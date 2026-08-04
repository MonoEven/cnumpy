/**
 * cnumpy SIMD-optimized operations (SSE2 baseline for x64)
 * Provides vectorized fast paths for contiguous float64 arrays.
 */
#include "../include/cnumpy/cnumpy_internal.h"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <emmintrin.h>  /* SSE2 */
#endif

/* =========================================================================
 * SIMD Binary Operations: out[i] = a[i] OP b[i]
 * ========================================================================= */
#define CNP_SSE2_BINARY_BODY(vector_operation, scalar_operator) \
    int64_t i = 0; \
    for (; i + 7 < n; i += 8) { \
        _mm_storeu_pd(out + i, vector_operation( \
            _mm_loadu_pd(a + i), _mm_loadu_pd(b + i))); \
        _mm_storeu_pd(out + i + 2, vector_operation( \
            _mm_loadu_pd(a + i + 2), _mm_loadu_pd(b + i + 2))); \
        _mm_storeu_pd(out + i + 4, vector_operation( \
            _mm_loadu_pd(a + i + 4), _mm_loadu_pd(b + i + 4))); \
        _mm_storeu_pd(out + i + 6, vector_operation( \
            _mm_loadu_pd(a + i + 6), _mm_loadu_pd(b + i + 6))); \
    } \
    for (; i + 1 < n; i += 2) { \
        _mm_storeu_pd(out + i, vector_operation( \
            _mm_loadu_pd(a + i), _mm_loadu_pd(b + i))); \
    } \
    for (; i < n; ++i) out[i] = a[i] scalar_operator b[i]

void cnp_sse2_add(const double *a, const double *b, double *out, int64_t n) {
    CNP_SSE2_BINARY_BODY(_mm_add_pd, +);
}

void cnp_sse2_subtract(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_SSE2_BINARY_BODY(_mm_sub_pd, -);
}

void cnp_sse2_multiply(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_SSE2_BINARY_BODY(_mm_mul_pd, *);
}

void cnp_sse2_divide(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_SSE2_BINARY_BODY(_mm_div_pd, /);
}

#undef CNP_SSE2_BINARY_BODY

static bool cnp_sse2_scalar_is_nan(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) ==
               UINT64_C(0x7ff0000000000000) &&
           (bits & UINT64_C(0x000fffffffffffff)) != 0;
}

static __forceinline __m128d cnp_sse2_maximum_vector(
    __m128d left, __m128d right) {
    __m128d left_nan = _mm_cmpunord_pd(left, left);
    __m128d right_nan = _mm_cmpunord_pd(right, right);
    __m128d unordered = _mm_or_pd(left_nan, right_nan);
    __m128d nan_value = _mm_or_pd(
        _mm_and_pd(left_nan, left), _mm_andnot_pd(left_nan, right));
    __m128d maximum = _mm_max_pd(left, right);
    return _mm_or_pd(_mm_and_pd(unordered, nan_value),
                     _mm_andnot_pd(unordered, maximum));
}

static __forceinline __m128d cnp_sse2_select(
    __m128d mask, __m128d when_true, __m128d when_false) {
    return _mm_or_pd(
        _mm_and_pd(mask, when_true),
        _mm_andnot_pd(mask, when_false));
}

static __forceinline __m128d cnp_sse2_minimum_vector(
    __m128d left, __m128d right) {
    __m128d left_nan = _mm_cmpunord_pd(left, left);
    __m128d right_nan = _mm_cmpunord_pd(right, right);
    __m128d unordered = _mm_or_pd(left_nan, right_nan);
    __m128d nan_value = cnp_sse2_select(left_nan, left, right);
    return cnp_sse2_select(
        unordered, nan_value, _mm_min_pd(left, right));
}

static __forceinline __m128d cnp_sse2_fmax_vector(
    __m128d left, __m128d right) {
    __m128d right_nan = _mm_cmpunord_pd(right, right);
    __m128d result = cnp_sse2_select(
        right_nan, left, _mm_max_pd(left, right));
    __m128d zero = _mm_setzero_pd();
    __m128d both_zero = _mm_and_pd(
        _mm_cmpeq_pd(left, zero), _mm_cmpeq_pd(right, zero));
    return cnp_sse2_select(
        both_zero, _mm_and_pd(left, right), result);
}

static __forceinline __m128d cnp_sse2_fmin_vector(
    __m128d left, __m128d right) {
    __m128d right_nan = _mm_cmpunord_pd(right, right);
    __m128d result = cnp_sse2_select(
        right_nan, left, _mm_min_pd(left, right));
    __m128d zero = _mm_setzero_pd();
    __m128d both_zero = _mm_and_pd(
        _mm_cmpeq_pd(left, zero), _mm_cmpeq_pd(right, zero));
    return cnp_sse2_select(
        both_zero, _mm_or_pd(left, right), result);
}

#define CNP_SSE2_EXTREMA_MAXIMUM 0
#define CNP_SSE2_EXTREMA_MINIMUM 1
#define CNP_SSE2_EXTREMA_FMAX 2
#define CNP_SSE2_EXTREMA_FMIN 3

static void cnp_sse2_scalar_extrema(
    double left, double right, double *result, int operation) {
    uint64_t left_bits;
    uint64_t right_bits;
    memcpy(&left_bits, &left, sizeof(left_bits));
    memcpy(&right_bits, &right, sizeof(right_bits));
    bool left_nan = cnp_sse2_scalar_is_nan(left);
    bool right_nan = cnp_sse2_scalar_is_nan(right);
    uint64_t selected_bits;
    if (operation == CNP_SSE2_EXTREMA_FMAX ||
            operation == CNP_SSE2_EXTREMA_FMIN) {
        if (left_nan) selected_bits = right_nan ? left_bits : right_bits;
        else if (right_nan) selected_bits = left_bits;
        else if (left == 0.0 && right == 0.0) {
            selected_bits = operation == CNP_SSE2_EXTREMA_FMAX
                ? left_bits & right_bits : left_bits | right_bits;
        } else {
            bool select_left = operation == CNP_SSE2_EXTREMA_FMAX
                ? left > right : left < right;
            selected_bits = select_left ? left_bits : right_bits;
        }
    } else if (left_nan) selected_bits = left_bits;
    else if (right_nan) selected_bits = right_bits;
    else {
        bool select_left = operation == CNP_SSE2_EXTREMA_MAXIMUM
            ? left > right : left < right;
        selected_bits = select_left ? left_bits : right_bits;
    }
    memcpy(result, &selected_bits, sizeof(selected_bits));
}

#define CNP_SSE2_EXTREMA_BODY(vector_function, scalar_operation) \
    int64_t i = 0; \
    for (; i + 7 < n; i += 8) { \
        _mm_storeu_pd(out + i, vector_function( \
            _mm_loadu_pd(a + i), _mm_loadu_pd(b + i))); \
        _mm_storeu_pd(out + i + 2, vector_function( \
            _mm_loadu_pd(a + i + 2), _mm_loadu_pd(b + i + 2))); \
        _mm_storeu_pd(out + i + 4, vector_function( \
            _mm_loadu_pd(a + i + 4), _mm_loadu_pd(b + i + 4))); \
        _mm_storeu_pd(out + i + 6, vector_function( \
            _mm_loadu_pd(a + i + 6), _mm_loadu_pd(b + i + 6))); \
    } \
    for (; i + 1 < n; i += 2) { \
        _mm_storeu_pd(out + i, vector_function( \
            _mm_loadu_pd(a + i), _mm_loadu_pd(b + i))); \
    } \
    for (; i < n; ++i) \
        cnp_sse2_scalar_extrema(a[i], b[i], out + i, scalar_operation)

void cnp_sse2_maximum(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_SSE2_EXTREMA_BODY(
        cnp_sse2_maximum_vector, CNP_SSE2_EXTREMA_MAXIMUM);
}

void cnp_sse2_minimum(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_SSE2_EXTREMA_BODY(
        cnp_sse2_minimum_vector, CNP_SSE2_EXTREMA_MINIMUM);
}

void cnp_sse2_fmax(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_SSE2_EXTREMA_BODY(
        cnp_sse2_fmax_vector, CNP_SSE2_EXTREMA_FMAX);
}

void cnp_sse2_fmin(
    const double *a, const double *b, double *out, int64_t n) {
    CNP_SSE2_EXTREMA_BODY(
        cnp_sse2_fmin_vector, CNP_SSE2_EXTREMA_FMIN);
}

#undef CNP_SSE2_EXTREMA_BODY
#undef CNP_SSE2_EXTREMA_MAXIMUM
#undef CNP_SSE2_EXTREMA_MINIMUM
#undef CNP_SSE2_EXTREMA_FMAX
#undef CNP_SSE2_EXTREMA_FMIN

#define CNP_SSE2_LOGICAL_AND 0
#define CNP_SSE2_LOGICAL_OR 1
#define CNP_SSE2_LOGICAL_XOR 2

static __forceinline bool cnp_sse2_scalar_truth_f64(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7fffffffffffffff)) != 0;
}

static __forceinline int cnp_sse2_truth_mask_f64x2(
    const double *values) {
    const __m128i magnitude_mask = _mm_set_epi32(
        INT32_MAX, -1, INT32_MAX, -1);
    __m128i magnitude = _mm_and_si128(
        _mm_loadu_si128((const __m128i*)values), magnitude_mask);
    __m128i zero32 = _mm_cmpeq_epi32(
        magnitude, _mm_setzero_si128());
    __m128i paired = _mm_shuffle_epi32(
        zero32, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i zero64 = _mm_and_si128(zero32, paired);
    int zero_mask = _mm_movemask_pd(_mm_castsi128_pd(zero64));
    return zero_mask ^ 3;
}

static __forceinline uint16_t cnp_sse2_pack_bool2(int mask) {
    return (uint16_t)(
        (mask & 1) |
        ((mask & 2) << 7));
}

static __forceinline void cnp_sse2_logical_binary2(
    const double *a,
    const double *b,
    uint8_t *out,
    int operation) {
    int left_mask = cnp_sse2_truth_mask_f64x2(a);
    int right_mask = cnp_sse2_truth_mask_f64x2(b);
    int result_mask = operation == CNP_SSE2_LOGICAL_AND
        ? left_mask & right_mask
        : operation == CNP_SSE2_LOGICAL_OR
        ? left_mask | right_mask
        : left_mask ^ right_mask;
    uint16_t packed = cnp_sse2_pack_bool2(result_mask);
    memcpy(out, &packed, sizeof(packed));
}

static void cnp_sse2_logical_binary(
    const double *a,
    const double *b,
    uint8_t *out,
    int64_t n,
    int operation) {
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        cnp_sse2_logical_binary2(a + i, b + i, out + i, operation);
        cnp_sse2_logical_binary2(
            a + i + 2, b + i + 2, out + i + 2, operation);
        cnp_sse2_logical_binary2(
            a + i + 4, b + i + 4, out + i + 4, operation);
        cnp_sse2_logical_binary2(
            a + i + 6, b + i + 6, out + i + 6, operation);
    }
    for (; i + 1 < n; i += 2)
        cnp_sse2_logical_binary2(a + i, b + i, out + i, operation);
    for (; i < n; ++i) {
        bool left = cnp_sse2_scalar_truth_f64(a[i]);
        bool right = cnp_sse2_scalar_truth_f64(b[i]);
        out[i] = (uint8_t)(operation == CNP_SSE2_LOGICAL_AND
            ? left && right
            : operation == CNP_SSE2_LOGICAL_OR
            ? left || right
            : left != right);
    }
}

void cnp_sse2_logical_and(
    const double *a, const double *b, uint8_t *out, int64_t n) {
    cnp_sse2_logical_binary(
        a, b, out, n, CNP_SSE2_LOGICAL_AND);
}

void cnp_sse2_logical_or(
    const double *a, const double *b, uint8_t *out, int64_t n) {
    cnp_sse2_logical_binary(
        a, b, out, n, CNP_SSE2_LOGICAL_OR);
}

void cnp_sse2_logical_xor(
    const double *a, const double *b, uint8_t *out, int64_t n) {
    cnp_sse2_logical_binary(
        a, b, out, n, CNP_SSE2_LOGICAL_XOR);
}

void cnp_sse2_logical_not(
    const double *a, uint8_t *out, int64_t n) {
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        for (int block = 0; block < 4; ++block) {
            int truth_mask = cnp_sse2_truth_mask_f64x2(
                a + i + block * 2);
            uint16_t packed = cnp_sse2_pack_bool2(truth_mask ^ 3);
            memcpy(out + i + block * 2, &packed, sizeof(packed));
        }
    }
    for (; i + 1 < n; i += 2) {
        int truth_mask = cnp_sse2_truth_mask_f64x2(a + i);
        uint16_t packed = cnp_sse2_pack_bool2(truth_mask ^ 3);
        memcpy(out + i, &packed, sizeof(packed));
    }
    for (; i < n; ++i)
        out[i] = (uint8_t)!cnp_sse2_scalar_truth_f64(a[i]);
}

#undef CNP_SSE2_LOGICAL_AND
#undef CNP_SSE2_LOGICAL_OR
#undef CNP_SSE2_LOGICAL_XOR

/* =========================================================================
 * SIMD Unary Operations
 * ========================================================================= */
void cnp_sse2_sqrt(const double *a, double *out, int64_t n) {
    int64_t i = 0;
    for (; i + 1 < n; i += 2) {
        __m128d va = _mm_loadu_pd(a + i);
        _mm_storeu_pd(out + i, _mm_sqrt_pd(va));
    }
    for (; i < n; i++) out[i] = sqrt(a[i]);
}

void cnp_sse2_absolute(const double *a, double *out, int64_t n) {
    /* Mask to clear sign bit */
    __m128d sign_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
    int64_t i = 0;
    for (; i + 1 < n; i += 2) {
        __m128d va = _mm_loadu_pd(a + i);
        _mm_storeu_pd(out + i, _mm_and_pd(va, sign_mask));
    }
    for (; i < n; i++) out[i] = fabs(a[i]);
}

void cnp_sse2_negative(const double *a, double *out, int64_t n) {
    __m128d sign = _mm_castsi128_pd(_mm_set_epi32(
        (int)0x80000000u, 0, (int)0x80000000u, 0));
    int64_t i = 0;
    for (; i + 1 < n; i += 2) {
        __m128d va = _mm_loadu_pd(a + i);
        _mm_storeu_pd(out + i, _mm_xor_pd(va, sign));
    }
    for (; i < n; i++) out[i] = -a[i];
}

void cnp_sse2_floor(const double *a, double *out, int64_t n) {
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        out[i] = floor(a[i]);
        out[i + 1] = floor(a[i + 1]);
        out[i + 2] = floor(a[i + 2]);
        out[i + 3] = floor(a[i + 3]);
    }
    for (; i < n; ++i) out[i] = floor(a[i]);
}

void cnp_sse2_sin_f32(const float *a, float *out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = sinf(a[i]);
}

void cnp_sse2_tanh_f32(const float *a, float *out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = tanhf(a[i]);
}

void cnp_sse2_tanh_f64(const double *a, double *out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = tanh(a[i]);
}

/* =========================================================================
 * SIMD Reductions
 * ========================================================================= */
double cnp_simd_sum(const double *data, int64_t n) {
    __m128d acc0 = _mm_setzero_pd();
    __m128d acc1 = _mm_setzero_pd();
    int64_t i = 0;

    /* Two accumulators for ILP */
    for (; i + 3 < n; i += 4) {
        acc0 = _mm_add_pd(acc0, _mm_loadu_pd(data + i));
        acc1 = _mm_add_pd(acc1, _mm_loadu_pd(data + i + 2));
    }
    acc0 = _mm_add_pd(acc0, acc1);

    /* Horizontal sum */
    __m128d hi = _mm_unpackhi_pd(acc0, acc0);
    acc0 = _mm_add_sd(acc0, hi);
    double result;
    _mm_store_sd(&result, acc0);

    /* Handle remaining */
    for (; i < n; i++) result += data[i];
    return result;
}

double cnp_sse2_dot(const double *a, const double *b, int64_t n) {
    __m128d acc0 = _mm_setzero_pd();
    __m128d acc1 = _mm_setzero_pd();
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        acc0 = _mm_add_pd(acc0, _mm_mul_pd(_mm_loadu_pd(a + i), _mm_loadu_pd(b + i)));
        acc1 = _mm_add_pd(acc1, _mm_mul_pd(_mm_loadu_pd(a + i + 2), _mm_loadu_pd(b + i + 2)));
    }
    acc0 = _mm_add_pd(acc0, acc1);
    __m128d hi = _mm_unpackhi_pd(acc0, acc0);
    acc0 = _mm_add_sd(acc0, hi);
    double result;
    _mm_store_sd(&result, acc0);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

/* =========================================================================
 * SIMD Fill
 * ========================================================================= */
void cnp_sse2_arange(
    double *out, double start, double step, int64_t n) {
    const __m128d lane01 = _mm_set_pd(step, 0.0);
    const __m128d lane23 = _mm_set_pd(3.0 * step, 2.0 * step);
    const __m128d lane45 = _mm_set_pd(5.0 * step, 4.0 * step);
    const __m128d lane67 = _mm_set_pd(7.0 * step, 6.0 * step);
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m128d base = _mm_set1_pd(start + (double)i * step);
        _mm_storeu_pd(out + i, _mm_add_pd(base, lane01));
        _mm_storeu_pd(out + i + 2, _mm_add_pd(base, lane23));
        _mm_storeu_pd(out + i + 4, _mm_add_pd(base, lane45));
        _mm_storeu_pd(out + i + 6, _mm_add_pd(base, lane67));
    }
    for (; i < n; ++i) out[i] = start + (double)i * step;
}

void cnp_simd_fill(double *out, double value, int64_t n) {
    __m128d vval = _mm_set1_pd(value);
    int64_t i = 0;
    for (; i + 1 < n; i += 2) {
        _mm_storeu_pd(out + i, vval);
    }
    for (; i < n; i++) out[i] = value;
}

void cnp_simd_zeros(double *out, int64_t n) {
    __m128d vzero = _mm_setzero_pd();
    int64_t i = 0;
    for (; i + 1 < n; i += 2) {
        _mm_storeu_pd(out + i, vzero);
    }
    for (; i < n; i++) out[i] = 0.0;
}
