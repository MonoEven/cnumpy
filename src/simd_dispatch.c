/**
 * SSE2-safe runtime selection for optional AVX2 reduction kernels.
 */
#include <cnumpy/cnumpy_internal.h>
#include <windows.h>
#include <intrin.h>

typedef void (*CnpBinaryElementwiseKernel)(
    const double *, const double *, double *, int64_t);
typedef void (*CnpUnaryElementwiseKernel)(
    const double *, double *, int64_t);
typedef void (*CnpBinaryLogicalKernel)(
    const double *, const double *, uint8_t *, int64_t);
typedef void (*CnpUnaryLogicalKernel)(
    const double *, uint8_t *, int64_t);
typedef void (*CnpUnaryFloatElementwiseKernel)(
    const float *, float *, int64_t);
typedef double (*CnpDotKernel)(const double *, const double *, int64_t);
typedef void (*CnpArangeKernel)(
    double *, double, double, int64_t);

static INIT_ONCE g_dispatch_once = INIT_ONCE_STATIC_INIT;
static CnpBinaryElementwiseKernel g_add_kernel = cnp_sse2_add;
static CnpBinaryElementwiseKernel g_subtract_kernel = cnp_sse2_subtract;
static CnpBinaryElementwiseKernel g_multiply_kernel = cnp_sse2_multiply;
static CnpBinaryElementwiseKernel g_divide_kernel = cnp_sse2_divide;
static CnpBinaryElementwiseKernel g_maximum_kernel = cnp_sse2_maximum;
static CnpBinaryElementwiseKernel g_minimum_kernel = cnp_sse2_minimum;
static CnpBinaryElementwiseKernel g_fmax_kernel = cnp_sse2_fmax;
static CnpBinaryElementwiseKernel g_fmin_kernel = cnp_sse2_fmin;
static CnpBinaryLogicalKernel g_logical_and_kernel = cnp_sse2_logical_and;
static CnpBinaryLogicalKernel g_logical_or_kernel = cnp_sse2_logical_or;
static CnpBinaryLogicalKernel g_logical_xor_kernel = cnp_sse2_logical_xor;
static CnpUnaryLogicalKernel g_logical_not_kernel = cnp_sse2_logical_not;
static CnpUnaryElementwiseKernel g_negative_kernel = cnp_sse2_negative;
static CnpUnaryElementwiseKernel g_absolute_kernel = cnp_sse2_absolute;
static CnpUnaryElementwiseKernel g_sqrt_kernel = cnp_sse2_sqrt;
static CnpUnaryElementwiseKernel g_floor_kernel = cnp_sse2_floor;
static CnpUnaryFloatElementwiseKernel g_sin_f32_kernel = cnp_sse2_sin_f32;
static CnpUnaryFloatElementwiseKernel g_tanh_f32_kernel = cnp_sse2_tanh_f32;
static CnpUnaryElementwiseKernel g_tanh_f64_kernel = cnp_sse2_tanh_f64;
static CnpGemmTileKernel g_gemm_tile_kernel = cnp_sse2_gemm_tile;
static CnpDotKernel g_dot_kernel = cnp_sse2_dot;
static CnpArangeKernel g_arange_kernel = cnp_sse2_arange;
static int g_simd_level = CNP_SIMD_LEVEL_SSE2;

static bool cpu_and_os_support_avx2(bool *supports_fma3) {
    int cpu_info[4];
    int maximum_leaf;

    __cpuid(cpu_info, 0);
    maximum_leaf = cpu_info[0];
    *supports_fma3 = false;
    if (maximum_leaf < 1) return false;

    __cpuid(cpu_info, 1);
    *supports_fma3 = (cpu_info[2] & (1 << 12)) != 0;
    if (!(cpu_info[2] & (1 << 27)) || !(cpu_info[2] & (1 << 28)))
        return false;
    if ((_xgetbv(0) & 0x6) != 0x6) return false;
    if (maximum_leaf < 7) return false;

    __cpuidex(cpu_info, 7, 0);
    return (cpu_info[1] & (1 << 5)) != 0;
}

static BOOL CALLBACK initialize_dispatch(
    PINIT_ONCE once, PVOID parameter, PVOID *context) {
    (void)once;
    (void)parameter;
    (void)context;
    bool supports_fma3 = false;
    if (cpu_and_os_support_avx2(&supports_fma3)) {
        g_add_kernel = cnp_avx2_add;
        g_subtract_kernel = cnp_avx2_subtract;
        g_multiply_kernel = cnp_avx2_multiply;
        g_divide_kernel = cnp_avx2_divide;
        g_maximum_kernel = cnp_avx2_maximum;
        g_minimum_kernel = cnp_avx2_minimum;
        g_fmax_kernel = cnp_avx2_fmax;
        g_fmin_kernel = cnp_avx2_fmin;
        g_logical_and_kernel = cnp_avx2_logical_and;
        g_logical_or_kernel = cnp_avx2_logical_or;
        g_logical_xor_kernel = cnp_avx2_logical_xor;
        g_logical_not_kernel = cnp_avx2_logical_not;
        g_negative_kernel = cnp_avx2_negative;
        g_absolute_kernel = cnp_avx2_absolute;
        g_sqrt_kernel = cnp_avx2_sqrt;
        g_floor_kernel = cnp_avx2_floor;
        if (supports_fma3) {
            g_sin_f32_kernel = cnp_avx2_sin_f32;
            g_tanh_f32_kernel = cnp_avx2_tanh_f32;
            g_tanh_f64_kernel = cnp_avx2_tanh_f64;
        }
        g_gemm_tile_kernel = cnp_avx2_gemm_tile;
        g_dot_kernel = cnp_avx2_dot;
        g_arange_kernel = cnp_avx2_arange;
        g_simd_level = CNP_SIMD_LEVEL_AVX2;
    }
    return TRUE;
}

CNP_STATUS cnp_simd_init_dispatch(void) {
    if (!InitOnceExecuteOnce(
            &g_dispatch_once, initialize_dispatch, NULL, NULL)) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_simd_init_dispatch",
                      "InitOnceExecuteOnce failed with Win32 error %lu",
                      GetLastError());
        return CNP_ERR_GENERIC;
    }
    return CNP_OK;
}

int cnp_simd_runtime_level(void) {
    if (cnp_simd_init_dispatch() != CNP_OK) return 0;
    return g_simd_level;
}

void cnp_simd_add(
    const double *a, const double *b, double *out, int64_t n) {
    g_add_kernel(a, b, out, n);
}

void cnp_simd_subtract(
    const double *a, const double *b, double *out, int64_t n) {
    g_subtract_kernel(a, b, out, n);
}

void cnp_simd_multiply(
    const double *a, const double *b, double *out, int64_t n) {
    g_multiply_kernel(a, b, out, n);
}

void cnp_simd_divide(
    const double *a, const double *b, double *out, int64_t n) {
    g_divide_kernel(a, b, out, n);
}

void cnp_simd_maximum(
    const double *a, const double *b, double *out, int64_t n) {
    g_maximum_kernel(a, b, out, n);
}

void cnp_simd_minimum(
    const double *a, const double *b, double *out, int64_t n) {
    g_minimum_kernel(a, b, out, n);
}

void cnp_simd_fmax(
    const double *a, const double *b, double *out, int64_t n) {
    g_fmax_kernel(a, b, out, n);
}

void cnp_simd_fmin(
    const double *a, const double *b, double *out, int64_t n) {
    g_fmin_kernel(a, b, out, n);
}

void cnp_simd_logical_and(
    const double *a, const double *b, uint8_t *out, int64_t n) {
    g_logical_and_kernel(a, b, out, n);
}

void cnp_simd_logical_or(
    const double *a, const double *b, uint8_t *out, int64_t n) {
    g_logical_or_kernel(a, b, out, n);
}

void cnp_simd_logical_xor(
    const double *a, const double *b, uint8_t *out, int64_t n) {
    g_logical_xor_kernel(a, b, out, n);
}

void cnp_simd_logical_not(
    const double *a, uint8_t *out, int64_t n) {
    g_logical_not_kernel(a, out, n);
}

void cnp_simd_negative(const double *a, double *out, int64_t n) {
    g_negative_kernel(a, out, n);
}

void cnp_simd_absolute(const double *a, double *out, int64_t n) {
    g_absolute_kernel(a, out, n);
}

void cnp_simd_abs(const double *a, double *out, int64_t n) {
    cnp_simd_absolute(a, out, n);
}

void cnp_simd_sqrt(const double *a, double *out, int64_t n) {
    g_sqrt_kernel(a, out, n);
}

void cnp_simd_floor(const double *a, double *out, int64_t n) {
    g_floor_kernel(a, out, n);
}

void cnp_simd_sin_f32(const float *a, float *out, int64_t n) {
    g_sin_f32_kernel(a, out, n);
}

void cnp_simd_tanh_f32(const float *a, float *out, int64_t n) {
    g_tanh_f32_kernel(a, out, n);
}

void cnp_simd_tanh_f64(const double *a, double *out, int64_t n) {
    g_tanh_f64_kernel(a, out, n);
}

void cnp_simd_gemm_tile(
    const double *a, const double *b, double *c,
    int64_t m, int64_t n, int64_t k,
    int64_t row_begin, int64_t row_end) {
    g_gemm_tile_kernel(a, b, c, m, n, k, row_begin, row_end);
}

double cnp_simd_dot(const double *a, const double *b, int64_t n) {
    return g_dot_kernel(a, b, n);
}

void cnp_simd_arange(
    double *out, double start, double step, int64_t n) {
    g_arange_kernel(out, start, step, n);
}
