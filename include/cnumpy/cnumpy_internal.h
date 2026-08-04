/**
 * cnumpy internal utilities - not part of public API
 */
#ifndef CNUMPY_INTERNAL_H
#define CNUMPY_INTERNAL_H

#include "cnumpy.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdarg.h>
#include <stdio.h>

/* Internal memory tracking */
extern size_t g_cnp_allocated_memory;

/* Memory allocation wrappers */
void* cnp_malloc(size_t size);
void* cnp_calloc(size_t count, size_t size);
void* cnp_realloc(void *ptr, size_t old_size, size_t new_size);
void  cnp_free(void *ptr, size_t size);
void* cnp_virtual_alloc(size_t size);
CNP_STATUS cnp_virtual_free(void *ptr, size_t size);

/* Error reporting */
void cnp_set_error(CNP_STATUS status, const char *func, const char *fmt, ...);
void cnp_relabel_error(const char *func);
void cnp_reset_string_functions(void);
void cnp_structured_cleanup(void);

CnpArray *cnp_array_adopt_external_data(
    int ndim, const int64_t *shape, CNP_TYPE dtype, CNP_ORDER order,
    void *data, uint32_t extra_flags,
    void *owner, CnpArrayOwnerRelease owner_release,
    const char *function_name);

/* Shared NPY parser for standalone files and ZIP members. */
CnpArray *cnp_npy_load_buffer(
    const uint8_t *buffer, size_t size, const char *function_name);
bool cnp_npy_save_buffer(
    const CnpArray *arr, uint8_t **buffer, size_t *size,
    const char *function_name);
bool cnp_inflate_raw(
    const uint8_t *source, size_t source_size,
    uint8_t *destination, size_t destination_size,
    size_t *written, const char *function_name);

/* Shared text representation and delimited-file contracts. */
char *cnp_text_array_string(
    const CnpArray *arr, const char *format,
    int precision, int64_t threshold, int edgeitems, bool suppress_small,
    const char *function_name);
bool cnp_text_float_format_is_valid(
    const char *format, const char *function_name);
CnpArray *cnp_text_load_file(
    const char *filename, const char *delimiter,
    int skip_header, int max_rows, bool missing_values,
    CNP_TYPE dtype, const char *function_name);

/* Internal helper: compute total size from shape */
int64_t cnp_compute_size(int ndim, const int64_t *shape);

/* Internal helper: compute strides from shape */
void cnp_compute_strides(int ndim, const int64_t *shape, int elsize, CNP_ORDER order, int64_t *strides);

/* Internal helper: derive contiguous-layout flags from shape and strides */
uint32_t cnp_compute_layout_flags(
    int ndim, const int64_t *shape, const int64_t *strides, int itemsize);

/* Internal helper: validate array metadata and compute ndarray.nbytes. */
bool cnp_array_nbytes_checked(
    const CnpArray *arr, const char *function_name, int64_t *nbytes);

/* Internal NumPy interp implementation shared by legacy public projections. */
CnpArray *cnp_interp_core(
    const CnpArray *x, const CnpArray *xp, const CnpArray *fp,
    bool explicit_bounds, double left, double right,
    const char *function_name);

/* Internal stride-aware FFT axis kernels shared by fft.c and fft_ext.c. */
CnpArray *cnp_fft_axis_transform(
    const CnpArray *source, int axis, int64_t length,
    bool inverse, const char *function_name);
CnpArray *cnp_fft_axis_real_forward(
    const CnpArray *source, int axis, int64_t length,
    const char *function_name);
CnpArray *cnp_fft_axis_real_inverse(
    const CnpArray *source, int axis, int64_t length,
    const char *function_name);

/* Internal parsed contraction path that deliberately bypasses public
 * einsum's product aliases, so those aliases can share the exact generic
 * dtype/stride/broadcast engine without recursive fast dispatch. */
CnpArray *cnp_einsum_generic(
    const char *subscripts, int narrays,
    const CnpArray *const *arrays, const char *function_name);
CNP_STATUS cnp_multiply_scalar_values(
    const void *left, CNP_TYPE left_type,
    const void *right, CNP_TYPE right_type,
    void *output, CNP_TYPE output_type,
    const char *function_name);

/* Internal helper: convert multi-index to flat offset */
int64_t cnp_multi_to_offset(int ndim, const int64_t *indices, const int64_t *strides);

/* Internal helper: normalize axis (handle negative) */
int cnp_normalize_axis(int axis, int ndim);

/* Internal helper: get element as double from raw data */
double cnp_get_element_double(const void *data, int64_t offset, CNP_TYPE dtype);

/* Internal helper: set element from double to raw data */
void cnp_set_element_double(void *data, int64_t offset, CNP_TYPE dtype, double value);

/* Internal helper: get element as int64 from raw data */
int64_t cnp_get_element_int(const void *data, int64_t offset, CNP_TYPE dtype);

/* Internal helper: set element from int64 to raw data */
void cnp_set_element_int(void *data, int64_t offset, CNP_TYPE dtype, int64_t value);

/* Internal helper: create result array for binary ops with broadcasting */
CnpArray* cnp_binary_op_prepare(const CnpArray *a, const CnpArray *b, CNP_TYPE out_dtype);

/* Internal helper: create result array for unary ops */
CnpArray* cnp_unary_op_prepare(const CnpArray *a, CNP_TYPE out_dtype);

/* Internal: generic binary operation with broadcasting */
typedef double (*cnp_binary_func)(double a, double b);
CnpArray* cnp_binary_op(const CnpArray *a, const CnpArray *b, cnp_binary_func func, CNP_TYPE out_dtype);

/* Internal: generic unary operation */
typedef double (*cnp_unary_func)(double a);
CnpArray* cnp_unary_op(const CnpArray *a, cnp_unary_func func, CNP_TYPE out_dtype);

typedef enum {
    CNP_UNARY_ROUND_RINT = 0,
    CNP_UNARY_ROUND_TRUNCATE,
    CNP_UNARY_ROUND_FLOOR,
    CNP_UNARY_ROUND_CEIL
} CnpUnaryRoundingMode;

CnpArray* cnp_unary_op_rounding(
    const CnpArray *source,
    CnpUnaryRoundingMode mode,
    const char *function_name);

CnpArray* cnp_unary_op_absolute(
    const CnpArray *source,
    bool floating_result,
    const char *function_name);

CNP_STATUS cnp_add_into_promoted(
    const CnpArray *left,
    const CnpArray *right,
    CnpArray *output);

CNP_STATUS cnp_cast_scalar_value(
    const void *source,
    CNP_TYPE source_type,
    void *destination,
    CNP_TYPE destination_type,
    const char *function_name);
CNP_STATUS cnp_compare_numeric_elements(
    const void *left, CNP_TYPE left_type,
    const void *right, CNP_TYPE right_type,
    CNP_TYPE comparison_type, int *order,
    const char *function_name);
static inline int cnp_compare_numpy_doubles(double left, double right) {
    bool left_nan = isnan(left);
    bool right_nan = isnan(right);
    if (left_nan) return right_nan ? 0 : 1;
    if (right_nan) return -1;
    if (left < right) return -1;
    if (left > right) return 1;
    return 0;
}
static inline uint64_t cnp_double_to_sortable(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if (isnan(value)) return UINT64_MAX;
    if (value == 0.0) return UINT64_C(0x8000000000000000);
    if (bits & UINT64_C(0x8000000000000000)) return ~bits;
    return bits ^ UINT64_C(0x8000000000000000);
}

typedef struct {
    const CnpArray *array;
    int axis;
    bool axis_none;
    int64_t outer;
    int64_t axis_length;
    int64_t axis_stride;
    int64_t inner;
    int result_ndim;
    int64_t result_shape[CNP_MAXDIMS];
} CnpReductionTraversal;

bool cnp_reduction_resolve_axis(
    const CnpArray *arr, int axis, bool axis_none,
    const char *function_name, int *resolved_axis);
bool cnp_reduction_resolve_axis_strict_scalar(
    const CnpArray *arr, int axis, bool axis_none,
    const char *function_name, int *resolved_axis);
void cnp_reduction_traversal_init(
    const CnpArray *arr, int resolved_axis,
    CnpReductionTraversal *traversal);
int64_t cnp_reduction_source_offset(
    const CnpReductionTraversal *traversal,
    int64_t outer, int64_t inner, int64_t item);
double cnp_reduction_sum_contiguous_double(
    const double *values, int64_t length);

/* Internal: comparison operation */
typedef bool (*cnp_compare_func)(double a, double b);
CnpArray* cnp_compare_op(const CnpArray *a, const CnpArray *b, cnp_compare_func func);

/* Internal: logical operation */
typedef bool (*cnp_logical_func)(bool a, bool b);
typedef void (*cnp_logical_f64_binary_kernel)(
    const double *left, const double *right,
    uint8_t *result, int64_t size);
CnpArray* cnp_logical_op(
    const CnpArray *a, const CnpArray *b,
    cnp_logical_func func,
    cnp_logical_f64_binary_kernel simd_kernel,
    const char *function_name);

CNP_STATUS cnp_validate_logical_truth_dtype(
    CNP_TYPE dtype, bool allow_temporal,
    const char *function_name);
CNP_STATUS cnp_scalar_truth(
    const void *source, CNP_TYPE dtype,
    bool *truth, const char *function_name);
CNP_ORDER cnp_logical_result_order(
    const CnpArray *left, const CnpArray *right);

/* Internal: check if value is truthy */
bool cnp_value_is_true(double val);

/* Internal: swap bytes */
void cnp_swap_bytes(void *data, int size);

/* Internal: type promotion for arithmetic */
CNP_TYPE cnp_promote_type(CNP_TYPE a, CNP_TYPE b);
CNP_TYPE cnp_promote_type_full(CNP_TYPE a, CNP_TYPE b);

/* Shared NumPy-compatible matrix-power implementation. */
CnpArray* cnp_matrix_power_impl(
    const CnpArray *a, int64_t exponent, const char *function_name);

/* Internal: check if type is floating point */
bool cnp_type_is_float(CNP_TYPE type);

/* Internal: check if type is integer */
bool cnp_type_is_integer(CNP_TYPE type);

/* Internal: check if type is complex */
bool cnp_type_is_complex(CNP_TYPE type);

/* Internal: check if type is unsigned */
bool cnp_type_is_unsigned(CNP_TYPE type);

/* Internal: create a 0-d array (scalar) */
CnpArray* cnp_scalar_array(double value, CNP_TYPE dtype);

/* Internal: ensure array is contiguous */
CnpArray* cnp_ensure_contiguous(const CnpArray *arr, CNP_ORDER order);

/* Internal: eigvals-only path that always preserves complex work results. */
CNP_STATUS cnp_linalg_eigvals_force_complex(
    const CnpArray *a, CnpArray **eigenvalues);

/* Internal: construct a view from caller-computed metadata. */
CnpArray* cnp_array_view_from_metadata(
    CnpArray *base, int ndim, const int64_t *shape,
    const int64_t *strides, int64_t offset,
    uint32_t layout_flags);

/* Internal: construct a contiguous reshape view with merged metadata. */
CnpArray* cnp_array_reshape_view(CnpArray *src, int ndim,
                                 const int64_t *shape, CNP_ORDER order);

/* Internal: apply slice to compute new shape/strides/offset */
CNP_STATUS cnp_apply_slice(const CnpArray *arr, const CnpSlice *slice, int axis,
                            int64_t *new_dim, int64_t *new_stride, int64_t *offset);

/* Random state global */
extern CnpRandomState g_cnp_random_state;

/* Internal random functions */
double cnp_random_double(void);
double cnp_random_gauss(void);
uint64_t cnp_random_uint64(void);
CnpArray *cnp_random_output_new(
    int ndim, const int64_t *shape, CNP_TYPE type,
    const char *function_name);
double cnp_random_gamma_sample(double shape_param);
int64_t cnp_random_poisson_sample(double lambda);
int64_t cnp_random_binomial_sample(int64_t n, double probability);

/* Shared NumPy polynomial common-type rule. */
bool cnp_polynomial_common_type(
    CNP_TYPE left, CNP_TYPE right, CNP_TYPE *result);

typedef enum {
    CNP_POLYNOMIAL_POWER,
    CNP_POLYNOMIAL_CHEBYSHEV,
    CNP_POLYNOMIAL_LEGENDRE,
    CNP_POLYNOMIAL_HERMITE,
    CNP_POLYNOMIAL_LAGUERRE
} CnpPolynomialBasis;

/* Shared NumPy polynomial least-squares implementation. */
CnpArray* cnp_polynomial_fit_basis(
    const CnpArray *x, const CnpArray *y, int deg,
    CnpPolynomialBasis basis, const char *function_name);

/* =========================================================================
 * SIMD-optimized operations (simd_ops.c)
 * ========================================================================= */
#define CNP_SIMD_LEVEL_SSE2 1
#define CNP_SIMD_LEVEL_AVX2 2

CNP_STATUS cnp_simd_init_dispatch(void);
int cnp_simd_runtime_level(void);

typedef void (*CnpGemmTileKernel)(
    const double *a, const double *b, double *c,
    int64_t m, int64_t n, int64_t k,
    int64_t row_begin, int64_t row_end);
CNP_STATUS cnp_gemm_f64(
    const double *a, const double *b, double *c,
    int64_t m, int64_t n, int64_t k);
CNP_STATUS cnp_gemm_thread_pool_init(void);
void cnp_gemm_thread_pool_cleanup(void);
void cnp_sse2_gemm_tile(
    const double *a, const double *b, double *c,
    int64_t m, int64_t n, int64_t k,
    int64_t row_begin, int64_t row_end);
void cnp_avx2_gemm_tile(
    const double *a, const double *b, double *c,
    int64_t m, int64_t n, int64_t k,
    int64_t row_begin, int64_t row_end);
void cnp_simd_gemm_tile(
    const double *a, const double *b, double *c,
    int64_t m, int64_t n, int64_t k,
    int64_t row_begin, int64_t row_end);

void cnp_sse2_add(
    const double *a, const double *b, double *out, int64_t n);
void cnp_sse2_subtract(
    const double *a, const double *b, double *out, int64_t n);
void cnp_sse2_multiply(
    const double *a, const double *b, double *out, int64_t n);
void cnp_sse2_divide(
    const double *a, const double *b, double *out, int64_t n);
void cnp_sse2_maximum(
    const double *a, const double *b, double *out, int64_t n);
void cnp_sse2_minimum(
    const double *a, const double *b, double *out, int64_t n);
void cnp_sse2_fmax(
    const double *a, const double *b, double *out, int64_t n);
void cnp_sse2_fmin(
    const double *a, const double *b, double *out, int64_t n);
void cnp_sse2_logical_and(
    const double *a, const double *b, uint8_t *out, int64_t n);
void cnp_sse2_logical_or(
    const double *a, const double *b, uint8_t *out, int64_t n);
void cnp_sse2_logical_xor(
    const double *a, const double *b, uint8_t *out, int64_t n);
void cnp_sse2_logical_not(
    const double *a, uint8_t *out, int64_t n);
void cnp_sse2_negative(const double *a, double *out, int64_t n);
void cnp_sse2_absolute(const double *a, double *out, int64_t n);
void cnp_sse2_sqrt(const double *a, double *out, int64_t n);
void cnp_sse2_floor(const double *a, double *out, int64_t n);
void cnp_sse2_sin_f32(const float *a, float *out, int64_t n);
void cnp_sse2_tanh_f32(const float *a, float *out, int64_t n);
void cnp_sse2_tanh_f64(const double *a, double *out, int64_t n);
void cnp_sse2_arange(
    double *out, double start, double step, int64_t n);
void cnp_avx2_add(
    const double *a, const double *b, double *out, int64_t n);
void cnp_avx2_subtract(
    const double *a, const double *b, double *out, int64_t n);
void cnp_avx2_multiply(
    const double *a, const double *b, double *out, int64_t n);
void cnp_avx2_divide(
    const double *a, const double *b, double *out, int64_t n);
void cnp_avx2_maximum(
    const double *a, const double *b, double *out, int64_t n);
void cnp_avx2_minimum(
    const double *a, const double *b, double *out, int64_t n);
void cnp_avx2_fmax(
    const double *a, const double *b, double *out, int64_t n);
void cnp_avx2_fmin(
    const double *a, const double *b, double *out, int64_t n);
void cnp_avx2_logical_and(
    const double *a, const double *b, uint8_t *out, int64_t n);
void cnp_avx2_logical_or(
    const double *a, const double *b, uint8_t *out, int64_t n);
void cnp_avx2_logical_xor(
    const double *a, const double *b, uint8_t *out, int64_t n);
void cnp_avx2_logical_not(
    const double *a, uint8_t *out, int64_t n);
void cnp_avx2_negative(const double *a, double *out, int64_t n);
void cnp_avx2_absolute(const double *a, double *out, int64_t n);
void cnp_avx2_sqrt(const double *a, double *out, int64_t n);
void cnp_avx2_floor(const double *a, double *out, int64_t n);
void cnp_avx2_sin_f32(const float *a, float *out, int64_t n);
void cnp_avx2_tanh_f32(const float *a, float *out, int64_t n);
void cnp_avx2_tanh_f64(const double *a, double *out, int64_t n);
void cnp_avx2_arange(
    double *out, double start, double step, int64_t n);

double cnp_sse2_dot(const double *a, const double *b, int64_t n);
double cnp_avx2_dot(const double *a, const double *b, int64_t n);

void cnp_simd_add(const double *a, const double *b, double *out, int64_t n);
void cnp_simd_subtract(const double *a, const double *b, double *out, int64_t n);
void cnp_simd_multiply(const double *a, const double *b, double *out, int64_t n);
void cnp_simd_divide(const double *a, const double *b, double *out, int64_t n);
void cnp_simd_maximum(
    const double *a, const double *b, double *out, int64_t n);
void cnp_simd_minimum(
    const double *a, const double *b, double *out, int64_t n);
void cnp_simd_fmax(
    const double *a, const double *b, double *out, int64_t n);
void cnp_simd_fmin(
    const double *a, const double *b, double *out, int64_t n);
void cnp_simd_logical_and(
    const double *a, const double *b, uint8_t *out, int64_t n);
void cnp_simd_logical_or(
    const double *a, const double *b, uint8_t *out, int64_t n);
void cnp_simd_logical_xor(
    const double *a, const double *b, uint8_t *out, int64_t n);
void cnp_simd_logical_not(
    const double *a, uint8_t *out, int64_t n);
void cnp_simd_sqrt(const double *a, double *out, int64_t n);
void cnp_simd_absolute(const double *a, double *out, int64_t n);
void cnp_simd_abs(const double *a, double *out, int64_t n);
void cnp_simd_negative(const double *a, double *out, int64_t n);
void cnp_simd_floor(const double *a, double *out, int64_t n);
void cnp_simd_sin_f32(const float *a, float *out, int64_t n);
void cnp_simd_tanh_f32(const float *a, float *out, int64_t n);
void cnp_simd_tanh_f64(const double *a, double *out, int64_t n);
void cnp_simd_arange(
    double *out, double start, double step, int64_t n);
double cnp_simd_sum(const double *data, int64_t n);
double cnp_simd_dot(const double *a, const double *b, int64_t n);
void cnp_simd_fill(double *out, double value, int64_t n);
void cnp_simd_zeros(double *out, int64_t n);

#endif /* CNUMPY_INTERNAL_H */
