/**
 * Stable AHK v2 DllCall ABI for allocation-free and batch operations.
 */
#ifndef CNUMPY_AHK_H
#define CNUMPY_AHK_H

#include "cnumpy.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CNP_AHK_BATCH_ADD_INTO       1u
#define CNP_AHK_BATCH_SQRT_INTO      2u
#define CNP_AHK_BATCH_CUMSUM_INTO    3u
#define CNP_AHK_BATCH_SUM_SCALAR     4u

/* Number of logical callback items offered per native-to-AHK crossing. */
#define CNP_AHK_CALLBACK_BATCH_SIZE 256

#define CNP_AHK_SIMD_SSE2 1
#define CNP_AHK_SIMD_AVX2 2

#define CNP_AHK_METADATA_ABI_VERSION 1u
#define CNP_AHK_METADATA_SIZE 544u

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t ndim;
    int32_t dtype;
    int32_t itemsize;
    uint32_t flags;
    int64_t size;
    int64_t shape[CNP_MAXDIMS];
} CnpAhkMetadata;

typedef struct {
    uint32_t opcode;
    uint32_t reserved;
    void *input0;
    void *input1;
    void *output;
    int64_t axis;
} CnpAhkBatchCommand;

/* Pointer-only callback ABI for AutoHotkey's untyped native callback thunk. */
typedef CNP_STATUS (CNP_CALL *CnpAhkLineCallback)(
    const double *line, int64_t length, void *userdata, double *result);
typedef CNP_STATUS (CNP_CALL *CnpAhkCoordinateCallback)(
    const int64_t *coordinates, int ndim,
    void *userdata, double *result);
typedef CNP_STATUS (CNP_CALL *CnpAhkIteratorCallback)(
    void *userdata, double *result);
typedef CNP_STATUS (CNP_CALL *CnpAhkUnaryCallback)(
    const double *value, void *userdata, double *result);

/*
 * Bulk callback ABI.  The callback must set produced_count on success.
 * A nonzero status aborts the operation; there is no scalar retry.
 */
typedef CNP_STATUS (CNP_CALL *CnpAhkLineBatchCallback)(
    const double *lines, int64_t line_count, int64_t line_length,
    double *results, int64_t result_capacity,
    int64_t *produced_count, void *userdata);
typedef CNP_STATUS (CNP_CALL *CnpAhkCoordinateBatchCallback)(
    const int64_t *coordinates, int64_t point_count, int ndim,
    double *results, int64_t result_capacity,
    int64_t *produced_count, void *userdata);
typedef CNP_STATUS (CNP_CALL *CnpAhkIteratorBatchCallback)(
    double *results, int64_t result_capacity,
    int64_t *produced_count, void *userdata);
typedef CNP_STATUS (CNP_CALL *CnpAhkUnaryBatchCallback)(
    const double *values, int64_t value_count,
    double *results, int64_t result_capacity,
    int64_t *produced_count, void *userdata);

CNP_API void *CNP_CALL cnp_ahk_apply_along_axis(
    CnpAhkLineCallback callback, void *userdata,
    int axis, void *source);
CNP_API void *CNP_CALL cnp_ahk_apply_over_axes(
    CnpAhkLineCallback callback, void *userdata,
    int naxes, const int *axes, void *source);
CNP_API void *CNP_CALL cnp_ahk_fromfunction(
    CnpAhkCoordinateCallback callback, void *userdata,
    int ndim, const int64_t *shape);
CNP_API void *CNP_CALL cnp_ahk_fromiter(
    CnpAhkIteratorCallback callback, void *userdata,
    int64_t count, int dtype);
CNP_API void *CNP_CALL cnp_ahk_frompyfunc(
    CnpAhkUnaryCallback callback, void *userdata, void *source);
CNP_API void *CNP_CALL cnp_ahk_vectorize(
    CnpAhkUnaryCallback callback, void *userdata, void *source);
CNP_API void *CNP_CALL cnp_ahk_piecewise(
    void *source, int nconditions, void *const *conditions,
    CnpAhkUnaryCallback callback, void *userdata);

CNP_API void *CNP_CALL cnp_ahk_apply_along_axis_v2(
    CnpAhkLineBatchCallback callback, void *userdata,
    int axis, void *source,
    int result_ndim, const int64_t *result_shape);
CNP_API void *CNP_CALL cnp_ahk_apply_over_axes_v2(
    CnpAhkLineBatchCallback callback, void *userdata,
    int naxes, const int *axes, void *source);
CNP_API void *CNP_CALL cnp_ahk_fromfunction_v2(
    CnpAhkCoordinateBatchCallback callback, void *userdata,
    int ndim, const int64_t *shape);
CNP_API void *CNP_CALL cnp_ahk_fromiter_v2(
    CnpAhkIteratorBatchCallback callback, void *userdata,
    int64_t count, int dtype);
CNP_API void *CNP_CALL cnp_ahk_frompyfunc_v2(
    CnpAhkUnaryBatchCallback callback, void *userdata, void *source);
CNP_API void *CNP_CALL cnp_ahk_vectorize_v2(
    CnpAhkUnaryBatchCallback callback, void *userdata, void *source);
CNP_API void *CNP_CALL cnp_ahk_piecewise_v2(
    void *source, int nconditions, void *const *conditions,
    CnpAhkUnaryBatchCallback callback, void *userdata);

CNP_API int CNP_CALL cnp_ahk_add_into(
    void *left, void *right, void *out);
CNP_API int CNP_CALL cnp_ahk_sqrt_into(
    void *source, void *out);
CNP_API int CNP_CALL cnp_ahk_cumsum_into(
    void *source, int axis, void *out);
CNP_API int CNP_CALL cnp_ahk_sum_into_scalar(
    void *source, double *out_value);
CNP_API void *CNP_CALL cnp_ahk_sum_array(void *source, int axis);
CNP_API void *CNP_CALL cnp_ahk_prod_array(void *source, int axis);
CNP_API void *CNP_CALL cnp_ahk_mean_array(void *source, int axis);
CNP_API void *CNP_CALL cnp_ahk_sum_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_prod_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_mean_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_average_v2(
    void *source, int axis, int axis_none, void *weights);
CNP_API void *CNP_CALL cnp_ahk_var_v2(
    void *handle, int axis, int axis_none, int ddof);
CNP_API void *CNP_CALL cnp_ahk_std_v2(
    void *handle, int axis, int axis_none, int ddof);
CNP_API void *CNP_CALL cnp_ahk_max_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_min_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_argmax_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_argmin_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_any_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_all_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_ptp_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_cumsum_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_cumprod_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_nansum_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_nanprod_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_nanmean_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_nanvar_v2(
    void *handle, int axis, int axis_none, int ddof);
CNP_API void *CNP_CALL cnp_ahk_nanstd_v2(
    void *handle, int axis, int axis_none, int ddof);
CNP_API void *CNP_CALL cnp_ahk_nanmax_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_nanmin_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_nanargmax_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_nanargmin_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_median_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_nanmedian_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_percentile_v2(
    void *handle, double q, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_nanpercentile_v2(
    void *handle, double q, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_quantile_v2(
    void *handle, double q, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_nanquantile_v2(
    void *handle, double q, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_nancumsum_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_nancumprod_v2(
    void *handle, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_var_array(void *source, int axis);
CNP_API void *CNP_CALL cnp_ahk_std_array(void *source, int axis);
CNP_API void *CNP_CALL cnp_ahk_max_array(void *source, int axis);
CNP_API void *CNP_CALL cnp_ahk_min_array(void *source, int axis);
CNP_API void *CNP_CALL cnp_ahk_atleast_1d(void *source);
CNP_API void *CNP_CALL cnp_ahk_atleast_2d(void *source);
CNP_API void *CNP_CALL cnp_ahk_atleast_3d(void *source);
CNP_API void *CNP_CALL cnp_ahk_transpose_copy(void *source);
CNP_API int CNP_CALL cnp_ahk_execute_batch(
    const CnpAhkBatchCommand *commands,
    int64_t command_count,
    int64_t *failed_index);
CNP_API int CNP_CALL cnp_ahk_simd_level(void);
CNP_API int CNP_CALL cnp_ahk_set_num_threads(int count);
CNP_API int CNP_CALL cnp_ahk_get_num_threads(void);
CNP_API int CNP_CALL cnp_ahk_get_metadata(
    void *handle, CnpAhkMetadata *out_metadata, uint32_t metadata_size);
CNP_API void *CNP_CALL cnp_ahk_concatenate_many(
    void *const *handles, int count, int axis);
CNP_API void *CNP_CALL cnp_ahk_sort(void *handle, int axis);
CNP_API void *CNP_CALL cnp_ahk_argsort(void *handle, int axis);
CNP_API void *CNP_CALL cnp_ahk_sort_v2(
    void *handle, int axis, int axis_none, int kind);
CNP_API void *CNP_CALL cnp_ahk_argsort_v2(
    void *handle, int axis, int axis_none, int kind);
CNP_API void *CNP_CALL cnp_ahk_msort(void *handle);
CNP_API void *CNP_CALL cnp_ahk_sort_complex(void *handle);
CNP_API void *CNP_CALL cnp_ahk_partition_v2(
    void *handle, const int64_t *kth, int kth_count,
    int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_argpartition_v2(
    void *handle, const int64_t *kth, int kth_count,
    int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_searchsorted_v2(
    void *source, void *values, const char *side, void *sorter);
CNP_API void *CNP_CALL cnp_ahk_digitize(
    void *x, void *bins, int right);
CNP_API void *CNP_CALL cnp_ahk_lexsort_v2(
    void *const *handles, int count, int axis);
CNP_API int CNP_CALL cnp_ahk_unique_v2(
    void *handle, int return_index, int return_inverse, int return_counts,
    void **results, int result_capacity);
CNP_API void *CNP_CALL cnp_ahk_intersect1d(
    void *left, void *right, int assume_unique);
CNP_API void *CNP_CALL cnp_ahk_union1d(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_setdiff1d(
    void *left, void *right, int assume_unique);
CNP_API void *CNP_CALL cnp_ahk_setxor1d(
    void *left, void *right, int assume_unique);
CNP_API void *CNP_CALL cnp_ahk_in1d(
    void *left, void *right, int assume_unique, int invert);
CNP_API void *CNP_CALL cnp_ahk_isin(
    void *element, void *test_elements, int assume_unique, int invert);
CNP_API void *CNP_CALL cnp_ahk_clip_array(
    void *source, void *a_min, void *a_max);
CNP_API void *CNP_CALL cnp_ahk_power(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_float_power(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_heaviside(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_isnan(void *source);
CNP_API void *CNP_CALL cnp_ahk_isinf(void *source);
CNP_API void *CNP_CALL cnp_ahk_isfinite(void *source);
CNP_API void *CNP_CALL cnp_ahk_signbit(void *source);
CNP_API int CNP_CALL cnp_ahk_iscomplexobj(void *source);
CNP_API int CNP_CALL cnp_ahk_isrealobj(void *source);
CNP_API int CNP_CALL cnp_ahk_isscalar(void *source);
CNP_API int CNP_CALL cnp_ahk_divmod(
    void *left, void *right, void **results, int result_capacity);
CNP_API void *CNP_CALL cnp_ahk_gcd(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_lcm(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_equal(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_not_equal(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_less(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_less_equal(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_greater(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_greater_equal(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_maximum(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_minimum(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_fmax(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_fmin(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_logical_and(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_logical_or(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_logical_xor(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_logical_not(void *source);
CNP_API void *CNP_CALL cnp_ahk_bitwise_and(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_bitwise_or(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_bitwise_xor(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_left_shift(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_right_shift(void *left, void *right);
CNP_API void *CNP_CALL cnp_ahk_invert(void *source);
CNP_API void *CNP_CALL cnp_ahk_bitwise_not(void *source);
CNP_API int CNP_CALL cnp_ahk_broadcast_arrays_v2(
    void *const *handles, int count,
    void **results, int result_capacity);
CNP_API int CNP_CALL cnp_ahk_meshgrid_v2(
    void *const *handles, int count,
    int sparse, int indexing_ij, int copy,
    void **results, int result_capacity);
CNP_API int CNP_CALL cnp_ahk_split_sections_v2(
    void *handle, int sections, int axis,
    void **results, int result_capacity);
CNP_API int CNP_CALL cnp_ahk_split_indices_v2(
    void *handle, int nindices, const int64_t *indices, int axis,
    void **results, int result_capacity);
CNP_API int CNP_CALL cnp_ahk_array_split_sections_v2(
    void *handle, int sections, int axis,
    void **results, int result_capacity);
CNP_API int CNP_CALL cnp_ahk_array_split_indices_v2(
    void *handle, int nindices, const int64_t *indices, int axis,
    void **results, int result_capacity);
CNP_API void *CNP_CALL cnp_ahk_take_v2(
    void *source, void *indices, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_take_along_axis_v2(
    void *source, void *indices, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_compress_v2(
    void *condition, void *source, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_delete_v2(
    void *source, void *obj, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_insert_v2(
    void *source, int64_t obj, void *values, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_insert_array_v2(
    void *source, void *obj, void *values, int axis, int axis_none);
CNP_API void *CNP_CALL cnp_ahk_softmax(void *source, int axis);
CNP_API void *CNP_CALL cnp_ahk_log_softmax(void *source, int axis);
CNP_API void *CNP_CALL cnp_ahk_trapz(
    void *y, void *x, double dx, int axis);
CNP_API void *CNP_CALL cnp_ahk_packbits_v2(
    void *source, int axis, int axis_none, int bitorder);
CNP_API void *CNP_CALL cnp_ahk_unpackbits_v2(
    void *source, int axis, int axis_none,
    int64_t count, int count_none, int bitorder);
CNP_API void *CNP_CALL cnp_ahk_einsum(
    const char *subscripts, int narrays, void **arrays);
CNP_API int CNP_CALL cnp_ahk_linalg_eig(
    void *source, void **results, int result_capacity);
CNP_API void *CNP_CALL cnp_ahk_linalg_cholesky(void *source);
CNP_API void *CNP_CALL cnp_ahk_linalg_det_v2(void *source);
CNP_API int CNP_CALL cnp_ahk_linalg_slogdet_v2(
    void *source, void **results, int result_capacity);
CNP_API int CNP_CALL cnp_ahk_linalg_eigh_v2(
    void *source, int upper, void **results, int result_capacity);
CNP_API void *CNP_CALL cnp_ahk_eigvalsh_v2(
    void *source, int upper);
CNP_API int CNP_CALL cnp_ahk_linalg_svd_v2(
    void *source, int full_matrices, int compute_uv, int hermitian,
    void **results, int result_capacity);
CNP_API void *CNP_CALL cnp_ahk_linalg_solve(void *a, void *b);
CNP_API int CNP_CALL cnp_ahk_linalg_lstsq_v2(
    void *a, void *b, double rcond, int rcond_none,
    void **results, int result_capacity);
CNP_API void *CNP_CALL cnp_ahk_linalg_cond_v2(void *source);
CNP_API void CNP_CALL cnp_ahk_random_seed_v2(uint64_t seed);
CNP_API void *CNP_CALL cnp_ahk_random_choice_v2(
    void *population,
    int size_ndim,
    const int64_t *size_shape,
    int size_none,
    int replace,
    void *probabilities);
CNP_API void *CNP_CALL cnp_ahk_random_permutation(void *source);
CNP_API int CNP_CALL cnp_ahk_random_shuffle(void *source);

#ifdef __cplusplus
}
#endif

#endif /* CNUMPY_AHK_H */
