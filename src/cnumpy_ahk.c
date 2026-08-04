/**
 * cnumpy_ahk.c - AHK v2 DllCall-compatible wrapper for cnumpy
 *
 * Design principles:
 * - All exports use C ABI (__cdecl), flat functions
 * - Arrays are opaque handles (Ptr in AHK)
 * - Data passes through caller-allocated buffers (double*)
 * - Status codes: 0 = OK, negative = error
 * - No complex structs cross the ABI boundary
 *
 * AHK usage:
 *   h := DllCall("cnumpy_ahk\cnp_ahk_create", "Int", ndim, "Ptr", shapeBuf, "Int", dtype, "Ptr")
 *   DllCall("cnumpy_ahk\cnp_ahk_set_data", "Ptr", h, "Ptr", dataBuf, "Int64", count)
 *   DllCall("cnumpy_ahk\cnp_ahk_get_data", "Ptr", h, "Ptr", outBuf, "Int64", count)
 *   DllCall("cnumpy_ahk\cnp_ahk_free", "Ptr", h)
 */
#include <cnumpy/cnumpy_internal.h>
#include <cnumpy/cnumpy_ahk.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* =========================================================================
 * Status codes
 * ========================================================================= */
#define AHK_OK              0
#define AHK_ERR_NULL       -1
#define AHK_ERR_SHAPE      -2
#define AHK_ERR_DTYPE      -3
#define AHK_ERR_MEMORY     -4
#define AHK_ERR_SIZE       -5
#define AHK_ERR_AXIS       -6
#define AHK_ERR_GENERIC    -7

/* =========================================================================
 * Lifecycle
 * ========================================================================= */

/* Initialize cnumpy library. Call once before any other function. */
__declspec(dllexport) int __cdecl cnp_ahk_init(void) {
    return (int)cnp_init();
}

/* Cleanup cnumpy library. Call once when done. */
__declspec(dllexport) int __cdecl cnp_ahk_cleanup(void) {
    cnp_cleanup();
    return AHK_OK;
}

/* Get version string. Returns pointer to static string. */
__declspec(dllexport) const char* __cdecl cnp_ahk_version(void) {
    return cnp_version();
}

__declspec(dllexport) int __cdecl cnp_ahk_simd_level(void) {
    return cnp_simd_runtime_level();
}

__declspec(dllexport) int __cdecl cnp_ahk_set_num_threads(int count) {
    return (int)cnp_set_num_threads(count);
}

__declspec(dllexport) int __cdecl cnp_ahk_get_num_threads(void) {
    return cnp_get_num_threads();
}

typedef struct {
    CnpAhkLineCallback callback;
    void *userdata;
    CNP_STATUS callback_status;
} CnpAhkLineContext;

typedef struct {
    CnpAhkCoordinateCallback callback;
    void *userdata;
    CNP_STATUS callback_status;
} CnpAhkCoordinateContext;

typedef struct {
    CnpAhkIteratorCallback callback;
    void *userdata;
    CNP_STATUS callback_status;
} CnpAhkIteratorContext;

typedef struct {
    CnpAhkUnaryCallback callback;
    void *userdata;
    CNP_STATUS callback_status;
} CnpAhkUnaryContext;

static double cnp_ahk_line_adapter(
    const double *line, int64_t length, void *raw_context) {
    CnpAhkLineContext *context = (CnpAhkLineContext*)raw_context;
    if (context->callback_status != CNP_OK) return 0.0;
    double result = 0.0;
    context->callback_status = context->callback(
        line, length, context->userdata, &result);
    return result;
}

static double cnp_ahk_coordinate_adapter(
    const int64_t *coordinates, int ndim, void *raw_context) {
    CnpAhkCoordinateContext *context =
        (CnpAhkCoordinateContext*)raw_context;
    if (context->callback_status != CNP_OK) return 0.0;
    double result = 0.0;
    context->callback_status = context->callback(
        coordinates, ndim, context->userdata, &result);
    return result;
}

static double cnp_ahk_iterator_adapter(void *raw_context) {
    CnpAhkIteratorContext *context = (CnpAhkIteratorContext*)raw_context;
    if (context->callback_status != CNP_OK) return 0.0;
    double result = 0.0;
    context->callback_status = context->callback(
        context->userdata, &result);
    return result;
}

static double cnp_ahk_unary_adapter(double value, void *raw_context) {
    CnpAhkUnaryContext *context = (CnpAhkUnaryContext*)raw_context;
    if (context->callback_status != CNP_OK) return 0.0;
    double result = 0.0;
    context->callback_status = context->callback(
        &value, context->userdata, &result);
    return result;
}

static void *cnp_ahk_callback_result(
    CnpArray *result, CNP_STATUS callback_status,
    const char *function_name) {
    if (callback_status != CNP_OK) {
        if (result) cnp_array_free(result);
        cnp_set_error(
            callback_status, function_name,
            "AutoHotkey callback returned status %d",
            (int)callback_status);
        return NULL;
    }
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

CNP_API void *CNP_CALL cnp_ahk_apply_along_axis(
    CnpAhkLineCallback callback, void *userdata,
    int axis, void *source) {
    const char *function_name = "cnp_ahk_apply_along_axis";
    if (!callback) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "callback is required");
        return NULL;
    }
    CnpAhkLineContext context = {callback, userdata, CNP_OK};
    CnpArray *result = cnp_apply_along_axis(
        cnp_ahk_line_adapter, axis, (const CnpArray*)source, &context);
    return cnp_ahk_callback_result(
        result, context.callback_status, function_name);
}

CNP_API void *CNP_CALL cnp_ahk_apply_over_axes(
    CnpAhkLineCallback callback, void *userdata,
    int naxes, const int *axes, void *source) {
    const char *function_name = "cnp_ahk_apply_over_axes";
    if (!callback) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "callback is required");
        return NULL;
    }
    CnpAhkLineContext context = {callback, userdata, CNP_OK};
    CnpArray *result = cnp_apply_over_axes(
        cnp_ahk_line_adapter, naxes, axes,
        (const CnpArray*)source, &context);
    return cnp_ahk_callback_result(
        result, context.callback_status, function_name);
}

CNP_API void *CNP_CALL cnp_ahk_fromfunction(
    CnpAhkCoordinateCallback callback, void *userdata,
    int ndim, const int64_t *shape) {
    const char *function_name = "cnp_ahk_fromfunction";
    if (!callback) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "callback is required");
        return NULL;
    }
    CnpAhkCoordinateContext context = {callback, userdata, CNP_OK};
    CnpArray *result = cnp_fromfunction(
        cnp_ahk_coordinate_adapter, ndim, shape, &context);
    return cnp_ahk_callback_result(
        result, context.callback_status, function_name);
}

CNP_API void *CNP_CALL cnp_ahk_fromiter(
    CnpAhkIteratorCallback callback, void *userdata,
    int64_t count, int dtype) {
    const char *function_name = "cnp_ahk_fromiter";
    if (!callback) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "callback is required");
        return NULL;
    }
    CnpAhkIteratorContext context = {callback, userdata, CNP_OK};
    CnpArray *result = cnp_fromiter(
        cnp_ahk_iterator_adapter, &context, count, (CNP_TYPE)dtype);
    return cnp_ahk_callback_result(
        result, context.callback_status, function_name);
}

static void *cnp_ahk_unary_call(
    CnpAhkUnaryCallback callback, void *userdata, void *source,
    bool vectorized, const char *function_name) {
    if (!callback) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "callback is required");
        return NULL;
    }
    CnpAhkUnaryContext context = {callback, userdata, CNP_OK};
    CnpArray *result = vectorized
        ? cnp_vectorize(
            cnp_ahk_unary_adapter, (const CnpArray*)source, &context)
        : cnp_frompyfunc(
            cnp_ahk_unary_adapter, (const CnpArray*)source, &context);
    return cnp_ahk_callback_result(
        result, context.callback_status, function_name);
}

CNP_API void *CNP_CALL cnp_ahk_frompyfunc(
    CnpAhkUnaryCallback callback, void *userdata, void *source) {
    return cnp_ahk_unary_call(
        callback, userdata, source, false, "cnp_ahk_frompyfunc");
}

CNP_API void *CNP_CALL cnp_ahk_vectorize(
    CnpAhkUnaryCallback callback, void *userdata, void *source) {
    return cnp_ahk_unary_call(
        callback, userdata, source, true, "cnp_ahk_vectorize");
}

CNP_API void *CNP_CALL cnp_ahk_piecewise(
    void *source, int nconditions, void *const *conditions,
    CnpAhkUnaryCallback callback, void *userdata) {
    const char *function_name = "cnp_ahk_piecewise";
    if (!callback) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "callback is required");
        return NULL;
    }
    CnpAhkUnaryContext context = {callback, userdata, CNP_OK};
    CnpArray *result = cnp_piecewise(
        (const CnpArray*)source, nconditions,
        (const CnpArray**)conditions,
        cnp_ahk_unary_adapter, &context);
    return cnp_ahk_callback_result(
        result, context.callback_status, function_name);
}

static bool cnp_ahk_bulk_real_dtype(
    const CnpArray *source, const char *function_name) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "source array is required");
        return false;
    }
    CNP_TYPE dtype = source->dtype->type_num;
    if (dtype == CNP_BOOL || cnp_type_is_integer(dtype) ||
            cnp_type_is_float(dtype)) {
        return true;
    }
    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "source dtype %d cannot be represented by the real double callback",
        (int)dtype);
    return false;
}

static bool cnp_ahk_bulk_callback_status(
    CNP_STATUS status, int64_t produced_count, int64_t capacity,
    const char *function_name) {
    if (status != CNP_OK) {
        cnp_set_error(
            status, function_name,
            "AutoHotkey bulk callback returned status %d", (int)status);
        return false;
    }
    if (produced_count < 0 || produced_count > capacity) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "bulk callback produced %lld values for capacity %lld",
            (long long)produced_count, (long long)capacity);
        return false;
    }
    return true;
}

static bool cnp_ahk_bulk_callback_exact(
    CNP_STATUS status, int64_t produced_count, int64_t expected_count,
    int64_t capacity, const char *function_name) {
    if (!cnp_ahk_bulk_callback_status(
            status, produced_count, capacity, function_name)) {
        return false;
    }
    if (produced_count != expected_count) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "bulk callback produced %lld values; expected %lld",
            (long long)produced_count, (long long)expected_count);
        return false;
    }
    return true;
}

static bool cnp_ahk_bulk_allocation_size(
    int64_t count, size_t itemsize, size_t *bytes,
    const char *description, const char *function_name) {
    if (count < 0 ||
            (uint64_t)count > (uint64_t)(SIZE_MAX / itemsize)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "%s exceeds addressable memory", description);
        return false;
    }
    size_t positive_count = (size_t)(count > 0 ? count : 1);
    *bytes = positive_count * itemsize;
    return true;
}

static int64_t cnp_ahk_bulk_flat_offset(
    const CnpArray *array, int64_t flat_index) {
    int64_t coordinates[CNP_MAXDIMS] = {0};
    int64_t remaining = flat_index;
    for (int dimension = array->ndim - 1; dimension >= 0; --dimension) {
        coordinates[dimension] = remaining % array->shape[dimension];
        remaining /= array->shape[dimension];
    }
    return array->offset + cnp_multi_to_offset(
        array->ndim, coordinates, array->strides);
}

static void cnp_ahk_bulk_line_coordinates(
    const CnpArray *source, int axis, int64_t line_index,
    int64_t *coordinates) {
    int64_t remaining = line_index;
    for (int dimension = source->ndim - 1; dimension >= 0; --dimension) {
        if (dimension == axis) {
            coordinates[dimension] = 0;
            continue;
        }
        coordinates[dimension] = remaining % source->shape[dimension];
        remaining /= source->shape[dimension];
    }
}

static bool cnp_ahk_bulk_result_shape(
    int result_ndim, const int64_t *result_shape,
    int64_t *result_size, const char *function_name) {
    if (result_ndim < 0 || result_ndim > CNP_MAXDIMS) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "callback result ndim must be in [0, %d], got %d",
            CNP_MAXDIMS, result_ndim);
        return false;
    }
    if (result_ndim > 0 && !result_shape) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "callback result shape is required");
        return false;
    }
    int64_t product = 1;
    for (int dimension = 0; dimension < result_ndim; ++dimension) {
        int64_t length = result_shape[dimension];
        if (length < 0) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "callback result dimension %d is negative", dimension);
            return false;
        }
        if (length != 0 && product != 0 && product > INT64_MAX / length) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "callback result shape product exceeds int64");
            return false;
        }
        product *= length;
    }
    *result_size = product;
    return true;
}

static CnpArray *cnp_ahk_apply_along_axis_bulk(
    CnpAhkLineBatchCallback callback, void *userdata,
    int axis, const CnpArray *source,
    int result_ndim, const int64_t *result_shape,
    const char *function_name) {
    if (!callback) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "callback is required");
        return NULL;
    }
    if (!cnp_ahk_bulk_real_dtype(source, function_name)) return NULL;
    if (axis < 0) axis += source->ndim;
    if (axis < 0 || axis >= source->ndim) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "axis %d is out of bounds for array of dimension %d",
            axis, source->ndim);
        return NULL;
    }
    if (source->ndim - 1 + result_ndim > CNP_MAXDIMS) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "callback result rank exceeds maximum rank %d", CNP_MAXDIMS);
        return NULL;
    }

    int64_t callback_result_size = 0;
    if (!cnp_ahk_bulk_result_shape(
            result_ndim, result_shape,
            &callback_result_size, function_name)) {
        return NULL;
    }

    int64_t line_count = 1;
    for (int dimension = 0; dimension < source->ndim; ++dimension) {
        if (dimension == axis) continue;
        if (source->shape[dimension] == 0) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "cannot apply along an axis when an iteration dimension is zero");
            return NULL;
        }
        if (line_count > INT64_MAX / source->shape[dimension]) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "callback line count exceeds int64");
            return NULL;
        }
        line_count *= source->shape[dimension];
    }

    int output_ndim = source->ndim - 1 + result_ndim;
    int64_t output_shape[CNP_MAXDIMS] = {0};
    int64_t storage_shape[CNP_MAXDIMS] = {0};
    int output_dimension = 0;
    for (int dimension = 0; dimension < axis; ++dimension)
        output_shape[output_dimension++] = source->shape[dimension];
    for (int dimension = 0; dimension < result_ndim; ++dimension)
        output_shape[output_dimension++] = result_shape[dimension];
    for (int dimension = axis + 1;
         dimension < source->ndim; ++dimension) {
        output_shape[output_dimension++] = source->shape[dimension];
    }

    int storage_dimension = 0;
    for (int dimension = 0; dimension < source->ndim; ++dimension) {
        if (dimension != axis)
            storage_shape[storage_dimension++] = source->shape[dimension];
    }
    for (int dimension = 0; dimension < result_ndim; ++dimension)
        storage_shape[storage_dimension++] = result_shape[dimension];

    const int64_t *allocated_shape = result_ndim > 0
        ? storage_shape : output_shape;
    CnpArray *storage = cnp_array_new(
        output_ndim, output_ndim > 0 ? allocated_shape : NULL,
        CNP_DOUBLE, CNP_ORDER_C);
    if (!storage) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t batch_limit = line_count < CNP_AHK_CALLBACK_BATCH_SIZE
        ? line_count : CNP_AHK_CALLBACK_BATCH_SIZE;
    int64_t axis_length = source->shape[axis];
    int64_t line_values = batch_limit;
    if (axis_length > 0 && line_values > INT64_MAX / axis_length) {
        cnp_array_free(storage);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "callback line batch exceeds addressable memory");
        return NULL;
    }
    line_values *= axis_length;
    int64_t result_values = batch_limit;
    if (callback_result_size > 0 &&
            result_values > INT64_MAX / callback_result_size) {
        cnp_array_free(storage);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "callback result batch exceeds addressable memory");
        return NULL;
    }
    result_values *= callback_result_size;

    size_t line_bytes = 0;
    size_t result_bytes = 0;
    if (!cnp_ahk_bulk_allocation_size(
            line_values, sizeof(double), &line_bytes,
            "callback line batch", function_name) ||
        !cnp_ahk_bulk_allocation_size(
            result_values, sizeof(double), &result_bytes,
            "callback result batch", function_name)) {
        cnp_array_free(storage);
        return NULL;
    }
    double *lines = (double*)cnp_malloc(line_bytes);
    double *callback_results = (double*)cnp_malloc(result_bytes);
    if (!lines || !callback_results) {
        if (callback_results) cnp_free(callback_results, result_bytes);
        if (lines) cnp_free(lines, line_bytes);
        cnp_array_free(storage);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate bulk callback buffers");
        return NULL;
    }

    for (int64_t first_line = 0;
         first_line < line_count; first_line += batch_limit) {
        int64_t current_lines = line_count - first_line;
        if (current_lines > batch_limit) current_lines = batch_limit;
        for (int64_t local_line = 0;
             local_line < current_lines; ++local_line) {
            int64_t coordinates[CNP_MAXDIMS] = {0};
            cnp_ahk_bulk_line_coordinates(
                source, axis, first_line + local_line, coordinates);
            for (int64_t item = 0; item < axis_length; ++item) {
                coordinates[axis] = item;
                int64_t offset = source->offset + cnp_multi_to_offset(
                    source->ndim, coordinates, source->strides);
                lines[local_line * axis_length + item] =
                    cnp_get_element_double(
                        source->data, offset, source->dtype->type_num);
            }
        }

        int64_t capacity = current_lines * callback_result_size;
        int64_t produced_count = -1;
        CNP_STATUS status = callback(
            lines, current_lines, axis_length,
            callback_results, capacity, &produced_count, userdata);
        if (!cnp_ahk_bulk_callback_exact(
                status, produced_count, capacity, capacity,
                function_name)) {
            cnp_free(callback_results, result_bytes);
            cnp_free(lines, line_bytes);
            cnp_array_free(storage);
            return NULL;
        }
        memcpy(
            (double*)storage->data + first_line * callback_result_size,
            callback_results, (size_t)capacity * sizeof(double));
    }

    cnp_free(callback_results, result_bytes);
    cnp_free(lines, line_bytes);
    if (result_ndim == 0) return storage;

    int64_t output_strides[CNP_MAXDIMS] = {0};
    for (int dimension = 0; dimension < axis; ++dimension)
        output_strides[dimension] = storage->strides[dimension];
    int callback_stride_start = source->ndim - 1;
    for (int dimension = 0; dimension < result_ndim; ++dimension) {
        output_strides[axis + dimension] =
            storage->strides[callback_stride_start + dimension];
    }
    int post_storage_dimension = axis;
    for (int dimension = axis + result_ndim;
         dimension < output_ndim; ++dimension) {
        output_strides[dimension] =
            storage->strides[post_storage_dimension++];
    }
    CnpArray *result = cnp_array_view_from_metadata(
        storage, output_ndim, output_shape, output_strides,
        storage->offset, 0);
    cnp_array_decref(storage);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    return result;
}

CNP_API void *CNP_CALL cnp_ahk_apply_along_axis_v2(
    CnpAhkLineBatchCallback callback, void *userdata,
    int axis, void *source,
    int result_ndim, const int64_t *result_shape) {
    return (void*)cnp_ahk_apply_along_axis_bulk(
        callback, userdata, axis, (const CnpArray*)source,
        result_ndim, result_shape, "cnp_ahk_apply_along_axis_v2");
}

CNP_API void *CNP_CALL cnp_ahk_apply_over_axes_v2(
    CnpAhkLineBatchCallback callback, void *userdata,
    int naxes, const int *axes, void *source) {
    const char *function_name = "cnp_ahk_apply_over_axes_v2";
    if (!callback) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "callback is required");
        return NULL;
    }
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "source array is required");
        return NULL;
    }
    if (naxes < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "axis count must be non-negative");
        return NULL;
    }
    if (naxes > 0 && !axes) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "axes are required");
        return NULL;
    }

    CnpArray *current = cnp_array_copy((const CnpArray*)source);
    if (!current) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int index = 0; index < naxes; ++index) {
        int axis = axes[index];
        if (axis < 0) axis += current->ndim;
        if (axis < 0 || axis >= current->ndim) {
            cnp_array_free(current);
            cnp_set_error(
                CNP_ERR_AXIS, function_name,
                "axis %d is out of bounds for array of dimension %d",
                axes[index], ((const CnpArray*)source)->ndim);
            return NULL;
        }
        CnpArray *reduced = cnp_ahk_apply_along_axis_bulk(
            callback, userdata, axis, current, 0, NULL, function_name);
        cnp_array_free(current);
        if (!reduced) return NULL;
        current = cnp_expand_dims(reduced, axis);
        cnp_array_decref(reduced);
        if (!current) {
            cnp_relabel_error(function_name);
            return NULL;
        }
    }
    return (void*)current;
}

CNP_API void *CNP_CALL cnp_ahk_fromfunction_v2(
    CnpAhkCoordinateBatchCallback callback, void *userdata,
    int ndim, const int64_t *shape) {
    const char *function_name = "cnp_ahk_fromfunction_v2";
    if (!callback) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "callback is required");
        return NULL;
    }
    if (ndim < 0 || ndim > CNP_MAXDIMS) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "ndim must be in [0, %d], got %d", CNP_MAXDIMS, ndim);
        return NULL;
    }
    if (ndim > 0 && !shape) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "shape is required");
        return NULL;
    }

    CnpArray *result = cnp_array_new(
        ndim, ndim > 0 ? shape : NULL, CNP_DOUBLE, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (result->size == 0) return (void*)result;

    int64_t batch_limit = result->size < CNP_AHK_CALLBACK_BATCH_SIZE
        ? result->size : CNP_AHK_CALLBACK_BATCH_SIZE;
    int64_t coordinate_count = batch_limit;
    if (ndim > 0 && coordinate_count > INT64_MAX / ndim) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "coordinate callback batch exceeds addressable memory");
        return NULL;
    }
    coordinate_count *= ndim;
    size_t coordinate_bytes = 0;
    size_t result_bytes = 0;
    if (!cnp_ahk_bulk_allocation_size(
            coordinate_count, sizeof(int64_t), &coordinate_bytes,
            "coordinate callback batch", function_name) ||
        !cnp_ahk_bulk_allocation_size(
            batch_limit, sizeof(double), &result_bytes,
            "coordinate result batch", function_name)) {
        cnp_array_free(result);
        return NULL;
    }
    int64_t *coordinates = (int64_t*)cnp_malloc(coordinate_bytes);
    double *callback_results = (double*)cnp_malloc(result_bytes);
    if (!coordinates || !callback_results) {
        if (callback_results) cnp_free(callback_results, result_bytes);
        if (coordinates) cnp_free(coordinates, coordinate_bytes);
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate coordinate callback buffers");
        return NULL;
    }

    int64_t current_coordinates[CNP_MAXDIMS] = {0};
    double *destination = (double*)result->data;
    for (int64_t first = 0; first < result->size; first += batch_limit) {
        int64_t count = result->size - first;
        if (count > batch_limit) count = batch_limit;
        for (int64_t point = 0; point < count; ++point) {
            for (int dimension = 0; dimension < ndim; ++dimension) {
                coordinates[point * ndim + dimension] =
                    current_coordinates[dimension];
            }
            for (int dimension = ndim - 1; dimension >= 0; --dimension) {
                ++current_coordinates[dimension];
                if (current_coordinates[dimension] < shape[dimension]) break;
                current_coordinates[dimension] = 0;
            }
        }
        int64_t produced_count = -1;
        CNP_STATUS status = callback(
            coordinates, count, ndim,
            callback_results, count, &produced_count, userdata);
        if (!cnp_ahk_bulk_callback_exact(
                status, produced_count, count, count, function_name)) {
            cnp_free(callback_results, result_bytes);
            cnp_free(coordinates, coordinate_bytes);
            cnp_array_free(result);
            return NULL;
        }
        memcpy(
            destination + first, callback_results,
            (size_t)count * sizeof(double));
    }

    cnp_free(callback_results, result_bytes);
    cnp_free(coordinates, coordinate_bytes);
    return (void*)result;
}

static bool cnp_ahk_fromiter_dtype(int dtype, const char *function_name) {
    CNP_TYPE type = (CNP_TYPE)dtype;
    if (type == CNP_BOOL || cnp_type_is_integer(type) ||
            cnp_type_is_float(type) || cnp_type_is_complex(type)) {
        return true;
    }
    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "dtype %d cannot be constructed from double callback values", dtype);
    return false;
}

static bool cnp_ahk_fromiter_grow(
    double **storage, int64_t *capacity, int64_t required,
    const char *function_name) {
    if (required <= *capacity) return true;
    int64_t new_capacity = *capacity > 0
        ? *capacity : CNP_AHK_CALLBACK_BATCH_SIZE;
    while (new_capacity < required) {
        if (new_capacity > INT64_MAX / 2) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "iterator result exceeds int64 capacity");
            return false;
        }
        new_capacity *= 2;
    }
    size_t new_bytes = 0;
    if (!cnp_ahk_bulk_allocation_size(
            new_capacity, sizeof(double), &new_bytes,
            "iterator result", function_name)) {
        return false;
    }
    double *replacement = (double*)cnp_malloc(new_bytes);
    if (!replacement) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to grow iterator result buffer");
        return false;
    }
    if (*storage) {
        size_t old_bytes = (size_t)*capacity * sizeof(double);
        memcpy(replacement, *storage, old_bytes);
        cnp_free(*storage, old_bytes);
    }
    *storage = replacement;
    *capacity = new_capacity;
    return true;
}

CNP_API void *CNP_CALL cnp_ahk_fromiter_v2(
    CnpAhkIteratorBatchCallback callback, void *userdata,
    int64_t count, int dtype) {
    const char *function_name = "cnp_ahk_fromiter_v2";
    if (!callback) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "iterator callback is required");
        return NULL;
    }
    if (count < -1) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "count must be -1 or non-negative");
        return NULL;
    }
    if (!cnp_ahk_fromiter_dtype(dtype, function_name)) return NULL;

    size_t batch_bytes =
        (size_t)CNP_AHK_CALLBACK_BATCH_SIZE * sizeof(double);
    double *batch = (double*)cnp_malloc(batch_bytes);
    if (!batch) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate iterator callback buffer");
        return NULL;
    }

    double *storage = NULL;
    int64_t storage_capacity = 0;
    int64_t total = 0;
    bool unknown_count = count == -1;
    while (unknown_count || total < count) {
        int64_t capacity = CNP_AHK_CALLBACK_BATCH_SIZE;
        if (!unknown_count && count - total < capacity)
            capacity = count - total;
        int64_t produced_count = -1;
        CNP_STATUS status = callback(
            batch, capacity, &produced_count, userdata);
        if (!cnp_ahk_bulk_callback_status(
                status, produced_count, capacity, function_name)) {
            if (storage) cnp_free(
                storage, (size_t)storage_capacity * sizeof(double));
            cnp_free(batch, batch_bytes);
            return NULL;
        }
        if (!unknown_count && produced_count != capacity) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "iterator callback produced %lld values; expected %lld",
                (long long)produced_count, (long long)capacity);
            if (storage) cnp_free(
                storage, (size_t)storage_capacity * sizeof(double));
            cnp_free(batch, batch_bytes);
            return NULL;
        }
        if (produced_count > 0) {
            if (total > INT64_MAX - produced_count ||
                !cnp_ahk_fromiter_grow(
                    &storage, &storage_capacity,
                    total + produced_count, function_name)) {
                if (storage) cnp_free(
                    storage, (size_t)storage_capacity * sizeof(double));
                cnp_free(batch, batch_bytes);
                return NULL;
            }
            memcpy(
                storage + total, batch,
                (size_t)produced_count * sizeof(double));
            total += produced_count;
        }
        if (unknown_count && produced_count < capacity) break;
    }
    cnp_free(batch, batch_bytes);

    int64_t shape[1] = {total};
    CnpArray *result = cnp_array_new(
        1, shape, (CNP_TYPE)dtype, CNP_ORDER_C);
    if (!result) {
        if (storage) cnp_free(
            storage, (size_t)storage_capacity * sizeof(double));
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int64_t index = 0; index < total; ++index) {
        cnp_set_element_double(
            result->data, index * result->dtype->elsize,
            result->dtype->type_num, storage[index]);
    }
    if (storage) cnp_free(
        storage, (size_t)storage_capacity * sizeof(double));
    return (void*)result;
}

static CnpArray *cnp_ahk_unary_bulk(
    CnpAhkUnaryBatchCallback callback, void *userdata,
    const CnpArray *source, const char *function_name) {
    if (!callback) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "callback is required");
        return NULL;
    }
    if (!cnp_ahk_bulk_real_dtype(source, function_name)) return NULL;
    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
            ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, CNP_DOUBLE, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (source->size == 0) return result;

    size_t batch_bytes =
        (size_t)CNP_AHK_CALLBACK_BATCH_SIZE * sizeof(double);
    double *inputs = (double*)cnp_malloc(batch_bytes);
    double *outputs = (double*)cnp_malloc(batch_bytes);
    if (!inputs || !outputs) {
        if (outputs) cnp_free(outputs, batch_bytes);
        if (inputs) cnp_free(inputs, batch_bytes);
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate unary callback buffers");
        return NULL;
    }

    bool contiguous_output =
        (result->flags & CNP_ARRAY_C_CONTIGUOUS) != 0;
    for (int64_t first = 0;
         first < source->size; first += CNP_AHK_CALLBACK_BATCH_SIZE) {
        int64_t count = source->size - first;
        if (count > CNP_AHK_CALLBACK_BATCH_SIZE)
            count = CNP_AHK_CALLBACK_BATCH_SIZE;
        for (int64_t index = 0; index < count; ++index)
            inputs[index] = cnp_array_flat_get(source, first + index);
        int64_t produced_count = -1;
        CNP_STATUS status = callback(
            inputs, count, outputs, count, &produced_count, userdata);
        if (!cnp_ahk_bulk_callback_exact(
                status, produced_count, count, count, function_name)) {
            cnp_free(outputs, batch_bytes);
            cnp_free(inputs, batch_bytes);
            cnp_array_free(result);
            return NULL;
        }
        if (contiguous_output) {
            memcpy(
                (double*)result->data + first, outputs,
                (size_t)count * sizeof(double));
        } else {
            for (int64_t index = 0; index < count; ++index) {
                int64_t offset = cnp_ahk_bulk_flat_offset(
                    result, first + index);
                *(double*)((char*)result->data + offset) = outputs[index];
            }
        }
    }

    cnp_free(outputs, batch_bytes);
    cnp_free(inputs, batch_bytes);
    return result;
}

CNP_API void *CNP_CALL cnp_ahk_frompyfunc_v2(
    CnpAhkUnaryBatchCallback callback, void *userdata, void *source) {
    return (void*)cnp_ahk_unary_bulk(
        callback, userdata, (const CnpArray*)source,
        "cnp_ahk_frompyfunc_v2");
}

CNP_API void *CNP_CALL cnp_ahk_vectorize_v2(
    CnpAhkUnaryBatchCallback callback, void *userdata, void *source) {
    return (void*)cnp_ahk_unary_bulk(
        callback, userdata, (const CnpArray*)source,
        "cnp_ahk_vectorize_v2");
}

static bool cnp_ahk_piecewise_conditions(
    const CnpArray *source, int nconditions,
    void *const *conditions, const char *function_name) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "x array is required");
        return false;
    }
    if (nconditions < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "condition count must be non-negative");
        return false;
    }
    if (nconditions > 0 && !conditions) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "condition list is required");
        return false;
    }
    for (int condition = 0; condition < nconditions; ++condition) {
        const CnpArray *current = (const CnpArray*)conditions[condition];
        if (!current) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "condition %d is required", condition);
            return false;
        }
        bool scalar = current->ndim == 0;
        bool same_shape = current->ndim == source->ndim;
        for (int dimension = 0;
             same_shape && dimension < source->ndim; ++dimension) {
            same_shape =
                current->shape[dimension] == source->shape[dimension];
        }
        if (!scalar && !same_shape) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "condition %d must be scalar or have the exact x shape",
                condition);
            return false;
        }
    }
    return true;
}

CNP_API void *CNP_CALL cnp_ahk_piecewise_v2(
    void *source, int nconditions, void *const *conditions,
    CnpAhkUnaryBatchCallback callback, void *userdata) {
    const char *function_name = "cnp_ahk_piecewise_v2";
    const CnpArray *input = (const CnpArray*)source;
    if (!cnp_ahk_piecewise_conditions(
            input, nconditions, conditions, function_name)) {
        return NULL;
    }
    if (nconditions > 0 && !callback) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "callback is required");
        return NULL;
    }
    if (!cnp_ahk_bulk_real_dtype(input, function_name)) return NULL;

    CnpArray *result = cnp_array_zeros(
        input->ndim, input->shape,
        input->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (nconditions == 0 || input->size == 0) return (void*)result;

    size_t value_bytes =
        (size_t)CNP_AHK_CALLBACK_BATCH_SIZE * sizeof(double);
    size_t index_bytes =
        (size_t)CNP_AHK_CALLBACK_BATCH_SIZE * sizeof(int64_t);
    double *inputs = (double*)cnp_malloc(value_bytes);
    double *outputs = (double*)cnp_malloc(value_bytes);
    int64_t *destinations = (int64_t*)cnp_malloc(index_bytes);
    if (!inputs || !outputs || !destinations) {
        if (destinations) cnp_free(destinations, index_bytes);
        if (outputs) cnp_free(outputs, value_bytes);
        if (inputs) cnp_free(inputs, value_bytes);
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate piecewise callback buffers");
        return NULL;
    }

    for (int condition = 0; condition < nconditions; ++condition) {
        const CnpArray *current = (const CnpArray*)conditions[condition];
        int64_t source_index = 0;
        while (source_index < input->size) {
            int64_t gathered = 0;
            while (source_index < input->size &&
                   gathered < CNP_AHK_CALLBACK_BATCH_SIZE) {
                int64_t condition_offset = current->ndim == 0
                    ? current->offset
                    : cnp_ahk_bulk_flat_offset(current, source_index);
                uint8_t truth = 0;
                CNP_STATUS cast_status = cnp_cast_scalar_value(
                    (const char*)current->data + condition_offset,
                    current->dtype->type_num,
                    &truth, CNP_BOOL, function_name);
                if (cast_status != CNP_OK) {
                    cnp_free(destinations, index_bytes);
                    cnp_free(outputs, value_bytes);
                    cnp_free(inputs, value_bytes);
                    cnp_array_free(result);
                    return NULL;
                }
                if (truth) {
                    destinations[gathered] = source_index;
                    inputs[gathered] =
                        cnp_array_flat_get(input, source_index);
                    ++gathered;
                }
                ++source_index;
            }
            if (gathered == 0) continue;
            int64_t produced_count = -1;
            CNP_STATUS status = callback(
                inputs, gathered, outputs, gathered,
                &produced_count, userdata);
            if (!cnp_ahk_bulk_callback_exact(
                    status, produced_count, gathered, gathered,
                    function_name)) {
                cnp_free(destinations, index_bytes);
                cnp_free(outputs, value_bytes);
                cnp_free(inputs, value_bytes);
                cnp_array_free(result);
                return NULL;
            }
            for (int64_t index = 0; index < gathered; ++index) {
                cnp_set_element_double(
                    result->data,
                    destinations[index] * result->dtype->elsize,
                    result->dtype->type_num, outputs[index]);
            }
        }
    }

    cnp_free(destinations, index_bytes);
    cnp_free(outputs, value_bytes);
    cnp_free(inputs, value_bytes);
    return (void*)result;
}

/* =========================================================================
 * Array creation / destruction
 * ========================================================================= */

/*
 * Create a new array with given shape and dtype.
 * shape_buf: int64 array of ndim elements
 * dtype: CNP_TYPE enum value (1=bool, 2=int8, 3=uint8, 4=int16, 5=uint16,
 *        6=int32, 7=uint32, 8=int64/long, 9=uint64/ulong, 10=int64/longlong,
 *        11=uint64/ulONGLONG, 12=float32, 13=float64, 14=longdouble,
 *        15=complex64, 16=complex128)
 * Returns: opaque handle (Ptr), or NULL on failure
 */
__declspec(dllexport) void* __cdecl cnp_ahk_create(int ndim, const int64_t *shape_buf, int dtype) {
    if (ndim <= 0 || ndim > 32 || !shape_buf) return NULL;
    CnpArray *arr = cnp_array_zeros(ndim, (int64_t*)shape_buf, (CNP_TYPE)dtype, CNP_ORDER_C);
    return (void*)arr;
}

/* Create array from double data buffer. Data is copied. */
__declspec(dllexport) void* __cdecl cnp_ahk_from_doubles(int ndim, const int64_t *shape_buf,
                                                          const double *data, int64_t count) {
    if (ndim < 0 || (ndim > 0 && !shape_buf) || !data || count <= 0) return NULL;
    if (cnp_init() != CNP_OK) return NULL;
    CnpArray *arr = cnp_array_new(ndim, (int64_t*)shape_buf, CNP_DOUBLE, CNP_ORDER_C);
    if (!arr) return NULL;
    int64_t copy_count = (count < arr->size) ? count : arr->size;
    memcpy(arr->data, data, copy_count * sizeof(double));
    return (void*)arr;
}

/* Create array from int64 data buffer. */
__declspec(dllexport) void* __cdecl cnp_ahk_from_ints(int ndim, const int64_t *shape_buf,
                                                       const int64_t *data, int64_t count) {
    if (ndim < 0 || (ndim > 0 && !shape_buf) || !data || count <= 0) return NULL;
    if (cnp_init() != CNP_OK) return NULL;
    CnpArray *arr = cnp_array_new(ndim, (int64_t*)shape_buf, CNP_LONGLONG, CNP_ORDER_C);
    if (!arr) return NULL;
    int64_t copy_count = (count < arr->size) ? count : arr->size;
    memcpy(arr->data, data, copy_count * sizeof(int64_t));
    return (void*)arr;
}

/* Free an array handle. */
__declspec(dllexport) int __cdecl cnp_ahk_free(void *handle) {
    if (!handle) return AHK_ERR_NULL;
    cnp_clear_error();
    cnp_array_decref((CnpArray*)handle);
    return (int)cnp_get_error(NULL);
}

/* =========================================================================
 * Array properties
 * ========================================================================= */

/* Get number of dimensions. */
__declspec(dllexport) int __cdecl cnp_ahk_ndim(void *handle) {
    if (!handle) return -1;
    return ((CnpArray*)handle)->ndim;
}

/* Get total element count. */
__declspec(dllexport) int64_t __cdecl cnp_ahk_size(void *handle) {
    if (!handle) return -1;
    return ((CnpArray*)handle)->size;
}

/* Get shape into caller buffer. Returns ndim. */
__declspec(dllexport) int __cdecl cnp_ahk_shape(void *handle, int64_t *out_buf) {
    if (!handle || !out_buf) return -1;
    CnpArray *arr = (CnpArray*)handle;
    for (int d = 0; d < arr->ndim; d++) out_buf[d] = arr->shape[d];
    return arr->ndim;
}

/* Get dtype type number. */
__declspec(dllexport) int __cdecl cnp_ahk_dtype(void *handle) {
    if (!handle) return -1;
    return (int)((CnpArray*)handle)->dtype->type_num;
}

/* Get read-only array flags for ownership/layout diagnostics. */
__declspec(dllexport) uint32_t __cdecl cnp_ahk_flags(void *handle) {
    if (!handle) return 0;
    return ((CnpArray*)handle)->flags;
}

/* Report storage sharing without exposing a writable data pointer. */
__declspec(dllexport) int __cdecl cnp_ahk_shares_data(void *a, void *b) {
    if (!a || !b) return 0;
    CnpArray *left = (CnpArray*)a;
    CnpArray *right = (CnpArray*)b;
    return left->data == right->data && left->offset == right->offset;
}

/* Get element size in bytes. */
__declspec(dllexport) int __cdecl cnp_ahk_itemsize(void *handle) {
    if (!handle) return -1;
    return (int)((CnpArray*)handle)->dtype->elsize;
}

typedef char CnpAhkMetadataSizeCheck[
    sizeof(CnpAhkMetadata) == CNP_AHK_METADATA_SIZE ? 1 : -1];

__declspec(dllexport) int __cdecl cnp_ahk_get_metadata(
    void *handle, CnpAhkMetadata *out_metadata, uint32_t metadata_size) {
    const char *function_name = "cnp_ahk_get_metadata";
    if (!handle || !out_metadata) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "array handle and metadata output must not be null");
        return CNP_ERR_GENERIC;
    }
    if (metadata_size < CNP_AHK_METADATA_SIZE) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "metadata buffer is %u bytes; %u bytes are required",
                      metadata_size, CNP_AHK_METADATA_SIZE);
        return CNP_ERR_GENERIC;
    }

    CnpArray *array = (CnpArray*)handle;
    CnpAhkMetadata metadata;
    memset(&metadata, 0, sizeof(metadata));
    metadata.abi_version = CNP_AHK_METADATA_ABI_VERSION;
    metadata.struct_size = CNP_AHK_METADATA_SIZE;
    metadata.ndim = array->ndim;
    metadata.dtype = (int32_t)array->dtype->type_num;
    metadata.itemsize = array->dtype->elsize;
    metadata.flags = array->flags;
    metadata.size = array->size;
    for (int dimension = 0; dimension < array->ndim; ++dimension)
        metadata.shape[dimension] = array->shape[dimension];
    memcpy(out_metadata, &metadata, sizeof(metadata));
    return CNP_OK;
}

/* =========================================================================
 * Data access
 * ========================================================================= */

/* Copy array data as doubles into caller buffer. Returns count copied. */
__declspec(dllexport) int64_t __cdecl cnp_ahk_get_doubles(void *handle, double *out_buf, int64_t max_count) {
    if (!handle || !out_buf || max_count <= 0) return -1;
    CnpArray *arr = (CnpArray*)handle;
    int64_t count = (arr->size < max_count) ? arr->size : max_count;
    for (int64_t i = 0; i < count; i++) {
        out_buf[i] = cnp_array_flat_get(arr, i);
    }
    return count;
}

/* Set array data from doubles buffer. Returns count set. */
__declspec(dllexport) int64_t __cdecl cnp_ahk_set_doubles(void *handle, const double *data, int64_t count) {
    if (!handle || !data || count <= 0) return -1;
    CnpArray *arr = (CnpArray*)handle;
    int64_t n = (arr->size < count) ? arr->size : count;
    for (int64_t i = 0; i < n; i++) {
        CNP_STATUS status = cnp_array_flat_set(arr, i, data[i]);
        if (status != CNP_OK) return status;
    }
    return n;
}

/* Get single element as double. */
__declspec(dllexport) double __cdecl cnp_ahk_get_item(void *handle, int64_t index) {
    return cnp_array_flat_get((CnpArray*)handle, index);
}

/* Set single element from double. */
__declspec(dllexport) int __cdecl cnp_ahk_set_item(void *handle, int64_t index, double value) {
    return cnp_array_flat_set((CnpArray*)handle, index, value);
}

/* =========================================================================
 * Array creation shortcuts
 * ========================================================================= */

__declspec(dllexport) void* __cdecl cnp_ahk_zeros(int ndim, const int64_t *shape_buf) {
    if (ndim <= 0 || !shape_buf) return NULL;
    return (void*)cnp_array_zeros(ndim, (int64_t*)shape_buf, CNP_DOUBLE, CNP_ORDER_C);
}

__declspec(dllexport) void* __cdecl cnp_ahk_ones(int ndim, const int64_t *shape_buf) {
    if (ndim <= 0 || !shape_buf) return NULL;
    return (void*)cnp_array_ones(ndim, (int64_t*)shape_buf, CNP_DOUBLE, CNP_ORDER_C);
}

__declspec(dllexport) void* __cdecl cnp_ahk_full(int ndim, const int64_t *shape_buf, double value) {
    if (ndim <= 0 || !shape_buf) return NULL;
    return (void*)cnp_array_full(ndim, (int64_t*)shape_buf, value, CNP_DOUBLE, CNP_ORDER_C);
}

__declspec(dllexport) void* __cdecl cnp_ahk_arange(double start, double stop, double step) {
    return (void*)cnp_arange(start, stop, step, CNP_DOUBLE);
}

__declspec(dllexport) void* __cdecl cnp_ahk_linspace(double start, double stop, int64_t num) {
    return (void*)cnp_linspace(start, stop, num, 1, CNP_DOUBLE);
}

__declspec(dllexport) void* __cdecl cnp_ahk_eye(int64_t n, int64_t m, int k) {
    if (m <= 0) m = n;
    return (void*)cnp_eye(n, m, k, CNP_DOUBLE);
}

__declspec(dllexport) void* __cdecl cnp_ahk_identity(int64_t n) {
    return (void*)cnp_identity(n, CNP_DOUBLE);
}

/* =========================================================================
 * Math operations (unary) - return new array
 * ========================================================================= */

#define UNARY_OP(name, func) \
__declspec(dllexport) void* __cdecl cnp_ahk_##name(void *handle) { \
    if (!handle) return NULL; \
    return (void*)func((CnpArray*)handle); \
}

__declspec(dllexport) void* __cdecl cnp_ahk_negative(void *source) {
    const char *function_name = "cnp_ahk_negative";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_negative((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_positive(void *source) {
    const char *function_name = "cnp_ahk_positive";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_positive((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

UNARY_OP(absolute, cnp_absolute)
UNARY_OP(fabs, cnp_fabs)

__declspec(dllexport) void* __cdecl cnp_ahk_sqrt(void *source) {
    const char *function_name = "cnp_ahk_sqrt";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_sqrt((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_cbrt(void *source) {
    const char *function_name = "cnp_ahk_cbrt";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_cbrt((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_cos(void *source) {
    const char *function_name = "cnp_ahk_cos";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_cos((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_sin(void *source) {
    const char *function_name = "cnp_ahk_sin";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_sin((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_tan(void *source) {
    const char *function_name = "cnp_ahk_tan";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_tan((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_sinh(void *source) {
    const char *function_name = "cnp_ahk_sinh";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_sinh((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_cosh(void *source) {
    const char *function_name = "cnp_ahk_cosh";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_cosh((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_tanh(void *source) {
    const char *function_name = "cnp_ahk_tanh";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_tanh((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_arcsinh(void *source) {
    const char *function_name = "cnp_ahk_arcsinh";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_arcsinh((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_arccosh(void *source) {
    const char *function_name = "cnp_ahk_arccosh";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_arccosh((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_arctanh(void *source) {
    const char *function_name = "cnp_ahk_arctanh";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_arctanh((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_arcsin(void *source) {
    const char *function_name = "cnp_ahk_arcsin";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_arcsin((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_arccos(void *source) {
    const char *function_name = "cnp_ahk_arccos";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_arccos((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_arctan(void *source) {
    const char *function_name = "cnp_ahk_arctan";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_arctan((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_conj(void *source) {
    const char *function_name = "cnp_ahk_conj";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_conj((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_conjugate(void *source) {
    const char *function_name = "cnp_ahk_conjugate";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_conjugate((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_square(void *source) {
    const char *function_name = "cnp_ahk_square";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_square((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_exp(void *source) {
    const char *function_name = "cnp_ahk_exp";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_exp((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_exp2(void *source) {
    const char *function_name = "cnp_ahk_exp2";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_exp2((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_expm1(void *source) {
    const char *function_name = "cnp_ahk_expm1";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_expm1((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_log(void *source) {
    const char *function_name = "cnp_ahk_log";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_log((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_log2(void *source) {
    const char *function_name = "cnp_ahk_log2";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_log2((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_log10(void *source) {
    const char *function_name = "cnp_ahk_log10";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_log10((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_log1p(void *source) {
    const char *function_name = "cnp_ahk_log1p";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_log1p((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_logaddexp(
    void *left, void *right) {
    const char *function_name = "cnp_ahk_logaddexp";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_logaddexp(
        (CnpArray*)left, (CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_logaddexp2(
    void *left, void *right) {
    const char *function_name = "cnp_ahk_logaddexp2";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_logaddexp2(
        (CnpArray*)left, (CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

UNARY_OP(floor, cnp_floor)
UNARY_OP(ceil, cnp_ceil)
UNARY_OP(rint, cnp_rint)
UNARY_OP(fix, cnp_fix)
UNARY_OP(trunc, cnp_trunc)

__declspec(dllexport) void* __cdecl cnp_ahk_sign(void *source) {
    const char *function_name = "cnp_ahk_sign";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_sign((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_reciprocal(void *source) {
    const char *function_name = "cnp_ahk_reciprocal";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_reciprocal((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

/* =========================================================================
 * Math operations (binary) - return new array
 * ========================================================================= */

#define BINARY_OP(name, func) \
__declspec(dllexport) void* __cdecl cnp_ahk_##name(void *a, void *b) { \
    if (!a || !b) return NULL; \
    return (void*)func((CnpArray*)a, (CnpArray*)b); \
}

__declspec(dllexport) void* __cdecl cnp_ahk_add(
    void *left, void *right) {
    const char *function_name = "cnp_ahk_add";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_add(
        (CnpArray*)left, (CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_subtract(
    void *left, void *right) {
    const char *function_name = "cnp_ahk_subtract";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_subtract(
        (CnpArray*)left, (CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_multiply(
    void *left, void *right) {
    const char *function_name = "cnp_ahk_multiply";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_multiply(
        (CnpArray*)left, (CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_divide(
    void *left, void *right) {
    const char *function_name = "cnp_ahk_divide";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_divide(
        (CnpArray*)left, (CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_true_divide(
    void *left, void *right) {
    const char *function_name = "cnp_ahk_true_divide";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_true_divide(
        (CnpArray*)left, (CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_floor_divide(
    void *left, void *right) {
    const char *function_name = "cnp_ahk_floor_divide";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_floor_divide(
        (CnpArray*)left, (CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_power(
    void *left, void *right) {
    const char *function_name = "cnp_ahk_power";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_power(
        (CnpArray*)left, (CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_float_power(
    void *left, void *right) {
    const char *function_name = "cnp_ahk_float_power";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_float_power(
        (CnpArray*)left, (CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_heaviside(
    void *left, void *right) {
    const char *function_name = "cnp_ahk_heaviside";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_heaviside(
        (CnpArray*)left, (CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_mod(
    void *left, void *right) {
    const char *function_name = "cnp_ahk_mod";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_mod(
        (CnpArray*)left, (CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_remainder(
    void *left, void *right) {
    const char *function_name = "cnp_ahk_remainder";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_remainder(
        (CnpArray*)left, (CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) int __cdecl cnp_ahk_divmod(
    void *left, void *right, void **results, int result_capacity) {
    const char *function_name = "cnp_ahk_divmod";
    CnpArray *quotient = NULL;
    CnpArray *remainder = NULL;
    CNP_STATUS status;

    if (results && result_capacity > 0) results[0] = NULL;
    if (results && result_capacity > 1) results[1] = NULL;
    if (!results) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "result buffer must not be null");
        return CNP_ERR_GENERIC;
    }
    if (result_capacity < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "result buffer capacity must be at least two");
        return CNP_ERR_SHAPE;
    }

    status = cnp_divmod(
        (const CnpArray*)left,
        (const CnpArray*)right,
        &quotient,
        &remainder);
    if (status != CNP_OK) {
        cnp_relabel_error(function_name);
        return status;
    }
    results[0] = quotient;
    results[1] = remainder;
    return CNP_OK;
}

__declspec(dllexport) void* __cdecl cnp_ahk_fmod(
    void *left, void *right) {
    const char *function_name = "cnp_ahk_fmod";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_fmod(
        (CnpArray*)left, (CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

typedef CnpArray* (CNP_CALL *CnpComparisonFunction)(
    const CnpArray *left, const CnpArray *right);

static void* cnp_ahk_comparison(
    void *left,
    void *right,
    CnpComparisonFunction function,
    const char *function_name) {
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = function(
        (const CnpArray*)left, (const CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_equal(
    void *left, void *right) {
    return cnp_ahk_comparison(
        left, right, cnp_equal, "cnp_ahk_equal");
}

__declspec(dllexport) void* __cdecl cnp_ahk_not_equal(
    void *left, void *right) {
    return cnp_ahk_comparison(
        left, right, cnp_not_equal, "cnp_ahk_not_equal");
}

__declspec(dllexport) void* __cdecl cnp_ahk_less(
    void *left, void *right) {
    return cnp_ahk_comparison(
        left, right, cnp_less, "cnp_ahk_less");
}

__declspec(dllexport) void* __cdecl cnp_ahk_less_equal(
    void *left, void *right) {
    return cnp_ahk_comparison(
        left, right, cnp_less_equal, "cnp_ahk_less_equal");
}

__declspec(dllexport) void* __cdecl cnp_ahk_greater(
    void *left, void *right) {
    return cnp_ahk_comparison(
        left, right, cnp_greater, "cnp_ahk_greater");
}

__declspec(dllexport) void* __cdecl cnp_ahk_greater_equal(
    void *left, void *right) {
    return cnp_ahk_comparison(
        left, right, cnp_greater_equal, "cnp_ahk_greater_equal");
}

__declspec(dllexport) void* __cdecl cnp_ahk_arctan2(
    void *y, void *x) {
    const char *function_name = "cnp_ahk_arctan2";
    if (!y || !x) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "y and x arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_arctan2(
        (CnpArray*)y, (CnpArray*)x);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_hypot(
    void *x, void *y) {
    const char *function_name = "cnp_ahk_hypot";
    if (!x || !y) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "x and y arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_hypot(
        (CnpArray*)x, (CnpArray*)y);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

typedef CnpArray* (CNP_CALL *CnpAngleConversionFunction)(
    const CnpArray *source);

static void* cnp_ahk_angle_conversion(
    void *source,
    CnpAngleConversionFunction function,
    const char *function_name) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = function((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_degrees(void *source) {
    return cnp_ahk_angle_conversion(
        source, cnp_degrees, "cnp_ahk_degrees");
}

__declspec(dllexport) void* __cdecl cnp_ahk_radians(void *source) {
    return cnp_ahk_angle_conversion(
        source, cnp_radians, "cnp_ahk_radians");
}

__declspec(dllexport) void* __cdecl cnp_ahk_deg2rad(void *source) {
    return cnp_ahk_angle_conversion(
        source, cnp_deg2rad, "cnp_ahk_deg2rad");
}

__declspec(dllexport) void* __cdecl cnp_ahk_rad2deg(void *source) {
    return cnp_ahk_angle_conversion(
        source, cnp_rad2deg, "cnp_ahk_rad2deg");
}
typedef CnpArray* (CNP_CALL *CnpExtremaFunction)(
    const CnpArray *left, const CnpArray *right);

static void* cnp_ahk_extrema(
    void *left,
    void *right,
    CnpExtremaFunction function,
    const char *function_name) {
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = function(
        (const CnpArray*)left, (const CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_maximum(
    void *left, void *right) {
    return cnp_ahk_extrema(
        left, right, cnp_maximum, "cnp_ahk_maximum");
}

__declspec(dllexport) void* __cdecl cnp_ahk_minimum(
    void *left, void *right) {
    return cnp_ahk_extrema(
        left, right, cnp_minimum, "cnp_ahk_minimum");
}

__declspec(dllexport) void* __cdecl cnp_ahk_fmax(
    void *left, void *right) {
    return cnp_ahk_extrema(
        left, right, cnp_fmax, "cnp_ahk_fmax");
}

__declspec(dllexport) void* __cdecl cnp_ahk_fmin(
    void *left, void *right) {
    return cnp_ahk_extrema(
        left, right, cnp_fmin, "cnp_ahk_fmin");
}

typedef CnpArray* (CNP_CALL *CnpLogicalBinaryFunction)(
    const CnpArray *left, const CnpArray *right);

static void* cnp_ahk_logical_binary(
    void *left,
    void *right,
    CnpLogicalBinaryFunction function,
    const char *function_name) {
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = function(
        (const CnpArray*)left, (const CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_logical_and(
    void *left, void *right) {
    return cnp_ahk_logical_binary(
        left, right, cnp_logical_and, "cnp_ahk_logical_and");
}

__declspec(dllexport) void* __cdecl cnp_ahk_logical_or(
    void *left, void *right) {
    return cnp_ahk_logical_binary(
        left, right, cnp_logical_or, "cnp_ahk_logical_or");
}

__declspec(dllexport) void* __cdecl cnp_ahk_logical_xor(
    void *left, void *right) {
    return cnp_ahk_logical_binary(
        left, right, cnp_logical_xor, "cnp_ahk_logical_xor");
}

__declspec(dllexport) void* __cdecl cnp_ahk_logical_not(void *source) {
    const char *function_name = "cnp_ahk_logical_not";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_logical_not((const CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

static void* cnp_ahk_bitwise_binary(
    void *left,
    void *right,
    CnpLogicalBinaryFunction function,
    const char *function_name) {
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = function(
        (const CnpArray*)left, (const CnpArray*)right);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

static void* cnp_ahk_array_unary(
    void *source,
    CnpArray* (CNP_CALL *function)(const CnpArray *),
    const char *function_name) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = function((const CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_isnan(void *source) {
    return cnp_ahk_array_unary(
        source, cnp_isnan, "cnp_ahk_isnan");
}

__declspec(dllexport) void* __cdecl cnp_ahk_isinf(void *source) {
    return cnp_ahk_array_unary(
        source, cnp_isinf, "cnp_ahk_isinf");
}

__declspec(dllexport) void* __cdecl cnp_ahk_isfinite(void *source) {
    return cnp_ahk_array_unary(
        source, cnp_isfinite, "cnp_ahk_isfinite");
}

__declspec(dllexport) void* __cdecl cnp_ahk_signbit(void *source) {
    return cnp_ahk_array_unary(
        source, cnp_signbit, "cnp_ahk_signbit");
}

typedef bool (CNP_CALL *CnpAhkObjectKindPredicate)(const CnpArray *source);

static int cnp_ahk_object_kind_predicate(
    void *source, CnpAhkObjectKindPredicate predicate) {
    cnp_clear_error();
    bool result = predicate((const CnpArray*)source);
    CNP_STATUS status = cnp_get_error(NULL);
    return status == CNP_OK ? (result ? 1 : 0) : (int)status;
}

__declspec(dllexport) int __cdecl cnp_ahk_iscomplexobj(void *source) {
    return cnp_ahk_object_kind_predicate(source, cnp_iscomplexobj);
}

__declspec(dllexport) int __cdecl cnp_ahk_isrealobj(void *source) {
    return cnp_ahk_object_kind_predicate(source, cnp_isrealobj);
}

__declspec(dllexport) int __cdecl cnp_ahk_isscalar(void *source) {
    return cnp_ahk_object_kind_predicate(source, cnp_isscalar);
}

__declspec(dllexport) void* __cdecl cnp_ahk_gcd(
    void *left, void *right) {
    return cnp_ahk_bitwise_binary(
        left, right, cnp_gcd, "cnp_ahk_gcd");
}

__declspec(dllexport) void* __cdecl cnp_ahk_lcm(
    void *left, void *right) {
    return cnp_ahk_bitwise_binary(
        left, right, cnp_lcm, "cnp_ahk_lcm");
}

__declspec(dllexport) void* __cdecl cnp_ahk_bitwise_and(
    void *left, void *right) {
    return cnp_ahk_bitwise_binary(
        left, right, cnp_bitwise_and, "cnp_ahk_bitwise_and");
}

__declspec(dllexport) void* __cdecl cnp_ahk_bitwise_or(
    void *left, void *right) {
    return cnp_ahk_bitwise_binary(
        left, right, cnp_bitwise_or, "cnp_ahk_bitwise_or");
}

__declspec(dllexport) void* __cdecl cnp_ahk_bitwise_xor(
    void *left, void *right) {
    return cnp_ahk_bitwise_binary(
        left, right, cnp_bitwise_xor, "cnp_ahk_bitwise_xor");
}

__declspec(dllexport) void* __cdecl cnp_ahk_left_shift(
    void *left, void *right) {
    return cnp_ahk_bitwise_binary(
        left, right, cnp_left_shift, "cnp_ahk_left_shift");
}

__declspec(dllexport) void* __cdecl cnp_ahk_right_shift(
    void *left, void *right) {
    return cnp_ahk_bitwise_binary(
        left, right, cnp_right_shift, "cnp_ahk_right_shift");
}

__declspec(dllexport) void* __cdecl cnp_ahk_invert(void *source) {
    return cnp_ahk_array_unary(
        source, cnp_invert, "cnp_ahk_invert");
}

__declspec(dllexport) void* __cdecl cnp_ahk_bitwise_not(void *source) {
    return cnp_ahk_array_unary(
        source, cnp_bitwise_not, "cnp_ahk_bitwise_not");
}

/* =========================================================================
 * Reductions - return scalar double
 * ========================================================================= */

static double cnp_ahk_reduction_scalar(CnpArray *result) {
    if (!result) return 0.0;
    double value = cnp_array_flat_get(result, 0);
    cnp_array_free(result);
    return value;
}

__declspec(dllexport) double __cdecl cnp_ahk_sum(void *handle, int axis) {
    if (!handle) return 0.0;
    CnpArray *arr = (CnpArray*)handle;
    return cnp_ahk_reduction_scalar(cnp_sum_v2(
        arr, axis, axis == CNP_AXIS_NONE, CNP_NOTYPE));
}

__declspec(dllexport) double __cdecl cnp_ahk_prod(void *handle, int axis) {
    if (!handle) return 0.0;
    CnpArray *arr = (CnpArray*)handle;
    return cnp_ahk_reduction_scalar(cnp_prod_v2(
        arr, axis, axis == CNP_AXIS_NONE, CNP_NOTYPE));
}

__declspec(dllexport) double __cdecl cnp_ahk_mean(void *handle, int axis) {
    if (!handle) return 0.0;
    CnpArray *arr = (CnpArray*)handle;
    return cnp_ahk_reduction_scalar(cnp_mean_v2(
        arr, axis, axis == CNP_AXIS_NONE, CNP_NOTYPE));
}

__declspec(dllexport) double __cdecl cnp_ahk_std(void *handle, int axis) {
    if (!handle) return 0.0;
    CnpArray *arr = (CnpArray*)handle;
    return cnp_ahk_reduction_scalar(cnp_std_v2(
        arr, axis, axis == CNP_AXIS_NONE, 0, CNP_NOTYPE));
}

__declspec(dllexport) double __cdecl cnp_ahk_var(void *handle, int axis) {
    if (!handle) return 0.0;
    CnpArray *arr = (CnpArray*)handle;
    return cnp_ahk_reduction_scalar(cnp_var_v2(
        arr, axis, axis == CNP_AXIS_NONE, 0, CNP_NOTYPE));
}

__declspec(dllexport) double __cdecl cnp_ahk_max(void *handle, int axis) {
    if (!handle) return 0.0;
    CnpArray *arr = (CnpArray*)handle;
    return cnp_ahk_reduction_scalar(cnp_max_v2(
        arr, axis, axis == CNP_AXIS_NONE));
}

__declspec(dllexport) double __cdecl cnp_ahk_min(void *handle, int axis) {
    if (!handle) return 0.0;
    CnpArray *arr = (CnpArray*)handle;
    return cnp_ahk_reduction_scalar(cnp_min_v2(
        arr, axis, axis == CNP_AXIS_NONE));
}

__declspec(dllexport) void* __cdecl cnp_ahk_sum_array(void *source, int axis) {
    if (!source) return NULL;
    return (void*)cnp_sum((CnpArray*)source, axis, CNP_NOTYPE);
}

__declspec(dllexport) void* __cdecl cnp_ahk_prod_array(void *source, int axis) {
    if (!source) return NULL;
    return (void*)cnp_prod((CnpArray*)source, axis, CNP_NOTYPE);
}

__declspec(dllexport) void* __cdecl cnp_ahk_mean_array(void *source, int axis) {
    if (!source) return NULL;
    return (void*)cnp_mean((CnpArray*)source, axis, CNP_NOTYPE);
}

__declspec(dllexport) void* __cdecl cnp_ahk_var_array(void *source, int axis) {
    if (!source) return NULL;
    return (void*)cnp_var((CnpArray*)source, axis, 0, CNP_NOTYPE);
}

__declspec(dllexport) void* __cdecl cnp_ahk_std_array(void *source, int axis) {
    if (!source) return NULL;
    return (void*)cnp_std((CnpArray*)source, axis, 0, CNP_NOTYPE);
}

__declspec(dllexport) void* __cdecl cnp_ahk_max_array(void *source, int axis) {
    if (!source) return NULL;
    return (void*)cnp_max((CnpArray*)source, axis);
}

__declspec(dllexport) void* __cdecl cnp_ahk_min_array(void *source, int axis) {
    if (!source) return NULL;
    return (void*)cnp_min((CnpArray*)source, axis);
}

__declspec(dllexport) int64_t __cdecl cnp_ahk_argmax(void *handle, int axis) {
    if (!handle) return -1;
    CnpArray *result = cnp_argmax((CnpArray*)handle, axis);
    if (!result) return -1;
    int64_t val = (int64_t)cnp_array_flat_get(result, 0);
    cnp_array_free(result);
    return val;
}

__declspec(dllexport) int64_t __cdecl cnp_ahk_argmin(void *handle, int axis) {
    if (!handle) return -1;
    CnpArray *result = cnp_argmin((CnpArray*)handle, axis);
    if (!result) return -1;
    int64_t val = (int64_t)cnp_array_flat_get(result, 0);
    cnp_array_free(result);
    return val;
}

/* Reductions returning arrays (along axis) */
__declspec(dllexport) void* __cdecl cnp_ahk_cumsum(void *handle, int axis) {
    if (!handle) return NULL;
    return (void*)cnp_cumsum((CnpArray*)handle, axis, CNP_NOTYPE);
}

__declspec(dllexport) void* __cdecl cnp_ahk_cumprod(void *handle, int axis) {
    if (!handle) return NULL;
    return (void*)cnp_cumprod((CnpArray*)handle, axis, CNP_NOTYPE);
}

#define AHK_AXIS_REDUCTION_V2(name, function) \
__declspec(dllexport) void* __cdecl cnp_ahk_##name##_v2( \
        void *handle, int axis, int axis_none) { \
    const char *function_name = "cnp_ahk_" #name "_v2"; \
    if (!handle) { \
        cnp_set_error(CNP_ERR_GENERIC, function_name, \
                      "source array must not be null"); \
        return NULL; \
    } \
    CnpArray *result = function( \
        (CnpArray*)handle, axis, axis_none != 0); \
    if (!result) cnp_relabel_error(function_name); \
    return (void*)result; \
}

#define AHK_DTYPE_REDUCTION_V2(name, function) \
__declspec(dllexport) void* __cdecl cnp_ahk_##name##_v2( \
        void *handle, int axis, int axis_none) { \
    const char *function_name = "cnp_ahk_" #name "_v2"; \
    if (!handle) { \
        cnp_set_error(CNP_ERR_GENERIC, function_name, \
                      "source array must not be null"); \
        return NULL; \
    } \
    CnpArray *result = function( \
        (CnpArray*)handle, axis, axis_none != 0, CNP_NOTYPE); \
    if (!result) cnp_relabel_error(function_name); \
    return (void*)result; \
}

#define AHK_DEVIATION_REDUCTION_V2(name, function) \
__declspec(dllexport) void* __cdecl cnp_ahk_##name##_v2( \
        void *handle, int axis, int axis_none, int ddof) { \
    const char *function_name = "cnp_ahk_" #name "_v2"; \
    if (!handle) { \
        cnp_set_error(CNP_ERR_GENERIC, function_name, \
                      "source array must not be null"); \
        return NULL; \
    } \
    CnpArray *result = function( \
        (CnpArray*)handle, axis, axis_none != 0, ddof, CNP_NOTYPE); \
    if (!result) cnp_relabel_error(function_name); \
    return (void*)result; \
}

#define AHK_PERCENTILE_REDUCTION_V2(name, function) \
__declspec(dllexport) void* __cdecl cnp_ahk_##name##_v2( \
        void *handle, double q, int axis, int axis_none) { \
    const char *function_name = "cnp_ahk_" #name "_v2"; \
    if (!handle) { \
        cnp_set_error(CNP_ERR_GENERIC, function_name, \
                      "source array must not be null"); \
        return NULL; \
    } \
    CnpArray *result = function( \
        (CnpArray*)handle, q, axis, axis_none != 0); \
    if (!result) cnp_relabel_error(function_name); \
    return (void*)result; \
}

AHK_DTYPE_REDUCTION_V2(sum, cnp_sum_v2)
AHK_DTYPE_REDUCTION_V2(prod, cnp_prod_v2)
AHK_DTYPE_REDUCTION_V2(mean, cnp_mean_v2)

__declspec(dllexport) void* __cdecl cnp_ahk_average_v2(
        void *source, int axis, int axis_none, void *weights) {
    const char *function_name = "cnp_ahk_average_v2";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_average_v2(
        (CnpArray*)source, axis, axis_none != 0,
        (CnpArray*)weights);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

AHK_DEVIATION_REDUCTION_V2(var, cnp_var_v2)
AHK_DEVIATION_REDUCTION_V2(std, cnp_std_v2)
AHK_AXIS_REDUCTION_V2(max, cnp_max_v2)
AHK_AXIS_REDUCTION_V2(min, cnp_min_v2)
AHK_AXIS_REDUCTION_V2(argmax, cnp_argmax_v2)
AHK_AXIS_REDUCTION_V2(argmin, cnp_argmin_v2)
AHK_AXIS_REDUCTION_V2(any, cnp_any_v2)
AHK_AXIS_REDUCTION_V2(all, cnp_all_v2)
AHK_AXIS_REDUCTION_V2(ptp, cnp_ptp_v2)
AHK_DTYPE_REDUCTION_V2(cumsum, cnp_cumsum_v2)
AHK_DTYPE_REDUCTION_V2(cumprod, cnp_cumprod_v2)
AHK_DTYPE_REDUCTION_V2(nansum, cnp_nansum_v2)
AHK_DTYPE_REDUCTION_V2(nanprod, cnp_nanprod_v2)
AHK_DTYPE_REDUCTION_V2(nanmean, cnp_nanmean_v2)
AHK_DEVIATION_REDUCTION_V2(nanvar, cnp_nanvar_v2)
AHK_DEVIATION_REDUCTION_V2(nanstd, cnp_nanstd_v2)
AHK_AXIS_REDUCTION_V2(nanmax, cnp_nanmax_v2)
AHK_AXIS_REDUCTION_V2(nanmin, cnp_nanmin_v2)
AHK_AXIS_REDUCTION_V2(nanargmax, cnp_nanargmax_v2)
AHK_AXIS_REDUCTION_V2(nanargmin, cnp_nanargmin_v2)
AHK_AXIS_REDUCTION_V2(median, cnp_median_v2)
AHK_AXIS_REDUCTION_V2(nanmedian, cnp_nanmedian_v2)
AHK_PERCENTILE_REDUCTION_V2(percentile, cnp_percentile_v2)
AHK_PERCENTILE_REDUCTION_V2(nanpercentile, cnp_nanpercentile_v2)
AHK_PERCENTILE_REDUCTION_V2(quantile, cnp_quantile_v2)
AHK_PERCENTILE_REDUCTION_V2(nanquantile, cnp_nanquantile_v2)
AHK_DTYPE_REDUCTION_V2(nancumsum, cnp_nancumsum_v2)
AHK_DTYPE_REDUCTION_V2(nancumprod, cnp_nancumprod_v2)

#undef AHK_PERCENTILE_REDUCTION_V2
#undef AHK_DEVIATION_REDUCTION_V2
#undef AHK_DTYPE_REDUCTION_V2
#undef AHK_AXIS_REDUCTION_V2

/* =========================================================================
 * Axis-aware miscellaneous numerical operations
 * ========================================================================= */
__declspec(dllexport) void* __cdecl cnp_ahk_softmax(
    void *source, int axis) {
    const char *function_name = "cnp_ahk_softmax";
    if (!source) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_softmax((CnpArray*)source, axis);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_log_softmax(
    void *source, int axis) {
    const char *function_name = "cnp_ahk_log_softmax";
    if (!source) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_log_softmax((CnpArray*)source, axis);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_trapz(
    void *y, void *x, double dx, int axis) {
    const char *function_name = "cnp_ahk_trapz";
    if (!y) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "y array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_trapz(
        (CnpArray*)y, (CnpArray*)x, dx, axis);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_angle(
    void *source, int degrees) {
    const char *function_name = "cnp_ahk_angle";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_angle((CnpArray*)source, degrees != 0);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_real(void *source) {
    const char *function_name = "cnp_ahk_real";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_real((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_imag(void *source) {
    const char *function_name = "cnp_ahk_imag";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_imag((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_real_if_close(
    void *source, double tolerance) {
    const char *function_name = "cnp_ahk_real_if_close";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_real_if_close(
        (CnpArray*)source, tolerance);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_convolve(
    void *left, void *right, int mode) {
    const char *function_name = "cnp_ahk_convolve";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_convolve(
        (CnpArray*)left, (CnpArray*)right, mode);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_correlate(
    void *left, void *right, int mode) {
    const char *function_name = "cnp_ahk_correlate";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays must not be null");
        return NULL;
    }
    CnpArray *result = cnp_correlate(
        (CnpArray*)left, (CnpArray*)right, mode);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_packbits_v2(
    void *source, int axis, int axis_none, int bitorder) {
    const char *function_name = "cnp_ahk_packbits_v2";
    if (!source) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_packbits_v2(
        (CnpArray*)source, axis, axis_none != 0,
        (CNP_BITORDER)bitorder);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_unpackbits_v2(
    void *source, int axis, int axis_none,
    int64_t count, int count_none, int bitorder) {
    const char *function_name = "cnp_ahk_unpackbits_v2";
    if (!source) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_unpackbits_v2(
        (CnpArray*)source, axis, axis_none != 0,
        count, count_none != 0, (CNP_BITORDER)bitorder);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

/* =========================================================================
 * Shape manipulation
 * ========================================================================= */

__declspec(dllexport) void* __cdecl cnp_ahk_atleast_1d(void *source) {
    const char *function_name = "cnp_ahk_atleast_1d";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_atleast_1d((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_atleast_2d(void *source) {
    const char *function_name = "cnp_ahk_atleast_2d";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_atleast_2d((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_atleast_3d(void *source) {
    const char *function_name = "cnp_ahk_atleast_3d";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_atleast_3d((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_reshape(void *handle, int ndim, const int64_t *new_shape) {
    if (!handle || ndim <= 0 || !new_shape) return NULL;
    return (void*)cnp_reshape((CnpArray*)handle, ndim, (int64_t*)new_shape, CNP_ORDER_C);
}

__declspec(dllexport) void* __cdecl cnp_ahk_transpose(void *handle) {
    if (!handle) return NULL;
    return (void*)cnp_transpose((CnpArray*)handle, NULL);
}

__declspec(dllexport) void* __cdecl cnp_ahk_transpose_copy(void *source) {
    if (!source) return NULL;
    CnpArray *view = cnp_transpose((CnpArray*)source, NULL);
    if (!view) return NULL;
    CnpArray *result = cnp_array_copy(view);
    cnp_array_free(view);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_flatten(void *handle) {
    if (!handle) return NULL;
    return (void*)cnp_flatten((CnpArray*)handle, CNP_ORDER_C);
}

__declspec(dllexport) void* __cdecl cnp_ahk_ravel(void *handle) {
    if (!handle) return NULL;
    return (void*)cnp_ravel((CnpArray*)handle, CNP_ORDER_C);
}

__declspec(dllexport) void* __cdecl cnp_ahk_squeeze(void *handle) {
    if (!handle) return NULL;
    return (void*)cnp_squeeze((CnpArray*)handle, -1);
}

/* =========================================================================
 * Linear algebra
 * ========================================================================= */

__declspec(dllexport) void* __cdecl cnp_ahk_dot(void *a, void *b) {
    if (!a || !b) return NULL;
    return (void*)cnp_dot((CnpArray*)a, (CnpArray*)b);
}

__declspec(dllexport) void* __cdecl cnp_ahk_matmul(void *a, void *b) {
    if (!a || !b) return NULL;
    return (void*)cnp_matmul((CnpArray*)a, (CnpArray*)b);
}

__declspec(dllexport) void* __cdecl cnp_ahk_einsum(
    const char *subscripts, int narrays, void **arrays) {
    if (!subscripts) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_ahk_einsum",
                      "subscripts must not be null");
        return NULL;
    }
    if (!arrays) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_ahk_einsum",
                      "operand pointer array must not be null");
        return NULL;
    }
    return (void*)cnp_einsum(
        subscripts, narrays, (const CnpArray *const *)arrays);
}

__declspec(dllexport) int __cdecl cnp_ahk_linalg_eig(
    void *source, void **results, int result_capacity) {
    CnpArray *eigenvalues = NULL;
    CnpArray *eigenvectors = NULL;
    CNP_STATUS status;
    if (results) {
        for (int index = 0; index < result_capacity; index++) {
            results[index] = NULL;
        }
    }
    if (!results) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_ahk_linalg_eig",
            "result buffer must not be null");
        return CNP_ERR_GENERIC;
    }
    if (result_capacity < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, "cnp_ahk_linalg_eig",
            "result buffer capacity must be at least two");
        return CNP_ERR_SHAPE;
    }
    status = cnp_linalg_eig(
        (CnpArray*)source, &eigenvalues, &eigenvectors);
    if (status != CNP_OK) return status;
    results[0] = eigenvalues;
    results[1] = eigenvectors;
    return CNP_OK;
}

__declspec(dllexport) void* __cdecl cnp_ahk_linalg_cholesky(
        void *source) {
    CnpArray *result = NULL;
    CNP_STATUS status = cnp_linalg_cholesky(
        (CnpArray*)source, &result);
    if (status != CNP_OK) return NULL;
    return (void*)result;
}

__declspec(dllexport) int __cdecl cnp_ahk_linalg_eigh_v2(
    void *source, int upper, void **results, int result_capacity) {
    CnpArray *eigenvalues = NULL;
    CnpArray *eigenvectors = NULL;
    CNP_STATUS status;
    int index;

    if (results) {
        for (index = 0; index < result_capacity; index++) {
            results[index] = NULL;
        }
    }
    if (!results) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_ahk_linalg_eigh_v2",
            "result buffer must not be null");
        return CNP_ERR_GENERIC;
    }
    if (result_capacity < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, "cnp_ahk_linalg_eigh_v2",
            "result buffer capacity must be at least two");
        return CNP_ERR_SHAPE;
    }

    status = cnp_linalg_eigh_v2(
        (CnpArray*)source, upper != 0,
        &eigenvalues, &eigenvectors);
    if (status != CNP_OK) {
        if (eigenvalues) cnp_array_free(eigenvalues);
        if (eigenvectors) cnp_array_free(eigenvectors);
        return status;
    }
    results[0] = eigenvalues;
    results[1] = eigenvectors;
    return CNP_OK;
}

__declspec(dllexport) void* __cdecl cnp_ahk_eigvalsh_v2(
    void *source, int upper) {
    return (void*)cnp_eigvalsh_v2(
        (CnpArray*)source, upper != 0);
}

__declspec(dllexport) int __cdecl cnp_ahk_linalg_svd_v2(
    void *source,
    int full_matrices,
    int compute_uv,
    int hermitian,
    void **results,
    int result_capacity) {
    CnpArray *u = NULL;
    CnpArray *s = NULL;
    CnpArray *vh = NULL;
    int required_capacity = compute_uv ? 3 : 1;
    CNP_STATUS status;
    int index;

    if (results) {
        for (index = 0; index < result_capacity; index++) {
            results[index] = NULL;
        }
    }
    if (!results) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_ahk_linalg_svd_v2",
            "result buffer must not be null");
        return CNP_ERR_GENERIC;
    }
    if (result_capacity < required_capacity) {
        cnp_set_error(
            CNP_ERR_SHAPE, "cnp_ahk_linalg_svd_v2",
            compute_uv
                ? "result buffer capacity must be at least three"
                : "result buffer capacity must be at least one");
        return CNP_ERR_SHAPE;
    }

    status = cnp_linalg_svd_v2(
        (CnpArray*)source,
        full_matrices != 0,
        compute_uv != 0,
        hermitian != 0,
        compute_uv ? &u : NULL,
        &s,
        compute_uv ? &vh : NULL);
    if (status != CNP_OK) {
        if (u) cnp_array_free(u);
        if (s) cnp_array_free(s);
        if (vh) cnp_array_free(vh);
        return status;
    }

    if (compute_uv) {
        results[0] = u;
        results[1] = s;
        results[2] = vh;
    } else {
        results[0] = s;
    }
    return CNP_OK;
}

__declspec(dllexport) double __cdecl cnp_ahk_linalg_det(void *handle) {
    if (!handle) return 0.0;
    CnpArray *result = cnp_linalg_det((CnpArray*)handle);
    if (!result) return 0.0;
    double val = cnp_array_flat_get(result, 0);
    cnp_array_free(result);
    return val;
}

__declspec(dllexport) void* __cdecl cnp_ahk_linalg_det_v2(void *source) {
    return (void*)cnp_linalg_det((CnpArray*)source);
}

__declspec(dllexport) int __cdecl cnp_ahk_linalg_slogdet_v2(
        void *source, void **results, int result_capacity) {
    CnpArray *sign = NULL;
    CnpArray *logabsdet = NULL;
    CNP_STATUS status;

    if (results) {
        for (int index = 0; index < result_capacity; index++) {
            results[index] = NULL;
        }
    }
    if (!results) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_ahk_linalg_slogdet_v2",
            "result buffer must not be null");
        return CNP_ERR_GENERIC;
    }
    if (result_capacity < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, "cnp_ahk_linalg_slogdet_v2",
            "result buffer capacity must be at least two");
        return CNP_ERR_SHAPE;
    }

    status = cnp_linalg_slogdet_v2(
        (CnpArray*)source, &sign, &logabsdet);
    if (status != CNP_OK) {
        if (sign) cnp_array_free(sign);
        if (logabsdet) cnp_array_free(logabsdet);
        return status;
    }
    results[0] = sign;
    results[1] = logabsdet;
    return CNP_OK;
}

__declspec(dllexport) void* __cdecl cnp_ahk_linalg_inv(void *handle) {
    if (!handle) return NULL;
    return (void*)cnp_linalg_inv((CnpArray*)handle);
}

__declspec(dllexport) double __cdecl cnp_ahk_linalg_norm(void *handle) {
    if (!handle) return 0.0;
    CnpArray *result = cnp_linalg_norm((CnpArray*)handle, NULL, 2);
    if (!result) return 0.0;
    double val = cnp_array_flat_get(result, 0);
    cnp_array_free(result);
    return val;
}

__declspec(dllexport) void* __cdecl cnp_ahk_linalg_solve(void *a, void *b) {
    const char *function_name = "cnp_ahk_linalg_solve";
    CnpArray *result = NULL;
    CNP_STATUS status = cnp_linalg_solve(
        (CnpArray*)a, (CnpArray*)b, &result);
    if (status != CNP_OK || !result) {
        if (result) cnp_array_free(result);
        cnp_relabel_error(function_name);
        return NULL;
    }
    return (void*)result;
}

__declspec(dllexport) int __cdecl cnp_ahk_linalg_lstsq_v2(
        void *a, void *b, double rcond, int rcond_none,
        void **results, int result_capacity) {
    const char *function_name = "cnp_ahk_linalg_lstsq_v2";
    CnpArray *x = NULL;
    CnpArray *residuals = NULL;
    CnpArray *rank = NULL;
    CnpArray *singular_values = NULL;
    CNP_STATUS status;
    if (results) {
        for (int index = 0; index < result_capacity; index++) {
            results[index] = NULL;
        }
    }
    if (!results) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "result buffer must not be null");
        return CNP_ERR_GENERIC;
    }
    if (result_capacity < 4) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "result buffer capacity must be at least four");
        return CNP_ERR_SHAPE;
    }
    status = cnp_linalg_lstsq_v2(
        (CnpArray*)a, (CnpArray*)b,
        rcond, rcond_none != 0,
        &x, &residuals, &rank, &singular_values);
    if (status != CNP_OK) {
        if (x) cnp_array_free(x);
        if (residuals) cnp_array_free(residuals);
        if (rank) cnp_array_free(rank);
        if (singular_values) cnp_array_free(singular_values);
        cnp_relabel_error(function_name);
        return status;
    }
    results[0] = x;
    results[1] = residuals;
    results[2] = rank;
    results[3] = singular_values;
    return CNP_OK;
}

__declspec(dllexport) void* __cdecl cnp_ahk_linalg_cond_v2(void *source) {
    const char *function_name = "cnp_ahk_linalg_cond_v2";
    CnpArray *result = cnp_linalg_cond_v2((CnpArray*)source);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

/* =========================================================================
 * Sorting
 * ========================================================================= */

__declspec(dllexport) void* __cdecl cnp_ahk_sort(void *handle, int axis) {
    if (!handle) return NULL;
    return (void*)cnp_sort((CnpArray*)handle, axis, CNP_SORT_QUICKSORT);
}

__declspec(dllexport) void* __cdecl cnp_ahk_argsort(void *handle, int axis) {
    if (!handle) return NULL;
    return (void*)cnp_argsort((CnpArray*)handle, axis, CNP_SORT_QUICKSORT);
}

__declspec(dllexport) void* __cdecl cnp_ahk_sort_v2(
    void *handle, int axis, int axis_none, int kind) {
    return (void*)cnp_sort_v2(
        (CnpArray*)handle, axis, axis_none != 0, (CNP_SORT_KIND)kind);
}

__declspec(dllexport) void* __cdecl cnp_ahk_msort(void *handle) {
    const char *function_name = "cnp_ahk_msort";
    if (!handle) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_msort((CnpArray*)handle);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_sort_complex(void *handle) {
    const char *function_name = "cnp_ahk_sort_complex";
    if (!handle) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    CnpArray *result = cnp_sort_complex((CnpArray*)handle);
    if (!result) cnp_relabel_error(function_name);
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_argsort_v2(
    void *handle, int axis, int axis_none, int kind) {
    return (void*)cnp_argsort_v2(
        (CnpArray*)handle, axis, axis_none != 0, (CNP_SORT_KIND)kind);
}

__declspec(dllexport) void* __cdecl cnp_ahk_partition_v2(
    void *handle, const int64_t *kth, int kth_count,
    int axis, int axis_none) {
    CnpArray *result = cnp_partition_v2(
        (CnpArray*)handle, kth, kth_count, axis, axis_none != 0);
    if (!result) cnp_relabel_error("cnp_ahk_partition_v2");
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_argpartition_v2(
    void *handle, const int64_t *kth, int kth_count,
    int axis, int axis_none) {
    CnpArray *result = cnp_argpartition_v2(
        (CnpArray*)handle, kth, kth_count, axis, axis_none != 0);
    if (!result) cnp_relabel_error("cnp_ahk_argpartition_v2");
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_searchsorted_v2(
    void *source, void *values, const char *side, void *sorter) {
    CnpArray *result = cnp_searchsorted_v2(
        (CnpArray*)source, (CnpArray*)values,
        side, (CnpArray*)sorter);
    if (!result) cnp_relabel_error("cnp_ahk_searchsorted_v2");
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_digitize(
    void *x, void *bins, int right) {
    CnpArray *result = cnp_digitize(
        (CnpArray*)x, (CnpArray*)bins, right != 0);
    if (!result) cnp_relabel_error("cnp_ahk_digitize");
    return (void*)result;
}

__declspec(dllexport) void* __cdecl cnp_ahk_lexsort_v2(
    void *const *handles, int count, int axis) {
    CnpArray *result = cnp_lexsort_v2(
        count, (const CnpArray**)handles, axis);
    if (!result) cnp_relabel_error("cnp_ahk_lexsort_v2");
    return (void*)result;
}

__declspec(dllexport) int __cdecl cnp_ahk_unique_v2(
    void *handle, int return_index, int return_inverse, int return_counts,
    void **results, int result_capacity) {
    return (int)cnp_unique_v2(
        (CnpArray*)handle,
        return_index != 0, return_inverse != 0, return_counts != 0,
        (CnpArray**)results, result_capacity);
}

__declspec(dllexport) void* __cdecl cnp_ahk_intersect1d(
    void *left, void *right, int assume_unique) {
    return (void*)cnp_intersect1d(
        (CnpArray*)left, (CnpArray*)right, assume_unique != 0);
}

__declspec(dllexport) void* __cdecl cnp_ahk_union1d(
    void *left, void *right) {
    return (void*)cnp_union1d((CnpArray*)left, (CnpArray*)right);
}

__declspec(dllexport) void* __cdecl cnp_ahk_setdiff1d(
    void *left, void *right, int assume_unique) {
    return (void*)cnp_setdiff1d(
        (CnpArray*)left, (CnpArray*)right, assume_unique != 0);
}

__declspec(dllexport) void* __cdecl cnp_ahk_setxor1d(
    void *left, void *right, int assume_unique) {
    return (void*)cnp_setxor1d(
        (CnpArray*)left, (CnpArray*)right, assume_unique != 0);
}

__declspec(dllexport) void* __cdecl cnp_ahk_in1d(
    void *left, void *right, int assume_unique, int invert) {
    return (void*)cnp_in1d(
        (CnpArray*)left, (CnpArray*)right,
        assume_unique != 0, invert != 0);
}

__declspec(dllexport) void* __cdecl cnp_ahk_isin(
    void *element, void *test_elements, int assume_unique, int invert) {
    return (void*)cnp_isin(
        (CnpArray*)element, (CnpArray*)test_elements,
        assume_unique != 0, invert != 0);
}

/* =========================================================================
 * Comparison / logic
 * ========================================================================= */

__declspec(dllexport) void* __cdecl cnp_ahk_clip(void *handle, double lo, double hi) {
    if (!handle) return NULL;
    return (void*)cnp_clip((CnpArray*)handle, lo, hi);
}

__declspec(dllexport) void* __cdecl cnp_ahk_clip_array(
        void *source, void *a_min, void *a_max) {
    return (void*)cnp_clip_array(
        (CnpArray*)source, (CnpArray*)a_min, (CnpArray*)a_max);
}

__declspec(dllexport) void* __cdecl cnp_ahk_where(void *cond, void *x, void *y) {
    if (!cond || !x || !y) return NULL;
    return (void*)cnp_where((CnpArray*)cond, (CnpArray*)x, (CnpArray*)y);
}

/* =========================================================================
 * Random
 * ========================================================================= */

__declspec(dllexport) void __cdecl cnp_ahk_random_seed(int seed) {
    cnp_random_seed((uint64_t)seed);
}

__declspec(dllexport) void __cdecl cnp_ahk_random_seed_v2(uint64_t seed) {
    cnp_random_seed(seed);
}

__declspec(dllexport) void* __cdecl cnp_ahk_random_choice_v2(
    void *population,
    int size_ndim,
    const int64_t *size_shape,
    int size_none,
    int replace,
    void *probabilities) {
    return (void*)cnp_random_choice_v2(
        (const CnpArray*)population,
        size_ndim,
        size_shape,
        size_none != 0,
        replace != 0,
        (const CnpArray*)probabilities);
}

__declspec(dllexport) void* __cdecl cnp_ahk_random_permutation(void *source) {
    return (void*)cnp_random_permutation((const CnpArray*)source);
}

__declspec(dllexport) int __cdecl cnp_ahk_random_shuffle(void *source) {
    cnp_clear_error();
    cnp_random_shuffle((CnpArray*)source);
    return cnp_get_error(NULL);
}

__declspec(dllexport) void* __cdecl cnp_ahk_random_random(int ndim, const int64_t *shape_buf) {
    if (ndim <= 0 || !shape_buf) return NULL;
    return (void*)cnp_random_random(ndim, (int64_t*)shape_buf);
}

__declspec(dllexport) void* __cdecl cnp_ahk_random_normal(int ndim, const int64_t *shape_buf,
                                                           double mean, double std) {
    if (ndim <= 0 || !shape_buf) return NULL;
    return (void*)cnp_random_normal(mean, std, ndim, (int64_t*)shape_buf);
}

__declspec(dllexport) void* __cdecl cnp_ahk_random_uniform(int ndim, const int64_t *shape_buf,
                                                            double lo, double hi) {
    if (ndim <= 0 || !shape_buf) return NULL;
    return (void*)cnp_random_uniform(lo, hi, ndim, (int64_t*)shape_buf);
}

__declspec(dllexport) void* __cdecl cnp_ahk_random_randint(int ndim, const int64_t *shape_buf,
                                                            int64_t lo, int64_t hi) {
    if (ndim <= 0 || !shape_buf) return NULL;
    return (void*)cnp_random_randint(lo, hi, ndim, (int64_t*)shape_buf);
}

/* =========================================================================
 * FFT
 * ========================================================================= */

__declspec(dllexport) void* __cdecl cnp_ahk_fft(void *handle) {
    if (!handle) return NULL;
    return (void*)cnp_fft((CnpArray*)handle, -1);
}

__declspec(dllexport) void* __cdecl cnp_ahk_ifft(void *handle) {
    if (!handle) return NULL;
    return (void*)cnp_ifft((CnpArray*)handle, -1);
}

__declspec(dllexport) void* __cdecl cnp_ahk_rfft(void *handle) {
    if (!handle) return NULL;
    return (void*)cnp_rfft((CnpArray*)handle, -1);
}

/* =========================================================================
 * Concatenation / stacking
 * ========================================================================= */

__declspec(dllexport) void* __cdecl cnp_ahk_concatenate(void *a, void *b, int axis) {
    if (!a || !b) return NULL;
    CnpArray *arrs[2] = { (CnpArray*)a, (CnpArray*)b };
    return (void*)cnp_concatenate(2, arrs, axis);
}

__declspec(dllexport) void* __cdecl cnp_ahk_concatenate_many(
    void *const *handles, int count, int axis) {
    const char *function_name = "cnp_ahk_concatenate_many";
    if (!handles || count <= 0) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "handles must contain at least one array");
        return NULL;
    }
    for (int index = 0; index < count; ++index) {
        if (!handles[index]) {
            cnp_set_error(CNP_ERR_GENERIC, function_name,
                          "array handle %d must not be null", index);
            return NULL;
        }
    }
    CnpArray *first = (CnpArray*)handles[0];
    int normalized_axis = cnp_normalize_axis(axis, first->ndim);
    if (normalized_axis < 0 || normalized_axis >= first->ndim) {
        cnp_set_error(CNP_ERR_AXIS, function_name,
                      "axis %d is out of bounds for %d dimensions",
                      axis, first->ndim);
        return NULL;
    }
    return (void*)cnp_concatenate(
        count, (CnpArray**)handles, normalized_axis);
}

__declspec(dllexport) void* __cdecl cnp_ahk_vstack(void *a, void *b) {
    if (!a || !b) return NULL;
    CnpArray *arrs[2] = { (CnpArray*)a, (CnpArray*)b };
    return (void*)cnp_vstack(2, arrs);
}

__declspec(dllexport) void* __cdecl cnp_ahk_hstack(void *a, void *b) {
    if (!a || !b) return NULL;
    CnpArray *arrs[2] = { (CnpArray*)a, (CnpArray*)b };
    return (void*)cnp_hstack(2, arrs);
}

__declspec(dllexport) int __cdecl cnp_ahk_broadcast_arrays_v2(
    void *const *handles, int count,
    void **results, int result_capacity) {
    return (int)cnp_broadcast_arrays_v2(
        count, (CnpArray *const *)handles,
        (CnpArray **)results, result_capacity);
}

__declspec(dllexport) int __cdecl cnp_ahk_meshgrid_v2(
    void *const *handles, int count,
    int sparse, int indexing_ij, int copy,
    void **results, int result_capacity) {
    return (int)cnp_meshgrid_v2(
        count, (CnpArray *const *)handles,
        sparse != 0, indexing_ij != 0, copy != 0,
        (CnpArray **)results, result_capacity);
}

__declspec(dllexport) int __cdecl cnp_ahk_split_sections_v2(
    void *handle, int sections, int axis,
    void **results, int result_capacity) {
    return (int)cnp_split_sections_v2(
        (CnpArray*)handle, sections, axis,
        (CnpArray**)results, result_capacity);
}

__declspec(dllexport) int __cdecl cnp_ahk_split_indices_v2(
    void *handle, int nindices, const int64_t *indices, int axis,
    void **results, int result_capacity) {
    return (int)cnp_split_indices_v2(
        (CnpArray*)handle, nindices, indices, axis,
        (CnpArray**)results, result_capacity);
}

__declspec(dllexport) int __cdecl cnp_ahk_array_split_sections_v2(
    void *handle, int sections, int axis,
    void **results, int result_capacity) {
    return (int)cnp_array_split_sections_v2(
        (CnpArray*)handle, sections, axis,
        (CnpArray**)results, result_capacity);
}

__declspec(dllexport) int __cdecl cnp_ahk_array_split_indices_v2(
    void *handle, int nindices, const int64_t *indices, int axis,
    void **results, int result_capacity) {
    return (int)cnp_array_split_indices_v2(
        (CnpArray*)handle, nindices, indices, axis,
        (CnpArray**)results, result_capacity);
}

__declspec(dllexport) void* __cdecl cnp_ahk_take_v2(
    void *source, void *indices, int axis, int axis_none) {
    return (void*)cnp_take_v2(
        (CnpArray*)source, (CnpArray*)indices, axis, axis_none != 0);
}

__declspec(dllexport) void* __cdecl cnp_ahk_take_along_axis_v2(
    void *source, void *indices, int axis, int axis_none) {
    return (void*)cnp_take_along_axis_v2(
        (CnpArray*)source, (CnpArray*)indices, axis, axis_none != 0);
}

__declspec(dllexport) void* __cdecl cnp_ahk_compress_v2(
    void *condition, void *source, int axis, int axis_none) {
    return (void*)cnp_compress_v2(
        (CnpArray*)condition, (CnpArray*)source, axis, axis_none != 0);
}

__declspec(dllexport) void* __cdecl cnp_ahk_delete_v2(
    void *source, void *obj, int axis, int axis_none) {
    return (void*)cnp_delete_v2(
        (CnpArray*)source, (CnpArray*)obj, axis, axis_none != 0);
}

__declspec(dllexport) void* __cdecl cnp_ahk_insert_v2(
    void *source, int64_t obj, void *values, int axis, int axis_none) {
    return (void*)cnp_insert_v2(
        (CnpArray*)source, obj, (CnpArray*)values, axis, axis_none != 0);
}

__declspec(dllexport) void* __cdecl cnp_ahk_insert_array_v2(
    void *source, void *obj, void *values, int axis, int axis_none) {
    return (void*)cnp_insert_array_v2(
        (CnpArray*)source, (CnpArray*)obj, (CnpArray*)values,
        axis, axis_none != 0);
}

/* =========================================================================
 * Utility
 * ========================================================================= */

/* Copy array (deep copy). */
__declspec(dllexport) void* __cdecl cnp_ahk_copy(void *handle) {
    if (!handle) return NULL;
    return (void*)cnp_copy((CnpArray*)handle);
}

/* Astype - convert dtype. */
__declspec(dllexport) void* __cdecl cnp_ahk_astype(void *handle, int dtype) {
    if (!handle) return NULL;
    return (void*)cnp_astype((CnpArray*)handle, (CNP_TYPE)dtype, CNP_CAST_UNSAFE);
}

/* Get allocated memory (for debugging). */
__declspec(dllexport) int64_t __cdecl cnp_ahk_allocated_memory(void) {
    return (int64_t)cnp_get_allocated_memory();
}

/* Allclose check. */
__declspec(dllexport) int __cdecl cnp_ahk_allclose(void *a, void *b, double rtol, double atol) {
    return cnp_allclose((CnpArray*)a, (CnpArray*)b, rtol, atol) ? 1 : 0;
}

__declspec(dllexport) int __cdecl cnp_ahk_allclose_v2(
    void *a, void *b, double rtol, double atol,
    int equal_nan, int *result) {
    const char *function_name = "cnp_ahk_allclose_v2";
    if (!result) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "result output is required");
        return CNP_ERR_GENERIC;
    }
    *result = 0;
    bool native_result = false;
    CNP_STATUS status = cnp_allclose_v2(
        (CnpArray*)a, (CnpArray*)b,
        rtol, atol, equal_nan != 0, &native_result);
    if (status != CNP_OK) {
        cnp_relabel_error(function_name);
        return status;
    }
    *result = native_result ? 1 : 0;
    return CNP_OK;
}

/* Array to string representation. */
__declspec(dllexport) int __cdecl cnp_ahk_to_string(void *handle, char *buf, int64_t buf_size) {
    if (!handle || !buf || buf_size <= 0) return AHK_ERR_NULL;
    char *text = cnp_array_string_v2((CnpArray*)handle, false);
    if (!text) return (int)cnp_get_error(NULL);
    size_t length = strlen(text);
    if (length >= (size_t)buf_size) {
        cnp_free(text, length + 1);
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_ahk_to_string",
            "output buffer is too small");
        return AHK_ERR_SIZE;
    }
    memcpy(buf, text, length + 1);
    cnp_free(text, length + 1);
    return (int)length;
}
