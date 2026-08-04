/**
 * cnumpy stride tricks and additional array manipulation
 * Corresponds to numpy.lib.stride_tricks, numpy.lib.array_utils
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>

static int64_t stride_utility_flat_offset(
    const CnpArray *array, int64_t flat_index) {
    int64_t offset = array->offset;
    for (int dimension = array->ndim - 1;
            dimension >= 0; --dimension) {
        int64_t coordinate = flat_index % array->shape[dimension];
        flat_index /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

static CNP_STATUS stride_utility_element_truth(
    const CnpArray *array, int64_t flat_index,
    bool *truth, const char *function_name) {
    uint8_t value = 0;
    int64_t offset = stride_utility_flat_offset(array, flat_index);
    CNP_STATUS status = cnp_cast_scalar_value(
        (const char*)array->data + offset,
        array->dtype->type_num,
        &value, CNP_BOOL, function_name);
    if (status == CNP_OK) *truth = value != 0;
    return status;
}

/* =========================================================================
 * cnp_as_strided - Create a view with new shape and strides
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_as_strided(const CnpArray *arr, int ndim, const int64_t *shape, const int64_t *strides) {
    const char *function_name = "cnp_as_strided";
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "array is NULL");
        return NULL;
    }
    if (ndim < 0 || ndim > CNP_MAXDIMS || (ndim > 0 && !shape)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "invalid as_strided shape metadata");
        return NULL;
    }
    int64_t default_strides[CNP_MAXDIMS];
    const int64_t *view_strides = strides;
    if (!view_strides) {
        if (ndim != arr->ndim) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "strides are required when the view rank changes");
            return NULL;
        }
        if (ndim > 0)
            memcpy(
                default_strides, arr->strides,
                (size_t)ndim * sizeof(int64_t));
        view_strides = default_strides;
    }
    CnpArray *result = cnp_array_view_from_metadata(
        (CnpArray*)arr, ndim, shape, view_strides,
        arr->offset, 0);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_sliding_window_view - Create sliding window view
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_sliding_window_view(const CnpArray *arr, int64_t window_size, int axis) {
    const char *function_name = "cnp_sliding_window_view";
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "array is NULL");
        return NULL;
    }
    if (window_size <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "window size must be positive");
        return NULL;
    }
    if (axis < 0) axis += arr->ndim;
    if (axis < 0 || axis >= arr->ndim) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "axis %d is outside the array rank", axis);
        return NULL;
    }
    if (window_size > arr->shape[axis]) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "window size exceeds the selected dimension");
        return NULL;
    }
    if (arr->ndim >= CNP_MAXDIMS) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "sliding window result rank exceeds CNP_MAXDIMS");
        return NULL;
    }

    /* Output shape: same as input but axis dimension reduced */
    int out_ndim = arr->ndim + 1;
    int64_t out_shape[CNP_MAXDIMS];
    int64_t out_strides[CNP_MAXDIMS];

    for (int i = 0; i < arr->ndim; i++) {
        if (i < axis) {
            out_shape[i] = arr->shape[i];
            out_strides[i] = arr->strides[i];
        } else if (i == axis) {
            out_shape[i] = arr->shape[axis] - window_size + 1;
            out_strides[i] = arr->strides[axis];
        } else {
            out_shape[i] = arr->shape[i];
            out_strides[i] = arr->strides[i];
        }
    }
    /* Last dimension is the window */
    out_shape[arr->ndim] = window_size;
    out_strides[arr->ndim] = arr->strides[axis];

    CnpArray *result = cnp_as_strided(
        arr, out_ndim, out_shape, out_strides);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    result->flags &= ~CNP_ARRAY_WRITEABLE;
    return result;
}

/* cnp_broadcast_to is already defined in shape.c */
/* cnp_broadcast_arrays is already defined in broadcast.c */
/* cnp_diag is already defined in array.c */

/* =========================================================================
 * cnp_diagonal - Return diagonal of 2D array
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_diagonal(const CnpArray *arr, int offset, int axis1, int axis2) {
    const char *function_name = "cnp_diagonal";
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "array is NULL");
        return NULL;
    }
    if (arr->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "array must have at least two dimensions");
        return NULL;
    }
    if (axis1 < 0) axis1 += arr->ndim;
    if (axis2 < 0) axis2 += arr->ndim;
    if (axis1 < 0 || axis1 >= arr->ndim ||
            axis2 < 0 || axis2 >= arr->ndim) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "diagonal axes are outside the array rank");
        return NULL;
    }
    if (axis1 == axis2) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "axis1 and axis2 cannot be the same");
        return NULL;
    }

    int64_t rows = arr->shape[axis1];
    int64_t columns = arr->shape[axis2];
    int64_t diagonal_offset = (int64_t)offset;
    int64_t diagonal_length;
    if (diagonal_offset >= 0) {
        int64_t available = diagonal_offset < columns
            ? columns - diagonal_offset : 0;
        diagonal_length = rows < available ? rows : available;
    } else {
        int64_t shift = -diagonal_offset;
        int64_t available = shift < rows ? rows - shift : 0;
        diagonal_length = available < columns ? available : columns;
    }

    int output_ndim = arr->ndim - 1;
    int64_t output_shape[CNP_MAXDIMS] = {0};
    int64_t output_strides[CNP_MAXDIMS] = {0};
    int output_axis = 0;
    for (int dimension = 0; dimension < arr->ndim; ++dimension) {
        if (dimension == axis1 || dimension == axis2) continue;
        output_shape[output_axis] = arr->shape[dimension];
        output_strides[output_axis] = arr->strides[dimension];
        ++output_axis;
    }
    output_shape[output_ndim - 1] = diagonal_length;
    output_strides[output_ndim - 1] =
        arr->strides[axis1] + arr->strides[axis2];
    int64_t view_offset = arr->offset;
    if (diagonal_length > 0) {
        if (diagonal_offset >= 0)
            view_offset += diagonal_offset * arr->strides[axis2];
        else
            view_offset += -diagonal_offset * arr->strides[axis1];
    }
    CnpArray *result = cnp_array_view_from_metadata(
        (CnpArray*)arr, output_ndim,
        output_shape, output_strides, view_offset, 0);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    result->flags &= ~CNP_ARRAY_WRITEABLE;
    return result;
}

/* cnp_trace is already defined in reduce.c */

/* =========================================================================
 * cnp_fill_diagonal - Fill diagonal of 2D array
 * ========================================================================= */
CNP_API CNP_STATUS CNP_CALL cnp_fill_diagonal(CnpArray *arr, double val) {
    const char *function_name = "cnp_fill_diagonal";
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "array is NULL");
        return CNP_ERR_GENERIC;
    }
    if (arr->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "array must have at least two dimensions");
        return CNP_ERR_SHAPE;
    }
    if (!(arr->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "array is not writeable");
        return CNP_ERR_GENERIC;
    }
    if (arr->ndim > 2) {
        for (int dimension = 1; dimension < arr->ndim; ++dimension) {
            if (arr->shape[dimension] != arr->shape[0]) {
                cnp_set_error(
                    CNP_ERR_SHAPE, function_name,
                    "all dimensions of an array of dimension greater than two must be equal");
                return CNP_ERR_SHAPE;
            }
        }
    }

    int itemsize = arr->dtype->elsize;
    void *converted = cnp_malloc((size_t)itemsize);
    if (!converted) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate a converted diagonal value");
        return CNP_ERR_MEMORY;
    }
    CNP_STATUS status = cnp_cast_scalar_value(
        &val, CNP_DOUBLE, converted,
        arr->dtype->type_num, function_name);
    if (status != CNP_OK) {
        cnp_free(converted, (size_t)itemsize);
        return status;
    }

    int64_t count = arr->shape[0];
    if (arr->ndim == 2 && arr->shape[1] < count)
        count = arr->shape[1];
    for (int64_t index = 0; index < count; ++index) {
        int64_t destination_offset = arr->offset;
        for (int dimension = 0; dimension < arr->ndim; ++dimension)
            destination_offset += index * arr->strides[dimension];
        memcpy(
            (char*)arr->data + destination_offset,
            converted, (size_t)itemsize);
    }
    cnp_free(converted, (size_t)itemsize);
    return CNP_OK;
}

/* =========================================================================
 * cnp_extract - Return elements satisfying condition
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_extract(const CnpArray *condition, const CnpArray *arr) {
    const char *function_name = "cnp_extract";
    if (!condition || !arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "condition and array are required");
        return NULL;
    }
    if (condition->size > arr->size) {
        cnp_set_error(
            CNP_ERR_INDEX, function_name,
            "condition has more entries than the flattened array");
        return NULL;
    }

    int64_t count = 0;
    for (int64_t index = 0; index < condition->size; ++index) {
        bool selected = false;
        if (stride_utility_element_truth(
                condition, index, &selected, function_name) != CNP_OK)
            return NULL;
        if (selected) ++count;
    }

    int64_t shape[1] = {count};
    CnpArray *result = cnp_array_new(
        1, shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int itemsize = result->dtype->elsize;
    int64_t output_index = 0;
    for (int64_t index = 0; index < condition->size; ++index) {
        bool selected = false;
        CNP_STATUS status = stride_utility_element_truth(
            condition, index, &selected, function_name);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        if (!selected) continue;
        int64_t source_offset = stride_utility_flat_offset(arr, index);
        memcpy(
            (char*)result->data + output_index * itemsize,
            (const char*)arr->data + source_offset,
            (size_t)itemsize);
        ++output_index;
    }
    return result;
}

/* =========================================================================
 * cnp_nonzero - Return indices of non-zero elements
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_nonzero(const CnpArray *arr) {
    const char *function_name = "cnp_nonzero";
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "array is NULL");
        return NULL;
    }
    CnpArray *result = cnp_array_nonzero(arr);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_apply_along_axis - Apply function along axis
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_apply_along_axis(
    double (*func)(const double*, int64_t, void*),
    int axis, const CnpArray *arr, void *userdata)
{
    const char *function_name = "cnp_apply_along_axis";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "source array is required");
        return NULL;
    }
    if (!func) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "callback is required");
        return NULL;
    }
    if (!(arr->dtype->type_num == CNP_BOOL ||
          cnp_type_is_integer(arr->dtype->type_num) ||
          cnp_type_is_float(arr->dtype->type_num))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %d cannot be represented by the real double callback",
            (int)arr->dtype->type_num);
        return NULL;
    }
    if (axis < 0) axis += arr->ndim;
    if (axis < 0 || axis >= arr->ndim) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "axis %d is out of bounds for array of dimension %d",
            axis, arr->ndim);
        return NULL;
    }

    for (int dimension = 0; dimension < arr->ndim; ++dimension) {
        if (dimension != axis && arr->shape[dimension] == 0) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "cannot apply along an axis when an iteration dimension is zero");
            return NULL;
        }
    }

    /* Output shape: same as input but without the axis dimension */
    int out_ndim = arr->ndim - 1;
    int64_t out_shape[CNP_MAXDIMS];
    int oi = 0;
    for (int i = 0; i < arr->ndim; i++) {
        if (i != axis) out_shape[oi++] = arr->shape[i];
    }

    CnpArray *result = cnp_array_new(
        out_ndim, out_ndim > 0 ? out_shape : NULL,
        CNP_DOUBLE, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    /* Iterate over all positions except along axis */
    int64_t axis_len = arr->shape[axis];
    if ((uint64_t)axis_len > SIZE_MAX / sizeof(double)) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "axis line exceeds addressable memory");
        return NULL;
    }
    size_t line_bytes = (size_t)(axis_len > 0 ? axis_len : 1) *
        sizeof(double);
    double *line = (double*)cnp_malloc(line_bytes);
    if (!line) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate callback line buffer");
        return NULL;
    }

    double *out = (double*)result->data;
    int64_t total_out = result->size;
    for (int64_t i = 0; i < total_out; i++) {
        /* Convert output flat index to multi-index (without axis) */
        int64_t indices[CNP_MAXDIMS] = {0};
        int64_t tmp = i;
        for (int d = out_ndim - 1; d >= 0; d--) {
            indices[d] = tmp % result->shape[d];
            tmp /= result->shape[d];
        }
        /* Insert axis dimension */
        int64_t full_indices[CNP_MAXDIMS];
        int fi = 0;
        for (int d = 0; d < arr->ndim; d++) {
            if (d == axis) {
                full_indices[d] = 0;
            } else {
                full_indices[d] = indices[fi++];
            }
        }
        /* Extract line along axis */
        for (int64_t k = 0; k < axis_len; k++) {
            full_indices[axis] = k;
            int64_t off = arr->offset + cnp_multi_to_offset(
                arr->ndim, full_indices, arr->strides);
            line[k] = cnp_get_element_double(arr->data, off, arr->dtype->type_num);
        }
        out[i] = func(line, axis_len, userdata);
    }

    cnp_free(line, line_bytes);
    return result;
}

/* =========================================================================
 * cnp_apply_over_axes - Apply function over multiple axes
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_apply_over_axes(
    double (*func)(const double*, int64_t, void*),
    int naxes, const int *axes, const CnpArray *arr, void *userdata)
{
    const char *function_name = "cnp_apply_over_axes";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "source array is required");
        return NULL;
    }
    if (!func) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "callback is required");
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

    CnpArray *current = cnp_array_copy(arr);
    if (!current) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    for (int ax = 0; ax < naxes; ax++) {
        int axis = axes[ax];
        if (axis < 0) axis += current->ndim;
        if (axis < 0 || axis >= current->ndim) {
            cnp_array_free(current);
            cnp_set_error(
                CNP_ERR_AXIS, function_name,
                "axis %d is out of bounds for array of dimension %d",
                axes[ax], arr->ndim);
            return NULL;
        }
        CnpArray *reduced = cnp_apply_along_axis(
            func, axis, current, userdata);
        cnp_array_free(current);
        if (!reduced) {
            cnp_relabel_error(function_name);
            return NULL;
        }
        current = cnp_expand_dims(reduced, axis);
        cnp_array_decref(reduced);
        if (!current) {
            cnp_relabel_error(function_name);
            return NULL;
        }
    }
    return current;
}

/* =========================================================================
 * cnp_frompyfunc - Create array from scalar function
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_frompyfunc(double (*func)(double, void*), const CnpArray *arr, void *userdata) {
    const char *function_name = "cnp_frompyfunc";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "source array is required");
        return NULL;
    }
    if (!func) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "callback is required");
        return NULL;
    }
    if (!(arr->dtype->type_num == CNP_BOOL ||
          cnp_type_is_integer(arr->dtype->type_num) ||
          cnp_type_is_float(arr->dtype->type_num))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %d cannot be represented by the real double callback",
            (int)arr->dtype->type_num);
        return NULL;
    }
    CNP_ORDER order =
        (arr->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(arr->flags & CNP_ARRAY_C_CONTIGUOUS)
            ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        arr->ndim, arr->shape, CNP_DOUBLE, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int64_t i = 0; i < arr->size; i++) {
        double value = func(cnp_array_flat_get(arr, i), userdata);
        int64_t coordinates[CNP_MAXDIMS] = {0};
        int64_t remaining = i;
        for (int dimension = arr->ndim - 1;
             dimension >= 0; --dimension) {
            coordinates[dimension] = remaining % arr->shape[dimension];
            remaining /= arr->shape[dimension];
        }
        int64_t output_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        *(double*)((char*)result->data + output_offset) = value;
    }
    return result;
}

/* =========================================================================
 * cnp_vectorize - Vectorize a scalar function over array
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_vectorize(double (*func)(double, void*), const CnpArray *arr, void *userdata) {
    CnpArray *result = cnp_frompyfunc(func, arr, userdata);
    if (!result) cnp_relabel_error("cnp_vectorize");
    return result;
}

/* =========================================================================
 * cnp_trim_zeros - Trim leading/trailing zeros
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_trim_zeros(const CnpArray *arr, const char *trim) {
    const char *function_name = "cnp_trim_zeros";
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "array is NULL");
        return NULL;
    }
    if (arr->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "array must be one-dimensional");
        return NULL;
    }
    if (!trim) trim = "fb";

    int64_t start = 0, end = arr->size;

    if (strchr(trim, 'f')) {
        while (start < end) {
            bool truth = false;
            if (stride_utility_element_truth(
                    arr, start, &truth, function_name) != CNP_OK)
                return NULL;
            if (truth) break;
            ++start;
        }
    }
    if (strchr(trim, 'b')) {
        while (end > start) {
            bool truth = false;
            if (stride_utility_element_truth(
                    arr, end - 1, &truth, function_name) != CNP_OK)
                return NULL;
            if (truth) break;
            --end;
        }
    }

    int64_t new_size = end - start;
    int64_t shape[1] = {new_size};
    int64_t strides[1] = {arr->strides[0]};
    int64_t view_offset = arr->offset + start * arr->strides[0];
    CnpArray *result = cnp_array_view_from_metadata(
        (CnpArray*)arr, 1, shape, strides, view_offset, 0);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    return result;
}

/* cnp_swapaxes is already defined in shape.c */
/* cnp_moveaxis is already defined in shape.c */

/* =========================================================================
 * cnp_rollaxis - Roll axis to new position
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_rollaxis(const CnpArray *arr, int axis, int start) {
    const char *function_name = "cnp_rollaxis";
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "array is NULL");
        return NULL;
    }
    if (axis < 0) axis += arr->ndim;
    if (start < 0) start += arr->ndim;
    if (axis < 0 || axis >= arr->ndim) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "axis is outside the array rank");
        return NULL;
    }
    if (start < 0 || start > arr->ndim) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "start is outside the valid insertion range");
        return NULL;
    }

    int dest = (start > axis) ? start - 1 : start;
    CnpArray *result = cnp_moveaxis(arr, axis, dest);
    if (!result) cnp_relabel_error(function_name);
    return result;
}
