/**
 * cnumpy indexing and slicing operations
 */
#include "../include/cnumpy/cnumpy_internal.h"

/* =========================================================================
 * Slice helpers
 * ========================================================================= */
CNP_STATUS cnp_apply_slice(const CnpArray *arr, const CnpSlice *slice, int axis,
                            int64_t *new_dim, int64_t *new_stride, int64_t *offset) {
    int64_t dim = arr->shape[axis];
    int64_t stride = arr->strides[axis];

    int64_t start, stop, step;

    /* Step */
    step = slice->has_step ? slice->step : 1;
    if (step == 0) {
        cnp_set_error(CNP_ERR_INDEX, "slice", "slice step cannot be zero");
        return CNP_ERR_INDEX;
    }

    /* Start */
    if (slice->has_start) {
        start = slice->start;
        if (start < 0) start += dim;
        if (start < 0) start = (step > 0) ? 0 : dim - 1;
        if (start >= dim) start = (step > 0) ? dim : dim - 1;
    } else {
        start = (step > 0) ? 0 : dim - 1;
    }

    /* Stop */
    if (slice->has_stop) {
        stop = slice->stop;
        if (stop < 0) stop += dim;
        if (stop < 0) stop = (step > 0) ? 0 : -1;
        if (stop > dim) stop = dim;
    } else {
        stop = (step > 0) ? dim : -1;
    }

    /* Compute new dimension */
    int64_t n;
    if (step > 0) {
        n = (stop > start) ? (stop - start + step - 1) / step : 0;
    } else {
        n = (start > stop) ? (start - stop - step - 1) / (-step) : 0;
    }

    *new_dim = n;
    *new_stride = stride * step;
    *offset = start * stride;
    return CNP_OK;
}

/* =========================================================================
 * Slicing
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_array_slice(CnpArray *arr, int ndim_slices, const CnpSlice *slices) {
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_array_slice", "source array is required");
        return NULL;
    }
    if (!slices) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_array_slice", "slice metadata is required");
        return NULL;
    }

    int ndim = arr->ndim;
    int64_t new_shape[CNP_MAXDIMS];
    int64_t new_strides[CNP_MAXDIMS];
    int64_t total_offset = arr->offset;
    int new_ndim = 0;

    for (int i = 0; i < ndim; i++) {
        if (i < ndim_slices) {
            int64_t dim, stride, off;
            CNP_STATUS st = cnp_apply_slice(arr, &slices[i], i, &dim, &stride, &off);
            if (st != CNP_OK) return NULL;
            new_shape[new_ndim] = dim;
            new_strides[new_ndim] = stride;
            total_offset += off;
            new_ndim++;
        } else {
            new_shape[new_ndim] = arr->shape[i];
            new_strides[new_ndim] = arr->strides[i];
            new_ndim++;
        }
    }

    return cnp_array_view_from_metadata(
        arr, new_ndim, new_shape, new_strides, total_offset, 0);
}

/* =========================================================================
 * Get item (single element or sub-array)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_array_getitem(CnpArray *arr, const int64_t *indices) {
    const char *function_name = "cnp_array_getitem";
    if (!arr || (arr->ndim > 0 && !indices)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "array and complete indices are required");
        return NULL;
    }
    int64_t offset = arr->offset;
    for (int dimension = 0; dimension < arr->ndim; ++dimension) {
        int64_t index = indices[dimension];
        int64_t original = index;
        if (index < 0) index += arr->shape[dimension];
        if (index < 0 || index >= arr->shape[dimension]) {
            cnp_set_error(
                CNP_ERR_INDEX, function_name,
                "index %lld is out of bounds for axis %d with size %lld",
                (long long)original, dimension,
                (long long)arr->shape[dimension]);
            return NULL;
        }
        offset += index * arr->strides[dimension];
    }
    CnpArray *result = cnp_array_new(
        0, NULL, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    memcpy(
        result->data, (const char*)arr->data + offset,
        (size_t)arr->dtype->elsize);
    return result;
}

/* =========================================================================
 * Take (fancy indexing with 1D index array)
 * ========================================================================= */
typedef struct {
    const CnpArray *array;
    int axis;
    int64_t outer;
    int64_t axis_size;
    int64_t inner;
    bool axis_none;
    bool inner_contiguous;
} CnpAxisTraversal;

static int64_t indexing_flat_offset(const CnpArray *array, int64_t flat) {
    int64_t offset = array->offset;
    for (int dimension = array->ndim - 1; dimension >= 0; --dimension) {
        int64_t coordinate = flat % array->shape[dimension];
        flat /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

static bool indexing_is_signed_integer(CNP_TYPE type) {
    return type == CNP_BYTE || type == CNP_SHORT || type == CNP_INT ||
           type == CNP_LONG || type == CNP_LONGLONG;
}

static bool indexing_is_unsigned_integer(CNP_TYPE type) {
    return type == CNP_UBYTE || type == CNP_USHORT || type == CNP_UINT ||
           type == CNP_ULONG || type == CNP_ULONGLONG;
}

static int64_t indexing_read_signed_integer(
    const CnpArray *array, int64_t flat) {
    return cnp_get_element_int(
        array->data, indexing_flat_offset(array, flat),
        array->dtype->type_num);
}

static uint64_t indexing_read_unsigned_integer(
    const CnpArray *array, int64_t flat) {
    const char *pointer = (const char*)array->data +
        indexing_flat_offset(array, flat);
    switch (array->dtype->type_num) {
        case CNP_BOOL: return *(const int8_t*)pointer != 0;
        case CNP_UBYTE: return *(const uint8_t*)pointer;
        case CNP_USHORT: return *(const uint16_t*)pointer;
        case CNP_UINT: return *(const uint32_t*)pointer;
        case CNP_ULONG:
        case CNP_ULONGLONG: return *(const uint64_t*)pointer;
        default: return (uint64_t)indexing_read_signed_integer(array, flat);
    }
}

static bool indexing_require_integer(
    const CnpArray *array, const char *function_name, const char *argument) {
    if (array && cnp_type_is_integer(array->dtype->type_num)) return true;
    cnp_set_error(CNP_ERR_TYPE, function_name,
                  "%s must have an integer dtype", argument);
    return false;
}

static bool indexing_require_take_indices(
    const CnpArray *array, const char *function_name) {
    if (array && (array->dtype->type_num == CNP_BOOL ||
                  indexing_is_signed_integer(array->dtype->type_num) ||
                  (indexing_is_unsigned_integer(array->dtype->type_num) &&
                   array->dtype->elsize < (int)sizeof(int64_t))))
        return true;
    cnp_set_error(CNP_ERR_TYPE, function_name,
                  "indices must have an integer dtype safely castable to int64");
    return false;
}

static bool indexing_element_truth_at(
    const CnpArray *array, int64_t offset) {
    const char *pointer = (const char*)array->data + offset;
    switch (array->dtype->type_num) {
        case CNP_BOOL:
        case CNP_BYTE: return *(const int8_t*)pointer != 0;
        case CNP_UBYTE: return *(const uint8_t*)pointer != 0;
        case CNP_SHORT: return *(const int16_t*)pointer != 0;
        case CNP_USHORT: return *(const uint16_t*)pointer != 0;
        case CNP_INT: return *(const int32_t*)pointer != 0;
        case CNP_UINT: return *(const uint32_t*)pointer != 0;
        case CNP_LONG:
        case CNP_LONGLONG: return *(const int64_t*)pointer != 0;
        case CNP_ULONG:
        case CNP_ULONGLONG: return *(const uint64_t*)pointer != 0;
        case CNP_FLOAT: {
            float value = *(const float*)pointer;
            return value < 0.0f || value > 0.0f || isnan(value);
        }
        case CNP_DOUBLE: {
            double value = *(const double*)pointer;
            return value < 0.0 || value > 0.0 || isnan(value);
        }
        case CNP_LONGDOUBLE: {
            long double value = *(const long double*)pointer;
            return value < 0.0L || value > 0.0L || isnan(value);
        }
        case CNP_CFLOAT: {
            const cnp_cfloat *value = (const cnp_cfloat*)pointer;
            return value->real < 0.0f || value->real > 0.0f ||
                value->imag < 0.0f || value->imag > 0.0f ||
                isnan(value->real) || isnan(value->imag);
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *value = (const cnp_cdouble*)pointer;
            return value->real < 0.0 || value->real > 0.0 ||
                value->imag < 0.0 || value->imag > 0.0 ||
                isnan(value->real) || isnan(value->imag);
        }
        case CNP_CLONGDOUBLE: {
            const cnp_clongdouble *value = (const cnp_clongdouble*)pointer;
            return value->real < 0.0L || value->real > 0.0L ||
                value->imag < 0.0L || value->imag > 0.0L ||
                isnan(value->real) || isnan(value->imag);
        }
        default: return false;
    }
}

static bool indexing_element_truth(const CnpArray *array, int64_t flat) {
    return indexing_element_truth_at(array, indexing_flat_offset(array, flat));
}

static bool indexing_axis_traversal(
    const CnpArray *array, int axis, bool axis_none,
    const char *function_name, CnpAxisTraversal *traversal) {
    if (!array) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "source array must not be null");
        return false;
    }
    traversal->array = array;
    traversal->axis_none = axis_none;
    traversal->outer = 1;
    traversal->inner = 1;
    if (axis_none) {
        traversal->axis = 0;
        traversal->axis_size = array->size;
        traversal->inner_contiguous =
            (array->flags & CNP_ARRAY_C_CONTIGUOUS) != 0;
        return true;
    }
    int normalized = axis;
    if (normalized < 0) normalized += array->ndim;
    if (normalized < 0 || normalized >= array->ndim) {
        cnp_set_error(CNP_ERR_AXIS, function_name,
                      "axis %d is out of bounds for array of dimension %d",
                      axis, array->ndim);
        return false;
    }
    traversal->axis = normalized;
    traversal->axis_size = array->shape[normalized];
    for (int dimension = 0; dimension < normalized; ++dimension)
        traversal->outer *= array->shape[dimension];
    for (int dimension = normalized + 1;
         dimension < array->ndim; ++dimension)
        traversal->inner *= array->shape[dimension];

    int64_t expected_stride = array->dtype->elsize;
    traversal->inner_contiguous = true;
    for (int dimension = array->ndim - 1;
         dimension > normalized; --dimension) {
        if (array->shape[dimension] > 1 &&
            array->strides[dimension] != expected_stride)
            traversal->inner_contiguous = false;
        expected_stride *= array->shape[dimension];
    }
    return true;
}

static int64_t indexing_outer_offset(
    const CnpAxisTraversal *traversal, int64_t outer) {
    const CnpArray *array = traversal->array;
    int64_t offset = array->offset;
    for (int dimension = traversal->axis - 1;
         dimension >= 0; --dimension) {
        int64_t coordinate = outer % array->shape[dimension];
        outer /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

static void indexing_copy_axis_block(
    const CnpAxisTraversal *traversal, int64_t outer, int64_t axis_index,
    char *destination) {
    const CnpArray *array = traversal->array;
    int itemsize = array->dtype->elsize;
    if (traversal->axis_none) {
        int64_t offset = traversal->inner_contiguous
            ? array->offset + axis_index * itemsize
            : indexing_flat_offset(array, axis_index);
        memcpy(destination, (const char*)array->data + offset, itemsize);
        return;
    }
    int64_t base = indexing_outer_offset(traversal, outer) +
                   axis_index * array->strides[traversal->axis];
    if (traversal->inner_contiguous) {
        memcpy(destination, (const char*)array->data + base,
               (size_t)traversal->inner * itemsize);
        return;
    }
    for (int64_t inner = 0; inner < traversal->inner; ++inner) {
        int64_t remaining = inner;
        int64_t offset = base;
        for (int dimension = array->ndim - 1;
             dimension > traversal->axis; --dimension) {
            int64_t coordinate = remaining % array->shape[dimension];
            remaining /= array->shape[dimension];
            offset += coordinate * array->strides[dimension];
        }
        memcpy(destination + inner * itemsize,
               (const char*)array->data + offset, itemsize);
    }
}

static void indexing_copy_axis_element(
    const CnpAxisTraversal *traversal, int64_t axis_index,
    const int64_t *source_coords, char *destination) {
    const CnpArray *array = traversal->array;
    int64_t source_offset;
    if (traversal->axis_none) {
        source_offset = traversal->inner_contiguous
            ? array->offset + axis_index * array->dtype->elsize
            : indexing_flat_offset(array, axis_index);
    } else {
        source_offset = array->offset + cnp_multi_to_offset(
            array->ndim, source_coords, array->strides);
    }
    memcpy(destination, (const char*)array->data + source_offset,
           array->dtype->elsize);
}

static bool indexing_normalize_index(
    int64_t *index, int64_t length,
    const char *function_name, int axis) {
    int64_t original = *index;
    if (length == 0) {
        cnp_set_error(CNP_ERR_INDEX, function_name,
                      "index %lld is out of bounds for axis %d with size 0",
                      (long long)original, axis);
        return false;
    }
    if (*index < 0) {
        if (*index < -length) {
            cnp_set_error(CNP_ERR_INDEX, function_name,
                          "index %lld is out of bounds for axis %d with size %lld",
                          (long long)original, axis, (long long)length);
            return false;
        }
        *index += length;
    }
    if (*index >= length) {
        cnp_set_error(CNP_ERR_INDEX, function_name,
                      "index %lld is out of bounds for axis %d with size %lld",
                      (long long)original, axis, (long long)length);
        return false;
    }
    return true;
}

static bool indexing_read_normalized_index(
    const CnpArray *indices, int64_t flat, int64_t length, bool allow_end,
    const char *function_name, int axis, int64_t *normalized) {
    if (indexing_is_unsigned_integer(indices->dtype->type_num)) {
        uint64_t value = indexing_read_unsigned_integer(indices, flat);
        uint64_t limit = (uint64_t)length;
        if (value > limit || (!allow_end && value == limit)) {
            cnp_set_error(CNP_ERR_INDEX, function_name,
                          "index %llu is out of bounds for axis %d with size %lld",
                          (unsigned long long)value, axis, (long long)length);
            return false;
        }
        *normalized = (int64_t)value;
        return true;
    }
    int64_t value = indices->dtype->type_num == CNP_BOOL
        ? (int64_t)indexing_read_unsigned_integer(indices, flat)
        : indexing_read_signed_integer(indices, flat);
    int64_t minimum = -length;
    if (value < minimum || value > length || (!allow_end && value == length)) {
        cnp_set_error(CNP_ERR_INDEX, function_name,
                      "index %lld is out of bounds for axis %d with size %lld",
                      (long long)value, axis, (long long)length);
        return false;
    }
    if (value < 0) value += length;
    *normalized = value;
    return true;
}

static CnpArray *indexing_take_impl(
    const char *function_name, const CnpArray *arr,
    const CnpArray *indices, int axis, bool axis_none) {
    if (!indices || !indexing_require_take_indices(indices, function_name))
        return NULL;
    bool scalar_axis = !axis_none && arr && arr->ndim == 0 &&
        (axis == 0 || axis == -1);
    if (!axis_none && arr && arr->ndim == 0 && !scalar_axis) {
        cnp_set_error(CNP_ERR_AXIS, function_name,
                      "axis %d is out of bounds for array of dimension 1", axis);
        return NULL;
    }
    bool flattened = axis_none || scalar_axis;
    CnpAxisTraversal traversal;
    if (!indexing_axis_traversal(
            arr, axis, flattened, function_name, &traversal)) return NULL;

    int output_ndim = flattened
        ? indices->ndim : arr->ndim - 1 + indices->ndim;
    if (output_ndim > CNP_MAXDIMS) {
        cnp_set_error(CNP_ERR_SHAPE, function_name,
                      "result has too many dimensions: %d", output_ndim);
        return NULL;
    }
    int64_t output_shape[CNP_MAXDIMS];
    if (flattened) {
        for (int dimension = 0; dimension < indices->ndim; ++dimension)
            output_shape[dimension] = indices->shape[dimension];
    } else {
        int output_dimension = 0;
        for (int dimension = 0; dimension < traversal.axis; ++dimension)
            output_shape[output_dimension++] = arr->shape[dimension];
        for (int dimension = 0; dimension < indices->ndim; ++dimension)
            output_shape[output_dimension++] = indices->shape[dimension];
        for (int dimension = traversal.axis + 1;
             dimension < arr->ndim; ++dimension)
            output_shape[output_dimension++] = arr->shape[dimension];
    }
    CnpArray *result = cnp_array_new(
        output_ndim, output_shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) return NULL;

    int itemsize = arr->dtype->elsize;
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t item = 0; item < indices->size; ++item) {
            int64_t index = indexing_read_signed_integer(indices, item);
            if (!indexing_normalize_index(
                    &index, traversal.axis_size, function_name,
                    flattened ? 0 : traversal.axis)) {
                cnp_array_decref(result);
                return NULL;
            }
            char *destination = (char*)result->data +
                (outer * indices->size + item) * traversal.inner * itemsize;
            indexing_copy_axis_block(
                &traversal, outer, index, destination);
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_take_v2(
    const CnpArray *arr, const CnpArray *indices,
    int axis, bool axis_none) {
    return indexing_take_impl(
        "cnp_take_v2", arr, indices, axis, axis_none);
}

CNP_API CnpArray* CNP_CALL cnp_array_take(
    const CnpArray *arr, const CnpArray *indices, int axis) {
    return indexing_take_impl(
        "cnp_array_take", arr, indices, axis, axis < 0);
}

static void indexing_flat_to_coords(
    int ndim, const int64_t *shape, int64_t flat, int64_t *coords) {
    for (int dimension = ndim - 1; dimension >= 0; --dimension) {
        coords[dimension] = flat % shape[dimension];
        flat /= shape[dimension];
    }
}

CNP_API CnpArray* CNP_CALL cnp_take_along_axis_v2(
    const CnpArray *arr, const CnpArray *indices,
    int axis, bool axis_none) {
    const char *function_name = "cnp_take_along_axis_v2";
    if (!indices || !indexing_require_integer(
            indices, function_name, "indices")) return NULL;
    CnpAxisTraversal traversal;
    if (!indexing_axis_traversal(
            arr, axis, axis_none, function_name, &traversal)) return NULL;
    int normalized_axis = traversal.axis;
    int result_ndim = arr->ndim;
    if (axis_none) {
        result_ndim = 1;
        normalized_axis = 0;
        if (indices->ndim != 1) {
            cnp_set_error(CNP_ERR_SHAPE, function_name,
                          "indices must be one-dimensional when axis is None");
            return NULL;
        }
    } else {
        if (indices->ndim != arr->ndim) {
            cnp_set_error(CNP_ERR_SHAPE, function_name,
                          "indices and source must have the same rank");
            return NULL;
        }
    }

    int64_t result_shape[CNP_MAXDIMS];
    if (axis_none) {
        result_shape[0] = indices->shape[0];
    } else {
        for (int dimension = 0; dimension < arr->ndim; ++dimension) {
            if (dimension == normalized_axis) {
                result_shape[dimension] = indices->shape[dimension];
            } else if (arr->shape[dimension] == indices->shape[dimension] ||
                       arr->shape[dimension] == 1) {
                result_shape[dimension] = indices->shape[dimension];
            } else if (indices->shape[dimension] == 1) {
                result_shape[dimension] = arr->shape[dimension];
            } else {
                cnp_set_error(CNP_ERR_BROADCAST, function_name,
                              "source and indices do not broadcast at axis %d",
                              dimension);
                return NULL;
            }
        }
    }
    CnpArray *result = cnp_array_new(
        result_ndim, result_shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) return NULL;

    int64_t output_coords[CNP_MAXDIMS];
    int64_t index_coords[CNP_MAXDIMS];
    int64_t source_coords[CNP_MAXDIMS];
    int itemsize = arr->dtype->elsize;
    for (int64_t flat = 0; flat < result->size; ++flat) {
        indexing_flat_to_coords(
            result_ndim, result_shape, flat, output_coords);
        int64_t index_offset;
        if (axis_none) {
            index_offset = indices->offset +
                output_coords[0] * indices->strides[0];
        } else {
            for (int dimension = 0; dimension < result_ndim; ++dimension)
                index_coords[dimension] = indices->shape[dimension] == 1
                    ? 0 : output_coords[dimension];
            index_offset = indices->offset + cnp_multi_to_offset(
                indices->ndim, index_coords, indices->strides);
        }
        uint64_t raw_unsigned = 0;
        int64_t selected;
        if (indexing_is_unsigned_integer(indices->dtype->type_num)) {
            const char *pointer = (const char*)indices->data + index_offset;
            switch (indices->dtype->type_num) {
                case CNP_UBYTE: raw_unsigned = *(const uint8_t*)pointer; break;
                case CNP_USHORT: raw_unsigned = *(const uint16_t*)pointer; break;
                case CNP_UINT: raw_unsigned = *(const uint32_t*)pointer; break;
                default: raw_unsigned = *(const uint64_t*)pointer; break;
            }
            selected = raw_unsigned <= INT64_MAX
                ? (int64_t)raw_unsigned
                : -(int64_t)(UINT64_MAX - raw_unsigned) - 1;
        } else {
            selected = cnp_get_element_int(
                indices->data, index_offset, indices->dtype->type_num);
        }
        if (!indexing_normalize_index(
                &selected, traversal.axis_size, function_name,
                normalized_axis)) {
            cnp_array_decref(result);
            return NULL;
        }
        if (!axis_none) {
            for (int dimension = 0; dimension < result_ndim; ++dimension)
                source_coords[dimension] = dimension == normalized_axis
                    ? selected
                    : (arr->shape[dimension] == 1
                        ? 0 : output_coords[dimension]);
        }
        indexing_copy_axis_element(
            &traversal, selected, source_coords,
            (char*)result->data + flat * itemsize);
    }
    return result;
}

static bool indexing_put_broadcast_shape(
    const CnpArray *arr, const CnpArray *indices, int axis,
    int64_t *iteration_shape, int64_t *iteration_size,
    const char *function_name) {
    *iteration_size = 1;
    for (int dimension = 0; dimension < arr->ndim; ++dimension) {
        int64_t length;
        if (dimension == axis) {
            length = indices->shape[dimension];
        } else if (arr->shape[dimension] == indices->shape[dimension] ||
                   arr->shape[dimension] == 1) {
            length = indices->shape[dimension];
        } else if (indices->shape[dimension] == 1) {
            length = arr->shape[dimension];
        } else {
            cnp_set_error(
                CNP_ERR_BROADCAST, function_name,
                "destination and indices do not broadcast at axis %d",
                dimension);
            return false;
        }
        iteration_shape[dimension] = length;
        if (length == 0) {
            *iteration_size = 0;
        } else if (*iteration_size > 0) {
            if (*iteration_size > INT64_MAX / length) {
                cnp_set_error(
                    CNP_ERR_MEMORY, function_name,
                    "broadcast iteration size exceeds int64 capacity");
                return false;
            }
            *iteration_size *= length;
        }
    }
    return true;
}

static bool indexing_put_values_broadcastable(
    const CnpArray *values, int target_ndim, const int64_t *target_shape,
    const char *function_name) {
    int leading = values->ndim - target_ndim;
    if (leading > 0) {
        for (int dimension = 0; dimension < leading; ++dimension) {
            if (values->shape[dimension] != 1) {
                cnp_set_error(
                    CNP_ERR_BROADCAST, function_name,
                    "values shape cannot broadcast to indexed destination shape");
                return false;
            }
        }
    }
    for (int target_dimension = 0;
         target_dimension < target_ndim; ++target_dimension) {
        int source_dimension = target_dimension + leading;
        if (source_dimension < 0) continue;
        int64_t source_length = values->shape[source_dimension];
        if (source_length != 1 &&
                source_length != target_shape[target_dimension]) {
            cnp_set_error(
                CNP_ERR_BROADCAST, function_name,
                "values shape cannot broadcast at axis %d", target_dimension);
            return false;
        }
    }
    return true;
}

static int64_t indexing_put_broadcast_offset(
    const CnpArray *array, int target_ndim,
    const int64_t *target_coordinates) {
    int leading = array->ndim - target_ndim;
    int64_t offset = array->offset;
    for (int source_dimension = 0;
         source_dimension < array->ndim; ++source_dimension) {
        int target_dimension = source_dimension - leading;
        int64_t coordinate = target_dimension < 0 ||
            array->shape[source_dimension] == 1
                ? 0 : target_coordinates[target_dimension];
        offset += coordinate * array->strides[source_dimension];
    }
    return offset;
}

static bool indexing_put_read_index(
    const CnpArray *indices, int64_t offset, int64_t axis_length,
    int axis, int64_t *normalized, const char *function_name) {
    const char *pointer = (const char*)indices->data + offset;
    if (indexing_is_unsigned_integer(indices->dtype->type_num)) {
        uint64_t value;
        switch (indices->dtype->type_num) {
            case CNP_UBYTE: value = *(const uint8_t*)pointer; break;
            case CNP_USHORT: value = *(const uint16_t*)pointer; break;
            case CNP_UINT: value = *(const uint32_t*)pointer; break;
            default: value = *(const uint64_t*)pointer; break;
        }
        if (value >= (uint64_t)axis_length) {
            cnp_set_error(
                CNP_ERR_INDEX, function_name,
                "index %llu is out of bounds for axis %d with size %lld",
                (unsigned long long)value, axis, (long long)axis_length);
            return false;
        }
        *normalized = (int64_t)value;
        return true;
    }
    int64_t value = cnp_get_element_int(
        indices->data, offset, indices->dtype->type_num);
    if (!indexing_normalize_index(
            &value, axis_length, function_name, axis)) return false;
    *normalized = value;
    return true;
}

CNP_API CNP_STATUS CNP_CALL cnp_put_along_axis(
    CnpArray *arr, const CnpArray *indices,
    const CnpArray *values, int axis) {
    const char *function_name = "cnp_put_along_axis";
    if (!arr || !indices || !values) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "destination, indices, and values arrays are required");
        return CNP_ERR_GENERIC;
    }
    if (!(arr->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "destination array is not writeable");
        return CNP_ERR_GENERIC;
    }
    if (!indexing_require_integer(indices, function_name, "indices"))
        return CNP_ERR_TYPE;
    int normalized_axis = axis;
    if (normalized_axis < 0) normalized_axis += arr->ndim;
    if (normalized_axis < 0 || normalized_axis >= arr->ndim) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "axis %d is out of bounds for array of dimension %d",
            axis, arr->ndim);
        return CNP_ERR_AXIS;
    }
    if (indices->ndim != arr->ndim) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "indices and destination must have the same rank");
        return CNP_ERR_SHAPE;
    }

    int64_t iteration_shape[CNP_MAXDIMS];
    int64_t iteration_size;
    if (!indexing_put_broadcast_shape(
            arr, indices, normalized_axis,
            iteration_shape, &iteration_size, function_name))
        return cnp_get_error(NULL);
    if (!indexing_put_values_broadcastable(
            values, arr->ndim, iteration_shape, function_name))
        return CNP_ERR_BROADCAST;
    if (iteration_size == 0) return CNP_OK;

    size_t index_bytes;
    size_t value_bytes;
    if ((uint64_t)iteration_size > SIZE_MAX / sizeof(int64_t) ||
            (uint64_t)iteration_size >
                SIZE_MAX / (size_t)arr->dtype->elsize) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "staged mutation exceeds addressable memory");
        return CNP_ERR_MEMORY;
    }
    index_bytes = (size_t)iteration_size * sizeof(int64_t);
    value_bytes = (size_t)iteration_size * (size_t)arr->dtype->elsize;
    int64_t *normalized_indices = (int64_t*)cnp_malloc(index_bytes);
    char *staged_values = (char*)cnp_malloc(value_bytes);
    if (!normalized_indices || !staged_values) {
        cnp_free(normalized_indices, index_bytes);
        cnp_free(staged_values, value_bytes);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate atomic mutation staging buffers");
        return CNP_ERR_MEMORY;
    }

    CNP_STATUS status = CNP_OK;
    int64_t coordinates[CNP_MAXDIMS];
    int64_t index_coordinates[CNP_MAXDIMS];
    for (int64_t flat = 0; flat < iteration_size; ++flat) {
        indexing_flat_to_coords(
            arr->ndim, iteration_shape, flat, coordinates);
        for (int dimension = 0; dimension < arr->ndim; ++dimension)
            index_coordinates[dimension] =
                indices->shape[dimension] == 1
                    ? 0 : coordinates[dimension];
        int64_t index_offset = indices->offset + cnp_multi_to_offset(
            indices->ndim, index_coordinates, indices->strides);
        if (!indexing_put_read_index(
                indices, index_offset, arr->shape[normalized_axis],
                normalized_axis, &normalized_indices[flat], function_name)) {
            status = CNP_ERR_INDEX;
            break;
        }
        int64_t value_offset = indexing_put_broadcast_offset(
            values, arr->ndim, coordinates);
        status = cnp_cast_scalar_value(
            (const char*)values->data + value_offset,
            values->dtype->type_num,
            staged_values + flat * arr->dtype->elsize,
            arr->dtype->type_num, function_name);
        if (status != CNP_OK) break;
    }

    if (status == CNP_OK) {
        int64_t destination_coordinates[CNP_MAXDIMS];
        for (int64_t flat = 0; flat < iteration_size; ++flat) {
            indexing_flat_to_coords(
                arr->ndim, iteration_shape, flat, coordinates);
            for (int dimension = 0; dimension < arr->ndim; ++dimension) {
                destination_coordinates[dimension] =
                    dimension == normalized_axis
                        ? normalized_indices[flat]
                        : (arr->shape[dimension] == 1
                            ? 0 : coordinates[dimension]);
            }
            int64_t destination_offset = arr->offset + cnp_multi_to_offset(
                arr->ndim, destination_coordinates, arr->strides);
            memcpy(
                (char*)arr->data + destination_offset,
                staged_values + flat * arr->dtype->elsize,
                (size_t)arr->dtype->elsize);
        }
    }
    cnp_free(normalized_indices, index_bytes);
    cnp_free(staged_values, value_bytes);
    return status;
}

CNP_API CnpArray* CNP_CALL cnp_compress_v2(
    const CnpArray *condition, const CnpArray *arr,
    int axis, bool axis_none) {
    const char *function_name = "cnp_compress_v2";
    if (!condition || condition->ndim != 1) {
        cnp_set_error(CNP_ERR_SHAPE, function_name,
                      "condition must be one-dimensional");
        return NULL;
    }
    bool scalar_axis = !axis_none && arr && arr->ndim == 0 &&
        (axis == 0 || axis == -1);
    if (!axis_none && arr && arr->ndim == 0 && !scalar_axis) {
        cnp_set_error(CNP_ERR_AXIS, function_name,
                      "axis %d is out of bounds for array of dimension 1", axis);
        return NULL;
    }
    bool flattened = axis_none || scalar_axis;
    CnpAxisTraversal traversal;
    if (!indexing_axis_traversal(
            arr, axis, flattened, function_name, &traversal)) return NULL;
    int64_t selected_count = 0;
    for (int64_t item = 0; item < condition->size; ++item) {
        bool selected = indexing_element_truth(condition, item);
        if (selected && item >= traversal.axis_size) {
            cnp_set_error(CNP_ERR_INDEX, function_name,
                          "index %lld is out of bounds for axis %d with size %lld",
                           (long long)item, flattened ? 0 : traversal.axis,
                          (long long)traversal.axis_size);
            return NULL;
        }
        if (selected) ++selected_count;
    }
    int output_ndim = flattened ? 1 : arr->ndim;
    int64_t output_shape[CNP_MAXDIMS];
    if (flattened) output_shape[0] = selected_count;
    else {
        for (int dimension = 0; dimension < arr->ndim; ++dimension)
            output_shape[dimension] = arr->shape[dimension];
        output_shape[traversal.axis] = selected_count;
    }
    CnpArray *result = cnp_array_new(
        output_ndim, output_shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) return NULL;
    int itemsize = arr->dtype->elsize;
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        int64_t destination_axis = 0;
        int64_t limit = condition->size < traversal.axis_size
            ? condition->size : traversal.axis_size;
        for (int64_t item = 0; item < limit; ++item) {
            if (!indexing_element_truth(condition, item)) continue;
            char *destination = (char*)result->data +
                (outer * selected_count + destination_axis) *
                traversal.inner * itemsize;
            indexing_copy_axis_block(
                &traversal, outer, item, destination);
            ++destination_axis;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_delete_v2(
    const CnpArray *arr, const CnpArray *obj,
    int axis, bool axis_none) {
    const char *function_name = "cnp_delete_v2";
    if (!obj) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "obj must not be null");
        return NULL;
    }
    bool boolean_object = obj->dtype->type_num == CNP_BOOL;
    if (!boolean_object &&
        !indexing_require_integer(obj, function_name, "obj"))
        return NULL;
    CnpAxisTraversal traversal;
    if (!indexing_axis_traversal(
            arr, axis, axis_none, function_name, &traversal)) return NULL;
    if (boolean_object &&
        (obj->ndim != 1 || obj->size != traversal.axis_size)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "boolean array argument obj must be one-dimensional and "
            "match the axis length of %lld",
            (long long)traversal.axis_size);
        return NULL;
    }
    size_t mask_size = traversal.axis_size > 0
        ? (size_t)traversal.axis_size : 1u;
    unsigned char *deleted = (unsigned char*)cnp_calloc(mask_size, 1);
    if (!deleted) {
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "failed to allocate deletion mask");
        return NULL;
    }
    int64_t unique_count = 0;
    if (boolean_object) {
        for (int64_t item = 0; item < obj->size; ++item) {
            if (indexing_element_truth(obj, item)) {
                deleted[item] = 1;
                ++unique_count;
            }
        }
    } else {
        for (int64_t item = 0; item < obj->size; ++item) {
            int64_t index;
            if (!indexing_read_normalized_index(
                    obj, item, traversal.axis_size, false, function_name,
                    axis_none ? 0 : traversal.axis, &index)) {
                cnp_free(deleted, mask_size);
                return NULL;
            }
            if (!deleted[index]) {
                deleted[index] = 1;
                ++unique_count;
            }
        }
    }
    int64_t output_axis_size = traversal.axis_size - unique_count;
    int output_ndim = axis_none ? 1 : arr->ndim;
    int64_t output_shape[CNP_MAXDIMS];
    if (axis_none) output_shape[0] = output_axis_size;
    else {
        for (int dimension = 0; dimension < arr->ndim; ++dimension)
            output_shape[dimension] = arr->shape[dimension];
        output_shape[traversal.axis] = output_axis_size;
    }
    CnpArray *result = cnp_array_new(
        output_ndim, output_shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_free(deleted, mask_size);
        return NULL;
    }
    int itemsize = arr->dtype->elsize;
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        int64_t destination_axis = 0;
        for (int64_t item = 0; item < traversal.axis_size; ++item) {
            if (deleted[item]) continue;
            char *destination = (char*)result->data +
                (outer * output_axis_size + destination_axis) *
                traversal.inner * itemsize;
            indexing_copy_axis_block(
                &traversal, outer, item, destination);
            ++destination_axis;
        }
    }
    cnp_free(deleted, mask_size);
    return result;
}

typedef enum {
    CNP_INDEX_VALUE_SIGNED,
    CNP_INDEX_VALUE_UNSIGNED,
    CNP_INDEX_VALUE_FLOATING,
    CNP_INDEX_VALUE_COMPLEX
} CnpIndexValueKind;

typedef struct {
    CnpIndexValueKind kind;
    int64_t signed_value;
    uint64_t unsigned_value;
    long double real;
    long double imaginary;
} CnpIndexValue;

static bool indexing_read_cast_value(
    const CnpArray *source, int64_t source_offset,
    const char *function_name, CnpIndexValue *value) {
    const char *pointer = (const char*)source->data + source_offset;
    value->imaginary = 0.0L;
    switch (source->dtype->type_num) {
        case CNP_BOOL:
            value->kind = CNP_INDEX_VALUE_SIGNED;
            value->signed_value = *(const int8_t*)pointer != 0;
            return true;
        case CNP_BYTE:
            value->kind = CNP_INDEX_VALUE_SIGNED;
            value->signed_value = *(const int8_t*)pointer;
            return true;
        case CNP_SHORT:
            value->kind = CNP_INDEX_VALUE_SIGNED;
            value->signed_value = *(const int16_t*)pointer;
            return true;
        case CNP_INT:
            value->kind = CNP_INDEX_VALUE_SIGNED;
            value->signed_value = *(const int32_t*)pointer;
            return true;
        case CNP_LONG:
        case CNP_LONGLONG:
            value->kind = CNP_INDEX_VALUE_SIGNED;
            value->signed_value = *(const int64_t*)pointer;
            return true;
        case CNP_UBYTE:
            value->kind = CNP_INDEX_VALUE_UNSIGNED;
            value->unsigned_value = *(const uint8_t*)pointer;
            return true;
        case CNP_USHORT:
            value->kind = CNP_INDEX_VALUE_UNSIGNED;
            value->unsigned_value = *(const uint16_t*)pointer;
            return true;
        case CNP_UINT:
            value->kind = CNP_INDEX_VALUE_UNSIGNED;
            value->unsigned_value = *(const uint32_t*)pointer;
            return true;
        case CNP_ULONG:
        case CNP_ULONGLONG:
            value->kind = CNP_INDEX_VALUE_UNSIGNED;
            value->unsigned_value = *(const uint64_t*)pointer;
            return true;
        case CNP_FLOAT:
            value->kind = CNP_INDEX_VALUE_FLOATING;
            value->real = *(const float*)pointer;
            return true;
        case CNP_DOUBLE:
            value->kind = CNP_INDEX_VALUE_FLOATING;
            value->real = *(const double*)pointer;
            return true;
        case CNP_LONGDOUBLE:
            value->kind = CNP_INDEX_VALUE_FLOATING;
            value->real = *(const long double*)pointer;
            return true;
        case CNP_CFLOAT: {
            const cnp_cfloat *complex_value = (const cnp_cfloat*)pointer;
            value->kind = CNP_INDEX_VALUE_COMPLEX;
            value->real = complex_value->real;
            value->imaginary = complex_value->imag;
            return true;
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *complex_value = (const cnp_cdouble*)pointer;
            value->kind = CNP_INDEX_VALUE_COMPLEX;
            value->real = complex_value->real;
            value->imaginary = complex_value->imag;
            return true;
        }
        case CNP_CLONGDOUBLE: {
            const cnp_clongdouble *complex_value =
                (const cnp_clongdouble*)pointer;
            value->kind = CNP_INDEX_VALUE_COMPLEX;
            value->real = complex_value->real;
            value->imaginary = complex_value->imag;
            return true;
        }
        default:
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "cannot cast inserted values from unsupported dtype %d",
                (int)source->dtype->type_num);
            return false;
    }
}

static uint64_t indexing_value_bits(const CnpIndexValue *value) {
    if (value->kind == CNP_INDEX_VALUE_SIGNED)
        return (uint64_t)value->signed_value;
    return value->unsigned_value;
}

static long double indexing_value_real(const CnpIndexValue *value) {
    if (value->kind == CNP_INDEX_VALUE_SIGNED)
        return (long double)value->signed_value;
    if (value->kind == CNP_INDEX_VALUE_UNSIGNED)
        return (long double)value->unsigned_value;
    return value->real;
}

static bool indexing_copy_cast_element(
    const CnpArray *source, int64_t source_offset,
    CnpArray *destination, int64_t destination_offset,
    const char *function_name) {
    if (source->dtype->type_num == destination->dtype->type_num) {
        memcpy((char*)destination->data + destination_offset,
               (const char*)source->data + source_offset,
               destination->dtype->elsize);
        return true;
    }
    char *target = (char*)destination->data + destination_offset;
    if (destination->dtype->type_num == CNP_BOOL) {
        *(int8_t*)target = indexing_element_truth_at(source, source_offset)
            ? 1 : 0;
        return true;
    }
    CnpIndexValue value;
    if (!indexing_read_cast_value(
            source, source_offset, function_name, &value)) return false;
    uint64_t bits = 0;
    if (value.kind == CNP_INDEX_VALUE_SIGNED ||
        value.kind == CNP_INDEX_VALUE_UNSIGNED)
        bits = indexing_value_bits(&value);
    switch (destination->dtype->type_num) {
        case CNP_BYTE: {
            if (value.kind == CNP_INDEX_VALUE_FLOATING ||
                value.kind == CNP_INDEX_VALUE_COMPLEX) {
                *(int8_t*)target = (int8_t)value.real;
            } else {
                uint8_t narrowed = (uint8_t)bits;
                memcpy(target, &narrowed, sizeof(narrowed));
            }
            return true;
        }
        case CNP_UBYTE:
            *(uint8_t*)target =
                value.kind == CNP_INDEX_VALUE_FLOATING ||
                value.kind == CNP_INDEX_VALUE_COMPLEX
                ? (uint8_t)value.real : (uint8_t)bits;
            return true;
        case CNP_SHORT: {
            if (value.kind == CNP_INDEX_VALUE_FLOATING ||
                value.kind == CNP_INDEX_VALUE_COMPLEX) {
                *(int16_t*)target = (int16_t)value.real;
            } else {
                uint16_t narrowed = (uint16_t)bits;
                memcpy(target, &narrowed, sizeof(narrowed));
            }
            return true;
        }
        case CNP_USHORT:
            *(uint16_t*)target =
                value.kind == CNP_INDEX_VALUE_FLOATING ||
                value.kind == CNP_INDEX_VALUE_COMPLEX
                ? (uint16_t)value.real : (uint16_t)bits;
            return true;
        case CNP_INT: {
            if (value.kind == CNP_INDEX_VALUE_FLOATING ||
                value.kind == CNP_INDEX_VALUE_COMPLEX) {
                *(int32_t*)target = (int32_t)value.real;
            } else {
                uint32_t narrowed = (uint32_t)bits;
                memcpy(target, &narrowed, sizeof(narrowed));
            }
            return true;
        }
        case CNP_UINT:
            *(uint32_t*)target =
                value.kind == CNP_INDEX_VALUE_FLOATING ||
                value.kind == CNP_INDEX_VALUE_COMPLEX
                ? (uint32_t)value.real : (uint32_t)bits;
            return true;
        case CNP_LONG:
        case CNP_LONGLONG:
            if (value.kind == CNP_INDEX_VALUE_FLOATING ||
                value.kind == CNP_INDEX_VALUE_COMPLEX) {
                *(int64_t*)target = (int64_t)value.real;
            } else {
                memcpy(target, &bits, sizeof(bits));
            }
            return true;
        case CNP_ULONG:
        case CNP_ULONGLONG:
            *(uint64_t*)target =
                value.kind == CNP_INDEX_VALUE_FLOATING ||
                value.kind == CNP_INDEX_VALUE_COMPLEX
                ? (uint64_t)value.real : bits;
            return true;
        case CNP_FLOAT:
            *(float*)target = (float)indexing_value_real(&value);
            return true;
        case CNP_DOUBLE:
            *(double*)target = (double)indexing_value_real(&value);
            return true;
        case CNP_LONGDOUBLE:
            *(long double*)target = indexing_value_real(&value);
            return true;
        case CNP_CFLOAT: {
            cnp_cfloat *complex_target = (cnp_cfloat*)target;
            complex_target->real = (float)indexing_value_real(&value);
            complex_target->imag = value.kind == CNP_INDEX_VALUE_COMPLEX
                ? (float)value.imaginary : 0.0f;
            return true;
        }
        case CNP_CDOUBLE: {
            cnp_cdouble *complex_target = (cnp_cdouble*)target;
            complex_target->real = (double)indexing_value_real(&value);
            complex_target->imag = value.kind == CNP_INDEX_VALUE_COMPLEX
                ? (double)value.imaginary : 0.0;
            return true;
        }
        case CNP_CLONGDOUBLE: {
            cnp_clongdouble *complex_target = (cnp_clongdouble*)target;
            complex_target->real = indexing_value_real(&value);
            complex_target->imag = value.kind == CNP_INDEX_VALUE_COMPLEX
                ? value.imaginary : 0.0L;
            return true;
        }
        default:
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "cannot cast inserted values to unsupported dtype %d",
                (int)destination->dtype->type_num);
            return false;
    }
}

static bool indexing_insert_values_shape(
    const CnpArray *arr, const CnpArray *values, int axis,
    bool move_first_axis, const char *function_name,
    int64_t *padded_shape, int64_t *broadcast_shape,
    int *leading_dimensions) {
    *leading_dimensions = arr->ndim - values->ndim;
    if (*leading_dimensions < 0) {
        int extra_source_dimensions = -*leading_dimensions;
        for (int dimension = 0;
             dimension < extra_source_dimensions; ++dimension) {
            if (values->shape[dimension] != 1) {
                cnp_set_error(
                    CNP_ERR_BROADCAST, function_name,
                    "values leading dimension %d with length %lld cannot "
                    "broadcast to source rank %d",
                    dimension, (long long)values->shape[dimension], arr->ndim);
                return false;
            }
        }
    }
    for (int dimension = 0; dimension < arr->ndim; ++dimension) {
        int source_dimension = dimension - *leading_dimensions;
        padded_shape[dimension] =
            source_dimension < 0 || source_dimension >= values->ndim
            ? 1 : values->shape[source_dimension];
    }
    if (move_first_axis) {
        int output_dimension = 0;
        for (int dimension = 1; dimension < arr->ndim; ++dimension) {
            if (output_dimension == axis) ++output_dimension;
            broadcast_shape[output_dimension++] = padded_shape[dimension];
        }
        broadcast_shape[axis] = padded_shape[0];
    } else {
        for (int dimension = 0; dimension < arr->ndim; ++dimension)
            broadcast_shape[dimension] = padded_shape[dimension];
    }
    for (int dimension = 0; dimension < arr->ndim; ++dimension) {
        if (dimension == axis) continue;
        if (broadcast_shape[dimension] != 1 &&
            broadcast_shape[dimension] != arr->shape[dimension]) {
            cnp_set_error(CNP_ERR_BROADCAST, function_name,
                          "values do not broadcast at axis %d", dimension);
            return false;
        }
    }
    return true;
}

static int64_t indexing_insert_value_offset(
    const CnpArray *values, int result_ndim, int axis,
    int leading_dimensions, bool move_first_axis,
    const int64_t *broadcast_shape, const int64_t *target_coords) {
    int64_t broadcast_coords[CNP_MAXDIMS];
    int64_t padded_coords[CNP_MAXDIMS];
    int64_t value_coords[CNP_MAXDIMS];
    for (int dimension = 0; dimension < result_ndim; ++dimension)
        broadcast_coords[dimension] = broadcast_shape[dimension] == 1
            ? 0 : target_coords[dimension];
    if (move_first_axis) {
        padded_coords[0] = broadcast_coords[axis];
        int padded_dimension = 1;
        for (int dimension = 0; dimension < result_ndim; ++dimension) {
            if (dimension == axis) continue;
            padded_coords[padded_dimension++] = broadcast_coords[dimension];
        }
    } else {
        for (int dimension = 0; dimension < result_ndim; ++dimension)
            padded_coords[dimension] = broadcast_coords[dimension];
    }
    for (int dimension = 0; dimension < values->ndim; ++dimension) {
        int padded_dimension = leading_dimensions + dimension;
        value_coords[dimension] = padded_dimension < 0
            ? 0 : padded_coords[padded_dimension];
    }
    return values->offset + cnp_multi_to_offset(
        values->ndim, value_coords, values->strides);
}

static CnpArray *indexing_insert_single(
    const CnpArray *arr, int64_t position, const CnpArray *values,
    int axis, bool axis_none, bool move_first_axis,
    const char *function_name) {
    if (!arr || !values) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "source and values must not be null");
        return NULL;
    }
    CnpAxisTraversal traversal;
    if (!indexing_axis_traversal(
            arr, axis, axis_none, function_name, &traversal)) return NULL;
    if (position < 0 || position > traversal.axis_size) {
        cnp_set_error(CNP_ERR_INDEX, function_name,
                      "index %lld is out of bounds for axis %d with size %lld",
                      (long long)position, axis_none ? 0 : traversal.axis,
                      (long long)traversal.axis_size);
        return NULL;
    }

    int result_ndim = axis_none ? 1 : arr->ndim;
    int64_t virtual_source_shape[CNP_MAXDIMS];
    if (axis_none) virtual_source_shape[0] = arr->size;
    else {
        for (int dimension = 0; dimension < arr->ndim; ++dimension)
            virtual_source_shape[dimension] = arr->shape[dimension];
    }
    CnpArray virtual_source = *arr;
    virtual_source.ndim = result_ndim;
    virtual_source.shape = virtual_source_shape;

    int64_t padded_shape[CNP_MAXDIMS];
    int64_t broadcast_shape[CNP_MAXDIMS];
    int leading_dimensions = 0;
    if (!indexing_insert_values_shape(
            &virtual_source, values, traversal.axis, move_first_axis,
            function_name, padded_shape, broadcast_shape,
            &leading_dimensions)) return NULL;
    int64_t inserted_count = broadcast_shape[traversal.axis];
    int64_t output_axis_size = traversal.axis_size + inserted_count;
    int64_t output_shape[CNP_MAXDIMS];
    for (int dimension = 0; dimension < result_ndim; ++dimension)
        output_shape[dimension] = virtual_source_shape[dimension];
    output_shape[traversal.axis] = output_axis_size;
    CnpArray *result = cnp_array_new(
        result_ndim, output_shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) return NULL;

    int itemsize = arr->dtype->elsize;
    int64_t target_coords[CNP_MAXDIMS] = {0};
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        int64_t remaining_outer = outer;
        for (int dimension = traversal.axis - 1;
             dimension >= 0; --dimension) {
            target_coords[dimension] =
                remaining_outer % virtual_source_shape[dimension];
            remaining_outer /= virtual_source_shape[dimension];
        }
        int64_t destination_axis = 0;
        for (int64_t source_axis = 0;
             source_axis < position; ++source_axis) {
            char *destination = (char*)result->data +
                (outer * output_axis_size + destination_axis) *
                traversal.inner * itemsize;
            indexing_copy_axis_block(
                &traversal, outer, source_axis, destination);
            ++destination_axis;
        }
        for (int64_t inserted_axis = 0;
             inserted_axis < inserted_count; ++inserted_axis) {
            target_coords[traversal.axis] = inserted_axis;
            for (int64_t inner = 0; inner < traversal.inner; ++inner) {
                int64_t remaining_inner = inner;
                for (int dimension = result_ndim - 1;
                     dimension > traversal.axis; --dimension) {
                    target_coords[dimension] =
                        remaining_inner % virtual_source_shape[dimension];
                    remaining_inner /= virtual_source_shape[dimension];
                }
                int64_t source_offset = indexing_insert_value_offset(
                    values, result_ndim, traversal.axis,
                    leading_dimensions, move_first_axis,
                    broadcast_shape, target_coords);
                int64_t destination_offset =
                    ((outer * output_axis_size + destination_axis) *
                     traversal.inner + inner) * itemsize;
                if (!indexing_copy_cast_element(
                        values, source_offset, result, destination_offset,
                        function_name)) {
                    cnp_array_free(result);
                    return NULL;
                }
            }
            ++destination_axis;
        }
        for (int64_t source_axis = position;
             source_axis < traversal.axis_size; ++source_axis) {
            char *destination = (char*)result->data +
                (outer * output_axis_size + destination_axis) *
                traversal.inner * itemsize;
            indexing_copy_axis_block(
                &traversal, outer, source_axis, destination);
            ++destination_axis;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_insert_v2(
    const CnpArray *arr, int64_t obj, const CnpArray *values,
    int axis, bool axis_none) {
    const char *function_name = "cnp_insert_v2";
    CnpAxisTraversal traversal;
    if (!indexing_axis_traversal(
            arr, axis, axis_none, function_name, &traversal)) return NULL;
    int64_t position = obj;
    if (position < 0) position += traversal.axis_size;
    if (position < 0 || position > traversal.axis_size) {
        cnp_set_error(CNP_ERR_INDEX, function_name,
                      "index %lld is out of bounds for axis %d with size %lld",
                      (long long)obj, axis_none ? 0 : traversal.axis,
                      (long long)traversal.axis_size);
        return NULL;
    }
    return indexing_insert_single(
        arr, position, values, axis, axis_none, true, function_name);
}

typedef struct {
    int64_t position;
    int64_t original;
} CnpInsertPosition;

static int indexing_compare_insert_positions(
    const void *left_pointer, const void *right_pointer) {
    const CnpInsertPosition *left =
        (const CnpInsertPosition*)left_pointer;
    const CnpInsertPosition *right =
        (const CnpInsertPosition*)right_pointer;
    if (left->position < right->position) return -1;
    if (left->position > right->position) return 1;
    if (left->original < right->original) return -1;
    if (left->original > right->original) return 1;
    return 0;
}

static CnpArray *indexing_insert_multiple(
    const CnpArray *arr, const CnpArray *obj, const CnpArray *values,
    int axis, bool axis_none, const char *function_name) {
    CnpAxisTraversal traversal;
    if (!indexing_axis_traversal(
            arr, axis, axis_none, function_name, &traversal)) return NULL;
    int result_ndim = axis_none ? 1 : arr->ndim;
    int64_t virtual_source_shape[CNP_MAXDIMS];
    if (axis_none) virtual_source_shape[0] = arr->size;
    else {
        for (int dimension = 0; dimension < arr->ndim; ++dimension)
            virtual_source_shape[dimension] = arr->shape[dimension];
    }
    CnpArray virtual_source = *arr;
    virtual_source.ndim = result_ndim;
    virtual_source.shape = virtual_source_shape;

    int64_t padded_shape[CNP_MAXDIMS];
    int64_t broadcast_shape[CNP_MAXDIMS];
    int leading_dimensions = 0;
    if (!indexing_insert_values_shape(
            &virtual_source, values, traversal.axis, false,
            function_name, padded_shape, broadcast_shape,
            &leading_dimensions)) return NULL;
    int64_t inserted_count = obj->size;
    if (broadcast_shape[traversal.axis] != 1 &&
        broadcast_shape[traversal.axis] != inserted_count) {
        cnp_set_error(
            CNP_ERR_BROADCAST, function_name,
            "values axis length %lld cannot broadcast to %lld inserted values",
            (long long)broadcast_shape[traversal.axis],
            (long long)inserted_count);
        return NULL;
    }
    if (inserted_count > INT64_MAX - traversal.axis_size) {
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "inserted axis length exceeds int64 capacity");
        return NULL;
    }
    int64_t output_axis_size = traversal.axis_size + inserted_count;
    if ((uint64_t)inserted_count > SIZE_MAX / sizeof(CnpInsertPosition) ||
        (uint64_t)output_axis_size > SIZE_MAX / sizeof(int64_t)) {
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "insertion metadata exceeds addressable memory");
        return NULL;
    }

    size_t position_bytes = (size_t)inserted_count *
        sizeof(CnpInsertPosition);
    CnpInsertPosition *positions = inserted_count > 0
        ? (CnpInsertPosition*)cnp_malloc(position_bytes) : NULL;
    size_t owner_bytes = output_axis_size > 0
        ? (size_t)output_axis_size * sizeof(int64_t) : sizeof(int64_t);
    int64_t *insertion_owner = (int64_t*)cnp_malloc(owner_bytes);
    if ((inserted_count > 0 && !positions) || !insertion_owner) {
        if (positions) cnp_free(positions, position_bytes);
        if (insertion_owner) cnp_free(insertion_owner, owner_bytes);
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "failed to allocate insertion metadata");
        return NULL;
    }
    for (int64_t item = 0; item < inserted_count; ++item) {
        int64_t position;
        if (!indexing_read_normalized_index(
                obj, item, traversal.axis_size, true, function_name,
                axis_none ? 0 : traversal.axis, &position)) {
            cnp_free(positions, position_bytes);
            cnp_free(insertion_owner, owner_bytes);
            return NULL;
        }
        positions[item].position = position;
        positions[item].original = item;
    }
    if (inserted_count > 1) {
        qsort(positions, (size_t)inserted_count,
              sizeof(CnpInsertPosition), indexing_compare_insert_positions);
    }
    for (int64_t item = 0; item < output_axis_size; ++item)
        insertion_owner[item] = -1;
    for (int64_t rank = 0; rank < inserted_count; ++rank) {
        int64_t adjusted = positions[rank].position + rank;
        insertion_owner[adjusted] = positions[rank].original;
    }
    if (positions) cnp_free(positions, position_bytes);

    int64_t output_shape[CNP_MAXDIMS];
    for (int dimension = 0; dimension < result_ndim; ++dimension)
        output_shape[dimension] = virtual_source_shape[dimension];
    output_shape[traversal.axis] = output_axis_size;
    CnpArray *result = cnp_array_new(
        result_ndim, output_shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_free(insertion_owner, owner_bytes);
        return NULL;
    }

    int itemsize = arr->dtype->elsize;
    int64_t target_coords[CNP_MAXDIMS] = {0};
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        int64_t remaining_outer = outer;
        for (int dimension = traversal.axis - 1;
             dimension >= 0; --dimension) {
            target_coords[dimension] =
                remaining_outer % virtual_source_shape[dimension];
            remaining_outer /= virtual_source_shape[dimension];
        }
        int64_t source_axis = 0;
        for (int64_t destination_axis = 0;
             destination_axis < output_axis_size; ++destination_axis) {
            int64_t insertion = insertion_owner[destination_axis];
            if (insertion < 0) {
                char *destination = (char*)result->data +
                    (outer * output_axis_size + destination_axis) *
                    traversal.inner * itemsize;
                indexing_copy_axis_block(
                    &traversal, outer, source_axis, destination);
                ++source_axis;
                continue;
            }
            target_coords[traversal.axis] = insertion;
            for (int64_t inner = 0; inner < traversal.inner; ++inner) {
                int64_t remaining_inner = inner;
                for (int dimension = result_ndim - 1;
                     dimension > traversal.axis; --dimension) {
                    target_coords[dimension] =
                        remaining_inner % virtual_source_shape[dimension];
                    remaining_inner /= virtual_source_shape[dimension];
                }
                int64_t source_offset = indexing_insert_value_offset(
                    values, result_ndim, traversal.axis,
                    leading_dimensions, false,
                    broadcast_shape, target_coords);
                int64_t destination_offset =
                    ((outer * output_axis_size + destination_axis) *
                     traversal.inner + inner) * itemsize;
                if (!indexing_copy_cast_element(
                        values, source_offset, result, destination_offset,
                        function_name)) {
                    cnp_array_free(result);
                    cnp_free(insertion_owner, owner_bytes);
                    return NULL;
                }
            }
        }
    }
    cnp_free(insertion_owner, owner_bytes);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_insert_array_v2(
    const CnpArray *arr, const CnpArray *obj, const CnpArray *values,
    int axis, bool axis_none) {
    const char *function_name = "cnp_insert_array_v2";
    if (!arr || !obj || !values) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "source, obj, and values must not be null");
        return NULL;
    }
    if (obj->dtype->type_num != CNP_BOOL &&
        !indexing_require_integer(obj, function_name, "obj"))
        return NULL;
    if (obj->ndim > 1) {
        cnp_set_error(CNP_ERR_SHAPE, function_name,
                      "index array argument obj must be one-dimensional or scalar");
        return NULL;
    }
    CnpAxisTraversal traversal;
    if (!indexing_axis_traversal(
            arr, axis, axis_none, function_name, &traversal)) return NULL;
    if (obj->ndim == 0 || obj->size == 1) {
        int64_t position;
        if (!indexing_read_normalized_index(
                obj, 0, traversal.axis_size, true, function_name,
                axis_none ? 0 : traversal.axis, &position)) return NULL;
        return indexing_insert_single(
            arr, position, values, axis, axis_none,
            obj->ndim == 0, function_name);
    }
    return indexing_insert_multiple(
        arr, obj, values, axis, axis_none, function_name);
}

/* =========================================================================
 * Where (ternary selection)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_array_where(const CnpArray *condition, const CnpArray *x, const CnpArray *y) {
    const char *function_name = "cnp_array_where";
    if (!condition) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "condition array is required");
        return NULL;
    }

    /* If x and y are NULL, return nonzero indices */
    if (!x && !y) {
        CnpArray *result = cnp_array_nonzero(condition);
        if (!result) cnp_relabel_error(function_name);
        return result;
    }

    if (!x || !y) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "x and y must either both be provided or both be omitted");
        return NULL;
    }
    CnpArray *result = cnp_where(condition, x, y);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * Nonzero
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_array_nonzero(const CnpArray *arr) {
    const char *function_name = "cnp_array_nonzero";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return NULL;
    }

    /* Count nonzero elements */
    int64_t count = 0;
    int64_t coords[CNP_MAXDIMS] = {0};
    for (int64_t i = 0; i < arr->size; i++) {
        if (indexing_element_truth(arr, i)) count++;
        for (int d = arr->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < arr->shape[d]) break;
            coords[d] = 0;
        }
    }

    /* Create result: tuple of arrays (ndim x count) */
    int ndim = arr->ndim > 0 ? arr->ndim : 1;
    int64_t shape[2] = {ndim, count};
    CnpArray *result = cnp_array_new(2, shape, CNP_LONGLONG, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t idx = 0;
    memset(coords, 0, sizeof(coords));
    for (int64_t i = 0; i < arr->size; i++) {
        if (indexing_element_truth(arr, i)) {
            for (int d = 0; d < ndim; d++) {
                *((int64_t*)result->data + d * count + idx) = coords[d];
            }
            idx++;
        }
        for (int d = arr->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < arr->shape[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}

/* =========================================================================
 * Boolean indexing
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_array_boolean_index(CnpArray *arr, const CnpArray *mask) {
    const char *function_name = "cnp_array_boolean_index";
    if (!arr || !mask) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array and mask are required");
        return NULL;
    }
    if (mask->dtype->type_num != CNP_BOOL) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name, "mask dtype must be bool");
        return NULL;
    }
    if (arr->ndim != mask->ndim) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "mask and source must have the same shape");
        return NULL;
    }
    for (int dimension = 0; dimension < arr->ndim; ++dimension) {
        if (arr->shape[dimension] != mask->shape[dimension]) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "mask and source must have the same shape");
            return NULL;
        }
    }

    /* Count true elements */
    int64_t count = 0;
    for (int64_t i = 0; i < mask->size; i++) {
        if (indexing_element_truth(mask, i)) count++;
    }

    int64_t shape[1] = {count};
    CnpArray *result = cnp_array_new(1, shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int elsize = result->dtype->elsize;
    int64_t out_idx = 0;
    for (int64_t i = 0; i < arr->size; i++) {
        if (indexing_element_truth(mask, i)) {
            int64_t src_off = indexing_flat_offset(arr, i);
            memcpy(
                (char*)result->data + out_idx * elsize,
                (const char*)arr->data + src_off, (size_t)elsize);
            out_idx++;
        }
    }
    return result;
}

/* =========================================================================
 * Fancy indexing
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_array_fancy_index(CnpArray *arr, const CnpArray *indices, int axis) {
    const char *function_name = "cnp_array_fancy_index";
    CnpArray *result = cnp_array_take(arr, indices, axis);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * Count nonzero
 * ========================================================================= */
CNP_API int64_t CNP_CALL cnp_count_nonzero(const CnpArray *arr, int axis) {
    const char *function_name = "cnp_count_nonzero";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return 0;
    }
    if (axis != CNP_AXIS_NONE) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "legacy scalar result supports only axis=None");
        return 0;
    }
    int64_t count = 0;
    for (int64_t i = 0; i < arr->size; i++) {
        if (indexing_element_truth(arr, i)) count++;
    }
    return count;
}

CNP_API CnpArray* CNP_CALL cnp_count_nonzero_v2(
        const CnpArray *arr, int axis,
        bool axis_none, bool keepdims) {
    const char *function_name = "cnp_count_nonzero_v2";
    int resolved_axis;
    if (!cnp_reduction_resolve_axis(
            arr, axis, axis_none, function_name, &resolved_axis))
        return NULL;

    CnpReductionTraversal traversal;
    cnp_reduction_traversal_init(
        arr, resolved_axis, &traversal);
    int output_ndim;
    int64_t output_shape[CNP_MAXDIMS];
    if (axis_none) {
        output_ndim = keepdims ? arr->ndim : 0;
        for (int dimension = 0;
             dimension < output_ndim; ++dimension)
            output_shape[dimension] = 1;
    } else if (arr->ndim == 0) {
        output_ndim = 0;
    } else if (keepdims) {
        output_ndim = arr->ndim;
        memcpy(output_shape, arr->shape,
               sizeof(int64_t) * arr->ndim);
        output_shape[resolved_axis] = 1;
    } else {
        output_ndim = traversal.result_ndim;
        memcpy(output_shape, traversal.result_shape,
               sizeof(int64_t) * traversal.result_ndim);
    }

    CnpArray *result = cnp_array_new(
        output_ndim, output_ndim ? output_shape : NULL,
        CNP_LONGLONG, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t *counts = (int64_t*)result->data;
    for (int64_t outer = 0;
         outer < traversal.outer; ++outer) {
        for (int64_t inner = 0;
             inner < traversal.inner; ++inner) {
            int64_t count = 0;
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item) {
                int64_t source_offset = cnp_reduction_source_offset(
                    &traversal, outer, inner, item);
                if (indexing_element_truth_at(arr, source_offset))
                    ++count;
            }
            counts[outer * traversal.inner + inner] = count;
        }
    }
    return result;
}

/* =========================================================================
 * Argwhere
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_argwhere(const CnpArray *arr) {
    const char *function_name = "cnp_argwhere";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return NULL;
    }

    int64_t count = 0;
    int64_t coords[CNP_MAXDIMS] = {0};
    for (int64_t i = 0; i < arr->size; i++) {
        if (indexing_element_truth(arr, i)) count++;
        for (int d = arr->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < arr->shape[d]) break;
            coords[d] = 0;
        }
    }

    int ndim = arr->ndim;
    int64_t shape[2] = {count, ndim};
    CnpArray *result = cnp_array_new(2, shape, CNP_LONGLONG, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t idx = 0;
    memset(coords, 0, sizeof(coords));
    for (int64_t i = 0; i < arr->size; i++) {
        if (indexing_element_truth(arr, i)) {
            for (int d = 0; d < ndim; d++) {
                *((int64_t*)result->data + idx * ndim + d) = coords[d];
            }
            idx++;
        }
        for (int d = arr->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < arr->shape[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}

/* =========================================================================
 * Flatnonzero
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_flatnonzero(const CnpArray *arr) {
    const char *function_name = "cnp_flatnonzero";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return NULL;
    }

    int64_t count = 0;
    for (int64_t i = 0; i < arr->size; i++) {
        if (indexing_element_truth(arr, i)) count++;
    }

    int64_t shape[1] = {count};
    CnpArray *result = cnp_array_new(1, shape, CNP_LONGLONG, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t idx = 0;
    for (int64_t i = 0; i < arr->size; i++) {
        if (indexing_element_truth(arr, i)) {
            *((int64_t*)result->data + idx) = i;
            idx++;
        }
    }
    return result;
}
