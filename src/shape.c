/**
 * cnumpy shape manipulation - reshape, transpose, concatenate, split, etc.
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <limits.h>

static bool shape_require_array(
    const CnpArray *array, const char *function_name) {
    if (array) return true;
    cnp_set_error(
        CNP_ERR_GENERIC, function_name, "source array is required");
    return false;
}

static bool shape_require_array_list(
    int narrays, CnpArray *const *arrays, const char *function_name) {
    if (narrays <= 0 || !arrays) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "at least one input array is required");
        return false;
    }
    for (int index = 0; index < narrays; ++index) {
        if (!arrays[index]) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "input array %d is null", index);
            return false;
        }
    }
    return true;
}

static bool shape_normalize_axis_checked(
    int axis, int ndim, const char *function_name, int *normalized_axis) {
    int resolved = axis;
    if (resolved < 0) resolved += ndim;
    if (resolved < 0 || resolved >= ndim) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "axis %d is out of bounds for array of dimension %d",
            axis, ndim);
        return false;
    }
    *normalized_axis = resolved;
    return true;
}

/* =========================================================================
 * Reshape and related
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_reshape(const CnpArray *arr, int ndim, const int64_t *newshape, CNP_ORDER order) {
    const char *function_name = "cnp_reshape";
    if (!shape_require_array(arr, function_name)) return NULL;
    if (ndim < 0 || ndim > CNP_MAXDIMS || (ndim > 0 && !newshape)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name, "invalid target rank %d", ndim);
        return NULL;
    }

    /* Handle -1 in shape */
    int64_t shape[CNP_MAXDIMS];
    int neg_idx = -1;
    int64_t known_size = 1;
    for (int i = 0; i < ndim; i++) {
        shape[i] = newshape[i];
        if (newshape[i] == -1) {
            if (neg_idx >= 0) {
                cnp_set_error(CNP_ERR_SHAPE, "cnp_reshape", "Can only specify one unknown dimension");
                return NULL;
            }
            neg_idx = i;
        } else {
            known_size *= newshape[i];
        }
    }
    if (neg_idx >= 0) {
        if (known_size == 0) {
            cnp_set_error(CNP_ERR_SHAPE, "cnp_reshape", "Cannot reshape array of size %lld into shape with 0",
                         (long long)arr->size);
            return NULL;
        }
        shape[neg_idx] = arr->size / known_size;
    }

    int64_t new_size = cnp_compute_size(ndim, shape);
    if (new_size != arr->size) {
        cnp_set_error(CNP_ERR_SHAPE, "cnp_reshape",
                     "Cannot reshape array of size %lld into shape (%lld elements)",
                     (long long)arr->size, (long long)new_size);
        return NULL;
    }

    /* If contiguous, create a view */
    if ((order == CNP_ORDER_C && (arr->flags & CNP_ARRAY_C_CONTIGUOUS)) ||
        (order == CNP_ORDER_F && (arr->flags & CNP_ARRAY_F_CONTIGUOUS))) {
        return cnp_array_reshape_view((CnpArray*)arr, ndim, shape, order);
    }

    /* Non-contiguous: copy data */
    CnpArray *result = cnp_array_new(ndim, shape, arr->dtype->type_num, order);
    if (!result) return NULL;

    int elsize = arr->dtype->elsize;
    int64_t coords[CNP_MAXDIMS] = {0};
    for (int64_t i = 0; i < arr->size; i++) {
        int64_t src_offset = arr->offset + cnp_multi_to_offset(arr->ndim, coords, arr->strides);
        memcpy((char*)result->data + i * elsize, (char*)arr->data + src_offset, elsize);
        for (int d = arr->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < arr->shape[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_ravel(const CnpArray *arr, CNP_ORDER order) {
    if (!shape_require_array(arr, "cnp_ravel")) return NULL;
    int64_t shape[1] = {arr->size};
    CnpArray *result = cnp_reshape(arr, 1, shape, order);
    if (!result) cnp_relabel_error("cnp_ravel");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_flatten(const CnpArray *arr, CNP_ORDER order) {
    if (!shape_require_array(arr, "cnp_flatten")) return NULL;
    /* flatten always returns a copy */
    int64_t shape[1] = {arr->size};
    CnpArray *result = cnp_array_new(1, shape, arr->dtype->type_num, order);
    if (!result) return NULL;

    int elsize = arr->dtype->elsize;
    bool c_traversal = order == CNP_ORDER_C || arr->ndim <= 1;
    if (c_traversal && (arr->flags & CNP_ARRAY_C_CONTIGUOUS)) {
        size_t bytes = (size_t)arr->size * (size_t)elsize;
        memcpy(result->data, (const char*)arr->data + arr->offset, bytes);
        return result;
    }

    int64_t coords[CNP_MAXDIMS] = {0};
    for (int64_t i = 0; i < arr->size; i++) {
        int64_t src_offset = arr->offset + cnp_multi_to_offset(arr->ndim, coords, arr->strides);
        memcpy((char*)result->data + i * elsize, (char*)arr->data + src_offset, elsize);
        for (int d = arr->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < arr->shape[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}

/* =========================================================================
 * Transpose and axis manipulation
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_transpose(const CnpArray *arr, const int *axes) {
    if (!arr) return NULL;
    int ndim = arr->ndim;
    int64_t shape[CNP_MAXDIMS];
    int64_t strides[CNP_MAXDIMS];

    if (axes == NULL) {
        /* Reverse all axes */
        for (int i = 0; i < ndim; i++) {
            shape[i] = arr->shape[ndim - 1 - i];
            strides[i] = arr->strides[ndim - 1 - i];
        }
    } else {
        bool seen_axes[CNP_MAXDIMS] = {false};
        for (int i = 0; i < ndim; i++) {
            int ax = axes[i];
            if (ax < 0) ax += ndim;
            if (ax < 0 || ax >= ndim) {
                cnp_set_error(CNP_ERR_AXIS, "cnp_transpose", "Invalid axis: %d", axes[i]);
                return NULL;
            }
            if (seen_axes[ax]) {
                cnp_set_error(CNP_ERR_AXIS, "cnp_transpose",
                              "Duplicate axis: %d", axes[i]);
                return NULL;
            }
            seen_axes[ax] = true;
            shape[i] = arr->shape[ax];
            strides[i] = arr->strides[ax];
        }
    }

    return cnp_array_view_from_metadata(
        (CnpArray*)arr, ndim, shape, strides, arr->offset, 0);
}

CNP_API CnpArray* CNP_CALL cnp_swapaxes(const CnpArray *arr, int axis1, int axis2) {
    if (!shape_require_array(arr, "cnp_swapaxes")) return NULL;
    int ndim = arr->ndim;
    if (!shape_normalize_axis_checked(
            axis1, ndim, "cnp_swapaxes", &axis1) ||
        !shape_normalize_axis_checked(
            axis2, ndim, "cnp_swapaxes", &axis2)) return NULL;

    int axes[CNP_MAXDIMS];
    for (int i = 0; i < ndim; i++) axes[i] = i;
    axes[axis1] = axis2;
    axes[axis2] = axis1;
    return cnp_transpose(arr, axes);
}

CNP_API CnpArray* CNP_CALL cnp_moveaxis(const CnpArray *arr, int src, int dst) {
    if (!shape_require_array(arr, "cnp_moveaxis")) return NULL;
    int ndim = arr->ndim;
    if (!shape_normalize_axis_checked(
            src, ndim, "cnp_moveaxis", &src) ||
        !shape_normalize_axis_checked(
            dst, ndim, "cnp_moveaxis", &dst)) return NULL;

    int axes[CNP_MAXDIMS];
    int idx = 0;
    for (int i = 0; i < ndim; i++)
        if (i != src) axes[idx++] = i;
    for (int i = ndim - 1; i > dst; --i) axes[i] = axes[i - 1];
    axes[dst] = src;
    return cnp_transpose(arr, axes);
}

CNP_API CnpArray* CNP_CALL cnp_squeeze(const CnpArray *arr, int axis) {
    if (!shape_require_array(arr, "cnp_squeeze")) return NULL;

    if (axis >= 0) {
        if (!shape_normalize_axis_checked(
                axis, arr->ndim, "cnp_squeeze", &axis)) return NULL;
        if (arr->shape[axis] != 1) {
            cnp_set_error(CNP_ERR_SHAPE, "cnp_squeeze", "Cannot squeeze axis %d with size %lld",
                         axis, (long long)arr->shape[axis]);
            return NULL;
        }
        int new_ndim = arr->ndim - 1;
        int64_t new_shape[CNP_MAXDIMS];
        int j = 0;
        for (int i = 0; i < arr->ndim; i++) {
            if (i != axis) new_shape[j++] = arr->shape[i];
        }
        return cnp_reshape(arr, new_ndim, new_shape, CNP_ORDER_C);
    }

    /* Squeeze all dimensions of size 1 */
    int new_ndim = 0;
    int64_t new_shape[CNP_MAXDIMS];
    for (int i = 0; i < arr->ndim; i++) {
        if (arr->shape[i] != 1) new_shape[new_ndim++] = arr->shape[i];
    }
    if (new_ndim == 0) {
        int64_t s[1] = {1};
        return cnp_reshape(arr, 0, s, CNP_ORDER_C);
    }
    return cnp_reshape(arr, new_ndim, new_shape, CNP_ORDER_C);
}

CNP_API CnpArray* CNP_CALL cnp_expand_dims(const CnpArray *arr, int axis) {
    if (!shape_require_array(arr, "cnp_expand_dims")) return NULL;
    int new_ndim = arr->ndim + 1;
    if (new_ndim > CNP_MAXDIMS ||
        !shape_normalize_axis_checked(
            axis, new_ndim, "cnp_expand_dims", &axis)) return NULL;

    int64_t new_shape[CNP_MAXDIMS];
    int j = 0;
    for (int i = 0; i < new_ndim; i++) {
        if (i == axis) new_shape[i] = 1;
        else new_shape[i] = arr->shape[j++];
    }
    return cnp_reshape(arr, new_ndim, new_shape, CNP_ORDER_C);
}

CNP_API CnpArray* CNP_CALL cnp_broadcast_to(const CnpArray *arr, int ndim, const int64_t *shape) {
    if (!shape_require_array(arr, "cnp_broadcast_to")) return NULL;
    if (ndim < arr->ndim || ndim > CNP_MAXDIMS ||
        (ndim > 0 && !shape)) {
        cnp_set_error(CNP_ERR_BROADCAST, "cnp_broadcast_to",
                      "Invalid broadcast rank: %d", ndim);
        return NULL;
    }

    int offset = ndim - arr->ndim;
    int64_t new_shape[CNP_MAXDIMS];
    int64_t new_strides[CNP_MAXDIMS];
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) {
            cnp_set_error(CNP_ERR_SHAPE, "cnp_broadcast_to",
                          "Negative dimension at axis %d", i);
            return NULL;
        }
        new_shape[i] = shape[i];
        new_strides[i] = 0;
    }
    for (int i = 0; i < arr->ndim; i++) {
        int target_axis = offset + i;
        if (arr->shape[i] != 1 &&
            arr->shape[i] != shape[target_axis]) {
            cnp_set_error(CNP_ERR_BROADCAST, "cnp_broadcast_to",
                          "Cannot broadcast axis %d", i);
            return NULL;
        }
        new_strides[target_axis] =
            arr->shape[i] == 1 ? 0 : arr->strides[i];
    }
    return cnp_array_view_from_metadata(
        (CnpArray*)arr, ndim, new_shape, new_strides,
        arr->offset, 0);
}

static void cnp_release_array_results(CnpArray **results, int count) {
    if (!results) return;
    for (int index = 0; index < count; ++index) {
        if (results[index]) {
            cnp_array_free(results[index]);
            results[index] = NULL;
        }
    }
}

CNP_API CNP_STATUS CNP_CALL cnp_broadcast_arrays_v2(
    int narrays, CnpArray *const *arrays,
    CnpArray **results, int result_capacity) {
    const char *function_name = "cnp_broadcast_arrays_v2";
    if (narrays < 0 || result_capacity < 0) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "array count and result capacity must be non-negative");
        return CNP_ERR_GENERIC;
    }
    if (results) {
        for (int index = 0; index < result_capacity; ++index)
            results[index] = NULL;
    }
    if (result_capacity < narrays) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "result capacity %d is smaller than array count %d",
                      result_capacity, narrays);
        return CNP_ERR_GENERIC;
    }
    if ((narrays > 0 && !arrays) || (result_capacity > 0 && !results)) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "input and result arrays must not be null");
        return CNP_ERR_GENERIC;
    }
    if (narrays == 0) return CNP_OK;

    int output_ndim = 0;
    for (int array_index = 0; array_index < narrays; ++array_index) {
        if (!arrays[array_index]) {
            cnp_set_error(CNP_ERR_GENERIC, function_name,
                          "input array %d is null", array_index);
            return CNP_ERR_GENERIC;
        }
        if (arrays[array_index]->ndim > output_ndim)
            output_ndim = arrays[array_index]->ndim;
    }

    int64_t output_shape[CNP_MAXDIMS];
    for (int axis = 0; axis < output_ndim; ++axis)
        output_shape[axis] = 1;

    for (int array_index = 0; array_index < narrays; ++array_index) {
        const CnpArray *array = arrays[array_index];
        int leading = output_ndim - array->ndim;
        for (int source_axis = 0; source_axis < array->ndim; ++source_axis) {
            int output_axis = leading + source_axis;
            int64_t dimension = array->shape[source_axis];
            int64_t current = output_shape[output_axis];
            if (dimension < 0) {
                cnp_set_error(CNP_ERR_SHAPE, function_name,
                              "input %d has negative dimension at axis %d",
                              array_index, source_axis);
                return CNP_ERR_SHAPE;
            }
            if (current == 1) {
                output_shape[output_axis] = dimension;
            } else if (dimension != 1 && dimension != current) {
                cnp_set_error(
                    CNP_ERR_BROADCAST, function_name,
                    "input %d axis %d has length %lld; output axis %d has length %lld",
                    array_index, source_axis, (long long)dimension,
                    output_axis, (long long)current);
                return CNP_ERR_BROADCAST;
            }
        }
    }

    for (int array_index = 0; array_index < narrays; ++array_index) {
        results[array_index] = cnp_broadcast_to(
            arrays[array_index], output_ndim, output_shape);
        if (!results[array_index]) {
            cnp_release_array_results(results, narrays);
            return cnp_get_error(NULL);
        }
    }
    return CNP_OK;
}

CNP_API CnpArray* CNP_CALL cnp_broadcast_arrays(int narrays, CnpArray **arrays) {
    /* Returns first array broadcast (simplified - full version returns all) */
    if (!shape_require_array_list(
            narrays, arrays, "cnp_broadcast_arrays")) return NULL;
    /* TODO: return array of broadcast arrays */
    return cnp_array_copy(arrays[0]);
}

/* =========================================================================
 * Joining arrays
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_concatenate(int narrays, CnpArray **arrays, int axis) {
    const char *function_name = "cnp_concatenate";
    if (!shape_require_array_list(narrays, arrays, function_name)) return NULL;

    CnpArray *first = arrays[0];
    if (!shape_normalize_axis_checked(
            axis, first->ndim, function_name, &axis)) return NULL;

    /* Compute output shape */
    int ndim = first->ndim;
    int64_t out_shape[CNP_MAXDIMS];
    for (int i = 0; i < ndim; i++) out_shape[i] = first->shape[i];

    int64_t total_axis_size = first->shape[axis];
    for (int n = 1; n < narrays; n++) {
        if (arrays[n]->ndim != ndim) {
            cnp_set_error(CNP_ERR_SHAPE, "cnp_concatenate", "All arrays must have same ndim");
            return NULL;
        }
        for (int i = 0; i < ndim; i++) {
            if (i != axis && arrays[n]->shape[i] != out_shape[i]) {
                cnp_set_error(CNP_ERR_SHAPE, "cnp_concatenate", "Shape mismatch on axis %d", i);
                return NULL;
            }
        }
        total_axis_size += arrays[n]->shape[axis];
    }
    out_shape[axis] = total_axis_size;

    CNP_TYPE result_type = first->dtype->type_num;
    for (int n = 1; n < narrays; ++n) {
        result_type = cnp_promote_type_full(
            result_type, arrays[n]->dtype->type_num);
        if (result_type == CNP_NOTYPE) {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "input dtypes do not have a common NumPy dtype");
            return NULL;
        }
    }

    CnpArray *result = cnp_array_new(
        ndim, out_shape, result_type, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    bool can_copy_blocks =
        (first->flags & CNP_ARRAY_C_CONTIGUOUS) != 0 &&
        first->dtype->type_num == result_type;
    for (int n = 1; n < narrays && can_copy_blocks; n++) {
        can_copy_blocks =
            (arrays[n]->flags & CNP_ARRAY_C_CONTIGUOUS) != 0 &&
            arrays[n]->dtype->type_num == result_type;
    }

    if (can_copy_blocks) {
        int64_t outer = 1;
        for (int d = 0; d < axis; d++) outer *= out_shape[d];

        size_t inner_bytes = (size_t)result->dtype->elsize;
        for (int d = axis + 1; d < ndim; d++)
            inner_bytes *= (size_t)out_shape[d];

        char *dst = (char*)result->data;
        for (int64_t outer_index = 0; outer_index < outer; outer_index++) {
            for (int n = 0; n < narrays; n++) {
                const CnpArray *src = arrays[n];
                size_t src_block_bytes =
                    (size_t)src->shape[axis] * inner_bytes;
                const char *src_block =
                    (const char*)src->data + src->offset +
                    (size_t)outer_index * src_block_bytes;
                memcpy(dst, src_block, src_block_bytes);
                dst += src_block_bytes;
            }
        }
        return result;
    }

    /* Copy data */
    int64_t axis_offset = 0;
    int elsize = result->dtype->elsize;
    for (int n = 0; n < narrays; n++) {
        CnpArray *src = arrays[n];
        int64_t coords[CNP_MAXDIMS] = {0};
        for (int64_t i = 0; i < src->size; i++) {
            int64_t src_off = src->offset + cnp_multi_to_offset(src->ndim, coords, src->strides);
            /* Compute destination coordinates */
            int64_t dst_coords[CNP_MAXDIMS];
            for (int d = 0; d < ndim; d++) dst_coords[d] = coords[d];
            dst_coords[axis] += axis_offset;
            int64_t dst_off = cnp_multi_to_offset(ndim, dst_coords, result->strides);
            CNP_STATUS cast_status = cnp_cast_scalar_value(
                (char*)src->data + src_off, src->dtype->type_num,
                (char*)result->data + dst_off, result_type,
                function_name);
            if (cast_status != CNP_OK) {
                cnp_array_free(result);
                return NULL;
            }

            for (int d = src->ndim - 1; d >= 0; d--) {
                coords[d]++;
                if (coords[d] < src->shape[d]) break;
                coords[d] = 0;
            }
        }
        axis_offset += src->shape[axis];
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_stack(int narrays, CnpArray **arrays, int axis) {
    if (!shape_require_array_list(narrays, arrays, "cnp_stack")) return NULL;
    int normalized_axis = axis;
    if (!shape_normalize_axis_checked(
            axis, arrays[0]->ndim + 1,
            "cnp_stack", &normalized_axis)) return NULL;
    axis = normalized_axis;

    /* Expand dims on each array then concatenate */
    CnpArray *expanded[CNP_MAXARGS];
    for (int i = 0; i < narrays; i++) {
        expanded[i] = cnp_expand_dims(arrays[i], axis);
        if (!expanded[i]) {
            for (int j = 0; j < i; j++) cnp_array_free(expanded[j]);
            return NULL;
        }
    }

    CnpArray *result = cnp_concatenate(narrays, expanded, axis);
    for (int i = 0; i < narrays; i++) cnp_array_free(expanded[i]);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_vstack(int narrays, CnpArray **arrays) {
    if (!shape_require_array_list(narrays, arrays, "cnp_vstack")) return NULL;
    /* Ensure at least 2D */
    CnpArray *reshaped[CNP_MAXARGS];
    for (int i = 0; i < narrays; i++) {
        if (arrays[i]->ndim == 1) {
            int64_t shape[2] = {1, arrays[i]->shape[0]};
            reshaped[i] = cnp_reshape(arrays[i], 2, shape, CNP_ORDER_C);
        } else {
            reshaped[i] = arrays[i];
            cnp_array_incref(reshaped[i]);
        }
    }
    CnpArray *result = cnp_concatenate(narrays, reshaped, 0);
    for (int i = 0; i < narrays; i++) {
        if (reshaped[i] != arrays[i]) cnp_array_free(reshaped[i]);
        else cnp_array_decref(reshaped[i]);
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_hstack(int narrays, CnpArray **arrays) {
    if (!shape_require_array_list(narrays, arrays, "cnp_hstack")) return NULL;
    if (arrays[0]->ndim == 1) {
        return cnp_concatenate(narrays, arrays, 0);
    }
    return cnp_concatenate(narrays, arrays, 1);
}

CNP_API CnpArray* CNP_CALL cnp_dstack(int narrays, CnpArray **arrays) {
    if (!shape_require_array_list(narrays, arrays, "cnp_dstack")) return NULL;
    CnpArray *reshaped[CNP_MAXARGS];
    for (int i = 0; i < narrays; i++) {
        CnpArray *a = arrays[i];
        if (a->ndim == 1) {
            int64_t shape[3] = {1, a->shape[0], 1};
            reshaped[i] = cnp_reshape(a, 3, shape, CNP_ORDER_C);
        } else if (a->ndim == 2) {
            int64_t shape[3] = {a->shape[0], a->shape[1], 1};
            reshaped[i] = cnp_reshape(a, 3, shape, CNP_ORDER_C);
        } else {
            reshaped[i] = a;
            cnp_array_incref(reshaped[i]);
        }
    }
    CnpArray *result = cnp_concatenate(narrays, reshaped, 2);
    for (int i = 0; i < narrays; i++) {
        if (reshaped[i] != arrays[i]) cnp_array_free(reshaped[i]);
        else cnp_array_decref(reshaped[i]);
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_column_stack(int narrays, CnpArray **arrays) {
    if (!shape_require_array_list(
            narrays, arrays, "cnp_column_stack")) return NULL;
    CnpArray *reshaped[CNP_MAXARGS];
    for (int i = 0; i < narrays; i++) {
        if (arrays[i]->ndim == 1) {
            reshaped[i] = cnp_expand_dims(arrays[i], 1);
        } else {
            reshaped[i] = arrays[i];
            cnp_array_incref(reshaped[i]);
        }
    }
    CnpArray *result = cnp_concatenate(narrays, reshaped, 1);
    for (int i = 0; i < narrays; i++) {
        if (reshaped[i] != arrays[i]) cnp_array_free(reshaped[i]);
        else cnp_array_decref(reshaped[i]);
    }
    return result;
}

/* =========================================================================
 * Splitting arrays
 * ========================================================================= */
static void split_clear_results(CnpArray **results, int result_capacity) {
    if (!results || result_capacity <= 0) return;
    for (int index = 0; index < result_capacity; ++index)
        results[index] = NULL;
}

static CNP_STATUS split_validate(
    const char *function_name, const CnpArray *arr, int axis,
    CnpArray **results, int result_capacity, int required_results,
    int *normalized_axis) {
    split_clear_results(results, result_capacity);
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "source array must not be null");
        return CNP_ERR_GENERIC;
    }
    if (result_capacity < 0) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "result capacity must be non-negative");
        return CNP_ERR_GENERIC;
    }
    if (result_capacity < required_results) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "result capacity %d is smaller than required count %d",
                      result_capacity, required_results);
        return CNP_ERR_GENERIC;
    }
    if (required_results > 0 && !results) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "result array must not be null");
        return CNP_ERR_GENERIC;
    }
    int resolved_axis = axis;
    if (resolved_axis < 0) resolved_axis += arr->ndim;
    if (resolved_axis < 0 || resolved_axis >= arr->ndim) {
        cnp_set_error(CNP_ERR_AXIS, function_name,
                      "axis %d is out of bounds for array of dimension %d",
                      axis, arr->ndim);
        return CNP_ERR_AXIS;
    }
    *normalized_axis = resolved_axis;
    return CNP_OK;
}

static CnpArray *split_create_view(
    const CnpArray *arr, int axis, int64_t start, int64_t stop) {
    CnpSlice slices[CNP_MAXDIMS];
    for (int dimension = 0; dimension < arr->ndim; ++dimension) {
        slices[dimension].start = 0;
        slices[dimension].stop = 0;
        slices[dimension].step = 1;
        slices[dimension].has_start = false;
        slices[dimension].has_stop = false;
        slices[dimension].has_step = false;
    }
    slices[axis].start = start;
    slices[axis].stop = stop;
    slices[axis].step = 1;
    slices[axis].has_start = true;
    slices[axis].has_stop = true;
    slices[axis].has_step = true;
    return cnp_array_slice((CnpArray*)arr, arr->ndim, slices);
}

static void split_release_partial(CnpArray **results, int created) {
    for (int index = 0; index < created; ++index) {
        if (results[index]) cnp_array_decref(results[index]);
        results[index] = NULL;
    }
}

static CNP_STATUS split_sections_impl(
    const char *function_name, const CnpArray *arr, int sections, int axis,
    bool allow_unequal, CnpArray **results, int result_capacity) {
    split_clear_results(results, result_capacity);
    if (sections <= 0) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "number of sections must be greater than zero");
        return CNP_ERR_GENERIC;
    }

    int normalized_axis = 0;
    CNP_STATUS status = split_validate(
        function_name, arr, axis, results, result_capacity, sections,
        &normalized_axis);
    if (status != CNP_OK) return status;

    int64_t axis_size = arr->shape[normalized_axis];
    if (!allow_unequal && axis_size % sections != 0) {
        cnp_set_error(CNP_ERR_SHAPE, function_name,
                      "array split does not result in an equal division");
        return CNP_ERR_SHAPE;
    }

    int64_t base_size = axis_size / sections;
    int64_t remainder = axis_size % sections;
    int64_t start = 0;
    for (int section = 0; section < sections; ++section) {
        int64_t length = base_size;
        if (allow_unequal && section < remainder) ++length;
        int64_t stop = start + length;
        results[section] = split_create_view(
            arr, normalized_axis, start, stop);
        if (!results[section]) {
            split_release_partial(results, section);
            return CNP_ERR_MEMORY;
        }
        start = stop;
    }
    return CNP_OK;
}

static int64_t split_normalize_index(int64_t index, int64_t axis_size) {
    if (index < 0) index += axis_size;
    if (index < 0) return 0;
    if (index > axis_size) return axis_size;
    return index;
}

static CNP_STATUS split_indices_impl(
    const char *function_name, const CnpArray *arr,
    int nindices, const int64_t *indices, int axis,
    CnpArray **results, int result_capacity) {
    split_clear_results(results, result_capacity);
    if (nindices < 0 || nindices == INT_MAX) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "index count is out of range: %d", nindices);
        return CNP_ERR_GENERIC;
    }
    if (nindices > 0 && !indices) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "indices must not be null when index count is positive");
        return CNP_ERR_GENERIC;
    }

    int output_count = nindices + 1;
    int normalized_axis = 0;
    CNP_STATUS status = split_validate(
        function_name, arr, axis, results, result_capacity, output_count,
        &normalized_axis);
    if (status != CNP_OK) return status;

    int64_t axis_size = arr->shape[normalized_axis];
    int64_t start = 0;
    for (int output = 0; output < output_count; ++output) {
        int64_t stop = output < nindices
            ? split_normalize_index(indices[output], axis_size)
            : axis_size;
        results[output] = split_create_view(
            arr, normalized_axis, start, stop);
        if (!results[output]) {
            split_release_partial(results, output);
            return CNP_ERR_MEMORY;
        }
        start = stop;
    }
    return CNP_OK;
}

CNP_API CNP_STATUS CNP_CALL cnp_split_sections_v2(
    const CnpArray *arr, int sections, int axis,
    CnpArray **results, int result_capacity) {
    return split_sections_impl(
        "cnp_split_sections_v2", arr, sections, axis, false,
        results, result_capacity);
}

CNP_API CNP_STATUS CNP_CALL cnp_split_indices_v2(
    const CnpArray *arr, int nindices, const int64_t *indices, int axis,
    CnpArray **results, int result_capacity) {
    return split_indices_impl(
        "cnp_split_indices_v2", arr, nindices, indices, axis,
        results, result_capacity);
}

CNP_API CNP_STATUS CNP_CALL cnp_array_split_sections_v2(
    const CnpArray *arr, int sections, int axis,
    CnpArray **results, int result_capacity) {
    return split_sections_impl(
        "cnp_array_split_sections_v2", arr, sections, axis, true,
        results, result_capacity);
}

CNP_API CNP_STATUS CNP_CALL cnp_array_split_indices_v2(
    const CnpArray *arr, int nindices, const int64_t *indices, int axis,
    CnpArray **results, int result_capacity) {
    return split_indices_impl(
        "cnp_array_split_indices_v2", arr, nindices, indices, axis,
        results, result_capacity);
}

static CNP_STATUS split_legacy_dispatch(
    const char *function_name, const CnpArray *arr, int nsections,
    int64_t *indices_or_sections, int axis, bool allow_unequal,
    CnpArray **results) {
    if (indices_or_sections) {
        if (nsections <= 0) {
            split_clear_results(results, nsections);
            cnp_set_error(CNP_ERR_GENERIC, function_name,
                          "legacy output count must be greater than zero");
            return CNP_ERR_GENERIC;
        }
        return split_indices_impl(
            function_name, arr, nsections - 1, indices_or_sections, axis,
            results, nsections);
    }
    return split_sections_impl(
        function_name, arr, nsections, axis, allow_unequal,
        results, nsections > 0 ? nsections : 0);
}

CNP_API CNP_STATUS CNP_CALL cnp_split(
    const CnpArray *arr, int nsections, int64_t *indices_or_sections,
    int axis, CnpArray **result) {
    return split_legacy_dispatch(
        "cnp_split", arr, nsections, indices_or_sections, axis, false, result);
}

CNP_API CNP_STATUS CNP_CALL cnp_array_split(
    const CnpArray *arr, int nsections, int axis, CnpArray **result) {
    return split_sections_impl(
        "cnp_array_split", arr, nsections, axis, true,
        result, nsections > 0 ? nsections : 0);
}

CNP_API CNP_STATUS CNP_CALL cnp_hsplit(
    const CnpArray *arr, int nsections, int64_t *indices_or_sections,
    CnpArray **result) {
    if (!arr || arr->ndim < 1) {
        split_clear_results(result, nsections);
        cnp_set_error(CNP_ERR_SHAPE, "cnp_hsplit",
                      "hsplit only works on arrays of 1 or more dimensions");
        return CNP_ERR_SHAPE;
    }
    int axis = arr->ndim == 1 ? 0 : 1;
    return split_legacy_dispatch(
        "cnp_hsplit", arr, nsections, indices_or_sections, axis, false, result);
}

CNP_API CNP_STATUS CNP_CALL cnp_vsplit(
    const CnpArray *arr, int nsections, int64_t *indices_or_sections,
    CnpArray **result) {
    if (!arr || arr->ndim < 2) {
        split_clear_results(result, nsections);
        cnp_set_error(CNP_ERR_SHAPE, "cnp_vsplit",
                      "vsplit only works on arrays of 2 or more dimensions");
        return CNP_ERR_SHAPE;
    }
    return split_legacy_dispatch(
        "cnp_vsplit", arr, nsections, indices_or_sections, 0, false, result);
}

CNP_API CNP_STATUS CNP_CALL cnp_dsplit(
    const CnpArray *arr, int nsections, int64_t *indices_or_sections,
    CnpArray **result) {
    if (!arr || arr->ndim < 3) {
        split_clear_results(result, nsections);
        cnp_set_error(CNP_ERR_SHAPE, "cnp_dsplit",
                      "dsplit only works on arrays of 3 or more dimensions");
        return CNP_ERR_SHAPE;
    }
    return split_legacy_dispatch(
        "cnp_dsplit", arr, nsections, indices_or_sections, 2, false, result);
}

/* =========================================================================
 * Tiling and repeating
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_tile(const CnpArray *arr, int nreps, const int64_t *reps) {
    if (!shape_require_array(arr, "cnp_tile")) return NULL;

    /* Compute output shape */
    int out_ndim = arr->ndim > nreps ? arr->ndim : nreps;
    int64_t out_shape[CNP_MAXDIMS];
    for (int i = 0; i < out_ndim; i++) out_shape[i] = 1;

    int arr_offset = out_ndim - arr->ndim;
    int rep_offset = out_ndim - nreps;
    for (int i = 0; i < arr->ndim; i++) out_shape[arr_offset + i] = arr->shape[i];
    for (int i = 0; i < nreps; i++) out_shape[rep_offset + i] *= reps[i];

    CnpArray *result = cnp_array_new(out_ndim, out_shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) return NULL;

    int elsize = result->dtype->elsize;
    int64_t coords[CNP_MAXDIMS] = {0};
    for (int64_t i = 0; i < result->size; i++) {
        /* Map output coords to input coords */
        int64_t src_coords[CNP_MAXDIMS] = {0};
        for (int d = 0; d < arr->ndim; d++) {
            src_coords[d] = coords[arr_offset + d] % arr->shape[d];
        }
        int64_t src_off = arr->offset + cnp_multi_to_offset(arr->ndim, src_coords, arr->strides);
        int64_t dst_off = i * elsize;
        memcpy((char*)result->data + dst_off, (char*)arr->data + src_off, elsize);

        for (int d = out_ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < out_shape[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_repeat(const CnpArray *arr, int64_t repeats, int axis) {
    if (!shape_require_array(arr, "cnp_repeat")) return NULL;

    if (axis < 0) {
        /* Flatten and repeat */
        CnpArray *flat = cnp_flatten(arr, CNP_ORDER_C);
        int64_t new_size = flat->size * repeats;
        int64_t shape[1] = {new_size};
        CnpArray *result = cnp_array_new(1, shape, arr->dtype->type_num, CNP_ORDER_C);
        if (!result) { cnp_array_free(flat); return NULL; }

        int elsize = result->dtype->elsize;
        for (int64_t i = 0; i < flat->size; i++) {
            for (int64_t r = 0; r < repeats; r++) {
                memcpy(
                    (char*)result->data + (i * repeats + r) * elsize,
                    (char*)flat->data + flat->offset + i * elsize,
                    (size_t)elsize);
            }
        }
        cnp_array_free(flat);
        return result;
    }

    if (!shape_normalize_axis_checked(
            axis, arr->ndim, "cnp_repeat", &axis)) return NULL;
    int64_t out_shape[CNP_MAXDIMS];
    for (int i = 0; i < arr->ndim; i++) out_shape[i] = arr->shape[i];
    out_shape[axis] *= repeats;

    CnpArray *result = cnp_array_new(arr->ndim, out_shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) return NULL;

    int elsize = result->dtype->elsize;
    int64_t output_coords[CNP_MAXDIMS] = {0};
    for (int64_t output_index = 0;
         output_index < result->size; ++output_index) {
        int64_t source_coords[CNP_MAXDIMS];
        for (int dimension = 0; dimension < arr->ndim; ++dimension)
            source_coords[dimension] = output_coords[dimension];
        source_coords[axis] /= repeats;
        int64_t source_offset = arr->offset + cnp_multi_to_offset(
            arr->ndim, source_coords, arr->strides);
        memcpy(
            (char*)result->data + output_index * elsize,
            (char*)arr->data + source_offset, (size_t)elsize);
        for (int dimension = arr->ndim - 1;
             dimension >= 0; --dimension) {
            ++output_coords[dimension];
            if (output_coords[dimension] < result->shape[dimension]) break;
            output_coords[dimension] = 0;
        }
    }
    return result;
}

/* =========================================================================
 * Flip, roll, rot90
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_flip(const CnpArray *arr, int axis) {
    if (!shape_require_array(arr, "cnp_flip")) return NULL;
    if (axis >= 0 && !shape_normalize_axis_checked(
            axis, arr->ndim, "cnp_flip", &axis)) return NULL;
    CnpArray *result = cnp_array_new(arr->ndim, arr->shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) return NULL;

    int elsize = result->dtype->elsize;
    int64_t coords[CNP_MAXDIMS] = {0};
    for (int64_t i = 0; i < arr->size; i++) {
        int64_t dst_coords[CNP_MAXDIMS];
        for (int d = 0; d < arr->ndim; d++) dst_coords[d] = coords[d];
        if (axis < 0) {
            /* Flip all axes */
            for (int d = 0; d < arr->ndim; d++) dst_coords[d] = arr->shape[d] - 1 - coords[d];
        } else {
            dst_coords[axis] = arr->shape[axis] - 1 - coords[axis];
        }
        int64_t src_off = arr->offset + cnp_multi_to_offset(arr->ndim, coords, arr->strides);
        int64_t dst_off = cnp_multi_to_offset(arr->ndim, dst_coords, result->strides);
        memcpy((char*)result->data + dst_off, (char*)arr->data + src_off, elsize);

        for (int d = arr->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < arr->shape[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_rot90(const CnpArray *arr, int k, int axis1, int axis2) {
    if (!shape_require_array(arr, "cnp_rot90")) return NULL;
    if (arr->ndim < 2) {
        cnp_set_error(
            CNP_ERR_AXIS, "cnp_rot90",
            "input must have at least two dimensions");
        return NULL;
    }
    if (!shape_normalize_axis_checked(
            axis1, arr->ndim, "cnp_rot90", &axis1) ||
        !shape_normalize_axis_checked(
            axis2, arr->ndim, "cnp_rot90", &axis2)) return NULL;
    if (axis1 == axis2) {
        cnp_set_error(
            CNP_ERR_AXIS, "cnp_rot90",
            "rotation axes must be different");
        return NULL;
    }
    k = ((k % 4) + 4) % 4;
    if (k == 0) return cnp_array_copy(arr);

    int axes_arr[CNP_MAXDIMS];
    for (int dimension = 0; dimension < arr->ndim; ++dimension)
        axes_arr[dimension] = dimension;
    int temporary_axis = axes_arr[axis1];
    axes_arr[axis1] = axes_arr[axis2];
    axes_arr[axis2] = temporary_axis;

    if (k == 1) {
        CnpArray *flipped = cnp_flip(arr, axis2);
        if (!flipped) return NULL;
        CnpArray *result = cnp_transpose(flipped, axes_arr);
        cnp_array_decref(flipped);
        if (!result) cnp_relabel_error("cnp_rot90");
        return result;
    }
    if (k == 2) {
        CnpArray *first_flip = cnp_flip(arr, axis1);
        if (!first_flip) return NULL;
        CnpArray *result = cnp_flip(first_flip, axis2);
        cnp_array_free(first_flip);
        if (!result) cnp_relabel_error("cnp_rot90");
        return result;
    }

    CnpArray *transposed = cnp_transpose(arr, axes_arr);
    if (!transposed) return NULL;
    CnpArray *result = cnp_flip(transposed, axis2);
    cnp_array_free(transposed);
    if (!result) cnp_relabel_error("cnp_rot90");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_roll(const CnpArray *arr, int64_t shift, int axis) {
    if (!shape_require_array(arr, "cnp_roll")) return NULL;
    if (axis >= 0 && !shape_normalize_axis_checked(
            axis, arr->ndim, "cnp_roll", &axis)) return NULL;
    CnpArray *result = cnp_array_new(arr->ndim, arr->shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) return NULL;

    int elsize = result->dtype->elsize;
    int64_t coords[CNP_MAXDIMS] = {0};
    for (int64_t i = 0; i < arr->size; i++) {
        int64_t dst_coords[CNP_MAXDIMS];
        for (int d = 0; d < arr->ndim; d++) dst_coords[d] = coords[d];
        if (axis < 0) {
            /* Roll flattened */
            int64_t flat_idx = i;
            int64_t new_flat = (flat_idx + shift) % arr->size;
            if (new_flat < 0) new_flat += arr->size;
            int64_t src_off = arr->offset + cnp_multi_to_offset(arr->ndim, coords, arr->strides);
            memcpy((char*)result->data + new_flat * elsize, (char*)arr->data + src_off, elsize);
        } else {
            dst_coords[axis] = (coords[axis] + shift) % arr->shape[axis];
            if (dst_coords[axis] < 0) dst_coords[axis] += arr->shape[axis];
            int64_t src_off = arr->offset + cnp_multi_to_offset(arr->ndim, coords, arr->strides);
            int64_t dst_off = cnp_multi_to_offset(arr->ndim, dst_coords, result->strides);
            memcpy((char*)result->data + dst_off, (char*)arr->data + src_off, elsize);
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
 * Append, insert, delete
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_append(const CnpArray *arr, const CnpArray *values, int axis) {
    if (!shape_require_array(arr, "cnp_append") ||
            !shape_require_array(values, "cnp_append")) return NULL;
    if (axis < 0) {
        CnpArray *flat_arr = cnp_flatten(arr, CNP_ORDER_C);
        CnpArray *flat_val = cnp_flatten(values, CNP_ORDER_C);
        CnpArray *arrs[2] = {flat_arr, flat_val};
        CnpArray *result = cnp_concatenate(2, arrs, 0);
        cnp_array_free(flat_arr);
        cnp_array_free(flat_val);
        return result;
    }
    if (!shape_normalize_axis_checked(
            axis, arr->ndim, "cnp_append", &axis)) return NULL;
    CnpArray *arrs[2] = {(CnpArray*)arr, (CnpArray*)values};
    CnpArray *result = cnp_concatenate(2, arrs, axis);
    if (!result) cnp_relabel_error("cnp_append");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_delete(const CnpArray *arr, const CnpArray *obj, int axis) {
    return cnp_delete_v2(arr, obj, axis, axis < 0);
}

CNP_API CnpArray* CNP_CALL cnp_insert(CnpArray *arr, int64_t obj, const CnpArray *values, int axis) {
    return cnp_insert_v2(arr, obj, values, axis, axis < 0);
}

/* =========================================================================
 * Pad
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_pad(const CnpArray *arr, int64_t pad_width, double constant_value) {
    if (!shape_require_array(arr, "cnp_pad")) return NULL;
    int64_t out_shape[CNP_MAXDIMS];
    for (int i = 0; i < arr->ndim; i++) out_shape[i] = arr->shape[i] + 2 * pad_width;

    CnpArray *result = cnp_array_full(arr->ndim, out_shape, constant_value, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) return NULL;

    int elsize = result->dtype->elsize;
    int64_t coords[CNP_MAXDIMS] = {0};
    for (int64_t i = 0; i < arr->size; i++) {
        int64_t dst_coords[CNP_MAXDIMS];
        for (int d = 0; d < arr->ndim; d++) dst_coords[d] = coords[d] + pad_width;
        int64_t src_off = arr->offset + cnp_multi_to_offset(arr->ndim, coords, arr->strides);
        int64_t dst_off = cnp_multi_to_offset(arr->ndim, dst_coords, result->strides);
        memcpy((char*)result->data + dst_off, (char*)arr->data + src_off, elsize);

        for (int d = arr->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < arr->shape[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}
