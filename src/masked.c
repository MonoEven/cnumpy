/**
 * cnumpy masked array support
 * Corresponds to numpy.ma module
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>

/* CnpMaskedArray is defined in cnumpy.h */

static bool masked_require_array(const CnpArray *array,
                                 const char *function_name,
                                 const char *role) {
    if (array) return true;
    cnp_set_error(CNP_ERR_GENERIC, function_name, "%s array is required", role);
    return false;
}

static bool masked_require_owner(const CnpMaskedArray *ma,
                                 const char *function_name) {
    if (ma) return true;
    cnp_set_error(CNP_ERR_GENERIC, function_name, "Masked array is required");
    return false;
}

static bool masked_same_shape(const CnpArray *left, const CnpArray *right,
                              const char *function_name,
                              const char *right_role) {
    if (left->ndim != right->ndim) {
        cnp_set_error(CNP_ERR_SHAPE, function_name,
                      "%s ndim %d does not match data ndim %d",
                      right_role, right->ndim, left->ndim);
        return false;
    }
    for (int axis = 0; axis < left->ndim; ++axis) {
        if (left->shape[axis] != right->shape[axis]) {
            cnp_set_error(CNP_ERR_SHAPE, function_name,
                          "%s shape differs at axis %d", right_role, axis);
            return false;
        }
    }
    return true;
}

static bool masked_validate_mask(const CnpArray *data, const CnpArray *mask,
                                 const char *function_name) {
    if (!masked_require_array(mask, function_name, "Mask")) return false;
    if (mask->dtype->type_num != CNP_BOOL) {
        cnp_set_error(CNP_ERR_TYPE, function_name,
                      "Mask dtype must be bool, got %s", mask->dtype->name);
        return false;
    }
    return masked_same_shape(data, mask, function_name, "Mask");
}

static CnpArray* masked_new_mask(const CnpArray *data,
                                 const char *function_name) {
    CnpArray *mask = cnp_array_new(
        data->ndim, data->shape, CNP_BOOL, CNP_ORDER_C);
    if (!mask) cnp_relabel_error(function_name);
    return mask;
}

static CnpMaskedArray* masked_finish(const CnpArray *data, CnpArray *mask,
                                     double fill_value,
                                     const char *function_name) {
    CnpMaskedArray *ma = cnp_masked_array_create(data, mask, fill_value);
    cnp_array_decref(mask);
    if (!ma) cnp_relabel_error(function_name);
    return ma;
}

/* =========================================================================
 * cnp_masked_array_create - Create a masked array
 * ========================================================================= */
CNP_API CnpMaskedArray* CNP_CALL cnp_masked_array_create(const CnpArray *data, const CnpArray *mask, double fill_value) {
    const char *function_name = "cnp_masked_array_create";
    if (!masked_require_array(data, function_name, "Data")) return NULL;
    if (mask && !masked_validate_mask(data, mask, function_name)) return NULL;

    CnpMaskedArray *ma = (CnpMaskedArray*)cnp_calloc(1, sizeof(CnpMaskedArray));
    if (!ma) {
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "Failed to allocate masked array");
        return NULL;
    }

    ma->data = cnp_array_copy(data);
    if (!ma->data) {
        cnp_free(ma, sizeof(CnpMaskedArray));
        cnp_relabel_error(function_name);
        return NULL;
    }

    if (mask) {
        ma->mask = cnp_array_copy(mask);
    } else {
        /* Create all-false mask */
        ma->mask = cnp_array_zeros(data->ndim, data->shape, CNP_BOOL, CNP_ORDER_C);
    }

    if (!ma->mask) {
        cnp_array_decref(ma->data);
        cnp_free(ma, sizeof(CnpMaskedArray));
        cnp_relabel_error(function_name);
        return NULL;
    }

    ma->fill_value = fill_value;
    return ma;
}

/* =========================================================================
 * cnp_masked_array_free - Free a masked array
 * ========================================================================= */
CNP_API void CNP_CALL cnp_masked_array_free(CnpMaskedArray *ma) {
    if (!ma) return;
    if (ma->data) cnp_array_decref(ma->data);
    if (ma->mask) cnp_array_decref(ma->mask);
    cnp_free(ma, sizeof(CnpMaskedArray));
}

/* =========================================================================
 * cnp_masked_array_get_data - Get underlying data array
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_masked_array_get_data(const CnpMaskedArray *ma) {
    if (!masked_require_owner(ma, "cnp_masked_array_get_data")) return NULL;
    CnpArray *view = cnp_array_view(ma->data);
    if (!view) cnp_relabel_error("cnp_masked_array_get_data");
    return view;
}

/* =========================================================================
 * cnp_masked_array_get_mask - Get mask array
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_masked_array_get_mask(const CnpMaskedArray *ma) {
    if (!masked_require_owner(ma, "cnp_masked_array_get_mask")) return NULL;
    CnpArray *view = cnp_array_view(ma->mask);
    if (!view) cnp_relabel_error("cnp_masked_array_get_mask");
    return view;
}

/* =========================================================================
 * cnp_masked_array_set_mask - Set mask from boolean array
 * ========================================================================= */
CNP_API CNP_STATUS CNP_CALL cnp_masked_array_set_mask(CnpMaskedArray *ma, const CnpArray *mask) {
    const char *function_name = "cnp_masked_array_set_mask";
    if (!masked_require_owner(ma, function_name)) return CNP_ERR_GENERIC;
    if (!masked_validate_mask(ma->data, mask, function_name)) {
        CnpErrorState error;
        return cnp_get_error(&error);
    }

    CnpArray *replacement = cnp_array_copy(mask);
    if (!replacement) {
        cnp_relabel_error(function_name);
        return CNP_ERR_MEMORY;
    }

    CnpArray *previous = ma->mask;
    ma->mask = replacement;
    if (previous) cnp_array_decref(previous);
    return CNP_OK;
}

/* =========================================================================
 * cnp_masked_array_filled - Return filled array (masked values replaced)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_masked_array_filled(const CnpMaskedArray *ma, double fill_value) {
    const char *function_name = "cnp_masked_array_filled";
    if (!masked_require_owner(ma, function_name)) return NULL;

    CnpArray *result = cnp_array_copy(ma->data);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int elsize = result->dtype->elsize;
    for (int64_t i = 0; i < result->size && i < ma->mask->size; i++) {
        double m = cnp_array_flat_get(ma->mask, i);
        if (m != 0.0) {
            cnp_set_element_double(result->data, i * elsize, result->dtype->type_num, fill_value);
        }
    }
    return result;
}

/* =========================================================================
 * cnp_masked_array_compressed - Return non-masked values as 1D array
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_masked_array_compressed(const CnpMaskedArray *ma) {
    const char *function_name = "cnp_masked_array_compressed";
    if (!masked_require_owner(ma, function_name)) return NULL;

    /* Count non-masked elements */
    int64_t count = 0;
    for (int64_t i = 0; i < ma->data->size && i < ma->mask->size; i++) {
        if (cnp_array_flat_get(ma->mask, i) == 0.0) count++;
    }

    int64_t shape[1] = {count};
    CnpArray *result = cnp_array_new(1, shape, ma->data->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int elsize = result->dtype->elsize;
    int64_t idx = 0;
    for (int64_t i = 0; i < ma->data->size && i < ma->mask->size; i++) {
        if (cnp_array_flat_get(ma->mask, i) == 0.0) {
            memcpy((char*)result->data + idx * elsize,
                   (const char*)ma->data->data + ma->data->offset + i * elsize,
                   (size_t)elsize);
            idx++;
        }
    }
    return result;
}

/* =========================================================================
 * cnp_masked_array_count - Count non-masked elements
 * ========================================================================= */
CNP_API int64_t CNP_CALL cnp_masked_array_count(const CnpMaskedArray *ma) {
    if (!masked_require_owner(ma, "cnp_masked_array_count")) return -1;

    int64_t count = 0;
    for (int64_t i = 0; i < ma->data->size && i < ma->mask->size; i++) {
        if (cnp_array_flat_get(ma->mask, i) == 0.0) count++;
    }
    return count;
}

/* =========================================================================
 * cnp_masked_array_sum - Sum of non-masked elements
 * ========================================================================= */
CNP_API double CNP_CALL cnp_masked_array_sum(const CnpMaskedArray *ma) {
    if (!masked_require_owner(ma, "cnp_masked_array_sum")) return NAN;

    double sum = 0.0;
    for (int64_t i = 0; i < ma->data->size && i < ma->mask->size; i++) {
        if (cnp_array_flat_get(ma->mask, i) == 0.0) {
            sum += cnp_array_flat_get(ma->data, i);
        }
    }
    return sum;
}

/* =========================================================================
 * cnp_masked_array_mean - Mean of non-masked elements
 * ========================================================================= */
CNP_API double CNP_CALL cnp_masked_array_mean(const CnpMaskedArray *ma) {
    if (!masked_require_owner(ma, "cnp_masked_array_mean")) return NAN;

    int64_t count = cnp_masked_array_count(ma);
    if (count == 0) return 0.0;

    return cnp_masked_array_sum(ma) / (double)count;
}

/* =========================================================================
 * cnp_masked_array_std - Standard deviation of non-masked elements
 * ========================================================================= */
CNP_API double CNP_CALL cnp_masked_array_std(const CnpMaskedArray *ma) {
    if (!masked_require_owner(ma, "cnp_masked_array_std")) return NAN;

    int64_t count = cnp_masked_array_count(ma);
    if (count <= 1) return 0.0;

    double mean = cnp_masked_array_mean(ma);
    double sum_sq = 0.0;
    for (int64_t i = 0; i < ma->data->size && i < ma->mask->size; i++) {
        if (cnp_array_flat_get(ma->mask, i) == 0.0) {
            double diff = cnp_array_flat_get(ma->data, i) - mean;
            sum_sq += diff * diff;
        }
    }
    return sqrt(sum_sq / (double)count);
}

/* =========================================================================
 * cnp_masked_array_min - Minimum of non-masked elements
 * ========================================================================= */
CNP_API double CNP_CALL cnp_masked_array_min(const CnpMaskedArray *ma) {
    if (!masked_require_owner(ma, "cnp_masked_array_min")) return NAN;

    double min_val = INFINITY;
    bool found = false;
    for (int64_t i = 0; i < ma->data->size && i < ma->mask->size; i++) {
        if (cnp_array_flat_get(ma->mask, i) == 0.0) {
            double v = cnp_array_flat_get(ma->data, i);
            if (!found || v < min_val) {
                min_val = v;
                found = true;
            }
        }
    }
    return found ? min_val : 0.0;
}

/* =========================================================================
 * cnp_masked_array_max - Maximum of non-masked elements
 * ========================================================================= */
CNP_API double CNP_CALL cnp_masked_array_max(const CnpMaskedArray *ma) {
    if (!masked_require_owner(ma, "cnp_masked_array_max")) return NAN;

    double max_val = -INFINITY;
    bool found = false;
    for (int64_t i = 0; i < ma->data->size && i < ma->mask->size; i++) {
        if (cnp_array_flat_get(ma->mask, i) == 0.0) {
            double v = cnp_array_flat_get(ma->data, i);
            if (!found || v > max_val) {
                max_val = v;
                found = true;
            }
        }
    }
    return found ? max_val : 0.0;
}

/* =========================================================================
 * cnp_masked_where - Create masked array from condition
 * ========================================================================= */
CNP_API CnpMaskedArray* CNP_CALL cnp_masked_where(const CnpArray *condition, const CnpArray *data, double fill_value) {
    const char *function_name = "cnp_masked_where";
    if (!masked_require_array(condition, function_name, "Condition") ||
        !masked_require_array(data, function_name, "Data")) return NULL;
    if (!masked_same_shape(data, condition, function_name, "Condition"))
        return NULL;

    /* Mask where condition is true */
    CnpArray *mask = masked_new_mask(data, function_name);
    if (!mask) return NULL;

    int8_t *mask_data = (int8_t*)mask->data;
    for (int64_t i = 0; i < data->size && i < condition->size; i++) {
        mask_data[i] = (cnp_array_flat_get(condition, i) != 0.0) ? 1 : 0;
    }

    return masked_finish(data, mask, fill_value, function_name);
}

/* =========================================================================
 * cnp_masked_invalid - Mask invalid values (NaN, Inf)
 * ========================================================================= */
CNP_API CnpMaskedArray* CNP_CALL cnp_masked_invalid(const CnpArray *data, double fill_value) {
    const char *function_name = "cnp_masked_invalid";
    if (!masked_require_array(data, function_name, "Data")) return NULL;

    CnpArray *mask = masked_new_mask(data, function_name);
    if (!mask) return NULL;

    int8_t *mask_data = (int8_t*)mask->data;
    for (int64_t i = 0; i < data->size; i++) {
        double v = cnp_array_flat_get(data, i);
        mask_data[i] = (isnan(v) || isinf(v)) ? 1 : 0;
    }

    return masked_finish(data, mask, fill_value, function_name);
}

/* =========================================================================
 * cnp_masked_greater - Mask values greater than threshold
 * ========================================================================= */
CNP_API CnpMaskedArray* CNP_CALL cnp_masked_greater(const CnpArray *data, double value, double fill_value) {
    const char *function_name = "cnp_masked_greater";
    if (!masked_require_array(data, function_name, "Data")) return NULL;

    CnpArray *mask = masked_new_mask(data, function_name);
    if (!mask) return NULL;

    int8_t *mask_data = (int8_t*)mask->data;
    for (int64_t i = 0; i < data->size; i++) {
        mask_data[i] = (cnp_array_flat_get(data, i) > value) ? 1 : 0;
    }

    return masked_finish(data, mask, fill_value, function_name);
}

/* =========================================================================
 * cnp_masked_less - Mask values less than threshold
 * ========================================================================= */
CNP_API CnpMaskedArray* CNP_CALL cnp_masked_less(const CnpArray *data, double value, double fill_value) {
    const char *function_name = "cnp_masked_less";
    if (!masked_require_array(data, function_name, "Data")) return NULL;

    CnpArray *mask = masked_new_mask(data, function_name);
    if (!mask) return NULL;

    int8_t *mask_data = (int8_t*)mask->data;
    for (int64_t i = 0; i < data->size; i++) {
        mask_data[i] = (cnp_array_flat_get(data, i) < value) ? 1 : 0;
    }

    return masked_finish(data, mask, fill_value, function_name);
}

/* =========================================================================
 * cnp_masked_equal - Mask values equal to given value
 * ========================================================================= */
CNP_API CnpMaskedArray* CNP_CALL cnp_masked_equal(const CnpArray *data, double value, double fill_value) {
    const char *function_name = "cnp_masked_equal";
    if (!masked_require_array(data, function_name, "Data")) return NULL;

    CnpArray *mask = masked_new_mask(data, function_name);
    if (!mask) return NULL;

    int8_t *mask_data = (int8_t*)mask->data;
    for (int64_t i = 0; i < data->size; i++) {
        double element = cnp_array_flat_get(data, i);
        mask_data[i] = !isnan(element) && element == value ? 1 : 0;
    }

    return masked_finish(data, mask, fill_value, function_name);
}

/* =========================================================================
 * cnp_masked_not_equal - Mask values not equal to given value
 * ========================================================================= */
CNP_API CnpMaskedArray* CNP_CALL cnp_masked_not_equal(const CnpArray *data, double value, double fill_value) {
    const char *function_name = "cnp_masked_not_equal";
    if (!masked_require_array(data, function_name, "Data")) return NULL;

    CnpArray *mask = masked_new_mask(data, function_name);
    if (!mask) return NULL;

    int8_t *mask_data = (int8_t*)mask->data;
    for (int64_t i = 0; i < data->size; i++) {
        double element = cnp_array_flat_get(data, i);
        mask_data[i] = isnan(element) || element != value ? 1 : 0;
    }

    return masked_finish(data, mask, fill_value, function_name);
}

/* =========================================================================
 * cnp_masked_inside - Mask values inside range [v1, v2]
 * ========================================================================= */
CNP_API CnpMaskedArray* CNP_CALL cnp_masked_inside(const CnpArray *data, double v1, double v2, double fill_value) {
    const char *function_name = "cnp_masked_inside";
    if (!masked_require_array(data, function_name, "Data")) return NULL;

    CnpArray *mask = masked_new_mask(data, function_name);
    if (!mask) return NULL;

    int8_t *mask_data = (int8_t*)mask->data;
    for (int64_t i = 0; i < data->size; i++) {
        double v = cnp_array_flat_get(data, i);
        mask_data[i] = (v >= v1 && v <= v2) ? 1 : 0;
    }

    return masked_finish(data, mask, fill_value, function_name);
}

/* =========================================================================
 * cnp_masked_outside - Mask values outside range [v1, v2]
 * ========================================================================= */
CNP_API CnpMaskedArray* CNP_CALL cnp_masked_outside(const CnpArray *data, double v1, double v2, double fill_value) {
    const char *function_name = "cnp_masked_outside";
    if (!masked_require_array(data, function_name, "Data")) return NULL;

    CnpArray *mask = masked_new_mask(data, function_name);
    if (!mask) return NULL;

    int8_t *mask_data = (int8_t*)mask->data;
    for (int64_t i = 0; i < data->size; i++) {
        double v = cnp_array_flat_get(data, i);
        mask_data[i] = !isnan(v) && (v < v1 || v > v2) ? 1 : 0;
    }

    return masked_finish(data, mask, fill_value, function_name);
}
