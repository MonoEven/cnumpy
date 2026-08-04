/**
 * cnumpy set operations corresponding to NumPy 1.25 arraysetops.
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>

static bool set_dtype_is_supported(CNP_TYPE dtype) {
    return dtype == CNP_BOOL ||
           (dtype >= CNP_BYTE && dtype <= CNP_LONGDOUBLE) ||
           dtype == CNP_HALF;
}

#define CNP_SET_EQUAL_TYPED(type) \
    return *(const type*)left == *(const type*)right

static bool set_scalar_equal(
    const void *left_value, const void *right_value,
    CNP_TYPE dtype, bool equal_nan) {
    const char *left = (const char*)left_value;
    const char *right = (const char*)right_value;
    switch (dtype) {
        case CNP_BOOL:
        case CNP_BYTE: CNP_SET_EQUAL_TYPED(int8_t);
        case CNP_UBYTE: CNP_SET_EQUAL_TYPED(uint8_t);
        case CNP_SHORT: CNP_SET_EQUAL_TYPED(int16_t);
        case CNP_USHORT: CNP_SET_EQUAL_TYPED(uint16_t);
        case CNP_INT: CNP_SET_EQUAL_TYPED(int32_t);
        case CNP_UINT: CNP_SET_EQUAL_TYPED(uint32_t);
        case CNP_LONG:
        case CNP_LONGLONG: CNP_SET_EQUAL_TYPED(int64_t);
        case CNP_ULONG:
        case CNP_ULONGLONG: CNP_SET_EQUAL_TYPED(uint64_t);
        case CNP_FLOAT: {
            float a = *(const float*)left;
            float b = *(const float*)right;
            bool a_nan = isnan(a);
            bool b_nan = isnan(b);
            if (a_nan || b_nan)
                return equal_nan && a_nan && b_nan;
            return a == b;
        }
        case CNP_DOUBLE: {
            double a = *(const double*)left;
            double b = *(const double*)right;
            bool a_nan = isnan(a);
            bool b_nan = isnan(b);
            if (a_nan || b_nan)
                return equal_nan && a_nan && b_nan;
            return a == b;
        }
        case CNP_LONGDOUBLE: {
            long double a = *(const long double*)left;
            long double b = *(const long double*)right;
            bool a_nan = isnan(a);
            bool b_nan = isnan(b);
            if (a_nan || b_nan)
                return equal_nan && a_nan && b_nan;
            return a == b;
        }
        case CNP_HALF: {
            double a = cnp_half_to_float(*(const uint16_t*)left);
            double b = cnp_half_to_float(*(const uint16_t*)right);
            bool a_nan = isnan(a);
            bool b_nan = isnan(b);
            if (a_nan || b_nan)
                return equal_nan && a_nan && b_nan;
            return a == b;
        }
        default:
            return false;
    }
}

static bool set_values_equal(
    const void *data, int64_t left_index, int64_t right_index,
    CNP_TYPE dtype, int element_size, bool equal_nan) {
    const char *bytes = (const char*)data;
    return set_scalar_equal(
        bytes + left_index * element_size,
        bytes + right_index * element_size,
        dtype, equal_nan);
}

#undef CNP_SET_EQUAL_TYPED

#define CNP_SET_COMPARE_TYPED(type) do { \
    type left = *(const type*)left_value; \
    type right = *(const type*)right_value; \
    return left < right ? -1 : (left > right ? 1 : 0); \
} while (0)

static int set_compare_long_double(long double left, long double right) {
    bool left_nan = isnan(left);
    bool right_nan = isnan(right);
    if (left_nan && right_nan) return 0;
    if (left_nan) return 1;
    if (right_nan) return -1;
    return left < right ? -1 : (left > right ? 1 : 0);
}

static int set_scalar_compare(
    const void *left_value, const void *right_value, CNP_TYPE dtype) {
    switch (dtype) {
        case CNP_BOOL:
        case CNP_BYTE: CNP_SET_COMPARE_TYPED(int8_t);
        case CNP_UBYTE: CNP_SET_COMPARE_TYPED(uint8_t);
        case CNP_SHORT: CNP_SET_COMPARE_TYPED(int16_t);
        case CNP_USHORT: CNP_SET_COMPARE_TYPED(uint16_t);
        case CNP_INT: CNP_SET_COMPARE_TYPED(int32_t);
        case CNP_UINT: CNP_SET_COMPARE_TYPED(uint32_t);
        case CNP_LONG:
        case CNP_LONGLONG: CNP_SET_COMPARE_TYPED(int64_t);
        case CNP_ULONG:
        case CNP_ULONGLONG: CNP_SET_COMPARE_TYPED(uint64_t);
        case CNP_FLOAT:
            return cnp_compare_numpy_doubles(
                (double)*(const float*)left_value,
                (double)*(const float*)right_value);
        case CNP_DOUBLE:
            return cnp_compare_numpy_doubles(
                *(const double*)left_value,
                *(const double*)right_value);
        case CNP_LONGDOUBLE:
            return set_compare_long_double(
                *(const long double*)left_value,
                *(const long double*)right_value);
        case CNP_HALF:
            return cnp_compare_numpy_doubles(
                cnp_half_to_float(*(const uint16_t*)left_value),
                cnp_half_to_float(*(const uint16_t*)right_value));
        default:
            return 0;
    }
}

#undef CNP_SET_COMPARE_TYPED

static bool membership_sorted_contains(
    const void *sorted_data, int64_t count, const void *value,
    CNP_TYPE dtype, int element_size) {
    const char *bytes = (const char*)sorted_data;
    int64_t low = 0;
    int64_t high = count;
    while (low < high) {
        int64_t middle = low + (high - low) / 2;
        int comparison = set_scalar_compare(
            bytes + middle * element_size, value, dtype);
        if (comparison < 0)
            low = middle + 1;
        else
            high = middle;
    }
    return low < count && set_scalar_equal(
        bytes + low * element_size, value, dtype, false);
}

static void release_arrays(CnpArray **arrays, int count) {
    for (int index = 0; index < count; ++index) {
        if (arrays[index]) {
            cnp_array_free(arrays[index]);
            arrays[index] = NULL;
        }
    }
}

static CnpArray *flatten_as(
    const CnpArray *arr, CNP_TYPE dtype, const char *function_name) {
    CnpArray *flat = cnp_flatten(arr, CNP_ORDER_C);
    if (!flat) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (flat->dtype->type_num == dtype) return flat;

    CnpArray *cast = cnp_astype(flat, dtype, CNP_CAST_UNSAFE);
    cnp_array_free(flat);
    if (!cast) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    return cast;
}

static bool prepare_common_flat(
    const CnpArray *left_source, const CnpArray *right_source,
    CnpArray **left, CnpArray **right, CNP_TYPE *dtype,
    const char *function_name) {
    *left = NULL;
    *right = NULL;
    if (!left_source || !right_source) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "input arrays must not be NULL");
        return false;
    }
    *dtype = cnp_promote_type(
        left_source->dtype->type_num, right_source->dtype->type_num);
    if (!set_dtype_is_supported(*dtype)) {
        cnp_set_error(CNP_ERR_TYPE, function_name,
                      "Set operation dtype %d is not supported", (int)*dtype);
        return false;
    }

    *left = flatten_as(left_source, *dtype, function_name);
    if (!*left) return false;
    *right = flatten_as(right_source, *dtype, function_name);
    if (!*right) {
        cnp_array_free(*left);
        *left = NULL;
        return false;
    }
    return true;
}

static CnpArray *concatenate_flat(
    const CnpArray *left, const CnpArray *right,
    CNP_TYPE dtype, const char *function_name) {
    if (left->size > INT64_MAX - right->size) {
        cnp_set_error(CNP_ERR_SHAPE, function_name,
                      "Concatenated set input is too large");
        return NULL;
    }
    int64_t total = left->size + right->size;
    CnpArray *result = cnp_array_new(1, &total, dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    size_t left_bytes = (size_t)left->size * left->dtype->elsize;
    size_t right_bytes = (size_t)right->size * right->dtype->elsize;
    memcpy(result->data, (const char*)left->data + left->offset, left_bytes);
    memcpy((char*)result->data + left_bytes,
           (const char*)right->data + right->offset, right_bytes);
    return result;
}

static CnpArray *unique_values(
    const CnpArray *arr, const char *function_name) {
    CnpArray *outputs[1] = {NULL};
    CNP_STATUS status = cnp_unique_v2(
        arr, false, false, false, outputs, 1);
    if (status != CNP_OK) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    return outputs[0];
}

static CnpArray *set_source(
    const CnpArray *arr, bool assume_unique, const char *function_name) {
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "input array must not be NULL");
        return NULL;
    }
    if (assume_unique) {
        if (!set_dtype_is_supported(arr->dtype->type_num)) {
            cnp_set_error(CNP_ERR_TYPE, function_name,
                          "Set operation dtype %d is not supported",
                          (int)arr->dtype->type_num);
            return NULL;
        }
        return flatten_as(arr, arr->dtype->type_num, function_name);
    }
    return unique_values(arr, function_name);
}

static CnpArray *unique_values_only(
    const CnpArray *flat, const char *function_name) {
    CnpArray *sorted = cnp_sort_v2(
        flat, 0, false, CNP_SORT_QUICKSORT);
    if (!sorted) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    const char *sorted_data =
        (const char*)sorted->data + sorted->offset;
    int64_t count = sorted->size;
    int64_t unique_count = count > 0 ? 1 : 0;
    for (int64_t position = 1; position < count; ++position) {
        if (!set_values_equal(
                sorted_data, position - 1, position,
                sorted->dtype->type_num, sorted->dtype->elsize, true)) {
            ++unique_count;
        }
    }

    CnpArray *result = cnp_array_new(
        1, &unique_count, sorted->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_array_free(sorted);
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (count > 0) {
        int64_t destination = 0;
        memcpy(
            (char*)result->data + destination++ * result->dtype->elsize,
            sorted_data,
            sorted->dtype->elsize);
        for (int64_t position = 1; position < count; ++position) {
            if (set_values_equal(
                    sorted_data, position - 1, position,
                    sorted->dtype->type_num,
                    sorted->dtype->elsize, true)) {
                continue;
            }
            memcpy(
                (char*)result->data +
                    destination++ * result->dtype->elsize,
                sorted_data + position * sorted->dtype->elsize,
                sorted->dtype->elsize);
        }
    }
    cnp_array_free(sorted);
    return result;
}

/* =========================================================================
 * unique
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_unique(
    const CnpArray *arr, bool return_index,
    bool return_inverse, bool return_counts) {
    (void)return_index;
    (void)return_inverse;
    (void)return_counts;
    CnpArray *results[1] = {NULL};
    CNP_STATUS status = cnp_unique_v2(
        arr, false, false, false, results, 1);
    if (status != CNP_OK) {
        cnp_relabel_error("cnp_unique");
        return NULL;
    }
    return results[0];
}

CNP_API CNP_STATUS CNP_CALL cnp_unique_v2(
    const CnpArray *arr, bool return_index, bool return_inverse,
    bool return_counts, CnpArray **results, int result_capacity) {
    int required = 1 + (return_index ? 1 : 0) +
                   (return_inverse ? 1 : 0) + (return_counts ? 1 : 0);
    if (!results) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_unique_v2",
                      "results must not be NULL");
        return CNP_ERR_GENERIC;
    }
    if (result_capacity < required) {
        for (int index = 0; index < result_capacity; ++index)
            results[index] = NULL;
        cnp_set_error(CNP_ERR_SHAPE, "cnp_unique_v2",
                      "result capacity %d is smaller than required %d",
                      result_capacity, required);
        return CNP_ERR_SHAPE;
    }
    for (int index = 0; index < result_capacity; ++index)
        results[index] = NULL;
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_unique_v2",
                      "arr must not be NULL");
        return CNP_ERR_GENERIC;
    }
    if (!set_dtype_is_supported(arr->dtype->type_num)) {
        cnp_set_error(CNP_ERR_TYPE, "cnp_unique_v2",
                      "Sorting dtype %d is not supported",
                      (int)arr->dtype->type_num);
        return CNP_ERR_TYPE;
    }

    CnpArray *flat = cnp_flatten(arr, CNP_ORDER_C);
    if (!flat) {
        cnp_relabel_error("cnp_unique_v2");
        return cnp_get_error(NULL);
    }
    if (!return_index && !return_inverse && !return_counts) {
        CnpArray *values = unique_values_only(flat, "cnp_unique_v2");
        cnp_array_free(flat);
        if (!values) return cnp_get_error(NULL);
        results[0] = values;
        return CNP_OK;
    }
    CNP_SORT_KIND kind = return_index
        ? CNP_SORT_STABLE : CNP_SORT_QUICKSORT;
    CnpArray *order = cnp_argsort_v2(flat, 0, false, kind);
    if (!order) {
        cnp_array_free(flat);
        cnp_relabel_error("cnp_unique_v2");
        return cnp_get_error(NULL);
    }

    int64_t count = flat->size;
    const char *flat_data = (const char*)flat->data + flat->offset;
    const int64_t *sorted_indices =
        (const int64_t*)((const char*)order->data + order->offset);
    int64_t unique_count = count > 0 ? 1 : 0;
    for (int64_t position = 1; position < count; ++position) {
        if (!set_values_equal(
                flat_data, sorted_indices[position - 1],
                sorted_indices[position], flat->dtype->type_num,
                flat->dtype->elsize, true)) {
            ++unique_count;
        }
    }

    CnpArray *created[4] = {NULL, NULL, NULL, NULL};
    int created_count = 0;
    int64_t unique_shape[1] = {unique_count};
    int64_t input_shape[1] = {count};
    created[created_count++] = cnp_array_new(
        1, unique_shape, flat->dtype->type_num, CNP_ORDER_C);
    if (return_index)
        created[created_count++] = cnp_array_new(
            1, unique_shape, CNP_LONGLONG, CNP_ORDER_C);
    if (return_inverse)
        created[created_count++] = cnp_array_new(
            1, input_shape, CNP_LONGLONG, CNP_ORDER_C);
    if (return_counts)
        created[created_count++] = cnp_array_new(
            1, unique_shape, CNP_LONGLONG, CNP_ORDER_C);
    for (int index = 0; index < created_count; ++index) {
        if (!created[index]) {
            release_arrays(created, created_count);
            cnp_array_free(order);
            cnp_array_free(flat);
            cnp_relabel_error("cnp_unique_v2");
            return cnp_get_error(NULL);
        }
    }

    CnpArray *index_result = NULL;
    CnpArray *inverse_result = NULL;
    CnpArray *counts_result = NULL;
    int optional_index = 1;
    if (return_index) index_result = created[optional_index++];
    if (return_inverse) inverse_result = created[optional_index++];
    if (return_counts) counts_result = created[optional_index++];

    int64_t group = 0;
    int64_t start = 0;
    for (int64_t position = 1; position <= count; ++position) {
        bool boundary = position == count;
        if (!boundary) {
            boundary = !set_values_equal(
                flat_data, sorted_indices[position - 1],
                sorted_indices[position], flat->dtype->type_num,
                flat->dtype->elsize, true);
        }
        if (!boundary) continue;

        int64_t representative = sorted_indices[start];
        memcpy(
            (char*)created[0]->data + group * created[0]->dtype->elsize,
            flat_data + representative * flat->dtype->elsize,
            flat->dtype->elsize);
        if (index_result)
            ((int64_t*)index_result->data)[group] = representative;
        if (counts_result)
            ((int64_t*)counts_result->data)[group] = position - start;
        if (inverse_result) {
            for (int64_t member = start; member < position; ++member)
                ((int64_t*)inverse_result->data)[sorted_indices[member]] = group;
        }
        ++group;
        start = position;
    }

    cnp_array_free(order);
    cnp_array_free(flat);
    for (int index = 0; index < required; ++index) {
        results[index] = created[index];
        created[index] = NULL;
    }
    return CNP_OK;
}

/* =========================================================================
 * intersection and union
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_intersect1d(
    const CnpArray *ar1, const CnpArray *ar2, bool assume_unique) {
    CnpArray *sources[2] = {
        set_source(ar1, assume_unique, "cnp_intersect1d"), NULL
    };
    if (!sources[0]) return NULL;
    sources[1] = set_source(ar2, assume_unique, "cnp_intersect1d");
    if (!sources[1]) {
        release_arrays(sources, 2);
        return NULL;
    }

    CnpArray *left = NULL;
    CnpArray *right = NULL;
    CNP_TYPE dtype;
    if (!prepare_common_flat(
            sources[0], sources[1], &left, &right, &dtype,
            "cnp_intersect1d")) {
        release_arrays(sources, 2);
        return NULL;
    }
    release_arrays(sources, 2);
    CnpArray *combined = concatenate_flat(
        left, right, dtype, "cnp_intersect1d");
    cnp_array_free(left);
    cnp_array_free(right);
    if (!combined) return NULL;
    CnpArray *sorted = cnp_sort_v2(
        combined, 0, false, CNP_SORT_QUICKSORT);
    cnp_array_free(combined);
    if (!sorted) {
        cnp_relabel_error("cnp_intersect1d");
        return NULL;
    }

    const char *data = (const char*)sorted->data + sorted->offset;
    int64_t result_count = 0;
    for (int64_t index = 1; index < sorted->size; ++index) {
        if (set_values_equal(
                data, index - 1, index, dtype,
                sorted->dtype->elsize, false)) {
            ++result_count;
        }
    }
    CnpArray *result = cnp_array_new(
        1, &result_count, dtype, CNP_ORDER_C);
    if (!result) {
        cnp_array_free(sorted);
        cnp_relabel_error("cnp_intersect1d");
        return NULL;
    }
    int64_t destination = 0;
    for (int64_t index = 1; index < sorted->size; ++index) {
        if (set_values_equal(
                data, index - 1, index, dtype,
                sorted->dtype->elsize, false)) {
            memcpy((char*)result->data +
                       destination++ * result->dtype->elsize,
                   data + (index - 1) * sorted->dtype->elsize,
                   sorted->dtype->elsize);
        }
    }
    cnp_array_free(sorted);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_union1d(
    const CnpArray *ar1, const CnpArray *ar2) {
    CnpArray *left = NULL;
    CnpArray *right = NULL;
    CNP_TYPE dtype;
    if (!prepare_common_flat(
            ar1, ar2, &left, &right, &dtype, "cnp_union1d"))
        return NULL;
    CnpArray *combined = concatenate_flat(
        left, right, dtype, "cnp_union1d");
    cnp_array_free(left);
    cnp_array_free(right);
    if (!combined) return NULL;
    CnpArray *result = unique_values(combined, "cnp_union1d");
    cnp_array_free(combined);
    return result;
}

/* =========================================================================
 * difference and symmetric difference
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_setdiff1d(
    const CnpArray *ar1, const CnpArray *ar2, bool assume_unique) {
    CnpArray *left = set_source(ar1, assume_unique, "cnp_setdiff1d");
    if (!left) return NULL;
    CnpArray *right = set_source(ar2, assume_unique, "cnp_setdiff1d");
    if (!right) {
        cnp_array_free(left);
        return NULL;
    }
    CnpArray *mask = cnp_in1d(left, right, true, true);
    cnp_array_free(right);
    if (!mask) {
        cnp_array_free(left);
        cnp_relabel_error("cnp_setdiff1d");
        return NULL;
    }

    const int8_t *mask_data =
        (const int8_t*)((const char*)mask->data + mask->offset);
    int64_t result_count = 0;
    for (int64_t index = 0; index < mask->size; ++index)
        if (mask_data[index]) ++result_count;
    CnpArray *result = cnp_array_new(
        1, &result_count, left->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_array_free(mask);
        cnp_array_free(left);
        cnp_relabel_error("cnp_setdiff1d");
        return NULL;
    }
    const char *left_data = (const char*)left->data + left->offset;
    int64_t destination = 0;
    for (int64_t index = 0; index < mask->size; ++index) {
        if (!mask_data[index]) continue;
        memcpy((char*)result->data +
                   destination++ * result->dtype->elsize,
               left_data + index * left->dtype->elsize,
               left->dtype->elsize);
    }
    cnp_array_free(mask);
    cnp_array_free(left);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_setxor1d(
    const CnpArray *ar1, const CnpArray *ar2, bool assume_unique) {
    CnpArray *sources[2] = {
        set_source(ar1, assume_unique, "cnp_setxor1d"), NULL
    };
    if (!sources[0]) return NULL;
    sources[1] = set_source(ar2, assume_unique, "cnp_setxor1d");
    if (!sources[1]) {
        release_arrays(sources, 2);
        return NULL;
    }
    CnpArray *left = NULL;
    CnpArray *right = NULL;
    CNP_TYPE dtype;
    if (!prepare_common_flat(
            sources[0], sources[1], &left, &right, &dtype,
            "cnp_setxor1d")) {
        release_arrays(sources, 2);
        return NULL;
    }
    release_arrays(sources, 2);
    CnpArray *combined = concatenate_flat(
        left, right, dtype, "cnp_setxor1d");
    cnp_array_free(left);
    cnp_array_free(right);
    if (!combined) return NULL;
    CnpArray *sorted = cnp_sort_v2(
        combined, 0, false, CNP_SORT_QUICKSORT);
    cnp_array_free(combined);
    if (!sorted) {
        cnp_relabel_error("cnp_setxor1d");
        return NULL;
    }

    const char *data = (const char*)sorted->data + sorted->offset;
    int64_t result_count = 0;
    for (int64_t index = 0; index < sorted->size; ++index) {
        bool different_left = index == 0 || !set_values_equal(
            data, index - 1, index, dtype,
            sorted->dtype->elsize, false);
        bool different_right = index + 1 == sorted->size || !set_values_equal(
            data, index, index + 1, dtype,
            sorted->dtype->elsize, false);
        if (different_left && different_right) ++result_count;
    }
    CnpArray *result = cnp_array_new(
        1, &result_count, dtype, CNP_ORDER_C);
    if (!result) {
        cnp_array_free(sorted);
        cnp_relabel_error("cnp_setxor1d");
        return NULL;
    }
    int64_t destination = 0;
    for (int64_t index = 0; index < sorted->size; ++index) {
        bool different_left = index == 0 || !set_values_equal(
            data, index - 1, index, dtype,
            sorted->dtype->elsize, false);
        bool different_right = index + 1 == sorted->size || !set_values_equal(
            data, index, index + 1, dtype,
            sorted->dtype->elsize, false);
        if (!different_left || !different_right) continue;
        memcpy((char*)result->data +
                   destination++ * result->dtype->elsize,
               data + index * sorted->dtype->elsize,
               sorted->dtype->elsize);
    }
    cnp_array_free(sorted);
    return result;
}

/* =========================================================================
 * membership
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_in1d(
    const CnpArray *ar1, const CnpArray *ar2,
    bool assume_unique, bool invert) {
    (void)assume_unique;
    CnpArray *left = NULL;
    CnpArray *right = NULL;
    CNP_TYPE dtype;
    if (!prepare_common_flat(
            ar1, ar2, &left, &right, &dtype, "cnp_in1d"))
        return NULL;
    CnpArray *sorted_right = cnp_sort_v2(
        right, 0, false, CNP_SORT_QUICKSORT);
    if (!sorted_right) {
        cnp_array_free(left);
        cnp_array_free(right);
        cnp_relabel_error("cnp_in1d");
        return NULL;
    }
    cnp_array_free(right);
    right = NULL;
    int64_t result_size = left->size;
    CnpArray *result = cnp_array_new(
        1, &result_size, CNP_BOOL, CNP_ORDER_C);
    if (!result) {
        cnp_array_free(left);
        cnp_array_free(sorted_right);
        cnp_relabel_error("cnp_in1d");
        return NULL;
    }

    const char *left_data = (const char*)left->data + left->offset;
    const char *right_data =
        (const char*)sorted_right->data + sorted_right->offset;
    int8_t *result_data = (int8_t*)result->data;
    for (int64_t left_index = 0; left_index < left->size; ++left_index) {
        bool found = membership_sorted_contains(
            right_data, sorted_right->size,
            left_data + left_index * left->dtype->elsize,
            dtype, left->dtype->elsize);
        result_data[left_index] = (invert ? !found : found) ? 1 : 0;
    }
    cnp_array_free(left);
    cnp_array_free(sorted_right);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_isin(
    const CnpArray *element, const CnpArray *test_elements,
    bool assume_unique, bool invert) {
    if (!element || !test_elements) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_isin",
                      "input arrays must not be NULL");
        return NULL;
    }
    CnpArray *flat_result = cnp_in1d(
        element, test_elements, assume_unique, invert);
    if (!flat_result) {
        cnp_relabel_error("cnp_isin");
        return NULL;
    }
    CnpArray *result = cnp_reshape(
        flat_result, element->ndim, element->shape, CNP_ORDER_C);
    cnp_array_decref(flat_result);
    if (!result) cnp_relabel_error("cnp_isin");
    return result;
}
