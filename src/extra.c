/**
 * cnumpy extra utility functions - select, piecewise, digitize,
 * bincount, unravel_index, ravel_multi_index, triu_indices, tril_indices
 * Corresponds to various numpy utility functions
 */
#include "../include/cnumpy/cnumpy_internal.h"

/* =========================================================================
 * digitize - Return indices of bins to which each value belongs
 * ========================================================================= */
static bool digitize_real_dtype_is_supported(CNP_TYPE dtype) {
    return dtype == CNP_BOOL || cnp_type_is_integer(dtype) ||
           cnp_type_is_float(dtype);
}

static int64_t digitize_flat_offset(
    const CnpArray *array, int64_t flat_index) {
    int64_t offset = array->offset;
    for (int dimension = array->ndim - 1; dimension >= 0; --dimension) {
        int64_t coordinate = flat_index % array->shape[dimension];
        flat_index /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

static CNP_STATUS digitize_bin_direction(
    const CnpArray *bins, int *direction, const char *function_name) {
    *direction = 1;
    if (bins->size < 2) return CNP_OK;

    bool increasing = true;
    bool decreasing = true;
    CNP_TYPE dtype = bins->dtype->type_num;
    if ((bins->flags & CNP_ARRAY_C_CONTIGUOUS) && dtype == CNP_DOUBLE) {
        const double *values = (const double*)(
            (const char*)bins->data + bins->offset);
        for (int64_t index = 1; index < bins->size; ++index) {
            int order = cnp_compare_numpy_doubles(
                values[index - 1], values[index]);
            if (order < 0) decreasing = false;
            if (order > 0) increasing = false;
            if (!increasing && !decreasing) {
                cnp_set_error(
                    CNP_ERR_VALUE, function_name,
                    "bins must be monotonically increasing or decreasing");
                return CNP_ERR_VALUE;
            }
        }
        *direction = increasing ? 1 : -1;
        return CNP_OK;
    }

    for (int64_t index = 1; index < bins->size; ++index) {
        int64_t previous_offset = digitize_flat_offset(bins, index - 1);
        int64_t current_offset = digitize_flat_offset(bins, index);
        const void *previous =
            (const char*)bins->data + previous_offset;
        const void *current =
            (const char*)bins->data + current_offset;
        int order = 0;
        CNP_STATUS status = cnp_compare_numeric_elements(
            previous, dtype, current, dtype,
            dtype, &order, function_name);
        if (status != CNP_OK) return status;
        if (order < 0) decreasing = false;
        if (order > 0) increasing = false;
        if (!increasing && !decreasing) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "bins must be monotonically increasing or decreasing");
            return CNP_ERR_VALUE;
        }
    }
    *direction = increasing ? 1 : -1;
    return CNP_OK;
}

static bool digitize_decreasing_contiguous_float64(
    const CnpArray *x, const CnpArray *bins,
    bool right, CnpArray **result_out,
    const char *function_name) {
    *result_out = NULL;
    if (!(x->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            !(bins->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            x->dtype->type_num != CNP_DOUBLE ||
            bins->dtype->type_num != CNP_DOUBLE) {
        return false;
    }

    CnpArray *result = cnp_array_new(
        x->ndim, x->shape, CNP_LONGLONG, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return true;
    }
    int64_t query_count = x->size;
    if (query_count == 0) {
        *result_out = result;
        return true;
    }

    int64_t bin_count = bins->size;
    const double *queries = (const double*)(
        (const char*)x->data + x->offset);
    const double *bin_values = bin_count > 0
        ? (const double*)((const char*)bins->data + bins->offset)
        : NULL;
    int64_t *indices = (int64_t*)result->data;
    for (int64_t index = 0; index < query_count; ++index) {
        double query = queries[index];
        int64_t low = 0;
        int64_t high = bin_count;
        while (low < high) {
            int64_t middle = low + (high - low) / 2;
            int64_t bin_index = bin_count - 1 - middle;
            int order = cnp_compare_numpy_doubles(
                bin_values[bin_index], query);
            if (order < 0 || (!right && order == 0)) low = middle + 1;
            else high = middle;
        }
        indices[index] = bin_count - low;
    }
    *result_out = result;
    return true;
}

CNP_API CnpArray* CNP_CALL cnp_digitize(
    const CnpArray *x, const CnpArray *bins, bool right) {
    const char *function_name = "cnp_digitize";
    if (!x) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "x array is NULL");
        return NULL;
    }
    if (!bins) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "bins array is NULL");
        return NULL;
    }
    if (bins->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "bins must be one-dimensional, got rank %d", bins->ndim);
        return NULL;
    }
    if (cnp_type_is_complex(x->dtype->type_num)) {
        cnp_set_error(CNP_ERR_TYPE, function_name, "x may not be complex");
        return NULL;
    }
    if (cnp_type_is_complex(bins->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "complex bins cannot be compared by digitize");
        return NULL;
    }
    if (!digitize_real_dtype_is_supported(x->dtype->type_num) ||
            !digitize_real_dtype_is_supported(bins->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "digitize requires boolean, integer, or real floating dtypes");
        return NULL;
    }

    int direction = 1;
    CNP_STATUS status = digitize_bin_direction(
        bins, &direction, function_name);
    if (status != CNP_OK) return NULL;

    if (direction < 0) {
        CnpArray *fast_result = NULL;
        if (digitize_decreasing_contiguous_float64(
                x, bins, right, &fast_result, function_name)) {
            return fast_result;
        }
    }

    const char *side = right ? "left" : "right";
    if (direction > 0) {
        CnpArray *result = cnp_searchsorted_v2(bins, x, side, NULL);
        if (!result) cnp_relabel_error(function_name);
        return result;
    }

    CnpSlice reverse = {0};
    reverse.step = -1;
    reverse.has_step = true;
    CnpArray *reversed_bins = cnp_array_slice(
        (CnpArray*)bins, 1, &reverse);
    if (!reversed_bins) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpArray *result = cnp_searchsorted_v2(
        reversed_bins, x, side, NULL);
    cnp_array_free(reversed_bins);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    int64_t bin_count = bins->size;
    int64_t *indices = (int64_t*)result->data;
    for (int64_t index = 0; index < result->size; ++index)
        indices[index] = bin_count - indices[index];
    return result;
}

/* =========================================================================
 * bincount - Count number of occurrences of each value
 * ========================================================================= */
static int64_t discrete_flat_offset(
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

static CNP_STATUS discrete_read_int64(
    const CnpArray *array, int64_t flat_index,
    int64_t *value, const char *function_name) {
    int64_t offset = discrete_flat_offset(array, flat_index);
    return cnp_cast_scalar_value(
        (const char*)array->data + offset,
        array->dtype->type_num,
        value, CNP_LONGLONG, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_bincount(
    const CnpArray *x, const CnpArray *weights, int64_t minlength) {
    const char *function_name = "cnp_bincount";
    if (!x) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "x array is NULL");
        return NULL;
    }
    if (x->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "x must be one-dimensional, got rank %d", x->ndim);
        return NULL;
    }
    CNP_TYPE x_type = x->dtype->type_num;
    if (!cnp_dtype_can_cast(x_type, CNP_LONGLONG, CNP_CAST_SAFE)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "x dtype %s cannot be safely cast to int64", x->dtype->name);
        return NULL;
    }
    if (minlength < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "minlength must not be negative");
        return NULL;
    }
    if (weights) {
        CNP_TYPE weights_type = weights->dtype->type_num;
        bool real_numeric = weights_type == CNP_BOOL ||
            cnp_type_is_integer(weights_type) ||
            cnp_type_is_float(weights_type);
        if (!real_numeric) {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "weights must have a real numeric dtype");
            return NULL;
        }
        if (weights->ndim != 1 || weights->size != x->size) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "weights and x must be one-dimensional arrays of equal length");
            return NULL;
        }
    }

    int64_t max_value = -1;
    for (int64_t index = 0; index < x->size; ++index) {
        int64_t value = 0;
        if (discrete_read_int64(
                x, index, &value, function_name) != CNP_OK)
            return NULL;
        if (value < 0) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "x must not contain negative values");
            return NULL;
        }
        if (value > max_value) max_value = value;
    }
    if (max_value == INT64_MAX) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "bincount output length exceeds int64");
        return NULL;
    }

    int64_t output_length = max_value + 1;
    if (minlength > output_length) output_length = minlength;
    int64_t shape[1] = {output_length};
    CNP_TYPE result_type = weights ? CNP_DOUBLE : CNP_LONGLONG;
    CnpArray *result = cnp_array_zeros(
        1, shape, result_type, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    for (int64_t index = 0; index < x->size; ++index) {
        int64_t bin = 0;
        CNP_STATUS status = discrete_read_int64(
            x, index, &bin, function_name);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        if (!weights) {
            ((int64_t*)result->data)[bin]++;
            continue;
        }
        int64_t weight_offset = discrete_flat_offset(weights, index);
        double weight = 0.0;
        status = cnp_cast_scalar_value(
            (const char*)weights->data + weight_offset,
            weights->dtype->type_num,
            &weight, CNP_DOUBLE, function_name);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        ((double*)result->data)[bin] += weight;
    }
    return result;
}

/* =========================================================================
 * select - Return array based on conditions and choices
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_select(int nconditions, const CnpArray **condlist,
                                       const CnpArray **choicelist, double default_val) {
    const char *function_name = "cnp_select";
    if (nconditions <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "at least one condition and choice are required");
        return NULL;
    }
    if (!condlist || !choicelist) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "condition and choice pointer lists are required");
        return NULL;
    }
    for (int condition = 0; condition < nconditions; ++condition) {
        if (!condlist[condition] || !choicelist[condition]) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "condition and choice %d are required", condition);
            return NULL;
        }
    }

    CnpArray *current = cnp_array_from_scalar(default_val, CNP_DOUBLE);
    if (!current) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int condition = nconditions - 1; condition >= 0; --condition) {
        CnpArray *next = cnp_where(
            condlist[condition], choicelist[condition], current);
        cnp_array_free(current);
        if (!next) {
            cnp_relabel_error(function_name);
            return NULL;
        }
        current = next;
    }
    return current;
}

/* =========================================================================
 * piecewise - Evaluate a piecewise-defined function
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_piecewise(const CnpArray *x, int nconditions,
                                          const CnpArray **condlist,
                                          double (*func)(double, void*), void *userdata) {
    const char *function_name = "cnp_piecewise";
    if (!x) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "x array is required");
        return NULL;
    }
    if (nconditions < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "condition count must be non-negative");
        return NULL;
    }
    if (nconditions > 0 && (!condlist || !func)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "condition list and callback are required");
        return NULL;
    }
    if (!(x->dtype->type_num == CNP_BOOL ||
          cnp_type_is_integer(x->dtype->type_num) ||
          cnp_type_is_float(x->dtype->type_num))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "x dtype %d cannot be represented by the real double callback",
            (int)x->dtype->type_num);
        return NULL;
    }
    for (int condition = 0; condition < nconditions; ++condition) {
        const CnpArray *current = condlist[condition];
        if (!current) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "condition %d is required", condition);
            return NULL;
        }
        bool scalar = current->ndim == 0;
        bool same_shape = current->ndim == x->ndim;
        for (int dimension = 0;
             same_shape && dimension < x->ndim; ++dimension) {
            same_shape = current->shape[dimension] == x->shape[dimension];
        }
        if (!scalar && !same_shape) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "condition %d must be scalar or have the exact x shape",
                condition);
            return NULL;
        }
    }

    CnpArray *result = cnp_array_zeros(
        x->ndim, x->shape, x->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    for (int condition = 0; condition < nconditions; ++condition) {
        const CnpArray *current = condlist[condition];
        for (int64_t index = 0; index < x->size; ++index) {
            int64_t condition_offset = current->ndim == 0
                ? current->offset : discrete_flat_offset(current, index);
            uint8_t truth = 0;
            CNP_STATUS status = cnp_cast_scalar_value(
                (const char*)current->data + condition_offset,
                current->dtype->type_num,
                &truth, CNP_BOOL, function_name);
            if (status != CNP_OK) {
                cnp_array_free(result);
                return NULL;
            }
            if (!truth) continue;
            double value = func(cnp_array_flat_get(x, index), userdata);
            cnp_set_element_double(
                result->data, index * result->dtype->elsize,
                result->dtype->type_num, value);
        }
    }
    return result;
}

/* =========================================================================
 * unravel_index - Convert flat index to multi-dimensional index
 * ========================================================================= */
static bool discrete_validate_index_shape(
    int ndim, const int64_t *shape, int64_t *total_size,
    const char *function_name) {
    if (ndim <= 0 || ndim > CNP_MAXDIMS) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "ndim must be in [1, %d], got %d", CNP_MAXDIMS, ndim);
        return false;
    }
    if (!shape) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "shape is NULL");
        return false;
    }
    int64_t product = 1;
    for (int dimension = 0; dimension < ndim; ++dimension) {
        if (shape[dimension] < 0) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "shape dimension %d must not be negative", dimension);
            return false;
        }
        if (shape[dimension] != 0 && product != 0 &&
                product > INT64_MAX / shape[dimension]) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "shape product exceeds int64");
            return false;
        }
        product *= shape[dimension];
    }
    *total_size = product;
    return true;
}

static bool discrete_index_dtype_is_supported(CNP_TYPE dtype) {
    return dtype == CNP_BOOL || cnp_type_is_integer(dtype);
}

CNP_API CnpArray* CNP_CALL cnp_unravel_index(
    const CnpArray *indices, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_unravel_index";
    if (!indices) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "indices array is NULL");
        return NULL;
    }
    if (!discrete_index_dtype_is_supported(indices->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "indices must have an integer or boolean dtype");
        return NULL;
    }
    int64_t total_size = 0;
    if (!discrete_validate_index_shape(
            ndim, shape, &total_size, function_name))
        return NULL;

    int64_t count = indices->size;
    int64_t output_shape[2] = {ndim, count};
    CnpArray *result = cnp_array_new(
        2, output_shape, CNP_LONGLONG, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    int64_t *output = (int64_t*)result->data;
    for (int64_t index = 0; index < count; ++index) {
        int64_t flat_index = 0;
        CNP_STATUS status = discrete_read_int64(
            indices, index, &flat_index, function_name);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        if (flat_index < 0 || flat_index >= total_size) {
            cnp_set_error(
                CNP_ERR_INDEX, function_name,
                "index %lld is outside [0, %lld)",
                (long long)flat_index, (long long)total_size);
            cnp_array_free(result);
            return NULL;
        }
        int64_t remaining = flat_index;
        for (int dimension = ndim - 1; dimension >= 0; --dimension) {
            output[(int64_t)dimension * count + index] =
                remaining % shape[dimension];
            remaining /= shape[dimension];
        }
    }
    return result;
}

/* =========================================================================
 * ravel_multi_index - Convert multi-dimensional index to flat index
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_ravel_multi_index(
    const CnpArray *multi_index, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_ravel_multi_index";
    if (!multi_index) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "multi_index array is NULL");
        return NULL;
    }
    if (!discrete_index_dtype_is_supported(
            multi_index->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "multi_index must have an integer or boolean dtype");
        return NULL;
    }
    int64_t total_size = 0;
    if (!discrete_validate_index_shape(
            ndim, shape, &total_size, function_name))
        return NULL;
    if (multi_index->ndim != 2 || multi_index->shape[0] != ndim) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "multi_index shape must be (ndim, count)");
        return NULL;
    }

    int64_t count = multi_index->shape[1];
    int64_t output_shape[1] = {count};
    CnpArray *result = cnp_array_new(
        1, output_shape, CNP_LONGLONG, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    int64_t *output = (int64_t*)result->data;
    for (int64_t index = 0; index < count; ++index) {
        int64_t flat_index = 0;
        int64_t stride = 1;
        for (int dimension = ndim - 1; dimension >= 0; --dimension) {
            int64_t coordinate = 0;
            CNP_STATUS status = discrete_read_int64(
                multi_index,
                (int64_t)dimension * count + index,
                &coordinate, function_name);
            if (status != CNP_OK) {
                cnp_array_free(result);
                return NULL;
            }
            if (coordinate < 0 || coordinate >= shape[dimension]) {
                cnp_set_error(
                    CNP_ERR_INDEX, function_name,
                    "coordinate %lld is out of bounds for axis %d with size %lld",
                    (long long)coordinate, dimension,
                    (long long)shape[dimension]);
                cnp_array_free(result);
                return NULL;
            }
            flat_index += coordinate * stride;
            stride *= shape[dimension];
        }
        output[index] = flat_index;
    }
    return result;
}

/* =========================================================================
 * triu_indices - Indices for upper triangle of (m, n) matrix
 * Returns array of shape (2, count) with [row_indices, col_indices]
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_triu_indices(int64_t n, int64_t k, int64_t m) {
    const char *function_name = "cnp_triu_indices";
    if (n < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name, "n must not be negative");
        return NULL;
    }
    if (m <= 0) m = n;

    /* Count elements first */
    int64_t count = 0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < m; j++) {
            if (j - i >= k) count++;
        }
    }

    int64_t shape[2] = {2, count};
    CnpArray *result = cnp_array_new(
        2, shape, CNP_INT, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int32_t *data = (int32_t*)result->data;
    int64_t idx = 0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < m; j++) {
            if (j - i >= k) {
                data[idx] = (int32_t)i;          /* row */
                data[count + idx] = (int32_t)j;  /* col */
                idx++;
            }
        }
    }
    return result;
}

/* =========================================================================
 * tril_indices - Indices for lower triangle of (m, n) matrix
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_tril_indices(int64_t n, int64_t k, int64_t m) {
    const char *function_name = "cnp_tril_indices";
    if (n < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name, "n must not be negative");
        return NULL;
    }
    if (m <= 0) m = n;

    int64_t count = 0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < m; j++) {
            if (j - i <= k) count++;
        }
    }

    int64_t shape[2] = {2, count};
    CnpArray *result = cnp_array_new(
        2, shape, CNP_INT, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int32_t *data = (int32_t*)result->data;
    int64_t idx = 0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < m; j++) {
            if (j - i <= k) {
                data[idx] = (int32_t)i;
                data[count + idx] = (int32_t)j;
                idx++;
            }
        }
    }
    return result;
}

/* =========================================================================
 * ediff1d - Differences between consecutive elements
 * ========================================================================= */
static CNP_STATUS ediff1d_subtract_element(
    const void *left, const void *right, void *destination,
    CNP_TYPE dtype, const char *function_name) {
#define CNP_EDIFF_INTEGER_SUBTRACT(unsigned_type) \
    do { \
        unsigned_type left_value = 0; \
        unsigned_type right_value = 0; \
        memcpy(&left_value, left, sizeof(left_value)); \
        memcpy(&right_value, right, sizeof(right_value)); \
        unsigned_type difference = \
            (unsigned_type)(left_value - right_value); \
        memcpy(destination, &difference, sizeof(difference)); \
        return CNP_OK; \
    } while (0)
    switch (dtype) {
        case CNP_BYTE:
        case CNP_UBYTE:
            CNP_EDIFF_INTEGER_SUBTRACT(uint8_t);
        case CNP_SHORT:
        case CNP_USHORT:
            CNP_EDIFF_INTEGER_SUBTRACT(uint16_t);
        case CNP_INT:
        case CNP_UINT:
            CNP_EDIFF_INTEGER_SUBTRACT(uint32_t);
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
            CNP_EDIFF_INTEGER_SUBTRACT(uint64_t);
        case CNP_HALF: {
            uint16_t left_bits = 0;
            uint16_t right_bits = 0;
            memcpy(&left_bits, left, sizeof(left_bits));
            memcpy(&right_bits, right, sizeof(right_bits));
            float difference = (float)cnp_half_to_float(left_bits) -
                (float)cnp_half_to_float(right_bits);
            uint16_t result_bits = cnp_float_to_half((double)difference);
            memcpy(destination, &result_bits, sizeof(result_bits));
            return CNP_OK;
        }
        case CNP_FLOAT: {
            float left_value = 0.0f;
            float right_value = 0.0f;
            memcpy(&left_value, left, sizeof(left_value));
            memcpy(&right_value, right, sizeof(right_value));
            float difference = left_value - right_value;
            memcpy(destination, &difference, sizeof(difference));
            return CNP_OK;
        }
        case CNP_DOUBLE: {
            double left_value = 0.0;
            double right_value = 0.0;
            memcpy(&left_value, left, sizeof(left_value));
            memcpy(&right_value, right, sizeof(right_value));
            double difference = left_value - right_value;
            memcpy(destination, &difference, sizeof(difference));
            return CNP_OK;
        }
        case CNP_LONGDOUBLE: {
            long double left_value = 0.0L;
            long double right_value = 0.0L;
            memcpy(&left_value, left, sizeof(left_value));
            memcpy(&right_value, right, sizeof(right_value));
            long double difference = left_value - right_value;
            memcpy(destination, &difference, sizeof(difference));
            return CNP_OK;
        }
        case CNP_CFLOAT: {
            cnp_cfloat left_value = {0};
            cnp_cfloat right_value = {0};
            memcpy(&left_value, left, sizeof(left_value));
            memcpy(&right_value, right, sizeof(right_value));
            cnp_cfloat difference = {
                left_value.real - right_value.real,
                left_value.imag - right_value.imag};
            memcpy(destination, &difference, sizeof(difference));
            return CNP_OK;
        }
        case CNP_CDOUBLE: {
            cnp_cdouble left_value = {0};
            cnp_cdouble right_value = {0};
            memcpy(&left_value, left, sizeof(left_value));
            memcpy(&right_value, right, sizeof(right_value));
            cnp_cdouble difference = {
                left_value.real - right_value.real,
                left_value.imag - right_value.imag};
            memcpy(destination, &difference, sizeof(difference));
            return CNP_OK;
        }
        case CNP_CLONGDOUBLE: {
            cnp_clongdouble left_value = {0};
            cnp_clongdouble right_value = {0};
            memcpy(&left_value, left, sizeof(left_value));
            memcpy(&right_value, right, sizeof(right_value));
            cnp_clongdouble difference = {
                left_value.real - right_value.real,
                left_value.imag - right_value.imag};
            memcpy(destination, &difference, sizeof(difference));
            return CNP_OK;
        }
        default:
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "array dtype does not support subtraction");
            return CNP_ERR_TYPE;
    }
#undef CNP_EDIFF_INTEGER_SUBTRACT
}

CNP_API CnpArray* CNP_CALL cnp_ediff1d(
    const CnpArray *arr, double to_begin, double to_end,
    bool has_begin, bool has_end) {
    const char *function_name = "cnp_ediff1d";
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "array is NULL");
        return NULL;
    }
    CNP_TYPE dtype = arr->dtype->type_num;
    bool supported = cnp_type_is_integer(dtype) ||
        cnp_type_is_float(dtype) || cnp_type_is_complex(dtype);
    if (!supported) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "array dtype %s does not support ediff1d", arr->dtype->name);
        return NULL;
    }
    if ((has_begin || has_end) &&
            !cnp_dtype_can_cast(
                CNP_DOUBLE, dtype, CNP_CAST_SAME_KIND)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "boundary values must be compatible with the array dtype under same_kind casting");
        return NULL;
    }

    int64_t difference_count = arr->size > 0 ? arr->size - 1 : 0;
    int64_t boundary_count = (has_begin ? 1 : 0) + (has_end ? 1 : 0);
    if (difference_count > INT64_MAX - boundary_count) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "ediff1d output length exceeds int64");
        return NULL;
    }
    int64_t output_length = difference_count + boundary_count;
    int64_t shape[1] = {output_length};
    CnpArray *result = cnp_array_new(
        1, shape, dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t output_index = 0;
    if (has_begin) {
        CNP_STATUS status = cnp_cast_scalar_value(
            &to_begin, CNP_DOUBLE,
            (char*)result->data + output_index * result->dtype->elsize,
            dtype, function_name);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        ++output_index;
    }
    for (int64_t index = 1; index < arr->size; ++index) {
        int64_t left_offset = discrete_flat_offset(arr, index);
        int64_t right_offset = discrete_flat_offset(arr, index - 1);
        CNP_STATUS status = ediff1d_subtract_element(
            (const char*)arr->data + left_offset,
            (const char*)arr->data + right_offset,
            (char*)result->data + output_index * result->dtype->elsize,
            dtype, function_name);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        ++output_index;
    }
    if (has_end) {
        CNP_STATUS status = cnp_cast_scalar_value(
            &to_end, CNP_DOUBLE,
            (char*)result->data + output_index * result->dtype->elsize,
            dtype, function_name);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
    }
    return result;
}

/* =========================================================================
 * trapz - Integrate along given axis using composite trapezoidal rule
 * ========================================================================= */
typedef struct {
    const CnpArray *array;
    bool one_dimensional;
    int axis;
    int64_t difference_length;
} CnpTrapzXTraversal;

typedef struct {
    CnpTrapzXTraversal x;
    int difference_ndim;
    int difference_axis;
    int64_t difference_shape[CNP_MAXDIMS];
    int product_ndim;
    int64_t product_shape[CNP_MAXDIMS];
    int reduction_axis;
    int result_ndim;
    int64_t result_shape[CNP_MAXDIMS];
    bool use_slice_path;
} CnpTrapzPlan;

typedef struct {
    CNP_TYPE dtype;
    uint64_t integer_bits;
    double real;
} CnpTrapzValue;

static bool trapz_real_numeric_dtype(
    const CnpArray *array, const char *argument,
    const char *function_name) {
    char kind = array->dtype->kind;
    if (kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f')
        return true;
    cnp_set_error(CNP_ERR_TYPE, function_name,
                  "%s must have a real numeric dtype", argument);
    return false;
}

static CnpTrapzValue trapz_integer_value(
    CNP_TYPE dtype, uint64_t bits) {
    CnpTrapzValue value = {dtype, bits, 0.0};
    return value;
}

static CnpTrapzValue trapz_real_value(
    CNP_TYPE dtype, double real) {
    CnpTrapzValue value = {dtype, 0, real};
    return value;
}

static CnpTrapzValue trapz_failed_value(void) {
    return trapz_real_value(CNP_NOTYPE, 0.0);
}

static CnpTrapzValue trapz_unsupported_value(
    const char *stage, CNP_TYPE dtype) {
    cnp_set_error(CNP_ERR_TYPE, "cnp_trapz",
                  "unsupported internal %s dtype %d", stage, (int)dtype);
    return trapz_failed_value();
}

static bool trapz_value_as_signed(
    CnpTrapzValue value, int64_t *converted) {
#define CNP_TRAPZ_SIGNED_VALUE(signed_type, unsigned_type) \
    do { \
        unsigned_type bits = (unsigned_type)value.integer_bits; \
        signed_type result; \
        memcpy(&result, &bits, sizeof(result)); \
        *converted = (int64_t)result; \
        return true; \
    } while (0)
    switch (value.dtype) {
        case CNP_BOOL:
            *converted = value.integer_bits != 0;
            return true;
        case CNP_BYTE:
            CNP_TRAPZ_SIGNED_VALUE(int8_t, uint8_t);
        case CNP_UBYTE:
            *converted = (int64_t)(uint8_t)value.integer_bits;
            return true;
        case CNP_SHORT:
            CNP_TRAPZ_SIGNED_VALUE(int16_t, uint16_t);
        case CNP_USHORT:
            *converted = (int64_t)(uint16_t)value.integer_bits;
            return true;
        case CNP_INT:
            CNP_TRAPZ_SIGNED_VALUE(int32_t, uint32_t);
        case CNP_UINT:
            *converted = (int64_t)(uint32_t)value.integer_bits;
            return true;
        case CNP_LONG:
        case CNP_LONGLONG:
            CNP_TRAPZ_SIGNED_VALUE(int64_t, uint64_t);
        case CNP_ULONG:
        case CNP_ULONGLONG: {
            uint64_t bits = value.integer_bits;
            int64_t result;
            memcpy(&result, &bits, sizeof(result));
            *converted = result;
            return true;
        }
        case CNP_HALF:
        case CNP_FLOAT:
        case CNP_DOUBLE:
        case CNP_LONGDOUBLE:
            *converted = (int64_t)value.real;
            return true;
        default:
            cnp_set_error(CNP_ERR_TYPE, "cnp_trapz",
                          "unsupported internal conversion dtype %d",
                          (int)value.dtype);
            return false;
    }
#undef CNP_TRAPZ_SIGNED_VALUE
}

static bool trapz_value_as_unsigned(
    CnpTrapzValue value, uint64_t *converted) {
    switch (value.dtype) {
        case CNP_BOOL:
            *converted = value.integer_bits != 0;
            return true;
        case CNP_BYTE:
        case CNP_SHORT:
        case CNP_INT:
        case CNP_LONG:
        case CNP_LONGLONG: {
            int64_t signed_value;
            if (!trapz_value_as_signed(value, &signed_value)) return false;
            *converted = (uint64_t)signed_value;
            return true;
        }
        case CNP_UBYTE:
            *converted = (uint8_t)value.integer_bits;
            return true;
        case CNP_USHORT:
            *converted = (uint16_t)value.integer_bits;
            return true;
        case CNP_UINT:
            *converted = (uint32_t)value.integer_bits;
            return true;
        case CNP_ULONG:
        case CNP_ULONGLONG:
            *converted = value.integer_bits;
            return true;
        case CNP_HALF:
        case CNP_FLOAT:
        case CNP_DOUBLE:
        case CNP_LONGDOUBLE:
            *converted = (uint64_t)value.real;
            return true;
        default:
            cnp_set_error(CNP_ERR_TYPE, "cnp_trapz",
                          "unsupported internal conversion dtype %d",
                          (int)value.dtype);
            return false;
    }
}

static bool trapz_value_as_double(
    CnpTrapzValue value, double *converted) {
    switch (value.dtype) {
        case CNP_BOOL:
        case CNP_BYTE:
        case CNP_SHORT:
        case CNP_INT:
        case CNP_LONG:
        case CNP_LONGLONG: {
            int64_t signed_value;
            if (!trapz_value_as_signed(value, &signed_value)) return false;
            *converted = (double)signed_value;
            return true;
        }
        case CNP_UBYTE:
        case CNP_USHORT:
        case CNP_UINT:
        case CNP_ULONG:
        case CNP_ULONGLONG: {
            uint64_t unsigned_value;
            if (!trapz_value_as_unsigned(value, &unsigned_value)) return false;
            *converted = (double)unsigned_value;
            return true;
        }
        case CNP_HALF:
        case CNP_FLOAT:
        case CNP_DOUBLE:
        case CNP_LONGDOUBLE:
            *converted = value.real;
            return true;
        default:
            cnp_set_error(CNP_ERR_TYPE, "cnp_trapz",
                          "unsupported internal conversion dtype %d",
                          (int)value.dtype);
            return false;
    }
}

static CnpTrapzValue trapz_pair_value(
    const CnpArray *array, int64_t left_offset,
    int64_t right_offset, bool difference) {
    const char *left = (const char*)array->data + left_offset;
    const char *right = (const char*)array->data + right_offset;
#define CNP_TRAPZ_UNSIGNED_PAIR(type) \
    do { \
        type left_value = *(const type*)left; \
        type right_value = *(const type*)right; \
        type result = difference \
            ? (type)(left_value - right_value) \
            : (type)(left_value + right_value); \
        return trapz_integer_value( \
            array->dtype->type_num, (uint64_t)result); \
    } while (0)
#define CNP_TRAPZ_SIGNED_PAIR(signed_type, unsigned_type) \
    do { \
        unsigned_type left_value = \
            (unsigned_type)*(const signed_type*)left; \
        unsigned_type right_value = \
            (unsigned_type)*(const signed_type*)right; \
        unsigned_type bits = difference \
            ? left_value - right_value : left_value + right_value; \
        return trapz_integer_value( \
            array->dtype->type_num, (uint64_t)bits); \
    } while (0)
    switch (array->dtype->type_num) {
        case CNP_BOOL: {
            bool left_value = *(const int8_t*)left != 0;
            bool right_value = *(const int8_t*)right != 0;
            return trapz_integer_value(
                CNP_BOOL,
                difference
                    ? (uint64_t)(left_value != right_value)
                    : (uint64_t)(left_value || right_value));
        }
        case CNP_BYTE:
            CNP_TRAPZ_SIGNED_PAIR(int8_t, uint8_t);
        case CNP_UBYTE:
            CNP_TRAPZ_UNSIGNED_PAIR(uint8_t);
        case CNP_SHORT:
            CNP_TRAPZ_SIGNED_PAIR(int16_t, uint16_t);
        case CNP_USHORT:
            CNP_TRAPZ_UNSIGNED_PAIR(uint16_t);
        case CNP_INT:
            CNP_TRAPZ_SIGNED_PAIR(int32_t, uint32_t);
        case CNP_UINT:
            CNP_TRAPZ_UNSIGNED_PAIR(uint32_t);
        case CNP_LONG:
        case CNP_LONGLONG:
            CNP_TRAPZ_SIGNED_PAIR(int64_t, uint64_t);
        case CNP_ULONG:
        case CNP_ULONGLONG:
            CNP_TRAPZ_UNSIGNED_PAIR(uint64_t);
        case CNP_HALF: {
            float left_value = (float)cnp_half_to_float(
                *(const uint16_t*)left);
            float right_value = (float)cnp_half_to_float(
                *(const uint16_t*)right);
            float value = difference
                ? left_value - right_value : left_value + right_value;
            uint16_t rounded = cnp_float_to_half((double)value);
            return trapz_real_value(
                CNP_HALF, cnp_half_to_float(rounded));
        }
        case CNP_FLOAT: {
            float left_value = *(const float*)left;
            float right_value = *(const float*)right;
            return trapz_real_value(
                CNP_FLOAT,
                (double)(difference
                    ? left_value - right_value
                    : left_value + right_value));
        }
        case CNP_DOUBLE:
            return trapz_real_value(
                CNP_DOUBLE,
                difference
                    ? *(const double*)left - *(const double*)right
                    : *(const double*)left + *(const double*)right);
        case CNP_LONGDOUBLE:
            return trapz_real_value(
                CNP_LONGDOUBLE,
                (double)(difference
                    ? *(const long double*)left - *(const long double*)right
                    : *(const long double*)left + *(const long double*)right));
        default:
            return trapz_unsupported_value(
                "pair arithmetic", array->dtype->type_num);
    }
#undef CNP_TRAPZ_SIGNED_PAIR
#undef CNP_TRAPZ_UNSIGNED_PAIR
}

static CNP_TYPE trapz_result_type(
    const CnpArray *y, const CnpArray *x, double dx) {
    CNP_TYPE y_type = y->dtype->type_num;
    CNP_TYPE product_type;
    if (x) {
        product_type = cnp_promote_type(
            y_type, x->dtype->type_num);
    } else if (y_type == CNP_HALF) {
        double magnitude = fabs(dx);
        product_type = !isfinite(dx) || magnitude <= 65504.0
            ? CNP_HALF : magnitude <= FLT_MAX ? CNP_FLOAT : CNP_DOUBLE;
    } else if (y_type == CNP_FLOAT) {
        product_type = !isfinite(dx) || fabs(dx) <= FLT_MAX
            ? CNP_FLOAT : CNP_DOUBLE;
    } else {
        product_type = y_type;
    }
    return cnp_type_is_float(product_type)
        ? product_type : CNP_DOUBLE;
}

static double trapz_round_real(double value, CNP_TYPE dtype) {
    if (dtype == CNP_HALF)
        return cnp_half_to_float(cnp_float_to_half(value));
    if (dtype == CNP_FLOAT) return (double)(float)value;
    if (dtype == CNP_LONGDOUBLE) return (double)(long double)value;
    return value;
}

static CnpTrapzValue trapz_multiply_typed(
    CnpTrapzValue left, CnpTrapzValue right, CNP_TYPE dtype) {
#define CNP_TRAPZ_UNSIGNED_PRODUCT(type) \
    do { \
        uint64_t left_value; \
        uint64_t right_value; \
        if (!trapz_value_as_unsigned(left, &left_value) || \
                !trapz_value_as_unsigned(right, &right_value)) \
            return trapz_failed_value(); \
        uint64_t bits = \
            (uint64_t)(type)left_value * \
            (uint64_t)(type)right_value; \
        return trapz_integer_value(dtype, (uint64_t)(type)bits); \
    } while (0)
#define CNP_TRAPZ_SIGNED_PRODUCT(signed_type, unsigned_type) \
    do { \
        int64_t left_value; \
        int64_t right_value; \
        if (!trapz_value_as_signed(left, &left_value) || \
                !trapz_value_as_signed(right, &right_value)) \
            return trapz_failed_value(); \
        uint64_t bits = \
            (uint64_t)(unsigned_type)(signed_type) \
                left_value * \
            (uint64_t)(unsigned_type)(signed_type) \
                right_value; \
        return trapz_integer_value( \
            dtype, (uint64_t)(unsigned_type)bits); \
    } while (0)
    switch (dtype) {
        case CNP_BOOL: {
            double left_value;
            double right_value;
            if (!trapz_value_as_double(left, &left_value) ||
                    !trapz_value_as_double(right, &right_value))
                return trapz_failed_value();
            return trapz_integer_value(
                CNP_BOOL,
                left_value != 0.0 && right_value != 0.0);
        }
        case CNP_BYTE:
            CNP_TRAPZ_SIGNED_PRODUCT(int8_t, uint8_t);
        case CNP_UBYTE:
            CNP_TRAPZ_UNSIGNED_PRODUCT(uint8_t);
        case CNP_SHORT:
            CNP_TRAPZ_SIGNED_PRODUCT(int16_t, uint16_t);
        case CNP_USHORT:
            CNP_TRAPZ_UNSIGNED_PRODUCT(uint16_t);
        case CNP_INT:
            CNP_TRAPZ_SIGNED_PRODUCT(int32_t, uint32_t);
        case CNP_UINT:
            CNP_TRAPZ_UNSIGNED_PRODUCT(uint32_t);
        case CNP_LONG:
        case CNP_LONGLONG:
            CNP_TRAPZ_SIGNED_PRODUCT(int64_t, uint64_t);
        case CNP_ULONG:
        case CNP_ULONGLONG:
            CNP_TRAPZ_UNSIGNED_PRODUCT(uint64_t);
        case CNP_HALF:
        case CNP_FLOAT:
        case CNP_DOUBLE:
        case CNP_LONGDOUBLE: {
            double left_value;
            double right_value;
            if (!trapz_value_as_double(left, &left_value) ||
                    !trapz_value_as_double(right, &right_value))
                return trapz_failed_value();
            return trapz_real_value(
                dtype,
                trapz_round_real(
                    left_value * right_value,
                    dtype));
        }
        default:
            return trapz_unsupported_value("product", dtype);
    }
#undef CNP_TRAPZ_SIGNED_PRODUCT
#undef CNP_TRAPZ_UNSIGNED_PRODUCT
}

static bool trapz_accumulate_panel(
    double total, CnpTrapzValue width, CnpTrapzValue pair_sum,
    CNP_TYPE product_type, CNP_TYPE output_type,
    double *next_total) {
    CnpTrapzValue product = trapz_multiply_typed(
        width, pair_sum, product_type);
    if (product.dtype == CNP_NOTYPE) return false;
    double product_value;
    if (!trapz_value_as_double(product, &product_value)) return false;
    double panel = trapz_round_real(
        product_value / 2.0, output_type);
    *next_total = trapz_round_real(total + panel, output_type);
    return true;
}

static bool trapz_resolve_x_axis(
    const CnpArray *x, int axis, int *resolved_axis) {
    const char *function_name = "cnp_trapz";
    if (x->ndim == 0) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "x must be at least one-dimensional");
        return false;
    }
    int normalized = axis;
    if (normalized < 0) normalized += x->ndim;
    if (normalized < 0 || normalized >= x->ndim) {
        cnp_set_error(CNP_ERR_AXIS, function_name,
                      "axis %d is out of bounds for x of dimension %d",
                      axis, x->ndim);
        return false;
    }
    *resolved_axis = normalized;
    return true;
}

static bool trapz_broadcast_dimension(
    int64_t y_length, int64_t x_length, int dimension,
    int64_t *result_length) {
    if (y_length == x_length || x_length == 1) {
        *result_length = y_length;
        return true;
    }
    if (y_length == 1) {
        *result_length = x_length;
        return true;
    }
    cnp_set_error(CNP_ERR_BROADCAST, "cnp_trapz",
                  "x differences cannot be broadcast to y integration slices at dimension %d",
                  dimension);
    return false;
}

static bool trapz_prepare_plan(
    const CnpArray *y, const CnpArray *x, int original_axis,
    int y_axis, CnpTrapzPlan *plan) {
    int64_t y_pair_shape[CNP_MAXDIMS];
    memcpy(y_pair_shape, y->shape, sizeof(int64_t) * y->ndim);
    y_pair_shape[y_axis] = y->shape[y_axis] > 0
        ? y->shape[y_axis] - 1 : 0;

    memset(plan, 0, sizeof(*plan));
    plan->x.array = x;
    if (!x) {
        plan->product_ndim = y->ndim;
        memcpy(plan->product_shape, y_pair_shape,
               sizeof(int64_t) * y->ndim);
    } else if (x->ndim == 1) {
        plan->x.one_dimensional = true;
        plan->x.axis = 0;
        plan->x.difference_length = x->shape[0] > 0
            ? x->shape[0] - 1 : 0;
        plan->difference_ndim = y->ndim;
        plan->difference_axis = y_axis;
        for (int dimension = 0; dimension < y->ndim; ++dimension)
            plan->difference_shape[dimension] = 1;
        plan->difference_shape[y_axis] = plan->x.difference_length;
        plan->product_ndim = y->ndim;
    } else {
        int x_axis;
        if (!trapz_resolve_x_axis(x, original_axis, &x_axis))
            return false;
        plan->x.axis = x_axis;
        plan->x.difference_length = x->shape[x_axis] > 0
            ? x->shape[x_axis] - 1 : 0;
        plan->difference_ndim = x->ndim;
        plan->difference_axis = x_axis;
        memcpy(plan->difference_shape, x->shape,
               sizeof(int64_t) * x->ndim);
        plan->difference_shape[x_axis] = plan->x.difference_length;
        plan->product_ndim = y->ndim > x->ndim ? y->ndim : x->ndim;
    }

    if (x) {
        int y_offset = plan->product_ndim - y->ndim;
        int x_offset = plan->product_ndim - plan->difference_ndim;
        for (int dimension = 0;
             dimension < plan->product_ndim; ++dimension) {
            int64_t y_length = dimension < y_offset
                ? 1 : y_pair_shape[dimension - y_offset];
            int64_t x_length = dimension < x_offset
                ? 1 : plan->difference_shape[dimension - x_offset];
            if (!trapz_broadcast_dimension(
                    y_length, x_length, dimension,
                    &plan->product_shape[dimension]))
                return false;
        }
    }

    int reduction_axis = original_axis;
    if (reduction_axis < 0) reduction_axis += plan->product_ndim;
    if (reduction_axis < 0 || reduction_axis >= plan->product_ndim) {
        cnp_set_error(CNP_ERR_AXIS, "cnp_trapz",
                      "axis %d is out of bounds for broadcast result of dimension %d",
                      original_axis, plan->product_ndim);
        return false;
    }
    plan->reduction_axis = reduction_axis;
    plan->result_ndim = plan->product_ndim - 1;
    int result_dimension = 0;
    for (int dimension = 0;
         dimension < plan->product_ndim; ++dimension) {
        if (dimension != reduction_axis)
            plan->result_shape[result_dimension++] =
                plan->product_shape[dimension];
    }
    if (plan->result_ndim == 0) plan->result_shape[0] = 1;

    plan->use_slice_path =
        plan->product_ndim == y->ndim && reduction_axis == y_axis;
    for (int dimension = 0;
         plan->use_slice_path && dimension < y->ndim; ++dimension) {
        if (plan->product_shape[dimension] != y_pair_shape[dimension])
            plan->use_slice_path = false;
    }
    if (plan->use_slice_path && x && !plan->x.one_dimensional &&
            (x->ndim != y->ndim || plan->x.axis != y_axis))
        plan->use_slice_path = false;
    return true;
}

static void trapz_product_coordinates(
    const CnpTrapzPlan *plan, int64_t output_index,
    int64_t *coordinates) {
    memset(coordinates, 0, sizeof(int64_t) * CNP_MAXDIMS);
    for (int dimension = plan->product_ndim - 1;
         dimension >= 0; --dimension) {
        if (dimension == plan->reduction_axis) continue;
        int64_t length = plan->product_shape[dimension];
        if (length > 0) {
            coordinates[dimension] = output_index % length;
            output_index /= length;
        }
    }
}

static void trapz_y_pair_offsets(
    const CnpArray *y, int y_axis, const CnpTrapzPlan *plan,
    const int64_t *product_coordinates,
    int64_t *lower_offset, int64_t *upper_offset) {
    int y_offset = plan->product_ndim - y->ndim;
    *lower_offset = y->offset;
    *upper_offset = y->offset;
    for (int dimension = 0; dimension < y->ndim; ++dimension) {
        int64_t pair_length = dimension == y_axis
            ? (y->shape[dimension] > 0 ? y->shape[dimension] - 1 : 0)
            : y->shape[dimension];
        int64_t coordinate = pair_length == 1
            ? 0 : product_coordinates[y_offset + dimension];
        *lower_offset += coordinate * y->strides[dimension];
        *upper_offset += (coordinate +
            (dimension == y_axis ? 1 : 0)) * y->strides[dimension];
    }
}

static CnpTrapzValue trapz_product_x_difference(
    const CnpTrapzPlan *plan, const int64_t *product_coordinates) {
    const CnpArray *x = plan->x.array;
    if (plan->x.one_dimensional) {
        int x_offset = plan->product_ndim - plan->difference_ndim;
        int product_axis = x_offset + plan->difference_axis;
        int64_t coordinate = plan->x.difference_length == 1
            ? 0 : product_coordinates[product_axis];
        int64_t lower_offset = x->offset + coordinate * x->strides[0];
        int64_t upper_offset = lower_offset + x->strides[0];
        return trapz_pair_value(
            x, upper_offset, lower_offset, true);
    }

    int x_offset = plan->product_ndim - x->ndim;
    int64_t lower_offset = x->offset;
    int64_t upper_offset = x->offset;
    for (int dimension = 0; dimension < x->ndim; ++dimension) {
        int64_t difference_length = plan->difference_shape[dimension];
        int64_t coordinate = difference_length == 1
            ? 0 : product_coordinates[x_offset + dimension];
        lower_offset += coordinate * x->strides[dimension];
        upper_offset += (coordinate +
            (dimension == plan->x.axis ? 1 : 0)) * x->strides[dimension];
    }
    return trapz_pair_value(x, upper_offset, lower_offset, true);
}

static bool trapz_fill_broadcast_result(
    CnpArray *result, const CnpArray *y, const CnpArray *x,
    double dx, int y_axis, const CnpTrapzPlan *plan) {
    int64_t coordinates[CNP_MAXDIMS];
    int64_t reduction_length =
        plan->product_shape[plan->reduction_axis];
    CNP_TYPE product_type = x
        ? cnp_promote_type(y->dtype->type_num, x->dtype->type_num)
        : result->dtype->type_num;
    for (int64_t output_index = 0;
         output_index < result->size; ++output_index) {
        trapz_product_coordinates(plan, output_index, coordinates);
        double total = 0.0;
        for (int64_t item = 0; item < reduction_length; ++item) {
            coordinates[plan->reduction_axis] = item;
            int64_t lower_offset;
            int64_t upper_offset;
            trapz_y_pair_offsets(
                y, y_axis, plan, coordinates,
                &lower_offset, &upper_offset);
            CnpTrapzValue pair_sum = trapz_pair_value(
                y, lower_offset, upper_offset, false);
            CnpTrapzValue width = x
                ? trapz_product_x_difference(plan, coordinates)
                : trapz_real_value(CNP_DOUBLE, dx);
            if (!trapz_accumulate_panel(
                total, width, pair_sum,
                product_type, result->dtype->type_num, &total))
                return false;
        }
        cnp_set_element_double(
            result->data,
            output_index * result->dtype->elsize,
            result->dtype->type_num, total);
    }
    return true;
}

static void trapz_slice_coordinates(
    const CnpReductionTraversal *traversal,
    int64_t outer, int64_t inner, int64_t *coordinates) {
    const CnpArray *array = traversal->array;
    memset(coordinates, 0, sizeof(int64_t) * CNP_MAXDIMS);
    for (int dimension = traversal->axis - 1;
         dimension >= 0; --dimension) {
        coordinates[dimension] = outer % array->shape[dimension];
        outer /= array->shape[dimension];
    }
    for (int dimension = array->ndim - 1;
         dimension > traversal->axis; --dimension) {
        coordinates[dimension] = inner % array->shape[dimension];
        inner /= array->shape[dimension];
    }
}

static CnpTrapzValue trapz_x_difference(
    const CnpTrapzXTraversal *traversal,
    const int64_t *y_coordinates, int64_t item) {
    const CnpArray *x = traversal->array;
    if (traversal->one_dimensional) {
        int64_t lower = traversal->difference_length == 1 ? 0 : item;
        int64_t lower_offset = x->offset + lower * x->strides[0];
        int64_t upper_offset = lower_offset + x->strides[0];
        return trapz_pair_value(
            x, upper_offset, lower_offset, true);
    }

    int64_t lower_offset = x->offset;
    int64_t upper_offset = x->offset;
    for (int dimension = 0; dimension < x->ndim; ++dimension) {
        int64_t coordinate;
        if (dimension == traversal->axis) {
            coordinate = traversal->difference_length == 1 ? 0 : item;
            lower_offset += coordinate * x->strides[dimension];
            upper_offset += (coordinate + 1) * x->strides[dimension];
        } else {
            coordinate = x->shape[dimension] == 1
                ? 0 : y_coordinates[dimension];
            lower_offset += coordinate * x->strides[dimension];
            upper_offset += coordinate * x->strides[dimension];
        }
    }
    return trapz_pair_value(x, upper_offset, lower_offset, true);
}

static bool trapz_contiguous_double_slices(
    CnpArray *result, const CnpArray *y,
    double dx, int64_t axis_length) {
    int64_t pair_count = axis_length > 0 ? axis_length - 1 : 0;
    if (pair_count == 0) {
        for (int64_t output_index = 0;
             output_index < result->size; ++output_index)
            ((double*)result->data)[output_index] = 0.0;
        return true;
    }
    if ((uint64_t)pair_count > SIZE_MAX / sizeof(double)) {
        cnp_set_error(CNP_ERR_MEMORY, "cnp_trapz",
                      "contiguous panel workspace exceeds addressable memory");
        return false;
    }

    size_t workspace_size = (size_t)pair_count * sizeof(double);
    double *panel_values = (double*)cnp_malloc(workspace_size);
    if (!panel_values) {
        cnp_set_error(CNP_ERR_MEMORY, "cnp_trapz",
                      "failed to allocate contiguous panel workspace");
        return false;
    }

    const double *values = (const double*)(
        (const char*)y->data + y->offset);
    for (int64_t output_index = 0;
         output_index < result->size; ++output_index) {
        const double *source = values + output_index * axis_length;
        for (int64_t item = 0; item < pair_count; ++item)
            panel_values[item] =
                (source[item] + source[item + 1]) * dx / 2.0;
        ((double*)result->data)[output_index] =
            cnp_reduction_sum_contiguous_double(
                panel_values, pair_count);
    }
    cnp_free(panel_values, workspace_size);
    return true;
}

CNP_API CnpArray* CNP_CALL cnp_trapz(
    const CnpArray *y, const CnpArray *x, double dx, int axis) {
    const char *function_name = "cnp_trapz";
    int resolved_axis;
    if (!cnp_reduction_resolve_axis_strict_scalar(
            y, axis, false, function_name, &resolved_axis))
        return NULL;
    if (!trapz_real_numeric_dtype(y, "y", function_name)) return NULL;

    CnpReductionTraversal y_traversal;
    cnp_reduction_traversal_init(y, resolved_axis, &y_traversal);
    int64_t pair_count = y_traversal.axis_length > 0
        ? y_traversal.axis_length - 1 : 0;
    if (x) {
        if (!trapz_real_numeric_dtype(x, "x", function_name)) return NULL;
    }
    CnpTrapzPlan plan;
    if (!trapz_prepare_plan(y, x, axis, resolved_axis, &plan)) return NULL;

    CNP_TYPE output_type = trapz_result_type(y, x, dx);
    CnpArray *result = cnp_array_new(
        plan.result_ndim, plan.result_shape,
        output_type, CNP_ORDER_C);
    if (!result) return NULL;

    bool contiguous_double_slices = plan.use_slice_path &&
        x == NULL &&
        y->dtype->type_num == CNP_DOUBLE &&
        output_type == CNP_DOUBLE &&
        (y->flags & CNP_ARRAY_C_CONTIGUOUS) != 0 &&
        resolved_axis == y->ndim - 1;
    if (contiguous_double_slices) {
        if (!trapz_contiguous_double_slices(
                result, y, dx, y_traversal.axis_length)) {
            cnp_array_free(result);
            return NULL;
        }
        return result;
    }

    if (!plan.use_slice_path) {
        if (!trapz_fill_broadcast_result(
                result, y, x, dx, resolved_axis, &plan)) {
            cnp_array_free(result);
            return NULL;
        }
        return result;
    }

    CnpTrapzXTraversal x_traversal = plan.x;
    CNP_TYPE product_type = x
        ? cnp_promote_type(y->dtype->type_num, x->dtype->type_num)
        : output_type;

    int64_t coordinates[CNP_MAXDIMS];
    for (int64_t outer = 0; outer < y_traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < y_traversal.inner; ++inner) {
            trapz_slice_coordinates(
                &y_traversal, outer, inner, coordinates);
            int64_t output_index = outer * y_traversal.inner + inner;
            double total = 0.0;
            for (int64_t item = 0; item < pair_count; ++item) {
                int64_t lower_offset = cnp_reduction_source_offset(
                    &y_traversal, outer, inner, item);
                int64_t upper_offset = cnp_reduction_source_offset(
                    &y_traversal, outer, inner, item + 1);
                CnpTrapzValue pair_sum = trapz_pair_value(
                    y, lower_offset, upper_offset, false);
                CnpTrapzValue width = x
                    ? trapz_x_difference(
                        &x_traversal, coordinates, item)
                    : trapz_real_value(CNP_DOUBLE, dx);
                if (!trapz_accumulate_panel(
                    total, width, pair_sum,
                    product_type, output_type, &total)) {
                    cnp_array_free(result);
                    return NULL;
                }
            }
            cnp_set_element_double(
                result->data,
                output_index * result->dtype->elsize,
                output_type, total);
        }
    }
    return result;
}

/* =========================================================================
 * sinc - Return sinc function: sin(pi*x)/(pi*x)
 * ========================================================================= */
static CNP_TYPE sinc_result_type(CNP_TYPE source_type) {
    if (source_type == CNP_BOOL || cnp_type_is_integer(source_type))
        return CNP_DOUBLE;
    if (cnp_type_is_float(source_type) ||
            cnp_type_is_complex(source_type))
        return source_type;
    return CNP_NOTYPE;
}

static cnp_clongdouble sinc_complex_value(cnp_clongdouble source) {
    const long double pi =
        3.141592653589793238462643383279502884L;
    long double real = pi * source.real;
    long double imag = pi * source.imag;
    cnp_clongdouble result = {1.0L, 0.0L};
    if (real == 0.0L && imag == 0.0L) return result;

    long double sine_real = sinl(real) * coshl(imag);
    long double sine_imag = cosl(real) * sinhl(imag);
    if (fabsl(real) >= fabsl(imag)) {
        long double ratio = imag / real;
        long double denominator = real + imag * ratio;
        result.real = (sine_real + sine_imag * ratio) / denominator;
        result.imag = (sine_imag - sine_real * ratio) / denominator;
    } else {
        long double ratio = real / imag;
        long double denominator = real * ratio + imag;
        result.real = (sine_real * ratio + sine_imag) / denominator;
        result.imag = (sine_imag * ratio - sine_real) / denominator;
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_sinc(const CnpArray *x) {
    const char *function_name = "cnp_sinc";
    if (!x) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "x array is NULL");
        return NULL;
    }
    CNP_TYPE result_type = sinc_result_type(x->dtype->type_num);
    if (result_type == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "x dtype %s does not support sinc", x->dtype->name);
        return NULL;
    }

    CnpArray *result = cnp_array_new(
        x->ndim, x->shape, result_type, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    const long double pi =
        3.141592653589793238462643383279502884L;
    for (int64_t index = 0; index < x->size; ++index) {
        int64_t source_offset = discrete_flat_offset(x, index);
        void *destination = (char*)result->data +
            index * result->dtype->elsize;
        CNP_STATUS status;
        if (cnp_type_is_complex(x->dtype->type_num)) {
            cnp_clongdouble source = {0};
            status = cnp_cast_scalar_value(
                (const char*)x->data + source_offset,
                x->dtype->type_num,
                &source, CNP_CLONGDOUBLE, function_name);
            if (status == CNP_OK) {
                cnp_clongdouble value = sinc_complex_value(source);
                status = cnp_cast_scalar_value(
                    &value, CNP_CLONGDOUBLE,
                    destination, result_type, function_name);
            }
        } else {
            long double source = 0.0L;
            status = cnp_cast_scalar_value(
                (const char*)x->data + source_offset,
                x->dtype->type_num,
                &source, CNP_LONGDOUBLE, function_name);
            if (status == CNP_OK) {
                long double scaled = pi * source;
                long double value = scaled == 0.0L
                    ? 1.0L : sinl(scaled) / scaled;
                status = cnp_cast_scalar_value(
                    &value, CNP_LONGDOUBLE,
                    destination, result_type, function_name);
            }
        }
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
    }
    return result;
}

/* =========================================================================
 * angle - Return the counterclockwise phase of each numeric element
 * numpy.angle(z, deg=False)
 * ========================================================================= */

static CNP_TYPE angle_result_type(CNP_TYPE source_type) {
    switch (source_type) {
        case CNP_BYTE:
        case CNP_UBYTE:
        case CNP_HALF:
            return CNP_HALF;
        case CNP_SHORT:
        case CNP_USHORT:
        case CNP_FLOAT:
        case CNP_CFLOAT:
            return CNP_FLOAT;
        case CNP_LONGDOUBLE:
        case CNP_CLONGDOUBLE:
            return CNP_LONGDOUBLE;
        case CNP_BOOL:
        case CNP_INT:
        case CNP_UINT:
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
        case CNP_DOUBLE:
        case CNP_CDOUBLE:
            return CNP_DOUBLE;
        default:
            return CNP_NOTYPE;
    }
}

static CNP_STATUS angle_validate_array(const CnpArray *array) {
    const char *function_name = "cnp_angle";
    if (!array) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return CNP_ERR_GENERIC;
    }
    if (!array->dtype ||
            angle_result_type(array->dtype->type_num) == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input array must have a numeric dtype");
        return CNP_ERR_TYPE;
    }
    if (array->ndim < 0 || array->ndim > CNP_MAXDIMS ||
            (array->ndim > 0 && (!array->shape || !array->strides))) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input array has invalid shape metadata");
        return CNP_ERR_SHAPE;
    }
    if (array->size > 0 && !array->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array requires a data buffer");
        return CNP_ERR_GENERIC;
    }
    return CNP_OK;
}

static void angle_write_half(
    void *destination, cnp_clongdouble value, bool degrees) {
    volatile float radians = atan2f(
        (float)value.imag, (float)value.real);
    uint16_t radians_bits = cnp_float_to_half((double)radians);
    if (!degrees) {
        memcpy(destination, &radians_bits, sizeof(radians_bits));
        return;
    }
    volatile float rounded_radians =
        (float)cnp_half_to_float(radians_bits);
    uint16_t scale_bits = cnp_float_to_half(57.29577951308232);
    volatile float rounded_scale =
        (float)cnp_half_to_float(scale_bits);
    volatile float result = rounded_radians * rounded_scale;
    uint16_t result_bits = cnp_float_to_half((double)result);
    memcpy(destination, &result_bits, sizeof(result_bits));
}

static void angle_write_value(
    void *destination, CNP_TYPE result_type,
    cnp_clongdouble value, bool degrees) {
    if (result_type == CNP_HALF) {
        angle_write_half(destination, value, degrees);
        return;
    }
    if (result_type == CNP_FLOAT) {
        volatile float result = atan2f(
            (float)value.imag, (float)value.real);
        if (degrees) {
            volatile float scale = (float)57.29577951308232;
            result = result * scale;
        }
        float stored = result;
        memcpy(destination, &stored, sizeof(stored));
        return;
    }
    if (result_type == CNP_LONGDOUBLE) {
        volatile long double result = atan2l(value.imag, value.real);
        if (degrees)
            result = result * (long double)57.29577951308232;
        long double stored = result;
        memcpy(destination, &stored, sizeof(stored));
        return;
    }
    volatile double result = atan2(
        (double)value.imag, (double)value.real);
    if (degrees) result = result * 57.29577951308232;
    double stored = result;
    memcpy(destination, &stored, sizeof(stored));
}

static bool angle_contiguous_real_float64(
    const CnpArray *source, CnpArray *result, bool degrees) {
    if (source->dtype->type_num != CNP_DOUBLE ||
            result->dtype->type_num != CNP_DOUBLE ||
            !(source->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            !(result->flags & CNP_ARRAY_C_CONTIGUOUS))
        return false;
    if (source->size == 0) return true;

    const char *source_data = (const char*)source->data + source->offset;
    char *result_data = (char*)result->data + result->offset;
    const uint64_t exponent_mask = 0x7ff0000000000000ULL;
    const uint64_t fraction_mask = 0x000fffffffffffffULL;
    const uint64_t sign_mask = 0x8000000000000000ULL;
    const uint64_t quiet_nan_mask = 0x0008000000000000ULL;
    const double negative_angle = degrees
        ? 180.0 : 3.14159265358979323846;

    for (int64_t index = 0; index < source->size; ++index) {
        uint64_t source_bits;
        memcpy(
            &source_bits,
            source_data + index * (int64_t)sizeof(double),
            sizeof(source_bits));
        uint64_t result_bits;
        if ((source_bits & exponent_mask) == exponent_mask &&
                (source_bits & fraction_mask) != 0) {
            result_bits = source_bits | quiet_nan_mask;
        } else {
            double value = (source_bits & sign_mask)
                ? negative_angle : 0.0;
            memcpy(&result_bits, &value, sizeof(result_bits));
        }
        memcpy(
            result_data + index * (int64_t)sizeof(double),
            &result_bits, sizeof(result_bits));
    }
    return true;
}

CNP_API CnpArray* CNP_CALL cnp_angle(const CnpArray *z, bool deg) {
    const char *function_name = "cnp_angle";
    if (angle_validate_array(z) != CNP_OK) return NULL;

    CNP_TYPE result_type = angle_result_type(z->dtype->type_num);
    CNP_ORDER result_order =
        (z->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(z->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        z->ndim, z->shape, result_type, result_order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (angle_contiguous_real_float64(z, result, deg))
        return result;

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < z->size; ++index) {
        int64_t source_offset = z->offset +
            cnp_multi_to_offset(z->ndim, coordinates, z->strides);
        int64_t destination_offset = result->offset +
            cnp_multi_to_offset(
                result->ndim, coordinates, result->strides);
        cnp_clongdouble value;
        CNP_STATUS status = cnp_cast_scalar_value(
            (const char*)z->data + source_offset,
            z->dtype->type_num,
            &value,
            CNP_CLONGDOUBLE,
            function_name);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            return NULL;
        }
        angle_write_value(
            (char*)result->data + destination_offset,
            result_type, value, deg);

        for (int axis = z->ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < z->shape[axis]) break;
            coordinates[axis] = 0;
        }
    }
    return result;
}

/* =========================================================================
 * real/imag - Return NumPy-compatible real and imaginary components
 * ========================================================================= */
static CNP_TYPE real_imag_component_type(CNP_TYPE source_type) {
    if (source_type == CNP_CFLOAT) return CNP_FLOAT;
    if (source_type == CNP_CDOUBLE) return CNP_DOUBLE;
    if (source_type == CNP_CLONGDOUBLE) return CNP_LONGDOUBLE;
    return CNP_NOTYPE;
}

static CnpArray *real_imag_complex_view(
    const CnpArray *source, bool imaginary, const char *function_name) {
    CNP_TYPE component_type = real_imag_component_type(
        source->dtype->type_num);
    if (component_type == CNP_NOTYPE) return NULL;

    CnpDtype *component_dtype = cnp_dtype_new(component_type);
    if (!component_dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "complex component dtype is unavailable");
        return NULL;
    }
    int64_t offset = source->offset +
        (imaginary ? component_dtype->elsize : 0);
    CnpArray *view = cnp_array_view_from_metadata(
        (CnpArray*)source, source->ndim, source->shape,
        source->strides, offset, 0);
    if (!view) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    view->dtype = component_dtype;
    view->flags &= ~(
        CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS);
    view->flags |= cnp_compute_layout_flags(
        view->ndim, view->shape, view->strides,
        component_dtype->elsize);
    return view;
}

static CNP_STATUS real_imag_validate_array(
    const CnpArray *source, const char *function_name) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return CNP_ERR_GENERIC;
    }
    CNP_TYPE source_type = source->dtype
        ? source->dtype->type_num : CNP_NOTYPE;
    if (angle_result_type(source_type) == CNP_NOTYPE &&
            source_type != CNP_DATETIME &&
            source_type != CNP_TIMEDELTA) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input array must have a numeric, datetime, or timedelta dtype");
        return CNP_ERR_TYPE;
    }
    if (source->ndim < 0 || source->ndim > CNP_MAXDIMS ||
            (source->ndim > 0 && (!source->shape || !source->strides))) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input array has invalid shape metadata");
        return CNP_ERR_SHAPE;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array requires a data buffer");
        return CNP_ERR_GENERIC;
    }
    return CNP_OK;
}

CNP_API CnpArray* CNP_CALL cnp_real(const CnpArray *z) {
    const char *function_name = "cnp_real";
    if (real_imag_validate_array(z, function_name) != CNP_OK)
        return NULL;

    if (cnp_type_is_complex(z->dtype->type_num))
        return real_imag_complex_view(z, false, function_name);
    cnp_array_incref((CnpArray*)z);
    return (CnpArray*)z;
}

CNP_API CnpArray* CNP_CALL cnp_imag(const CnpArray *z) {
    const char *function_name = "cnp_imag";
    if (real_imag_validate_array(z, function_name) != CNP_OK)
        return NULL;

    if (cnp_type_is_complex(z->dtype->type_num))
        return real_imag_complex_view(z, true, function_name);
    CNP_ORDER order =
        (z->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(z->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_zeros(
        z->ndim, z->shape, z->dtype->type_num, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    result->flags &= ~CNP_ARRAY_WRITEABLE;
    return result;
}

/* =========================================================================
 * convolve/correlate - NumPy 1.25 one-dimensional signal products
 * Public mode values retain the cnumpy ABI: 0=full, 1=same, 2=valid.
 * ========================================================================= */
typedef union {
    uint8_t boolean;
    int8_t signed_byte;
    uint8_t unsigned_byte;
    int16_t signed_short;
    uint16_t unsigned_short;
    int32_t signed_int;
    uint32_t unsigned_int;
    int64_t signed_longlong;
    uint64_t unsigned_longlong;
    uint16_t half;
    float floating;
    double double_precision;
    long double long_double;
    cnp_cfloat complex_float;
    cnp_cdouble complex_double;
    cnp_clongdouble complex_long_double;
} CnpSignalScalar;

static bool signal_dtype_is_numeric(CNP_TYPE dtype) {
    return dtype == CNP_BOOL || cnp_type_is_integer(dtype) ||
        cnp_type_is_float(dtype) || cnp_type_is_complex(dtype);
}

static const void *signal_element_pointer(
    const CnpArray *array, int64_t index) {
    int64_t offset = array->offset;
    if (array->ndim == 1) offset += index * array->strides[0];
    return (const char*)array->data + offset;
}

static CNP_STATUS signal_read_promoted(
    const CnpArray *array,
    int64_t index,
    CNP_TYPE result_dtype,
    CnpSignalScalar *value,
    const char *function_name) {
    const void *source = signal_element_pointer(array, index);
    memset(value, 0, sizeof(*value));
    if (array->dtype->type_num == result_dtype) {
        memcpy(value, source, (size_t)array->dtype->elsize);
        return CNP_OK;
    }
    return cnp_cast_scalar_value(
        source, array->dtype->type_num,
        value, result_dtype, function_name);
}

static int64_t signal_kernel_index(
    int64_t logical_index, int64_t kernel_length, bool reverse_kernel) {
    return reverse_kernel
        ? kernel_length - logical_index - 1
        : logical_index;
}

static bool signal_dot_contiguous_float64(
    const CnpArray *data,
    int64_t data_start,
    const CnpArray *kernel,
    int64_t kernel_start,
    int64_t kernel_length,
    int64_t count,
    bool reverse_kernel,
    CNP_TYPE result_dtype,
    void *output) {
    if (result_dtype != CNP_DOUBLE ||
            data->dtype->type_num != CNP_DOUBLE ||
            kernel->dtype->type_num != CNP_DOUBLE ||
            !(data->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            !(kernel->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            (data->ndim == 1 &&
             data->strides[0] != (int64_t)sizeof(double)) ||
            (kernel->ndim == 1 &&
             kernel->strides[0] != (int64_t)sizeof(double)))
        return false;

    const double *data_values = (const double*)(
        (const char*)data->data + data->offset) + data_start;
    int64_t physical_kernel_start = signal_kernel_index(
        kernel_start, kernel_length, reverse_kernel);
    const double *kernel_values = (const double*)(
        (const char*)kernel->data + kernel->offset) +
        physical_kernel_start;
    int64_t kernel_step = reverse_kernel ? -1 : 1;
    double sum = 0.0;
    if (count == 8) {
        sum = sum + data_values[0] * kernel_values[0 * kernel_step];
        sum = sum + data_values[1] * kernel_values[1 * kernel_step];
        sum = sum + data_values[2] * kernel_values[2 * kernel_step];
        sum = sum + data_values[3] * kernel_values[3 * kernel_step];
        sum = sum + data_values[4] * kernel_values[4 * kernel_step];
        sum = sum + data_values[5] * kernel_values[5 * kernel_step];
        sum = sum + data_values[6] * kernel_values[6 * kernel_step];
        sum = sum + data_values[7] * kernel_values[7 * kernel_step];
    } else {
        for (int64_t index = 0; index < count; ++index) {
            sum = sum + data_values[index] *
                kernel_values[index * kernel_step];
        }
    }
    double result = sum;
    memcpy(output, &result, sizeof(result));
    return true;
}

#pragma float_control(precise, on, push)
static bool signal_central_contiguous_float64(
    const CnpArray *data,
    int64_t data_start,
    const CnpArray *kernel,
    int64_t kernel_start,
    int64_t kernel_length,
    bool reverse_kernel,
    CNP_TYPE result_dtype,
    CnpArray *result,
    int64_t output_start,
    int64_t output_count,
    bool reverse_output) {
    if (kernel_length != 8 || result_dtype != CNP_DOUBLE ||
            data->dtype->type_num != CNP_DOUBLE ||
            kernel->dtype->type_num != CNP_DOUBLE ||
            !(data->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            !(kernel->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            data->strides[0] != (int64_t)sizeof(double) ||
            kernel->strides[0] != (int64_t)sizeof(double))
        return false;

    const double *data_values = (const double*)(
        (const char*)data->data + data->offset) + data_start;
    int64_t physical_kernel_start = signal_kernel_index(
        kernel_start, kernel_length, reverse_kernel);
    const double *kernel_values = (const double*)(
        (const char*)kernel->data + kernel->offset) +
        physical_kernel_start;
    int64_t kernel_step = reverse_kernel ? -1 : 1;
    const double kernel_0 = kernel_values[0 * kernel_step];
    const double kernel_1 = kernel_values[1 * kernel_step];
    const double kernel_2 = kernel_values[2 * kernel_step];
    const double kernel_3 = kernel_values[3 * kernel_step];
    const double kernel_4 = kernel_values[4 * kernel_step];
    const double kernel_5 = kernel_values[5 * kernel_step];
    const double kernel_6 = kernel_values[6 * kernel_step];
    const double kernel_7 = kernel_values[7 * kernel_step];
    double *output_values = (double*)result->data;
    for (int64_t index = 0; index < output_count; ++index) {
        double sum = 0.0;
        sum += data_values[index + 0] * kernel_0;
        sum += data_values[index + 1] * kernel_1;
        sum += data_values[index + 2] * kernel_2;
        sum += data_values[index + 3] * kernel_3;
        sum += data_values[index + 4] * kernel_4;
        sum += data_values[index + 5] * kernel_5;
        sum += data_values[index + 6] * kernel_6;
        sum += data_values[index + 7] * kernel_7;
        int64_t destination_index = reverse_output
            ? result->size - output_start - index - 1
            : output_start + index;
        output_values[destination_index] = sum;
    }
    return true;
}
#pragma float_control(pop)

static CNP_STATUS signal_dot(
    const CnpArray *data,
    int64_t data_start,
    const CnpArray *kernel,
    int64_t kernel_start,
    int64_t kernel_length,
    int64_t count,
    bool reverse_kernel,
    bool conjugate_data,
    bool conjugate_kernel,
    CNP_TYPE result_dtype,
    void *output,
    const char *function_name) {
    if (signal_dot_contiguous_float64(
            data, data_start, kernel, kernel_start,
            kernel_length, count, reverse_kernel,
            result_dtype, output))
        return CNP_OK;

    CnpSignalScalar left;
    CnpSignalScalar right;
    CNP_STATUS status;

#define SIGNAL_READ_PAIR(INDEX) \
    status = signal_read_promoted( \
        data, data_start + (INDEX), result_dtype, \
        &left, function_name); \
    if (status != CNP_OK) return status; \
    status = signal_read_promoted( \
        kernel, \
        signal_kernel_index( \
            kernel_start + (INDEX), kernel_length, reverse_kernel), \
        result_dtype, &right, function_name); \
    if (status != CNP_OK) return status

    switch (result_dtype) {
        case CNP_BOOL: {
            uint8_t sum = 0;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                if (left.boolean != 0 && right.boolean != 0) {
                    sum = 1;
                    break;
                }
            }
            memcpy(output, &sum, sizeof(sum));
            return CNP_OK;
        }
        case CNP_BYTE: {
            uint32_t sum = 0;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                sum += (uint32_t)(int32_t)left.signed_byte *
                    (uint32_t)(int32_t)right.signed_byte;
            }
            uint8_t result = (uint8_t)sum;
            memcpy(output, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_UBYTE: {
            uint32_t sum = 0;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                sum += (uint32_t)left.unsigned_byte *
                    (uint32_t)right.unsigned_byte;
            }
            uint8_t result = (uint8_t)sum;
            memcpy(output, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_SHORT: {
            uint32_t sum = 0;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                sum += (uint32_t)(int32_t)left.signed_short *
                    (uint32_t)(int32_t)right.signed_short;
            }
            uint16_t result = (uint16_t)sum;
            memcpy(output, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_USHORT: {
            uint32_t sum = 0;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                sum += (uint32_t)left.unsigned_short *
                    (uint32_t)right.unsigned_short;
            }
            uint16_t result = (uint16_t)sum;
            memcpy(output, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_INT: {
            uint32_t sum = 0;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                sum += (uint32_t)left.signed_int *
                    (uint32_t)right.signed_int;
            }
            memcpy(output, &sum, sizeof(sum));
            return CNP_OK;
        }
        case CNP_UINT: {
            uint32_t sum = 0;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                sum += left.unsigned_int * right.unsigned_int;
            }
            memcpy(output, &sum, sizeof(sum));
            return CNP_OK;
        }
        case CNP_LONG:
        case CNP_LONGLONG: {
            uint64_t sum = 0;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                sum += (uint64_t)left.signed_longlong *
                    (uint64_t)right.signed_longlong;
            }
            memcpy(output, &sum, sizeof(sum));
            return CNP_OK;
        }
        case CNP_ULONG:
        case CNP_ULONGLONG: {
            uint64_t sum = 0;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                sum += left.unsigned_longlong *
                    right.unsigned_longlong;
            }
            memcpy(output, &sum, sizeof(sum));
            return CNP_OK;
        }
        case CNP_HALF: {
            volatile float sum = 0.0f;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                float left_value = (float)cnp_half_to_float(left.half);
                float right_value = (float)cnp_half_to_float(right.half);
                sum = sum + left_value * right_value;
            }
            uint16_t result = cnp_float_to_half((double)sum);
            memcpy(output, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_FLOAT: {
            volatile float sum = 0.0f;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                sum = sum + left.floating * right.floating;
            }
            float result = sum;
            memcpy(output, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_DOUBLE: {
            volatile double sum = 0.0;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                sum = sum +
                    left.double_precision * right.double_precision;
            }
            double result = sum;
            memcpy(output, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_LONGDOUBLE: {
            volatile long double sum = 0.0L;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                sum = sum + left.long_double * right.long_double;
            }
            long double result = sum;
            memcpy(output, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_CFLOAT: {
            volatile float real_sum = 0.0f;
            volatile float imaginary_sum = 0.0f;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                float left_imaginary = conjugate_data
                    ? -left.complex_float.imag
                    : left.complex_float.imag;
                float right_imaginary = conjugate_kernel
                    ? -right.complex_float.imag
                    : right.complex_float.imag;
                real_sum = real_sum +
                    left.complex_float.real * right.complex_float.real -
                    left_imaginary * right_imaginary;
                imaginary_sum = imaginary_sum +
                    left.complex_float.real * right_imaginary +
                    left_imaginary * right.complex_float.real;
            }
            cnp_cfloat result = {real_sum, imaginary_sum};
            memcpy(output, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_CDOUBLE: {
            volatile double real_sum = 0.0;
            volatile double imaginary_sum = 0.0;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                double left_imaginary = conjugate_data
                    ? -left.complex_double.imag
                    : left.complex_double.imag;
                double right_imaginary = conjugate_kernel
                    ? -right.complex_double.imag
                    : right.complex_double.imag;
                real_sum = real_sum +
                    left.complex_double.real * right.complex_double.real -
                    left_imaginary * right_imaginary;
                imaginary_sum = imaginary_sum +
                    left.complex_double.real * right_imaginary +
                    left_imaginary * right.complex_double.real;
            }
            cnp_cdouble result = {real_sum, imaginary_sum};
            memcpy(output, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_CLONGDOUBLE: {
            volatile long double real_sum = 0.0L;
            volatile long double imaginary_sum = 0.0L;
            for (int64_t index = 0; index < count; ++index) {
                SIGNAL_READ_PAIR(index);
                long double left_imaginary = conjugate_data
                    ? -left.complex_long_double.imag
                    : left.complex_long_double.imag;
                long double right_imaginary = conjugate_kernel
                    ? -right.complex_long_double.imag
                    : right.complex_long_double.imag;
                real_sum = real_sum +
                    left.complex_long_double.real *
                        right.complex_long_double.real -
                    left_imaginary * right_imaginary;
                imaginary_sum = imaginary_sum +
                    left.complex_long_double.real * right_imaginary +
                    left_imaginary * right.complex_long_double.real;
            }
            cnp_clongdouble result = {real_sum, imaginary_sum};
            memcpy(output, &result, sizeof(result));
            return CNP_OK;
        }
        default:
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "result dtype %d does not provide a numeric dot product",
                (int)result_dtype);
            return CNP_ERR_TYPE;
    }

#undef SIGNAL_READ_PAIR
}

static CNP_STATUS signal_validate_array(
    const CnpArray *array,
    bool allow_scalar,
    const char *role,
    const char *function_name) {
    if (!array) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "%s array is required", role);
        return CNP_ERR_GENERIC;
    }
    if (!array->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "%s array requires a dtype", role);
        return CNP_ERR_TYPE;
    }
    if ((!allow_scalar && array->ndim != 1) ||
            (allow_scalar && array->ndim != 0 && array->ndim != 1)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "%s array must be one-dimensional%s",
            role, allow_scalar ? " or scalar" : "");
        return CNP_ERR_SHAPE;
    }
    if (!signal_dtype_is_numeric(array->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "%s array dtype must be numeric", role);
        return CNP_ERR_TYPE;
    }
    if (array->size > 0 && !array->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "%s array requires a data buffer", role);
        return CNP_ERR_GENERIC;
    }
    return CNP_OK;
}

static CnpArray *signal_product(
    const CnpArray *left,
    const CnpArray *right,
    int mode,
    bool convolution,
    const char *function_name) {
    CNP_STATUS status = signal_validate_array(
        left, convolution, "first", function_name);
    if (status != CNP_OK) return NULL;
    status = signal_validate_array(
        right, convolution, "second", function_name);
    if (status != CNP_OK) return NULL;
    if (mode < 0 || mode > 2) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "mode must be 0 (full), 1 (same), or 2 (valid)");
        return NULL;
    }

    int64_t left_length = left->ndim == 0 ? 1 : left->size;
    int64_t right_length = right->ndim == 0 ? 1 : right->size;
    const CnpArray *data;
    const CnpArray *kernel;
    int64_t data_length;
    int64_t kernel_length;
    bool reverse_kernel = false;
    bool conjugate_data = false;
    bool conjugate_kernel = false;
    bool reverse_output = false;

    if (convolution) {
        data = left;
        kernel = right;
        data_length = left_length;
        kernel_length = right_length;
        if (kernel_length > data_length) {
            const CnpArray *array_swap = data;
            data = kernel;
            kernel = array_swap;
            int64_t length_swap = data_length;
            data_length = kernel_length;
            kernel_length = length_swap;
        }
        if (data_length == 0) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name, "a cannot be empty");
            return NULL;
        }
        if (kernel_length == 0) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name, "v cannot be empty");
            return NULL;
        }
        reverse_kernel = true;
    } else {
        if (left_length == 0) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "first array argument cannot be empty");
            return NULL;
        }
        if (right_length == 0) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "second array argument cannot be empty");
            return NULL;
        }
        if (left_length >= right_length) {
            data = left;
            kernel = right;
            data_length = left_length;
            kernel_length = right_length;
            conjugate_kernel = true;
        } else {
            data = right;
            kernel = left;
            data_length = right_length;
            kernel_length = left_length;
            conjugate_data = true;
            reverse_output = true;
        }
    }

    CNP_TYPE result_dtype = cnp_promote_type_full(
        left->dtype->type_num, right->dtype->type_num);
    if (result_dtype == CNP_NOTYPE ||
            !signal_dtype_is_numeric(result_dtype)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input array dtypes do not have a common numeric dtype");
        return NULL;
    }

    int64_t result_length;
    int64_t left_overlap;
    int64_t right_overlap;
    if (mode == 0) {
        if (data_length > INT64_MAX - kernel_length + 1) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "full result length exceeds INT64_MAX");
            return NULL;
        }
        result_length = data_length + kernel_length - 1;
        left_overlap = kernel_length - 1;
        right_overlap = kernel_length - 1;
    } else if (mode == 1) {
        result_length = data_length;
        left_overlap = kernel_length / 2;
        right_overlap = kernel_length - left_overlap - 1;
    } else {
        result_length = data_length - kernel_length + 1;
        left_overlap = 0;
        right_overlap = 0;
    }

    int64_t shape[1] = {result_length};
    CnpArray *result = cnp_array_new(
        1, shape, result_dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t data_start = 0;
    int64_t kernel_start = left_overlap;
    int64_t count = kernel_length - left_overlap;
    int64_t output_index = 0;
    int64_t central_count = data_length - kernel_length + 1;
    for (int64_t index = 0; index < left_overlap; ++index) {
        int64_t destination_index = reverse_output
            ? result_length - output_index - 1
            : output_index;
        status = signal_dot(
            data, data_start, kernel, kernel_start,
            kernel_length, count, reverse_kernel,
            conjugate_data, conjugate_kernel, result_dtype,
            (char*)result->data +
                destination_index * result->dtype->elsize,
            function_name);
        if (status != CNP_OK) goto fail;
        ++output_index;
        ++count;
        --kernel_start;
    }
    if (signal_central_contiguous_float64(
            data, data_start, kernel, kernel_start,
            kernel_length, reverse_kernel, result_dtype,
            result, output_index, central_count, reverse_output)) {
        output_index += central_count;
        data_start += central_count;
        central_count = 0;
    }
    for (int64_t index = 0; index < central_count; ++index) {
        int64_t destination_index = reverse_output
            ? result_length - output_index - 1
            : output_index;
        status = signal_dot(
            data, data_start, kernel, kernel_start,
            kernel_length, count, reverse_kernel,
            conjugate_data, conjugate_kernel, result_dtype,
            (char*)result->data +
                destination_index * result->dtype->elsize,
            function_name);
        if (status != CNP_OK) goto fail;
        ++output_index;
        ++data_start;
    }
    for (int64_t index = 0; index < right_overlap; ++index) {
        --count;
        int64_t destination_index = reverse_output
            ? result_length - output_index - 1
            : output_index;
        status = signal_dot(
            data, data_start, kernel, kernel_start,
            kernel_length, count, reverse_kernel,
            conjugate_data, conjugate_kernel, result_dtype,
            (char*)result->data +
                destination_index * result->dtype->elsize,
            function_name);
        if (status != CNP_OK) goto fail;
        ++output_index;
        ++data_start;
    }
    return result;

fail:
    cnp_array_free(result);
    cnp_relabel_error(function_name);
    return NULL;
}

CNP_API CnpArray* CNP_CALL cnp_convolve(
    const CnpArray *a, const CnpArray *v, int mode) {
    return signal_product(a, v, mode, true, "cnp_convolve");
}

CNP_API CnpArray* CNP_CALL cnp_correlate(
    const CnpArray *a, const CnpArray *v, int mode) {
    return signal_product(a, v, mode, false, "cnp_correlate");
}
