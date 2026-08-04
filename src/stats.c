/**
 * cnumpy statistics functions - median, percentile, histogram, corrcoef, etc.
 */
#include "../include/cnumpy/cnumpy_internal.h"

/* =========================================================================
 * Median
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_median(const CnpArray *arr, int axis) {
    CnpArray *result = cnp_median_v2(
        arr, axis, axis == CNP_AXIS_NONE);
    if (!result) cnp_relabel_error("cnp_median");
    return result;
}

/* =========================================================================
 * Percentile and Quantile
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_percentile(const CnpArray *arr, double q, int axis) {
    CnpArray *result = cnp_percentile_v2(
        arr, q, axis, axis == CNP_AXIS_NONE);
    if (!result) cnp_relabel_error("cnp_percentile");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_quantile(const CnpArray *arr, double q, int axis) {
    CnpArray *result = cnp_quantile_v2(
        arr, q, axis, axis == CNP_AXIS_NONE);
    if (!result) cnp_relabel_error("cnp_quantile");
    return result;
}

/* =========================================================================
 * Histogram
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_histogram(const CnpArray *arr, int64_t bins, double range_min, double range_max) {
    const char *function_name = "cnp_histogram";
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            arr, function_name, &ignored_nbytes)) return NULL;
    if (arr->ndim < 0 || arr->ndim > CNP_MAXDIMS ||
            (arr->ndim > 0 && (!arr->shape || !arr->strides)) ||
            (arr->size > 0 && !arr->data)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source array shape metadata and data buffer must be valid");
        return NULL;
    }
    CNP_TYPE source_type = arr->dtype->type_num;
    if (!(source_type == CNP_BOOL ||
          cnp_type_is_integer(source_type) ||
          cnp_type_is_float(source_type))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array must have a real numeric dtype");
        return NULL;
    }
    if (bins <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "bin count must be positive");
        return NULL;
    }

    bool automatic_range = range_min == range_max;
    if (automatic_range) {
        if (arr->size == 0) {
            range_min = 0.0;
            range_max = 1.0;
        } else {
            range_min = cnp_array_flat_get(arr, 0);
            range_max = range_min;
            for (int64_t index = 0; index < arr->size; ++index) {
                double value = cnp_array_flat_get(arr, index);
                if (!isfinite(value)) {
                    cnp_set_error(
                        CNP_ERR_VALUE, function_name,
                        "autodetected range must be finite");
                    return NULL;
                }
                if (value < range_min) range_min = value;
                if (value > range_max) range_max = value;
            }
            if (range_min == range_max) {
                range_min -= 0.5;
                range_max += 0.5;
            }
        }
    } else if (!isfinite(range_min) || !isfinite(range_max) ||
            range_min > range_max) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "range endpoints must be finite and ordered");
        return NULL;
    }

    int64_t shape[1] = {bins};
    CnpArray *result = cnp_array_zeros(
        1, shape, CNP_LONGLONG, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    int64_t *counts = (int64_t*)result->data;
    double span = range_max - range_min;
    for (int64_t index = 0; index < arr->size; ++index) {
        double value = cnp_array_flat_get(arr, index);
        if (!(value >= range_min && value <= range_max)) continue;
        int64_t bin = value == range_max
            ? bins - 1
            : (int64_t)(((value - range_min) / span) * (double)bins);
        if (bin >= 0 && bin < bins) ++counts[bin];
    }
    return result;
}

/* =========================================================================
 * Correlation coefficient
 * ========================================================================= */
typedef struct {
    double real;
    double imag;
} CnpCovValue;

static bool cov_numeric_dtype(CNP_TYPE type) {
    return type == CNP_BOOL || cnp_type_is_integer(type) ||
        cnp_type_is_float(type) || cnp_type_is_complex(type);
}

static bool cov_validate_input(
        const CnpArray *array, const char *role,
        const char *function_name) {
    if (!array) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "%s array is required", role);
        return false;
    }
    if (!array->dtype || !cov_numeric_dtype(array->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "%s array must have a numeric dtype", role);
        return false;
    }
    if (array->ndim < 0 || array->ndim > 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "%s array must have at most two dimensions", role);
        return false;
    }
    return true;
}

static int64_t cov_variable_count(const CnpArray *array, bool rowvar) {
    if (array->ndim < 2) return 1;
    return rowvar ? array->shape[0] : array->shape[1];
}

static int64_t cov_observation_count(const CnpArray *array, bool rowvar) {
    if (array->ndim == 0) return 1;
    if (array->ndim == 1) return array->shape[0];
    return rowvar ? array->shape[1] : array->shape[0];
}

static CnpCovValue cov_read_value(
        const CnpArray *array, bool rowvar,
        int64_t variable, int64_t observation) {
    int64_t offset = array->offset;
    if (array->ndim == 1) {
        offset += observation * array->strides[0];
    } else if (array->ndim == 2) {
        offset += rowvar
            ? variable * array->strides[0] +
                observation * array->strides[1]
            : observation * array->strides[0] +
                variable * array->strides[1];
    }
    const char *pointer = (const char*)array->data + offset;
    switch (array->dtype->type_num) {
        case CNP_CFLOAT: {
            const cnp_cfloat *value = (const cnp_cfloat*)pointer;
            return (CnpCovValue){value->real, value->imag};
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *value = (const cnp_cdouble*)pointer;
            return (CnpCovValue){value->real, value->imag};
        }
        case CNP_CLONGDOUBLE: {
            const cnp_clongdouble *value = (const cnp_clongdouble*)pointer;
            return (CnpCovValue){(double)value->real, (double)value->imag};
        }
        default:
            return (CnpCovValue){cnp_get_element_double(
                array->data, offset, array->dtype->type_num), 0.0};
    }
}

static CnpCovValue cov_combined_value(
        const CnpArray *left, const CnpArray *right,
        bool rowvar, int64_t left_variables,
        int64_t variable, int64_t observation) {
    if (variable < left_variables) {
        return cov_read_value(left, rowvar, variable, observation);
    }
    return cov_read_value(
        right, rowvar, variable - left_variables, observation);
}

static CnpArray *covariance_execute(
        const CnpArray *m, const CnpArray *y,
        bool rowvar, int ddof, const char *function_name) {
    CnpCovValue *means = NULL;
    CnpArray *result = NULL;
    int64_t left_variables;
    int64_t right_variables = 0;
    int64_t observations;
    int64_t variables;
    CNP_TYPE result_type;

    if (!cov_validate_input(m, "primary", function_name)) return NULL;
    if (y && !cov_validate_input(y, "secondary", function_name)) return NULL;
    left_variables = cov_variable_count(m, rowvar);
    observations = cov_observation_count(m, rowvar);
    if (y) {
        right_variables = cov_variable_count(y, rowvar);
        int64_t right_observations = cov_observation_count(y, rowvar);
        if (right_observations != observations) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "primary and secondary observation counts differ: %lld and %lld",
                (long long)observations, (long long)right_observations);
            return NULL;
        }
    }
    if (right_variables > INT64_MAX - left_variables) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "combined variable count overflows int64");
        return NULL;
    }
    variables = left_variables + right_variables;
    result_type = cnp_type_is_complex(m->dtype->type_num) ||
            (y && cnp_type_is_complex(y->dtype->type_num))
        ? CNP_CDOUBLE : CNP_DOUBLE;

    if ((uint64_t)variables > SIZE_MAX / sizeof(CnpCovValue)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "covariance mean workspace is too large");
        return NULL;
    }
    if (variables != 0) {
        means = (CnpCovValue*)cnp_calloc(
            (size_t)variables, sizeof(CnpCovValue));
        if (!means) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "failed to allocate covariance mean workspace");
            return NULL;
        }
    }
    for (int64_t variable = 0; variable < variables; variable++) {
        for (int64_t observation = 0;
                observation < observations; observation++) {
            CnpCovValue value = cov_combined_value(
                m, y, rowvar, left_variables,
                variable, observation);
            means[variable].real += value.real;
            means[variable].imag += value.imag;
        }
        means[variable].real /= (double)observations;
        means[variable].imag /= (double)observations;
    }

    if (variables == 1) {
        result = cnp_array_new(0, NULL, result_type, CNP_ORDER_C);
    } else {
        int64_t shape[2] = {variables, variables};
        result = cnp_array_new(2, shape, result_type, CNP_ORDER_C);
    }
    if (!result) {
        if (means) {
            cnp_free(means, (size_t)variables * sizeof(CnpCovValue));
        }
        cnp_relabel_error(function_name);
        return NULL;
    }

    double divisor = (double)observations - (double)ddof;
    if (divisor <= 0.0) divisor = 0.0;
    for (int64_t left_variable = 0;
            left_variable < variables; left_variable++) {
        for (int64_t right_variable = 0;
                right_variable < variables; right_variable++) {
            CnpCovValue sum = {0.0, 0.0};
            for (int64_t observation = 0;
                    observation < observations; observation++) {
                CnpCovValue left = cov_combined_value(
                    m, y, rowvar, left_variables,
                    left_variable, observation);
                CnpCovValue right = cov_combined_value(
                    m, y, rowvar, left_variables,
                    right_variable, observation);
                left.real -= means[left_variable].real;
                left.imag -= means[left_variable].imag;
                right.real -= means[right_variable].real;
                right.imag -= means[right_variable].imag;
                sum.real += left.real * right.real + left.imag * right.imag;
                sum.imag += left.imag * right.real - left.real * right.imag;
            }
            int64_t index = variables == 1
                ? 0 : left_variable * variables + right_variable;
            if (result_type == CNP_CDOUBLE) {
                ((cnp_cdouble*)result->data)[index].real = sum.real / divisor;
                ((cnp_cdouble*)result->data)[index].imag = sum.imag / divisor;
            } else {
                ((double*)result->data)[index] = sum.real / divisor;
            }
        }
    }
    if (means) cnp_free(means, (size_t)variables * sizeof(CnpCovValue));
    return result;
}

static double corrcoef_clip_component(double value) {
    if (value > 1.0) return 1.0;
    if (value < -1.0) return -1.0;
    return value;
}

CNP_API CnpArray* CNP_CALL cnp_corrcoef(
        const CnpArray *x, const CnpArray *y) {
    const char *function_name = "cnp_corrcoef";
    CnpArray *result = covariance_execute(
        x, y, true, 1, function_name);
    if (!result) return NULL;
    if (result->ndim == 0) {
        if (result->dtype->type_num == CNP_CDOUBLE) {
            cnp_cdouble *value = (cnp_cdouble*)result->data;
            double denominator = hypot(value->real, value->imag);
            value->real /= denominator;
            value->imag /= denominator;
        } else {
            double *value = (double*)result->data;
            *value /= *value;
        }
        return result;
    }

    int64_t variables = result->shape[0];
    for (int64_t row = 0; row < variables; row++) {
        double row_variance = result->dtype->type_num == CNP_CDOUBLE
            ? ((cnp_cdouble*)result->data)[row * variables + row].real
            : ((double*)result->data)[row * variables + row];
        double row_std = sqrt(row_variance);
        for (int64_t column = 0; column < variables; column++) {
            if (column == row) continue;
            double column_variance = result->dtype->type_num == CNP_CDOUBLE
                ? ((cnp_cdouble*)result->data)[column * variables + column].real
                : ((double*)result->data)[column * variables + column];
            double scale = row_std * sqrt(column_variance);
            int64_t index = row * variables + column;
            if (result->dtype->type_num == CNP_CDOUBLE) {
                cnp_cdouble *value = &((cnp_cdouble*)result->data)[index];
                value->real = corrcoef_clip_component(value->real / scale);
                value->imag = corrcoef_clip_component(value->imag / scale);
            } else {
                double *value = &((double*)result->data)[index];
                *value = corrcoef_clip_component(*value / scale);
            }
        }
    }
    for (int64_t diagonal = 0; diagonal < variables; diagonal++) {
        int64_t index = diagonal * variables + diagonal;
        if (result->dtype->type_num == CNP_CDOUBLE) {
            cnp_cdouble *value = &((cnp_cdouble*)result->data)[index];
            double denominator = value->real;
            value->real /= denominator;
            value->imag /= denominator;
        } else {
            double *value = &((double*)result->data)[index];
            *value /= *value;
        }
    }
    return result;
}

/* =========================================================================
 * Covariance
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_cov(
        const CnpArray *m, const CnpArray *y, int rowvar, int ddof) {
    const char *function_name = "cnp_cov";
    if (rowvar != 0 && rowvar != 1) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "rowvar must be either 0 or 1");
        return NULL;
    }
    return covariance_execute(
        m, y, rowvar != 0, ddof, function_name);
}

/* =========================================================================
 * Weighted average
 * ========================================================================= */
static bool average_numeric_dtype(CNP_TYPE type) {
    switch (type) {
        case CNP_BOOL:
        case CNP_BYTE:
        case CNP_UBYTE:
        case CNP_SHORT:
        case CNP_USHORT:
        case CNP_INT:
        case CNP_UINT:
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
        case CNP_HALF:
        case CNP_FLOAT:
        case CNP_DOUBLE:
        case CNP_LONGDOUBLE:
        case CNP_CFLOAT:
        case CNP_CDOUBLE:
        case CNP_CLONGDOUBLE:
            return true;
        default:
            return false;
    }
}

static bool average_validate_array(
        const CnpArray *array, const char *role,
        const char *function_name) {
    if (!array) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "%s array is required", role);
        return false;
    }
    if (!array->dtype ||
            !average_numeric_dtype(array->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "%s array must have a numeric dtype", role);
        return false;
    }
    if (array->ndim < 0 || array->ndim > CNP_MAXDIMS ||
            (array->ndim > 0 &&
             (!array->shape || !array->strides))) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "%s array has invalid shape metadata", role);
        return false;
    }
    if (array->size < 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "%s array has invalid shape metadata", role);
        return false;
    }
    int64_t expected_size = 1;
    for (int dimension = 0; dimension < array->ndim; ++dimension) {
        int64_t extent = array->shape[dimension];
        if (extent < 0 ||
                (expected_size != 0 &&
                 extent > INT64_MAX / expected_size)) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "%s array has invalid shape metadata", role);
            return false;
        }
        expected_size *= extent;
    }
    if (expected_size != array->size) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "%s array size does not match its shape", role);
        return false;
    }
    if (array->size > 0 && !array->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "%s array requires a data buffer", role);
        return false;
    }
    return true;
}

static cnp_cfloat average_divide_cfloat(
        cnp_cfloat numerator, cnp_cfloat denominator) {
    cnp_cfloat result;
    if (denominator.imag == 0.0f) {
        result.real = numerator.real / denominator.real;
        result.imag = numerator.imag / denominator.real;
        return result;
    }
    if (fabsf(denominator.real) >= fabsf(denominator.imag)) {
        float ratio = denominator.imag / denominator.real;
        float scale = 1.0f + ratio * ratio;
        float real = numerator.real / denominator.real;
        float imag = numerator.imag / denominator.real;
        result.real = (real + imag * ratio) / scale;
        result.imag = (imag - real * ratio) / scale;
        return result;
    }
    float ratio = denominator.real / denominator.imag;
    float scale = 1.0f + ratio * ratio;
    float real = numerator.real / denominator.imag;
    float imag = numerator.imag / denominator.imag;
    result.real = (real * ratio + imag) / scale;
    result.imag = (imag * ratio - real) / scale;
    return result;
}

static cnp_cdouble average_divide_cdouble(
        cnp_cdouble numerator, cnp_cdouble denominator) {
    cnp_cdouble result;
    if (denominator.imag == 0.0) {
        result.real = numerator.real / denominator.real;
        result.imag = numerator.imag / denominator.real;
        return result;
    }
    if (fabs(denominator.real) >= fabs(denominator.imag)) {
        double ratio = denominator.imag / denominator.real;
        double scale = 1.0 + ratio * ratio;
        double real = numerator.real / denominator.real;
        double imag = numerator.imag / denominator.real;
        result.real = (real + imag * ratio) / scale;
        result.imag = (imag - real * ratio) / scale;
        return result;
    }
    double ratio = denominator.real / denominator.imag;
    double scale = 1.0 + ratio * ratio;
    double real = numerator.real / denominator.imag;
    double imag = numerator.imag / denominator.imag;
    result.real = (real * ratio + imag) / scale;
    result.imag = (imag * ratio - real) / scale;
    return result;
}

static cnp_clongdouble average_divide_clongdouble(
        cnp_clongdouble numerator,
        cnp_clongdouble denominator) {
    cnp_clongdouble result;
    if (denominator.imag == 0.0L) {
        result.real = numerator.real / denominator.real;
        result.imag = numerator.imag / denominator.real;
        return result;
    }
    if (fabsl(denominator.real) >= fabsl(denominator.imag)) {
        long double ratio = denominator.imag / denominator.real;
        long double scale = 1.0L + ratio * ratio;
        long double real = numerator.real / denominator.real;
        long double imag = numerator.imag / denominator.real;
        result.real = (real + imag * ratio) / scale;
        result.imag = (imag - real * ratio) / scale;
        return result;
    }
    long double ratio = denominator.real / denominator.imag;
    long double scale = 1.0L + ratio * ratio;
    long double real = numerator.real / denominator.imag;
    long double imag = numerator.imag / denominator.imag;
    result.real = (real * ratio + imag) / scale;
    result.imag = (imag * ratio - real) / scale;
    return result;
}

static CnpArray *average_real_result(
        const CnpArray *arr, const CnpArray *weights,
        int axis, bool axis_none, int resolved_axis,
        bool axis_weights, CNP_TYPE result_type,
        const char *function_name) {
    CnpArray *weights_view = NULL;
    CnpArray *source_cast = NULL;
    CnpArray *weights_cast = NULL;
    CnpArray *product = NULL;
    CnpArray *numerator = NULL;
    CnpArray *denominator = NULL;
    CnpArray *result = NULL;
    const CnpArray *product_weights = weights;

    if (axis_weights) {
        int64_t view_shape[CNP_MAXDIMS];
        int64_t view_strides[CNP_MAXDIMS];
        for (int dimension = 0; dimension < arr->ndim; ++dimension) {
            view_shape[dimension] = 1;
            view_strides[dimension] = 0;
        }
        view_shape[resolved_axis] = weights->shape[0];
        view_strides[resolved_axis] = weights->strides[0];
        weights_view = cnp_array_view_from_metadata(
            (CnpArray*)weights, arr->ndim,
            view_shape, view_strides, weights->offset, 0);
        if (!weights_view) goto cleanup;
        product_weights = weights_view;
    }

    const CnpArray *source_operand = arr;
    if (arr->dtype->type_num != result_type) {
        source_cast = cnp_astype(
            arr, result_type, CNP_CAST_UNSAFE);
        if (!source_cast) goto cleanup;
        source_operand = source_cast;
    }
    const CnpArray *weights_operand = product_weights;
    if (product_weights->dtype->type_num != result_type) {
        weights_cast = cnp_astype(
            product_weights, result_type, CNP_CAST_UNSAFE);
        if (!weights_cast) goto cleanup;
        weights_operand = weights_cast;
    }

    product = cnp_multiply(source_operand, weights_operand);
    if (!product) goto cleanup;
    if (product->dtype->type_num != result_type) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "weighted product did not preserve the resolved result dtype");
        goto cleanup;
    }
    numerator = cnp_sum_v2(
        product, axis, axis_none, result_type);
    if (!numerator) goto cleanup;
    denominator = axis_weights
        ? cnp_sum_v2(weights, 0, false, result_type)
        : cnp_sum_v2(weights, axis, axis_none, result_type);
    if (!denominator) goto cleanup;
    for (int64_t index = 0; index < denominator->size; ++index) {
        double value = cnp_get_element_double(
            denominator->data,
            denominator->offset + index * denominator->dtype->elsize,
            denominator->dtype->type_num);
        if (value == 0.0) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "weights sum to zero, cannot normalize");
            goto cleanup;
        }
    }
    result = cnp_divide(numerator, denominator);

cleanup:
    cnp_array_decref(denominator);
    cnp_array_decref(numerator);
    cnp_array_decref(product);
    cnp_array_decref(weights_cast);
    cnp_array_decref(source_cast);
    cnp_array_decref(weights_view);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_average(const CnpArray *arr, int axis, const CnpArray *weights) {
    CnpArray *result = cnp_average_v2(
        arr, axis, axis == CNP_AXIS_NONE, weights);
    if (!result) cnp_relabel_error("cnp_average");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_average_v2(
        const CnpArray *arr, int axis, bool axis_none,
        const CnpArray *weights) {
    const char *function_name = "cnp_average_v2";
    if (!average_validate_array(
            arr, "source", function_name)) return NULL;
    if (weights && !average_validate_array(
            weights, "weights", function_name)) return NULL;
    int resolved_axis;
    if (!cnp_reduction_resolve_axis(
            arr, axis, axis_none,
            function_name, &resolved_axis)) return NULL;
    if (!weights) {
        CnpArray *result = cnp_mean_v2(
            arr, axis, axis_none, CNP_NOTYPE);
        if (result && result->size == 0) {
            cnp_array_decref(result);
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "division by zero");
            return NULL;
        }
        if (!result) cnp_relabel_error(function_name);
        return result;
    }

    CnpReductionTraversal source_traversal;
    CnpReductionTraversal weights_traversal;
    cnp_reduction_traversal_init(
        arr, resolved_axis, &source_traversal);
    bool full_shape_weights = weights->ndim == arr->ndim;
    if (full_shape_weights) {
        for (int dimension = 0; dimension < arr->ndim; ++dimension) {
            if (weights->shape[dimension] != arr->shape[dimension]) {
                full_shape_weights = false;
                break;
            }
        }
    }
    bool axis_weights = !source_traversal.axis_none &&
        weights->ndim == 1 &&
        weights->shape[0] == source_traversal.axis_length;
    if (!full_shape_weights && !axis_weights) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            source_traversal.axis_none
                ? "weights must have the same shape as the source array"
                : "weights must match the source shape or selected axis");
        return NULL;
    }
    if (full_shape_weights)
        cnp_reduction_traversal_init(
            weights, resolved_axis, &weights_traversal);

    CNP_TYPE result_type = cnp_promote_type_full(
        arr->dtype->type_num, weights->dtype->type_num);
    if (arr->dtype->type_num == CNP_BOOL ||
            cnp_type_is_integer(arr->dtype->type_num))
        result_type = cnp_promote_type_full(result_type, CNP_DOUBLE);
    if (result_type == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source and weights do not have a common numeric dtype");
        return NULL;
    }
    if (result_type == CNP_FLOAT || result_type == CNP_DOUBLE)
        return average_real_result(
            arr, weights, axis, axis_none, resolved_axis,
            axis_weights, result_type, function_name);
    if (result_type != CNP_HALF &&
            result_type != CNP_LONGDOUBLE &&
            result_type != CNP_CFLOAT &&
            result_type != CNP_CDOUBLE &&
            result_type != CNP_CLONGDOUBLE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "resolved result dtype %d is not supported",
            (int)result_type);
        return NULL;
    }
    CnpArray *result = cnp_array_new(
        source_traversal.result_ndim,
        source_traversal.result_shape,
        result_type, CNP_ORDER_C);
    if (!result) return NULL;

    for (int64_t outer = 0;
         outer < source_traversal.outer; ++outer) {
        for (int64_t inner = 0;
             inner < source_traversal.inner; ++inner) {
            float weighted_sum_half = 0.0f;
            float weight_sum_half = 0.0f;
            cnp_cfloat weighted_sum_cfloat = {0.0f, 0.0f};
            cnp_cfloat weight_sum_cfloat = {0.0f, 0.0f};
            cnp_cdouble weighted_sum_cdouble = {0.0, 0.0};
            cnp_cdouble weight_sum_cdouble = {0.0, 0.0};
            cnp_clongdouble weighted_sum_clongdouble = {0.0L, 0.0L};
            cnp_clongdouble weight_sum_clongdouble = {0.0L, 0.0L};
            long double weighted_sum_longdouble = 0.0L;
            long double weight_sum_longdouble = 0.0L;
            for (int64_t item = 0;
                 item < source_traversal.axis_length; ++item) {
                int64_t source_offset = cnp_reduction_source_offset(
                    &source_traversal, outer, inner, item);
                int64_t weight_offset = axis_weights
                    ? weights->offset + item * weights->strides[0]
                    : cnp_reduction_source_offset(
                        &weights_traversal, outer, inner, item);
                if (result_type == CNP_CFLOAT) {
                    cnp_cfloat value;
                    cnp_cfloat weight;
                    CNP_STATUS status = cnp_cast_scalar_value(
                        (const char*)arr->data + source_offset,
                        arr->dtype->type_num,
                        &value, CNP_CFLOAT, function_name);
                    if (status == CNP_OK)
                        status = cnp_cast_scalar_value(
                            (const char*)weights->data + weight_offset,
                            weights->dtype->type_num,
                            &weight, CNP_CFLOAT, function_name);
                    if (status != CNP_OK) {
                        cnp_array_decref(result);
                        return NULL;
                    }
                    weighted_sum_cfloat.real +=
                        value.real * weight.real -
                        value.imag * weight.imag;
                    weighted_sum_cfloat.imag +=
                        value.real * weight.imag +
                        value.imag * weight.real;
                    weight_sum_cfloat.real += weight.real;
                    weight_sum_cfloat.imag += weight.imag;
                    continue;
                }
                if (result_type == CNP_HALF) {
                    float value;
                    float weight;
                    CNP_STATUS status = cnp_cast_scalar_value(
                        (const char*)arr->data + source_offset,
                        arr->dtype->type_num,
                        &value, CNP_FLOAT, function_name);
                    if (status == CNP_OK)
                        status = cnp_cast_scalar_value(
                            (const char*)weights->data + weight_offset,
                            weights->dtype->type_num,
                            &weight, CNP_FLOAT, function_name);
                    if (status != CNP_OK) {
                        cnp_array_decref(result);
                        return NULL;
                    }
                    uint16_t product = cnp_float_to_half(
                        (double)(value * weight));
                    weighted_sum_half +=
                        (float)cnp_half_to_float(product);
                    weight_sum_half += weight;
                    continue;
                }
                if (result_type == CNP_CDOUBLE) {
                    cnp_cdouble value;
                    cnp_cdouble weight;
                    CNP_STATUS status = cnp_cast_scalar_value(
                        (const char*)arr->data + source_offset,
                        arr->dtype->type_num,
                        &value, CNP_CDOUBLE, function_name);
                    if (status == CNP_OK)
                        status = cnp_cast_scalar_value(
                            (const char*)weights->data + weight_offset,
                            weights->dtype->type_num,
                            &weight, CNP_CDOUBLE, function_name);
                    if (status != CNP_OK) {
                        cnp_array_decref(result);
                        return NULL;
                    }
                    weighted_sum_cdouble.real +=
                        value.real * weight.real -
                        value.imag * weight.imag;
                    weighted_sum_cdouble.imag +=
                        value.real * weight.imag +
                        value.imag * weight.real;
                    weight_sum_cdouble.real += weight.real;
                    weight_sum_cdouble.imag += weight.imag;
                    continue;
                }
                if (result_type == CNP_CLONGDOUBLE) {
                    cnp_clongdouble value;
                    cnp_clongdouble weight;
                    CNP_STATUS status = cnp_cast_scalar_value(
                        (const char*)arr->data + source_offset,
                        arr->dtype->type_num,
                        &value, CNP_CLONGDOUBLE, function_name);
                    if (status == CNP_OK)
                        status = cnp_cast_scalar_value(
                            (const char*)weights->data + weight_offset,
                            weights->dtype->type_num,
                            &weight, CNP_CLONGDOUBLE, function_name);
                    if (status != CNP_OK) {
                        cnp_array_decref(result);
                        return NULL;
                    }
                    weighted_sum_clongdouble.real +=
                        value.real * weight.real -
                        value.imag * weight.imag;
                    weighted_sum_clongdouble.imag +=
                        value.real * weight.imag +
                        value.imag * weight.real;
                    weight_sum_clongdouble.real += weight.real;
                    weight_sum_clongdouble.imag += weight.imag;
                    continue;
                }
                if (result_type == CNP_LONGDOUBLE) {
                    long double value;
                    long double weight;
                    CNP_STATUS status = cnp_cast_scalar_value(
                        (const char*)arr->data + source_offset,
                        arr->dtype->type_num,
                        &value, CNP_LONGDOUBLE, function_name);
                    if (status == CNP_OK)
                        status = cnp_cast_scalar_value(
                            (const char*)weights->data + weight_offset,
                            weights->dtype->type_num,
                            &weight, CNP_LONGDOUBLE, function_name);
                    if (status != CNP_OK) {
                        cnp_array_decref(result);
                        return NULL;
                    }
                    weighted_sum_longdouble += value * weight;
                    weight_sum_longdouble += weight;
                    continue;
                }
            }
            bool zero_weight = result_type == CNP_CFLOAT
                ? weight_sum_cfloat.real == 0.0f &&
                    weight_sum_cfloat.imag == 0.0f
                : result_type == CNP_HALF
                ? weight_sum_half == 0.0f
                : result_type == CNP_CDOUBLE
                ? weight_sum_cdouble.real == 0.0 &&
                    weight_sum_cdouble.imag == 0.0
                : result_type == CNP_CLONGDOUBLE
                ? weight_sum_clongdouble.real == 0.0L &&
                    weight_sum_clongdouble.imag == 0.0L
                : result_type == CNP_LONGDOUBLE
                ? weight_sum_longdouble == 0.0L
                : false;
            if (zero_weight) {
                cnp_array_decref(result);
                cnp_set_error(
                    CNP_ERR_GENERIC, function_name,
                    "weights sum to zero, cannot normalize");
                return NULL;
            }
            int64_t output_index =
                outer * source_traversal.inner + inner;
            if (result_type == CNP_CFLOAT) {
                cnp_cfloat *output =
                    &((cnp_cfloat*)result->data)[output_index];
                *output = average_divide_cfloat(
                    weighted_sum_cfloat, weight_sum_cfloat);
            } else if (result_type == CNP_HALF) {
                ((uint16_t*)result->data)[output_index] =
                    cnp_float_to_half(
                        (double)(weighted_sum_half / weight_sum_half));
            } else if (result_type == CNP_CDOUBLE) {
                cnp_cdouble *output =
                    &((cnp_cdouble*)result->data)[output_index];
                *output = average_divide_cdouble(
                    weighted_sum_cdouble, weight_sum_cdouble);
            } else if (result_type == CNP_CLONGDOUBLE) {
                cnp_clongdouble *output =
                    &((cnp_clongdouble*)result->data)[output_index];
                *output = average_divide_clongdouble(
                    weighted_sum_clongdouble,
                    weight_sum_clongdouble);
            } else if (result_type == CNP_LONGDOUBLE) {
                ((long double*)result->data)[output_index] =
                    weighted_sum_longdouble / weight_sum_longdouble;
            }
        }
    }
    return result;
}

/* =========================================================================
 * Diff (n-th discrete difference)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_diff(const CnpArray *arr, int n, int axis) {
    const char *function_name = "cnp_diff";
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            arr, function_name, &ignored_nbytes)) return NULL;
    if (arr->ndim <= 0 || arr->ndim > CNP_MAXDIMS ||
            !arr->shape || !arr->strides ||
            (arr->size > 0 && !arr->data)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source must be a valid array with at least one dimension");
        return NULL;
    }
    if (!(arr->dtype->type_num == CNP_BOOL ||
          cnp_type_is_integer(arr->dtype->type_num) ||
          cnp_type_is_float(arr->dtype->type_num) ||
          cnp_type_is_complex(arr->dtype->type_num))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array must have a represented numeric dtype");
        return NULL;
    }
    if (n < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "order must be non-negative");
        return NULL;
    }
    int normalized_axis = axis < 0 ? axis + arr->ndim : axis;
    if (normalized_axis < 0 || normalized_axis >= arr->ndim) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "axis %d is out of bounds for array of dimension %d",
            axis, arr->ndim);
        return NULL;
    }

    CnpArray *current = cnp_array_copy(arr);
    if (!current) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int iteration = 0; iteration < n; ++iteration) {
        if (current->shape[normalized_axis] == 0) break;
        CnpSlice leading[CNP_MAXDIMS] = {0};
        CnpSlice trailing[CNP_MAXDIMS] = {0};
        leading[normalized_axis].start = 1;
        leading[normalized_axis].has_start = true;
        trailing[normalized_axis].stop =
            current->shape[normalized_axis] - 1;
        trailing[normalized_axis].has_stop = true;
        CnpArray *right = cnp_array_slice(
            current, current->ndim, leading);
        if (!right) {
            cnp_array_decref(current);
            cnp_relabel_error(function_name);
            return NULL;
        }
        CnpArray *left = cnp_array_slice(
            current, current->ndim, trailing);
        if (!left) {
            cnp_array_decref(right);
            cnp_array_decref(current);
            cnp_relabel_error(function_name);
            return NULL;
        }
        CnpArray *next = current->dtype->type_num == CNP_BOOL
            ? cnp_not_equal(right, left)
            : cnp_subtract(right, left);
        cnp_array_decref(left);
        cnp_array_decref(right);
        cnp_array_decref(current);
        if (!next) {
            cnp_relabel_error(function_name);
            return NULL;
        }
        current = next;
    }
    return current;
}

/* =========================================================================
 * Gradient
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_gradient(const CnpArray *arr, int axis) {
    const char *function_name = "cnp_gradient";
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            arr, function_name, &ignored_nbytes)) return NULL;
    if (arr->ndim <= 0 || arr->ndim > CNP_MAXDIMS ||
            !arr->shape || !arr->strides ||
            (arr->size > 0 && !arr->data)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source must be a valid array with at least one dimension");
        return NULL;
    }
    CNP_TYPE source_type = arr->dtype->type_num;
    if (source_type == CNP_BOOL ||
            !(cnp_type_is_integer(source_type) ||
              cnp_type_is_float(source_type) ||
              cnp_type_is_complex(source_type))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array must have a non-boolean represented numeric dtype");
        return NULL;
    }
    int normalized_axis = axis < 0 ? axis + arr->ndim : axis;
    if (normalized_axis < 0 || normalized_axis >= arr->ndim) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "axis %d is out of bounds for array of dimension %d",
            axis, arr->ndim);
        return NULL;
    }
    int64_t axis_size = arr->shape[normalized_axis];
    if (axis_size < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "axis %d must contain at least two elements",
            normalized_axis);
        return NULL;
    }
    CNP_TYPE result_type = cnp_type_is_integer(source_type)
        ? CNP_DOUBLE : source_type;
    CnpArray *result = cnp_array_new(
        arr->ndim, arr->shape, result_type, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < arr->size; ++index) {
        int64_t position = coordinates[normalized_axis];
        int64_t left_position = position == 0
            ? 0 : position - 1;
        int64_t right_position = position == axis_size - 1
            ? axis_size - 1 : position + 1;
        long double divisor =
            (position == 0 || position == axis_size - 1) ? 1.0L : 2.0L;
        coordinates[normalized_axis] = left_position;
        int64_t left_offset = arr->offset + cnp_multi_to_offset(
            arr->ndim, coordinates, arr->strides);
        coordinates[normalized_axis] = right_position;
        int64_t right_offset = arr->offset + cnp_multi_to_offset(
            arr->ndim, coordinates, arr->strides);
        coordinates[normalized_axis] = position;

        cnp_clongdouble left_value = {0};
        cnp_clongdouble right_value = {0};
        CNP_STATUS status = cnp_cast_scalar_value(
            (const char*)arr->data + left_offset,
            source_type, &left_value, CNP_CLONGDOUBLE, function_name);
        if (status == CNP_OK)
            status = cnp_cast_scalar_value(
                (const char*)arr->data + right_offset,
                source_type, &right_value, CNP_CLONGDOUBLE, function_name);
        cnp_clongdouble difference = {
            (right_value.real - left_value.real) / divisor,
            (right_value.imag - left_value.imag) / divisor,
        };
        if (status == CNP_OK)
            status = cnp_cast_scalar_value(
                &difference, CNP_CLONGDOUBLE,
                (char*)result->data + index * result->dtype->elsize,
                result_type, function_name);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            return NULL;
        }

        for (int dimension = arr->ndim - 1; dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < arr->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

/* =========================================================================
 * Interpolation
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_interp(const CnpArray *x, const CnpArray *xp, const CnpArray *fp) {
    return cnp_interp_core(
        x, xp, fp, false, 0.0, 0.0, "cnp_interp");
}
