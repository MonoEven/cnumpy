/**
 * cnumpy polynomial subpackage extensions
 * Corresponds to numpy.polynomial: chebyshev, legendre, hermite, laguerre
 *   Each with: polyval, polyfit, polyder, polyint, polyadd, polysub, polymul, polyfromroots
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>

typedef struct {
    double real;
    double imag;
} CnpBasisCoefficient;

typedef enum {
    CNP_CALCULUS_CHEBYSHEV,
    CNP_CALCULUS_LEGENDRE,
    CNP_CALCULUS_HERMITE,
    CNP_CALCULUS_LAGUERRE
} CnpCalculusBasis;

static bool basis_calculus_dtype_supported(CNP_TYPE dtype) {
    return dtype == CNP_BOOL || cnp_type_is_integer(dtype) ||
           cnp_type_is_float(dtype) || cnp_type_is_complex(dtype);
}

static CNP_TYPE basis_calculus_result_dtype(CNP_TYPE source_dtype) {
    if (source_dtype == CNP_BOOL || cnp_type_is_integer(source_dtype))
        return CNP_DOUBLE;
    return source_dtype;
}

static int64_t basis_calculus_slice_count(const CnpArray *array) {
    int64_t count = 1;
    for (int dimension = 1; dimension < array->ndim; ++dimension)
        count *= array->shape[dimension];
    return count;
}

static const void *basis_calculus_pointer(
    const CnpArray *array, int64_t degree, int64_t slice) {
    int64_t coordinates[CNP_MAXDIMS] = {0};
    if (array->ndim > 0) {
        coordinates[0] = degree;
        for (int dimension = array->ndim - 1; dimension >= 1; --dimension) {
            int64_t length = array->shape[dimension];
            coordinates[dimension] = length > 0 ? slice % length : 0;
            if (length > 0) slice /= length;
        }
    }
    return cnp_array_at(array, coordinates);
}

static CnpBasisCoefficient basis_calculus_read(
    const CnpArray *array, int64_t degree, int64_t slice) {
    const void *pointer = basis_calculus_pointer(array, degree, slice);
    CnpBasisCoefficient value = {0.0, 0.0};
    switch (array->dtype->type_num) {
        case CNP_HALF:
            value.real = cnp_half_to_float(*(const uint16_t*)pointer);
            break;
        case CNP_CFLOAT: {
            const cnp_cfloat *complex_value = (const cnp_cfloat*)pointer;
            value.real = complex_value->real;
            value.imag = complex_value->imag;
            break;
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *complex_value = (const cnp_cdouble*)pointer;
            value.real = complex_value->real;
            value.imag = complex_value->imag;
            break;
        }
        case CNP_CLONGDOUBLE: {
            const cnp_clongdouble *complex_value =
                (const cnp_clongdouble*)pointer;
            value.real = (double)complex_value->real;
            value.imag = (double)complex_value->imag;
            break;
        }
        default:
            value.real = cnp_get_element_double(
                pointer, 0, array->dtype->type_num);
            break;
    }
    return value;
}

static CnpBasisCoefficient basis_calculus_round(
    CnpBasisCoefficient value, CNP_TYPE dtype) {
    switch (dtype) {
        case CNP_HALF:
            value.real = cnp_half_to_float(cnp_float_to_half(value.real));
            value.imag = 0.0;
            break;
        case CNP_FLOAT:
            value.real = (float)value.real;
            value.imag = 0.0;
            break;
        case CNP_CFLOAT:
            value.real = (float)value.real;
            value.imag = (float)value.imag;
            break;
        case CNP_DOUBLE:
        case CNP_LONGDOUBLE:
            value.imag = 0.0;
            break;
        default:
            break;
    }
    return value;
}

static CnpBasisCoefficient basis_calculus_scale(
    CnpBasisCoefficient value, double factor, CNP_TYPE dtype) {
    value.real *= factor;
    value.imag *= factor;
    return basis_calculus_round(value, dtype);
}

static CnpBasisCoefficient basis_calculus_add_scaled(
    CnpBasisCoefficient target, CnpBasisCoefficient value,
    double factor, CNP_TYPE dtype) {
    target.real += value.real * factor;
    target.imag += value.imag * factor;
    return basis_calculus_round(target, dtype);
}

static void basis_calculus_store(
    CnpArray *result, int64_t index, CnpBasisCoefficient value) {
    switch (result->dtype->type_num) {
        case CNP_HALF:
            ((uint16_t*)result->data)[index] =
                cnp_float_to_half(value.real);
            break;
        case CNP_FLOAT:
            ((float*)result->data)[index] = (float)value.real;
            break;
        case CNP_DOUBLE:
            ((double*)result->data)[index] = value.real;
            break;
        case CNP_LONGDOUBLE:
            ((long double*)result->data)[index] = (long double)value.real;
            break;
        case CNP_CFLOAT:
            ((cnp_cfloat*)result->data)[index].real = (float)value.real;
            ((cnp_cfloat*)result->data)[index].imag = (float)value.imag;
            break;
        case CNP_CDOUBLE:
            ((cnp_cdouble*)result->data)[index].real = value.real;
            ((cnp_cdouble*)result->data)[index].imag = value.imag;
            break;
        case CNP_CLONGDOUBLE:
            ((cnp_clongdouble*)result->data)[index].real =
                (long double)value.real;
            ((cnp_clongdouble*)result->data)[index].imag =
                (long double)value.imag;
            break;
        default:
            break;
    }
}

static bool basis_series_coefficient_is_zero(
    CnpBasisCoefficient value) {
    return !isnan(value.real) && !isnan(value.imag) &&
           value.real == 0.0 && value.imag == 0.0;
}

static int64_t basis_series_trimmed_length(
    const CnpArray *array, CNP_TYPE dtype) {
    int64_t length = array->size;
    while (length > 1) {
        CnpBasisCoefficient value = basis_calculus_round(
            basis_calculus_read(array, length - 1, 0), dtype);
        if (!basis_series_coefficient_is_zero(value))
            break;
        --length;
    }
    return length;
}

static CnpArray *basis_series_add_or_subtract(
    const CnpArray *left, const CnpArray *right, bool subtract,
    const char *function_name) {
    CNP_TYPE result_dtype;
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input coefficient arrays are required");
        return NULL;
    }
    if (left->ndim > 1 || right->ndim > 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "coefficient arrays must be one-dimensional");
        return NULL;
    }
    if (left->size == 0 || right->size == 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "coefficient array is empty");
        return NULL;
    }
    if (!cnp_polynomial_common_type(
            left->dtype->type_num, right->dtype->type_num,
            &result_dtype)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "coefficient arrays have no common real or complex type");
        return NULL;
    }

    int64_t left_length = basis_series_trimmed_length(
        left, result_dtype);
    int64_t right_length = basis_series_trimmed_length(
        right, result_dtype);
    int64_t length = left_length > right_length
        ? left_length : right_length;
    if ((uint64_t)length >
            SIZE_MAX / sizeof(CnpBasisCoefficient)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "coefficient workspace size overflows");
        return NULL;
    }
    size_t workspace_bytes =
        (size_t)length * sizeof(CnpBasisCoefficient);
    CnpBasisCoefficient *workspace =
        (CnpBasisCoefficient*)cnp_malloc(workspace_bytes);
    if (!workspace) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    bool start_with_left = left_length > right_length;
    for (int64_t index = 0; index < length; ++index) {
        CnpBasisCoefficient value;
        if (subtract && !start_with_left) {
            value = basis_calculus_scale(
                basis_calculus_round(
                    basis_calculus_read(right, index, 0), result_dtype),
                -1.0, result_dtype);
            if (index < left_length) {
                value = basis_calculus_add_scaled(
                    value,
                    basis_calculus_round(
                        basis_calculus_read(left, index, 0), result_dtype),
                    1.0, result_dtype);
            }
        } else if (start_with_left) {
            value = basis_calculus_round(
                basis_calculus_read(left, index, 0), result_dtype);
            if (index < right_length) {
                value = basis_calculus_add_scaled(
                    value,
                    basis_calculus_round(
                        basis_calculus_read(right, index, 0), result_dtype),
                    subtract ? -1.0 : 1.0, result_dtype);
            }
        } else {
            value = basis_calculus_round(
                basis_calculus_read(right, index, 0), result_dtype);
            if (index < left_length) {
                value = basis_calculus_add_scaled(
                    value,
                    basis_calculus_round(
                        basis_calculus_read(left, index, 0), result_dtype),
                    1.0, result_dtype);
            }
        }
        workspace[index] = value;
    }
    while (length > 1 && basis_series_coefficient_is_zero(
            workspace[length - 1]))
        --length;

    int64_t shape[1] = {length};
    CnpArray *result = cnp_array_new(
        1, shape, result_dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
    } else {
        for (int64_t index = 0; index < length; ++index)
            basis_calculus_store(result, index, workspace[index]);
    }
    cnp_free(workspace, workspace_bytes);
    return result;
}

static bool basis_calculus_result_shape(
    const CnpArray *source, int64_t axis_length,
    int *result_ndim, int64_t *shape) {
    *result_ndim = source->ndim > 0 ? source->ndim : 1;
    if (source->ndim == 0) {
        shape[0] = axis_length;
        return true;
    }
    for (int dimension = 0; dimension < source->ndim; ++dimension)
        shape[dimension] = source->shape[dimension];
    shape[0] = axis_length;
    return true;
}

static CnpArray *basis_calculus_derivative(
    const CnpArray *source, int m, CnpCalculusBasis basis,
    const char *function_name) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "coefficient array is required");
        return NULL;
    }
    if (m < 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "derivative order must be non-negative");
        return NULL;
    }
    if (!basis_calculus_dtype_supported(source->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "coefficient dtype is not numeric");
        return NULL;
    }

    int64_t axis_length = source->ndim > 0 ? source->shape[0] : 1;
    int64_t result_axis_length = m == 0
        ? axis_length
        : m >= axis_length ? (axis_length > 0 ? 1 : 0) : axis_length - m;
    int result_ndim = 0;
    int64_t result_shape[CNP_MAXDIMS] = {0};
    basis_calculus_result_shape(
        source, result_axis_length, &result_ndim, result_shape);
    CNP_TYPE result_dtype = basis_calculus_result_dtype(
        source->dtype->type_num);
    CnpArray *result = cnp_array_new(
        result_ndim, result_shape, result_dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t slice_count = basis_calculus_slice_count(source);
    if (m > 0 && (axis_length == 0 || m >= axis_length)) {
        CnpBasisCoefficient zero = {0.0, 0.0};
        for (int64_t index = 0; index < result->size; ++index)
            basis_calculus_store(result, index, zero);
        return result;
    }
    if (m == 0) {
        for (int64_t degree = 0; degree < axis_length; ++degree) {
            for (int64_t slice = 0; slice < slice_count; ++slice) {
                basis_calculus_store(
                    result, degree * slice_count + slice,
                    basis_calculus_round(
                        basis_calculus_read(source, degree, slice),
                        result_dtype));
            }
        }
        return result;
    }
    if ((uint64_t)axis_length >
        SIZE_MAX / (2 * sizeof(CnpBasisCoefficient))) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "derivative workspace size overflows");
        return NULL;
    }

    size_t workspace_bytes =
        (size_t)axis_length * 2 * sizeof(CnpBasisCoefficient);
    CnpBasisCoefficient *workspace =
        (CnpBasisCoefficient*)cnp_malloc(workspace_bytes);
    if (!workspace) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "cannot allocate derivative workspace");
        return NULL;
    }
    CnpBasisCoefficient *current = workspace;
    CnpBasisCoefficient *next = workspace + axis_length;

    for (int64_t slice = 0; slice < slice_count; ++slice) {
        for (int64_t degree = 0; degree < axis_length; ++degree)
            current[degree] = basis_calculus_round(
                basis_calculus_read(source, degree, slice), result_dtype);
        int64_t current_length = axis_length;
        for (int order = 0; order < m; ++order) {
            int64_t next_length = current_length - 1;
            if (basis == CNP_CALCULUS_CHEBYSHEV) {
                for (int64_t degree = next_length; degree > 2; --degree) {
                    next[degree - 1] = basis_calculus_scale(
                        current[degree], 2.0 * degree, result_dtype);
                    current[degree - 2] = basis_calculus_add_scaled(
                        current[degree - 2], current[degree],
                        (double)degree / (degree - 2), result_dtype);
                }
                if (next_length > 1)
                    next[1] = basis_calculus_scale(
                        current[2], 4.0, result_dtype);
                next[0] = basis_calculus_round(current[1], result_dtype);
            } else if (basis == CNP_CALCULUS_LEGENDRE) {
                for (int64_t degree = next_length; degree > 2; --degree) {
                    next[degree - 1] = basis_calculus_scale(
                        current[degree], 2.0 * degree - 1.0, result_dtype);
                    current[degree - 2] = basis_calculus_add_scaled(
                        current[degree - 2], current[degree],
                        1.0, result_dtype);
                }
                if (next_length > 1)
                    next[1] = basis_calculus_scale(
                        current[2], 3.0, result_dtype);
                next[0] = basis_calculus_round(current[1], result_dtype);
            } else if (basis == CNP_CALCULUS_HERMITE) {
                for (int64_t degree = next_length; degree > 0; --degree)
                    next[degree - 1] = basis_calculus_scale(
                        current[degree], 2.0 * degree, result_dtype);
            } else {
                for (int64_t degree = next_length; degree > 1; --degree) {
                    next[degree - 1] = basis_calculus_scale(
                        current[degree], -1.0, result_dtype);
                    current[degree - 1] = basis_calculus_add_scaled(
                        current[degree - 1], current[degree],
                        1.0, result_dtype);
                }
                next[0] = basis_calculus_scale(
                    current[1], -1.0, result_dtype);
            }
            CnpBasisCoefficient *swap = current;
            current = next;
            next = swap;
            current_length = next_length;
        }
        for (int64_t degree = 0; degree < result_axis_length; ++degree)
            basis_calculus_store(
                result, degree * slice_count + slice, current[degree]);
        if (current != workspace) {
            CnpBasisCoefficient *swap = current;
            current = next;
            next = swap;
        }
    }
    cnp_free(workspace, workspace_bytes);
    return result;
}

static CnpBasisCoefficient basis_calculus_chebval(
    const CnpBasisCoefficient *coefficients, int64_t length, double x) {
    CnpBasisCoefficient c0 = {0.0, 0.0};
    CnpBasisCoefficient c1 = {0.0, 0.0};
    if (length == 1) {
        c0 = coefficients[0];
    } else if (length == 2) {
        c0 = coefficients[0];
        c1 = coefficients[1];
    } else {
        c0 = coefficients[length - 2];
        c1 = coefficients[length - 1];
        for (int64_t count = 3; count <= length; ++count) {
            CnpBasisCoefficient previous_c0 = c0;
            c0.real = coefficients[length - count].real - c1.real;
            c0.imag = coefficients[length - count].imag - c1.imag;
            c1.real = previous_c0.real + c1.real * (2.0 * x);
            c1.imag = previous_c0.imag + c1.imag * (2.0 * x);
        }
    }
    c0.real += c1.real * x;
    c0.imag += c1.imag * x;
    return c0;
}

static CnpBasisCoefficient basis_evaluation_add(
    CnpBasisCoefficient left, CnpBasisCoefficient right,
    CNP_TYPE dtype) {
    return basis_calculus_round(
        (CnpBasisCoefficient){
            left.real + right.real,
            left.imag + right.imag},
        dtype);
}

static CnpBasisCoefficient basis_evaluation_subtract(
    CnpBasisCoefficient left, CnpBasisCoefficient right,
    CNP_TYPE dtype) {
    return basis_calculus_round(
        (CnpBasisCoefficient){
            left.real - right.real,
            left.imag - right.imag},
        dtype);
}

static CnpBasisCoefficient basis_evaluation_multiply(
    CnpBasisCoefficient left, CnpBasisCoefficient right,
    CNP_TYPE dtype) {
    return basis_calculus_round(
        (CnpBasisCoefficient){
            left.real * right.real - left.imag * right.imag,
            left.real * right.imag + left.imag * right.real},
        dtype);
}

static const void *basis_evaluation_flat_pointer(
    const CnpArray *array, int64_t index) {
    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int dimension = array->ndim - 1; dimension >= 0; --dimension) {
        int64_t length = array->shape[dimension];
        coordinates[dimension] = length > 0 ? index % length : 0;
        if (length > 0) index /= length;
    }
    return cnp_array_at(array, coordinates);
}

static CnpBasisCoefficient basis_evaluation_read_point(
    const CnpArray *array, int64_t index) {
    const void *pointer = basis_evaluation_flat_pointer(array, index);
    CnpBasisCoefficient value = {0.0, 0.0};
    switch (array->dtype->type_num) {
        case CNP_HALF:
            value.real = cnp_half_to_float(*(const uint16_t*)pointer);
            break;
        case CNP_CFLOAT: {
            const cnp_cfloat *complex_value =
                (const cnp_cfloat*)pointer;
            value.real = complex_value->real;
            value.imag = complex_value->imag;
            break;
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *complex_value =
                (const cnp_cdouble*)pointer;
            value.real = complex_value->real;
            value.imag = complex_value->imag;
            break;
        }
        case CNP_CLONGDOUBLE: {
            const cnp_clongdouble *complex_value =
                (const cnp_clongdouble*)pointer;
            value.real = (double)complex_value->real;
            value.imag = (double)complex_value->imag;
            break;
        }
        default:
            value.real = cnp_get_element_double(
                pointer, 0, array->dtype->type_num);
            break;
    }
    return value;
}

static CnpBasisCoefficient basis_evaluation_value(
    const CnpBasisCoefficient *coefficients,
    int64_t coefficient_count,
    CnpBasisCoefficient point,
    CnpCalculusBasis basis,
    CNP_TYPE dtype) {
    CnpBasisCoefficient zero = {0.0, 0.0};
    CnpBasisCoefficient c0;
    CnpBasisCoefficient c1;
    CnpBasisCoefficient point_twice = basis_calculus_scale(
        point, 2.0, dtype);

    if (coefficient_count == 1) {
        c0 = coefficients[0];
        c1 = zero;
    } else if (coefficient_count == 2) {
        c0 = coefficients[0];
        c1 = coefficients[1];
    } else {
        int64_t remaining_degree = coefficient_count;
        c0 = coefficients[coefficient_count - 2];
        c1 = coefficients[coefficient_count - 1];
        for (int64_t count = 3;
             count <= coefficient_count; ++count) {
            CnpBasisCoefficient previous_c0 = c0;
            remaining_degree--;
            if (basis == CNP_CALCULUS_CHEBYSHEV) {
                c0 = basis_evaluation_subtract(
                    coefficients[coefficient_count - count], c1,
                    dtype);
                c1 = basis_evaluation_add(
                    previous_c0,
                    basis_evaluation_multiply(c1, point_twice, dtype),
                    dtype);
            } else if (basis == CNP_CALCULUS_LEGENDRE) {
                CnpBasisCoefficient lower = basis_calculus_scale(
                    basis_calculus_scale(
                        c1, (double)(remaining_degree - 1), dtype),
                    1.0 / (double)remaining_degree, dtype);
                CnpBasisCoefficient upper = basis_evaluation_multiply(
                    c1, point, dtype);
                upper = basis_calculus_scale(
                    upper,
                    (double)(2 * remaining_degree - 1), dtype);
                upper = basis_calculus_scale(
                    upper, 1.0 / (double)remaining_degree, dtype);
                c0 = basis_evaluation_subtract(
                    coefficients[coefficient_count - count], lower,
                    dtype);
                c1 = basis_evaluation_add(previous_c0, upper, dtype);
            } else if (basis == CNP_CALCULUS_HERMITE) {
                CnpBasisCoefficient lower = basis_calculus_scale(
                    c1, (double)(2 * (remaining_degree - 1)), dtype);
                c0 = basis_evaluation_subtract(
                    coefficients[coefficient_count - count], lower,
                    dtype);
                c1 = basis_evaluation_add(
                    previous_c0,
                    basis_evaluation_multiply(c1, point_twice, dtype),
                    dtype);
            } else {
                CnpBasisCoefficient lower = basis_calculus_scale(
                    basis_calculus_scale(
                        c1, (double)(remaining_degree - 1), dtype),
                    1.0 / (double)remaining_degree, dtype);
                CnpBasisCoefficient factor = basis_evaluation_subtract(
                    (CnpBasisCoefficient){
                        (double)(2 * remaining_degree - 1), 0.0},
                    point, dtype);
                CnpBasisCoefficient upper = basis_evaluation_multiply(
                    c1, factor, dtype);
                upper = basis_calculus_scale(
                    upper, 1.0 / (double)remaining_degree, dtype);
                c0 = basis_evaluation_subtract(
                    coefficients[coefficient_count - count], lower,
                    dtype);
                c1 = basis_evaluation_add(previous_c0, upper, dtype);
            }
        }
    }

    if (basis == CNP_CALCULUS_HERMITE) {
        return basis_evaluation_add(
            c0, basis_evaluation_multiply(c1, point_twice, dtype),
            dtype);
    }
    if (basis == CNP_CALCULUS_LAGUERRE) {
        CnpBasisCoefficient one_minus_point =
            basis_evaluation_subtract(
                (CnpBasisCoefficient){1.0, 0.0}, point, dtype);
        return basis_evaluation_add(
            c0,
            basis_evaluation_multiply(c1, one_minus_point, dtype),
            dtype);
    }
    return basis_evaluation_add(
        c0, basis_evaluation_multiply(c1, point, dtype), dtype);
}

static CnpArray *basis_evaluate(
    const CnpArray *x, const CnpArray *c,
    CnpCalculusBasis basis, const char *function_name) {
    int64_t coefficient_count;
    int coefficient_dimensions;
    int result_ndim;
    int64_t result_shape[CNP_MAXDIMS] = {0};
    int64_t slice_count;
    CNP_TYPE coefficient_dtype;
    CNP_TYPE result_dtype;
    CnpArray *result;
    CnpBasisCoefficient *workspace;
    size_t workspace_bytes;

    if (!x) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "point array is required");
        return NULL;
    }
    if (!c) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "coefficient array is required");
        return NULL;
    }
    if (!basis_calculus_dtype_supported(x->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "point dtype is not numeric");
        return NULL;
    }
    if (!basis_calculus_dtype_supported(c->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "coefficient dtype is not numeric");
        return NULL;
    }
    coefficient_count = c->ndim > 0 ? c->shape[0] : 1;
    if (coefficient_count == 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "coefficient axis is empty");
        return NULL;
    }
    coefficient_dimensions = c->ndim > 0 ? c->ndim - 1 : 0;
    if (coefficient_dimensions > CNP_MAXDIMS - x->ndim) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "tensor result exceeds the maximum rank");
        return NULL;
    }
    result_ndim = coefficient_dimensions + x->ndim;
    for (int dimension = 0;
         dimension < coefficient_dimensions; ++dimension) {
        result_shape[dimension] = c->shape[dimension + 1];
    }
    for (int dimension = 0; dimension < x->ndim; ++dimension) {
        result_shape[coefficient_dimensions + dimension] =
            x->shape[dimension];
    }
    coefficient_dtype = basis_calculus_result_dtype(
        c->dtype->type_num);
    result_dtype = cnp_promote_type(
        coefficient_dtype, x->dtype->type_num);
    if (result_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "point and coefficient dtypes cannot be promoted");
        return NULL;
    }
    result = cnp_array_new(
        result_ndim, result_shape, result_dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    slice_count = basis_calculus_slice_count(c);
    if (slice_count == 0 || x->size == 0) return result;
    if ((uint64_t)coefficient_count >
            SIZE_MAX / sizeof(CnpBasisCoefficient)) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "coefficient workspace size overflows");
        return NULL;
    }
    workspace_bytes =
        (size_t)coefficient_count * sizeof(CnpBasisCoefficient);
    workspace = (CnpBasisCoefficient*)cnp_malloc(workspace_bytes);
    if (!workspace) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "cannot allocate coefficient workspace");
        return NULL;
    }

    for (int64_t slice = 0; slice < slice_count; ++slice) {
        for (int64_t degree = 0;
             degree < coefficient_count; ++degree) {
            workspace[degree] = basis_calculus_round(
                basis_calculus_read(c, degree, slice), result_dtype);
        }
        for (int64_t point_index = 0;
             point_index < x->size; ++point_index) {
            CnpBasisCoefficient point = basis_calculus_round(
                basis_evaluation_read_point(x, point_index),
                result_dtype);
            basis_calculus_store(
                result, slice * x->size + point_index,
                basis_evaluation_value(
                    workspace, coefficient_count, point,
                    basis, result_dtype));
        }
    }
    cnp_free(workspace, workspace_bytes);
    return result;
}

/* =========================================================================
 * Chebyshev polynomials
 * numpy.polynomial.chebyshev
 * ========================================================================= */

/* Evaluate Chebyshev series using Clenshaw's algorithm */
CNP_API CnpArray* CNP_CALL cnp_chebval(const CnpArray *x, const CnpArray *c) {
    return basis_evaluate(
        x, c, CNP_CALCULUS_CHEBYSHEV, "cnp_chebval");
}

/* Differentiate Chebyshev series */
CNP_API CnpArray* CNP_CALL cnp_chebder(const CnpArray *c, int m) {
    return basis_calculus_derivative(
        c, m, CNP_CALCULUS_CHEBYSHEV, "cnp_chebder");
}

/* Integrate Chebyshev series */
CNP_API CnpArray* CNP_CALL cnp_chebint(const CnpArray *c, int m, double lbnd) {
    const char *function_name = "cnp_chebint";
    if (!c) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "coefficient array is required");
        return NULL;
    }
    if (m < 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "integral order must be non-negative");
        return NULL;
    }
    if (!basis_calculus_dtype_supported(c->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "coefficient dtype is not numeric");
        return NULL;
    }
    int64_t axis_length = c->ndim > 0 ? c->shape[0] : 1;
    if (m == 0)
        return basis_calculus_derivative(
            c, 0, CNP_CALCULUS_CHEBYSHEV, function_name);
    if (axis_length == 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "coefficient axis is empty");
        return NULL;
    }

    int64_t slice_count = basis_calculus_slice_count(c);
    bool zero_constant_series = axis_length == 1;
    if (zero_constant_series) {
        for (int64_t slice = 0; slice < slice_count; ++slice) {
            CnpBasisCoefficient value = basis_calculus_read(c, 0, slice);
            if (value.real != 0.0 || value.imag != 0.0) {
                zero_constant_series = false;
                break;
            }
        }
    }
    if (!zero_constant_series && (int64_t)m > INT64_MAX - axis_length) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "integrated coefficient count overflows");
        return NULL;
    }
    int64_t result_axis_length = zero_constant_series
        ? 1 : axis_length + m;
    int result_ndim = 0;
    int64_t result_shape[CNP_MAXDIMS] = {0};
    basis_calculus_result_shape(
        c, result_axis_length, &result_ndim, result_shape);
    CNP_TYPE result_dtype = basis_calculus_result_dtype(
        c->dtype->type_num);
    CnpArray *result = cnp_array_new(
        result_ndim, result_shape, result_dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if ((uint64_t)result_axis_length >
        SIZE_MAX / (2 * sizeof(CnpBasisCoefficient))) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "integration workspace size overflows");
        return NULL;
    }

    size_t workspace_bytes =
        (size_t)result_axis_length * 2 * sizeof(CnpBasisCoefficient);
    CnpBasisCoefficient *workspace =
        (CnpBasisCoefficient*)cnp_malloc(workspace_bytes);
    if (!workspace) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "cannot allocate integration workspace");
        return NULL;
    }
    CnpBasisCoefficient *current = workspace;
    CnpBasisCoefficient *next = workspace + result_axis_length;
    for (int64_t slice = 0; slice < slice_count; ++slice) {
        for (int64_t degree = 0; degree < axis_length; ++degree)
            current[degree] = basis_calculus_round(
                basis_calculus_read(c, degree, slice), result_dtype);
        int64_t current_length = axis_length;
        if (!zero_constant_series) {
            for (int order = 0; order < m; ++order) {
                int64_t next_length = current_length + 1;
                memset(
                    next, 0,
                    (size_t)next_length * sizeof(CnpBasisCoefficient));
                next[1] = basis_calculus_round(current[0], result_dtype);
                if (current_length > 1)
                    next[2] = basis_calculus_scale(
                        current[1], 0.25, result_dtype);
                for (int64_t degree = 2; degree < current_length; ++degree) {
                    next[degree + 1] = basis_calculus_scale(
                        current[degree], 1.0 / (2.0 * (degree + 1)),
                        result_dtype);
                    next[degree - 1] = basis_calculus_add_scaled(
                        next[degree - 1], current[degree],
                        -1.0 / (2.0 * (degree - 1)), result_dtype);
                }
                CnpBasisCoefficient at_bound = basis_calculus_chebval(
                    next, next_length, lbnd);
                next[0] = basis_calculus_add_scaled(
                    next[0], at_bound, -1.0, result_dtype);
                CnpBasisCoefficient *swap = current;
                current = next;
                next = swap;
                current_length = next_length;
            }
        }
        for (int64_t degree = 0; degree < result_axis_length; ++degree)
            basis_calculus_store(
                result, degree * slice_count + slice, current[degree]);
        if (current != workspace) {
            CnpBasisCoefficient *swap = current;
            current = next;
            next = swap;
        }
    }
    cnp_free(workspace, workspace_bytes);
    return result;
}

/* Add two Chebyshev series */
CNP_API CnpArray* CNP_CALL cnp_chebadd(const CnpArray *c1, const CnpArray *c2) {
    return basis_series_add_or_subtract(
        c1, c2, false, "cnp_chebadd");
}

/* Subtract two Chebyshev series */
CNP_API CnpArray* CNP_CALL cnp_chebsub(const CnpArray *c1, const CnpArray *c2) {
    return basis_series_add_or_subtract(
        c1, c2, true, "cnp_chebsub");
}

/* =========================================================================
 * Legendre polynomials
 * numpy.polynomial.legendre
 * ========================================================================= */

/* Evaluate Legendre series */
CNP_API CnpArray* CNP_CALL cnp_legval(const CnpArray *x, const CnpArray *c) {
    return basis_evaluate(
        x, c, CNP_CALCULUS_LEGENDRE, "cnp_legval");
}

/* Differentiate Legendre series */
CNP_API CnpArray* CNP_CALL cnp_legder(const CnpArray *c, int m) {
    return basis_calculus_derivative(
        c, m, CNP_CALCULUS_LEGENDRE, "cnp_legder");
}

/* =========================================================================
 * Hermite polynomials (physicist's)
 * numpy.polynomial.hermite
 * ========================================================================= */

/* Evaluate Hermite series */
CNP_API CnpArray* CNP_CALL cnp_hermval(const CnpArray *x, const CnpArray *c) {
    return basis_evaluate(
        x, c, CNP_CALCULUS_HERMITE, "cnp_hermval");
}

/* Differentiate Hermite series */
CNP_API CnpArray* CNP_CALL cnp_hermder(const CnpArray *c, int m) {
    return basis_calculus_derivative(
        c, m, CNP_CALCULUS_HERMITE, "cnp_hermder");
}

/* =========================================================================
 * Laguerre polynomials
 * numpy.polynomial.laguerre
 * ========================================================================= */

/* Evaluate Laguerre series */
CNP_API CnpArray* CNP_CALL cnp_lagval(const CnpArray *x, const CnpArray *c) {
    return basis_evaluate(
        x, c, CNP_CALCULUS_LAGUERRE, "cnp_lagval");
}

/* Differentiate Laguerre series */
CNP_API CnpArray* CNP_CALL cnp_lagder(const CnpArray *c, int m) {
    return basis_calculus_derivative(
        c, m, CNP_CALCULUS_LAGUERRE, "cnp_lagder");
}

/* =========================================================================
 * Additional array utilities
 * numpy: diagflat, mgrid, ogrid, fill_diagonal (multi-dim)
 * ========================================================================= */

static int64_t poly_array_flat_offset(
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

/* diagflat - Create 2D array from flattened diagonal */
CNP_API CnpArray* CNP_CALL cnp_diagflat(const CnpArray *arr, int k) {
    const char *function_name = "cnp_diagflat";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "array is NULL");
        return NULL;
    }
    int64_t diagonal_offset = k >= 0 ? (int64_t)k : -(int64_t)k;
    if (arr->size > INT64_MAX - diagonal_offset) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "diagflat result dimension exceeds int64");
        return NULL;
    }
    int64_t n = arr->size + diagonal_offset;
    int64_t shape[2] = {n, n};
    CnpArray *result = cnp_array_zeros(
        2, shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    int itemsize = arr->dtype->elsize;
    for (int64_t index = 0; index < arr->size; ++index) {
        int64_t row = k >= 0 ? index : index - k;
        int64_t column = k >= 0 ? index + k : index;
        int64_t source_offset = poly_array_flat_offset(arr, index);
        int64_t destination_index = row * n + column;
        memcpy(
            (char*)result->data + destination_index * itemsize,
            (const char*)arr->data + source_offset,
            (size_t)itemsize);
    }
    return result;
}

static bool grid_axis_length(
    int64_t start, int64_t stop, int64_t step,
    int dimension, const char *function_name, int64_t *length) {
    if (step == 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "step for dimension %d must be nonzero", dimension);
        return false;
    }
    uint64_t count = 0;
    if (step > 0 && start < stop) {
        uint64_t distance = (uint64_t)stop - (uint64_t)start;
        count = UINT64_C(1) + (distance - UINT64_C(1)) / (uint64_t)step;
    } else if (step < 0 && start > stop) {
        uint64_t distance = (uint64_t)start - (uint64_t)stop;
        uint64_t magnitude = (uint64_t)(-(step + 1)) + UINT64_C(1);
        count = UINT64_C(1) + (distance - UINT64_C(1)) / magnitude;
    }
    if (count > (uint64_t)INT64_MAX) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "grid length for dimension %d overflows int64", dimension);
        return false;
    }
    *length = (int64_t)count;
    return true;
}

static void grid_release_outputs(CnpArray **result, int count) {
    if (!result) return;
    for (int index = 0; index < count; ++index) {
        if (result[index]) cnp_array_decref(result[index]);
        result[index] = NULL;
    }
}

/* mgrid - Open multi-dimensional meshgrid (returns ndim arrays) */
CNP_API CNP_STATUS CNP_CALL cnp_mgrid(int ndim, const int64_t *start, const int64_t *stop,
                                       const int64_t *step, CnpArray **result) {
    const char *function_name = "cnp_mgrid";
    if (result && ndim > 0 && ndim <= CNP_MAXDIMS)
        for (int dimension = 0; dimension < ndim; ++dimension)
            result[dimension] = NULL;
    if (ndim <= 0 || ndim > CNP_MAXDIMS ||
            !start || !stop || !step || !result) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "valid ndim, start, stop, step, and result pointers are required");
        return CNP_ERR_VALUE;
    }

    int64_t shape[CNP_MAXDIMS];
    int64_t total = 1;
    for (int d = 0; d < ndim; d++) {
        if (!grid_axis_length(
                start[d], stop[d], step[d], d,
                function_name, &shape[d])) return cnp_get_error(NULL);
        if (shape[d] == 0) {
            total = 0;
        } else if (total > 0) {
            if (total > INT64_MAX / shape[d]) {
                cnp_set_error(
                    CNP_ERR_SHAPE, function_name,
                    "dense grid element count overflows int64");
                return CNP_ERR_SHAPE;
            }
            total *= shape[d];
        }
    }

    for (int d = 0; d < ndim; d++) {
        result[d] = cnp_array_new(ndim, shape, CNP_LONGLONG, CNP_ORDER_C);
        if (!result[d]) {
            cnp_relabel_error(function_name);
            CNP_STATUS status = cnp_get_error(NULL);
            grid_release_outputs(result, d);
            return status;
        }
        int64_t *out = (int64_t*)result[d]->data;
        for (int64_t i = 0; i < total; i++) {
            int64_t coords[CNP_MAXDIMS];
            int64_t tmp = i;
            for (int dd = ndim - 1; dd >= 0; dd--) {
                coords[dd] = tmp % shape[dd];
                tmp /= shape[dd];
            }
            out[i] = start[d] + coords[d] * step[d];
        }
    }
    return CNP_OK;
}

/* ogrid - Open multi-dimensional meshgrid (sparse) */
CNP_API CNP_STATUS CNP_CALL cnp_ogrid(int ndim, const int64_t *start, const int64_t *stop,
                                       const int64_t *step, CnpArray **result) {
    const char *function_name = "cnp_ogrid";
    if (result && ndim > 0 && ndim <= CNP_MAXDIMS)
        for (int dimension = 0; dimension < ndim; ++dimension)
            result[dimension] = NULL;
    if (ndim <= 0 || ndim > CNP_MAXDIMS ||
            !start || !stop || !step || !result) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "valid ndim, start, stop, step, and result pointers are required");
        return CNP_ERR_VALUE;
    }

    for (int d = 0; d < ndim; d++) {
        int64_t n;
        if (!grid_axis_length(
                start[d], stop[d], step[d], d,
                function_name, &n)) {
            CNP_STATUS status = cnp_get_error(NULL);
            grid_release_outputs(result, d);
            return status;
        }
        int64_t shape[CNP_MAXDIMS];
        for (int dd = 0; dd < ndim; dd++) shape[dd] = (dd == d) ? n : 1;
        result[d] = cnp_array_new(ndim, shape, CNP_LONGLONG, CNP_ORDER_C);
        if (!result[d]) {
            cnp_relabel_error(function_name);
            CNP_STATUS status = cnp_get_error(NULL);
            grid_release_outputs(result, d);
            return status;
        }
        int64_t *out = (int64_t*)result[d]->data;
        for (int64_t i = 0; i < n; i++) {
            out[i] = start[d] + i * step[d];
        }
    }
    return CNP_OK;
}

/* Testing utilities: boolean projections of NumPy's testing assertions. */
static bool assertion_numeric_dtype(CNP_TYPE type) {
    return type == CNP_BOOL || cnp_type_is_integer(type) ||
           cnp_type_is_float(type) || cnp_type_is_complex(type);
}

static bool assertion_validate_array(
    const CnpArray *array, const char *role, const char *function_name) {
    if (!array) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "%s array is required", role);
        return false;
    }
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            array, function_name, &ignored_nbytes)) return false;
    if (!array->dtype || !assertion_numeric_dtype(array->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "%s array must have a represented numeric dtype", role);
        return false;
    }
    if (array->ndim < 0 || array->ndim > CNP_MAXDIMS ||
            (array->ndim > 0 && (!array->shape || !array->strides))) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "%s array has invalid shape metadata", role);
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

static bool assertion_shapes_compatible(
    const CnpArray *left, const CnpArray *right) {
    if (left->ndim == 0 || right->ndim == 0) return true;
    if (left->ndim != right->ndim) return false;
    for (int axis = 0; axis < left->ndim; ++axis) {
        if (left->shape[axis] != right->shape[axis]) return false;
    }
    return true;
}

static const void *assertion_flat_pointer(
    const CnpArray *array, int64_t flat_index) {
    int64_t offset = array->offset;
    if (array->ndim == 0)
        return (const char*)array->data + offset;
    for (int axis = array->ndim - 1; axis >= 0; --axis) {
        int64_t dimension = array->shape[axis];
        int64_t coordinate = dimension > 0 ? flat_index % dimension : 0;
        if (dimension > 0) flat_index /= dimension;
        offset += coordinate * array->strides[axis];
    }
    return (const char*)array->data + offset;
}

static bool assertion_read_value(
    const void *source, CNP_TYPE source_type,
    cnp_clongdouble *value, const char *function_name) {
    return cnp_cast_scalar_value(
        source, source_type, value, CNP_CLONGDOUBLE,
        function_name) == CNP_OK;
}

static bool assertion_value_is_nan(cnp_clongdouble value) {
    return isnan((double)value.real) || isnan((double)value.imag);
}

static bool assertion_values_equal(
    const void *left, CNP_TYPE left_type,
    const void *right, CNP_TYPE right_type,
    const char *function_name, bool *equal) {
    cnp_clongdouble left_value = {0};
    cnp_clongdouble right_value = {0};
    if (!assertion_read_value(
            left, left_type, &left_value, function_name) ||
        !assertion_read_value(
            right, right_type, &right_value, function_name)) return false;
    bool left_nan = assertion_value_is_nan(left_value);
    bool right_nan = assertion_value_is_nan(right_value);
    if (left_nan || right_nan) {
        *equal = left_nan && right_nan;
        return true;
    }

    CNP_TYPE comparison_type = cnp_promote_type_full(left_type, right_type);
    if (!assertion_numeric_dtype(comparison_type)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input dtypes do not have a represented numeric comparison type");
        return false;
    }
    int order = 0;
    CNP_STATUS status = cnp_compare_numeric_elements(
        left, left_type, right, right_type,
        comparison_type, &order, function_name);
    if (status != CNP_OK) return false;
    *equal = order == 0;
    return true;
}

static bool assertion_read_integer_magnitude(
    const void *source, CNP_TYPE type,
    bool *negative, uint64_t *magnitude) {
    int64_t signed_value;
    uint64_t unsigned_value;
    switch (type) {
        case CNP_BOOL:
        case CNP_BYTE:
            signed_value = *(const int8_t*)source;
            break;
        case CNP_UBYTE:
            *negative = false;
            *magnitude = *(const uint8_t*)source;
            return true;
        case CNP_SHORT:
            signed_value = *(const int16_t*)source;
            break;
        case CNP_USHORT:
            *negative = false;
            *magnitude = *(const uint16_t*)source;
            return true;
        case CNP_INT:
            signed_value = *(const int32_t*)source;
            break;
        case CNP_UINT:
            *negative = false;
            *magnitude = *(const uint32_t*)source;
            return true;
        case CNP_LONG:
        case CNP_LONGLONG:
            signed_value = *(const int64_t*)source;
            break;
        case CNP_ULONG:
        case CNP_ULONGLONG:
            unsigned_value = *(const uint64_t*)source;
            *negative = false;
            *magnitude = unsigned_value;
            return true;
        default:
            return false;
    }
    *negative = signed_value < 0;
    *magnitude = *negative
        ? UINT64_C(0) - (uint64_t)signed_value
        : (uint64_t)signed_value;
    return true;
}

static long double assertion_integer_difference(
    const void *left, CNP_TYPE left_type,
    const void *right, CNP_TYPE right_type) {
    bool left_negative;
    bool right_negative;
    uint64_t left_magnitude;
    uint64_t right_magnitude;
    assertion_read_integer_magnitude(
        left, left_type, &left_negative, &left_magnitude);
    assertion_read_integer_magnitude(
        right, right_type, &right_negative, &right_magnitude);
    if (left_negative == right_negative) {
        uint64_t difference = left_magnitude >= right_magnitude
            ? left_magnitude - right_magnitude
            : right_magnitude - left_magnitude;
        return (long double)difference;
    }
    return (long double)left_magnitude + (long double)right_magnitude;
}

static bool assertion_compare_elements(
    const CnpArray *left, const CnpArray *right,
    int64_t count, bool almost, long double tolerance,
    const char *function_name, bool *result) {
    for (int64_t index = 0; index < count; ++index) {
        const void *left_pointer = assertion_flat_pointer(
            left, left->ndim == 0 ? 0 : index);
        const void *right_pointer = assertion_flat_pointer(
            right, right->ndim == 0 ? 0 : index);
        CNP_TYPE left_type = left->dtype->type_num;
        CNP_TYPE right_type = right->dtype->type_num;
        bool equal = false;
        if (!assertion_values_equal(
                left_pointer, left_type, right_pointer, right_type,
                function_name, &equal)) return false;
        if (equal) continue;
        if (!almost) {
            *result = false;
            return true;
        }

        long double difference;
        bool left_integer = left_type == CNP_BOOL ||
            cnp_type_is_integer(left_type);
        bool right_integer = right_type == CNP_BOOL ||
            cnp_type_is_integer(right_type);
        if (left_integer && right_integer) {
            difference = assertion_integer_difference(
                left_pointer, left_type, right_pointer, right_type);
        } else {
            cnp_clongdouble left_value = {0};
            cnp_clongdouble right_value = {0};
            if (!assertion_read_value(
                    left_pointer, left_type, &left_value, function_name) ||
                !assertion_read_value(
                    right_pointer, right_type, &right_value, function_name))
                return false;
            if (assertion_value_is_nan(left_value) ||
                    assertion_value_is_nan(right_value)) {
                *result = false;
                return true;
            }
            difference = hypotl(
                left_value.real - right_value.real,
                left_value.imag - right_value.imag);
        }
        if (!(difference <= tolerance)) {
            *result = false;
            return true;
        }
    }
    *result = true;
    return true;
}

CNP_API bool CNP_CALL cnp_assert_array_equal(
    const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_assert_array_equal";
    if (!assertion_validate_array(a, "actual", function_name) ||
        !assertion_validate_array(b, "desired", function_name)) return false;
    if (!assertion_shapes_compatible(a, b)) return false;
    int64_t count = a->ndim == 0 ? b->size : a->size;
    bool result = false;
    return assertion_compare_elements(
        a, b, count, false, 0.0L, function_name, &result) && result;
}

CNP_API bool CNP_CALL cnp_assert_array_almost_equal(
    const CnpArray *a, const CnpArray *b, int decimal) {
    const char *function_name = "cnp_assert_array_almost_equal";
    if (!assertion_validate_array(a, "actual", function_name) ||
        !assertion_validate_array(b, "desired", function_name)) return false;
    if (!assertion_shapes_compatible(a, b)) return false;
    long double tolerance = 1.5L * powl(10.0L, -(long double)decimal);
    int64_t count = a->ndim == 0 ? b->size : a->size;
    bool result = false;
    return assertion_compare_elements(
        a, b, count, true, tolerance, function_name, &result) && result;
}

CNP_API bool CNP_CALL cnp_assert_allclose(
    const CnpArray *a, const CnpArray *b, double rtol, double atol) {
    const char *function_name = "cnp_assert_allclose";
    if (!assertion_validate_array(a, "actual", function_name) ||
        !assertion_validate_array(b, "desired", function_name)) return false;
    if (!assertion_shapes_compatible(a, b)) return false;
    bool result = false;
    CNP_STATUS status = cnp_allclose_v2(
        a, b, rtol, atol, true, &result);
    if (status != CNP_OK) {
        cnp_relabel_error(function_name);
        return false;
    }
    return result;
}
