/**
 * cnumpy polynomial extensions 2
 * numpy.polynomial: chebyshev, legendre, hermite, laguerre
 *   Additional: polymul, polyfit, poly2cheb, cheb2poly, leg2poly, poly2leg,
 *   chebpts1, chebpts2, herm2poly, poly2herm, lag2poly, poly2lag
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    double real;
    double imag;
} CnpPolynomialCoefficient;

static CnpArray *polynomial_multiply_basis(
    const CnpArray *left, const CnpArray *right,
    CnpPolynomialBasis basis, const char *function_name);

/* =========================================================================
 * Chebyshev multiplication
 * numpy.polynomial.chebyshev.chebmul(c1, c2)
 * Uses identity: T_i * T_j = (T_{i+j} + T_{|i-j|}) / 2
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_chebmul(const CnpArray *c1, const CnpArray *c2) {
    return polynomial_multiply_basis(
        c1, c2, CNP_POLYNOMIAL_CHEBYSHEV, "cnp_chebmul");
}

/* =========================================================================
 * Chebyshev fitting
 * numpy.polynomial.chebyshev.chebfit(x, y, deg)
 * Least squares fit using Chebyshev basis
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_chebfit(const CnpArray *x, const CnpArray *y, int deg) {
    return cnp_polynomial_fit_basis(
        x, y, deg, CNP_POLYNOMIAL_CHEBYSHEV, "cnp_chebfit");
}

/* =========================================================================
 * Chebyshev points (first kind)
 * numpy.polynomial.chebyshev.chebpts1(n)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_chebpts1(int64_t n) {
    const char *function_name = "cnp_chebpts1";
    if (n < 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name, "npts must be >= 1");
        return NULL;
    }
    int64_t shape[1] = {n};
    CnpArray *result = cnp_array_new(1, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    double *out = (double*)result->data;
    double scale = 0.5 * M_PI / (double)n;
    for (int64_t k = 0; k < n; k++) {
        double coordinate = 2.0 * (double)k - (double)n + 1.0;
        out[k] = sin(scale * coordinate);
    }
    return result;
}

/* =========================================================================
 * Chebyshev points (second kind)
 * numpy.polynomial.chebyshev.chebpts2(n)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_chebpts2(int64_t n) {
    const char *function_name = "cnp_chebpts2";
    if (n < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name, "npts must be >= 2");
        return NULL;
    }
    int64_t shape[1] = {n};
    CnpArray *result = cnp_array_new(1, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    double *out = (double*)result->data;
    double step = M_PI / (double)(n - 1);
    out[0] = -1.0;
    for (int64_t k = 1; k < n - 1; k++) {
        out[k] = cos(-M_PI + (double)k * step);
    }
    out[n - 1] = 1.0;
    return result;
}

static bool polynomial_conversion_dtype_supported(CNP_TYPE dtype) {
    switch (dtype) {
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

static CnpPolynomialCoefficient polynomial_conversion_coefficient(
    const CnpArray *array, int64_t index) {
    int64_t coordinate = index;
    const void *pointer = cnp_array_at(array, &coordinate);
    CnpPolynomialCoefficient value = {0.0, 0.0};
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

static bool polynomial_coefficient_is_zero(
    CnpPolynomialCoefficient value) {
    return value.real == 0.0 && value.imag == 0.0;
}

static CNP_TYPE polynomial_conversion_result_dtype(
    CNP_TYPE source_dtype, CnpPolynomialBasis basis, bool power_to_basis) {
    bool preserves_dtype =
        basis == CNP_POLYNOMIAL_CHEBYSHEV && !power_to_basis;
    if (preserves_dtype) {
        switch (source_dtype) {
            case CNP_HALF:
            case CNP_FLOAT:
            case CNP_DOUBLE:
            case CNP_LONGDOUBLE:
            case CNP_CFLOAT:
            case CNP_CDOUBLE:
            case CNP_CLONGDOUBLE:
                return source_dtype;
            default:
                return CNP_DOUBLE;
        }
    }
    if (source_dtype == CNP_CFLOAT ||
        source_dtype == CNP_CDOUBLE ||
        source_dtype == CNP_CLONGDOUBLE)
        return CNP_CDOUBLE;
    return CNP_DOUBLE;
}

static bool polynomial_result_coefficient_is_zero(
    CnpPolynomialCoefficient value, CNP_TYPE dtype) {
    switch (dtype) {
        case CNP_HALF:
            return cnp_half_to_float(cnp_float_to_half(value.real)) == 0.0;
        case CNP_FLOAT:
            return (float)value.real == 0.0f;
        case CNP_CFLOAT:
            return (float)value.real == 0.0f &&
                   (float)value.imag == 0.0f;
        case CNP_CDOUBLE:
        case CNP_CLONGDOUBLE:
            return value.real == 0.0 && value.imag == 0.0;
        default:
            return value.real == 0.0;
    }
}

static void polynomial_store_result_coefficient(
    CnpArray *result, int64_t index, CnpPolynomialCoefficient value) {
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

typedef struct {
    CnpPolynomialCoefficient *data;
    int64_t length;
    int64_t capacity;
    CNP_TYPE dtype;
} CnpPolynomialSeries;

static bool polynomial_dtype_is_integer(CNP_TYPE dtype) {
    switch (dtype) {
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
            return true;
        default:
            return false;
    }
}

static int polynomial_real_precision_rank(CNP_TYPE dtype) {
    switch (dtype) {
        case CNP_HALF: return 0;
        case CNP_FLOAT:
        case CNP_CFLOAT: return 1;
        case CNP_DOUBLE:
        case CNP_CDOUBLE: return 2;
        case CNP_LONGDOUBLE:
        case CNP_CLONGDOUBLE: return 3;
        default: return -1;
    }
}

static bool polynomial_dtype_is_complex(CNP_TYPE dtype) {
    return dtype == CNP_CFLOAT || dtype == CNP_CDOUBLE ||
        dtype == CNP_CLONGDOUBLE;
}

static CNP_TYPE polynomial_real_dtype_from_rank(int rank) {
    if (rank <= 0) return CNP_HALF;
    if (rank == 1) return CNP_FLOAT;
    if (rank == 2) return CNP_DOUBLE;
    return CNP_LONGDOUBLE;
}

static CNP_TYPE polynomial_complex_dtype_from_rank(int rank) {
    if (rank <= 1) return CNP_CFLOAT;
    if (rank == 2) return CNP_CDOUBLE;
    return CNP_CLONGDOUBLE;
}

bool cnp_polynomial_common_type(
    CNP_TYPE left, CNP_TYPE right, CNP_TYPE *result) {
    if (left == CNP_BOOL || right == CNP_BOOL ||
            !polynomial_conversion_dtype_supported(left) ||
            !polynomial_conversion_dtype_supported(right))
        return false;
    bool integer_input = polynomial_dtype_is_integer(left) ||
        polynomial_dtype_is_integer(right);
    bool complex_result = polynomial_dtype_is_complex(left) ||
        polynomial_dtype_is_complex(right);
    int left_rank = polynomial_real_precision_rank(left);
    int right_rank = polynomial_real_precision_rank(right);
    int rank = left_rank > right_rank ? left_rank : right_rank;
    if (integer_input && rank < 2) rank = 2;
    *result = complex_result
        ? polynomial_complex_dtype_from_rank(rank)
        : polynomial_real_dtype_from_rank(rank);
    return true;
}

static CnpPolynomialCoefficient polynomial_round_coefficient(
    CnpPolynomialCoefficient value, CNP_TYPE dtype) {
    switch (dtype) {
        case CNP_HALF:
            value.real = cnp_half_to_float(
                cnp_float_to_half(value.real));
            value.imag = 0.0;
            break;
        case CNP_FLOAT:
            value.real = (float)value.real;
            value.imag = 0.0;
            break;
        case CNP_LONGDOUBLE:
            value.real = (double)(long double)value.real;
            value.imag = 0.0;
            break;
        case CNP_CFLOAT:
            value.real = (float)value.real;
            value.imag = (float)value.imag;
            break;
        case CNP_CLONGDOUBLE:
            value.real = (double)(long double)value.real;
            value.imag = (double)(long double)value.imag;
            break;
        case CNP_DOUBLE:
            value.imag = 0.0;
            break;
        default:
            break;
    }
    return value;
}

static CnpPolynomialCoefficient polynomial_coefficient_add(
    CnpPolynomialCoefficient left, CnpPolynomialCoefficient right,
    CNP_TYPE dtype) {
    CnpPolynomialCoefficient result = {
        left.real + right.real,
        left.imag + right.imag
    };
    return polynomial_round_coefficient(result, dtype);
}

static CnpPolynomialCoefficient polynomial_coefficient_subtract(
    CnpPolynomialCoefficient left, CnpPolynomialCoefficient right,
    CNP_TYPE dtype) {
    CnpPolynomialCoefficient result = {
        left.real - right.real,
        left.imag - right.imag
    };
    return polynomial_round_coefficient(result, dtype);
}

static CnpPolynomialCoefficient polynomial_coefficient_multiply(
    CnpPolynomialCoefficient left, CnpPolynomialCoefficient right,
    CNP_TYPE dtype) {
    CnpPolynomialCoefficient result = {
        left.real * right.real - left.imag * right.imag,
        left.real * right.imag + left.imag * right.real
    };
    return polynomial_round_coefficient(result, dtype);
}

static CnpPolynomialCoefficient polynomial_coefficient_scale(
    CnpPolynomialCoefficient value, double factor, CNP_TYPE dtype) {
    CnpPolynomialCoefficient result = {
        value.real * factor,
        value.imag * factor
    };
    return polynomial_round_coefficient(result, dtype);
}

static CnpPolynomialCoefficient polynomial_coefficient_scale_ratio(
    CnpPolynomialCoefficient value, double numerator,
    double denominator, CNP_TYPE dtype) {
    CnpPolynomialCoefficient result = {
        (value.real * numerator) / denominator,
        (value.imag * numerator) / denominator
    };
    return polynomial_round_coefficient(result, dtype);
}

static CnpPolynomialCoefficient polynomial_coefficient_add_scale_ratio(
    CnpPolynomialCoefficient accumulator,
    CnpPolynomialCoefficient value, double numerator,
    double denominator, CNP_TYPE dtype) {
    CnpPolynomialCoefficient result = {
        accumulator.real + (value.real * numerator) / denominator,
        accumulator.imag + (value.imag * numerator) / denominator
    };
    return polynomial_round_coefficient(result, dtype);
}

static void polynomial_series_trim(CnpPolynomialSeries *series) {
    while (series->length > 1 && polynomial_coefficient_is_zero(
            series->data[series->length - 1]))
        --series->length;
}

static void polynomial_series_zero(CnpPolynomialSeries *series) {
    memset(
        series->data, 0,
        (size_t)series->capacity * sizeof(*series->data));
    series->length = 1;
}

static void polynomial_series_copy(
    CnpPolynomialSeries *destination,
    const CnpPolynomialSeries *source) {
    memcpy(
        destination->data, source->data,
        (size_t)source->length * sizeof(*source->data));
    destination->length = source->length;
}

static void polynomial_series_scale(
    CnpPolynomialSeries *destination,
    const CnpPolynomialSeries *source,
    CnpPolynomialCoefficient factor) {
    for (int64_t index = 0; index < source->length; ++index) {
        destination->data[index] = polynomial_coefficient_multiply(
            source->data[index], factor, destination->dtype);
    }
    destination->length = source->length;
    polynomial_series_trim(destination);
}

static void polynomial_series_scale_real(
    CnpPolynomialSeries *destination,
    const CnpPolynomialSeries *source, double factor) {
    for (int64_t index = 0; index < source->length; ++index) {
        destination->data[index] = polynomial_coefficient_scale(
            source->data[index], factor, destination->dtype);
    }
    destination->length = source->length;
    polynomial_series_trim(destination);
}

static void polynomial_series_divide_real(
    CnpPolynomialSeries *destination,
    const CnpPolynomialSeries *source, double divisor) {
    for (int64_t index = 0; index < source->length; ++index) {
        destination->data[index] = polynomial_coefficient_scale_ratio(
            source->data[index], 1.0, divisor, destination->dtype);
    }
    destination->length = source->length;
    polynomial_series_trim(destination);
}

static void polynomial_series_add(
    CnpPolynomialSeries *destination,
    const CnpPolynomialSeries *left,
    const CnpPolynomialSeries *right) {
    int64_t length = left->length > right->length
        ? left->length : right->length;
    CnpPolynomialCoefficient zero = {0.0, 0.0};
    for (int64_t index = 0; index < length; ++index) {
        CnpPolynomialCoefficient left_value =
            index < left->length ? left->data[index] : zero;
        CnpPolynomialCoefficient right_value =
            index < right->length ? right->data[index] : zero;
        destination->data[index] = polynomial_coefficient_add(
            left_value, right_value, destination->dtype);
    }
    destination->length = length;
    polynomial_series_trim(destination);
}

static void polynomial_series_subtract(
    CnpPolynomialSeries *destination,
    const CnpPolynomialSeries *left,
    const CnpPolynomialSeries *right) {
    int64_t length = left->length > right->length
        ? left->length : right->length;
    CnpPolynomialCoefficient zero = {0.0, 0.0};
    for (int64_t index = 0; index < length; ++index) {
        CnpPolynomialCoefficient left_value =
            index < left->length ? left->data[index] : zero;
        CnpPolynomialCoefficient right_value =
            index < right->length ? right->data[index] : zero;
        destination->data[index] = polynomial_coefficient_subtract(
            left_value, right_value, destination->dtype);
    }
    destination->length = length;
    polynomial_series_trim(destination);
}

static void polynomial_series_multiply_x(
    CnpPolynomialSeries *destination,
    const CnpPolynomialSeries *source,
    CnpPolynomialBasis basis) {
    polynomial_series_zero(destination);
    if (source->length == 1 && polynomial_coefficient_is_zero(
            source->data[0]))
        return;
    destination->length = source->length + 1;
    if (basis == CNP_POLYNOMIAL_LEGENDRE) {
        destination->data[1] = source->data[0];
        for (int64_t index = 1; index < source->length; ++index) {
            double denominator = 2.0 * index + 1.0;
            destination->data[index + 1] =
                polynomial_coefficient_scale_ratio(
                    source->data[index], index + 1.0, denominator,
                    destination->dtype);
            CnpPolynomialCoefficient lower =
                polynomial_coefficient_add_scale_ratio(
                    destination->data[index - 1], source->data[index],
                    (double)index, denominator,
                    destination->dtype);
            destination->data[index - 1] = lower;
        }
    } else if (basis == CNP_POLYNOMIAL_HERMITE) {
        destination->data[1] = polynomial_coefficient_scale(
            source->data[0], 0.5, destination->dtype);
        for (int64_t index = 1; index < source->length; ++index) {
            destination->data[index + 1] = polynomial_coefficient_scale(
                source->data[index], 0.5, destination->dtype);
            destination->data[index - 1] =
                polynomial_coefficient_add_scale_ratio(
                    destination->data[index - 1], source->data[index],
                    (double)index, 1.0, destination->dtype);
        }
    } else {
        destination->data[0] = source->data[0];
        destination->data[1] = polynomial_coefficient_scale(
            source->data[0], -1.0, destination->dtype);
        for (int64_t index = 1; index < source->length; ++index) {
            destination->data[index + 1] = polynomial_coefficient_scale(
                source->data[index], -(index + 1.0),
                destination->dtype);
            destination->data[index] =
                polynomial_coefficient_add_scale_ratio(
                    destination->data[index], source->data[index],
                    2.0 * index + 1.0, 1.0, destination->dtype);
            destination->data[index - 1] =
                polynomial_coefficient_add_scale_ratio(
                    destination->data[index - 1], source->data[index],
                    -(double)index, 1.0, destination->dtype);
        }
    }
    polynomial_series_trim(destination);
}

static CnpArray *polynomial_multiply_basis(
    const CnpArray *left, const CnpArray *right,
    CnpPolynomialBasis basis, const char *function_name) {
    enum { workspace_series_count = 9 };
    CNP_TYPE result_dtype;
    CNP_TYPE operation_dtype;
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input coefficient arrays are required");
        return NULL;
    }
    if (left->ndim != 1 || right->ndim != 1) {
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
            &operation_dtype)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "coefficient arrays have no common real or complex type");
        return NULL;
    }
    if (left->size > INT64_MAX - right->size + 1) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "polynomial product length overflows");
        return NULL;
    }
    int64_t capacity = left->size + right->size - 1;
    if ((uint64_t)capacity >
            SIZE_MAX / (workspace_series_count * sizeof(
                CnpPolynomialCoefficient))) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "polynomial product workspace size overflows");
        return NULL;
    }
    size_t workspace_count =
        (size_t)capacity * workspace_series_count;
    size_t workspace_bytes = workspace_count *
        sizeof(CnpPolynomialCoefficient);
    CnpPolynomialCoefficient *workspace =
        (CnpPolynomialCoefficient*)cnp_calloc(
            workspace_count, sizeof(CnpPolynomialCoefficient));
    if (!workspace) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    CnpPolynomialSeries series[workspace_series_count];
    for (int index = 0; index < workspace_series_count; ++index) {
        series[index].data = workspace + (size_t)index * capacity;
        series[index].length = 1;
        series[index].capacity = capacity;
        series[index].dtype = operation_dtype;
    }
    CnpPolynomialSeries *left_series = &series[0];
    CnpPolynomialSeries *right_series = &series[1];
    left_series->length = left->size;
    right_series->length = right->size;
    for (int64_t index = 0; index < left->size; ++index) {
        left_series->data[index] = polynomial_round_coefficient(
            polynomial_conversion_coefficient(left, index), operation_dtype);
    }
    for (int64_t index = 0; index < right->size; ++index) {
        right_series->data[index] = polynomial_round_coefficient(
            polynomial_conversion_coefficient(right, index), operation_dtype);
    }
    polynomial_series_trim(left_series);
    polynomial_series_trim(right_series);
    result_dtype = operation_dtype;
    if (basis != CNP_POLYNOMIAL_CHEBYSHEV &&
            (left_series->length == 1 || right_series->length == 1) &&
            polynomial_real_precision_rank(result_dtype) < 2) {
        result_dtype = polynomial_dtype_is_complex(result_dtype)
            ? CNP_CDOUBLE : CNP_DOUBLE;
    }

    CnpPolynomialSeries *output = &series[8];
    if (basis == CNP_POLYNOMIAL_CHEBYSHEV) {
        output->length = left_series->length + right_series->length - 1;
        for (int64_t left_index = 0;
                left_index < left_series->length; ++left_index) {
            for (int64_t right_index = 0;
                    right_index < right_series->length; ++right_index) {
                CnpPolynomialCoefficient product =
                    polynomial_coefficient_multiply(
                        left_series->data[left_index],
                        right_series->data[right_index], result_dtype);
                CnpPolynomialCoefficient half =
                    polynomial_coefficient_scale(
                        product, 0.5, result_dtype);
                int64_t sum_index = left_index + right_index;
                int64_t difference_index = left_index >= right_index
                    ? left_index - right_index
                    : right_index - left_index;
                output->data[sum_index] = polynomial_coefficient_add(
                    output->data[sum_index], half, result_dtype);
                output->data[difference_index] = polynomial_coefficient_add(
                    output->data[difference_index], half, result_dtype);
            }
        }
        polynomial_series_trim(output);
    } else {
        CnpPolynomialSeries *coefficients;
        CnpPolynomialSeries *multiplicand;
        if (left_series->length > right_series->length) {
            coefficients = right_series;
            multiplicand = left_series;
        } else {
            coefficients = left_series;
            multiplicand = right_series;
        }
        CnpPolynomialSeries *c0 = &series[2];
        CnpPolynomialSeries *c1 = &series[3];
        CnpPolynomialSeries *temporary = &series[4];
        CnpPolynomialSeries *work0 = &series[5];
        CnpPolynomialSeries *work1 = &series[6];
        CnpPolynomialSeries *work2 = &series[7];
        polynomial_series_zero(c0);
        polynomial_series_zero(c1);
        if (coefficients->length == 1) {
            polynomial_series_scale(
                c0, multiplicand, coefficients->data[0]);
        } else if (coefficients->length == 2) {
            polynomial_series_scale(
                c0, multiplicand, coefficients->data[0]);
            polynomial_series_scale(
                c1, multiplicand, coefficients->data[1]);
        } else {
            int64_t degree_count = coefficients->length;
            polynomial_series_scale(
                c0, multiplicand,
                coefficients->data[coefficients->length - 2]);
            polynomial_series_scale(
                c1, multiplicand,
                coefficients->data[coefficients->length - 1]);
            for (int64_t offset = 3;
                    offset <= coefficients->length; ++offset) {
                polynomial_series_copy(temporary, c0);
                --degree_count;
                polynomial_series_scale(
                    work0, multiplicand,
                    coefficients->data[coefficients->length - offset]);
                if (basis == CNP_POLYNOMIAL_HERMITE) {
                    polynomial_series_scale_real(
                        work1, c1, 2.0 * (degree_count - 1.0));
                    polynomial_series_subtract(c0, work0, work1);
                    polynomial_series_multiply_x(
                        work0, c1, basis);
                    polynomial_series_scale_real(work1, work0, 2.0);
                    polynomial_series_add(c1, temporary, work1);
                } else if (basis == CNP_POLYNOMIAL_LEGENDRE) {
                    polynomial_series_scale_real(
                        work1, c1, degree_count - 1.0);
                    polynomial_series_divide_real(
                        work2, work1, (double)degree_count);
                    polynomial_series_subtract(c0, work0, work2);
                    polynomial_series_multiply_x(
                        work0, c1, basis);
                    polynomial_series_scale_real(
                        work1, work0, 2.0 * degree_count - 1.0);
                    polynomial_series_divide_real(
                        work2, work1, (double)degree_count);
                    polynomial_series_add(c1, temporary, work2);
                } else {
                    polynomial_series_scale_real(
                        work1, c1, degree_count - 1.0);
                    polynomial_series_divide_real(
                        work2, work1, (double)degree_count);
                    polynomial_series_subtract(c0, work0, work2);
                    polynomial_series_multiply_x(
                        work0, c1, basis);
                    polynomial_series_scale_real(
                        work1, c1, 2.0 * degree_count - 1.0);
                    polynomial_series_subtract(work2, work1, work0);
                    polynomial_series_divide_real(
                        work1, work2, (double)degree_count);
                    polynomial_series_add(c1, temporary, work1);
                }
            }
        }
        polynomial_series_multiply_x(work0, c1, basis);
        if (basis == CNP_POLYNOMIAL_HERMITE) {
            polynomial_series_scale_real(work1, work0, 2.0);
            polynomial_series_add(output, c0, work1);
        } else if (basis == CNP_POLYNOMIAL_LEGENDRE) {
            polynomial_series_add(output, c0, work0);
        } else {
            polynomial_series_subtract(work1, c1, work0);
            polynomial_series_add(output, c0, work1);
        }
    }

    int64_t shape[1] = {output->length};
    CnpArray *result = cnp_array_new(
        1, shape, result_dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
    } else {
        for (int64_t index = 0; index < output->length; ++index) {
            polynomial_store_result_coefficient(
                result, index, output->data[index]);
        }
    }
    cnp_free(workspace, workspace_bytes);
    return result;
}

static bool polynomial_allocate_conversion_workspace(
    int64_t length, CnpPolynomialCoefficient **coefficients,
    double **work0, double **work1, double **work2,
    const char *function_name) {
    if ((uint64_t)length > SIZE_MAX / sizeof(**coefficients) ||
        (uint64_t)length > SIZE_MAX / sizeof(**work0)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "polynomial conversion workspace size overflows");
        return false;
    }
    *coefficients = (CnpPolynomialCoefficient*)cnp_calloc(
        (size_t)length, sizeof(**coefficients));
    *work0 = (double*)cnp_calloc((size_t)length, sizeof(**work0));
    *work1 = (double*)cnp_calloc((size_t)length, sizeof(**work1));
    *work2 = (double*)cnp_calloc((size_t)length, sizeof(**work2));
    if (!*coefficients || !*work0 || !*work1 || !*work2) {
        cnp_free(
            *coefficients,
            *coefficients ? (size_t)length * sizeof(**coefficients) : 0);
        cnp_free(*work0, *work0 ? (size_t)length * sizeof(**work0) : 0);
        cnp_free(*work1, *work1 ? (size_t)length * sizeof(**work1) : 0);
        cnp_free(*work2, *work2 ? (size_t)length * sizeof(**work2) : 0);
        *coefficients = NULL;
        *work0 = NULL;
        *work1 = NULL;
        *work2 = NULL;
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "cannot allocate polynomial conversion workspace");
        return false;
    }
    return true;
}

static CnpArray *polynomial_convert_basis(
    const CnpArray *source, CnpPolynomialBasis basis,
    bool power_to_basis, const char *function_name) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "coefficient array is required");
        return NULL;
    }
    if (source->ndim > 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "coefficient array is not 1-d");
        return NULL;
    }
    if (source->size == 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "coefficient array is empty");
        return NULL;
    }
    if (!polynomial_conversion_dtype_supported(source->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "coefficient dtype is not numeric");
        return NULL;
    }

    int64_t length = source->size;
    while (length > 1 && polynomial_coefficient_is_zero(
        polynomial_conversion_coefficient(source, length - 1)))
        --length;

    CnpPolynomialCoefficient *converted = NULL;
    double *work0 = NULL;
    double *work1 = NULL;
    double *work2 = NULL;
    if (!polynomial_allocate_conversion_workspace(
            length, &converted, &work0, &work1, &work2,
            function_name))
        return NULL;

    if (power_to_basis) {
        double *current = work0;
        double *next = work1;
        current[0] = 1.0;
        for (int64_t degree = 0; degree < length; ++degree) {
            CnpPolynomialCoefficient coefficient =
                polynomial_conversion_coefficient(source, degree);
            for (int64_t term = 0; term <= degree; ++term) {
                converted[term].real += coefficient.real * current[term];
                converted[term].imag += coefficient.imag * current[term];
            }
            if (degree + 1 == length) break;
            memset(next, 0, (size_t)length * sizeof(*next));
            if (basis == CNP_POLYNOMIAL_CHEBYSHEV) {
                next[1] += current[0];
                for (int64_t term = 1; term <= degree; ++term) {
                    next[term - 1] += 0.5 * current[term];
                    next[term + 1] += 0.5 * current[term];
                }
            } else {
                for (int64_t term = 0; term <= degree; ++term) {
                    double denominator = 2.0 * term + 1.0;
                    next[term + 1] += current[term] *
                        (term + 1.0) / denominator;
                    if (term > 0)
                        next[term - 1] += current[term] *
                            term / denominator;
                }
            }
            double *swap = current;
            current = next;
            next = swap;
        }
    } else {
        double *previous2 = work0;
        double *previous1 = work1;
        double *current = work2;
        previous2[0] = 1.0;
        CnpPolynomialCoefficient coefficient =
            polynomial_conversion_coefficient(source, 0);
        converted[0] = coefficient;
        if (length > 1) {
            previous1[1] = 1.0;
            coefficient = polynomial_conversion_coefficient(source, 1);
            converted[1] = coefficient;
        }
        for (int64_t degree = 2; degree < length; ++degree) {
            memset(current, 0, (size_t)length * sizeof(*current));
            double leading_factor =
                basis == CNP_POLYNOMIAL_CHEBYSHEV
                    ? 2.0 : (2.0 * degree - 1.0) / degree;
            double trailing_factor =
                basis == CNP_POLYNOMIAL_CHEBYSHEV
                    ? 1.0 : (degree - 1.0) / degree;
            for (int64_t term = 0; term < degree; ++term)
                current[term + 1] += leading_factor * previous1[term];
            for (int64_t term = 0; term <= degree - 2; ++term)
                current[term] -= trailing_factor * previous2[term];

            coefficient = polynomial_conversion_coefficient(source, degree);
            for (int64_t term = 0; term <= degree; ++term) {
                converted[term].real += coefficient.real * current[term];
                converted[term].imag += coefficient.imag * current[term];
            }
            double *swap = previous2;
            previous2 = previous1;
            previous1 = current;
            current = swap;
        }
    }

    CNP_TYPE result_dtype = polynomial_conversion_result_dtype(
        source->dtype->type_num, basis, power_to_basis);
    int64_t result_length = length;
    while (result_length > 1 && polynomial_result_coefficient_is_zero(
        converted[result_length - 1], result_dtype))
        --result_length;
    int64_t shape[1] = {result_length};
    CnpArray *result = cnp_array_new(
        1, shape, result_dtype, CNP_ORDER_C);
    if (!result) cnp_relabel_error(function_name);
    if (result) {
        for (int64_t index = 0; index < result_length; ++index)
            polynomial_store_result_coefficient(
                result, index, converted[index]);
    }

    size_t coefficient_bytes =
        (size_t)length * sizeof(*converted);
    size_t work_bytes = (size_t)length * sizeof(*work0);
    cnp_free(converted, coefficient_bytes);
    cnp_free(work0, work_bytes);
    cnp_free(work1, work_bytes);
    cnp_free(work2, work_bytes);
    return result;
}

/* =========================================================================
 * poly2cheb - Convert power series to Chebyshev
 * numpy.polynomial.chebyshev.poly2cheb(pol)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_poly2cheb(const CnpArray *pol) {
    return polynomial_convert_basis(
        pol, CNP_POLYNOMIAL_CHEBYSHEV, true, "cnp_poly2cheb");
}

/* =========================================================================
 * cheb2poly - Convert Chebyshev to power series
 * numpy.polynomial.chebyshev.cheb2poly(c)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_cheb2poly(const CnpArray *c) {
    return polynomial_convert_basis(
        c, CNP_POLYNOMIAL_CHEBYSHEV, false, "cnp_cheb2poly");
}

/* =========================================================================
 * Legendre multiplication
 * numpy.polynomial.legendre.legmul(c1, c2)
 * Uses: P_i * P_j = sum_k C_{ijk} P_k (linearization coefficients)
 * Simplified: convert to power, multiply, convert back
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_legmul(const CnpArray *c1, const CnpArray *c2) {
    return polynomial_multiply_basis(
        c1, c2, CNP_POLYNOMIAL_LEGENDRE, "cnp_legmul");
}

/* =========================================================================
 * Legendre fitting
 * numpy.polynomial.legendre.legfit(x, y, deg)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_legfit(const CnpArray *x, const CnpArray *y, int deg) {
    return cnp_polynomial_fit_basis(
        x, y, deg, CNP_POLYNOMIAL_LEGENDRE, "cnp_legfit");
}

/* =========================================================================
 * Hermite multiplication (physicist's)
 * numpy.polynomial.hermite.hermmul(c1, c2)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_hermmul(const CnpArray *c1, const CnpArray *c2) {
    return polynomial_multiply_basis(
        c1, c2, CNP_POLYNOMIAL_HERMITE, "cnp_hermmul");
}

/* =========================================================================
 * Hermite fitting
 * numpy.polynomial.hermite.hermfit(x, y, deg)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_hermfit(const CnpArray *x, const CnpArray *y, int deg) {
    return cnp_polynomial_fit_basis(
        x, y, deg, CNP_POLYNOMIAL_HERMITE, "cnp_hermfit");
}

/* =========================================================================
 * Laguerre multiplication
 * numpy.polynomial.laguerre.lagmul(c1, c2)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_lagmul(const CnpArray *c1, const CnpArray *c2) {
    return polynomial_multiply_basis(
        c1, c2, CNP_POLYNOMIAL_LAGUERRE, "cnp_lagmul");
}

/* =========================================================================
 * Laguerre fitting
 * numpy.polynomial.laguerre.lagfit(x, y, deg)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_lagfit(const CnpArray *x, const CnpArray *y, int deg) {
    return cnp_polynomial_fit_basis(
        x, y, deg, CNP_POLYNOMIAL_LAGUERRE, "cnp_lagfit");
}

/* =========================================================================
 * leg2poly - Convert Legendre to power series
 * numpy.polynomial.legendre.leg2poly(c)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_leg2poly(const CnpArray *c) {
    return polynomial_convert_basis(
        c, CNP_POLYNOMIAL_LEGENDRE, false, "cnp_leg2poly");
}

/* =========================================================================
 * poly2leg - Convert power series to Legendre
 * numpy.polynomial.legendre.poly2leg(pol)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_poly2leg(const CnpArray *pol) {
    return polynomial_convert_basis(
        pol, CNP_POLYNOMIAL_LEGENDRE, true, "cnp_poly2leg");
}
