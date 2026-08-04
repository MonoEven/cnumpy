/**
 * cnumpy polynomial operations - polyfit, polyval, polyder, polyint, roots
 * Corresponds to numpy.polynomial and numpy.poly* functions
 */
#include "../include/cnumpy/cnumpy_internal.h"

#include <float.h>

/* =========================================================================
 * polyder - Derivative of polynomial
 * ========================================================================= */
typedef struct {
    double real;
    double imag;
} CnpPowerCoefficient;

static bool power_calculus_dtype_supported(CNP_TYPE dtype) {
    return dtype == CNP_BOOL || cnp_type_is_integer(dtype) ||
           cnp_type_is_float(dtype) || cnp_type_is_complex(dtype);
}

static CnpPowerCoefficient power_coefficient_at(
    const CnpArray *array, int64_t index) {
    int64_t coordinate = index;
    const void *pointer = cnp_array_at(array, &coordinate);
    CnpPowerCoefficient value = {0.0, 0.0};
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

static uint64_t power_integer_bits_at(
    const CnpArray *array, int64_t index) {
    int64_t coordinate = index;
    const void *pointer = cnp_array_at(array, &coordinate);
    switch (array->dtype->type_num) {
        case CNP_BOOL: return (uint64_t)*(const int8_t*)pointer;
        case CNP_BYTE: return (uint64_t)(int64_t)*(const int8_t*)pointer;
        case CNP_UBYTE: return (uint64_t)*(const uint8_t*)pointer;
        case CNP_SHORT: return (uint64_t)(int64_t)*(const int16_t*)pointer;
        case CNP_USHORT: return (uint64_t)*(const uint16_t*)pointer;
        case CNP_INT: return (uint64_t)(int64_t)*(const int32_t*)pointer;
        case CNP_UINT: return (uint64_t)*(const uint32_t*)pointer;
        case CNP_LONG:
        case CNP_LONGLONG:
            return (uint64_t)*(const int64_t*)pointer;
        case CNP_ULONG:
        case CNP_ULONGLONG:
            return *(const uint64_t*)pointer;
        default:
            return 0;
    }
}

static CNP_TYPE power_derivative_result_dtype(CNP_TYPE source_dtype) {
    switch (source_dtype) {
        case CNP_BOOL:
        case CNP_BYTE:
        case CNP_UBYTE:
        case CNP_SHORT:
        case CNP_USHORT:
        case CNP_INT:
            return CNP_INT;
        case CNP_UINT:
        case CNP_LONG:
        case CNP_LONGLONG:
            return CNP_LONGLONG;
        case CNP_CFLOAT:
        case CNP_CDOUBLE:
        case CNP_CLONGDOUBLE:
            return CNP_CDOUBLE;
        default:
            return CNP_DOUBLE;
    }
}

static void power_store_coefficient(
    CnpArray *result, int64_t index, CnpPowerCoefficient value) {
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
            cnp_set_element_double(
                result->data,
                index * result->dtype->elsize,
                result->dtype->type_num,
                value.real);
            break;
    }
}

CNP_API CnpArray* CNP_CALL cnp_polyder(const CnpArray *p, int m) {
    const char *function_name = "cnp_polyder";
    if (!p) {
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
    if (p->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "coefficient array is not 1-d");
        return NULL;
    }
    if (!power_calculus_dtype_supported(p->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "coefficient dtype is not numeric");
        return NULL;
    }
    if (m == 0) {
        CnpArray *copy = cnp_array_copy(p);
        if (!copy) cnp_relabel_error(function_name);
        return copy;
    }

    int64_t length = p->size;
    int64_t result_length = length > m ? length - m : 0;
    int64_t shape[1] = {result_length};
    CNP_TYPE result_dtype = power_derivative_result_dtype(
        p->dtype->type_num);
    CnpArray *result = cnp_array_new(
        1, shape, result_dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    for (int64_t index = 0; index < result_length; ++index) {
        int64_t degree = length - 1 - index;
        if (result_dtype == CNP_INT) {
            uint32_t bits = (uint32_t)power_integer_bits_at(p, index);
            for (int order = 0; order < m; ++order)
                bits *= (uint32_t)(degree - order);
            memcpy(
                (char*)result->data + index * sizeof(bits),
                &bits, sizeof(bits));
        } else if (result_dtype == CNP_LONGLONG) {
            uint64_t bits = power_integer_bits_at(p, index);
            for (int order = 0; order < m; ++order)
                bits *= (uint64_t)(degree - order);
            memcpy(
                (char*)result->data + index * sizeof(bits),
                &bits, sizeof(bits));
        } else {
            CnpPowerCoefficient value = power_coefficient_at(p, index);
            double factor = 1.0;
            for (int order = 0; order < m; ++order)
                factor *= (double)(degree - order);
            value.real *= factor;
            value.imag *= factor;
            power_store_coefficient(result, index, value);
        }
    }
    return result;
}

/* =========================================================================
 * polyint - Integral of polynomial
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_polyint(const CnpArray *p, int m, const CnpArray *k) {
    const char *function_name = "cnp_polyint";
    if (!p) {
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
    if (p->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "coefficient array is not 1-d");
        return NULL;
    }
    if (!power_calculus_dtype_supported(p->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "coefficient dtype is not numeric");
        return NULL;
    }
    if (m == 0) {
        CnpArray *copy = cnp_array_copy(p);
        if (!copy) cnp_relabel_error(function_name);
        return copy;
    }
    if (k) {
        if (k->ndim > 1 || !power_calculus_dtype_supported(k->dtype->type_num)) {
            cnp_set_error(
                k->ndim > 1 ? CNP_ERR_SHAPE : CNP_ERR_TYPE,
                function_name,
                "integration constants must be a numeric scalar or 1-d array");
            return NULL;
        }
        if (k->size != 1 && k->size < m) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "integration constants must contain one value or at least m values");
            return NULL;
        }
    }
    if ((int64_t)m > INT64_MAX - p->size) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "integrated coefficient count overflows");
        return NULL;
    }

    int64_t result_length = p->size + m;
    if ((uint64_t)result_length >
        SIZE_MAX / sizeof(CnpPowerCoefficient)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "integration workspace size overflows");
        return NULL;
    }
    size_t workspace_bytes =
        (size_t)result_length * sizeof(CnpPowerCoefficient);
    CnpPowerCoefficient *coefficients =
        (CnpPowerCoefficient*)cnp_malloc(workspace_bytes);
    if (!coefficients) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "cannot allocate integration workspace");
        return NULL;
    }
    for (int64_t index = 0; index < p->size; ++index)
        coefficients[index] = power_coefficient_at(p, index);

    int64_t current_length = p->size;
    for (int order = 0; order < m; ++order) {
        for (int64_t index = 0; index < current_length; ++index) {
            double divisor = (double)(current_length - index);
            coefficients[index].real /= divisor;
            coefficients[index].imag /= divisor;
        }
        coefficients[current_length] =
            k ? power_coefficient_at(k, k->size == 1 ? 0 : order)
              : (CnpPowerCoefficient){0.0, 0.0};
        ++current_length;
    }

    bool complex_result = cnp_type_is_complex(p->dtype->type_num) ||
        (k && cnp_type_is_complex(k->dtype->type_num));
    int64_t shape[1] = {result_length};
    CnpArray *result = cnp_array_new(
        1, shape, complex_result ? CNP_CDOUBLE : CNP_DOUBLE,
        CNP_ORDER_C);
    if (!result) cnp_relabel_error(function_name);
    if (result) {
        for (int64_t index = 0; index < result_length; ++index)
            power_store_coefficient(result, index, coefficients[index]);
    }
    cnp_free(coefficients, workspace_bytes);
    return result;
}

static void power_store_integer_bits(
    CnpArray *result, int64_t index, uint64_t bits) {
    char *pointer = (char*)result->data +
        index * result->dtype->elsize;
    switch (result->dtype->type_num) {
        case CNP_BOOL:
        case CNP_BYTE:
        case CNP_UBYTE: {
            uint8_t value = (uint8_t)bits;
            memcpy(pointer, &value, sizeof(value));
            break;
        }
        case CNP_SHORT:
        case CNP_USHORT: {
            uint16_t value = (uint16_t)bits;
            memcpy(pointer, &value, sizeof(value));
            break;
        }
        case CNP_INT:
        case CNP_UINT: {
            uint32_t value = (uint32_t)bits;
            memcpy(pointer, &value, sizeof(value));
            break;
        }
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
            memcpy(pointer, &bits, sizeof(bits));
            break;
        default:
            break;
    }
}

static void power_store_arithmetic_coefficient(
    CnpArray *result, int64_t index, CnpPowerCoefficient value) {
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

static CnpPowerCoefficient power_arithmetic_coefficient_at(
    const CnpArray *array, int64_t index, CNP_TYPE dtype) {
    CnpPowerCoefficient value = power_coefficient_at(array, index);
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
        case CNP_DOUBLE:
        case CNP_LONGDOUBLE:
            value.imag = 0.0;
            break;
        case CNP_CFLOAT:
            value.real = (float)value.real;
            value.imag = (float)value.imag;
            break;
        default:
            break;
    }
    return value;
}

static void power_add_or_subtract_integer(
    CnpArray *result, int64_t index, uint64_t left,
    uint64_t right, bool subtract) {
    uint64_t value;
    switch (result->dtype->elsize) {
        case 1:
            value = (uint8_t)(subtract
                ? (uint8_t)left - (uint8_t)right
                : (uint8_t)left + (uint8_t)right);
            break;
        case 2:
            value = (uint16_t)(subtract
                ? (uint16_t)left - (uint16_t)right
                : (uint16_t)left + (uint16_t)right);
            break;
        case 4:
            value = (uint32_t)(subtract
                ? (uint32_t)left - (uint32_t)right
                : (uint32_t)left + (uint32_t)right);
            break;
        default:
            value = subtract ? left - right : left + right;
            break;
    }
    power_store_integer_bits(result, index, value);
}

static void power_add_or_subtract_real_or_complex(
    CnpArray *result, int64_t index, CnpPowerCoefficient left,
    CnpPowerCoefficient right, bool subtract) {
    CnpPowerCoefficient value;
    switch (result->dtype->type_num) {
        case CNP_HALF: {
            float real = subtract
                ? (float)left.real - (float)right.real
                : (float)left.real + (float)right.real;
            value = (CnpPowerCoefficient){real, 0.0};
            break;
        }
        case CNP_FLOAT: {
            float real = subtract
                ? (float)left.real - (float)right.real
                : (float)left.real + (float)right.real;
            value = (CnpPowerCoefficient){real, 0.0};
            break;
        }
        case CNP_CFLOAT: {
            float real = subtract
                ? (float)left.real - (float)right.real
                : (float)left.real + (float)right.real;
            float imag = subtract
                ? (float)left.imag - (float)right.imag
                : (float)left.imag + (float)right.imag;
            value = (CnpPowerCoefficient){real, imag};
            break;
        }
        default:
            value.real = subtract
                ? left.real - right.real : left.real + right.real;
            value.imag = subtract
                ? left.imag - right.imag : left.imag + right.imag;
            break;
    }
    power_store_arithmetic_coefficient(result, index, value);
}

static CnpArray *power_polynomial_add_or_subtract(
    const CnpArray *left, const CnpArray *right, bool subtract,
    const char *function_name) {
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "polynomial inputs are required");
        return NULL;
    }
    if (left->ndim > 1 || right->ndim > 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "polynomial inputs must be one-dimensional");
        return NULL;
    }
    if (!power_calculus_dtype_supported(left->dtype->type_num) ||
        !power_calculus_dtype_supported(right->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "polynomial input dtype is not numeric");
        return NULL;
    }

    CNP_TYPE result_dtype = cnp_promote_type(
        left->dtype->type_num, right->dtype->type_num);
    if (subtract && result_dtype == CNP_BOOL) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "boolean subtract is not supported");
        return NULL;
    }
    int64_t length = left->size > right->size
        ? left->size : right->size;
    int64_t shape[1] = {length};
    CnpArray *result = cnp_array_new(
        1, shape, result_dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t left_padding = length - left->size;
    int64_t right_padding = length - right->size;
    bool integer_result = result_dtype == CNP_BOOL ||
        cnp_type_is_integer(result_dtype);
    for (int64_t index = 0; index < length; ++index) {
        bool has_left = index >= left_padding;
        bool has_right = index >= right_padding;
        int64_t left_index = index - left_padding;
        int64_t right_index = index - right_padding;
        if (result_dtype == CNP_BOOL) {
            uint64_t left_value = has_left
                ? power_integer_bits_at(left, left_index) : 0;
            uint64_t right_value = has_right
                ? power_integer_bits_at(right, right_index) : 0;
            power_store_integer_bits(
                result, index, left_value != 0 || right_value != 0);
        } else if (integer_result) {
            uint64_t left_value = has_left
                ? power_integer_bits_at(left, left_index) : 0;
            uint64_t right_value = has_right
                ? power_integer_bits_at(right, right_index) : 0;
            power_add_or_subtract_integer(
                result, index, left_value, right_value, subtract);
        } else {
            CnpPowerCoefficient left_value = has_left
                ? power_arithmetic_coefficient_at(
                    left, left_index, result_dtype)
                : (CnpPowerCoefficient){0.0, 0.0};
            CnpPowerCoefficient right_value = has_right
                ? power_arithmetic_coefficient_at(
                    right, right_index, result_dtype)
                : (CnpPowerCoefficient){0.0, 0.0};
            power_add_or_subtract_real_or_complex(
                result, index, left_value, right_value, subtract);
        }
    }
    return result;
}

/* =========================================================================
 * polyadd - Add two polynomials
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_polyadd(const CnpArray *a, const CnpArray *b) {
    return power_polynomial_add_or_subtract(
        a, b, false, "cnp_polyadd");
}

/* =========================================================================
 * polysub - Subtract two polynomials
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_polysub(const CnpArray *a, const CnpArray *b) {
    return power_polynomial_add_or_subtract(
        a, b, true, "cnp_polysub");
}

/* =========================================================================
 * polymul - Multiply two polynomials
 * ========================================================================= */
static bool power_polynomial_coefficient_is_zero(
    const CnpArray *array, int64_t index) {
    if (array->dtype->type_num == CNP_BOOL ||
        cnp_type_is_integer(array->dtype->type_num))
        return power_integer_bits_at(array, index) == 0;
    CnpPowerCoefficient value = power_coefficient_at(array, index);
    return !isnan(value.real) && !isnan(value.imag) &&
           value.real == 0.0 && value.imag == 0.0;
}

static void power_polynomial_normalize_input(
    const CnpArray *array, int64_t *start, int64_t *length,
    bool *synthetic_zero) {
    int64_t first = 0;
    while (first < array->size &&
           power_polynomial_coefficient_is_zero(array, first))
        ++first;
    *synthetic_zero = first == array->size;
    *start = first;
    *length = *synthetic_zero ? 1 : array->size - first;
}

static uint64_t power_polynomial_integer_value(
    const CnpArray *array, int64_t start, bool synthetic_zero,
    int64_t index) {
    return synthetic_zero
        ? 0 : power_integer_bits_at(array, start + index);
}

static CnpPowerCoefficient power_polynomial_arithmetic_value(
    const CnpArray *array, int64_t start, bool synthetic_zero,
    int64_t index, CNP_TYPE dtype) {
    return synthetic_zero
        ? (CnpPowerCoefficient){0.0, 0.0}
        : power_arithmetic_coefficient_at(
            array, start + index, dtype);
}

static uint64_t power_polynomial_integer_multiply_add(
    CNP_TYPE dtype, uint64_t accumulator,
    uint64_t left, uint64_t right) {
    int size = cnp_dtype_itemsize(dtype);
    switch (size) {
        case 1:
            return (uint8_t)(
                (uint16_t)(uint8_t)accumulator +
                (uint16_t)(uint8_t)left * (uint16_t)(uint8_t)right);
        case 2:
            return (uint16_t)(
                (uint32_t)(uint16_t)accumulator +
                (uint32_t)(uint16_t)left * (uint32_t)(uint16_t)right);
        case 4:
            return (uint32_t)(
                (uint32_t)accumulator +
                (uint32_t)left * (uint32_t)right);
        default:
            return accumulator + left * right;
    }
}

static CnpArray *power_polynomial_multiply(
    const CnpArray *left, const CnpArray *right,
    const char *function_name) {
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "polynomial inputs are required");
        return NULL;
    }
    if (left->ndim > 1 || right->ndim > 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "polynomial inputs must be one-dimensional");
        return NULL;
    }
    if (!power_calculus_dtype_supported(left->dtype->type_num) ||
        !power_calculus_dtype_supported(right->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "polynomial input dtype is not numeric");
        return NULL;
    }

    int64_t left_start;
    int64_t left_length;
    bool left_zero;
    power_polynomial_normalize_input(
        left, &left_start, &left_length, &left_zero);
    int64_t right_start;
    int64_t right_length;
    bool right_zero;
    power_polynomial_normalize_input(
        right, &right_start, &right_length, &right_zero);
    if (left_length > INT64_MAX - right_length + 1) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "polynomial product length overflows");
        return NULL;
    }

    const CnpArray *first = left;
    int64_t first_start = left_start;
    int64_t first_length = left_length;
    bool first_zero = left_zero;
    const CnpArray *second = right;
    int64_t second_start = right_start;
    int64_t second_length = right_length;
    bool second_zero = right_zero;
    if (second_length > first_length) {
        first = right;
        first_start = right_start;
        first_length = right_length;
        first_zero = right_zero;
        second = left;
        second_start = left_start;
        second_length = left_length;
        second_zero = left_zero;
    }

    CNP_TYPE result_dtype = cnp_promote_type(
        left->dtype->type_num, right->dtype->type_num);
    int64_t length = first_length + second_length - 1;
    int64_t shape[1] = {length};
    CnpArray *result = cnp_array_new(
        1, shape, result_dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    for (int64_t output = 0; output < length; ++output) {
        int64_t first_begin = output >= second_length
            ? output - (second_length - 1) : 0;
        int64_t first_end = output < first_length
            ? output : first_length - 1;
        if (result_dtype == CNP_BOOL) {
            bool accumulator = false;
            for (int64_t first_index = first_begin;
                 first_index <= first_end; ++first_index) {
                int64_t second_index = output - first_index;
                bool first_value = power_polynomial_integer_value(
                    first, first_start, first_zero, first_index) != 0;
                bool second_value = power_polynomial_integer_value(
                    second, second_start, second_zero, second_index) != 0;
                accumulator = accumulator ||
                    (first_value && second_value);
            }
            power_store_integer_bits(result, output, accumulator);
        } else if (cnp_type_is_integer(result_dtype)) {
            uint64_t accumulator = 0;
            for (int64_t first_index = first_begin;
                 first_index <= first_end; ++first_index) {
                int64_t second_index = output - first_index;
                accumulator = power_polynomial_integer_multiply_add(
                    result_dtype, accumulator,
                    power_polynomial_integer_value(
                        first, first_start, first_zero, first_index),
                    power_polynomial_integer_value(
                        second, second_start, second_zero, second_index));
            }
            power_store_integer_bits(result, output, accumulator);
        } else if (result_dtype == CNP_HALF ||
                   result_dtype == CNP_FLOAT) {
            float accumulator = 0.0f;
            for (int64_t first_index = first_begin;
                 first_index <= first_end; ++first_index) {
                int64_t second_index = output - first_index;
                CnpPowerCoefficient first_value =
                    power_polynomial_arithmetic_value(
                        first, first_start, first_zero,
                        first_index, result_dtype);
                CnpPowerCoefficient second_value =
                    power_polynomial_arithmetic_value(
                        second, second_start, second_zero,
                        second_index, result_dtype);
                accumulator +=
                    (float)first_value.real * (float)second_value.real;
            }
            power_store_arithmetic_coefficient(
                result, output,
                (CnpPowerCoefficient){accumulator, 0.0});
        } else if (result_dtype == CNP_CFLOAT) {
            float accumulator_real = 0.0f;
            float accumulator_imag = 0.0f;
            for (int64_t first_index = first_begin;
                 first_index <= first_end; ++first_index) {
                int64_t second_index = output - first_index;
                CnpPowerCoefficient first_value =
                    power_polynomial_arithmetic_value(
                        first, first_start, first_zero,
                        first_index, result_dtype);
                CnpPowerCoefficient second_value =
                    power_polynomial_arithmetic_value(
                        second, second_start, second_zero,
                        second_index, result_dtype);
                float first_real = (float)first_value.real;
                float first_imag = (float)first_value.imag;
                float second_real = (float)second_value.real;
                float second_imag = (float)second_value.imag;
                accumulator_real += first_real * second_real -
                    first_imag * second_imag;
                accumulator_imag += first_real * second_imag +
                    first_imag * second_real;
            }
            power_store_arithmetic_coefficient(
                result, output,
                (CnpPowerCoefficient){
                    accumulator_real, accumulator_imag});
        } else {
            CnpPowerCoefficient accumulator = {0.0, 0.0};
            for (int64_t first_index = first_begin;
                 first_index <= first_end; ++first_index) {
                int64_t second_index = output - first_index;
                CnpPowerCoefficient first_value =
                    power_polynomial_arithmetic_value(
                        first, first_start, first_zero,
                        first_index, result_dtype);
                CnpPowerCoefficient second_value =
                    power_polynomial_arithmetic_value(
                        second, second_start, second_zero,
                        second_index, result_dtype);
                accumulator.real +=
                    first_value.real * second_value.real -
                    first_value.imag * second_value.imag;
                accumulator.imag +=
                    first_value.real * second_value.imag +
                    first_value.imag * second_value.real;
            }
            power_store_arithmetic_coefficient(
                result, output, accumulator);
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_polymul(const CnpArray *a, const CnpArray *b) {
    return power_polynomial_multiply(a, b, "cnp_polymul");
}

/* =========================================================================
 * polydiv - Divide two polynomials (returns quotient and remainder)
 * ========================================================================= */
static CNP_TYPE power_polynomial_division_input_dtype(CNP_TYPE dtype) {
    return dtype == CNP_BOOL || cnp_type_is_integer(dtype)
        ? CNP_DOUBLE : dtype;
}

static CnpPowerCoefficient power_polynomial_division_round(
    CnpPowerCoefficient value, CNP_TYPE dtype) {
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
        case CNP_DOUBLE:
        case CNP_LONGDOUBLE:
            value.imag = 0.0;
            break;
        case CNP_CFLOAT:
            value.real = (float)value.real;
            value.imag = (float)value.imag;
            break;
        default:
            break;
    }
    return value;
}

#if defined(_MSC_VER)
#pragma float_control(precise, on, push)
#endif

static CnpPowerCoefficient power_polynomial_division_input_value(
    const CnpArray *array, int64_t index, CNP_TYPE dtype) {
    CnpPowerCoefficient value = power_coefficient_at(array, index);
    switch (dtype) {
        case CNP_HALF: {
            float real = (float)value.real + 0.0f;
            value.real = cnp_half_to_float(
                cnp_float_to_half((double)real));
            value.imag = 0.0;
            break;
        }
        case CNP_FLOAT:
            value.real = (float)value.real + 0.0f;
            value.imag = 0.0;
            break;
        case CNP_DOUBLE:
        case CNP_LONGDOUBLE:
            value.real += 0.0;
            value.imag = 0.0;
            break;
        case CNP_CFLOAT:
            value.real = (float)value.real + 0.0f;
            value.imag = (float)value.imag + 0.0f;
            break;
        default:
            value.real += 0.0;
            value.imag += 0.0;
            break;
    }
    return value;
}

static bool power_polynomial_division_double_is_finite(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) !=
        UINT64_C(0x7ff0000000000000);
}

static CNP_TYPE power_polynomial_division_scalar_dtype(
    CnpPowerCoefficient value, bool complex_value) {
    if (complex_value) {
        if (power_polynomial_division_double_is_finite(value.real) &&
                power_polynomial_division_double_is_finite(value.imag) &&
                value.real > -3.4e38 && value.real < 3.4e38 &&
                value.imag > -3.4e38 && value.imag < 3.4e38)
            return CNP_CFLOAT;
        return CNP_CDOUBLE;
    }
    if (!power_polynomial_division_double_is_finite(value.real))
        return CNP_HALF;
    if (value.real > -65000.0 && value.real < 65000.0)
        return CNP_HALF;
    if (value.real > -3.4e38 && value.real < 3.4e38)
        return CNP_FLOAT;
    return CNP_DOUBLE;
}

static double power_polynomial_division_positive_infinity(void) {
    uint64_t bits = UINT64_C(0x7ff0000000000000);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static double power_polynomial_division_negative_nan(void) {
    uint64_t bits = UINT64_C(0xfff8000000000000);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static CnpPowerCoefficient power_polynomial_division_reciprocal(
    CnpPowerCoefficient denominator) {
    if (denominator.real == 0.0 && denominator.imag == 0.0) {
        return (CnpPowerCoefficient){
            power_polynomial_division_positive_infinity(),
            power_polynomial_division_negative_nan()};
    }
    if (fabs(denominator.real) >= fabs(denominator.imag)) {
        double ratio = denominator.imag / denominator.real;
        double scale = denominator.real + denominator.imag * ratio;
        return (CnpPowerCoefficient){
            1.0 / scale, -ratio / scale};
    }
    double ratio = denominator.real / denominator.imag;
    double scale = denominator.real * ratio + denominator.imag;
    return (CnpPowerCoefficient){
        ratio / scale, -1.0 / scale};
}

static CnpPowerCoefficient power_polynomial_division_multiply(
    CnpPowerCoefficient left, CnpPowerCoefficient right,
    CNP_TYPE dtype, bool fused) {
    left = power_polynomial_division_round(left, dtype);
    right = power_polynomial_division_round(right, dtype);
    switch (dtype) {
        case CNP_HALF: {
            float real = (float)left.real * (float)right.real;
            return power_polynomial_division_round(
                (CnpPowerCoefficient){real, 0.0}, dtype);
        }
        case CNP_FLOAT: {
            float real = (float)left.real * (float)right.real;
            return (CnpPowerCoefficient){real, 0.0};
        }
        case CNP_CFLOAT: {
            float left_real = (float)left.real;
            float left_imag = (float)left.imag;
            float right_real = (float)right.real;
            float right_imag = (float)right.imag;
            float real = fused
                ? fmaf(
                    left_real, right_real,
                    -(left_imag * right_imag))
                : left_real * right_real - left_imag * right_imag;
            float imag = fused
                ? fmaf(
                    left_real, right_imag,
                    left_imag * right_real)
                : left_real * right_imag + left_imag * right_real;
            return (CnpPowerCoefficient){real, imag};
        }
        case CNP_CDOUBLE:
        case CNP_CLONGDOUBLE:
            return fused
                ? (CnpPowerCoefficient){
                    fma(left.real, right.real,
                        -(left.imag * right.imag)),
                    fma(left.real, right.imag,
                        left.imag * right.real)}
                : (CnpPowerCoefficient){
                    left.real * right.real - left.imag * right.imag,
                    left.real * right.imag + left.imag * right.real};
        default:
            return (CnpPowerCoefficient){
                left.real * right.real, 0.0};
    }
}

static CnpPowerCoefficient power_polynomial_division_subtract(
    CnpPowerCoefficient left, CnpPowerCoefficient right,
    CNP_TYPE arithmetic_dtype, CNP_TYPE result_dtype) {
    left = power_polynomial_division_round(left, arithmetic_dtype);
    right = power_polynomial_division_round(right, arithmetic_dtype);
    CnpPowerCoefficient value;
    switch (arithmetic_dtype) {
        case CNP_HALF: {
            float real = (float)left.real - (float)right.real;
            value = (CnpPowerCoefficient){real, 0.0};
            break;
        }
        case CNP_FLOAT: {
            float real = (float)left.real - (float)right.real;
            value = (CnpPowerCoefficient){real, 0.0};
            break;
        }
        case CNP_CFLOAT: {
            float real = (float)left.real - (float)right.real;
            float imag = (float)left.imag - (float)right.imag;
            value = (CnpPowerCoefficient){real, imag};
            break;
        }
        default:
            value = (CnpPowerCoefficient){
                left.real - right.real,
                left.imag - right.imag};
            break;
    }
    return power_polynomial_division_round(value, result_dtype);
}

static bool power_polynomial_division_close_to_zero(
    CnpPowerCoefficient value) {
    if (!power_polynomial_division_double_is_finite(value.real) ||
        !power_polynomial_division_double_is_finite(value.imag))
        return false;
    return hypot(value.real, value.imag) <= 1.0e-8;
}

CNP_API CNP_STATUS CNP_CALL cnp_polydiv(
    const CnpArray *a, const CnpArray *b,
    CnpArray **quotient, CnpArray **remainder) {
    const char *function_name = "cnp_polydiv";
    if (quotient) *quotient = NULL;
    if (remainder) *remainder = NULL;
    if (!quotient || !remainder) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "output slots are required");
        return CNP_ERR_GENERIC;
    }
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "polynomial inputs are required");
        return CNP_ERR_GENERIC;
    }
    if (a->ndim > 1 || b->ndim > 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "polynomial inputs must be one-dimensional");
        return CNP_ERR_SHAPE;
    }
    if (a->size == 0 || b->size == 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "polynomial inputs must not be empty");
        return CNP_ERR_SHAPE;
    }
    if (!power_calculus_dtype_supported(a->dtype->type_num) ||
        !power_calculus_dtype_supported(b->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "polynomial input dtype is not numeric");
        return CNP_ERR_TYPE;
    }

    CNP_TYPE dividend_dtype = power_polynomial_division_input_dtype(
        a->dtype->type_num);
    CNP_TYPE divisor_dtype = power_polynomial_division_input_dtype(
        b->dtype->type_num);
    CNP_TYPE result_dtype = cnp_promote_type(
        dividend_dtype, divisor_dtype);
    int64_t quotient_length = a->size >= b->size
        ? a->size - b->size + 1 : 1;
    int64_t quotient_shape[1] = {quotient_length};
    CnpArray *quotient_result = cnp_array_new(
        1, quotient_shape, result_dtype, CNP_ORDER_C);
    if (!quotient_result) {
        cnp_relabel_error(function_name);
        return CNP_ERR_MEMORY;
    }
    for (int64_t index = 0; index < quotient_length; ++index) {
        power_store_arithmetic_coefficient(
            quotient_result, index,
            (CnpPowerCoefficient){0.0, 0.0});
    }

    int64_t workspace_shape[1] = {a->size};
    CnpArray *remainder_workspace = cnp_array_new(
        1, workspace_shape, result_dtype, CNP_ORDER_C);
    if (!remainder_workspace) {
        cnp_array_free(quotient_result);
        cnp_relabel_error(function_name);
        return CNP_ERR_MEMORY;
    }
    for (int64_t index = 0; index < a->size; ++index) {
        CnpPowerCoefficient value = power_polynomial_division_input_value(
            a, index, dividend_dtype);
        power_store_arithmetic_coefficient(
            remainder_workspace, index,
            power_polynomial_division_round(value, result_dtype));
    }

    CnpPowerCoefficient leading_divisor =
        power_polynomial_division_input_value(b, 0, divisor_dtype);
    bool complex_result = cnp_type_is_complex(result_dtype);
    CnpPowerCoefficient scale = cnp_type_is_complex(divisor_dtype)
        ? power_polynomial_division_reciprocal(leading_divisor)
        : (CnpPowerCoefficient){1.0 / leading_divisor.real, 0.0};
    int64_t division_steps = a->size >= b->size
        ? a->size - b->size + 1 : 0;
    for (int64_t step = 0; step < division_steps; ++step) {
        CnpPowerCoefficient remainder_value =
            power_arithmetic_coefficient_at(
                remainder_workspace, step, result_dtype);
        CnpPowerCoefficient factor = power_polynomial_division_multiply(
            scale, remainder_value,
            complex_result ? CNP_CDOUBLE : CNP_DOUBLE, false);
        power_store_arithmetic_coefficient(
            quotient_result, step,
            power_polynomial_division_round(factor, result_dtype));

        CNP_TYPE scalar_dtype = power_polynomial_division_scalar_dtype(
            factor, complex_result);
        CNP_TYPE product_dtype = cnp_promote_type(
            scalar_dtype, divisor_dtype);
        CNP_TYPE subtraction_dtype = cnp_promote_type(
            result_dtype, product_dtype);
        for (int64_t divisor_index = 0;
             divisor_index < b->size; ++divisor_index) {
            CnpPowerCoefficient divisor_value =
                power_polynomial_division_input_value(
                    b, divisor_index, divisor_dtype);
            CnpPowerCoefficient product =
                power_polynomial_division_multiply(
                    factor, divisor_value, product_dtype, true);
            int64_t remainder_index = step + divisor_index;
            CnpPowerCoefficient current =
                power_arithmetic_coefficient_at(
                    remainder_workspace, remainder_index,
                    result_dtype);
            CnpPowerCoefficient updated =
                power_polynomial_division_subtract(
                    current, product,
                    subtraction_dtype, result_dtype);
            power_store_arithmetic_coefficient(
                remainder_workspace, remainder_index, updated);
        }
    }

    int64_t remainder_start = 0;
    while (remainder_start + 1 < a->size) {
        CnpPowerCoefficient value = power_arithmetic_coefficient_at(
            remainder_workspace, remainder_start, result_dtype);
        if (!power_polynomial_division_close_to_zero(value)) break;
        ++remainder_start;
    }

    CnpArray *remainder_result = remainder_workspace;
    if (remainder_start != 0) {
        int64_t remainder_length = a->size - remainder_start;
        int64_t remainder_shape[1] = {remainder_length};
        remainder_result = cnp_array_new(
            1, remainder_shape, result_dtype, CNP_ORDER_C);
        if (!remainder_result) {
            cnp_array_free(remainder_workspace);
            cnp_array_free(quotient_result);
            cnp_relabel_error(function_name);
            return CNP_ERR_MEMORY;
        }
        for (int64_t index = 0; index < remainder_length; ++index) {
            CnpPowerCoefficient value = power_arithmetic_coefficient_at(
                remainder_workspace, remainder_start + index,
                result_dtype);
            power_store_arithmetic_coefficient(
                remainder_result, index, value);
        }
        cnp_array_free(remainder_workspace);
    }

    *quotient = quotient_result;
    *remainder = remainder_result;
    return CNP_OK;
}

#if defined(_MSC_VER)
#pragma float_control(pop)
#endif

/* =========================================================================
 * polyval - Evaluate polynomial at points
 * p[0]*x^(n-1) + p[1]*x^(n-2) + ... + p[n-1]
 * ========================================================================= */
typedef struct {
    uint64_t integer_bits;
    CnpPowerCoefficient arithmetic;
} CnpPowerEvaluationValue;

static const void *power_polynomial_evaluation_flat_pointer(
    const CnpArray *array, int64_t flat_index) {
    int64_t offset = array->offset;
    int64_t remaining = flat_index;
    for (int dimension = array->ndim - 1; dimension >= 0; --dimension) {
        int64_t coordinate = remaining % array->shape[dimension];
        remaining /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return (const char*)array->data + offset;
}

static uint64_t power_polynomial_evaluation_integer_bits(
    const void *pointer, CNP_TYPE dtype) {
    switch (dtype) {
        case CNP_BOOL: return (uint64_t)*(const int8_t*)pointer;
        case CNP_BYTE: return (uint64_t)(int64_t)*(const int8_t*)pointer;
        case CNP_UBYTE: return (uint64_t)*(const uint8_t*)pointer;
        case CNP_SHORT: return (uint64_t)(int64_t)*(const int16_t*)pointer;
        case CNP_USHORT: return (uint64_t)*(const uint16_t*)pointer;
        case CNP_INT: return (uint64_t)(int64_t)*(const int32_t*)pointer;
        case CNP_UINT: return (uint64_t)*(const uint32_t*)pointer;
        case CNP_LONG:
        case CNP_LONGLONG:
            return (uint64_t)*(const int64_t*)pointer;
        case CNP_ULONG:
        case CNP_ULONGLONG:
            return *(const uint64_t*)pointer;
        default:
            return 0;
    }
}

static CnpPowerEvaluationValue power_polynomial_evaluation_value_at(
    const CnpArray *array, int64_t flat_index) {
    const void *pointer = power_polynomial_evaluation_flat_pointer(
        array, flat_index);
    CnpPowerEvaluationValue value = {0};
    switch (array->dtype->type_num) {
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
            value.integer_bits = power_polynomial_evaluation_integer_bits(
                pointer, array->dtype->type_num);
            break;
        case CNP_HALF:
            value.arithmetic.real = cnp_half_to_float(
                *(const uint16_t*)pointer);
            break;
        case CNP_FLOAT:
            value.arithmetic.real = *(const float*)pointer;
            break;
        case CNP_DOUBLE:
            value.arithmetic.real = *(const double*)pointer;
            break;
        case CNP_LONGDOUBLE:
            value.arithmetic.real = (double)*(const long double*)pointer;
            break;
        case CNP_CFLOAT: {
            const cnp_cfloat *complex_value = (const cnp_cfloat*)pointer;
            value.arithmetic.real = complex_value->real;
            value.arithmetic.imag = complex_value->imag;
            break;
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *complex_value = (const cnp_cdouble*)pointer;
            value.arithmetic.real = complex_value->real;
            value.arithmetic.imag = complex_value->imag;
            break;
        }
        case CNP_CLONGDOUBLE: {
            const cnp_clongdouble *complex_value =
                (const cnp_clongdouble*)pointer;
            value.arithmetic.real = (double)complex_value->real;
            value.arithmetic.imag = (double)complex_value->imag;
            break;
        }
        default:
            break;
    }
    return value;
}

static int64_t power_polynomial_evaluation_signed_value(
    CnpPowerEvaluationValue value, CNP_TYPE dtype) {
    switch (dtype) {
        case CNP_BOOL: return value.integer_bits != 0;
        case CNP_BYTE: return (int8_t)(uint8_t)value.integer_bits;
        case CNP_SHORT: return (int16_t)(uint16_t)value.integer_bits;
        case CNP_INT: return (int32_t)(uint32_t)value.integer_bits;
        case CNP_LONG:
        case CNP_LONGLONG:
            return (int64_t)value.integer_bits;
        default:
            return (int64_t)value.integer_bits;
    }
}

static uint64_t power_polynomial_evaluation_unsigned_value(
    CnpPowerEvaluationValue value, CNP_TYPE dtype) {
    switch (dtype) {
        case CNP_BOOL: return value.integer_bits != 0;
        case CNP_UBYTE: return (uint8_t)value.integer_bits;
        case CNP_USHORT: return (uint16_t)value.integer_bits;
        case CNP_UINT: return (uint32_t)value.integer_bits;
        default: return value.integer_bits;
    }
}

static uint64_t power_polynomial_evaluation_integer_mask(CNP_TYPE dtype) {
    switch (cnp_dtype_itemsize(dtype)) {
        case 1: return UINT64_C(0xff);
        case 2: return UINT64_C(0xffff);
        case 4: return UINT64_C(0xffffffff);
        default: return UINT64_MAX;
    }
}

static uint64_t power_polynomial_evaluation_cast_integer_bits(
    CnpPowerEvaluationValue value, CNP_TYPE source_dtype,
    CNP_TYPE destination_dtype) {
    uint64_t bits = cnp_type_is_unsigned(source_dtype)
        ? power_polynomial_evaluation_unsigned_value(value, source_dtype)
        : (uint64_t)power_polynomial_evaluation_signed_value(
            value, source_dtype);
    return bits & power_polynomial_evaluation_integer_mask(
        destination_dtype);
}

static int power_polynomial_evaluation_scalar_category(CNP_TYPE dtype) {
    if (dtype == CNP_BOOL) return 0;
    if (cnp_type_is_integer(dtype)) return 1;
    return 2;
}

static CNP_TYPE power_polynomial_evaluation_minimum_unsigned(
    uint64_t value) {
    if (value <= UINT8_MAX) return CNP_UBYTE;
    if (value <= UINT16_MAX) return CNP_USHORT;
    if (value <= UINT32_MAX) return CNP_UINT;
    return CNP_ULONGLONG;
}

static CNP_TYPE power_polynomial_evaluation_minimum_signed(int64_t value) {
    if (value >= INT8_MIN && value <= INT8_MAX) return CNP_BYTE;
    if (value >= INT16_MIN && value <= INT16_MAX) return CNP_SHORT;
    if (value >= INT32_MIN && value <= INT32_MAX) return CNP_INT;
    return CNP_LONGLONG;
}

static CNP_TYPE power_polynomial_evaluation_minimum_scalar_dtype(
    CnpPowerEvaluationValue value, CNP_TYPE dtype) {
    if (dtype == CNP_BOOL) return CNP_BOOL;
    if (cnp_type_is_integer(dtype)) {
        if (cnp_type_is_unsigned(dtype))
            return power_polynomial_evaluation_minimum_unsigned(
                power_polynomial_evaluation_unsigned_value(value, dtype));
        int64_t signed_value = power_polynomial_evaluation_signed_value(
            value, dtype);
        return signed_value < 0
            ? power_polynomial_evaluation_minimum_signed(signed_value)
            : power_polynomial_evaluation_minimum_unsigned(
                (uint64_t)signed_value);
    }
    if (dtype == CNP_CFLOAT) return CNP_CFLOAT;
    if (dtype == CNP_CDOUBLE || dtype == CNP_CLONGDOUBLE) {
        if (power_polynomial_division_double_is_finite(
                value.arithmetic.real) &&
            power_polynomial_division_double_is_finite(
                value.arithmetic.imag) &&
            value.arithmetic.real > -3.4e38 &&
            value.arithmetic.real < 3.4e38 &&
            value.arithmetic.imag > -3.4e38 &&
            value.arithmetic.imag < 3.4e38)
            return CNP_CFLOAT;
        return dtype == CNP_CLONGDOUBLE ? CNP_CLONGDOUBLE : CNP_CDOUBLE;
    }
    if (dtype == CNP_HALF) return CNP_HALF;
    double real = value.arithmetic.real;
    if (!power_polynomial_division_double_is_finite(real) ||
        (real > -65000.0 && real < 65000.0))
        return CNP_HALF;
    if (dtype == CNP_FLOAT || (real > -3.4e38 && real < 3.4e38))
        return CNP_FLOAT;
    return dtype == CNP_LONGDOUBLE ? CNP_LONGDOUBLE : CNP_DOUBLE;
}

static bool power_polynomial_evaluation_integer_fits(
    CnpPowerEvaluationValue value, CNP_TYPE source_dtype,
    CNP_TYPE destination_dtype) {
    int bits = cnp_dtype_itemsize(destination_dtype) * 8;
    bool destination_unsigned = cnp_type_is_unsigned(destination_dtype);
    if (cnp_type_is_unsigned(source_dtype)) {
        uint64_t source = power_polynomial_evaluation_unsigned_value(
            value, source_dtype);
        if (destination_unsigned) {
            uint64_t maximum = bits == 64
                ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
            return source <= maximum;
        }
        uint64_t maximum = bits == 64
            ? (uint64_t)INT64_MAX
            : (UINT64_C(1) << (bits - 1)) - 1;
        return source <= maximum;
    }

    int64_t source = power_polynomial_evaluation_signed_value(
        value, source_dtype);
    if (destination_unsigned) {
        if (source < 0) return false;
        uint64_t maximum = bits == 64
            ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
        return (uint64_t)source <= maximum;
    }
    int64_t minimum = bits == 64
        ? INT64_MIN : -(INT64_C(1) << (bits - 1));
    int64_t maximum = bits == 64
        ? INT64_MAX : (INT64_C(1) << (bits - 1)) - 1;
    return source >= minimum && source <= maximum;
}

static CNP_TYPE power_polynomial_evaluation_add_dtype(
    CNP_TYPE array_dtype, CnpPowerEvaluationValue scalar,
    CNP_TYPE scalar_dtype) {
    int array_category = power_polynomial_evaluation_scalar_category(
        array_dtype);
    int scalar_category = power_polynomial_evaluation_scalar_category(
        scalar_dtype);
    if (array_category == 1 && scalar_category == 1 &&
        power_polynomial_evaluation_integer_fits(
            scalar, scalar_dtype, array_dtype))
        return array_dtype;
    CNP_TYPE effective_scalar_dtype = scalar_category > array_category
        ? scalar_dtype
        : power_polynomial_evaluation_minimum_scalar_dtype(
            scalar, scalar_dtype);
    return cnp_promote_type(array_dtype, effective_scalar_dtype);
}

static CnpPowerCoefficient power_polynomial_evaluation_arithmetic_cast(
    CnpPowerEvaluationValue value, CNP_TYPE source_dtype,
    CNP_TYPE destination_dtype) {
    if (source_dtype == CNP_BOOL || cnp_type_is_integer(source_dtype)) {
        bool source_unsigned = cnp_type_is_unsigned(source_dtype);
        if (destination_dtype == CNP_HALF ||
            destination_dtype == CNP_FLOAT ||
            destination_dtype == CNP_CFLOAT) {
            float real = source_unsigned
                ? (float)power_polynomial_evaluation_unsigned_value(
                    value, source_dtype)
                : (float)power_polynomial_evaluation_signed_value(
                    value, source_dtype);
            return power_polynomial_division_round(
                (CnpPowerCoefficient){real, 0.0}, destination_dtype);
        }
        double real = source_unsigned
            ? (double)power_polynomial_evaluation_unsigned_value(
                value, source_dtype)
            : (double)power_polynomial_evaluation_signed_value(
                value, source_dtype);
        return power_polynomial_division_round(
            (CnpPowerCoefficient){real, 0.0}, destination_dtype);
    }
    return power_polynomial_division_round(
        value.arithmetic, destination_dtype);
}

static CnpPowerEvaluationValue power_polynomial_evaluation_multiply(
    CnpPowerEvaluationValue left, CNP_TYPE left_dtype,
    CnpPowerEvaluationValue right, CNP_TYPE right_dtype,
    CNP_TYPE result_dtype) {
    CnpPowerEvaluationValue result = {0};
    if (result_dtype == CNP_BOOL) {
        result.integer_bits = left.integer_bits != 0 &&
            right.integer_bits != 0;
        return result;
    }
    if (cnp_type_is_integer(result_dtype)) {
        uint64_t left_bits =
            power_polynomial_evaluation_cast_integer_bits(
                left, left_dtype, result_dtype);
        uint64_t right_bits =
            power_polynomial_evaluation_cast_integer_bits(
                right, right_dtype, result_dtype);
        result.integer_bits = (left_bits * right_bits) &
            power_polynomial_evaluation_integer_mask(result_dtype);
        return result;
    }
    CnpPowerCoefficient left_value =
        power_polynomial_evaluation_arithmetic_cast(
            left, left_dtype, result_dtype);
    CnpPowerCoefficient right_value =
        power_polynomial_evaluation_arithmetic_cast(
            right, right_dtype, result_dtype);
    result.arithmetic = power_polynomial_division_multiply(
        left_value, right_value, result_dtype, true);
    return result;
}

static CnpPowerEvaluationValue power_polynomial_evaluation_add(
    CnpPowerEvaluationValue left, CNP_TYPE left_dtype,
    CnpPowerEvaluationValue right, CNP_TYPE right_dtype,
    CNP_TYPE result_dtype) {
    CnpPowerEvaluationValue result = {0};
    if (result_dtype == CNP_BOOL) {
        result.integer_bits = left.integer_bits != 0 ||
            right.integer_bits != 0;
        return result;
    }
    if (cnp_type_is_integer(result_dtype)) {
        uint64_t left_bits =
            power_polynomial_evaluation_cast_integer_bits(
                left, left_dtype, result_dtype);
        uint64_t right_bits =
            power_polynomial_evaluation_cast_integer_bits(
                right, right_dtype, result_dtype);
        result.integer_bits = (left_bits + right_bits) &
            power_polynomial_evaluation_integer_mask(result_dtype);
        return result;
    }

    CnpPowerCoefficient left_value =
        power_polynomial_evaluation_arithmetic_cast(
            left, left_dtype, result_dtype);
    CnpPowerCoefficient right_value =
        power_polynomial_evaluation_arithmetic_cast(
            right, right_dtype, result_dtype);
    switch (result_dtype) {
        case CNP_HALF:
        case CNP_FLOAT: {
            float real = (float)left_value.real + (float)right_value.real;
            result.arithmetic = power_polynomial_division_round(
                (CnpPowerCoefficient){real, 0.0}, result_dtype);
            break;
        }
        case CNP_CFLOAT: {
            float real = (float)left_value.real + (float)right_value.real;
            float imag = (float)left_value.imag + (float)right_value.imag;
            result.arithmetic = (CnpPowerCoefficient){real, imag};
            break;
        }
        default:
            result.arithmetic = (CnpPowerCoefficient){
                left_value.real + right_value.real,
                left_value.imag + right_value.imag};
            break;
    }
    return result;
}

static void power_polynomial_evaluation_store(
    CnpArray *result, int64_t index, CnpPowerEvaluationValue value) {
    const char *pointer = (const char*)
        power_polynomial_evaluation_flat_pointer(result, index);
    int64_t storage_index =
        (pointer - (const char*)result->data) / result->dtype->elsize;
    if (result->dtype->type_num == CNP_BOOL ||
        cnp_type_is_integer(result->dtype->type_num)) {
        power_store_integer_bits(result, storage_index, value.integer_bits);
        return;
    }
    power_store_arithmetic_coefficient(
        result, storage_index, value.arithmetic);
}

#if defined(_MSC_VER)
#pragma float_control(precise, on, push)
#endif

CNP_API CnpArray* CNP_CALL cnp_polyval(
    const CnpArray *p, const CnpArray *x) {
    const char *function_name = "cnp_polyval";
    if (!p || !x) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "coefficient and point arrays are required");
        return NULL;
    }
    if (p->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "coefficient array must be one-dimensional");
        return NULL;
    }
    if (!power_calculus_dtype_supported(p->dtype->type_num) ||
        !power_calculus_dtype_supported(x->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "coefficient and point dtypes must be numeric");
        return NULL;
    }

    CNP_TYPE result_dtype = x->dtype->type_num;
    for (int64_t coefficient_index = 0;
         coefficient_index < p->size; ++coefficient_index) {
        CNP_TYPE product_dtype = cnp_promote_type(
            result_dtype, x->dtype->type_num);
        CnpPowerEvaluationValue coefficient =
            power_polynomial_evaluation_value_at(p, coefficient_index);
        result_dtype = power_polynomial_evaluation_add_dtype(
            product_dtype, coefficient, p->dtype->type_num);
    }

    CNP_ORDER result_order =
        (x->flags & CNP_ARRAY_F_CONTIGUOUS) != 0 &&
        (x->flags & CNP_ARRAY_C_CONTIGUOUS) == 0
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        x->ndim, x->shape, result_dtype, result_order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    bool matching_contiguous_layout =
        ((x->flags & CNP_ARRAY_C_CONTIGUOUS) != 0 &&
         (result->flags & CNP_ARRAY_C_CONTIGUOUS) != 0) ||
        ((x->flags & CNP_ARRAY_F_CONTIGUOUS) != 0 &&
         (result->flags & CNP_ARRAY_F_CONTIGUOUS) != 0);
    if (p->dtype->type_num == CNP_DOUBLE &&
        x->dtype->type_num == CNP_DOUBLE &&
        result_dtype == CNP_DOUBLE && matching_contiguous_layout) {
        const double *point_values = (const double*)(
            (const char*)x->data + x->offset);
        double *result_values = (double*)result->data;
        memset(result_values, 0, (size_t)x->size * sizeof(double));
        for (int64_t coefficient_index = 0;
             coefficient_index < p->size; ++coefficient_index) {
            CnpPowerEvaluationValue coefficient =
                power_polynomial_evaluation_value_at(
                    p, coefficient_index);
            for (int64_t point_index = 0;
                 point_index < x->size; ++point_index) {
                double product = result_values[point_index] *
                    point_values[point_index];
                result_values[point_index] = product;
            }
            for (int64_t point_index = 0;
                 point_index < x->size; ++point_index) {
                double sum = result_values[point_index] +
                    coefficient.arithmetic.real;
                result_values[point_index] = sum;
            }
        }
        return result;
    }

    for (int64_t point_index = 0;
         point_index < x->size; ++point_index) {
        CnpPowerEvaluationValue point =
            power_polynomial_evaluation_value_at(x, point_index);
        CnpPowerEvaluationValue current = {0};
        CNP_TYPE current_dtype = x->dtype->type_num;
        for (int64_t coefficient_index = 0;
             coefficient_index < p->size; ++coefficient_index) {
            CNP_TYPE product_dtype = cnp_promote_type(
                current_dtype, x->dtype->type_num);
            CnpPowerEvaluationValue product =
                power_polynomial_evaluation_multiply(
                    current, current_dtype,
                    point, x->dtype->type_num,
                    product_dtype);
            CnpPowerEvaluationValue coefficient =
                power_polynomial_evaluation_value_at(
                    p, coefficient_index);
            CNP_TYPE next_dtype = power_polynomial_evaluation_add_dtype(
                product_dtype, coefficient, p->dtype->type_num);
            current = power_polynomial_evaluation_add(
                product, product_dtype,
                coefficient, p->dtype->type_num,
                next_dtype);
            current_dtype = next_dtype;
        }
        power_polynomial_evaluation_store(result, point_index, current);
    }
    return result;
}

#if defined(_MSC_VER)
#pragma float_control(pop)
#endif

/* =========================================================================
 * polyfit - Least squares polynomial fit
 * ========================================================================= */
static bool power_polynomial_fit_dtype_supported(CNP_TYPE dtype) {
    return dtype == CNP_BOOL || cnp_type_is_integer(dtype) ||
        dtype == CNP_HALF || dtype == CNP_FLOAT || dtype == CNP_DOUBLE ||
        dtype == CNP_CFLOAT || dtype == CNP_CDOUBLE;
}

static CnpPowerCoefficient power_polynomial_fit_value_at(
    const CnpArray *array, int64_t row, int64_t column) {
    int64_t coordinates[2] = {row, column};
    const void *pointer = cnp_array_at(array, coordinates);
    CnpPowerCoefficient value = {0.0, 0.0};
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
        default:
            value.real = cnp_get_element_double(
                pointer, 0, array->dtype->type_num);
            break;
    }
    return value;
}

static CnpPowerCoefficient power_polynomial_fit_contiguous_value(
    const CnpArray *array, int64_t index) {
    switch (array->dtype->type_num) {
        case CNP_FLOAT:
            return (CnpPowerCoefficient){
                ((const float*)array->data)[index], 0.0};
        case CNP_DOUBLE:
            return (CnpPowerCoefficient){
                ((const double*)array->data)[index], 0.0};
        case CNP_CFLOAT: {
            const cnp_cfloat *value =
                &((const cnp_cfloat*)array->data)[index];
            return (CnpPowerCoefficient){value->real, value->imag};
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *value =
                &((const cnp_cdouble*)array->data)[index];
            return (CnpPowerCoefficient){value->real, value->imag};
        }
        default:
            return (CnpPowerCoefficient){
                cnp_get_element_double(
                    array->data,
                    index * array->dtype->elsize,
                    array->dtype->type_num),
                0.0};
    }
}

static CnpPowerCoefficient power_polynomial_fit_conjugate(
    CnpPowerCoefficient value) {
    value.imag = -value.imag;
    return value;
}

static double power_polynomial_fit_default_rcond(
    const CnpArray *x) {
    double epsilon;
    switch (x->dtype->type_num) {
        case CNP_HALF:
            epsilon = 0.0009765625;
            break;
        case CNP_FLOAT:
        case CNP_CFLOAT:
            epsilon = FLT_EPSILON;
            break;
        default:
            epsilon = DBL_EPSILON;
            break;
    }
    return (double)x->shape[0] * epsilon;
}

static bool power_polynomial_fit_reduce_tall_system(
    CnpArray **lhs_pointer,
    const CnpArray *y,
    int64_t rhs_count,
    CNP_TYPE arithmetic_dtype,
    CnpPowerCoefficient **transformed_rhs_result,
    size_t *transformed_rhs_bytes_result,
    const char *function_name) {
    CnpArray *lhs = *lhs_pointer;
    int64_t rows = lhs->shape[0];
    int64_t columns = lhs->shape[1];
    int64_t rhs_value_count = 0;
    size_t householder_bytes;
    size_t transformed_rhs_bytes = 0;
    CnpPowerCoefficient *householder = NULL;
    CnpPowerCoefficient *transformed_rhs = NULL;
    CnpArray *reduced_lhs = NULL;
    int64_t reduced_shape[2] = {columns, columns};

    *transformed_rhs_result = NULL;
    *transformed_rhs_bytes_result = 0;
    if ((uint64_t)rows > SIZE_MAX / sizeof(CnpPowerCoefficient)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "Householder workspace size overflows");
        return false;
    }
    if (rhs_count != 0 && rows > INT64_MAX / rhs_count) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "transformed right-hand-side workspace size overflows");
        return false;
    }
    rhs_value_count = rows * rhs_count;
    if ((uint64_t)rhs_value_count >
            SIZE_MAX / sizeof(CnpPowerCoefficient)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "transformed right-hand-side workspace size overflows");
        return false;
    }

    householder_bytes = (size_t)rows * sizeof(CnpPowerCoefficient);
    householder = (CnpPowerCoefficient*)cnp_malloc(householder_bytes);
    if (!householder) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate Householder workspace");
        return false;
    }
    if (rhs_value_count != 0) {
        transformed_rhs_bytes =
            (size_t)rhs_value_count * sizeof(CnpPowerCoefficient);
        transformed_rhs = (CnpPowerCoefficient*)cnp_malloc(
            transformed_rhs_bytes);
        if (!transformed_rhs) {
            cnp_free(householder, householder_bytes);
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "failed to allocate transformed right-hand-side workspace");
            return false;
        }
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t rhs = 0; rhs < rhs_count; ++rhs) {
                transformed_rhs[row * rhs_count + rhs] =
                    power_polynomial_fit_value_at(y, row, rhs);
            }
        }
    }

    for (int64_t column = 0; column < columns; ++column) {
        int64_t length = rows - column;
        double norm = 0.0;
        double first_absolute;
        double vector_norm = 0.0;
        CnpPowerCoefficient phase;
        CnpPowerCoefficient alpha;
        if (arithmetic_dtype == CNP_DOUBLE) {
            double *matrix = (double*)lhs->data;
            double first;
            double real_alpha;
            for (int64_t index = 0; index < length; ++index) {
                double value = matrix[
                    (column + index) * columns + column];
                householder[index].real = value;
                householder[index].imag = 0.0;
                norm = hypot(norm, value);
            }
            if (norm == 0.0) continue;
            first = householder[0].real;
            real_alpha = -copysign(norm, first == 0.0 ? 1.0 : first);
            householder[0].real -= real_alpha;
            for (int64_t index = 0; index < length; ++index) {
                vector_norm = hypot(
                    vector_norm, householder[index].real);
            }
            if (vector_norm == 0.0) continue;
            for (int64_t index = 0; index < length; ++index) {
                householder[index].real /= vector_norm;
            }
            for (int64_t target_column = column;
                 target_column < columns; ++target_column) {
                double product = 0.0;
                for (int64_t index = 0; index < length; ++index) {
                    product += householder[index].real * matrix[
                        (column + index) * columns + target_column];
                }
                product *= 2.0;
                for (int64_t index = 0; index < length; ++index) {
                    matrix[(column + index) * columns + target_column] -=
                        householder[index].real * product;
                }
            }
            for (int64_t rhs = 0; rhs < rhs_count; ++rhs) {
                double product = 0.0;
                for (int64_t index = 0; index < length; ++index) {
                    product += householder[index].real *
                        transformed_rhs[
                            (column + index) * rhs_count + rhs].real;
                }
                product *= 2.0;
                for (int64_t index = 0; index < length; ++index) {
                    transformed_rhs[
                        (column + index) * rhs_count + rhs].real -=
                            householder[index].real * product;
                }
            }
            matrix[column * columns + column] = real_alpha;
            for (int64_t row = column + 1; row < rows; ++row) {
                matrix[row * columns + column] = 0.0;
            }
            continue;
        }
        for (int64_t index = 0; index < length; ++index) {
            CnpPowerCoefficient value =
                power_polynomial_fit_contiguous_value(
                    lhs, (column + index) * columns + column);
            householder[index] = value;
            norm = hypot(norm, hypot(value.real, value.imag));
        }
        if (norm == 0.0) continue;
        first_absolute = hypot(
            householder[0].real, householder[0].imag);
        phase = first_absolute == 0.0
            ? (CnpPowerCoefficient){1.0, 0.0}
            : (CnpPowerCoefficient){
                householder[0].real / first_absolute,
                householder[0].imag / first_absolute};
        alpha = (CnpPowerCoefficient){
            -phase.real * norm, -phase.imag * norm};
        householder[0].real -= alpha.real;
        householder[0].imag -= alpha.imag;
        for (int64_t index = 0; index < length; ++index) {
            vector_norm = hypot(
                vector_norm,
                hypot(
                    householder[index].real,
                    householder[index].imag));
        }
        if (vector_norm == 0.0) continue;
        for (int64_t index = 0; index < length; ++index) {
            householder[index].real /= vector_norm;
            householder[index].imag /= vector_norm;
        }

        for (int64_t target_column = column;
             target_column < columns; ++target_column) {
            CnpPowerCoefficient product = {0.0, 0.0};
            for (int64_t index = 0; index < length; ++index) {
                CnpPowerCoefficient matrix_value =
                    power_polynomial_fit_contiguous_value(
                        lhs,
                        (column + index) * columns + target_column);
                CnpPowerCoefficient term =
                    power_polynomial_division_multiply(
                        power_polynomial_fit_conjugate(
                            householder[index]),
                        matrix_value, arithmetic_dtype, false);
                product.real += term.real;
                product.imag += term.imag;
            }
            product.real *= 2.0;
            product.imag *= 2.0;
            for (int64_t index = 0; index < length; ++index) {
                int64_t matrix_index =
                    (column + index) * columns + target_column;
                CnpPowerCoefficient matrix_value =
                    power_polynomial_fit_contiguous_value(
                        lhs, matrix_index);
                CnpPowerCoefficient correction =
                    power_polynomial_division_multiply(
                        householder[index], product,
                        arithmetic_dtype, false);
                matrix_value.real -= correction.real;
                matrix_value.imag -= correction.imag;
                power_store_coefficient(
                    lhs, matrix_index, matrix_value);
            }
        }
        for (int64_t rhs = 0; rhs < rhs_count; ++rhs) {
            CnpPowerCoefficient product = {0.0, 0.0};
            for (int64_t index = 0; index < length; ++index) {
                CnpPowerCoefficient term =
                    power_polynomial_division_multiply(
                        power_polynomial_fit_conjugate(
                            householder[index]),
                        transformed_rhs[(column + index) * rhs_count + rhs],
                        arithmetic_dtype, false);
                product.real += term.real;
                product.imag += term.imag;
            }
            product.real *= 2.0;
            product.imag *= 2.0;
            for (int64_t index = 0; index < length; ++index) {
                CnpPowerCoefficient correction =
                    power_polynomial_division_multiply(
                        householder[index], product,
                        arithmetic_dtype, false);
                transformed_rhs[
                    (column + index) * rhs_count + rhs].real -=
                        correction.real;
                transformed_rhs[
                    (column + index) * rhs_count + rhs].imag -=
                        correction.imag;
            }
        }
        power_store_coefficient(
            lhs, column * columns + column, alpha);
        for (int64_t row = column + 1; row < rows; ++row) {
            power_store_coefficient(
                lhs, row * columns + column,
                (CnpPowerCoefficient){0.0, 0.0});
        }
    }

    reduced_lhs = cnp_array_new(
        2, reduced_shape, arithmetic_dtype, CNP_ORDER_C);
    if (!reduced_lhs) {
        if (transformed_rhs)
            cnp_free(transformed_rhs, transformed_rhs_bytes);
        cnp_free(householder, householder_bytes);
        cnp_relabel_error(function_name);
        return false;
    }
    for (int64_t row = 0; row < columns; ++row) {
        for (int64_t column = 0; column < columns; ++column) {
            CnpPowerCoefficient value = row <= column
                ? power_polynomial_fit_contiguous_value(
                    lhs, row * columns + column)
                : (CnpPowerCoefficient){0.0, 0.0};
            power_store_coefficient(
                reduced_lhs, row * columns + column, value);
        }
    }

    cnp_free(householder, householder_bytes);
    cnp_array_free(lhs);
    *lhs_pointer = reduced_lhs;
    *transformed_rhs_result = transformed_rhs;
    *transformed_rhs_bytes_result = transformed_rhs_bytes;
    return true;
}

static CNP_TYPE polynomial_fit_normalized_dtype(CNP_TYPE dtype) {
    if (dtype == CNP_BOOL || cnp_type_is_integer(dtype)) return CNP_DOUBLE;
    return dtype;
}

static CnpPowerCoefficient polynomial_fit_round(
    CnpPowerCoefficient value, CNP_TYPE dtype) {
    return power_polynomial_division_round(value, dtype);
}

static CnpPowerCoefficient polynomial_fit_add(
    CnpPowerCoefficient left, CnpPowerCoefficient right, CNP_TYPE dtype) {
    return polynomial_fit_round(
        (CnpPowerCoefficient){
            left.real + right.real, left.imag + right.imag},
        dtype);
}

static CnpPowerCoefficient polynomial_fit_subtract(
    CnpPowerCoefficient left, CnpPowerCoefficient right, CNP_TYPE dtype) {
    return polynomial_fit_round(
        (CnpPowerCoefficient){
            left.real - right.real, left.imag - right.imag},
        dtype);
}

static CnpPowerCoefficient polynomial_fit_scale(
    CnpPowerCoefficient value, double factor, CNP_TYPE dtype) {
    return power_polynomial_division_multiply(
        value,
        polynomial_fit_round(
            (CnpPowerCoefficient){factor, 0.0}, dtype),
        dtype,
        false);
}

static CnpPowerCoefficient polynomial_fit_divide_real(
    CnpPowerCoefficient value, double denominator, CNP_TYPE dtype) {
    if (dtype == CNP_FLOAT || dtype == CNP_CFLOAT) {
        float divisor = (float)denominator;
        return polynomial_fit_round(
            (CnpPowerCoefficient){
                (float)value.real / divisor,
                (float)value.imag / divisor},
            dtype);
    }
    return polynomial_fit_round(
        (CnpPowerCoefficient){
            value.real / denominator, value.imag / denominator},
        dtype);
}

static bool polynomial_fit_value_is_finite(CnpPowerCoefficient value) {
    return isfinite(value.real) && isfinite(value.imag);
}

static double polynomial_fit_round_real(double value, CNP_TYPE dtype) {
    return dtype == CNP_FLOAT || dtype == CNP_CFLOAT
        ? (double)(float)value : value;
}

static double polynomial_fit_column_norm(
    const CnpArray *lhs, int64_t rows, int64_t columns,
    int64_t column, CNP_TYPE design_dtype) {
    double sum = 0.0;
    for (int64_t row = 0; row < rows; ++row) {
        CnpPowerCoefficient value = polynomial_fit_round(
            power_polynomial_fit_contiguous_value(
                lhs, row * columns + column),
            design_dtype);
        double square = polynomial_fit_round_real(
            polynomial_fit_round_real(
                value.real * value.real, design_dtype) +
            polynomial_fit_round_real(
                value.imag * value.imag, design_dtype),
            design_dtype);
        sum = polynomial_fit_round_real(sum + square, design_dtype);
    }
    if (design_dtype == CNP_FLOAT || design_dtype == CNP_CFLOAT)
        return (double)sqrtf((float)sum);
    return sqrt(sum);
}

static bool polynomial_fit_fill_design_row(
    CnpArray *lhs, int64_t row, int64_t coefficient_count,
    CnpPowerCoefficient x_value, CnpPolynomialBasis basis,
    CNP_TYPE design_dtype) {
    CnpPowerCoefficient previous = {1.0, 0.0};
    CnpPowerCoefficient current;
    x_value = polynomial_fit_round(x_value, design_dtype);
    if (!polynomial_fit_value_is_finite(x_value)) return false;

    if (basis == CNP_POLYNOMIAL_POWER) {
        CnpPowerCoefficient power = previous;
        for (int64_t exponent = 0;
             exponent < coefficient_count; ++exponent) {
            int64_t column = coefficient_count - 1 - exponent;
            power_store_coefficient(
                lhs, row * coefficient_count + column, power);
            power = power_polynomial_division_multiply(
                power, x_value, design_dtype, false);
        }
        return true;
    }

    power_store_coefficient(lhs, row * coefficient_count, previous);
    if (coefficient_count == 1) return true;
    switch (basis) {
        case CNP_POLYNOMIAL_CHEBYSHEV:
        case CNP_POLYNOMIAL_LEGENDRE:
            current = x_value;
            break;
        case CNP_POLYNOMIAL_HERMITE:
            current = polynomial_fit_scale(x_value, 2.0, design_dtype);
            break;
        case CNP_POLYNOMIAL_LAGUERRE:
            current = polynomial_fit_subtract(
                (CnpPowerCoefficient){1.0, 0.0},
                x_value,
                design_dtype);
            break;
        default:
            return false;
    }
    power_store_coefficient(lhs, row * coefficient_count + 1, current);

    for (int64_t degree = 2;
         degree < coefficient_count; ++degree) {
        CnpPowerCoefficient next;
        if (basis == CNP_POLYNOMIAL_CHEBYSHEV) {
            CnpPowerCoefficient twice_x =
                polynomial_fit_scale(x_value, 2.0, design_dtype);
            next = polynomial_fit_subtract(
                power_polynomial_division_multiply(
                    current, twice_x, design_dtype, false),
                previous,
                design_dtype);
        } else if (basis == CNP_POLYNOMIAL_LEGENDRE) {
            CnpPowerCoefficient leading = polynomial_fit_scale(
                power_polynomial_division_multiply(
                    current, x_value, design_dtype, false),
                2.0 * (double)degree - 1.0,
                design_dtype);
            CnpPowerCoefficient trailing = polynomial_fit_scale(
                previous, (double)degree - 1.0, design_dtype);
            next = polynomial_fit_divide_real(
                polynomial_fit_subtract(
                    leading, trailing, design_dtype),
                (double)degree,
                design_dtype);
        } else if (basis == CNP_POLYNOMIAL_HERMITE) {
            CnpPowerCoefficient twice_x =
                polynomial_fit_scale(x_value, 2.0, design_dtype);
            next = polynomial_fit_subtract(
                power_polynomial_division_multiply(
                    current, twice_x, design_dtype, false),
                polynomial_fit_scale(
                    previous,
                    2.0 * ((double)degree - 1.0),
                    design_dtype),
                design_dtype);
        } else {
            CnpPowerCoefficient factor = polynomial_fit_subtract(
                (CnpPowerCoefficient){
                    2.0 * (double)degree - 1.0, 0.0},
                x_value,
                design_dtype);
            next = polynomial_fit_divide_real(
                polynomial_fit_subtract(
                    power_polynomial_division_multiply(
                        current, factor, design_dtype, false),
                    polynomial_fit_scale(
                        previous,
                        (double)degree - 1.0,
                        design_dtype),
                    design_dtype),
                (double)degree,
                design_dtype);
        }
        power_store_coefficient(
            lhs, row * coefficient_count + degree, next);
        previous = current;
        current = next;
    }
    return true;
}

static double polynomial_fit_singular_value(
    const CnpArray *singular_values, int64_t index) {
    if (singular_values->dtype->type_num == CNP_FLOAT)
        return ((const float*)singular_values->data)[index];
    return ((const double*)singular_values->data)[index];
}

static CnpArray *polynomial_fit_solve(
    CnpArray *lhs, const CnpArray *y,
    int64_t rhs_count, CNP_TYPE result_dtype, double rcond,
    CnpPowerCoefficient *scale_reciprocals, size_t scale_bytes,
    const char *function_name) {
    int64_t sample_count = lhs->shape[0];
    int64_t coefficient_count = lhs->shape[1];
    int64_t result_shape[2] = {coefficient_count, rhs_count};
    int64_t singular_count;
    int64_t solve_row_count;
    int64_t projection_count = 0;
    bool reduced_tall_system = false;
    size_t projection_bytes = 0;
    size_t transformed_rhs_bytes = 0;
    CnpPowerCoefficient *projected_rhs = NULL;
    CnpPowerCoefficient *transformed_rhs = NULL;
    CnpArray *u = NULL;
    CnpArray *singular_values = NULL;
    CnpArray *vh = NULL;
    CnpArray *result = NULL;
    CNP_STATUS status;

    if (sample_count > coefficient_count) {
        if (!power_polynomial_fit_reduce_tall_system(
                &lhs, y, rhs_count, result_dtype,
                &transformed_rhs, &transformed_rhs_bytes,
                function_name)) {
            cnp_array_free(lhs);
            cnp_free(scale_reciprocals, scale_bytes);
            return NULL;
        }
        reduced_tall_system = true;
    }

    status = cnp_linalg_svd_v2(
        lhs, false, true, false, &u, &singular_values, &vh);
    cnp_array_free(lhs);
    if (status != CNP_OK) {
        if (u) cnp_array_free(u);
        if (singular_values) cnp_array_free(singular_values);
        if (vh) cnp_array_free(vh);
        if (transformed_rhs)
            cnp_free(transformed_rhs, transformed_rhs_bytes);
        cnp_free(scale_reciprocals, scale_bytes);
        cnp_relabel_error(function_name);
        return NULL;
    }

    singular_count = singular_values->shape[0];
    solve_row_count = reduced_tall_system
        ? coefficient_count : sample_count;
    if (rhs_count != 0 && singular_count > INT64_MAX / rhs_count) {
        cnp_array_free(u);
        cnp_array_free(singular_values);
        cnp_array_free(vh);
        if (transformed_rhs)
            cnp_free(transformed_rhs, transformed_rhs_bytes);
        cnp_free(scale_reciprocals, scale_bytes);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "projected right-hand-side workspace size overflows");
        return NULL;
    }
    projection_count = singular_count * rhs_count;
    if ((uint64_t)projection_count >
            SIZE_MAX / sizeof(CnpPowerCoefficient)) {
        cnp_array_free(u);
        cnp_array_free(singular_values);
        cnp_array_free(vh);
        if (transformed_rhs)
            cnp_free(transformed_rhs, transformed_rhs_bytes);
        cnp_free(scale_reciprocals, scale_bytes);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "projected right-hand-side workspace size overflows");
        return NULL;
    }
    if (projection_count != 0) {
        projection_bytes =
            (size_t)projection_count * sizeof(CnpPowerCoefficient);
        projected_rhs = (CnpPowerCoefficient*)cnp_malloc(projection_bytes);
        if (!projected_rhs) {
            cnp_array_free(u);
            cnp_array_free(singular_values);
            cnp_array_free(vh);
            if (transformed_rhs)
                cnp_free(transformed_rhs, transformed_rhs_bytes);
            cnp_free(scale_reciprocals, scale_bytes);
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "failed to allocate projected right-hand-side workspace");
            return NULL;
        }
    }

    result = cnp_array_new(
        y->ndim, result_shape, result_dtype, CNP_ORDER_C);
    if (!result) {
        if (projected_rhs) cnp_free(projected_rhs, projection_bytes);
        cnp_array_free(u);
        cnp_array_free(singular_values);
        cnp_array_free(vh);
        if (transformed_rhs)
            cnp_free(transformed_rhs, transformed_rhs_bytes);
        cnp_free(scale_reciprocals, scale_bytes);
        cnp_relabel_error(function_name);
        return NULL;
    }

    {
        double largest = singular_count == 0
            ? 0.0 : polynomial_fit_singular_value(singular_values, 0);
        double cutoff = rcond * largest;
        for (int64_t singular = 0;
             singular < singular_count; ++singular) {
            double singular_value = polynomial_fit_singular_value(
                singular_values, singular);
            double reciprocal = singular_value > cutoff
                ? 1.0 / singular_value : 0.0;
            for (int64_t rhs = 0; rhs < rhs_count; ++rhs) {
                CnpPowerCoefficient projection = {0.0, 0.0};
                for (int64_t row = 0; row < solve_row_count; ++row) {
                    CnpPowerCoefficient u_value =
                        power_polynomial_fit_contiguous_value(
                            u, row * singular_count + singular);
                    CnpPowerCoefficient y_value = reduced_tall_system
                        ? transformed_rhs[row * rhs_count + rhs]
                        : power_polynomial_fit_value_at(y, row, rhs);
                    CnpPowerCoefficient product =
                        power_polynomial_division_multiply(
                            power_polynomial_fit_conjugate(u_value),
                            y_value,
                            result_dtype,
                            false);
                    projection = polynomial_fit_add(
                        projection, product, result_dtype);
                }
                projection = polynomial_fit_scale(
                    projection, reciprocal, result_dtype);
                projected_rhs[singular * rhs_count + rhs] = projection;
            }
        }
    }
    if (transformed_rhs) {
        cnp_free(transformed_rhs, transformed_rhs_bytes);
        transformed_rhs = NULL;
    }

    for (int64_t coefficient = 0;
         coefficient < coefficient_count; ++coefficient) {
        for (int64_t rhs = 0; rhs < rhs_count; ++rhs) {
            CnpPowerCoefficient coefficient_value = {0.0, 0.0};
            for (int64_t singular = 0;
                 singular < singular_count; ++singular) {
                CnpPowerCoefficient vh_value =
                    power_polynomial_fit_contiguous_value(
                        vh, singular * coefficient_count + coefficient);
                CnpPowerCoefficient product =
                    power_polynomial_division_multiply(
                        power_polynomial_fit_conjugate(vh_value),
                        projected_rhs[singular * rhs_count + rhs],
                        result_dtype,
                        false);
                coefficient_value = polynomial_fit_add(
                    coefficient_value, product, result_dtype);
            }
            coefficient_value = power_polynomial_division_multiply(
                coefficient_value,
                scale_reciprocals[coefficient],
                result_dtype,
                false);
            power_store_coefficient(
                result,
                coefficient * rhs_count + rhs,
                coefficient_value);
        }
    }

    if (projected_rhs) cnp_free(projected_rhs, projection_bytes);
    cnp_array_free(u);
    cnp_array_free(singular_values);
    cnp_array_free(vh);
    cnp_free(scale_reciprocals, scale_bytes);
    return result;
}

CnpArray* cnp_polynomial_fit_basis(
    const CnpArray *x, const CnpArray *y, int deg,
    CnpPolynomialBasis basis, const char *function_name) {
    int64_t sample_count;
    int64_t coefficient_count;
    int64_t rhs_count;
    int64_t lhs_shape[2];
    size_t scale_bytes;
    CNP_TYPE design_dtype;
    CNP_TYPE result_dtype;
    double rcond;
    CnpPowerCoefficient *scale_reciprocals = NULL;
    CnpArray *lhs = NULL;

    if (!x || !y) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "x and y arrays are required");
        return NULL;
    }
    if (deg < 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "degree must be nonnegative");
        return NULL;
    }
    if (x->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "x must be one-dimensional");
        return NULL;
    }
    if (x->shape[0] == 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "x must not be empty");
        return NULL;
    }
    if (y->ndim != 1 && y->ndim != 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "y must be one- or two-dimensional");
        return NULL;
    }
    if (x->shape[0] != y->shape[0]) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "x and y must have the same length");
        return NULL;
    }
    if (!power_polynomial_fit_dtype_supported(x->dtype->type_num) ||
            !power_polynomial_fit_dtype_supported(y->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "x and y dtypes must be supported by linear algebra");
        return NULL;
    }
    if ((basis != CNP_POLYNOMIAL_POWER &&
            (x->dtype->type_num == CNP_HALF ||
             y->dtype->type_num == CNP_HALF)) ||
            (basis == CNP_POLYNOMIAL_POWER &&
             y->dtype->type_num == CNP_HALF)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "float16 is not supported by linear algebra");
        return NULL;
    }

    sample_count = x->shape[0];
    coefficient_count = (int64_t)deg + 1;
    rhs_count = y->ndim == 1 ? 1 : y->shape[1];
    if (basis == CNP_POLYNOMIAL_POWER) {
        result_dtype = cnp_type_is_complex(x->dtype->type_num) ||
            cnp_type_is_complex(y->dtype->type_num)
            ? CNP_CDOUBLE : CNP_DOUBLE;
        design_dtype = result_dtype;
    } else {
        CNP_TYPE x_dtype = polynomial_fit_normalized_dtype(
            x->dtype->type_num);
        CNP_TYPE y_dtype = polynomial_fit_normalized_dtype(
            y->dtype->type_num);
        design_dtype = x_dtype;
        result_dtype = cnp_promote_type(x_dtype, y_dtype);
        if (result_dtype != CNP_FLOAT && result_dtype != CNP_DOUBLE &&
                result_dtype != CNP_CFLOAT &&
                result_dtype != CNP_CDOUBLE) {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "x and y dtypes must be supported by linear algebra");
            return NULL;
        }
    }

    rcond = power_polynomial_fit_default_rcond(x);
    lhs_shape[0] = sample_count;
    lhs_shape[1] = coefficient_count;
    lhs = cnp_array_new(2, lhs_shape, result_dtype, CNP_ORDER_C);
    if (!lhs) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if ((uint64_t)coefficient_count >
            SIZE_MAX / sizeof(CnpPowerCoefficient)) {
        cnp_array_free(lhs);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "column scale workspace size overflows");
        return NULL;
    }
    scale_bytes = (size_t)coefficient_count * sizeof(CnpPowerCoefficient);
    scale_reciprocals = (CnpPowerCoefficient*)cnp_calloc(
        (size_t)coefficient_count, sizeof(CnpPowerCoefficient));
    if (!scale_reciprocals) {
        cnp_array_free(lhs);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate column scale workspace");
        return NULL;
    }

    for (int64_t row = 0; row < sample_count; ++row) {
        CnpPowerCoefficient x_value =
            power_polynomial_fit_value_at(x, row, 0);
        if (!polynomial_fit_fill_design_row(
                lhs, row, coefficient_count,
                x_value, basis, design_dtype)) {
            cnp_array_free(lhs);
            cnp_free(scale_reciprocals, scale_bytes);
            cnp_set_error(
                CNP_ERR_CONVERGENCE, function_name,
                "SVD did not converge in Linear Least Squares");
            return NULL;
        }
    }

    for (int64_t column = 0;
         column < coefficient_count; ++column) {
        double scale = polynomial_fit_column_norm(
            lhs, sample_count, coefficient_count,
            column, design_dtype);
        double reciprocal;
        if (scale == 0.0) scale = 1.0;
        reciprocal = design_dtype == CNP_FLOAT ||
                design_dtype == CNP_CFLOAT
            ? (double)((float)1.0f / (float)scale)
            : 1.0 / scale;
        scale_reciprocals[column] =
            (CnpPowerCoefficient){reciprocal, 0.0};
        for (int64_t row = 0; row < sample_count; ++row) {
            int64_t index = row * coefficient_count + column;
            CnpPowerCoefficient value = polynomial_fit_round(
                power_polynomial_fit_contiguous_value(lhs, index),
                design_dtype);
            value = power_polynomial_division_multiply(
                value,
                scale_reciprocals[column],
                design_dtype,
                false);
            power_store_coefficient(lhs, index, value);
        }
    }

    return polynomial_fit_solve(
        lhs, y, rhs_count, result_dtype, rcond,
        scale_reciprocals, scale_bytes, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_polyfit(
    const CnpArray *x, const CnpArray *y, int deg) {
    return cnp_polynomial_fit_basis(
        x, y, deg, CNP_POLYNOMIAL_POWER, "cnp_polyfit");
}

/* =========================================================================
 * poly - Build polynomial from roots
 * ========================================================================= */
static bool power_polynomial_construction_dtype(
    CNP_TYPE root_dtype, CNP_TYPE *result_dtype) {
    if (root_dtype == CNP_FLOAT) {
        *result_dtype = CNP_FLOAT;
        return true;
    }
    if (root_dtype == CNP_CFLOAT) {
        *result_dtype = CNP_CFLOAT;
        return true;
    }
    if (root_dtype == CNP_CDOUBLE || root_dtype == CNP_CLONGDOUBLE) {
        *result_dtype = CNP_CDOUBLE;
        return true;
    }
    if (root_dtype == CNP_BOOL || cnp_type_is_integer(root_dtype) ||
            root_dtype == CNP_HALF || root_dtype == CNP_DOUBLE ||
            root_dtype == CNP_LONGDOUBLE) {
        *result_dtype = CNP_DOUBLE;
        return true;
    }
    return false;
}

static int power_polynomial_root_compare(
    const void *left_pointer, const void *right_pointer) {
    const CnpPowerCoefficient *left =
        (const CnpPowerCoefficient*)left_pointer;
    const CnpPowerCoefficient *right =
        (const CnpPowerCoefficient*)right_pointer;
    if (left->real < right->real) return -1;
    if (left->real > right->real) return 1;
    if (left->imag < right->imag) return -1;
    if (left->imag > right->imag) return 1;
    return 0;
}

static bool power_polynomial_roots_are_conjugate_complete(
    const CnpArray *roots, bool *complete) {
    const char *function_name = "cnp_poly";
    int64_t root_count = roots->size;
    size_t root_bytes;
    CnpPowerCoefficient *values;
    CnpPowerCoefficient *conjugates;
    *complete = true;
    if (root_count == 0) return true;
    if ((uint64_t)root_count >
            SIZE_MAX / sizeof(CnpPowerCoefficient)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "conjugate-root workspace size overflows");
        return false;
    }
    root_bytes = (size_t)root_count * sizeof(CnpPowerCoefficient);
    values = (CnpPowerCoefficient*)cnp_malloc(root_bytes);
    if (!values) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate conjugate-root workspace");
        return false;
    }
    conjugates = (CnpPowerCoefficient*)cnp_malloc(root_bytes);
    if (!conjugates) {
        cnp_free(values, root_bytes);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate conjugate-root workspace");
        return false;
    }
    for (int64_t index = 0; index < root_count; ++index) {
        CnpPowerCoefficient value = power_coefficient_at(roots, index);
        if (isnan(value.real) || isnan(value.imag)) {
            *complete = false;
            cnp_free(values, root_bytes);
            cnp_free(conjugates, root_bytes);
            return true;
        }
        values[index] = value;
        conjugates[index] = (CnpPowerCoefficient){
            value.real, -value.imag};
    }
    qsort(
        values, (size_t)root_count,
        sizeof(CnpPowerCoefficient), power_polynomial_root_compare);
    qsort(
        conjugates, (size_t)root_count,
        sizeof(CnpPowerCoefficient), power_polynomial_root_compare);
    for (int64_t index = 0; index < root_count; ++index) {
        if (values[index].real != conjugates[index].real ||
                values[index].imag != conjugates[index].imag) {
            *complete = false;
            break;
        }
    }
    cnp_free(values, root_bytes);
    cnp_free(conjugates, root_bytes);
    return true;
}

static CnpArray *power_polynomial_from_roots(
    const CnpArray *roots,
    bool force_real_result,
    CNP_TYPE forced_result_dtype) {
    const char *function_name = "cnp_poly";
    int64_t root_count;
    size_t coefficient_bytes;
    CNP_TYPE arithmetic_dtype;
    CNP_TYPE result_dtype;
    CnpPowerCoefficient *coefficients;
    CnpArray *result;
    int64_t shape[1];
    if (!power_polynomial_construction_dtype(
            roots->dtype->type_num, &result_dtype)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "root dtype is not numeric");
        return NULL;
    }
    arithmetic_dtype = result_dtype;
    root_count = roots->size;
    if (root_count == 0) {
        result = cnp_scalar_array(1.0, CNP_DOUBLE);
        if (!result) cnp_relabel_error(function_name);
        return result;
    }
    if ((uint64_t)root_count + 1 >
            SIZE_MAX / sizeof(CnpPowerCoefficient)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "coefficient workspace size overflows");
        return NULL;
    }
    coefficient_bytes =
        (size_t)(root_count + 1) * sizeof(CnpPowerCoefficient);
    coefficients = (CnpPowerCoefficient*)cnp_calloc(
        (size_t)(root_count + 1), sizeof(CnpPowerCoefficient));
    if (!coefficients) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate coefficient workspace");
        return NULL;
    }
    coefficients[0].real = 1.0;
    for (int64_t root_index = 0;
         root_index < root_count; ++root_index) {
        CnpPowerCoefficient root =
            power_coefficient_at(roots, root_index);
        for (int64_t coefficient = root_index + 1;
             coefficient > 0; --coefficient) {
            CnpPowerCoefficient product =
                power_polynomial_division_multiply(
                    root, coefficients[coefficient - 1],
                    arithmetic_dtype, false);
            coefficients[coefficient] =
                power_polynomial_division_subtract(
                    coefficients[coefficient], product,
                    arithmetic_dtype, arithmetic_dtype);
        }
    }
    if (cnp_type_is_complex(arithmetic_dtype)) {
        if (force_real_result) {
            result_dtype = forced_result_dtype;
        } else {
            bool conjugate_complete;
            if (!power_polynomial_roots_are_conjugate_complete(
                    roots, &conjugate_complete)) {
                cnp_free(coefficients, coefficient_bytes);
                return NULL;
            }
            if (conjugate_complete) {
                result_dtype = arithmetic_dtype == CNP_CFLOAT
                    ? CNP_FLOAT : CNP_DOUBLE;
            }
        }
    }
    shape[0] = root_count + 1;
    result = cnp_array_new(
        1, shape, result_dtype, CNP_ORDER_C);
    if (result) {
        for (int64_t coefficient = 0;
             coefficient <= root_count; ++coefficient) {
            power_store_arithmetic_coefficient(
                result, coefficient, coefficients[coefficient]);
        }
    }
    cnp_free(coefficients, coefficient_bytes);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_poly(const CnpArray *roots) {
    const char *function_name = "cnp_poly";
    CnpArray *eigenvalues = NULL;
    CnpArray *eigenvectors = NULL;
    CnpArray *result;
    CNP_STATUS status;
    if (!roots) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array is required");
        return NULL;
    }
    if (roots->ndim == 1) {
        return power_polynomial_from_roots(
            roots, false, CNP_DOUBLE);
    }
    if (roots->ndim != 2 || roots->shape[0] == 0 ||
            roots->shape[0] != roots->shape[1]) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input must be 1d or non-empty square 2d array");
        return NULL;
    }
    status = cnp_linalg_eig(
        roots, &eigenvalues, &eigenvectors);
    if (status != CNP_OK) {
        if (eigenvalues) cnp_array_free(eigenvalues);
        if (eigenvectors) cnp_array_free(eigenvectors);
        cnp_relabel_error(function_name);
        return NULL;
    }
    cnp_array_free(eigenvectors);
    if (cnp_type_is_complex(roots->dtype->type_num)) {
        result = power_polynomial_from_roots(
            eigenvalues, false, CNP_DOUBLE);
    } else {
        CNP_TYPE matrix_result_dtype;
        if (!power_polynomial_construction_dtype(
                roots->dtype->type_num, &matrix_result_dtype)) {
            cnp_array_free(eigenvalues);
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "matrix dtype is not numeric");
            return NULL;
        }
        result = power_polynomial_from_roots(
            eigenvalues, true, matrix_result_dtype);
    }
    cnp_array_free(eigenvalues);
    return result;
}

/* =========================================================================
 * polyroots - Find roots of an ascending-power coefficient series
 * numpy.polynomial.polynomial.polyroots(c)
 * ========================================================================= */
static CNP_TYPE power_polyroots_common_dtype(CNP_TYPE source_dtype) {
    if (cnp_type_is_integer(source_dtype)) return CNP_DOUBLE;
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
            return CNP_NOTYPE;
    }
}

static bool power_polyroots_is_zero(CnpPowerCoefficient value) {
    uint64_t real_bits;
    uint64_t imaginary_bits;
    memcpy(&real_bits, &value.real, sizeof(real_bits));
    memcpy(&imaginary_bits, &value.imag, sizeof(imaginary_bits));
    return (real_bits & UINT64_C(0x7fffffffffffffff)) == 0 &&
           (imaginary_bits & UINT64_C(0x7fffffffffffffff)) == 0;
}

static CnpPowerCoefficient power_polyroots_divide(
    CnpPowerCoefficient numerator,
    CnpPowerCoefficient denominator,
    CNP_TYPE dtype) {
    if (dtype == CNP_CFLOAT) {
        float numerator_real = (float)numerator.real;
        float numerator_imag = (float)numerator.imag;
        float denominator_real = (float)denominator.real;
        float denominator_imag = (float)denominator.imag;
        if (fabsf(denominator_real) >= fabsf(denominator_imag)) {
            float ratio = denominator_imag / denominator_real;
            float scale = denominator_real + denominator_imag * ratio;
            return (CnpPowerCoefficient){
                (numerator_real + numerator_imag * ratio) / scale,
                (numerator_imag - numerator_real * ratio) / scale};
        }
        {
            float ratio = denominator_real / denominator_imag;
            float scale = denominator_real * ratio + denominator_imag;
            return (CnpPowerCoefficient){
                (numerator_real * ratio + numerator_imag) / scale,
                (numerator_imag * ratio - numerator_real) / scale};
        }
    }
    if (dtype == CNP_CDOUBLE || dtype == CNP_CLONGDOUBLE) {
        if (fabs(denominator.real) >= fabs(denominator.imag)) {
            double ratio = denominator.imag / denominator.real;
            double scale = denominator.real + denominator.imag * ratio;
            return (CnpPowerCoefficient){
                (numerator.real + numerator.imag * ratio) / scale,
                (numerator.imag - numerator.real * ratio) / scale};
        }
        {
            double ratio = denominator.real / denominator.imag;
            double scale = denominator.real * ratio + denominator.imag;
            return (CnpPowerCoefficient){
                (numerator.real * ratio + numerator.imag) / scale,
                (numerator.imag * ratio - numerator.real) / scale};
        }
    }
    if (dtype == CNP_FLOAT) {
        return (CnpPowerCoefficient){
            (float)numerator.real / (float)denominator.real, 0.0};
    }
    if (dtype == CNP_HALF) {
        float quotient = (float)numerator.real / (float)denominator.real;
        return power_polynomial_division_round(
            (CnpPowerCoefficient){quotient, 0.0}, CNP_HALF);
    }
    return (CnpPowerCoefficient){
        numerator.real / denominator.real, 0.0};
}

static int power_polyroots_compare_component(double left, double right) {
    bool left_nan = isnan(left);
    bool right_nan = isnan(right);
    if (left_nan) return right_nan ? 0 : 1;
    if (right_nan) return -1;
    if (left < right) return -1;
    if (left > right) return 1;
    return 0;
}

static int power_polyroots_float_compare(
    const void *left_pointer, const void *right_pointer) {
    return power_polyroots_compare_component(
        *(const float*)left_pointer, *(const float*)right_pointer);
}

static int power_polyroots_double_compare(
    const void *left_pointer, const void *right_pointer) {
    return power_polyroots_compare_component(
        *(const double*)left_pointer, *(const double*)right_pointer);
}

static int power_polyroots_cfloat_compare(
    const void *left_pointer, const void *right_pointer) {
    const cnp_cfloat *left = (const cnp_cfloat*)left_pointer;
    const cnp_cfloat *right = (const cnp_cfloat*)right_pointer;
    int comparison = power_polyroots_compare_component(
        left->real, right->real);
    if (comparison != 0) return comparison;
    return power_polyroots_compare_component(left->imag, right->imag);
}

static int power_polyroots_cdouble_compare(
    const void *left_pointer, const void *right_pointer) {
    const cnp_cdouble *left = (const cnp_cdouble*)left_pointer;
    const cnp_cdouble *right = (const cnp_cdouble*)right_pointer;
    int comparison = power_polyroots_compare_component(
        left->real, right->real);
    if (comparison != 0) return comparison;
    return power_polyroots_compare_component(left->imag, right->imag);
}

static void power_polyroots_sort(CnpArray *roots) {
    if (!roots || roots->size < 2) return;
    switch (roots->dtype->type_num) {
        case CNP_FLOAT:
            qsort(
                roots->data, (size_t)roots->size, sizeof(float),
                power_polyroots_float_compare);
            break;
        case CNP_DOUBLE:
            qsort(
                roots->data, (size_t)roots->size, sizeof(double),
                power_polyroots_double_compare);
            break;
        case CNP_CFLOAT:
            qsort(
                roots->data, (size_t)roots->size, sizeof(cnp_cfloat),
                power_polyroots_cfloat_compare);
            break;
        case CNP_CDOUBLE:
            qsort(
                roots->data, (size_t)roots->size, sizeof(cnp_cdouble),
                power_polyroots_cdouble_compare);
            break;
        default:
            break;
    }
}

static CnpPowerCoefficient power_polyroots_workspace_value_at(
    const CnpArray *array, int64_t index) {
    switch (array->dtype->type_num) {
        case CNP_FLOAT:
            return (CnpPowerCoefficient){
                ((const float*)array->data)[index], 0.0};
        case CNP_DOUBLE:
            return (CnpPowerCoefficient){
                ((const double*)array->data)[index], 0.0};
        case CNP_CFLOAT:
            return (CnpPowerCoefficient){
                ((const cnp_cfloat*)array->data)[index].real,
                ((const cnp_cfloat*)array->data)[index].imag};
        case CNP_CDOUBLE:
            return (CnpPowerCoefficient){
                ((const cnp_cdouble*)array->data)[index].real,
                ((const cnp_cdouble*)array->data)[index].imag};
        default:
            return (CnpPowerCoefficient){0.0, 0.0};
    }
}

static CNP_STATUS power_polyroots_balance_companion(
    CnpArray *companion, int64_t degree,
    const char *function_name) {
    bool changed;
    for (int64_t index = 0; index < companion->size; ++index) {
        CnpPowerCoefficient value =
            power_polyroots_workspace_value_at(companion, index);
        if (!isfinite(value.real) || !isfinite(value.imag)) {
            cnp_set_error(
                CNP_ERR_CONVERGENCE, function_name,
                "companion matrix contains NaN or infinity");
            return CNP_ERR_CONVERGENCE;
        }
    }
    do {
        changed = false;
        for (int64_t index = 0; index < degree; ++index) {
            double row_norm = 0.0;
            double column_norm = 0.0;
            for (int64_t other = 0; other < degree; ++other) {
                if (other == index) continue;
                CnpPowerCoefficient row_value =
                    power_polyroots_workspace_value_at(
                        companion, index * degree + other);
                CnpPowerCoefficient column_value =
                    power_polyroots_workspace_value_at(
                        companion, other * degree + index);
                row_norm += hypot(row_value.real, row_value.imag);
                column_norm += hypot(
                    column_value.real, column_value.imag);
            }
            if (!isfinite(row_norm) || !isfinite(column_norm)) {
                cnp_set_error(
                    CNP_ERR_CONVERGENCE, function_name,
                    "companion matrix contains NaN or infinity");
                return CNP_ERR_CONVERGENCE;
            }
            if (row_norm == 0.0 || column_norm == 0.0) continue;
            double factor = 1.0;
            double scaled_column = column_norm;
            double lower = row_norm / 2.0;
            while (scaled_column < lower) {
                factor *= 2.0;
                scaled_column *= 4.0;
            }
            double upper = row_norm * 2.0;
            while (scaled_column >= upper) {
                factor *= 0.5;
                scaled_column *= 0.25;
            }
            if ((scaled_column + row_norm) / factor >=
                    0.95 * (column_norm + row_norm)) {
                continue;
            }
            changed = true;
            for (int64_t other = 0; other < degree; ++other) {
                CnpPowerCoefficient row_value =
                    power_polyroots_workspace_value_at(
                        companion, index * degree + other);
                row_value.real /= factor;
                row_value.imag /= factor;
                power_store_arithmetic_coefficient(
                    companion, index * degree + other, row_value);

                CnpPowerCoefficient column_value =
                    power_polyroots_workspace_value_at(
                        companion, other * degree + index);
                column_value.real *= factor;
                column_value.imag *= factor;
                power_store_arithmetic_coefficient(
                    companion, other * degree + index, column_value);
            }
        }
    } while (changed);
    return CNP_OK;
}

static CnpPowerCoefficient power_polyroots_complex_multiply(
    CnpPowerCoefficient left, CnpPowerCoefficient right) {
    return (CnpPowerCoefficient){
        left.real * right.real - left.imag * right.imag,
        left.real * right.imag + left.imag * right.real};
}

static void power_polyroots_value_and_derivative(
    const CnpArray *coefficients,
    int64_t leading_index,
    int64_t coefficient_step,
    int64_t degree,
    CNP_TYPE coefficient_dtype,
    CnpPowerCoefficient point,
    CnpPowerCoefficient *value,
    CnpPowerCoefficient *derivative) {
    CnpPowerCoefficient current_value = power_polynomial_division_round(
        power_coefficient_at(coefficients, leading_index),
        coefficient_dtype);
    CnpPowerCoefficient current_derivative = {0.0, 0.0};
    for (int64_t index = 1; index <= degree; ++index) {
        current_derivative = power_polyroots_complex_multiply(
            current_derivative, point);
        current_derivative.real += current_value.real;
        current_derivative.imag += current_value.imag;
        current_value = power_polyroots_complex_multiply(
            current_value, point);
        CnpPowerCoefficient coefficient =
            power_polynomial_division_round(
                power_coefficient_at(
                    coefficients,
                    leading_index + index * coefficient_step),
                coefficient_dtype);
        current_value.real += coefficient.real;
        current_value.imag += coefficient.imag;
    }
    *value = current_value;
    *derivative = current_derivative;
}

static bool power_polyroots_refines_to_real(
    const CnpArray *coefficients,
    int64_t leading_index,
    int64_t coefficient_step,
    int64_t degree,
    CNP_TYPE coefficient_dtype,
    CnpPowerCoefficient root) {
    double initial_imaginary = fabs(root.imag);
    if (initial_imaginary == 0.0) return true;
    CnpPowerCoefficient refined = root;
    for (int iteration = 0; iteration < 2; ++iteration) {
        CnpPowerCoefficient value;
        CnpPowerCoefficient derivative;
        power_polyroots_value_and_derivative(
            coefficients, leading_index, coefficient_step,
            degree, coefficient_dtype,
            refined, &value, &derivative);
        if (!isfinite(value.real) || !isfinite(value.imag) ||
                !isfinite(derivative.real) ||
                !isfinite(derivative.imag) ||
                (derivative.real == 0.0 && derivative.imag == 0.0)) {
            return false;
        }
        CnpPowerCoefficient correction = power_polyroots_divide(
            value, derivative, CNP_CDOUBLE);
        if (!isfinite(correction.real) || !isfinite(correction.imag)) {
            return false;
        }
        refined.real -= correction.real;
        refined.imag -= correction.imag;
    }
    return fabs(refined.imag) <= initial_imaginary * 1.0e-3;
}

static CnpArray *power_polyroots_project_real_spectrum(
    const CnpArray *coefficients,
    int64_t leading_index,
    int64_t coefficient_step,
    int64_t degree,
    CNP_TYPE coefficient_dtype,
    CnpArray *roots,
    const char *function_name) {
    CNP_TYPE expected_complex_dtype;
    if (coefficient_dtype != CNP_FLOAT &&
            coefficient_dtype != CNP_DOUBLE) {
        return roots;
    }
    expected_complex_dtype = coefficient_dtype == CNP_FLOAT
        ? CNP_CFLOAT : CNP_CDOUBLE;
    if (roots->dtype->type_num != expected_complex_dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "complex eig workspace returned an unexpected dtype");
        cnp_array_free(roots);
        return NULL;
    }
    for (int64_t index = 0; index < roots->size; ++index) {
        if (!power_polyroots_refines_to_real(
                coefficients, leading_index, coefficient_step,
                degree, coefficient_dtype,
                power_coefficient_at(roots, index))) {
            return roots;
        }
    }
    int64_t shape[1] = {roots->size};
    CnpArray *real_roots = cnp_array_new(
        1, shape, coefficient_dtype, CNP_ORDER_C);
    if (!real_roots) {
        cnp_array_free(roots);
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int64_t index = 0; index < roots->size; ++index) {
        CnpPowerCoefficient value = power_coefficient_at(roots, index);
        power_store_arithmetic_coefficient(
            real_roots, index,
            (CnpPowerCoefficient){value.real, 0.0});
    }
    cnp_array_free(roots);
    return real_roots;
}

static CnpArray *power_polyroots_finalize_eigenvalues(CnpArray *roots) {
    if (!roots) return NULL;
    switch (roots->dtype->type_num) {
        case CNP_FLOAT:
            for (int64_t index = 0; index < roots->size; ++index) {
                if (((float*)roots->data)[index] == 0.0f) {
                    ((float*)roots->data)[index] = 0.0f;
                }
            }
            return roots;
        case CNP_DOUBLE: {
            int64_t carrier_shape[1];
            int64_t result_shape[1] = {roots->size};
            int64_t result_strides[1] = {2 * (int64_t)sizeof(double)};
            CnpArray *carrier;
            CnpArray *view;
            if (roots->size > INT64_MAX / 2) {
                cnp_set_error(
                    CNP_ERR_MEMORY, "cnp_polyroots",
                    "real eigenvalue storage size overflows");
                cnp_array_free(roots);
                return NULL;
            }
            carrier_shape[0] = 2 * roots->size;
            carrier = cnp_array_new(
                1, carrier_shape, CNP_DOUBLE, CNP_ORDER_C);
            if (!carrier) {
                cnp_array_free(roots);
                cnp_relabel_error("cnp_polyroots");
                return NULL;
            }
            for (int64_t index = 0; index < carrier->size; ++index) {
                ((double*)carrier->data)[index] = 0.0;
            }
            for (int64_t index = 0; index < roots->size; ++index) {
                double value = ((double*)roots->data)[index];
                ((double*)carrier->data)[2 * index] =
                    value == 0.0 ? 0.0 : value;
            }
            view = cnp_array_view_from_metadata(
                carrier, 1, result_shape, result_strides, 0, 0);
            cnp_array_decref(carrier);
            cnp_array_free(roots);
            if (!view) cnp_relabel_error("cnp_polyroots");
            return view;
        }
        case CNP_CFLOAT:
            for (int64_t index = 0; index < roots->size; ++index) {
                cnp_cfloat *value = &((cnp_cfloat*)roots->data)[index];
                if (value->real == 0.0f) value->real = 0.0f;
                if (value->imag == 0.0f) value->imag = 0.0f;
            }
            return roots;
        case CNP_CDOUBLE:
            for (int64_t index = 0; index < roots->size; ++index) {
                cnp_cdouble *value = &((cnp_cdouble*)roots->data)[index];
                if (value->real == 0.0) value->real = 0.0;
                if (value->imag == 0.0) value->imag = 0.0;
            }
            return roots;
        default:
            return roots;
    }
}

CNP_API CnpArray* CNP_CALL cnp_polyroots(const CnpArray *p) {
    const char *function_name = "cnp_polyroots";
    CNP_TYPE result_dtype;
    int64_t coefficient_count;
    int64_t degree;
    int64_t result_shape[1];
    CnpArray *companion = NULL;
    CnpArray *eigenvalues = NULL;
    CNP_STATUS status;
    if (!p) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "coefficient array is required");
        return NULL;
    }
    if (p->size == 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "Coefficient array is empty");
        return NULL;
    }
    if (p->ndim != 0 && p->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "Coefficient array is not 1-d");
        return NULL;
    }
    result_dtype = power_polyroots_common_dtype(p->dtype->type_num);
    if (result_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "Coefficient arrays have no common type");
        return NULL;
    }

    coefficient_count = p->size;
    while (coefficient_count > 1 && power_polyroots_is_zero(
            power_coefficient_at(p, coefficient_count - 1))) {
        coefficient_count--;
    }
    degree = coefficient_count - 1;
    result_shape[0] = degree;
    if (degree == 0) {
        CnpArray *result = cnp_array_new(
            1, result_shape, result_dtype, CNP_ORDER_C);
        if (!result) cnp_relabel_error(function_name);
        return result;
    }
    if (degree == 1) {
        CnpArray *result = cnp_array_new(
            1, result_shape, result_dtype, CNP_ORDER_C);
        if (!result) {
            cnp_relabel_error(function_name);
            return NULL;
        }
        CnpPowerCoefficient numerator = power_polynomial_division_round(
            power_coefficient_at(p, 0), result_dtype);
        CnpPowerCoefficient denominator = power_polynomial_division_round(
            power_coefficient_at(p, 1), result_dtype);
        numerator.real = -numerator.real;
        numerator.imag = -numerator.imag;
        power_store_arithmetic_coefficient(
            result, 0,
            power_polyroots_divide(numerator, denominator, result_dtype));
        return result;
    }
    if (result_dtype == CNP_HALF ||
            result_dtype == CNP_LONGDOUBLE ||
            result_dtype == CNP_CLONGDOUBLE) {
        const char *dtype_name = result_dtype == CNP_HALF
            ? "float16"
            : (result_dtype == CNP_LONGDOUBLE
                ? "longdouble" : "clongdouble");
        char message[128];
        snprintf(
            message, sizeof(message),
            "array type %s is unsupported in linalg", dtype_name);
        cnp_set_error(CNP_ERR_TYPE, function_name, message);
        return NULL;
    }

    int64_t companion_shape[2] = {degree, degree};
    companion = cnp_array_new(
        2, companion_shape, result_dtype, CNP_ORDER_C);
    if (!companion) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int64_t index = 0; index < companion->size; ++index) {
        power_store_arithmetic_coefficient(
            companion, index, (CnpPowerCoefficient){0.0, 0.0});
    }
    CnpPowerCoefficient leading = power_polynomial_division_round(
        power_coefficient_at(p, degree), result_dtype);
    for (int64_t row = 0; row < degree; ++row) {
        CnpPowerCoefficient numerator = power_polynomial_division_round(
            power_coefficient_at(p, degree - 1 - row), result_dtype);
        numerator.real = -numerator.real;
        numerator.imag = -numerator.imag;
        power_store_arithmetic_coefficient(
            companion, row * degree,
            power_polyroots_divide(numerator, leading, result_dtype));
        if (row + 1 < degree) {
            power_store_arithmetic_coefficient(
                companion, row * degree + row + 1,
                (CnpPowerCoefficient){1.0, 0.0});
        }
    }
    status = power_polyroots_balance_companion(
        companion, degree, function_name);
    if (status != CNP_OK) {
        cnp_array_free(companion);
        return NULL;
    }

    status = cnp_linalg_eigvals_force_complex(
        companion, &eigenvalues);
    cnp_array_free(companion);
    if (status != CNP_OK) {
        if (eigenvalues) cnp_array_free(eigenvalues);
        cnp_relabel_error(function_name);
        return NULL;
    }
    eigenvalues = power_polyroots_project_real_spectrum(
        p, degree, -1, degree, result_dtype,
        eigenvalues, function_name);
    if (!eigenvalues) return NULL;
    power_polyroots_sort(eigenvalues);
    return power_polyroots_finalize_eigenvalues(eigenvalues);
}

/* =========================================================================
 * roots - Find roots of a descending-power coefficient series
 * numpy.roots(p)
 * ========================================================================= */
static CNP_TYPE legacy_power_roots_working_dtype(CNP_TYPE source_dtype) {
    if (source_dtype == CNP_BOOL || cnp_type_is_integer(source_dtype)) {
        return CNP_DOUBLE;
    }
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
            return CNP_NOTYPE;
    }
}

static CnpArray *legacy_power_roots_float64_zeros(
    int64_t count, const char *function_name) {
    int64_t shape[1] = {count};
    CnpArray *result = cnp_array_new(
        1, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int64_t index = 0; index < count; ++index) {
        ((double*)result->data)[index] = 0.0;
    }
    return result;
}

static CnpArray *legacy_power_roots_append_zeros(
    CnpArray *roots, int64_t trailing_zeros,
    const char *function_name) {
    int64_t total;
    int64_t shape[1];
    CnpArray *result;
    if (roots->size > INT64_MAX - trailing_zeros) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "root result size overflows");
        cnp_array_free(roots);
        return NULL;
    }
    total = roots->size + trailing_zeros;
    shape[0] = total;
    result = cnp_array_new(
        1, shape, roots->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_array_free(roots);
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int64_t index = 0; index < roots->size; ++index) {
        power_store_arithmetic_coefficient(
            result, index, power_coefficient_at(roots, index));
    }
    for (int64_t index = roots->size; index < total; ++index) {
        power_store_arithmetic_coefficient(
            result, index, (CnpPowerCoefficient){0.0, 0.0});
    }
    cnp_array_free(roots);
    return result;
}

static bool legacy_power_roots_infinite_leading_result(
    const CnpArray *p,
    int64_t first_nonzero,
    int64_t last_nonzero,
    int64_t trailing_zeros,
    CNP_TYPE working_dtype,
    CnpPowerCoefficient leading,
    const char *function_name,
    CnpArray **result) {
    int64_t degree = last_nonzero - first_nonzero;
    int64_t shape[1] = {degree};
    CnpArray *roots;
    CnpPowerCoefficient numerator;
    CnpPowerCoefficient signed_zero;

    *result = NULL;
    for (int64_t index = first_nonzero + 1;
         index <= last_nonzero; ++index) {
        CnpPowerCoefficient coefficient = power_coefficient_at(p, index);
        if (!isfinite(coefficient.real) || !isfinite(coefficient.imag)) {
            return false;
        }
    }
    roots = cnp_array_new(
        1, shape, working_dtype, CNP_ORDER_C);
    if (!roots) {
        cnp_relabel_error(function_name);
        return true;
    }
    for (int64_t index = 0; index < degree; ++index) {
        power_store_arithmetic_coefficient(
            roots, index, (CnpPowerCoefficient){0.0, 0.0});
    }
    numerator = power_polynomial_division_round(
        power_coefficient_at(p, first_nonzero + 1),
        working_dtype);
    numerator.real = -numerator.real;
    numerator.imag = -numerator.imag;
    signed_zero = power_polyroots_divide(
        numerator, leading, working_dtype);
    power_store_arithmetic_coefficient(
        roots, degree - 1, signed_zero);
    *result = legacy_power_roots_append_zeros(
        roots, trailing_zeros, function_name);
    return true;
}

static void legacy_power_roots_order_real_conjugate_pairs(
    CnpArray *roots, CNP_TYPE coefficient_dtype) {
    if (coefficient_dtype != CNP_FLOAT &&
            coefficient_dtype != CNP_DOUBLE) {
        return;
    }
    if (roots->dtype->type_num == CNP_CFLOAT) {
        cnp_cfloat *values = (cnp_cfloat*)roots->data;
        for (int64_t index = 0; index + 1 < roots->size; ++index) {
            if (values[index].imag < 0.0f &&
                    values[index + 1].imag > 0.0f) {
                cnp_cfloat temporary = values[index];
                values[index] = values[index + 1];
                values[index + 1] = temporary;
                index++;
            }
        }
    } else if (roots->dtype->type_num == CNP_CDOUBLE) {
        cnp_cdouble *values = (cnp_cdouble*)roots->data;
        for (int64_t index = 0; index + 1 < roots->size; ++index) {
            if (values[index].imag < 0.0 &&
                    values[index + 1].imag > 0.0) {
                cnp_cdouble temporary = values[index];
                values[index] = values[index + 1];
                values[index + 1] = temporary;
                index++;
            }
        }
    }
}

CNP_API CnpArray* CNP_CALL cnp_roots(const CnpArray *p) {
    const char *function_name = "cnp_roots";
    CNP_TYPE working_dtype;
    int64_t first_nonzero;
    int64_t last_nonzero;
    int64_t trailing_zeros;
    int64_t degree;
    int64_t companion_shape[2];
    CnpArray *companion;
    CnpArray *eigenvalues = NULL;
    CnpPowerCoefficient leading;
    CNP_STATUS status;

    if (!p) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "coefficient array is required");
        return NULL;
    }
    if (p->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "Input must be a rank-1 array");
        return NULL;
    }
    if (p->size == 0) {
        return legacy_power_roots_float64_zeros(0, function_name);
    }
    working_dtype = legacy_power_roots_working_dtype(
        p->dtype->type_num);
    if (working_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "coefficient array must have a numeric dtype");
        return NULL;
    }

    first_nonzero = 0;
    while (first_nonzero < p->size && power_polyroots_is_zero(
            power_coefficient_at(p, first_nonzero))) {
        first_nonzero++;
    }
    if (first_nonzero == p->size) {
        return legacy_power_roots_float64_zeros(0, function_name);
    }
    last_nonzero = p->size - 1;
    while (last_nonzero > first_nonzero && power_polyroots_is_zero(
            power_coefficient_at(p, last_nonzero))) {
        last_nonzero--;
    }
    trailing_zeros = p->size - last_nonzero - 1;
    degree = last_nonzero - first_nonzero;
    if (degree == 0) {
        return legacy_power_roots_float64_zeros(
            trailing_zeros, function_name);
    }
    if (working_dtype == CNP_HALF ||
            working_dtype == CNP_LONGDOUBLE ||
            working_dtype == CNP_CLONGDOUBLE) {
        const char *dtype_name = working_dtype == CNP_HALF
            ? "float16"
            : (working_dtype == CNP_LONGDOUBLE
                ? "longdouble" : "clongdouble");
        char message[128];
        snprintf(
            message, sizeof(message),
            "array type %s is unsupported in linalg", dtype_name);
        cnp_set_error(CNP_ERR_TYPE, function_name, message);
        return NULL;
    }

    leading = power_polynomial_division_round(
        power_coefficient_at(p, first_nonzero), working_dtype);
    if ((working_dtype == CNP_FLOAT || working_dtype == CNP_DOUBLE) &&
            isinf(leading.real)) {
        CnpArray *infinite_result;
        if (legacy_power_roots_infinite_leading_result(
                p, first_nonzero, last_nonzero, trailing_zeros,
                working_dtype, leading, function_name,
                &infinite_result)) {
            return infinite_result;
        }
    }

    companion_shape[0] = degree;
    companion_shape[1] = degree;
    companion = cnp_array_new(
        2, companion_shape, working_dtype, CNP_ORDER_C);
    if (!companion) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int64_t index = 0; index < companion->size; ++index) {
        power_store_arithmetic_coefficient(
            companion, index, (CnpPowerCoefficient){0.0, 0.0});
    }
    for (int64_t row = 1; row < degree; ++row) {
        power_store_arithmetic_coefficient(
            companion, row * degree + row - 1,
            (CnpPowerCoefficient){1.0, 0.0});
    }
    for (int64_t column = 0; column < degree; ++column) {
        CnpPowerCoefficient numerator = power_polynomial_division_round(
            power_coefficient_at(p, first_nonzero + column + 1),
            working_dtype);
        numerator.real = -numerator.real;
        numerator.imag = -numerator.imag;
        power_store_arithmetic_coefficient(
            companion, column,
            power_polyroots_divide(
                numerator, leading, working_dtype));
    }
    status = power_polyroots_balance_companion(
        companion, degree, function_name);
    if (status != CNP_OK) {
        cnp_array_free(companion);
        return NULL;
    }

    status = cnp_linalg_eigvals_force_complex(
        companion, &eigenvalues);
    cnp_array_free(companion);
    if (status != CNP_OK) {
        if (eigenvalues) cnp_array_free(eigenvalues);
        cnp_relabel_error(function_name);
        return NULL;
    }
    eigenvalues = power_polyroots_project_real_spectrum(
        p, first_nonzero, 1, degree, working_dtype,
        eigenvalues, function_name);
    if (!eigenvalues) return NULL;
    legacy_power_roots_order_real_conjugate_pairs(
        eigenvalues, working_dtype);
    return legacy_power_roots_append_zeros(
        eigenvalues, trailing_zeros, function_name);
}

/* =========================================================================
 * vander - Generate Vandermonde matrix
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_vander(const CnpArray *x, int64_t n, bool increasing) {
    const char *function_name = "cnp_vander";
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            x, function_name, &ignored_nbytes)) return NULL;
    if (x->ndim != 1 || !x->shape || !x->strides ||
            (x->size > 0 && !x->data)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "x must be a valid one-dimensional array");
        return NULL;
    }
    if (!power_calculus_dtype_supported(x->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "x must have a represented numeric dtype");
        return NULL;
    }
    if (n < 0) n = x->size;

    int64_t m = x->size;
    int64_t shape[2] = {m, n};
    CNP_TYPE result_type = cnp_promote_types_public(
        x->dtype->type_num, CNP_INT);
    if (result_type == CNP_NOTYPE) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpArray *result = cnp_array_new(
        2, shape, result_type, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int itemsize = result->dtype->elsize;
    for (int64_t row = 0; row < m; ++row) {
        const void *source = (const char*)x->data +
            x->offset + row * x->strides[0];
        cnp_clongdouble base = {0};
        CNP_STATUS status = cnp_cast_scalar_value(
            source, x->dtype->type_num,
            &base, CNP_CLONGDOUBLE, function_name);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            return NULL;
        }

        cnp_clongdouble value = {1.0L, 0.0L};
        for (int64_t power = 0; power < n; ++power) {
            int64_t column = increasing ? power : n - 1 - power;
            void *destination = (char*)result->data +
                (row * n + column) * itemsize;
            status = cnp_cast_scalar_value(
                &value, CNP_CLONGDOUBLE,
                destination, result_type, function_name);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                return NULL;
            }
            cnp_clongdouble next = {
                value.real * base.real - value.imag * base.imag,
                value.real * base.imag + value.imag * base.real,
            };
            value = next;
        }
    }
    return result;
}
