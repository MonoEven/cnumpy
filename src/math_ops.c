/**
 * cnumpy mathematical operations - arithmetic, trig, exp/log, rounding
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <complex.h>

#ifdef _MSC_VER
typedef char cnp_msvc_cfloat_layout_must_match[
    sizeof(cnp_cfloat) == sizeof(_Fcomplex) ? 1 : -1];
typedef char cnp_msvc_cdouble_layout_must_match[
    sizeof(cnp_cdouble) == sizeof(_Dcomplex) ? 1 : -1];
typedef char cnp_msvc_clongdouble_layout_must_match[
    sizeof(cnp_clongdouble) == sizeof(_Lcomplex) ? 1 : -1];
#endif

/* =========================================================================
 * Helper: determine output type for binary arithmetic
 * ========================================================================= */
static CNP_TYPE arith_result_type(const CnpArray *a, const CnpArray *b) {
    CNP_TYPE ta = a->dtype->type_num;
    CNP_TYPE tb = b->dtype->type_num;
    /* If either is float/complex, promote */
    if (cnp_type_is_float(ta) || cnp_type_is_float(tb) ||
        cnp_type_is_complex(ta) || cnp_type_is_complex(tb)) {
        return cnp_promote_type(ta, tb);
    }
    /* Both integers: default to float64 for division, else promote */
    return cnp_promote_type(ta, tb);
}

static CNP_TYPE unary_result_type(const CnpArray *a) {
    CNP_TYPE t = a->dtype->type_num;
    if (cnp_type_is_float(t) || cnp_type_is_complex(t)) return t;
    return CNP_DOUBLE; /* Integer inputs produce float64 for most unary ops */
}

static bool same_shape(const CnpArray *a, const CnpArray *b) {
    if (a->ndim != b->ndim || a->size != b->size) return false;
    for (int axis = 0; axis < a->ndim; ++axis) {
        if (a->shape[axis] != b->shape[axis]) return false;
    }
    return true;
}

/* =========================================================================
 * Arithmetic operations
 * ========================================================================= */
typedef enum {
    CNP_ARITHMETIC_ADD,
    CNP_ARITHMETIC_SUBTRACT,
    CNP_ARITHMETIC_MULTIPLY,
    CNP_ARITHMETIC_DIVIDE,
    CNP_ARITHMETIC_FLOOR_DIVIDE,
    CNP_ARITHMETIC_REMAINDER,
    CNP_ARITHMETIC_FMOD
} CnpArithmeticOperation;

typedef enum {
    CNP_ARITHMETIC_SIGNED,
    CNP_ARITHMETIC_UNSIGNED,
    CNP_ARITHMETIC_FLOATING,
    CNP_ARITHMETIC_COMPLEX
} CnpArithmeticValueKind;

typedef struct {
    CnpArithmeticValueKind kind;
    int64_t signed_value;
    uint64_t unsigned_value;
    long double real;
    long double imaginary;
} CnpArithmeticValue;

typedef union {
    uint64_t integer;
    long double floating;
    cnp_clongdouble complex_value;
} CnpArithmeticScalarStorage;

static CNP_STATUS arithmetic_read_value(
    const void *source, CNP_TYPE dtype, CnpArithmeticValue *value,
    const char *function_name) {
    value->signed_value = 0;
    value->unsigned_value = 0;
    value->real = 0.0L;
    value->imaginary = 0.0L;
    switch (dtype) {
        case CNP_BOOL: {
            uint8_t source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_SIGNED;
            value->signed_value = source_value != 0;
            return CNP_OK;
        }
        case CNP_BYTE: {
            int8_t source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_SIGNED;
            value->signed_value = source_value;
            return CNP_OK;
        }
        case CNP_SHORT: {
            int16_t source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_SIGNED;
            value->signed_value = source_value;
            return CNP_OK;
        }
        case CNP_INT: {
            int32_t source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_SIGNED;
            value->signed_value = source_value;
            return CNP_OK;
        }
        case CNP_LONG:
        case CNP_LONGLONG: {
            int64_t source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_SIGNED;
            value->signed_value = source_value;
            return CNP_OK;
        }
        case CNP_UBYTE: {
            uint8_t source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_UNSIGNED;
            value->unsigned_value = source_value;
            return CNP_OK;
        }
        case CNP_USHORT: {
            uint16_t source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_UNSIGNED;
            value->unsigned_value = source_value;
            return CNP_OK;
        }
        case CNP_UINT: {
            uint32_t source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_UNSIGNED;
            value->unsigned_value = source_value;
            return CNP_OK;
        }
        case CNP_ULONG:
        case CNP_ULONGLONG: {
            uint64_t source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_UNSIGNED;
            value->unsigned_value = source_value;
            return CNP_OK;
        }
        case CNP_HALF: {
            uint16_t source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_FLOATING;
            value->real = (long double)cnp_half_to_float(source_value);
            return CNP_OK;
        }
        case CNP_FLOAT: {
            float source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_FLOATING;
            value->real = (long double)source_value;
            return CNP_OK;
        }
        case CNP_DOUBLE: {
            double source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_FLOATING;
            value->real = (long double)source_value;
            return CNP_OK;
        }
        case CNP_LONGDOUBLE: {
            long double source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_FLOATING;
            value->real = source_value;
            return CNP_OK;
        }
        case CNP_CFLOAT: {
            cnp_cfloat source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_COMPLEX;
            value->real = (long double)source_value.real;
            value->imaginary = (long double)source_value.imag;
            return CNP_OK;
        }
        case CNP_CDOUBLE: {
            cnp_cdouble source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_COMPLEX;
            value->real = (long double)source_value.real;
            value->imaginary = (long double)source_value.imag;
            return CNP_OK;
        }
        case CNP_CLONGDOUBLE: {
            cnp_clongdouble source_value;
            memcpy(&source_value, source, sizeof(source_value));
            value->kind = CNP_ARITHMETIC_COMPLEX;
            value->real = source_value.real;
            value->imaginary = source_value.imag;
            return CNP_OK;
        }
        default:
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "dtype %d is not supported for arithmetic", (int)dtype);
            return CNP_ERR_TYPE;
    }
}

static uint64_t arithmetic_integer_bits(const CnpArithmeticValue *value) {
    return value->kind == CNP_ARITHMETIC_SIGNED
        ? (uint64_t)value->signed_value
        : value->unsigned_value;
}

static float arithmetic_value_float(const CnpArithmeticValue *value) {
    if (value->kind == CNP_ARITHMETIC_SIGNED)
        return (float)value->signed_value;
    if (value->kind == CNP_ARITHMETIC_UNSIGNED)
        return (float)value->unsigned_value;
    return (float)value->real;
}

static double arithmetic_value_double(const CnpArithmeticValue *value) {
    if (value->kind == CNP_ARITHMETIC_SIGNED)
        return (double)value->signed_value;
    if (value->kind == CNP_ARITHMETIC_UNSIGNED)
        return (double)value->unsigned_value;
    return (double)value->real;
}

static long double arithmetic_value_longdouble(
    const CnpArithmeticValue *value) {
    if (value->kind == CNP_ARITHMETIC_SIGNED)
        return (long double)value->signed_value;
    if (value->kind == CNP_ARITHMETIC_UNSIGNED)
        return (long double)value->unsigned_value;
    return value->real;
}

static CNP_STATUS arithmetic_unsigned_result(
    CnpArithmeticOperation operation,
    uint64_t left,
    uint64_t right,
    uint64_t *result,
    const char *function_name) {
    switch (operation) {
        case CNP_ARITHMETIC_ADD:
            *result = left + right;
            return CNP_OK;
        case CNP_ARITHMETIC_SUBTRACT:
            *result = left - right;
            return CNP_OK;
        case CNP_ARITHMETIC_MULTIPLY:
            *result = left * right;
            return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_GENERIC, function_name,
        "invalid internal arithmetic operation %d", (int)operation);
    return CNP_ERR_GENERIC;
}

static int64_t arithmetic_signed_from_bits(uint64_t bits) {
    int64_t result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint64_t arithmetic_signed_floor_divide_bits(
    int64_t left, int64_t right) {
    if (right == 0) return 0;
    if (left == INT64_MIN && right == -1)
        return (uint64_t)INT64_MIN;
    int64_t quotient = left / right;
    int64_t remainder = left % right;
    if (remainder != 0 && ((left < 0) != (right < 0))) --quotient;
    return (uint64_t)quotient;
}

static uint64_t arithmetic_signed_remainder_bits(
    int64_t left, int64_t right) {
    if (right == 0 || (left == INT64_MIN && right == -1)) return 0;
    int64_t remainder = left % right;
    if (remainder != 0 && ((left < 0) != (right < 0)))
        remainder += right;
    return (uint64_t)remainder;
}

static uint64_t arithmetic_signed_fmod_bits(
    int64_t left, int64_t right) {
    if (right == 0 || (left == INT64_MIN && right == -1)) return 0;
    return (uint64_t)(left % right);
}

static CNP_STATUS arithmetic_store_integer_bits(
    void *destination,
    CNP_TYPE result_dtype,
    uint64_t result_bits,
    const char *function_name) {
    if (result_dtype == CNP_BYTE || result_dtype == CNP_UBYTE) {
        uint8_t result = (uint8_t)result_bits;
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_SHORT || result_dtype == CNP_USHORT) {
        uint16_t result = (uint16_t)result_bits;
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_INT || result_dtype == CNP_UINT) {
        uint32_t result = (uint32_t)result_bits;
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONG || result_dtype == CNP_ULONG ||
            result_dtype == CNP_LONGLONG ||
            result_dtype == CNP_ULONGLONG) {
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "result dtype %d is not an integer dtype", (int)result_dtype);
    return CNP_ERR_TYPE;
}

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static float arithmetic_divmod_float(
    float left, float right, float *modulus_result) {
    float modulus = fmodf(left, right);
    if (right == 0.0f) {
        *modulus_result = modulus;
        return left / right;
    }
    volatile float quotient = (left - modulus) / right;
    if (modulus != 0.0f) {
        if (isless(right, 0.0f) != isless(modulus, 0.0f)) {
            modulus += right;
            quotient -= 1.0f;
        }
    } else {
        modulus = copysignf(0.0f, right);
    }
    volatile float floored = floorf(quotient);
    if (isgreater(quotient - floored, 0.5f)) floored += 1.0f;
    if (floored == 0.0f)
        floored = copysignf(0.0f, left / right);
    *modulus_result = modulus;
    return floored;
}

static double arithmetic_divmod_double(
    double left, double right, double *modulus_result) {
    double modulus = fmod(left, right);
    if (right == 0.0) {
        *modulus_result = modulus;
        return left / right;
    }
    volatile double quotient = (left - modulus) / right;
    if (modulus != 0.0) {
        if (isless(right, 0.0) != isless(modulus, 0.0)) {
            modulus += right;
            quotient -= 1.0;
        }
    } else {
        modulus = copysign(0.0, right);
    }
    volatile double floored = floor(quotient);
    if (isgreater(quotient - floored, 0.5)) floored += 1.0;
    if (floored == 0.0)
        floored = copysign(0.0, left / right);
    *modulus_result = modulus;
    return floored;
}

static long double arithmetic_divmod_longdouble(
    long double left, long double right, long double *modulus_result) {
    long double modulus = fmodl(left, right);
    if (right == 0.0L) {
        *modulus_result = modulus;
        return left / right;
    }
    volatile long double quotient = (left - modulus) / right;
    if (modulus != 0.0L) {
        if (isless(right, 0.0L) != isless(modulus, 0.0L)) {
            modulus += right;
            quotient -= 1.0L;
        }
    } else {
        modulus = copysignl(0.0L, right);
    }
    volatile long double floored = floorl(quotient);
    if (isgreater(quotient - floored, 0.5L)) floored += 1.0L;
    if (floored == 0.0L)
        floored = copysignl(0.0L, left / right);
    *modulus_result = modulus;
    return floored;
}

static cnp_cfloat arithmetic_divide_cfloat(
    float left_real,
    float left_imaginary,
    float right_real,
    float right_imaginary) {
    float absolute_real = fabsf(right_real);
    float absolute_imaginary = fabsf(right_imaginary);
    cnp_cfloat result;
    if (absolute_real >= absolute_imaginary) {
        if (absolute_real == 0.0f && absolute_imaginary == 0.0f) {
            result.real = left_real / absolute_real;
            result.imag = left_imaginary / absolute_imaginary;
            return result;
        }
        volatile float ratio = right_imaginary / right_real;
        volatile float denominator =
            right_real + right_imaginary * ratio;
        volatile float scale = 1.0f / denominator;
        volatile float real_numerator =
            left_real + left_imaginary * ratio;
        volatile float imaginary_numerator =
            left_imaginary - left_real * ratio;
        result.real = real_numerator * scale;
        result.imag = imaginary_numerator * scale;
        return result;
    }

    volatile float ratio = right_real / right_imaginary;
    volatile float denominator =
        right_imaginary + right_real * ratio;
    volatile float scale = 1.0f / denominator;
    volatile float real_numerator =
        left_real * ratio + left_imaginary;
    volatile float imaginary_numerator =
        left_imaginary * ratio - left_real;
    result.real = real_numerator * scale;
    result.imag = imaginary_numerator * scale;
    return result;
}

static cnp_cdouble arithmetic_divide_cdouble(
    double left_real,
    double left_imaginary,
    double right_real,
    double right_imaginary) {
    double absolute_real = fabs(right_real);
    double absolute_imaginary = fabs(right_imaginary);
    cnp_cdouble result;
    if (absolute_real >= absolute_imaginary) {
        if (absolute_real == 0.0 && absolute_imaginary == 0.0) {
            result.real = left_real / absolute_real;
            result.imag = left_imaginary / absolute_imaginary;
            return result;
        }
        volatile double ratio = right_imaginary / right_real;
        volatile double denominator =
            right_real + right_imaginary * ratio;
        volatile double scale = 1.0 / denominator;
        volatile double real_numerator =
            left_real + left_imaginary * ratio;
        volatile double imaginary_numerator =
            left_imaginary - left_real * ratio;
        result.real = real_numerator * scale;
        result.imag = imaginary_numerator * scale;
        return result;
    }

    volatile double ratio = right_real / right_imaginary;
    volatile double denominator =
        right_imaginary + right_real * ratio;
    volatile double scale = 1.0 / denominator;
    volatile double real_numerator =
        left_real * ratio + left_imaginary;
    volatile double imaginary_numerator =
        left_imaginary * ratio - left_real;
    result.real = real_numerator * scale;
    result.imag = imaginary_numerator * scale;
    return result;
}

static cnp_clongdouble arithmetic_divide_clongdouble(
    long double left_real,
    long double left_imaginary,
    long double right_real,
    long double right_imaginary) {
    long double absolute_real = fabsl(right_real);
    long double absolute_imaginary = fabsl(right_imaginary);
    cnp_clongdouble result;
    if (absolute_real >= absolute_imaginary) {
        if (absolute_real == 0.0L && absolute_imaginary == 0.0L) {
            result.real = left_real / absolute_real;
            result.imag = left_imaginary / absolute_imaginary;
            return result;
        }
        volatile long double ratio = right_imaginary / right_real;
        volatile long double denominator =
            right_real + right_imaginary * ratio;
        volatile long double scale = 1.0L / denominator;
        volatile long double real_numerator =
            left_real + left_imaginary * ratio;
        volatile long double imaginary_numerator =
            left_imaginary - left_real * ratio;
        result.real = real_numerator * scale;
        result.imag = imaginary_numerator * scale;
        return result;
    }

    volatile long double ratio = right_real / right_imaginary;
    volatile long double denominator =
        right_imaginary + right_real * ratio;
    volatile long double scale = 1.0L / denominator;
    volatile long double real_numerator =
        left_real * ratio + left_imaginary;
    volatile long double imaginary_numerator =
        left_imaginary * ratio - left_real;
    result.real = real_numerator * scale;
    result.imag = imaginary_numerator * scale;
    return result;
}

static CNP_STATUS arithmetic_promoted_element(
    CnpArithmeticOperation operation,
    const void *left_source, CNP_TYPE left_dtype,
    const void *right_source, CNP_TYPE right_dtype,
    void *destination, CNP_TYPE result_dtype,
    const char *function_name) {
    if (operation != CNP_ARITHMETIC_ADD &&
            operation != CNP_ARITHMETIC_SUBTRACT &&
            operation != CNP_ARITHMETIC_MULTIPLY &&
            operation != CNP_ARITHMETIC_DIVIDE &&
            operation != CNP_ARITHMETIC_FLOOR_DIVIDE &&
            operation != CNP_ARITHMETIC_REMAINDER &&
            operation != CNP_ARITHMETIC_FMOD) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "invalid internal arithmetic operation %d", (int)operation);
        return CNP_ERR_GENERIC;
    }
    CnpArithmeticValue left;
    CnpArithmeticValue right;
    CNP_STATUS status = arithmetic_read_value(
        left_source, left_dtype, &left, function_name);
    if (status != CNP_OK) return status;
    status = arithmetic_read_value(
        right_source, right_dtype, &right, function_name);
    if (status != CNP_OK) return status;

    if (result_dtype == CNP_BOOL) {
        if (operation == CNP_ARITHMETIC_SUBTRACT ||
                operation == CNP_ARITHMETIC_DIVIDE ||
                operation == CNP_ARITHMETIC_FLOOR_DIVIDE ||
                operation == CNP_ARITHMETIC_REMAINDER ||
                operation == CNP_ARITHMETIC_FMOD) {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "operation has no supported boolean NumPy ufunc loop");
            return CNP_ERR_TYPE;
        }
        bool left_value = arithmetic_integer_bits(&left) != 0;
        bool right_value = arithmetic_integer_bits(&right) != 0;
        uint8_t result = operation == CNP_ARITHMETIC_ADD
            ? (uint8_t)(left_value || right_value)
            : (uint8_t)(left_value && right_value);
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (cnp_type_is_integer(result_dtype)) {
        if (operation == CNP_ARITHMETIC_DIVIDE) {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "division cannot produce integer result dtype %d",
                (int)result_dtype);
            return CNP_ERR_TYPE;
        }
        if ((left.kind != CNP_ARITHMETIC_SIGNED &&
             left.kind != CNP_ARITHMETIC_UNSIGNED) ||
                (right.kind != CNP_ARITHMETIC_SIGNED &&
                 right.kind != CNP_ARITHMETIC_UNSIGNED)) {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "inexact values cannot be evaluated in integer dtype %d",
                (int)result_dtype);
            return CNP_ERR_TYPE;
        }
        uint64_t left_bits = arithmetic_integer_bits(&left);
        uint64_t right_bits = arithmetic_integer_bits(&right);
        uint64_t result_bits;
        if (operation == CNP_ARITHMETIC_FLOOR_DIVIDE) {
            if (cnp_type_is_unsigned(result_dtype)) {
                result_bits = right_bits == 0
                    ? 0 : left_bits / right_bits;
            } else {
                result_bits = arithmetic_signed_floor_divide_bits(
                    arithmetic_signed_from_bits(left_bits),
                    arithmetic_signed_from_bits(right_bits));
            }
        } else if (operation == CNP_ARITHMETIC_REMAINDER) {
            if (cnp_type_is_unsigned(result_dtype)) {
                result_bits = right_bits == 0
                    ? 0 : left_bits % right_bits;
            } else {
                result_bits = arithmetic_signed_remainder_bits(
                    arithmetic_signed_from_bits(left_bits),
                    arithmetic_signed_from_bits(right_bits));
            }
        } else if (operation == CNP_ARITHMETIC_FMOD) {
            if (cnp_type_is_unsigned(result_dtype)) {
                result_bits = right_bits == 0
                    ? 0 : left_bits % right_bits;
            } else {
                result_bits = arithmetic_signed_fmod_bits(
                    arithmetic_signed_from_bits(left_bits),
                    arithmetic_signed_from_bits(right_bits));
            }
        } else {
            status = arithmetic_unsigned_result(
                operation,
                left_bits,
                right_bits,
                &result_bits,
                function_name);
            if (status != CNP_OK) return status;
        }
        return arithmetic_store_integer_bits(
            destination, result_dtype, result_bits, function_name);
    }
    if (result_dtype == CNP_HALF) {
        uint16_t left_bits = cnp_float_to_half(
            arithmetic_value_double(&left));
        uint16_t right_bits = cnp_float_to_half(
            arithmetic_value_double(&right));
        float left_value = (float)cnp_half_to_float(left_bits);
        float right_value = (float)cnp_half_to_float(right_bits);
        float modulus;
        volatile float result;
        if (operation == CNP_ARITHMETIC_FLOOR_DIVIDE)
            result = arithmetic_divmod_float(
                left_value, right_value, &modulus);
        else if (operation == CNP_ARITHMETIC_REMAINDER) {
            (void)arithmetic_divmod_float(
                left_value, right_value, &modulus);
            result = modulus;
        }
        else if (operation == CNP_ARITHMETIC_FMOD)
            result = fmodf(left_value, right_value);
        else if (operation == CNP_ARITHMETIC_DIVIDE)
            result = left_value / right_value;
        else if (operation == CNP_ARITHMETIC_SUBTRACT)
            result = left_value - right_value;
        else if (operation == CNP_ARITHMETIC_MULTIPLY)
            result = left_value * right_value;
        else
            result = left_value + right_value;
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float left_value = arithmetic_value_float(&left);
        float right_value = arithmetic_value_float(&right);
        float modulus;
        volatile float result;
        if (operation == CNP_ARITHMETIC_FLOOR_DIVIDE)
            result = arithmetic_divmod_float(
                left_value, right_value, &modulus);
        else if (operation == CNP_ARITHMETIC_REMAINDER) {
            (void)arithmetic_divmod_float(
                left_value, right_value, &modulus);
            result = modulus;
        }
        else if (operation == CNP_ARITHMETIC_FMOD)
            result = fmodf(left_value, right_value);
        else if (operation == CNP_ARITHMETIC_DIVIDE)
            result = left_value / right_value;
        else if (operation == CNP_ARITHMETIC_SUBTRACT)
            result = left_value - right_value;
        else if (operation == CNP_ARITHMETIC_MULTIPLY)
            result = left_value * right_value;
        else
            result = left_value + right_value;
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double left_value = arithmetic_value_double(&left);
        double right_value = arithmetic_value_double(&right);
        double modulus;
        volatile double result;
        if (operation == CNP_ARITHMETIC_FLOOR_DIVIDE)
            result = arithmetic_divmod_double(
                left_value, right_value, &modulus);
        else if (operation == CNP_ARITHMETIC_REMAINDER) {
            (void)arithmetic_divmod_double(
                left_value, right_value, &modulus);
            result = modulus;
        }
        else if (operation == CNP_ARITHMETIC_FMOD)
            result = fmod(left_value, right_value);
        else if (operation == CNP_ARITHMETIC_DIVIDE)
            result = left_value / right_value;
        else if (operation == CNP_ARITHMETIC_SUBTRACT)
            result = left_value - right_value;
        else if (operation == CNP_ARITHMETIC_MULTIPLY)
            result = left_value * right_value;
        else
            result = left_value + right_value;
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double left_value = arithmetic_value_longdouble(&left);
        long double right_value = arithmetic_value_longdouble(&right);
        long double modulus;
        volatile long double result;
        if (operation == CNP_ARITHMETIC_FLOOR_DIVIDE)
            result = arithmetic_divmod_longdouble(
                left_value, right_value, &modulus);
        else if (operation == CNP_ARITHMETIC_REMAINDER) {
            (void)arithmetic_divmod_longdouble(
                left_value, right_value, &modulus);
            result = modulus;
        }
        else if (operation == CNP_ARITHMETIC_FMOD)
            result = fmodl(left_value, right_value);
        else if (operation == CNP_ARITHMETIC_DIVIDE)
            result = left_value / right_value;
        else if (operation == CNP_ARITHMETIC_SUBTRACT)
            result = left_value - right_value;
        else if (operation == CNP_ARITHMETIC_MULTIPLY)
            result = left_value * right_value;
        else
            result = left_value + right_value;
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_CFLOAT) {
        float left_real = arithmetic_value_float(&left);
        float left_imaginary = (float)left.imaginary;
        float right_real = arithmetic_value_float(&right);
        float right_imaginary = (float)right.imaginary;
        cnp_cfloat result;
        if (operation == CNP_ARITHMETIC_DIVIDE) {
            result = arithmetic_divide_cfloat(
                left_real, left_imaginary,
                right_real, right_imaginary);
        } else if (operation == CNP_ARITHMETIC_MULTIPLY) {
            volatile float real_left = left_real * right_real;
            volatile float real_right = left_imaginary * right_imaginary;
            volatile float imaginary_left = left_real * right_imaginary;
            volatile float imaginary_right = left_imaginary * right_real;
            result.real = real_left - real_right;
            result.imag = imaginary_left + imaginary_right;
        } else if (operation == CNP_ARITHMETIC_SUBTRACT) {
            result.real = left_real - right_real;
            result.imag = left_imaginary - right_imaginary;
        } else {
            result.real = left_real + right_real;
            result.imag = left_imaginary + right_imaginary;
        }
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_CDOUBLE) {
        double left_real = arithmetic_value_double(&left);
        double left_imaginary = (double)left.imaginary;
        double right_real = arithmetic_value_double(&right);
        double right_imaginary = (double)right.imaginary;
        cnp_cdouble result;
        if (operation == CNP_ARITHMETIC_DIVIDE) {
            result = arithmetic_divide_cdouble(
                left_real, left_imaginary,
                right_real, right_imaginary);
        } else if (operation == CNP_ARITHMETIC_MULTIPLY) {
            volatile double real_left = left_real * right_real;
            volatile double real_right = left_imaginary * right_imaginary;
            volatile double imaginary_left = left_real * right_imaginary;
            volatile double imaginary_right = left_imaginary * right_real;
            result.real = real_left - real_right;
            result.imag = imaginary_left + imaginary_right;
        } else if (operation == CNP_ARITHMETIC_SUBTRACT) {
            result.real = left_real - right_real;
            result.imag = left_imaginary - right_imaginary;
        } else {
            result.real = left_real + right_real;
            result.imag = left_imaginary + right_imaginary;
        }
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_CLONGDOUBLE) {
        long double left_real = arithmetic_value_longdouble(&left);
        long double left_imaginary = left.imaginary;
        long double right_real = arithmetic_value_longdouble(&right);
        long double right_imaginary = right.imaginary;
        cnp_clongdouble result;
        if (operation == CNP_ARITHMETIC_DIVIDE) {
            result = arithmetic_divide_clongdouble(
                left_real, left_imaginary,
                right_real, right_imaginary);
        } else if (operation == CNP_ARITHMETIC_MULTIPLY) {
            volatile long double real_left = left_real * right_real;
            volatile long double real_right = left_imaginary * right_imaginary;
            volatile long double imaginary_left =
                left_real * right_imaginary;
            volatile long double imaginary_right =
                left_imaginary * right_real;
            result.real = real_left - real_right;
            result.imag = imaginary_left + imaginary_right;
        } else if (operation == CNP_ARITHMETIC_SUBTRACT) {
            result.real = left_real - right_real;
            result.imag = left_imaginary - right_imaginary;
        } else {
            result.real = left_real + right_real;
            result.imag = left_imaginary + right_imaginary;
        }
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }

    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "result dtype %d is not supported for arithmetic",
        (int)result_dtype);
    return CNP_ERR_TYPE;
}

static CNP_STATUS arithmetic_promoted_divmod_element(
    const void *left_source,
    CNP_TYPE left_dtype,
    const void *right_source,
    CNP_TYPE right_dtype,
    void *quotient_destination,
    void *remainder_destination,
    CNP_TYPE result_dtype,
    const char *function_name) {
    CnpArithmeticValue left;
    CnpArithmeticValue right;
    CNP_STATUS status = arithmetic_read_value(
        left_source, left_dtype, &left, function_name);
    if (status != CNP_OK) return status;
    status = arithmetic_read_value(
        right_source, right_dtype, &right, function_name);
    if (status != CNP_OK) return status;

    if (cnp_type_is_integer(result_dtype)) {
        if ((left.kind != CNP_ARITHMETIC_SIGNED &&
             left.kind != CNP_ARITHMETIC_UNSIGNED) ||
                (right.kind != CNP_ARITHMETIC_SIGNED &&
                 right.kind != CNP_ARITHMETIC_UNSIGNED)) {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "inexact values cannot be evaluated in integer dtype %d",
                (int)result_dtype);
            return CNP_ERR_TYPE;
        }
        uint64_t left_bits = arithmetic_integer_bits(&left);
        uint64_t right_bits = arithmetic_integer_bits(&right);
        uint64_t quotient_bits;
        uint64_t remainder_bits;
        if (cnp_type_is_unsigned(result_dtype)) {
            if (right_bits == 0) {
                quotient_bits = 0;
                remainder_bits = 0;
            } else {
                quotient_bits = left_bits / right_bits;
                remainder_bits = left_bits % right_bits;
            }
        } else {
            int64_t left_value = arithmetic_signed_from_bits(left_bits);
            int64_t right_value = arithmetic_signed_from_bits(right_bits);
            quotient_bits = arithmetic_signed_floor_divide_bits(
                left_value, right_value);
            remainder_bits = arithmetic_signed_remainder_bits(
                left_value, right_value);
        }
        status = arithmetic_store_integer_bits(
            quotient_destination, result_dtype,
            quotient_bits, function_name);
        if (status != CNP_OK) return status;
        return arithmetic_store_integer_bits(
            remainder_destination, result_dtype,
            remainder_bits, function_name);
    }
    if (result_dtype == CNP_HALF || result_dtype == CNP_FLOAT) {
        float left_value;
        float right_value;
        if (result_dtype == CNP_HALF) {
            uint16_t left_bits = cnp_float_to_half(
                arithmetic_value_double(&left));
            uint16_t right_bits = cnp_float_to_half(
                arithmetic_value_double(&right));
            left_value = (float)cnp_half_to_float(left_bits);
            right_value = (float)cnp_half_to_float(right_bits);
        } else {
            left_value = arithmetic_value_float(&left);
            right_value = arithmetic_value_float(&right);
        }
        float remainder_value;
        volatile float quotient_value = arithmetic_divmod_float(
            left_value, right_value, &remainder_value);
        if (result_dtype == CNP_HALF) {
            uint16_t quotient_bits = cnp_float_to_half(
                (double)quotient_value);
            uint16_t remainder_bits = cnp_float_to_half(
                (double)remainder_value);
            memcpy(
                quotient_destination, &quotient_bits,
                sizeof(quotient_bits));
            memcpy(
                remainder_destination, &remainder_bits,
                sizeof(remainder_bits));
            return CNP_OK;
        }
        memcpy(
            quotient_destination, (const void*)&quotient_value,
            sizeof(quotient_value));
        memcpy(
            remainder_destination, &remainder_value,
            sizeof(remainder_value));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double remainder_value;
        volatile double quotient_value = arithmetic_divmod_double(
            arithmetic_value_double(&left),
            arithmetic_value_double(&right),
            &remainder_value);
        memcpy(
            quotient_destination, (const void*)&quotient_value,
            sizeof(quotient_value));
        memcpy(
            remainder_destination, &remainder_value,
            sizeof(remainder_value));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double remainder_value;
        volatile long double quotient_value = arithmetic_divmod_longdouble(
            arithmetic_value_longdouble(&left),
            arithmetic_value_longdouble(&right),
            &remainder_value);
        memcpy(
            quotient_destination, (const void*)&quotient_value,
            sizeof(quotient_value));
        memcpy(
            remainder_destination, &remainder_value,
            sizeof(remainder_value));
        return CNP_OK;
    }

    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "result dtype %d is not supported for divmod",
        (int)result_dtype);
    return CNP_ERR_TYPE;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static int64_t arithmetic_broadcast_offset(
    const CnpArray *array,
    const int64_t *result_coordinates,
    int result_ndim) {
    int result_axis = result_ndim - array->ndim;
    int64_t offset = array->offset;
    for (int axis = 0; axis < array->ndim; ++axis) {
        int64_t coordinate = array->shape[axis] == 1
            ? 0 : result_coordinates[result_axis + axis];
        offset += coordinate * array->strides[axis];
    }
    return offset;
}

static bool arithmetic_input_broadcasts_to_output(
    const CnpArray *input, const CnpArray *output) {
    if (input->ndim > output->ndim) return false;
    int output_axis = output->ndim - input->ndim;
    for (int input_axis = 0; input_axis < input->ndim;
            ++input_axis, ++output_axis) {
        int64_t input_dimension = input->shape[input_axis];
        int64_t output_dimension = output->shape[output_axis];
        if (input_dimension != 1 &&
                input_dimension != output_dimension)
            return false;
    }
    return true;
}

static const CnpArray *arithmetic_storage_root(const CnpArray *array) {
    while (array && array->base) array = array->base;
    return array;
}

static bool arithmetic_arrays_share_storage(
    const CnpArray *left, const CnpArray *right) {
    if (left == right) return true;
    if (arithmetic_storage_root(left) == arithmetic_storage_root(right))
        return true;
    if (left->owner && left->owner == right->owner) return true;
    return left->data == right->data;
}

static bool arithmetic_arrays_have_same_element_mapping(
    const CnpArray *left, const CnpArray *right) {
    if (left->dtype->type_num != right->dtype->type_num ||
            left->ndim != right->ndim || left->size != right->size)
        return false;
    if (left->data != right->data || left->offset != right->offset)
        return false;
    for (int axis = 0; axis < left->ndim; ++axis) {
        if (left->shape[axis] != right->shape[axis] ||
                left->strides[axis] != right->strides[axis])
            return false;
    }
    return true;
}

static CNP_ORDER arithmetic_result_order(
    const CnpArray *left, const CnpArray *right) {
    const CnpArray *inputs[2] = {left, right};
    bool prefer_fortran = false;
    for (int input_index = 0; input_index < 2; ++input_index) {
        const CnpArray *input = inputs[input_index];
        bool c_contiguous =
            (input->flags & CNP_ARRAY_C_CONTIGUOUS) != 0;
        bool f_contiguous =
            (input->flags & CNP_ARRAY_F_CONTIGUOUS) != 0;
        if (input->ndim < 2 || input->size <= 1 ||
                (c_contiguous && f_contiguous))
            continue;
        if (f_contiguous && !c_contiguous) {
            prefer_fortran = true;
            continue;
        }
        return CNP_ORDER_C;
    }
    return prefer_fortran ? CNP_ORDER_F : CNP_ORDER_C;
}

static bool arithmetic_dtype_supported(CNP_TYPE dtype) {
    return dtype == CNP_BOOL || cnp_type_is_integer(dtype) ||
        cnp_type_is_float(dtype) || cnp_type_is_complex(dtype);
}

static CNP_STATUS arithmetic_validate_inputs(
    CnpArithmeticOperation operation,
    const CnpArray *left,
    const CnpArray *right,
    const char *function_name,
    const char *operation_name,
    CNP_TYPE *result_dtype) {
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays are required");
        return CNP_ERR_GENERIC;
    }
    if (!left->dtype || !right->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "left and right arrays must have dtypes");
        return CNP_ERR_TYPE;
    }
    if ((left->size > 0 && !left->data) ||
            (right->size > 0 && !right->data)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays require data buffers");
        return CNP_ERR_GENERIC;
    }

    CNP_TYPE promoted = arith_result_type(left, right);
    if (operation == CNP_ARITHMETIC_DIVIDE &&
            (promoted == CNP_BOOL || cnp_type_is_integer(promoted)))
        promoted = CNP_DOUBLE;
    if ((operation == CNP_ARITHMETIC_FLOOR_DIVIDE ||
         operation == CNP_ARITHMETIC_REMAINDER ||
         operation == CNP_ARITHMETIC_FMOD) &&
            promoted == CNP_BOOL)
        promoted = CNP_BYTE;
    if (!arithmetic_dtype_supported(left->dtype->type_num) ||
            !arithmetic_dtype_supported(right->dtype->type_num) ||
            !arithmetic_dtype_supported(promoted)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtypes %s and %s do not support %s",
            left->dtype->name,
            right->dtype->name,
            operation_name);
        return CNP_ERR_TYPE;
    }
    if ((operation == CNP_ARITHMETIC_FLOOR_DIVIDE ||
         operation == CNP_ARITHMETIC_REMAINDER ||
         operation == CNP_ARITHMETIC_FMOD) &&
            (cnp_type_is_complex(left->dtype->type_num) ||
             cnp_type_is_complex(right->dtype->type_num))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "complex %s has no supported NumPy ufunc loop",
            operation_name);
        return CNP_ERR_TYPE;
    }
    if (operation == CNP_ARITHMETIC_SUBTRACT &&
            promoted == CNP_BOOL) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "boolean subtraction has no supported NumPy ufunc loop");
        return CNP_ERR_TYPE;
    }
    *result_dtype = promoted;
    return CNP_OK;
}

static CnpArray* arithmetic_prepare_result(
    const CnpArray *left,
    const CnpArray *right,
    CNP_TYPE result_dtype,
    const char *function_name) {
    if (!cnp_can_broadcast(left, right)) {
        cnp_set_error(
            CNP_ERR_BROADCAST, function_name,
            "left and right arrays cannot be broadcast together");
        return NULL;
    }

    int result_ndim = left->ndim > right->ndim
        ? left->ndim : right->ndim;
    int64_t result_shape[CNP_MAXDIMS];
    for (int axis = 0; axis < result_ndim; ++axis) {
        int64_t left_dimension = axis < result_ndim - left->ndim
            ? 1 : left->shape[axis - (result_ndim - left->ndim)];
        int64_t right_dimension = axis < result_ndim - right->ndim
            ? 1 : right->shape[axis - (result_ndim - right->ndim)];
        result_shape[axis] = left_dimension == 1
            ? right_dimension : left_dimension;
    }
    return cnp_array_new(
        result_ndim, result_shape, result_dtype,
        arithmetic_result_order(left, right));
}

static CnpArray* arithmetic_promoted_arrays(
    CnpArithmeticOperation operation,
    const CnpArray *left,
    const CnpArray *right,
    CNP_TYPE result_dtype,
    const char *function_name) {
    CnpArray *result = arithmetic_prepare_result(
        left, right, result_dtype, function_name);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t left_offset = arithmetic_broadcast_offset(
            left, coordinates, result->ndim);
        int64_t right_offset = arithmetic_broadcast_offset(
            right, coordinates, result->ndim);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = arithmetic_promoted_element(
            operation,
            (const char*)left->data + left_offset,
            left->dtype->type_num,
            (const char*)right->data + right_offset,
            right->dtype->type_num,
            (char*)result->data + result_offset,
            result_dtype,
            function_name);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

CNP_STATUS cnp_add_into_promoted(
    const CnpArray *left,
    const CnpArray *right,
    CnpArray *output) {
    const char *function_name = "cnp_add_into";
    if (!left || !right || !output) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left, right, and output arrays are required");
        return CNP_ERR_GENERIC;
    }
    if (!left->dtype || !right->dtype || !output->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "left, right, and output arrays must have dtypes");
        return CNP_ERR_TYPE;
    }
    if (!(output->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "output array must be writeable");
        return CNP_ERR_GENERIC;
    }
    CNP_TYPE result_dtype = arith_result_type(left, right);
    if (result_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtypes %s and %s do not support addition",
            left->dtype->name, right->dtype->name);
        return CNP_ERR_TYPE;
    }
    if (!cnp_dtype_can_cast(
            result_dtype,
            output->dtype->type_num,
            CNP_CAST_SAME_KIND)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "cannot cast addition result dtype %d to output dtype %s "
            "under the same_kind rule",
            (int)result_dtype, output->dtype->name);
        return CNP_ERR_TYPE;
    }

    if (!arithmetic_input_broadcasts_to_output(left, output) ||
            !arithmetic_input_broadcasts_to_output(right, output)) {
        cnp_set_error(
            CNP_ERR_BROADCAST, function_name,
            "left and right shapes cannot broadcast to the output shape");
        return CNP_ERR_BROADCAST;
    }

    if ((left->size > 0 && !left->data) ||
            (right->size > 0 && !right->data) ||
            (output->size > 0 && !output->data)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left, right, and output arrays require data buffers");
        return CNP_ERR_GENERIC;
    }

    bool left_overlap_requires_temporary =
        arithmetic_arrays_share_storage(left, output) &&
        !arithmetic_arrays_have_same_element_mapping(left, output);
    bool right_overlap_requires_temporary =
        arithmetic_arrays_share_storage(right, output) &&
        !arithmetic_arrays_have_same_element_mapping(right, output);
    if (left_overlap_requires_temporary ||
            right_overlap_requires_temporary) {
        CnpArray *temporary = arithmetic_promoted_arrays(
            CNP_ARITHMETIC_ADD,
            left, right, result_dtype, function_name);
        if (!temporary) {
            cnp_relabel_error(function_name);
            return cnp_get_error(NULL);
        }
        CNP_STATUS status = cnp_copyto(
            output, temporary, CNP_CAST_SAME_KIND);
        cnp_array_free(temporary);
        if (status != CNP_OK) cnp_relabel_error(function_name);
        return status;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < output->size; ++index) {
        int64_t left_offset = arithmetic_broadcast_offset(
            left, coordinates, output->ndim);
        int64_t right_offset = arithmetic_broadcast_offset(
            right, coordinates, output->ndim);
        int64_t output_offset = output->offset + cnp_multi_to_offset(
            output->ndim, coordinates, output->strides);
        void *output_pointer = (char*)output->data + output_offset;
        CNP_STATUS status;
        if (output->dtype->type_num == result_dtype) {
            status = arithmetic_promoted_element(
                CNP_ARITHMETIC_ADD,
                (const char*)left->data + left_offset,
                left->dtype->type_num,
                (const char*)right->data + right_offset,
                right->dtype->type_num,
                output_pointer,
                result_dtype,
                function_name);
        } else {
            CnpArithmeticScalarStorage intermediate = {0};
            status = arithmetic_promoted_element(
                CNP_ARITHMETIC_ADD,
                (const char*)left->data + left_offset,
                left->dtype->type_num,
                (const char*)right->data + right_offset,
                right->dtype->type_num,
                &intermediate,
                result_dtype,
                function_name);
            if (status == CNP_OK) {
                status = cnp_cast_scalar_value(
                    &intermediate,
                    result_dtype,
                    output_pointer,
                    output->dtype->type_num,
                    function_name);
            }
        }
        if (status != CNP_OK) return status;
        for (int dimension = output->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < output->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return CNP_OK;
}

CNP_API CnpArray* CNP_CALL cnp_add(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_add";
    CNP_TYPE dt = CNP_NOTYPE;
    if (arithmetic_validate_inputs(
            CNP_ARITHMETIC_ADD,
            a, b, function_name, "addition", &dt) != CNP_OK)
        return NULL;
    if ((a->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (b->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        a->size > 0 &&
        a->dtype->type_num == CNP_DOUBLE &&
        b->dtype->type_num == CNP_DOUBLE && same_shape(a, b)) {
        CnpArray *result = cnp_array_new(
            a->ndim, a->shape, CNP_DOUBLE, CNP_ORDER_C);
        if (!result) {
            cnp_relabel_error(function_name);
            return NULL;
        }
        cnp_simd_add(
            (const double*)((const char*)a->data + a->offset),
            (const double*)((const char*)b->data + b->offset),
            (double*)result->data, a->size);
        return result;
    }
    return arithmetic_promoted_arrays(
        CNP_ARITHMETIC_ADD, a, b, dt, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_subtract(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_subtract";
    CNP_TYPE dt = CNP_NOTYPE;
    if (arithmetic_validate_inputs(
            CNP_ARITHMETIC_SUBTRACT,
            a, b, function_name, "subtraction", &dt) != CNP_OK)
        return NULL;
    if ((a->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (b->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        a->size > 0 &&
        a->dtype->type_num == CNP_DOUBLE &&
        b->dtype->type_num == CNP_DOUBLE && same_shape(a, b)) {
        CnpArray *result = cnp_array_new(
            a->ndim, a->shape, CNP_DOUBLE, CNP_ORDER_C);
        if (!result) {
            cnp_relabel_error(function_name);
            return NULL;
        }
        cnp_simd_subtract(
            (const double*)((const char*)a->data + a->offset),
            (const double*)((const char*)b->data + b->offset),
            (double*)result->data, a->size);
        return result;
    }
    return arithmetic_promoted_arrays(
        CNP_ARITHMETIC_SUBTRACT, a, b, dt, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_multiply(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_multiply";
    CNP_TYPE dt = CNP_NOTYPE;
    if (arithmetic_validate_inputs(
            CNP_ARITHMETIC_MULTIPLY,
            a, b, function_name, "multiplication", &dt) != CNP_OK)
        return NULL;
    if ((a->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (b->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        a->size > 0 &&
        a->dtype->type_num == CNP_DOUBLE &&
        b->dtype->type_num == CNP_DOUBLE && same_shape(a, b)) {
        CnpArray *result = cnp_array_new(
            a->ndim, a->shape, CNP_DOUBLE, CNP_ORDER_C);
        if (!result) {
            cnp_relabel_error(function_name);
            return NULL;
        }
        cnp_simd_multiply(
            (const double*)((const char*)a->data + a->offset),
            (const double*)((const char*)b->data + b->offset),
            (double*)result->data, a->size);
        return result;
    }
    return arithmetic_promoted_arrays(
        CNP_ARITHMETIC_MULTIPLY, a, b, dt, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_divide(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_divide";
    CNP_TYPE dt = CNP_NOTYPE;
    if (arithmetic_validate_inputs(
            CNP_ARITHMETIC_DIVIDE,
            a, b, function_name, "division", &dt) != CNP_OK)
        return NULL;
    if ((a->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (b->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        a->size > 0 &&
        a->dtype->type_num == CNP_DOUBLE &&
        b->dtype->type_num == CNP_DOUBLE && same_shape(a, b)) {
        CnpArray *result = cnp_array_new(
            a->ndim, a->shape, CNP_DOUBLE, CNP_ORDER_C);
        if (!result) {
            cnp_relabel_error(function_name);
            return NULL;
        }
        cnp_simd_divide(
            (const double*)((const char*)a->data + a->offset),
            (const double*)((const char*)b->data + b->offset),
            (double*)result->data, a->size);
        return result;
    }
    return arithmetic_promoted_arrays(
        CNP_ARITHMETIC_DIVIDE, a, b, dt, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_true_divide(const CnpArray *a, const CnpArray *b) {
    CnpArray *result = cnp_divide(a, b);
    if (!result) cnp_relabel_error("cnp_true_divide");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_floor_divide(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_floor_divide";
    CNP_TYPE dt = CNP_NOTYPE;
    if (arithmetic_validate_inputs(
            CNP_ARITHMETIC_FLOOR_DIVIDE,
            a, b, function_name, "floor division", &dt) != CNP_OK)
        return NULL;
    return arithmetic_promoted_arrays(
        CNP_ARITHMETIC_FLOOR_DIVIDE,
        a, b, dt, function_name);
}

static CNP_TYPE power_result_dtype(
    CNP_TYPE left_dtype, CNP_TYPE right_dtype, bool float_power) {
    if (!arithmetic_dtype_supported(left_dtype) ||
            !arithmetic_dtype_supported(right_dtype))
        return CNP_NOTYPE;
    if (!float_power) {
        if (left_dtype == CNP_BOOL && right_dtype == CNP_BOOL)
            return CNP_BYTE;
        return cnp_promote_type(left_dtype, right_dtype);
    }
    if (left_dtype == CNP_CLONGDOUBLE ||
            right_dtype == CNP_CLONGDOUBLE)
        return CNP_CLONGDOUBLE;
    if (cnp_type_is_complex(left_dtype) ||
            cnp_type_is_complex(right_dtype)) {
        if (left_dtype == CNP_LONGDOUBLE ||
                right_dtype == CNP_LONGDOUBLE)
            return CNP_CLONGDOUBLE;
        return CNP_CDOUBLE;
    }
    if (left_dtype == CNP_LONGDOUBLE ||
            right_dtype == CNP_LONGDOUBLE)
        return CNP_LONGDOUBLE;
    return CNP_DOUBLE;
}

static CNP_STATUS power_validate_inputs(
    const CnpArray *left,
    const CnpArray *right,
    bool float_power,
    const char *function_name,
    CNP_TYPE *result_dtype) {
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays are required");
        return CNP_ERR_GENERIC;
    }
    if (!left->dtype || !right->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "left and right arrays must have dtypes");
        return CNP_ERR_TYPE;
    }
    if ((left->size > 0 && !left->data) ||
            (right->size > 0 && !right->data)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays require data buffers");
        return CNP_ERR_GENERIC;
    }
    CNP_TYPE resolved = power_result_dtype(
        left->dtype->type_num, right->dtype->type_num, float_power);
    if (resolved == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtypes %s and %s do not support %s",
            left->dtype->name,
            right->dtype->name,
            float_power ? "float_power" : "power");
        return CNP_ERR_TYPE;
    }
    if (!cnp_can_broadcast(left, right)) {
        cnp_set_error(
            CNP_ERR_BROADCAST, function_name,
            "left and right arrays cannot be broadcast together");
        return CNP_ERR_BROADCAST;
    }
    *result_dtype = resolved;
    return CNP_OK;
}

static uint64_t power_integer_bits(uint64_t base, uint64_t exponent) {
    uint64_t result = UINT64_C(1);
    while (exponent != 0) {
        if (exponent & UINT64_C(1)) result *= base;
        exponent >>= 1;
        if (exponent != 0) base *= base;
    }
    return result;
}

static void power_store_integer_bits(
    void *destination, CNP_TYPE dtype, uint64_t bits) {
    if (dtype == CNP_BYTE || dtype == CNP_UBYTE) {
        uint8_t value = (uint8_t)bits;
        memcpy(destination, &value, sizeof(value));
    } else if (dtype == CNP_SHORT || dtype == CNP_USHORT) {
        uint16_t value = (uint16_t)bits;
        memcpy(destination, &value, sizeof(value));
    } else if (dtype == CNP_INT || dtype == CNP_UINT) {
        uint32_t value = (uint32_t)bits;
        memcpy(destination, &value, sizeof(value));
    } else {
        memcpy(destination, &bits, sizeof(bits));
    }
}

static cnp_cfloat power_cfloat(
    cnp_cfloat base, cnp_cfloat exponent) {
    if (exponent.real == 0.0f && exponent.imag == 0.0f) {
        cnp_cfloat one = {1.0f, 0.0f};
        return one;
    }
    if (base.real == 0.0f && base.imag == 0.0f &&
            !(exponent.imag == 0.0f && exponent.real > 0.0f)) {
        cnp_cfloat invalid = {NAN, NAN};
        return invalid;
    }
#ifdef _MSC_VER
    _Fcomplex native_base;
    _Fcomplex native_exponent;
    _Fcomplex native_result;
    cnp_cfloat result;
    memcpy(&native_base, &base, sizeof(native_base));
    memcpy(&native_exponent, &exponent, sizeof(native_exponent));
    native_result = cpowf(native_base, native_exponent);
    memcpy(&result, &native_result, sizeof(result));
#else
    float complex native_result = cpowf(
        CMPLXF(base.real, base.imag),
        CMPLXF(exponent.real, exponent.imag));
    cnp_cfloat result = {
        crealf(native_result), cimagf(native_result)};
#endif
    return result;
}

static cnp_cdouble power_cdouble(
    cnp_cdouble base, cnp_cdouble exponent) {
    if (exponent.real == 0.0 && exponent.imag == 0.0) {
        cnp_cdouble one = {1.0, 0.0};
        return one;
    }
    if (base.real == 0.0 && base.imag == 0.0 &&
            !(exponent.imag == 0.0 && exponent.real > 0.0)) {
        cnp_cdouble invalid = {NAN, NAN};
        return invalid;
    }
#ifdef _MSC_VER
    _Dcomplex native_base;
    _Dcomplex native_exponent;
    _Dcomplex native_result;
    cnp_cdouble result;
    memcpy(&native_base, &base, sizeof(native_base));
    memcpy(&native_exponent, &exponent, sizeof(native_exponent));
    native_result = cpow(native_base, native_exponent);
    memcpy(&result, &native_result, sizeof(result));
#else
    double complex native_result = cpow(
        CMPLX(base.real, base.imag),
        CMPLX(exponent.real, exponent.imag));
    cnp_cdouble result = {
        creal(native_result), cimag(native_result)};
#endif
    return result;
}

static cnp_clongdouble power_clongdouble(
    cnp_clongdouble base, cnp_clongdouble exponent) {
    if (exponent.real == 0.0L && exponent.imag == 0.0L) {
        cnp_clongdouble one = {1.0L, 0.0L};
        return one;
    }
    if (base.real == 0.0L && base.imag == 0.0L &&
            !(exponent.imag == 0.0L && exponent.real > 0.0L)) {
        cnp_clongdouble invalid = {NAN, NAN};
        return invalid;
    }
#ifdef _MSC_VER
    _Lcomplex native_base;
    _Lcomplex native_exponent;
    _Lcomplex native_result;
    cnp_clongdouble result;
    memcpy(&native_base, &base, sizeof(native_base));
    memcpy(&native_exponent, &exponent, sizeof(native_exponent));
    native_result = cpowl(native_base, native_exponent);
    memcpy(&result, &native_result, sizeof(result));
#else
    long double complex native_result = cpowl(
        CMPLXL(base.real, base.imag),
        CMPLXL(exponent.real, exponent.imag));
    cnp_clongdouble result = {
        creall(native_result), cimagl(native_result)};
#endif
    return result;
}

static CNP_STATUS power_element(
    const void *left_source,
    CNP_TYPE left_dtype,
    const void *right_source,
    CNP_TYPE right_dtype,
    void *destination,
    CNP_TYPE result_dtype,
    const char *function_name) {
    CnpArithmeticValue left;
    CnpArithmeticValue right;
    CNP_STATUS status = arithmetic_read_value(
        left_source, left_dtype, &left, function_name);
    if (status != CNP_OK) return status;
    status = arithmetic_read_value(
        right_source, right_dtype, &right, function_name);
    if (status != CNP_OK) return status;

    if (cnp_type_is_integer(result_dtype)) {
        if (right.kind == CNP_ARITHMETIC_SIGNED &&
                right.signed_value < 0) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "Integers to negative integer powers are not allowed");
            return CNP_ERR_GENERIC;
        }
        uint64_t exponent = right.kind == CNP_ARITHMETIC_SIGNED
            ? (uint64_t)right.signed_value
            : right.unsigned_value;
        uint64_t bits = power_integer_bits(
            arithmetic_integer_bits(&left), exponent);
        power_store_integer_bits(destination, result_dtype, bits);
        return CNP_OK;
    }
    if (result_dtype == CNP_HALF) {
        uint16_t left_bits = cnp_float_to_half(
            arithmetic_value_double(&left));
        uint16_t right_bits = cnp_float_to_half(
            arithmetic_value_double(&right));
        volatile float result = powf(
            (float)cnp_half_to_float(left_bits),
            (float)cnp_half_to_float(right_bits));
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        volatile float result = powf(
            arithmetic_value_float(&left),
            arithmetic_value_float(&right));
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        volatile double result = pow(
            arithmetic_value_double(&left),
            arithmetic_value_double(&right));
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        volatile long double result = powl(
            arithmetic_value_longdouble(&left),
            arithmetic_value_longdouble(&right));
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_CFLOAT) {
        cnp_cfloat base = {
            arithmetic_value_float(&left), (float)left.imaginary};
        cnp_cfloat exponent = {
            arithmetic_value_float(&right), (float)right.imaginary};
        cnp_cfloat result = power_cfloat(base, exponent);
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_CDOUBLE) {
        cnp_cdouble base = {
            arithmetic_value_double(&left), (double)left.imaginary};
        cnp_cdouble exponent = {
            arithmetic_value_double(&right), (double)right.imaginary};
        cnp_cdouble result = power_cdouble(base, exponent);
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_CLONGDOUBLE) {
        cnp_clongdouble base = {
            arithmetic_value_longdouble(&left), left.imaginary};
        cnp_clongdouble exponent = {
            arithmetic_value_longdouble(&right), right.imaginary};
        cnp_clongdouble result = power_clongdouble(base, exponent);
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "result dtype %d is not supported for power", (int)result_dtype);
    return CNP_ERR_TYPE;
}

static bool power_contiguous_typed(
    const CnpArray *left,
    const CnpArray *right,
    CnpArray *result,
    const char *function_name,
    CNP_STATUS *status) {
    bool c_contiguous =
        (left->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (right->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_C_CONTIGUOUS);
    bool f_contiguous =
        (left->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (right->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_F_CONTIGUOUS);
    if ((!c_contiguous && !f_contiguous) || !same_shape(left, right))
        return false;
    if (result->size == 0) {
        *status = CNP_OK;
        return true;
    }

    const char *left_values =
        (const char*)left->data + left->offset;
    const char *right_values =
        (const char*)right->data + right->offset;
    char *result_values = (char*)result->data + result->offset;
    int left_size = left->dtype->elsize;
    int right_size = right->dtype->elsize;
    int result_size = result->dtype->elsize;

    if (left->dtype->type_num == CNP_DOUBLE &&
            right->dtype->type_num == CNP_DOUBLE &&
            result->dtype->type_num == CNP_DOUBLE) {
        const double *left_doubles = (const double*)left_values;
        const double *right_doubles = (const double*)right_values;
        double *result_doubles = (double*)result_values;
        int64_t index = 0;
        for (; index + 3 < result->size; index += 4) {
            result_doubles[index] = pow(
                left_doubles[index], right_doubles[index]);
            result_doubles[index + 1] = pow(
                left_doubles[index + 1], right_doubles[index + 1]);
            result_doubles[index + 2] = pow(
                left_doubles[index + 2], right_doubles[index + 2]);
            result_doubles[index + 3] = pow(
                left_doubles[index + 3], right_doubles[index + 3]);
        }
        for (; index < result->size; ++index) {
            result_doubles[index] = pow(
                left_doubles[index], right_doubles[index]);
        }
        *status = CNP_OK;
        return true;
    }

    for (int64_t index = 0; index < result->size; ++index) {
        *status = power_element(
            left_values + index * left_size,
            left->dtype->type_num,
            right_values + index * right_size,
            right->dtype->type_num,
            result_values + index * result_size,
            result->dtype->type_num,
            function_name);
        if (*status != CNP_OK) return true;
    }
    *status = CNP_OK;
    return true;
}

static CnpArray *power_arrays(
    const CnpArray *left,
    const CnpArray *right,
    bool float_power,
    const char *function_name) {
    CNP_TYPE result_dtype = CNP_NOTYPE;
    CNP_STATUS status = power_validate_inputs(
        left, right, float_power, function_name, &result_dtype);
    if (status != CNP_OK) return NULL;
    CnpArray *result = arithmetic_prepare_result(
        left, right, result_dtype, function_name);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    if (power_contiguous_typed(
            left, right, result, function_name, &status)) {
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t left_offset = arithmetic_broadcast_offset(
            left, coordinates, result->ndim);
        int64_t right_offset = arithmetic_broadcast_offset(
            right, coordinates, result->ndim);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        status = power_element(
            (const char*)left->data + left_offset,
            left->dtype->type_num,
            (const char*)right->data + right_offset,
            right->dtype->type_num,
            (char*)result->data + result_offset,
            result_dtype,
            function_name);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_power(
    const CnpArray *a, const CnpArray *b) {
    return power_arrays(a, b, false, "cnp_power");
}

CNP_API CnpArray* CNP_CALL cnp_float_power(
    const CnpArray *a, const CnpArray *b) {
    return power_arrays(a, b, true, "cnp_float_power");
}

static int heaviside_loop_rank(CNP_TYPE dtype) {
    switch (dtype) {
        case CNP_BOOL:
        case CNP_BYTE:
        case CNP_UBYTE:
        case CNP_HALF:
            return 0;
        case CNP_SHORT:
        case CNP_USHORT:
        case CNP_FLOAT:
            return 1;
        case CNP_INT:
        case CNP_UINT:
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
        case CNP_DOUBLE:
            return 2;
        case CNP_LONGDOUBLE:
            return 3;
        default:
            return -1;
    }
}

static CNP_TYPE heaviside_loop_dtype(int rank) {
    switch (rank) {
        case 0: return CNP_HALF;
        case 1: return CNP_FLOAT;
        case 2: return CNP_DOUBLE;
        case 3: return CNP_LONGDOUBLE;
        default: return CNP_NOTYPE;
    }
}

static CNP_STATUS heaviside_validate_inputs(
    const CnpArray *left,
    const CnpArray *right,
    CNP_TYPE *result_dtype,
    const char *function_name) {
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays are required");
        return CNP_ERR_GENERIC;
    }
    if (!left->dtype || !right->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "left and right arrays must have dtypes");
        return CNP_ERR_TYPE;
    }
    if ((left->size > 0 && !left->data) ||
            (right->size > 0 && !right->data)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays require data buffers");
        return CNP_ERR_GENERIC;
    }

    int left_rank = heaviside_loop_rank(left->dtype->type_num);
    int right_rank = heaviside_loop_rank(right->dtype->type_num);
    if (left_rank < 0 || right_rank < 0) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtypes %s and %s do not support heaviside",
            left->dtype->name, right->dtype->name);
        return CNP_ERR_TYPE;
    }
    *result_dtype = heaviside_loop_dtype(
        left_rank > right_rank ? left_rank : right_rank);
    return CNP_OK;
}

static CNP_STATUS heaviside_store_value(
    void *destination,
    CNP_TYPE result_dtype,
    long double value,
    const char *function_name) {
    if (result_dtype == CNP_HALF) {
        uint16_t result = cnp_float_to_half((double)value);
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float result = (float)value;
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double result = (double)value;
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "result dtype %d is not supported for heaviside",
        (int)result_dtype);
    return CNP_ERR_TYPE;
}

static int heaviside_value_class(const CnpArithmeticValue *value) {
    if (value->kind == CNP_ARITHMETIC_SIGNED) {
        if (value->signed_value < 0) return -1;
        return value->signed_value > 0 ? 1 : 0;
    }
    if (value->kind == CNP_ARITHMETIC_UNSIGNED)
        return value->unsigned_value > 0 ? 1 : 0;
    if (isnan(value->real)) return 2;
    if (value->real < 0.0L) return -1;
    return value->real > 0.0L ? 1 : 0;
}

static CNP_STATUS heaviside_element(
    const void *left_source,
    CNP_TYPE left_dtype,
    const void *right_source,
    CNP_TYPE right_dtype,
    void *destination,
    CNP_TYPE result_dtype,
    const char *function_name) {
    CnpArithmeticValue left;
    CnpArithmeticValue right;
    CNP_STATUS status = arithmetic_read_value(
        left_source, left_dtype, &left, function_name);
    if (status != CNP_OK) return status;
    status = arithmetic_read_value(
        right_source, right_dtype, &right, function_name);
    if (status != CNP_OK) return status;

    int value_class = heaviside_value_class(&left);
    long double result_value;
    if (value_class < 0)
        result_value = 0.0L;
    else if (value_class > 1)
        result_value = (long double)NAN;
    else if (value_class > 0)
        result_value = 1.0L;
    else
        result_value = arithmetic_value_longdouble(&right);
    return heaviside_store_value(
        destination, result_dtype, result_value, function_name);
}

static bool heaviside_contiguous_typed(
    const CnpArray *left,
    const CnpArray *right,
    CnpArray *result,
    const char *function_name,
    CNP_STATUS *status) {
    bool c_contiguous =
        (left->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (right->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_C_CONTIGUOUS);
    bool f_contiguous =
        (left->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (right->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_F_CONTIGUOUS);
    if ((!c_contiguous && !f_contiguous) || !same_shape(left, right))
        return false;
    if (result->size == 0) {
        *status = CNP_OK;
        return true;
    }

    const char *left_values =
        (const char*)left->data + left->offset;
    const char *right_values =
        (const char*)right->data + right->offset;
    char *result_values = (char*)result->data + result->offset;

    if (left->dtype->type_num == CNP_DOUBLE &&
            right->dtype->type_num == CNP_DOUBLE &&
            result->dtype->type_num == CNP_DOUBLE) {
        const double *left_doubles = (const double*)left_values;
        const double *right_doubles = (const double*)right_values;
        double *result_doubles = (double*)result_values;
        int64_t index = 0;
        for (; index + 3 < result->size; index += 4) {
            for (int lane = 0; lane < 4; ++lane) {
                double value = left_doubles[index + lane];
                if (isnan(value))
                    result_doubles[index + lane] = NAN;
                else if (value < 0.0)
                    result_doubles[index + lane] = 0.0;
                else if (value > 0.0)
                    result_doubles[index + lane] = 1.0;
                else
                    result_doubles[index + lane] =
                        right_doubles[index + lane];
            }
        }
        for (; index < result->size; ++index) {
            double value = left_doubles[index];
            if (isnan(value))
                result_doubles[index] = NAN;
            else if (value < 0.0)
                result_doubles[index] = 0.0;
            else if (value > 0.0)
                result_doubles[index] = 1.0;
            else
                result_doubles[index] = right_doubles[index];
        }
        *status = CNP_OK;
        return true;
    }

    int left_size = left->dtype->elsize;
    int right_size = right->dtype->elsize;
    int result_size = result->dtype->elsize;
    for (int64_t index = 0; index < result->size; ++index) {
        *status = heaviside_element(
            left_values + index * left_size,
            left->dtype->type_num,
            right_values + index * right_size,
            right->dtype->type_num,
            result_values + index * result_size,
            result->dtype->type_num,
            function_name);
        if (*status != CNP_OK) return true;
    }
    *status = CNP_OK;
    return true;
}

static CnpArray *heaviside_arrays(
    const CnpArray *left,
    const CnpArray *right,
    const char *function_name) {
    CNP_TYPE result_dtype = CNP_NOTYPE;
    CNP_STATUS status = heaviside_validate_inputs(
        left, right, &result_dtype, function_name);
    if (status != CNP_OK) return NULL;

    CnpArray *result = arithmetic_prepare_result(
        left, right, result_dtype, function_name);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (heaviside_contiguous_typed(
            left, right, result, function_name, &status)) {
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t left_offset = arithmetic_broadcast_offset(
            left, coordinates, result->ndim);
        int64_t right_offset = arithmetic_broadcast_offset(
            right, coordinates, result->ndim);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        status = heaviside_element(
            (const char*)left->data + left_offset,
            left->dtype->type_num,
            (const char*)right->data + right_offset,
            right->dtype->type_num,
            (char*)result->data + result_offset,
            result_dtype,
            function_name);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_heaviside(
    const CnpArray *x1, const CnpArray *x2) {
    return heaviside_arrays(x1, x2, "cnp_heaviside");
}

CNP_API CnpArray* CNP_CALL cnp_mod(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_mod";
    CNP_TYPE dt = CNP_NOTYPE;
    if (arithmetic_validate_inputs(
            CNP_ARITHMETIC_REMAINDER,
            a, b, function_name, "remainder", &dt) != CNP_OK)
        return NULL;
    return arithmetic_promoted_arrays(
        CNP_ARITHMETIC_REMAINDER,
        a, b, dt, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_remainder(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_remainder";
    CNP_TYPE dt = CNP_NOTYPE;
    if (arithmetic_validate_inputs(
            CNP_ARITHMETIC_REMAINDER,
            a, b, function_name, "remainder", &dt) != CNP_OK)
        return NULL;
    return arithmetic_promoted_arrays(
        CNP_ARITHMETIC_REMAINDER,
        a, b, dt, function_name);
}

static CNP_STATUS divmod_validate_slots_and_inputs(
    const CnpArray *left,
    const CnpArray *right,
    CnpArray **quotient,
    CnpArray **remainder,
    const char *function_name) {
    if (quotient) *quotient = NULL;
    if (remainder && remainder != quotient) *remainder = NULL;
    if (!quotient || !remainder) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "quotient and remainder output slots are required");
        return CNP_ERR_GENERIC;
    }
    if (quotient == remainder) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "quotient and remainder output slots must be distinct");
        return CNP_ERR_GENERIC;
    }
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays are required");
        return CNP_ERR_GENERIC;
    }
    if (!left->dtype || !right->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "left and right arrays must have dtypes");
        return CNP_ERR_TYPE;
    }
    if ((left->size > 0 && !left->data) ||
            (right->size > 0 && !right->data)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays require data buffers");
        return CNP_ERR_GENERIC;
    }
    return CNP_OK;
}

static bool divmod_contiguous_typed(
    const CnpArray *left,
    const CnpArray *right,
    CnpArray *quotient,
    CnpArray *remainder,
    const char *function_name,
    CNP_STATUS *status) {
    bool c_contiguous =
        (left->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (right->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (quotient->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (remainder->flags & CNP_ARRAY_C_CONTIGUOUS);
    bool f_contiguous =
        (left->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (right->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (quotient->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (remainder->flags & CNP_ARRAY_F_CONTIGUOUS);
    if ((!c_contiguous && !f_contiguous) ||
            !same_shape(left, right) ||
            !same_shape(left, quotient) ||
            !same_shape(quotient, remainder))
        return false;
    if (quotient->size == 0) {
        *status = CNP_OK;
        return true;
    }

    const uint8_t *left_values =
        (const uint8_t*)left->data + left->offset;
    const uint8_t *right_values =
        (const uint8_t*)right->data + right->offset;
    uint8_t *quotient_values =
        (uint8_t*)quotient->data + quotient->offset;
    uint8_t *remainder_values =
        (uint8_t*)remainder->data + remainder->offset;
    if (left->dtype->type_num == CNP_DOUBLE &&
            right->dtype->type_num == CNP_DOUBLE &&
            quotient->dtype->type_num == CNP_DOUBLE &&
            remainder->dtype->type_num == CNP_DOUBLE) {
        const double *left_doubles = (const double*)left_values;
        const double *right_doubles = (const double*)right_values;
        double *quotient_doubles = (double*)quotient_values;
        double *remainder_doubles = (double*)remainder_values;
        for (int64_t index = 0; index < quotient->size; ++index) {
            quotient_doubles[index] = arithmetic_divmod_double(
                left_doubles[index], right_doubles[index],
                &remainder_doubles[index]);
        }
        *status = CNP_OK;
        return true;
    }

    int left_itemsize = left->dtype->elsize;
    int right_itemsize = right->dtype->elsize;
    int result_itemsize = quotient->dtype->elsize;
    for (int64_t index = 0; index < quotient->size; ++index) {
        *status = arithmetic_promoted_divmod_element(
            left_values + index * left_itemsize,
            left->dtype->type_num,
            right_values + index * right_itemsize,
            right->dtype->type_num,
            quotient_values + index * result_itemsize,
            remainder_values + index * result_itemsize,
            quotient->dtype->type_num,
            function_name);
        if (*status != CNP_OK) return true;
    }
    *status = CNP_OK;
    return true;
}

static CNP_STATUS divmod_numeric_arrays(
    const CnpArray *left,
    const CnpArray *right,
    CnpArray **quotient,
    CnpArray **remainder,
    const char *function_name) {
    CNP_TYPE result_dtype = CNP_NOTYPE;
    CNP_STATUS status = arithmetic_validate_inputs(
        CNP_ARITHMETIC_FLOOR_DIVIDE,
        left, right, function_name, "divmod", &result_dtype);
    if (status != CNP_OK) return status;

    CnpArray *quotient_result = arithmetic_prepare_result(
        left, right, result_dtype, function_name);
    if (!quotient_result) {
        status = cnp_get_error(NULL);
        cnp_relabel_error(function_name);
        return status;
    }
    CnpArray *remainder_result = arithmetic_prepare_result(
        left, right, result_dtype, function_name);
    if (!remainder_result) {
        status = cnp_get_error(NULL);
        cnp_array_free(quotient_result);
        cnp_relabel_error(function_name);
        return status;
    }

    if (divmod_contiguous_typed(
            left, right, quotient_result, remainder_result,
            function_name, &status)) {
        if (status != CNP_OK) {
            cnp_array_free(quotient_result);
            cnp_array_free(remainder_result);
            return status;
        }
        *quotient = quotient_result;
        *remainder = remainder_result;
        return CNP_OK;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < quotient_result->size; ++index) {
        int64_t left_offset = arithmetic_broadcast_offset(
            left, coordinates, quotient_result->ndim);
        int64_t right_offset = arithmetic_broadcast_offset(
            right, coordinates, quotient_result->ndim);
        int64_t quotient_offset = quotient_result->offset +
            cnp_multi_to_offset(
                quotient_result->ndim,
                coordinates,
                quotient_result->strides);
        int64_t remainder_offset = remainder_result->offset +
            cnp_multi_to_offset(
                remainder_result->ndim,
                coordinates,
                remainder_result->strides);
        status = arithmetic_promoted_divmod_element(
            (const char*)left->data + left_offset,
            left->dtype->type_num,
            (const char*)right->data + right_offset,
            right->dtype->type_num,
            (char*)quotient_result->data + quotient_offset,
            (char*)remainder_result->data + remainder_offset,
            result_dtype,
            function_name);
        if (status != CNP_OK) {
            cnp_array_free(quotient_result);
            cnp_array_free(remainder_result);
            return status;
        }
        for (int dimension = quotient_result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] <
                    quotient_result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }

    *quotient = quotient_result;
    *remainder = remainder_result;
    return CNP_OK;
}

static CNP_STATUS divmod_timedelta_arrays(
    const CnpArray *left,
    const CnpArray *right,
    CnpArray **quotient,
    CnpArray **remainder,
    const char *function_name) {
    CnpArray *quotient_result = arithmetic_prepare_result(
        left, right, CNP_LONGLONG, function_name);
    if (!quotient_result) {
        cnp_relabel_error(function_name);
        return cnp_get_error(NULL);
    }
    CnpArray *remainder_result = arithmetic_prepare_result(
        left, right, CNP_TIMEDELTA, function_name);
    if (!remainder_result) {
        CNP_STATUS status = cnp_get_error(NULL);
        cnp_array_free(quotient_result);
        cnp_relabel_error(function_name);
        return status;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < quotient_result->size; ++index) {
        int64_t left_offset = arithmetic_broadcast_offset(
            left, coordinates, quotient_result->ndim);
        int64_t right_offset = arithmetic_broadcast_offset(
            right, coordinates, quotient_result->ndim);
        int64_t quotient_offset = quotient_result->offset +
            cnp_multi_to_offset(
                quotient_result->ndim,
                coordinates,
                quotient_result->strides);
        int64_t remainder_offset = remainder_result->offset +
            cnp_multi_to_offset(
                remainder_result->ndim,
                coordinates,
                remainder_result->strides);
        int64_t left_value;
        int64_t right_value;
        int64_t quotient_value;
        int64_t remainder_value;
        memcpy(
            &left_value,
            (const char*)left->data + left_offset,
            sizeof(left_value));
        memcpy(
            &right_value,
            (const char*)right->data + right_offset,
            sizeof(right_value));

        if (left_value == INT64_MIN ||
                right_value == INT64_MIN || right_value == 0) {
            quotient_value = 0;
            remainder_value = INT64_MIN;
        } else {
            uint64_t quotient_bits = arithmetic_signed_floor_divide_bits(
                left_value, right_value);
            uint64_t remainder_bits = arithmetic_signed_remainder_bits(
                left_value, right_value);
            memcpy(
                &quotient_value, &quotient_bits,
                sizeof(quotient_value));
            memcpy(
                &remainder_value, &remainder_bits,
                sizeof(remainder_value));
        }
        memcpy(
            (char*)quotient_result->data + quotient_offset,
            &quotient_value,
            sizeof(quotient_value));
        memcpy(
            (char*)remainder_result->data + remainder_offset,
            &remainder_value,
            sizeof(remainder_value));

        for (int dimension = quotient_result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] <
                    quotient_result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }

    *quotient = quotient_result;
    *remainder = remainder_result;
    return CNP_OK;
}

CNP_API CNP_STATUS CNP_CALL cnp_divmod(
    const CnpArray *left,
    const CnpArray *right,
    CnpArray **quotient,
    CnpArray **remainder) {
    const char *function_name = "cnp_divmod";
    CNP_STATUS status = divmod_validate_slots_and_inputs(
        left, right, quotient, remainder, function_name);
    if (status != CNP_OK) return status;

    if (left->dtype->type_num == CNP_TIMEDELTA &&
            right->dtype->type_num == CNP_TIMEDELTA)
        return divmod_timedelta_arrays(
            left, right, quotient, remainder, function_name);
    return divmod_numeric_arrays(
        left, right, quotient, remainder, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_fmod(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_fmod";
    CNP_TYPE dt = CNP_NOTYPE;
    if (arithmetic_validate_inputs(
            CNP_ARITHMETIC_FMOD,
            a, b, function_name, "fmod", &dt) != CNP_OK)
        return NULL;
    return arithmetic_promoted_arrays(
        CNP_ARITHMETIC_FMOD,
        a, b, dt, function_name);
}

/* Unary arithmetic */
typedef enum {
    CNP_UNARY_SIGN_NEGATIVE,
    CNP_UNARY_SIGN_POSITIVE
} CnpUnarySignOperation;

static bool unary_numeric_timedelta_dtype_supported(CNP_TYPE dtype) {
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
        case CNP_FLOAT:
        case CNP_DOUBLE:
        case CNP_LONGDOUBLE:
        case CNP_CFLOAT:
        case CNP_CDOUBLE:
        case CNP_CLONGDOUBLE:
        case CNP_TIMEDELTA:
        case CNP_HALF:
            return true;
        default:
            return false;
    }
}

static void unary_sign_element(
    CnpUnarySignOperation operation,
    CNP_TYPE dtype,
    const void *source,
    void *destination) {
    int itemsize = cnp_dtype_itemsize(dtype);
    if (operation == CNP_UNARY_SIGN_POSITIVE) {
        memcpy(destination, source, (size_t)itemsize);
        return;
    }

    switch (dtype) {
        case CNP_BYTE:
        case CNP_UBYTE: {
            uint8_t bits;
            memcpy(&bits, source, sizeof(bits));
            bits = (uint8_t)(0u - bits);
            memcpy(destination, &bits, sizeof(bits));
            return;
        }
        case CNP_SHORT:
        case CNP_USHORT: {
            uint16_t bits;
            memcpy(&bits, source, sizeof(bits));
            bits = (uint16_t)(0u - bits);
            memcpy(destination, &bits, sizeof(bits));
            return;
        }
        case CNP_INT:
        case CNP_UINT: {
            uint32_t bits;
            memcpy(&bits, source, sizeof(bits));
            bits = UINT32_C(0) - bits;
            memcpy(destination, &bits, sizeof(bits));
            return;
        }
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
        case CNP_TIMEDELTA: {
            uint64_t bits;
            memcpy(&bits, source, sizeof(bits));
            bits = UINT64_C(0) - bits;
            memcpy(destination, &bits, sizeof(bits));
            return;
        }
        case CNP_HALF: {
            uint16_t bits;
            memcpy(&bits, source, sizeof(bits));
            bits ^= UINT16_C(0x8000);
            memcpy(destination, &bits, sizeof(bits));
            return;
        }
        case CNP_FLOAT: {
            uint32_t bits;
            memcpy(&bits, source, sizeof(bits));
            bits ^= UINT32_C(0x80000000);
            memcpy(destination, &bits, sizeof(bits));
            return;
        }
        case CNP_DOUBLE: {
            uint64_t bits;
            memcpy(&bits, source, sizeof(bits));
            bits ^= UINT64_C(0x8000000000000000);
            memcpy(destination, &bits, sizeof(bits));
            return;
        }
        case CNP_LONGDOUBLE: {
            long double value;
            memcpy(&value, source, sizeof(value));
            value = -value;
            memcpy(destination, &value, sizeof(value));
            return;
        }
        case CNP_CFLOAT: {
            uint32_t components[2];
            memcpy(components, source, sizeof(components));
            components[0] ^= UINT32_C(0x80000000);
            components[1] ^= UINT32_C(0x80000000);
            memcpy(destination, components, sizeof(components));
            return;
        }
        case CNP_CDOUBLE: {
            uint64_t components[2];
            memcpy(components, source, sizeof(components));
            components[0] ^= UINT64_C(0x8000000000000000);
            components[1] ^= UINT64_C(0x8000000000000000);
            memcpy(destination, components, sizeof(components));
            return;
        }
        case CNP_CLONGDOUBLE: {
            cnp_clongdouble value;
            memcpy(&value, source, sizeof(value));
            value.real = -value.real;
            value.imag = -value.imag;
            memcpy(destination, &value, sizeof(value));
            return;
        }
        default:
            return;
    }
}

static uint64_t unary_stride_magnitude(int64_t stride) {
    return stride < 0
        ? UINT64_C(0) - (uint64_t)stride
        : (uint64_t)stride;
}

static void unary_set_keep_order_layout(
    const CnpArray *source, CnpArray *result) {
    if (source->ndim <= 1 || source->size == 0 ||
            (source->flags & (
                CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS))) {
        return;
    }

    int axes[CNP_MAXDIMS];
    for (int dimension = 0; dimension < source->ndim; ++dimension) {
        axes[dimension] = source->ndim - dimension - 1;
    }
    for (int position = 1; position < source->ndim; ++position) {
        int axis = axes[position];
        uint64_t magnitude = unary_stride_magnitude(
            source->strides[axis]);
        int insertion = position;
        for (int previous = position - 1; previous >= 0; --previous) {
            int previous_axis = axes[previous];
            uint64_t previous_magnitude = unary_stride_magnitude(
                source->strides[previous_axis]);
            if (magnitude == 0 || previous_magnitude == 0 ||
                    magnitude == previous_magnitude) continue;
            if (magnitude > previous_magnitude) break;
            insertion = previous;
        }
        for (int shifted = position; shifted > insertion; --shifted) {
            axes[shifted] = axes[shifted - 1];
        }
        axes[insertion] = axis;
    }

    int64_t stride = result->dtype->elsize;
    for (int position = 0; position < source->ndim; ++position) {
        int axis = axes[position];
        result->strides[axis] = stride;
        stride *= result->shape[axis];
    }
    result->flags &= ~(
        CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS);
    result->flags |= cnp_compute_layout_flags(
        result->ndim, result->shape, result->strides,
        result->dtype->elsize);
}

static CnpArray* unary_sign_arrays(
    const CnpArray *source,
    CnpUnarySignOperation operation,
    const char *function_name,
    const char *operation_name) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }
    CNP_TYPE dtype = source->dtype->type_num;
    if (dtype == CNP_BOOL ||
            !unary_numeric_timedelta_dtype_supported(dtype)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support %s",
            source->dtype->name, operation_name);
        return NULL;
    }

    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);

    if (operation == CNP_UNARY_SIGN_NEGATIVE &&
            (source->flags & (
                CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) &&
            dtype == CNP_DOUBLE) {
        cnp_simd_negative(
            (const double*)((const char*)source->data + source->offset),
            (double*)result->data, source->size);
        return result;
    }
    if (operation == CNP_UNARY_SIGN_POSITIVE &&
            (source->flags & (
                CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS))) {
        memcpy(
            result->data,
            (const char*)source->data + source->offset,
            (size_t)(source->size * source->dtype->elsize));
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        unary_sign_element(
            operation,
            dtype,
            (const char*)source->data + source_offset,
            (char*)result->data + result_offset);
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

static uint16_t unary_signum_half_bits(uint16_t bits) {
    uint16_t magnitude = bits & UINT16_C(0x7fff);
    if (magnitude == 0) return 0;
    if (magnitude > UINT16_C(0x7c00)) return bits;
    return (bits & UINT16_C(0x8000)) | UINT16_C(0x3c00);
}

static uint32_t unary_signum_float_bits(uint32_t bits) {
    uint32_t magnitude = bits & UINT32_C(0x7fffffff);
    if (magnitude == 0) return 0;
    if (magnitude > UINT32_C(0x7f800000)) return bits;
    return (bits & UINT32_C(0x80000000)) | UINT32_C(0x3f800000);
}

static uint64_t unary_signum_double_bits(uint64_t bits) {
    uint64_t magnitude = bits & UINT64_C(0x7fffffffffffffff);
    if (magnitude == 0) return 0;
    if (magnitude > UINT64_C(0x7ff0000000000000)) return bits;
    return (bits & UINT64_C(0x8000000000000000)) |
        UINT64_C(0x3ff0000000000000);
}

static long double unary_signum_longdouble_value(long double value) {
    if (isnan(value)) return value;
    if (value > 0.0L) return 1.0L;
    if (value < 0.0L) return -1.0L;
    return 0.0L;
}

static void unary_signum_element(
    CNP_TYPE dtype, const void *source, void *destination) {
    switch (dtype) {
        case CNP_BYTE: {
            int8_t value;
            int8_t result;
            memcpy(&value, source, sizeof(value));
            result = (int8_t)((value > 0) - (value < 0));
            memcpy(destination, &result, sizeof(result));
            return;
        }
        case CNP_UBYTE: {
            uint8_t value;
            uint8_t result;
            memcpy(&value, source, sizeof(value));
            result = value == 0 ? 0 : 1;
            memcpy(destination, &result, sizeof(result));
            return;
        }
        case CNP_SHORT: {
            int16_t value;
            int16_t result;
            memcpy(&value, source, sizeof(value));
            result = (int16_t)((value > 0) - (value < 0));
            memcpy(destination, &result, sizeof(result));
            return;
        }
        case CNP_USHORT: {
            uint16_t value;
            uint16_t result;
            memcpy(&value, source, sizeof(value));
            result = value == 0 ? 0 : 1;
            memcpy(destination, &result, sizeof(result));
            return;
        }
        case CNP_INT: {
            int32_t value;
            int32_t result;
            memcpy(&value, source, sizeof(value));
            result = (value > 0) - (value < 0);
            memcpy(destination, &result, sizeof(result));
            return;
        }
        case CNP_UINT: {
            uint32_t value;
            uint32_t result;
            memcpy(&value, source, sizeof(value));
            result = value == 0 ? 0 : 1;
            memcpy(destination, &result, sizeof(result));
            return;
        }
        case CNP_LONG:
        case CNP_LONGLONG:
        case CNP_TIMEDELTA: {
            int64_t value;
            int64_t result;
            memcpy(&value, source, sizeof(value));
            result = (value > 0) - (value < 0);
            memcpy(destination, &result, sizeof(result));
            return;
        }
        case CNP_ULONG:
        case CNP_ULONGLONG: {
            uint64_t value;
            uint64_t result;
            memcpy(&value, source, sizeof(value));
            result = value == 0 ? 0 : 1;
            memcpy(destination, &result, sizeof(result));
            return;
        }
        case CNP_HALF: {
            uint16_t bits;
            memcpy(&bits, source, sizeof(bits));
            bits = unary_signum_half_bits(bits);
            memcpy(destination, &bits, sizeof(bits));
            return;
        }
        case CNP_FLOAT: {
            uint32_t bits;
            memcpy(&bits, source, sizeof(bits));
            bits = unary_signum_float_bits(bits);
            memcpy(destination, &bits, sizeof(bits));
            return;
        }
        case CNP_DOUBLE: {
            uint64_t bits;
            memcpy(&bits, source, sizeof(bits));
            bits = unary_signum_double_bits(bits);
            memcpy(destination, &bits, sizeof(bits));
            return;
        }
        case CNP_LONGDOUBLE: {
            long double value;
            memcpy(&value, source, sizeof(value));
            value = unary_signum_longdouble_value(value);
            memcpy(destination, &value, sizeof(value));
            return;
        }
        case CNP_CFLOAT: {
            uint32_t source_components[2];
            uint32_t result_components[2] = {0, 0};
            memcpy(source_components, source, sizeof(source_components));
            uint32_t real_magnitude =
                source_components[0] & UINT32_C(0x7fffffff);
            uint32_t imaginary_magnitude =
                source_components[1] & UINT32_C(0x7fffffff);
            if (real_magnitude > UINT32_C(0x7f800000) ||
                    imaginary_magnitude > UINT32_C(0x7f800000)) {
                result_components[0] = UINT32_C(0x7fc00000);
            } else {
                uint32_t selected = real_magnitude != 0
                    ? source_components[0] : source_components[1];
                result_components[0] = unary_signum_float_bits(selected);
            }
            memcpy(destination, result_components, sizeof(result_components));
            return;
        }
        case CNP_CDOUBLE: {
            uint64_t source_components[2];
            uint64_t result_components[2] = {0, 0};
            memcpy(source_components, source, sizeof(source_components));
            uint64_t real_magnitude =
                source_components[0] & UINT64_C(0x7fffffffffffffff);
            uint64_t imaginary_magnitude =
                source_components[1] & UINT64_C(0x7fffffffffffffff);
            if (real_magnitude > UINT64_C(0x7ff0000000000000) ||
                    imaginary_magnitude >
                        UINT64_C(0x7ff0000000000000)) {
                result_components[0] = UINT64_C(0x7ff8000000000000);
            } else {
                uint64_t selected = real_magnitude != 0
                    ? source_components[0] : source_components[1];
                result_components[0] = unary_signum_double_bits(selected);
            }
            memcpy(destination, result_components, sizeof(result_components));
            return;
        }
        case CNP_CLONGDOUBLE: {
            cnp_clongdouble source_value;
            cnp_clongdouble result = {0.0L, 0.0L};
            memcpy(&source_value, source, sizeof(source_value));
            if (isnan(source_value.real) || isnan(source_value.imag)) {
                result.real = (long double)NAN;
            } else {
                long double selected = source_value.real != 0.0L
                    ? source_value.real : source_value.imag;
                result.real = unary_signum_longdouble_value(selected);
            }
            memcpy(destination, &result, sizeof(result));
            return;
        }
        default:
            return;
    }
}

static CnpArray* unary_signum_arrays(const CnpArray *source) {
    const char *function_name = "cnp_sign";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }
    CNP_TYPE dtype = source->dtype->type_num;
    if (dtype == CNP_BOOL ||
            !unary_numeric_timedelta_dtype_supported(dtype)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support sign",
            source->dtype->name);
        return NULL;
    }

    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);

    int itemsize = source->dtype->elsize;
    if (source->flags & (
            CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) {
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        for (int64_t index = 0; index < source->size; ++index) {
            unary_signum_element(
                dtype,
                source_data + index * itemsize,
                result_data + index * itemsize);
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        unary_signum_element(
            dtype,
            (const char*)source->data + source_offset,
            (char*)result->data + result_offset);
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static cnp_cfloat unary_reciprocal_cfloat(cnp_cfloat source) {
    if (isnan(source.real) || isnan(source.imag)) {
        return arithmetic_divide_cfloat(
            1.0f, 0.0f, source.real, source.imag);
    }
    float absolute_real = fabsf(source.real);
    float absolute_imaginary = fabsf(source.imag);
    cnp_cfloat result;
    if (absolute_real >= absolute_imaginary) {
        volatile float ratio = source.imag / source.real;
        volatile float denominator =
            source.real + source.imag * ratio;
        volatile float scale = 1.0f / denominator;
        result.real = scale;
        result.imag = -ratio * scale;
    } else {
        volatile float ratio = source.real / source.imag;
        volatile float denominator =
            source.real * ratio + source.imag;
        volatile float scale = 1.0f / denominator;
        result.real = ratio * scale;
        result.imag = -scale;
    }
    return result;
}

static cnp_cdouble unary_reciprocal_cdouble(cnp_cdouble source) {
    if (isnan(source.real) || isnan(source.imag)) {
        return arithmetic_divide_cdouble(
            1.0, 0.0, source.real, source.imag);
    }
    double absolute_real = fabs(source.real);
    double absolute_imaginary = fabs(source.imag);
    cnp_cdouble result;
    if (absolute_real >= absolute_imaginary) {
        volatile double ratio = source.imag / source.real;
        volatile double denominator =
            source.real + source.imag * ratio;
        volatile double scale = 1.0 / denominator;
        result.real = scale;
        result.imag = -ratio * scale;
    } else {
        volatile double ratio = source.real / source.imag;
        volatile double denominator =
            source.real * ratio + source.imag;
        volatile double scale = 1.0 / denominator;
        result.real = ratio * scale;
        result.imag = -scale;
    }
    return result;
}

static cnp_clongdouble unary_reciprocal_clongdouble(
    cnp_clongdouble source) {
    if (isnan(source.real) || isnan(source.imag)) {
        return arithmetic_divide_clongdouble(
            1.0L, 0.0L, source.real, source.imag);
    }
    long double absolute_real = fabsl(source.real);
    long double absolute_imaginary = fabsl(source.imag);
    cnp_clongdouble result;
    if (absolute_real >= absolute_imaginary) {
        volatile long double ratio = source.imag / source.real;
        volatile long double denominator =
            source.real + source.imag * ratio;
        volatile long double scale = 1.0L / denominator;
        result.real = scale;
        result.imag = -ratio * scale;
    } else {
        volatile long double ratio = source.real / source.imag;
        volatile long double denominator =
            source.real * ratio + source.imag;
        volatile long double scale = 1.0L / denominator;
        result.real = ratio * scale;
        result.imag = -scale;
    }
    return result;
}

static CNP_STATUS unary_reciprocal_element(
    CNP_TYPE dtype, const void *source, void *destination) {
    switch (dtype) {
        case CNP_BOOL: {
            uint8_t value;
            int8_t result;
            memcpy(&value, source, sizeof(value));
            result = value == 0 ? 0 : 1;
            memcpy(destination, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_BYTE: {
            int8_t value;
            int8_t result;
            memcpy(&value, source, sizeof(value));
            result = (int8_t)((value == 1) - (value == -1));
            memcpy(destination, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_UBYTE: {
            uint8_t value;
            uint8_t result;
            memcpy(&value, source, sizeof(value));
            result = value == 1 ? 1 : 0;
            memcpy(destination, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_SHORT: {
            int16_t value;
            int16_t result;
            memcpy(&value, source, sizeof(value));
            result = (int16_t)((value == 1) - (value == -1));
            memcpy(destination, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_USHORT: {
            uint16_t value;
            uint16_t result;
            memcpy(&value, source, sizeof(value));
            result = value == 1 ? 1 : 0;
            memcpy(destination, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_INT: {
            int32_t value;
            int32_t result;
            memcpy(&value, source, sizeof(value));
            result = value == 0
                ? INT32_MIN
                : (value == 1) - (value == -1);
            memcpy(destination, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_UINT: {
            uint32_t value;
            uint32_t result;
            memcpy(&value, source, sizeof(value));
            result = value == 1 ? UINT32_C(1) : UINT32_C(0);
            memcpy(destination, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_LONG:
        case CNP_LONGLONG: {
            int64_t value;
            int64_t result;
            memcpy(&value, source, sizeof(value));
            result = value == 0
                ? INT64_MIN
                : (int64_t)((value == 1) - (value == -1));
            memcpy(destination, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_ULONG:
        case CNP_ULONGLONG: {
            uint64_t value;
            uint64_t result;
            memcpy(&value, source, sizeof(value));
            result = value == 0
                ? UINT64_C(0x8000000000000000)
                : value == 1 ? UINT64_C(1) : UINT64_C(0);
            memcpy(destination, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_HALF: {
            uint16_t bits;
            memcpy(&bits, source, sizeof(bits));
            float value = (float)cnp_half_to_float(bits);
            volatile float result = 1.0f / value;
            bits = cnp_float_to_half((double)result);
            memcpy(destination, &bits, sizeof(bits));
            return CNP_OK;
        }
        case CNP_FLOAT: {
            float value;
            volatile float result;
            memcpy(&value, source, sizeof(value));
            result = 1.0f / value;
            memcpy(destination, (const void*)&result, sizeof(result));
            return CNP_OK;
        }
        case CNP_DOUBLE: {
            double value;
            volatile double result;
            memcpy(&value, source, sizeof(value));
            result = 1.0 / value;
            memcpy(destination, (const void*)&result, sizeof(result));
            return CNP_OK;
        }
        case CNP_LONGDOUBLE: {
            long double value;
            volatile long double result;
            memcpy(&value, source, sizeof(value));
            result = 1.0L / value;
            memcpy(destination, (const void*)&result, sizeof(result));
            return CNP_OK;
        }
        case CNP_CFLOAT: {
            cnp_cfloat value;
            cnp_cfloat result;
            memcpy(&value, source, sizeof(value));
            result = unary_reciprocal_cfloat(value);
            memcpy(destination, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_CDOUBLE: {
            cnp_cdouble value;
            cnp_cdouble result;
            memcpy(&value, source, sizeof(value));
            result = unary_reciprocal_cdouble(value);
            memcpy(destination, &result, sizeof(result));
            return CNP_OK;
        }
        case CNP_CLONGDOUBLE: {
            cnp_clongdouble value;
            cnp_clongdouble result;
            memcpy(&value, source, sizeof(value));
            result = unary_reciprocal_clongdouble(value);
            memcpy(destination, &result, sizeof(result));
            return CNP_OK;
        }
        default:
            cnp_set_error(
                CNP_ERR_TYPE, "cnp_reciprocal",
                "source dtype %d does not support reciprocal", (int)dtype);
            return CNP_ERR_TYPE;
    }
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static CnpArray* unary_reciprocal_arrays(const CnpArray *source) {
    const char *function_name = "cnp_reciprocal";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }

    CNP_TYPE dtype = source->dtype->type_num;
    bool supported = dtype == CNP_BOOL || cnp_type_is_integer(dtype) ||
        cnp_type_is_float(dtype) || cnp_type_is_complex(dtype);
    if (!supported) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support reciprocal",
            source->dtype->name);
        return NULL;
    }

    CNP_TYPE result_dtype = dtype == CNP_BOOL ? CNP_BYTE : dtype;
    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);

    int source_itemsize = source->dtype->elsize;
    int result_itemsize = result->dtype->elsize;
    if (source->flags & (
            CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) {
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        for (int64_t index = 0; index < source->size; ++index) {
            CNP_STATUS status = unary_reciprocal_element(
                dtype,
                source_data + index * source_itemsize,
                result_data + index * result_itemsize);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                cnp_relabel_error(function_name);
                return NULL;
            }
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = unary_reciprocal_element(
            dtype,
            (const char*)source->data + source_offset,
            (char*)result->data + result_offset);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

static CNP_STATUS unary_conjugate_element(
    CNP_TYPE source_dtype,
    const void *source,
    void *destination,
    const char *function_name) {
    if (source_dtype == CNP_BOOL) {
        uint8_t value;
        int8_t result;
        memcpy(&value, source, sizeof(value));
        result = value != 0 ? 1 : 0;
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (source_dtype == CNP_CFLOAT) {
        uint32_t components[2];
        memcpy(components, source, sizeof(components));
        components[1] ^= UINT32_C(0x80000000);
        memcpy(destination, components, sizeof(components));
        return CNP_OK;
    }
    if (source_dtype == CNP_CDOUBLE) {
        uint64_t components[2];
        memcpy(components, source, sizeof(components));
        components[1] ^= UINT64_C(0x8000000000000000);
        memcpy(destination, components, sizeof(components));
        return CNP_OK;
    }
    if (source_dtype == CNP_CLONGDOUBLE) {
        cnp_clongdouble value;
        memcpy(&value, source, sizeof(value));
        long double opposite_sign = -copysignl(1.0L, value.imag);
        value.imag = copysignl(value.imag, opposite_sign);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "source dtype %d does not support the typed conjugate element loop",
        (int)source_dtype);
    return CNP_ERR_TYPE;
}

static CnpArray* unary_conjugate_arrays(
    const CnpArray *source,
    const char *function_name) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }

    CNP_TYPE source_dtype = source->dtype->type_num;
    bool supported = source_dtype == CNP_BOOL ||
        cnp_type_is_integer(source_dtype) ||
        cnp_type_is_float(source_dtype) ||
        cnp_type_is_complex(source_dtype);
    if (!supported) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support conjugate",
            source->dtype->name);
        return NULL;
    }

    CNP_TYPE result_dtype = source_dtype == CNP_BOOL
        ? CNP_BYTE : source_dtype;
    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);
    if (source->size == 0) return result;

    int source_itemsize = source->dtype->elsize;
    int result_itemsize = result->dtype->elsize;
    if (source->flags & (
            CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) {
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        if (source_dtype != CNP_BOOL &&
                !cnp_type_is_complex(source_dtype)) {
            memcpy(result_data, source_data,
                (size_t)source->size * (size_t)source_itemsize);
            return result;
        }
        for (int64_t index = 0; index < source->size; ++index) {
            CNP_STATUS status = unary_conjugate_element(
                source_dtype,
                source_data + index * source_itemsize,
                result_data + index * result_itemsize,
                function_name);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                cnp_relabel_error(function_name);
                return NULL;
            }
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        if (source_dtype == CNP_BOOL ||
                cnp_type_is_complex(source_dtype)) {
            CNP_STATUS status = unary_conjugate_element(
                source_dtype,
                (const char*)source->data + source_offset,
                (char*)result->data + result_offset,
                function_name);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                cnp_relabel_error(function_name);
                return NULL;
            }
        } else {
            memcpy(
                (char*)result->data + result_offset,
                (const char*)source->data + source_offset,
                (size_t)source_itemsize);
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

static CNP_STATUS unary_square_element(
    CNP_TYPE dtype, const void *source, void *destination) {
    if (dtype == CNP_BOOL) {
        uint8_t value;
        int8_t result;
        memcpy(&value, source, sizeof(value));
        result = value == 0 ? 0 : 1;
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
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
            return arithmetic_promoted_element(
                CNP_ARITHMETIC_MULTIPLY,
                source, dtype, source, dtype,
                destination, dtype, "cnp_square");
        default:
            cnp_set_error(
                CNP_ERR_TYPE, "cnp_square",
                "source dtype %d does not support square", (int)dtype);
            return CNP_ERR_TYPE;
    }
}

static CnpArray* unary_square_arrays(const CnpArray *source) {
    const char *function_name = "cnp_square";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }

    CNP_TYPE dtype = source->dtype->type_num;
    bool supported = dtype == CNP_BOOL || cnp_type_is_integer(dtype) ||
        cnp_type_is_float(dtype) || cnp_type_is_complex(dtype);
    if (!supported) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support square",
            source->dtype->name);
        return NULL;
    }

    CNP_TYPE result_dtype = dtype == CNP_BOOL ? CNP_BYTE : dtype;
    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);

    if (dtype == CNP_DOUBLE && source->size > 0 &&
            (source->flags & (
                CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS))) {
        const double *source_data = (const double*)(
            (const char*)source->data + source->offset);
        cnp_simd_multiply(
            source_data, source_data,
            (double*)((char*)result->data + result->offset),
            source->size);
        return result;
    }

    int source_itemsize = source->dtype->elsize;
    int result_itemsize = result->dtype->elsize;
    if (source->flags & (
            CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) {
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        for (int64_t index = 0; index < source->size; ++index) {
            CNP_STATUS status = unary_square_element(
                dtype,
                source_data + index * source_itemsize,
                result_data + index * result_itemsize);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                cnp_relabel_error(function_name);
                return NULL;
            }
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = unary_square_element(
            dtype,
            (const char*)source->data + source_offset,
            (char*)result->data + result_offset);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

static CNP_TYPE unary_sqrt_result_dtype(CNP_TYPE source_dtype) {
    switch (source_dtype) {
        case CNP_BOOL:
        case CNP_BYTE:
        case CNP_UBYTE:
        case CNP_HALF:
            return CNP_HALF;
        case CNP_SHORT:
        case CNP_USHORT:
        case CNP_FLOAT:
            return CNP_FLOAT;
        case CNP_INT:
        case CNP_UINT:
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
        case CNP_DOUBLE:
            return CNP_DOUBLE;
        case CNP_LONGDOUBLE:
            return CNP_LONGDOUBLE;
        case CNP_CFLOAT:
            return CNP_CFLOAT;
        case CNP_CDOUBLE:
            return CNP_CDOUBLE;
        case CNP_CLONGDOUBLE:
            return CNP_CLONGDOUBLE;
        default:
            return CNP_NOTYPE;
    }
}

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static float unary_sqrt_hypotf(float real, float imaginary) {
#ifdef _MSC_VER
    float (__cdecl *volatile function)(float, float) = _hypotf;
    return function(real, imaginary);
#else
    return hypotf(real, imaginary);
#endif
}

static double unary_sqrt_hypot(double real, double imaginary) {
#ifdef _MSC_VER
    double (__cdecl *volatile function)(double, double) = _hypot;
    return function(real, imaginary);
#else
    return hypot(real, imaginary);
#endif
}

static long double unary_sqrt_hypotl(
    long double real, long double imaginary) {
#ifdef _MSC_VER
    return (long double)unary_sqrt_hypot(
        (double)real, (double)imaginary);
#else
    return hypotl(real, imaginary);
#endif
}

static cnp_cfloat unary_sqrt_cfloat(cnp_cfloat source) {
    cnp_cfloat result;
    uint32_t source_real_bits;
    memcpy(&source_real_bits, &source.real, sizeof(source_real_bits));
    if (isinf(source.imag)) {
        result.real = INFINITY;
        result.imag = source.imag;
        return result;
    }
    if (isnan(source.real)) {
        volatile float difference = source.imag - source.imag;
        memcpy(&result.real, &source_real_bits, sizeof(source_real_bits));
        result.imag = difference / difference;
        return result;
    }
    if (isinf(source.real)) {
        if (source.real > 0.0f) {
            volatile float difference = source.imag - source.imag;
            result.real = INFINITY;
            result.imag = copysignf(difference, source.imag);
        } else {
            volatile float difference = source.imag - source.imag;
            result.real = fabsf(difference);
            result.imag = copysignf(INFINITY, source.imag);
        }
        return result;
    }
    if (source.real == 0.0f && source.imag == 0.0f) {
        result.real = 0.0f;
        result.imag = source.imag;
        return result;
    }
    float real = source.real;
    float imaginary = source.imag;
    float scale = 1.0f;
    const float threshold = FLT_MAX / (1.0f + sqrtf(2.0f));
    if (fabsf(real) >= threshold || fabsf(imaginary) >= threshold) {
        real *= 0.25f;
        imaginary *= 0.25f;
        scale = 2.0f;
    }
    volatile float magnitude = unary_sqrt_hypotf(real, imaginary);
    if (real >= 0.0f) {
        volatile float root = sqrtf(
            (magnitude + real) * 0.5f);
        result.real = root * scale;
        result.imag = imaginary / (2.0f * root);
    } else {
        volatile float root = sqrtf(
            (magnitude - real) * 0.5f);
        result.real =
            (fabsf(imaginary) / (2.0f * root)) * scale;
        result.imag = copysignf(root, imaginary);
    }
    return result;
}

static cnp_cdouble unary_sqrt_cdouble(cnp_cdouble source) {
    cnp_cdouble result;
    uint64_t source_real_bits;
    memcpy(&source_real_bits, &source.real, sizeof(source_real_bits));
    if (isinf(source.imag)) {
        result.real = INFINITY;
        result.imag = source.imag;
        return result;
    }
    if (isnan(source.real)) {
        volatile double difference = source.imag - source.imag;
        memcpy(&result.real, &source_real_bits, sizeof(source_real_bits));
        result.imag = difference / difference;
        return result;
    }
    if (isinf(source.real)) {
        if (source.real > 0.0) {
            volatile double difference = source.imag - source.imag;
            result.real = INFINITY;
            result.imag = copysign(difference, source.imag);
        } else {
            volatile double difference = source.imag - source.imag;
            result.real = fabs(difference);
            result.imag = copysign(INFINITY, source.imag);
        }
        return result;
    }
    if (source.real == 0.0 && source.imag == 0.0) {
        result.real = 0.0;
        result.imag = source.imag;
        return result;
    }
    double real = source.real;
    double imaginary = source.imag;
    double scale = 1.0;
    const double threshold = DBL_MAX / (1.0 + sqrt(2.0));
    if (fabs(real) >= threshold || fabs(imaginary) >= threshold) {
        real *= 0.25;
        imaginary *= 0.25;
        scale = 2.0;
    }
    volatile double magnitude = unary_sqrt_hypot(real, imaginary);
    if (real >= 0.0) {
        volatile double root = sqrt(
            (magnitude + real) * 0.5);
        result.real = root * scale;
        result.imag = imaginary / (2.0 * root);
    } else {
        volatile double root = sqrt(
            (magnitude - real) * 0.5);
        result.real = (fabs(imaginary) / (2.0 * root)) * scale;
        result.imag = copysign(root, imaginary);
    }
    return result;
}

static cnp_clongdouble unary_sqrt_clongdouble(
    cnp_clongdouble source) {
    cnp_clongdouble result;
    unsigned char source_real_bits[sizeof(long double)];
    memcpy(source_real_bits, &source.real, sizeof(source_real_bits));
    if (isinf(source.imag)) {
        result.real = INFINITY;
        result.imag = source.imag;
        return result;
    }
    if (isnan(source.real)) {
        volatile long double difference = source.imag - source.imag;
        memcpy(&result.real, source_real_bits, sizeof(source_real_bits));
        result.imag = difference / difference;
        return result;
    }
    if (isinf(source.real)) {
        if (source.real > 0.0L) {
            volatile long double difference = source.imag - source.imag;
            result.real = INFINITY;
            result.imag = copysignl(difference, source.imag);
        } else {
            volatile long double difference = source.imag - source.imag;
            result.real = fabsl(difference);
            result.imag = copysignl(INFINITY, source.imag);
        }
        return result;
    }
    if (source.real == 0.0L && source.imag == 0.0L) {
        result.real = 0.0L;
        result.imag = source.imag;
        return result;
    }
    long double real = source.real;
    long double imaginary = source.imag;
    long double scale = 1.0L;
    const long double threshold =
        LDBL_MAX / (1.0L + sqrtl(2.0L));
    if (fabsl(real) >= threshold || fabsl(imaginary) >= threshold) {
        real *= 0.25L;
        imaginary *= 0.25L;
        scale = 2.0L;
    }
    volatile long double magnitude = unary_sqrt_hypotl(real, imaginary);
    if (real >= 0.0L) {
        volatile long double root = sqrtl(
            (magnitude + real) * 0.5L);
        result.real = root * scale;
        result.imag = imaginary / (2.0L * root);
    } else {
        volatile long double root = sqrtl(
            (magnitude - real) * 0.5L);
        result.real =
            (fabsl(imaginary) / (2.0L * root)) * scale;
        result.imag = copysignl(root, imaginary);
    }
    return result;
}

static CNP_STATUS unary_sqrt_element(
    CNP_TYPE source_dtype,
    CNP_TYPE result_dtype,
    const void *source,
    void *destination) {
    if (source_dtype == CNP_CFLOAT) {
        cnp_cfloat value;
        cnp_cfloat result;
        uint32_t real_bits;
        uint32_t imaginary_bits;
        memcpy(&real_bits, source, sizeof(real_bits));
        memcpy(
            &imaginary_bits,
            (const char*)source + sizeof(real_bits),
            sizeof(imaginary_bits));
        memcpy(&value, source, sizeof(value));
        result = unary_sqrt_cfloat(value);
        memcpy(destination, &result, sizeof(result));
        if ((real_bits & UINT32_C(0x7f800000)) ==
                UINT32_C(0x7f800000) &&
                (real_bits & UINT32_C(0x007fffff)) != 0 &&
                (imaginary_bits & UINT32_C(0x7fffffff)) !=
                UINT32_C(0x7f800000)) {
            memcpy(destination, &real_bits, sizeof(real_bits));
        }
        return CNP_OK;
    }
    if (source_dtype == CNP_CDOUBLE) {
        cnp_cdouble value;
        cnp_cdouble result;
        uint64_t real_bits;
        uint64_t imaginary_bits;
        memcpy(&real_bits, source, sizeof(real_bits));
        memcpy(
            &imaginary_bits,
            (const char*)source + sizeof(real_bits),
            sizeof(imaginary_bits));
        memcpy(&value, source, sizeof(value));
        result = unary_sqrt_cdouble(value);
        memcpy(destination, &result, sizeof(result));
        if ((real_bits & UINT64_C(0x7ff0000000000000)) ==
                UINT64_C(0x7ff0000000000000) &&
                (real_bits & UINT64_C(0x000fffffffffffff)) != 0 &&
                (imaginary_bits & UINT64_C(0x7fffffffffffffff)) !=
                UINT64_C(0x7ff0000000000000)) {
            memcpy(destination, &real_bits, sizeof(real_bits));
        }
        return CNP_OK;
    }
    if (source_dtype == CNP_CLONGDOUBLE) {
        cnp_clongdouble value;
        cnp_clongdouble result;
        memcpy(&value, source, sizeof(value));
        result = unary_sqrt_clongdouble(value);
        memcpy(destination, &result, sizeof(result));
        if (!isinf(value.imag) && isnan(value.real)) {
            memcpy(destination, source, sizeof(value.real));
        }
        return CNP_OK;
    }

    CnpArithmeticValue value;
    CNP_STATUS status = arithmetic_read_value(
        source, source_dtype, &value, "cnp_sqrt");
    if (status != CNP_OK) return status;
    if (result_dtype == CNP_HALF) {
        uint16_t input_bits = cnp_float_to_half(
            arithmetic_value_double(&value));
        float input = (float)cnp_half_to_float(input_bits);
        volatile float result = sqrtf(input);
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float input = arithmetic_value_float(&value);
        volatile float result = sqrtf(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double input = arithmetic_value_double(&value);
        volatile double result = sqrt(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double input = arithmetic_value_longdouble(&value);
        volatile long double result = sqrtl(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, "cnp_sqrt",
        "result dtype %d does not support sqrt", (int)result_dtype);
    return CNP_ERR_TYPE;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static CnpArray* unary_sqrt_arrays(const CnpArray *source) {
    const char *function_name = "cnp_sqrt";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }

    CNP_TYPE source_dtype = source->dtype->type_num;
    CNP_TYPE result_dtype = unary_sqrt_result_dtype(source_dtype);
    if (result_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support sqrt",
            source->dtype->name);
        return NULL;
    }

    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);

    if (source_dtype == CNP_DOUBLE && source->size > 0 &&
            (source->flags & (
                CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS))) {
        cnp_simd_sqrt(
            (const double*)((const char*)source->data + source->offset),
            (double*)((char*)result->data + result->offset),
            source->size);
        return result;
    }

    int source_itemsize = source->dtype->elsize;
    int result_itemsize = result->dtype->elsize;
    if (source->flags & (
            CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) {
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        for (int64_t index = 0; index < source->size; ++index) {
            CNP_STATUS status = unary_sqrt_element(
                source_dtype, result_dtype,
                source_data + index * source_itemsize,
                result_data + index * result_itemsize);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                cnp_relabel_error(function_name);
                return NULL;
            }
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = unary_sqrt_element(
            source_dtype, result_dtype,
            (const char*)source->data + source_offset,
            (char*)result->data + result_offset);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

static CNP_TYPE unary_cbrt_result_dtype(CNP_TYPE source_dtype) {
    switch (source_dtype) {
        case CNP_BOOL:
        case CNP_BYTE:
        case CNP_UBYTE:
        case CNP_HALF:
            return CNP_HALF;
        case CNP_SHORT:
        case CNP_USHORT:
        case CNP_FLOAT:
            return CNP_FLOAT;
        case CNP_INT:
        case CNP_UINT:
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
        case CNP_DOUBLE:
            return CNP_DOUBLE;
        case CNP_LONGDOUBLE:
            return CNP_LONGDOUBLE;
        default:
            return CNP_NOTYPE;
    }
}

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static CNP_STATUS unary_cbrt_element(
    CNP_TYPE source_dtype,
    CNP_TYPE result_dtype,
    const void *source,
    void *destination) {
    if (source_dtype == CNP_HALF) {
        uint16_t bits;
        memcpy(&bits, source, sizeof(bits));
        if ((bits & UINT16_C(0x7c00)) == UINT16_C(0x7c00) &&
                (bits & UINT16_C(0x03ff)) != 0) {
            memcpy(destination, &bits, sizeof(bits));
            return CNP_OK;
        }
    }
    if (source_dtype == CNP_FLOAT) {
        uint32_t bits;
        memcpy(&bits, source, sizeof(bits));
        if ((bits & UINT32_C(0x7f800000)) ==
                UINT32_C(0x7f800000) &&
                (bits & UINT32_C(0x007fffff)) != 0) {
            memcpy(destination, &bits, sizeof(bits));
            return CNP_OK;
        }
    }
    if (source_dtype == CNP_DOUBLE) {
        uint64_t bits;
        memcpy(&bits, source, sizeof(bits));
        if ((bits & UINT64_C(0x7ff0000000000000)) ==
                UINT64_C(0x7ff0000000000000) &&
                (bits & UINT64_C(0x000fffffffffffff)) != 0) {
            memcpy(destination, &bits, sizeof(bits));
            return CNP_OK;
        }
    }
    if (source_dtype == CNP_LONGDOUBLE) {
        long double input;
        memcpy(&input, source, sizeof(input));
        if (isnan(input)) {
            memcpy(destination, source, sizeof(input));
            return CNP_OK;
        }
    }

    CnpArithmeticValue value;
    CNP_STATUS status = arithmetic_read_value(
        source, source_dtype, &value, "cnp_cbrt");
    if (status != CNP_OK) return status;
    if (result_dtype == CNP_HALF) {
        uint16_t input_bits = cnp_float_to_half(
            arithmetic_value_double(&value));
        float input = (float)cnp_half_to_float(input_bits);
        volatile float result = cbrtf(input);
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float input = arithmetic_value_float(&value);
        volatile float result = cbrtf(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double input = arithmetic_value_double(&value);
        volatile double result = cbrt(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double input = arithmetic_value_longdouble(&value);
        volatile long double result = cbrtl(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, "cnp_cbrt",
        "result dtype %d does not support cbrt", (int)result_dtype);
    return CNP_ERR_TYPE;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static CnpArray* unary_cbrt_arrays(const CnpArray *source) {
    const char *function_name = "cnp_cbrt";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }

    CNP_TYPE source_dtype = source->dtype->type_num;
    CNP_TYPE result_dtype = unary_cbrt_result_dtype(source_dtype);
    if (result_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support cbrt",
            source->dtype->name);
        return NULL;
    }

    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);

    int source_itemsize = source->dtype->elsize;
    int result_itemsize = result->dtype->elsize;
    if (source->flags & (
            CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) {
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        for (int64_t index = 0; index < source->size; ++index) {
            CNP_STATUS status = unary_cbrt_element(
                source_dtype, result_dtype,
                source_data + index * source_itemsize,
                result_data + index * result_itemsize);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                cnp_relabel_error(function_name);
                return NULL;
            }
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = unary_cbrt_element(
            source_dtype, result_dtype,
            (const char*)source->data + source_offset,
            (char*)result->data + result_offset);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

static CNP_TYPE unary_trigonometric_result_dtype(CNP_TYPE source_dtype) {
    switch (source_dtype) {
        case CNP_BOOL:
        case CNP_BYTE:
        case CNP_UBYTE:
        case CNP_HALF:
            return CNP_HALF;
        case CNP_SHORT:
        case CNP_USHORT:
        case CNP_FLOAT:
            return CNP_FLOAT;
        case CNP_INT:
        case CNP_UINT:
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
        case CNP_DOUBLE:
            return CNP_DOUBLE;
        case CNP_LONGDOUBLE:
            return CNP_LONGDOUBLE;
        case CNP_CFLOAT:
            return CNP_CFLOAT;
        case CNP_CDOUBLE:
            return CNP_CDOUBLE;
        case CNP_CLONGDOUBLE:
            return CNP_CLONGDOUBLE;
        default:
            return CNP_NOTYPE;
    }
}

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static cnp_cfloat unary_cos_cfloat(cnp_cfloat source) {
#ifdef _MSC_VER
    _Fcomplex native_source;
    _Fcomplex native_result;
    cnp_cfloat result;
    memcpy(&native_source, &source, sizeof(native_source));
    native_result = ccosf(native_source);
    memcpy(&result, &native_result, sizeof(result));
#else
    float complex native_result =
        ccosf(CMPLXF(source.real, source.imag));
    cnp_cfloat result = {crealf(native_result), cimagf(native_result)};
#endif
    return result;
}

static cnp_cdouble unary_cos_cdouble(cnp_cdouble source) {
#ifdef _MSC_VER
    _Dcomplex native_source;
    _Dcomplex native_result;
    cnp_cdouble result;
    memcpy(&native_source, &source, sizeof(native_source));
    native_result = ccos(native_source);
    memcpy(&result, &native_result, sizeof(result));
#else
    double complex native_result =
        ccos(CMPLX(source.real, source.imag));
    cnp_cdouble result = {creal(native_result), cimag(native_result)};
#endif
    return result;
}

static cnp_clongdouble unary_cos_clongdouble(
    cnp_clongdouble source) {
#ifdef _MSC_VER
    _Lcomplex native_source;
    _Lcomplex native_result;
    cnp_clongdouble result;
    memcpy(&native_source, &source, sizeof(native_source));
    native_result = ccosl(native_source);
    memcpy(&result, &native_result, sizeof(result));
#else
    long double complex native_result =
        ccosl(CMPLXL(source.real, source.imag));
    cnp_clongdouble result = {
        creall(native_result), cimagl(native_result)};
#endif
    return result;
}

static CNP_STATUS unary_cos_element(
    CNP_TYPE source_dtype,
    CNP_TYPE result_dtype,
    const void *source,
    void *destination) {
    if (source_dtype == CNP_CFLOAT) {
        cnp_cfloat value;
        memcpy(&value, source, sizeof(value));
        value = unary_cos_cfloat(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    if (source_dtype == CNP_CDOUBLE) {
        cnp_cdouble value;
        memcpy(&value, source, sizeof(value));
        value = unary_cos_cdouble(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    if (source_dtype == CNP_CLONGDOUBLE) {
        cnp_clongdouble value;
        memcpy(&value, source, sizeof(value));
        value = unary_cos_clongdouble(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }

    CnpArithmeticValue value;
    CNP_STATUS status = arithmetic_read_value(
        source, source_dtype, &value, "cnp_cos");
    if (status != CNP_OK) return status;
    if (result_dtype == CNP_HALF) {
        uint16_t input_bits = cnp_float_to_half(
            arithmetic_value_double(&value));
        float input = (float)cnp_half_to_float(input_bits);
        volatile float result = cosf(input);
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float input = arithmetic_value_float(&value);
        volatile float result = cosf(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double input = arithmetic_value_double(&value);
        volatile double result = cos(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double input = arithmetic_value_longdouble(&value);
        volatile long double result = cosl(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, "cnp_cos",
        "result dtype %d does not support cos", (int)result_dtype);
    return CNP_ERR_TYPE;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static CnpArray* unary_cos_arrays(const CnpArray *source) {
    const char *function_name = "cnp_cos";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }

    CNP_TYPE source_dtype = source->dtype->type_num;
    CNP_TYPE result_dtype = unary_trigonometric_result_dtype(source_dtype);
    if (result_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support cos",
            source->dtype->name);
        return NULL;
    }

    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);
    if (source->size == 0) return result;

    int source_itemsize = source->dtype->elsize;
    int result_itemsize = result->dtype->elsize;
    if (source->flags & (
            CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) {
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        for (int64_t index = 0; index < source->size; ++index) {
            CNP_STATUS status = unary_cos_element(
                source_dtype, result_dtype,
                source_data + index * source_itemsize,
                result_data + index * result_itemsize);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                cnp_relabel_error(function_name);
                return NULL;
            }
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = unary_cos_element(
            source_dtype, result_dtype,
            (const char*)source->data + source_offset,
            (char*)result->data + result_offset);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static cnp_cfloat unary_sin_cfloat(cnp_cfloat source) {
#ifdef _MSC_VER
    _Fcomplex native_source;
    _Fcomplex native_result;
    cnp_cfloat result;
    memcpy(&native_source, &source, sizeof(native_source));
    native_result = csinf(native_source);
    memcpy(&result, &native_result, sizeof(result));
#else
    float complex native_result =
        csinf(CMPLXF(source.real, source.imag));
    cnp_cfloat result = {crealf(native_result), cimagf(native_result)};
#endif
    return result;
}

static cnp_cdouble unary_sin_cdouble(cnp_cdouble source) {
#ifdef _MSC_VER
    _Dcomplex native_source;
    _Dcomplex native_result;
    cnp_cdouble result;
    memcpy(&native_source, &source, sizeof(native_source));
    native_result = csin(native_source);
    memcpy(&result, &native_result, sizeof(result));
#else
    double complex native_result =
        csin(CMPLX(source.real, source.imag));
    cnp_cdouble result = {creal(native_result), cimag(native_result)};
#endif
    return result;
}

static cnp_clongdouble unary_sin_clongdouble(
    cnp_clongdouble source) {
#ifdef _MSC_VER
    _Lcomplex native_source;
    _Lcomplex native_result;
    cnp_clongdouble result;
    memcpy(&native_source, &source, sizeof(native_source));
    native_result = csinl(native_source);
    memcpy(&result, &native_result, sizeof(result));
#else
    long double complex native_result =
        csinl(CMPLXL(source.real, source.imag));
    cnp_clongdouble result = {
        creall(native_result), cimagl(native_result)};
#endif
    return result;
}

static CNP_STATUS unary_sin_element(
    CNP_TYPE source_dtype,
    CNP_TYPE result_dtype,
    const void *source,
    void *destination) {
    if (source_dtype == CNP_CFLOAT) {
        cnp_cfloat value;
        memcpy(&value, source, sizeof(value));
        value = unary_sin_cfloat(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    if (source_dtype == CNP_CDOUBLE) {
        cnp_cdouble value;
        memcpy(&value, source, sizeof(value));
        value = unary_sin_cdouble(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    if (source_dtype == CNP_CLONGDOUBLE) {
        cnp_clongdouble value;
        memcpy(&value, source, sizeof(value));
        value = unary_sin_clongdouble(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }

    CnpArithmeticValue value;
    CNP_STATUS status = arithmetic_read_value(
        source, source_dtype, &value, "cnp_sin");
    if (status != CNP_OK) return status;
    if (result_dtype == CNP_HALF) {
        uint16_t input_bits = cnp_float_to_half(
            arithmetic_value_double(&value));
        float input = (float)cnp_half_to_float(input_bits);
        volatile float result = sinf(input);
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float input = arithmetic_value_float(&value);
        float result;
        cnp_simd_sin_f32(&input, &result, 1);
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double input = arithmetic_value_double(&value);
        volatile double result = sin(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double input = arithmetic_value_longdouble(&value);
        volatile long double result = sinl(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, "cnp_sin",
        "result dtype %d does not support sin", (int)result_dtype);
    return CNP_ERR_TYPE;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static CnpArray* unary_sin_arrays(const CnpArray *source) {
    const char *function_name = "cnp_sin";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }

    CNP_TYPE source_dtype = source->dtype->type_num;
    CNP_TYPE result_dtype = unary_trigonometric_result_dtype(source_dtype);
    if (result_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support sin",
            source->dtype->name);
        return NULL;
    }

    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);
    if (source->size == 0) return result;

    int source_itemsize = source->dtype->elsize;
    int result_itemsize = result->dtype->elsize;
    if (source->flags & (
            CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) {
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        if (source_dtype == CNP_FLOAT) {
            cnp_simd_sin_f32(
                (const float*)source_data,
                (float*)result_data,
                source->size);
            return result;
        }
        if (source_dtype == CNP_SHORT || source_dtype == CNP_USHORT) {
            enum { CNP_SIN_FLOAT_TILE_SIZE = 256 };
            float input_tile[CNP_SIN_FLOAT_TILE_SIZE];
            float output_tile[CNP_SIN_FLOAT_TILE_SIZE];
            for (int64_t tile_start = 0; tile_start < source->size;
                    tile_start += CNP_SIN_FLOAT_TILE_SIZE) {
                int64_t tile_size = source->size - tile_start;
                if (tile_size > CNP_SIN_FLOAT_TILE_SIZE)
                    tile_size = CNP_SIN_FLOAT_TILE_SIZE;
                for (int64_t index = 0; index < tile_size; ++index) {
                    if (source_dtype == CNP_SHORT) {
                        int16_t value;
                        memcpy(
                            &value,
                            source_data +
                                (tile_start + index) * source_itemsize,
                            sizeof(value));
                        input_tile[index] = (float)value;
                    } else {
                        uint16_t value;
                        memcpy(
                            &value,
                            source_data +
                                (tile_start + index) * source_itemsize,
                            sizeof(value));
                        input_tile[index] = (float)value;
                    }
                }
                cnp_simd_sin_f32(input_tile, output_tile, tile_size);
                memcpy(
                    result_data + tile_start * result_itemsize,
                    output_tile,
                    (size_t)tile_size * sizeof(float));
            }
            return result;
        }
        for (int64_t index = 0; index < source->size; ++index) {
            CNP_STATUS status = unary_sin_element(
                source_dtype, result_dtype,
                source_data + index * source_itemsize,
                result_data + index * result_itemsize);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                cnp_relabel_error(function_name);
                return NULL;
            }
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = unary_sin_element(
            source_dtype, result_dtype,
            (const char*)source->data + source_offset,
            (char*)result->data + result_offset);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static cnp_cfloat unary_tan_cfloat(cnp_cfloat source) {
#ifdef _MSC_VER
    _Fcomplex native_source;
    _Fcomplex native_result;
    cnp_cfloat result;
    memcpy(&native_source, &source, sizeof(native_source));
    native_result = ctanf(native_source);
    memcpy(&result, &native_result, sizeof(result));
#else
    float complex native_result =
        ctanf(CMPLXF(source.real, source.imag));
    cnp_cfloat result = {crealf(native_result), cimagf(native_result)};
#endif
    return result;
}

static cnp_cdouble unary_tan_cdouble(cnp_cdouble source) {
#ifdef _MSC_VER
    _Dcomplex native_source;
    _Dcomplex native_result;
    cnp_cdouble result;
    memcpy(&native_source, &source, sizeof(native_source));
    native_result = ctan(native_source);
    memcpy(&result, &native_result, sizeof(result));
#else
    double complex native_result =
        ctan(CMPLX(source.real, source.imag));
    cnp_cdouble result = {creal(native_result), cimag(native_result)};
#endif
    return result;
}

static cnp_clongdouble unary_tan_clongdouble(
    cnp_clongdouble source) {
#ifdef _MSC_VER
    _Lcomplex native_source;
    _Lcomplex native_result;
    cnp_clongdouble result;
    memcpy(&native_source, &source, sizeof(native_source));
    native_result = ctanl(native_source);
    memcpy(&result, &native_result, sizeof(result));
#else
    long double complex native_result =
        ctanl(CMPLXL(source.real, source.imag));
    cnp_clongdouble result = {
        creall(native_result), cimagl(native_result)};
#endif
    return result;
}

static CNP_STATUS unary_tan_element(
    CNP_TYPE source_dtype,
    CNP_TYPE result_dtype,
    const void *source,
    void *destination) {
    if (source_dtype == CNP_CFLOAT) {
        cnp_cfloat value;
        memcpy(&value, source, sizeof(value));
        value = unary_tan_cfloat(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    if (source_dtype == CNP_CDOUBLE) {
        cnp_cdouble value;
        memcpy(&value, source, sizeof(value));
        value = unary_tan_cdouble(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    if (source_dtype == CNP_CLONGDOUBLE) {
        cnp_clongdouble value;
        memcpy(&value, source, sizeof(value));
        value = unary_tan_clongdouble(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }

    CnpArithmeticValue value;
    CNP_STATUS status = arithmetic_read_value(
        source, source_dtype, &value, "cnp_tan");
    if (status != CNP_OK) return status;
    if (result_dtype == CNP_HALF) {
        uint16_t input_bits = cnp_float_to_half(
            arithmetic_value_double(&value));
        float input = (float)cnp_half_to_float(input_bits);
        volatile float result = tanf(input);
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float input = arithmetic_value_float(&value);
        volatile float result = tanf(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double input = arithmetic_value_double(&value);
        volatile double result = tan(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double input = arithmetic_value_longdouble(&value);
        volatile long double result = tanl(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, "cnp_tan",
        "result dtype %d does not support tan", (int)result_dtype);
    return CNP_ERR_TYPE;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static CnpArray* unary_tan_arrays(const CnpArray *source) {
    const char *function_name = "cnp_tan";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }

    CNP_TYPE source_dtype = source->dtype->type_num;
    CNP_TYPE result_dtype = unary_trigonometric_result_dtype(source_dtype);
    if (result_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support tan",
            source->dtype->name);
        return NULL;
    }

    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);
    if (source->size == 0) return result;

    int source_itemsize = source->dtype->elsize;
    int result_itemsize = result->dtype->elsize;
    if (source->flags & (
            CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) {
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        for (int64_t index = 0; index < source->size; ++index) {
            CNP_STATUS status = unary_tan_element(
                source_dtype, result_dtype,
                source_data + index * source_itemsize,
                result_data + index * result_itemsize);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                cnp_relabel_error(function_name);
                return NULL;
            }
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = unary_tan_element(
            source_dtype, result_dtype,
            (const char*)source->data + source_offset,
            (char*)result->data + result_offset);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static float unary_arcsin_numpy_f_float(
    float a, float b, float hypot_a_b) {
    if (b < 0.0f) return (hypot_a_b - b) / 2.0f;
    if (b == 0.0f) return a / 2.0f;
    return a * a / (hypot_a_b + b) / 2.0f;
}

static float unary_arcsin_numpy_hypot_float(float x, float y) {
    return (float)hypot((double)x, (double)y);
}

static void unary_arcsin_numpy_hard_work_float(
    float x,
    float y,
    float *rx,
    int *b_is_usable,
    float *b,
    float *sqrt_a2_minus_y2,
    float *new_y) {
    const float a_crossover = 10.0f;
    const float b_crossover = 0.6417f;
    const float four_sqrt_min = 4.3368086899420177e-19f;
    float r = unary_arcsin_numpy_hypot_float(x, y + 1.0f);
    float s = unary_arcsin_numpy_hypot_float(x, y - 1.0f);
    float a = (r + s) / 2.0f;
    float am1;
    float amy;

    if (a < 1.0f) a = 1.0f;
    if (a < a_crossover) {
        if (y == 1.0f && x < FLT_EPSILON * FLT_EPSILON / 128.0f) {
            *rx = sqrtf(x);
        }
        else if (x >= FLT_EPSILON * fabsf(y - 1.0f)) {
            am1 = unary_arcsin_numpy_f_float(x, 1.0f + y, r) +
                unary_arcsin_numpy_f_float(x, 1.0f - y, s);
            *rx = log1pf(am1 + sqrtf(am1 * (a + 1.0f)));
        }
        else if (y < 1.0f) {
            *rx = x / sqrtf((1.0f - y) * (1.0f + y));
        }
        else {
            *rx = log1pf(
                (y - 1.0f) + sqrtf((y - 1.0f) * (y + 1.0f)));
        }
    }
    else {
        *rx = logf(a + sqrtf(a * a - 1.0f));
    }

    *new_y = y;
    if (y < four_sqrt_min) {
        *b_is_usable = 0;
        *sqrt_a2_minus_y2 = a * (2.0f / FLT_EPSILON);
        *new_y = y * (2.0f / FLT_EPSILON);
        return;
    }

    *b = y / a;
    *b_is_usable = 1;
    if (*b > b_crossover) {
        *b_is_usable = 0;
        if (y == 1.0f && x < FLT_EPSILON / 128.0f) {
            *sqrt_a2_minus_y2 = sqrtf(x) * sqrtf((a + y) / 2.0f);
        }
        else if (x >= FLT_EPSILON * fabsf(y - 1.0f)) {
            amy = unary_arcsin_numpy_f_float(x, y + 1.0f, r) +
                unary_arcsin_numpy_f_float(x, y - 1.0f, s);
            *sqrt_a2_minus_y2 = sqrtf(amy * (a + y));
        }
        else if (y > 1.0f) {
            *sqrt_a2_minus_y2 =
                x * (4.0f / FLT_EPSILON / FLT_EPSILON) * y /
                sqrtf((y + 1.0f) * (y - 1.0f));
            *new_y = y * (4.0f / FLT_EPSILON / FLT_EPSILON);
        }
        else {
            *sqrt_a2_minus_y2 = sqrtf((1.0f - y) * (1.0f + y));
        }
    }
}

static void unary_arcsin_numpy_clog_large_float(
    float x, float y, float *real, float *imaginary) {
    const float numpy_e = 2.718281828459045235360287471352662498f;
    const float quarter_sqrt_max = 4.611685743549481e+18f;
    const float sqrt_min = 1.0842021724855044e-19f;
    float ax = fabsf(x);
    float ay = fabsf(y);
    if (ax < ay) {
        float swap = ax;
        ax = ay;
        ay = swap;
    }
    if (ax > FLT_MAX / 2.0f) {
        *real = logf(unary_arcsin_numpy_hypot_float(
            x / numpy_e, y / numpy_e)) + 1.0f;
    }
    else if (ax > quarter_sqrt_max || ay < sqrt_min) {
        *real = logf(unary_arcsin_numpy_hypot_float(x, y));
    }
    else {
        *real = logf(ax * ax + ay * ay) / 2.0f;
    }
    *imaginary = atan2f(y, x);
}

static cnp_cfloat unary_arcsin_numpy_asinh_float(cnp_cfloat source) {
    const float sqrt_6_epsilon = 8.4572793338e-4f;
    const float reciprocal_epsilon = 1.0f / FLT_EPSILON;
    const float loge2 = 0.693147180559945309417232121458176568f;
    float x = source.real;
    float y = source.imag;
    float ax = fabsf(x);
    float ay = fabsf(y);
    float wx;
    float wy;
    float rx;
    float ry;
    float b;
    float sqrt_a2_minus_y2;
    float new_y;
    int b_is_usable;

    if (isnan(x) || isnan(y)) {
        if (isinf(x)) {
            cnp_cfloat result = {x, y + y};
            return result;
        }
        if (isinf(y)) {
            cnp_cfloat result = {y, x + x};
            return result;
        }
        if (y == 0.0f) {
            cnp_cfloat result = {x + x, y};
            return result;
        }
        {
            cnp_cfloat result = {(float)NAN, (float)NAN};
            return result;
        }
    }

    if (ax > reciprocal_epsilon || ay > reciprocal_epsilon) {
        if (!signbit(x)) {
            unary_arcsin_numpy_clog_large_float(x, y, &wx, &wy);
            wx += loge2;
        }
        else {
            unary_arcsin_numpy_clog_large_float(-x, -y, &wx, &wy);
            wx += loge2;
        }
        {
            cnp_cfloat result = {
                copysignf(wx, x), copysignf(wy, y)};
            return result;
        }
    }
    if (x == 0.0f && y == 0.0f) return source;
    {
        volatile float inexact = 1.0f + 3.9443045e-31f;
        (void)inexact;
    }
    if (ax < sqrt_6_epsilon / 4.0f &&
            ay < sqrt_6_epsilon / 4.0f) {
        return source;
    }

    unary_arcsin_numpy_hard_work_float(
        ax, ay, &rx, &b_is_usable, &b,
        &sqrt_a2_minus_y2, &new_y);
    if (b_is_usable) ry = asinf(b);
    else ry = atan2f(new_y, sqrt_a2_minus_y2);
    {
        cnp_cfloat result = {
            copysignf(rx, x), copysignf(ry, y)};
        return result;
    }
}

static double unary_arcsin_numpy_f_double(
    double a, double b, double hypot_a_b) {
    if (b < 0.0) return (hypot_a_b - b) / 2.0;
    if (b == 0.0) return a / 2.0;
    return a * a / (hypot_a_b + b) / 2.0;
}

static double unary_arcsin_numpy_hypot_double(double x, double y) {
    return hypot(x, y);
}

static void unary_arcsin_numpy_hard_work_double(
    double x,
    double y,
    double *rx,
    int *b_is_usable,
    double *b,
    double *sqrt_a2_minus_y2,
    double *new_y) {
    const double a_crossover = 10.0;
    const double b_crossover = 0.6417;
    const double four_sqrt_min = 5.9666725849601654e-154;
    double r = unary_arcsin_numpy_hypot_double(x, y + 1.0);
    double s = unary_arcsin_numpy_hypot_double(x, y - 1.0);
    double a = (r + s) / 2.0;
    double am1;
    double amy;

    if (a < 1.0) a = 1.0;
    if (a < a_crossover) {
        if (y == 1.0 && x < DBL_EPSILON * DBL_EPSILON / 128.0) {
            *rx = sqrt(x);
        }
        else if (x >= DBL_EPSILON * fabs(y - 1.0)) {
            am1 = unary_arcsin_numpy_f_double(x, 1.0 + y, r) +
                unary_arcsin_numpy_f_double(x, 1.0 - y, s);
            *rx = log1p(am1 + sqrt(am1 * (a + 1.0)));
        }
        else if (y < 1.0) {
            *rx = x / sqrt((1.0 - y) * (1.0 + y));
        }
        else {
            *rx = log1p((y - 1.0) + sqrt((y - 1.0) * (y + 1.0)));
        }
    }
    else {
        *rx = log(a + sqrt(a * a - 1.0));
    }

    *new_y = y;
    if (y < four_sqrt_min) {
        *b_is_usable = 0;
        *sqrt_a2_minus_y2 = a * (2.0 / DBL_EPSILON);
        *new_y = y * (2.0 / DBL_EPSILON);
        return;
    }

    *b = y / a;
    *b_is_usable = 1;
    if (*b > b_crossover) {
        *b_is_usable = 0;
        if (y == 1.0 && x < DBL_EPSILON / 128.0) {
            *sqrt_a2_minus_y2 = sqrt(x) * sqrt((a + y) / 2.0);
        }
        else if (x >= DBL_EPSILON * fabs(y - 1.0)) {
            amy = unary_arcsin_numpy_f_double(x, y + 1.0, r) +
                unary_arcsin_numpy_f_double(x, y - 1.0, s);
            *sqrt_a2_minus_y2 = sqrt(amy * (a + y));
        }
        else if (y > 1.0) {
            *sqrt_a2_minus_y2 =
                x * (4.0 / DBL_EPSILON / DBL_EPSILON) * y /
                sqrt((y + 1.0) * (y - 1.0));
            *new_y = y * (4.0 / DBL_EPSILON / DBL_EPSILON);
        }
        else {
            *sqrt_a2_minus_y2 = sqrt((1.0 - y) * (1.0 + y));
        }
    }
}

static void unary_arcsin_numpy_clog_large_double(
    double x, double y, double *real, double *imaginary) {
    const double numpy_e = 2.718281828459045235360287471352662498;
    const double quarter_sqrt_max = 3.3519519824856489e+153;
    const double sqrt_min = 1.4916681462400413e-154;
    double ax = fabs(x);
    double ay = fabs(y);
    if (ax < ay) {
        double swap = ax;
        ax = ay;
        ay = swap;
    }
    if (ax > DBL_MAX / 2.0) {
        *real = log(unary_arcsin_numpy_hypot_double(
            x / numpy_e, y / numpy_e)) + 1.0;
    }
    else if (ax > quarter_sqrt_max || ay < sqrt_min) {
        *real = log(unary_arcsin_numpy_hypot_double(x, y));
    }
    else {
        *real = log(ax * ax + ay * ay) / 2.0;
    }
    *imaginary = atan2(y, x);
}

static cnp_cdouble unary_arcsin_numpy_asinh_double(cnp_cdouble source) {
    const double sqrt_6_epsilon = 3.65002414998885671e-08;
    const double reciprocal_epsilon = 1.0 / DBL_EPSILON;
    const double loge2 = 0.693147180559945309417232121458176568;
    double x = source.real;
    double y = source.imag;
    double ax = fabs(x);
    double ay = fabs(y);
    double wx;
    double wy;
    double rx;
    double ry;
    double b;
    double sqrt_a2_minus_y2;
    double new_y;
    int b_is_usable;

    if (isnan(x) || isnan(y)) {
        if (isinf(x)) {
            cnp_cdouble result = {x, y + y};
            return result;
        }
        if (isinf(y)) {
            cnp_cdouble result = {y, x + x};
            return result;
        }
        if (y == 0.0) {
            cnp_cdouble result = {x + x, y};
            return result;
        }
        {
            cnp_cdouble result = {NAN, NAN};
            return result;
        }
    }

    if (ax > reciprocal_epsilon || ay > reciprocal_epsilon) {
        if (!signbit(x)) {
            unary_arcsin_numpy_clog_large_double(x, y, &wx, &wy);
            wx += loge2;
        }
        else {
            unary_arcsin_numpy_clog_large_double(-x, -y, &wx, &wy);
            wx += loge2;
        }
        {
            cnp_cdouble result = {copysign(wx, x), copysign(wy, y)};
            return result;
        }
    }
    if (x == 0.0 && y == 0.0) return source;
    {
        volatile float inexact = 1.0f + 3.9443045e-31f;
        (void)inexact;
    }
    if (ax < sqrt_6_epsilon / 4.0 &&
            ay < sqrt_6_epsilon / 4.0) {
        return source;
    }

    unary_arcsin_numpy_hard_work_double(
        ax, ay, &rx, &b_is_usable, &b,
        &sqrt_a2_minus_y2, &new_y);
    if (b_is_usable) ry = asin(b);
    else ry = atan2(new_y, sqrt_a2_minus_y2);
    {
        cnp_cdouble result = {copysign(rx, x), copysign(ry, y)};
        return result;
    }
}

static cnp_cfloat unary_arcsin_cfloat(cnp_cfloat source) {
    cnp_cfloat transformed = {source.imag, source.real};
    cnp_cfloat intermediate =
        unary_arcsin_numpy_asinh_float(transformed);
    cnp_cfloat result = {intermediate.imag, intermediate.real};
    return result;
}

static cnp_cdouble unary_arcsin_cdouble(cnp_cdouble source) {
    cnp_cdouble transformed = {source.imag, source.real};
    cnp_cdouble intermediate =
        unary_arcsin_numpy_asinh_double(transformed);
    cnp_cdouble result = {intermediate.imag, intermediate.real};
    return result;
}

static cnp_clongdouble unary_arcsin_clongdouble(
    cnp_clongdouble source) {
#ifdef _MSC_VER
    cnp_cdouble narrowed = {(double)source.real, (double)source.imag};
    cnp_cdouble value = unary_arcsin_cdouble(narrowed);
    cnp_clongdouble result = {
        (long double)value.real, (long double)value.imag};
#else
    long double complex native_result =
        casinl(CMPLXL(source.real, source.imag));
    cnp_clongdouble result = {
        creall(native_result), cimagl(native_result)};
#endif
    return result;
}

static CNP_STATUS unary_arcsin_element(
    CNP_TYPE source_dtype,
    CNP_TYPE result_dtype,
    const void *source,
    void *destination) {
    if (source_dtype == CNP_CFLOAT) {
        cnp_cfloat value;
        memcpy(&value, source, sizeof(value));
        value = unary_arcsin_cfloat(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    if (source_dtype == CNP_CDOUBLE) {
        cnp_cdouble value;
        memcpy(&value, source, sizeof(value));
        value = unary_arcsin_cdouble(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    if (source_dtype == CNP_CLONGDOUBLE) {
        cnp_clongdouble value;
        memcpy(&value, source, sizeof(value));
        value = unary_arcsin_clongdouble(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }

    CnpArithmeticValue value;
    CNP_STATUS status = arithmetic_read_value(
        source, source_dtype, &value, "cnp_arcsin");
    if (status != CNP_OK) return status;
    if (result_dtype == CNP_HALF) {
        uint16_t input_bits = cnp_float_to_half(
            arithmetic_value_double(&value));
        float input = (float)cnp_half_to_float(input_bits);
        volatile float result = asinf(input);
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float input = arithmetic_value_float(&value);
        volatile float result = asinf(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double input = arithmetic_value_double(&value);
        volatile double result = asin(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double input = arithmetic_value_longdouble(&value);
        volatile long double result = asinl(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, "cnp_arcsin",
        "result dtype %d does not support arcsin", (int)result_dtype);
    return CNP_ERR_TYPE;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static CnpArray* unary_arcsin_arrays(const CnpArray *source) {
    const char *function_name = "cnp_arcsin";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }

    CNP_TYPE source_dtype = source->dtype->type_num;
    CNP_TYPE result_dtype = unary_trigonometric_result_dtype(source_dtype);
    if (result_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support arcsin",
            source->dtype->name);
        return NULL;
    }

    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);
    if (source->size == 0) return result;

    int source_itemsize = source->dtype->elsize;
    int result_itemsize = result->dtype->elsize;
    if (source->flags & (
            CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) {
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        for (int64_t index = 0; index < source->size; ++index) {
            CNP_STATUS status = unary_arcsin_element(
                source_dtype, result_dtype,
                source_data + index * source_itemsize,
                result_data + index * result_itemsize);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                cnp_relabel_error(function_name);
                return NULL;
            }
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = unary_arcsin_element(
            source_dtype, result_dtype,
            (const char*)source->data + source_offset,
            (char*)result->data + result_offset);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static cnp_cfloat unary_arccos_numpy_float(cnp_cfloat source) {
    const float sqrt_6_epsilon = 8.4572793338e-4f;
    const volatile float pi_over_2_low = 7.5497899549e-9f;
    const float reciprocal_epsilon = 1.0f / FLT_EPSILON;
    const float pi_over_2 =
        1.570796326794896619231321691639751442f;
    const float loge2 =
        0.693147180559945309417232121458176568f;
    float x = source.real;
    float y = source.imag;
    int sign_x = signbit(x);
    int sign_y = signbit(y);
    float abs_x = fabsf(x);
    float abs_y = fabsf(y);
    float wx;
    float wy;
    float rx;
    float ry;
    float b;
    float sqrt_a2_minus_x2;
    float new_x;
    int b_is_usable;

    if (isnan(x) || isnan(y)) {
        if (isinf(x)) {
            cnp_cfloat result = {y + y, -(float)INFINITY};
            return result;
        }
        if (isinf(y)) {
            cnp_cfloat result = {x + x, -y};
            return result;
        }
        if (x == 0.0f) {
            cnp_cfloat result = {pi_over_2 + pi_over_2_low, y + y};
            return result;
        }
        {
            cnp_cfloat result = {(float)NAN, (float)NAN};
            return result;
        }
    }

    if (abs_x > reciprocal_epsilon || abs_y > reciprocal_epsilon) {
        unary_arcsin_numpy_clog_large_float(x, y, &wx, &wy);
        rx = fabsf(wy);
        ry = wx + loge2;
        if (!sign_y) ry = -ry;
        {
            cnp_cfloat result = {rx, ry};
            return result;
        }
    }

    if (x == 1.0f && y == 0.0f) {
        cnp_cfloat result = {0.0f, -y};
        return result;
    }
    {
        volatile float inexact = 1.0f + 3.9443045e-31f;
        (void)inexact;
    }
    if (abs_x < sqrt_6_epsilon / 4.0f &&
            abs_y < sqrt_6_epsilon / 4.0f) {
        cnp_cfloat result = {
            pi_over_2 - (x - pi_over_2_low), -y};
        return result;
    }

    unary_arcsin_numpy_hard_work_float(
        abs_y, abs_x, &ry, &b_is_usable, &b,
        &sqrt_a2_minus_x2, &new_x);
    if (b_is_usable) {
        rx = !sign_x ? acosf(b) : acosf(-b);
    }
    else {
        rx = !sign_x
            ? atan2f(sqrt_a2_minus_x2, new_x)
            : atan2f(sqrt_a2_minus_x2, -new_x);
    }
    if (!sign_y) ry = -ry;
    {
        cnp_cfloat result = {rx, ry};
        return result;
    }
}

static cnp_cdouble unary_arccos_numpy_double(cnp_cdouble source) {
    const double sqrt_6_epsilon = 3.65002414998885671e-08;
    const volatile double pi_over_2_low = 6.1232339957367659e-17;
    const double reciprocal_epsilon = 1.0 / DBL_EPSILON;
    const double pi_over_2 =
        1.570796326794896619231321691639751442;
    const double loge2 =
        0.693147180559945309417232121458176568;
    double x = source.real;
    double y = source.imag;
    int sign_x = signbit(x);
    int sign_y = signbit(y);
    double abs_x = fabs(x);
    double abs_y = fabs(y);
    double wx;
    double wy;
    double rx;
    double ry;
    double b;
    double sqrt_a2_minus_x2;
    double new_x;
    int b_is_usable;

    if (isnan(x) || isnan(y)) {
        if (isinf(x)) {
            cnp_cdouble result = {y + y, -INFINITY};
            return result;
        }
        if (isinf(y)) {
            cnp_cdouble result = {x + x, -y};
            return result;
        }
        if (x == 0.0) {
            cnp_cdouble result = {pi_over_2 + pi_over_2_low, y + y};
            return result;
        }
        {
            cnp_cdouble result = {NAN, NAN};
            return result;
        }
    }

    if (abs_x > reciprocal_epsilon || abs_y > reciprocal_epsilon) {
        unary_arcsin_numpy_clog_large_double(x, y, &wx, &wy);
        rx = fabs(wy);
        ry = wx + loge2;
        if (!sign_y) ry = -ry;
        {
            cnp_cdouble result = {rx, ry};
            return result;
        }
    }

    if (x == 1.0 && y == 0.0) {
        cnp_cdouble result = {0.0, -y};
        return result;
    }
    {
        volatile float inexact = 1.0f + 3.9443045e-31f;
        (void)inexact;
    }
    if (abs_x < sqrt_6_epsilon / 4.0 &&
            abs_y < sqrt_6_epsilon / 4.0) {
        cnp_cdouble result = {
            pi_over_2 - (x - pi_over_2_low), -y};
        return result;
    }

    unary_arcsin_numpy_hard_work_double(
        abs_y, abs_x, &ry, &b_is_usable, &b,
        &sqrt_a2_minus_x2, &new_x);
    if (b_is_usable) {
        rx = !sign_x ? acos(b) : acos(-b);
    }
    else {
        rx = !sign_x
            ? atan2(sqrt_a2_minus_x2, new_x)
            : atan2(sqrt_a2_minus_x2, -new_x);
    }
    if (!sign_y) ry = -ry;
    {
        cnp_cdouble result = {rx, ry};
        return result;
    }
}

static cnp_cfloat unary_arccos_cfloat(cnp_cfloat source) {
    return unary_arccos_numpy_float(source);
}

static cnp_cdouble unary_arccos_cdouble(cnp_cdouble source) {
    return unary_arccos_numpy_double(source);
}

static cnp_clongdouble unary_arccos_clongdouble(
    cnp_clongdouble source) {
#ifdef _MSC_VER
    cnp_cdouble narrowed = {(double)source.real, (double)source.imag};
    cnp_cdouble value = unary_arccos_cdouble(narrowed);
    cnp_clongdouble result = {
        (long double)value.real, (long double)value.imag};
#else
    long double complex native_result =
        cacosl(CMPLXL(source.real, source.imag));
    cnp_clongdouble result = {
        creall(native_result), cimagl(native_result)};
#endif
    return result;
}

static CNP_STATUS unary_arccos_element(
    CNP_TYPE source_dtype,
    CNP_TYPE result_dtype,
    const void *source,
    void *destination) {
    if (source_dtype == CNP_CFLOAT) {
        cnp_cfloat value;
        memcpy(&value, source, sizeof(value));
        value = unary_arccos_cfloat(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    if (source_dtype == CNP_CDOUBLE) {
        cnp_cdouble value;
        memcpy(&value, source, sizeof(value));
        value = unary_arccos_cdouble(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    if (source_dtype == CNP_CLONGDOUBLE) {
        cnp_clongdouble value;
        memcpy(&value, source, sizeof(value));
        value = unary_arccos_clongdouble(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }

    CnpArithmeticValue value;
    CNP_STATUS status = arithmetic_read_value(
        source, source_dtype, &value, "cnp_arccos");
    if (status != CNP_OK) return status;
    if (result_dtype == CNP_HALF) {
        uint16_t input_bits = cnp_float_to_half(
            arithmetic_value_double(&value));
        float input = (float)cnp_half_to_float(input_bits);
        volatile float result = acosf(input);
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float input = arithmetic_value_float(&value);
        volatile float result = acosf(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double input = arithmetic_value_double(&value);
        volatile double result = acos(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double input = arithmetic_value_longdouble(&value);
        volatile long double result = acosl(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, "cnp_arccos",
        "result dtype %d does not support arccos", (int)result_dtype);
    return CNP_ERR_TYPE;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static CnpArray* unary_arccos_arrays(const CnpArray *source) {
    const char *function_name = "cnp_arccos";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }

    CNP_TYPE source_dtype = source->dtype->type_num;
    CNP_TYPE result_dtype = unary_trigonometric_result_dtype(source_dtype);
    if (result_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support arccos",
            source->dtype->name);
        return NULL;
    }

    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);
    if (source->size == 0) return result;

    int source_itemsize = source->dtype->elsize;
    int result_itemsize = result->dtype->elsize;
    if (source->flags & (
            CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) {
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        for (int64_t index = 0; index < source->size; ++index) {
            CNP_STATUS status = unary_arccos_element(
                source_dtype, result_dtype,
                source_data + index * source_itemsize,
                result_data + index * result_itemsize);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                cnp_relabel_error(function_name);
                return NULL;
            }
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = unary_arccos_element(
            source_dtype, result_dtype,
            (const char*)source->data + source_offset,
            (char*)result->data + result_offset);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static float unary_arctan_numpy_sum_squares_float(
    float x, float y) {
    const float sqrt_min = 1.0842022e-19f;
    if (y < sqrt_min) return x * x;
    return x * x + y * y;
}

static double unary_arctan_numpy_sum_squares_double(
    double x, double y) {
    const double sqrt_min = 1.4916681462400413e-154;
    if (y < sqrt_min) return x * x;
    return x * x + y * y;
}

static float unary_arctan_numpy_real_reciprocal_float(
    float x, float y) {
    const int32_t cutoff = FLT_MANT_DIG / 2 + 1;
    const int32_t bias = FLT_MAX_EXP - 1;
    float scale;
    uint32_t x_bits;
    uint32_t y_bits;
    int32_t x_exponent;
    int32_t y_exponent;
    memcpy(&x_bits, &x, sizeof(x_bits));
    memcpy(&y_bits, &y, sizeof(y_bits));
    x_exponent = (int32_t)(x_bits & UINT32_C(0x7f800000));
    y_exponent = (int32_t)(y_bits & UINT32_C(0x7f800000));
    if (x_exponent - y_exponent >= cutoff << 23 || isinf(x)) {
        return 1.0f / x;
    }
    if (y_exponent - x_exponent >= cutoff << 23) {
        return x / y / y;
    }
    if (x_exponent <=
            (bias + FLT_MAX_EXP / 2 - cutoff) << 23) {
        return x / (x * x + y * y);
    }
    {
        uint32_t scale_bits =
            UINT32_C(0x7f800000) - (uint32_t)x_exponent;
        memcpy(&scale, &scale_bits, sizeof(scale));
    }
    x *= scale;
    y *= scale;
    return x / (x * x + y * y) * scale;
}

static double unary_arctan_numpy_real_reciprocal_double(
    double x, double y) {
    const int32_t cutoff = DBL_MANT_DIG / 2 + 1;
    const int32_t bias = DBL_MAX_EXP - 1;
    double scale = 1.0;
    uint64_t x_bits;
    uint64_t y_bits;
    int32_t x_exponent;
    int32_t y_exponent;
    memcpy(&x_bits, &x, sizeof(x_bits));
    memcpy(&y_bits, &y, sizeof(y_bits));
    x_exponent = (int32_t)(
        (x_bits >> 32) & UINT64_C(0x7ff00000));
    y_exponent = (int32_t)(
        (y_bits >> 32) & UINT64_C(0x7ff00000));
    if (x_exponent - y_exponent >= cutoff << 20 || isinf(x)) {
        return 1.0 / x;
    }
    if (y_exponent - x_exponent >= cutoff << 20) {
        return x / y / y;
    }
    if (x_exponent <=
            (bias + DBL_MAX_EXP / 2 - cutoff) << 20) {
        return x / (x * x + y * y);
    }
    {
        uint64_t scale_bits;
        memcpy(&scale_bits, &scale, sizeof(scale_bits));
        scale_bits =
            (scale_bits & UINT64_C(0x00000000ffffffff)) |
            ((uint64_t)(UINT32_C(0x7ff00000) -
                (uint32_t)x_exponent) << 32);
        memcpy(&scale, &scale_bits, sizeof(scale));
    }
    x *= scale;
    y *= scale;
    return x / (x * x + y * y) * scale;
}

static cnp_cfloat unary_arctan_numpy_atanh_float(
    cnp_cfloat source) {
    const float sqrt_3_epsilon = 5.9801995673e-4f;
    const volatile float pi_over_2_low = 7.5497899549e-9f;
    const float reciprocal_epsilon = 1.0f / FLT_EPSILON;
    const float pi_over_2 =
        1.570796326794896619231321691639751442f;
    const float loge2 =
        0.693147180559945309417232121458176568f;
    float x = source.real;
    float y = source.imag;
    float abs_x = fabsf(x);
    float abs_y = fabsf(y);
    float rx;
    float ry;

    if (y == 0.0f && abs_x <= 1.0f) {
        cnp_cfloat result = {atanhf(x), y};
        return result;
    }
    if (x == 0.0f) {
        cnp_cfloat result = {x, atanf(y)};
        return result;
    }
    if (isnan(x) || isnan(y)) {
        if (isinf(x)) {
            cnp_cfloat result = {copysignf(0.0f, x), y + y};
            return result;
        }
        if (isinf(y)) {
            cnp_cfloat result = {
                copysignf(0.0f, x),
                copysignf(pi_over_2 + pi_over_2_low, y)};
            return result;
        }
        {
            cnp_cfloat result = {(float)NAN, (float)NAN};
            return result;
        }
    }
    if (abs_x > reciprocal_epsilon ||
            abs_y > reciprocal_epsilon) {
        cnp_cfloat result = {
            unary_arctan_numpy_real_reciprocal_float(x, y),
            copysignf(pi_over_2 + pi_over_2_low, y)};
        return result;
    }
    if (abs_x < sqrt_3_epsilon / 2.0f &&
            abs_y < sqrt_3_epsilon / 2.0f) {
        volatile float inexact = 1.0f + 3.9443045e-31f;
        (void)inexact;
        return source;
    }
    if (abs_x == 1.0f && abs_y < FLT_EPSILON) {
        rx = (loge2 - logf(abs_y)) / 2.0f;
    }
    else {
        rx = log1pf(
            4.0f * abs_x /
            unary_arctan_numpy_sum_squares_float(
                abs_x - 1.0f, abs_y)) / 4.0f;
    }
    if (abs_x == 1.0f) {
        ry = atan2f(2.0f, -abs_y) / 2.0f;
    }
    else if (abs_y < FLT_EPSILON) {
        ry = atan2f(
            2.0f * abs_y,
            (1.0f - abs_x) * (1.0f + abs_x)) / 2.0f;
    }
    else {
        ry = atan2f(
            2.0f * abs_y,
            (1.0f - abs_x) * (1.0f + abs_x) -
                abs_y * abs_y) / 2.0f;
    }
    {
        cnp_cfloat result = {
            copysignf(rx, x), copysignf(ry, y)};
        return result;
    }
}

static cnp_cdouble unary_arctan_numpy_atanh_double(
    cnp_cdouble source) {
    const double sqrt_3_epsilon = 2.5809568279517849e-8;
    const volatile double pi_over_2_low = 6.1232339957367659e-17;
    const double reciprocal_epsilon = 1.0 / DBL_EPSILON;
    const double pi_over_2 =
        1.570796326794896619231321691639751442;
    const double loge2 =
        0.693147180559945309417232121458176568;
    double x = source.real;
    double y = source.imag;
    double abs_x = fabs(x);
    double abs_y = fabs(y);
    double rx;
    double ry;

    if (y == 0.0 && abs_x <= 1.0) {
        cnp_cdouble result = {atanh(x), y};
        return result;
    }
    if (x == 0.0) {
        cnp_cdouble result = {x, atan(y)};
        return result;
    }
    if (isnan(x) || isnan(y)) {
        if (isinf(x)) {
            cnp_cdouble result = {copysign(0.0, x), y + y};
            return result;
        }
        if (isinf(y)) {
            cnp_cdouble result = {
                copysign(0.0, x),
                copysign(pi_over_2 + pi_over_2_low, y)};
            return result;
        }
        {
            cnp_cdouble result = {NAN, NAN};
            return result;
        }
    }
    if (abs_x > reciprocal_epsilon ||
            abs_y > reciprocal_epsilon) {
        cnp_cdouble result = {
            unary_arctan_numpy_real_reciprocal_double(x, y),
            copysign(pi_over_2 + pi_over_2_low, y)};
        return result;
    }
    if (abs_x < sqrt_3_epsilon / 2.0 &&
            abs_y < sqrt_3_epsilon / 2.0) {
        volatile float inexact = 1.0f + 3.9443045e-31f;
        (void)inexact;
        return source;
    }
    if (abs_x == 1.0 && abs_y < DBL_EPSILON) {
        rx = (loge2 - log(abs_y)) / 2.0;
    }
    else {
        rx = log1p(
            4.0 * abs_x /
            unary_arctan_numpy_sum_squares_double(
                abs_x - 1.0, abs_y)) / 4.0;
    }
    if (abs_x == 1.0) {
        ry = atan2(2.0, -abs_y) / 2.0;
    }
    else if (abs_y < DBL_EPSILON) {
        ry = atan2(
            2.0 * abs_y,
            (1.0 - abs_x) * (1.0 + abs_x)) / 2.0;
    }
    else {
        ry = atan2(
            2.0 * abs_y,
            (1.0 - abs_x) * (1.0 + abs_x) -
                abs_y * abs_y) / 2.0;
    }
    {
        cnp_cdouble result = {
            copysign(rx, x), copysign(ry, y)};
        return result;
    }
}

static cnp_cfloat unary_arctan_cfloat(cnp_cfloat source) {
#ifdef _MSC_VER
    cnp_cfloat transformed = {source.imag, source.real};
    cnp_cfloat intermediate =
        unary_arctan_numpy_atanh_float(transformed);
    cnp_cfloat result = {intermediate.imag, intermediate.real};
    return result;
#else
    float complex native_result =
        catanf(CMPLXF(source.real, source.imag));
    cnp_cfloat result = {
        crealf(native_result), cimagf(native_result)};
    return result;
#endif
}

static cnp_cdouble unary_arctan_cdouble(cnp_cdouble source) {
#ifdef _MSC_VER
    cnp_cdouble transformed = {source.imag, source.real};
    cnp_cdouble intermediate =
        unary_arctan_numpy_atanh_double(transformed);
    cnp_cdouble result = {intermediate.imag, intermediate.real};
    return result;
#else
    double complex native_result =
        catan(CMPLX(source.real, source.imag));
    cnp_cdouble result = {
        creal(native_result), cimag(native_result)};
    return result;
#endif
}

static cnp_clongdouble unary_arctan_clongdouble(
    cnp_clongdouble source) {
#ifdef _MSC_VER
    cnp_cdouble narrowed = {(double)source.real, (double)source.imag};
    cnp_cdouble value = unary_arctan_cdouble(narrowed);
    cnp_clongdouble result = {
        (long double)value.real, (long double)value.imag};
#else
    long double complex native_result =
        catanl(CMPLXL(source.real, source.imag));
    cnp_clongdouble result = {
        creall(native_result), cimagl(native_result)};
#endif
    return result;
}

static CNP_STATUS unary_arctan_element(
    CNP_TYPE source_dtype,
    CNP_TYPE result_dtype,
    const void *source,
    void *destination) {
    if (source_dtype == CNP_CFLOAT) {
        cnp_cfloat value;
        memcpy(&value, source, sizeof(value));
        value = unary_arctan_cfloat(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    if (source_dtype == CNP_CDOUBLE) {
        cnp_cdouble value;
        memcpy(&value, source, sizeof(value));
        value = unary_arctan_cdouble(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    if (source_dtype == CNP_CLONGDOUBLE) {
        cnp_clongdouble value;
        memcpy(&value, source, sizeof(value));
        value = unary_arctan_clongdouble(value);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }

    CnpArithmeticValue value;
    CNP_STATUS status = arithmetic_read_value(
        source, source_dtype, &value, "cnp_arctan");
    if (status != CNP_OK) return status;
    if (result_dtype == CNP_HALF) {
        uint16_t input_bits = cnp_float_to_half(
            arithmetic_value_double(&value));
        float input = (float)cnp_half_to_float(input_bits);
        volatile float result = atanf(input);
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float input = arithmetic_value_float(&value);
        volatile float result = atanf(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double input = arithmetic_value_double(&value);
        volatile double result = atan(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double input = arithmetic_value_longdouble(&value);
        volatile long double result = atanl(input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, "cnp_arctan",
        "result dtype %d does not support arctan", (int)result_dtype);
    return CNP_ERR_TYPE;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static CnpArray* unary_arctan_arrays(const CnpArray *source) {
    const char *function_name = "cnp_arctan";
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }

    CNP_TYPE source_dtype = source->dtype->type_num;
    CNP_TYPE result_dtype = unary_trigonometric_result_dtype(source_dtype);
    if (result_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support arctan",
            source->dtype->name);
        return NULL;
    }

    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);
    if (source->size == 0) return result;

    int source_itemsize = source->dtype->elsize;
    int result_itemsize = result->dtype->elsize;
    if (source->flags & (
            CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) {
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        for (int64_t index = 0; index < source->size; ++index) {
            CNP_STATUS status = unary_arctan_element(
                source_dtype, result_dtype,
                source_data + index * source_itemsize,
                result_data + index * result_itemsize);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                cnp_relabel_error(function_name);
                return NULL;
            }
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = unary_arctan_element(
            source_dtype, result_dtype,
            (const char*)source->data + source_offset,
            (char*)result->data + result_offset);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

static int real_binary_loop_rank(CNP_TYPE source_dtype) {
    switch (source_dtype) {
        case CNP_BOOL:
        case CNP_BYTE:
        case CNP_UBYTE:
        case CNP_HALF:
            return 0;
        case CNP_SHORT:
        case CNP_USHORT:
        case CNP_FLOAT:
            return 1;
        case CNP_INT:
        case CNP_UINT:
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
        case CNP_DOUBLE:
            return 2;
        case CNP_LONGDOUBLE:
            return 3;
        default:
            return -1;
    }
}

static CNP_TYPE real_binary_result_dtype(
    CNP_TYPE y_dtype, CNP_TYPE x_dtype) {
    int y_rank = real_binary_loop_rank(y_dtype);
    int x_rank = real_binary_loop_rank(x_dtype);
    if (y_rank < 0 || x_rank < 0) return CNP_NOTYPE;
    int result_rank = y_rank > x_rank ? y_rank : x_rank;
    switch (result_rank) {
        case 0:
            return CNP_HALF;
        case 1:
            return CNP_FLOAT;
        case 2:
            return CNP_DOUBLE;
        case 3:
            return CNP_LONGDOUBLE;
        default:
            return CNP_NOTYPE;
    }
}

static CNP_STATUS arctan2_validate_inputs(
    const CnpArray *y,
    const CnpArray *x,
    CNP_TYPE *result_dtype) {
    const char *function_name = "cnp_arctan2";
    if (!y || !x) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "y and x arrays are required");
        return CNP_ERR_GENERIC;
    }
    if (!y->dtype || !x->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "y and x arrays must have dtypes");
        return CNP_ERR_TYPE;
    }
    if (y->ndim < 0 || y->ndim > CNP_MAXDIMS ||
            x->ndim < 0 || x->ndim > CNP_MAXDIMS ||
            (y->ndim > 0 && (!y->shape || !y->strides)) ||
            (x->ndim > 0 && (!x->shape || !x->strides))) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "y and x arrays require valid shape metadata");
        return CNP_ERR_SHAPE;
    }
    if ((y->size > 0 && !y->data) ||
            (x->size > 0 && !x->data)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "y and x arrays require data buffers");
        return CNP_ERR_GENERIC;
    }
    CNP_TYPE resolved = real_binary_result_dtype(
        y->dtype->type_num, x->dtype->type_num);
    if (resolved == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "arctan2 does not support dtypes %s and %s",
            y->dtype->name, x->dtype->name);
        return CNP_ERR_TYPE;
    }
    if (!cnp_can_broadcast(y, x)) {
        cnp_set_error(
            CNP_ERR_BROADCAST, function_name,
            "y and x arrays cannot be broadcast together");
        return CNP_ERR_BROADCAST;
    }
    *result_dtype = resolved;
    return CNP_OK;
}

static CnpArray* arctan2_prepare_result(
    const CnpArray *y,
    const CnpArray *x,
    CNP_TYPE result_dtype) {
    const char *function_name = "cnp_arctan2";
    int result_ndim = y->ndim > x->ndim ? y->ndim : x->ndim;
    int64_t result_shape[CNP_MAXDIMS];
    for (int axis = 0; axis < result_ndim; ++axis) {
        int y_axis = axis - (result_ndim - y->ndim);
        int x_axis = axis - (result_ndim - x->ndim);
        int64_t y_dimension = y_axis < 0 ? 1 : y->shape[y_axis];
        int64_t x_dimension = x_axis < 0 ? 1 : x->shape[x_axis];
        result_shape[axis] = y_dimension == 1
            ? x_dimension : y_dimension;
    }
    CnpArray *result = cnp_array_new(
        result_ndim, result_shape, result_dtype,
        arithmetic_result_order(y, x));
    if (!result) cnp_relabel_error(function_name);
    return result;
}

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static CNP_STATUS arctan2_element(
    const void *y_source,
    CNP_TYPE y_dtype,
    const void *x_source,
    CNP_TYPE x_dtype,
    void *destination,
    CNP_TYPE result_dtype) {
    const char *function_name = "cnp_arctan2";
    CnpArithmeticValue y_value;
    CnpArithmeticValue x_value;
    CNP_STATUS status = arithmetic_read_value(
        y_source, y_dtype, &y_value, function_name);
    if (status != CNP_OK) return status;
    status = arithmetic_read_value(
        x_source, x_dtype, &x_value, function_name);
    if (status != CNP_OK) return status;

    if (result_dtype == CNP_HALF) {
        uint16_t y_bits = cnp_float_to_half(
            arithmetic_value_double(&y_value));
        uint16_t x_bits = cnp_float_to_half(
            arithmetic_value_double(&x_value));
        float y_input = (float)cnp_half_to_float(y_bits);
        float x_input = (float)cnp_half_to_float(x_bits);
        volatile float result = atan2f(y_input, x_input);
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float y_input = arithmetic_value_float(&y_value);
        float x_input = arithmetic_value_float(&x_value);
        volatile float result = atan2f(y_input, x_input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double y_input = arithmetic_value_double(&y_value);
        double x_input = arithmetic_value_double(&x_value);
        volatile double result = atan2(y_input, x_input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double y_input = arithmetic_value_longdouble(&y_value);
        long double x_input = arithmetic_value_longdouble(&x_value);
        volatile long double result = atan2l(y_input, x_input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "result dtype %d does not support arctan2",
        (int)result_dtype);
    return CNP_ERR_TYPE;
}

static bool arctan2_contiguous_typed(
    const CnpArray *y,
    const CnpArray *x,
    CnpArray *result) {
    bool c_contiguous =
        (y->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (x->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_C_CONTIGUOUS);
    bool f_contiguous =
        (y->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (x->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_F_CONTIGUOUS);
    if ((!c_contiguous && !f_contiguous) ||
            !same_shape(y, x) || !same_shape(y, result) ||
            y->dtype->type_num != result->dtype->type_num ||
            x->dtype->type_num != result->dtype->type_num)
        return false;
    if (result->size == 0) return true;

    const char *y_data = (const char*)y->data + y->offset;
    const char *x_data = (const char*)x->data + x->offset;
    char *result_data = (char*)result->data + result->offset;
    if (result->dtype->type_num == CNP_HALF) {
        for (int64_t index = 0; index < result->size; ++index) {
            uint16_t y_bits;
            uint16_t x_bits;
            memcpy(&y_bits, y_data + index * 2, sizeof(y_bits));
            memcpy(&x_bits, x_data + index * 2, sizeof(x_bits));
            float y_input = (float)cnp_half_to_float(y_bits);
            float x_input = (float)cnp_half_to_float(x_bits);
            volatile float value = atan2f(y_input, x_input);
            uint16_t result_bits = cnp_float_to_half((double)value);
            memcpy(
                result_data + index * 2,
                &result_bits, sizeof(result_bits));
        }
        return true;
    }
    if (result->dtype->type_num == CNP_FLOAT) {
        const float *y_values = (const float*)y_data;
        const float *x_values = (const float*)x_data;
        float *result_values = (float*)result_data;
        for (int64_t index = 0; index < result->size; ++index) {
            volatile float value = atan2f(
                y_values[index], x_values[index]);
            result_values[index] = value;
        }
        return true;
    }
    if (result->dtype->type_num == CNP_DOUBLE) {
        const double *y_values = (const double*)y_data;
        const double *x_values = (const double*)x_data;
        double *result_values = (double*)result_data;
        for (int64_t index = 0; index < result->size; ++index) {
            volatile double value = atan2(
                y_values[index], x_values[index]);
            result_values[index] = value;
        }
        return true;
    }
    if (result->dtype->type_num == CNP_LONGDOUBLE) {
        const long double *y_values = (const long double*)y_data;
        const long double *x_values = (const long double*)x_data;
        long double *result_values = (long double*)result_data;
        for (int64_t index = 0; index < result->size; ++index) {
            volatile long double value = atan2l(
                y_values[index], x_values[index]);
            result_values[index] = value;
        }
        return true;
    }
    return false;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static CnpArray* arctan2_arrays(
    const CnpArray *y, const CnpArray *x) {
    const char *function_name = "cnp_arctan2";
    CNP_TYPE result_dtype = CNP_NOTYPE;
    if (arctan2_validate_inputs(y, x, &result_dtype) != CNP_OK)
        return NULL;
    CnpArray *result = arctan2_prepare_result(y, x, result_dtype);
    if (!result) return NULL;
    if (arctan2_contiguous_typed(y, x, result)) return result;

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t y_offset = arithmetic_broadcast_offset(
            y, coordinates, result->ndim);
        int64_t x_offset = arithmetic_broadcast_offset(
            x, coordinates, result->ndim);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = arctan2_element(
            (const char*)y->data + y_offset,
            y->dtype->type_num,
            (const char*)x->data + x_offset,
            x->dtype->type_num,
            (char*)result->data + result_offset,
            result_dtype);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int axis = result->ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < result->shape[axis]) break;
            coordinates[axis] = 0;
        }
    }
    return result;
}

static CNP_STATUS hypot_validate_inputs(
    const CnpArray *x,
    const CnpArray *y,
    CNP_TYPE *result_dtype) {
    const char *function_name = "cnp_hypot";
    if (!x || !y) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "x and y arrays are required");
        return CNP_ERR_GENERIC;
    }
    if (!x->dtype || !y->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "x and y arrays must have dtypes");
        return CNP_ERR_TYPE;
    }
    if (x->ndim < 0 || x->ndim > CNP_MAXDIMS ||
            y->ndim < 0 || y->ndim > CNP_MAXDIMS ||
            (x->ndim > 0 && (!x->shape || !x->strides)) ||
            (y->ndim > 0 && (!y->shape || !y->strides))) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "x and y arrays require valid shape metadata");
        return CNP_ERR_SHAPE;
    }
    if ((x->size > 0 && !x->data) ||
            (y->size > 0 && !y->data)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "x and y arrays require data buffers");
        return CNP_ERR_GENERIC;
    }
    CNP_TYPE resolved = real_binary_result_dtype(
        x->dtype->type_num, y->dtype->type_num);
    if (resolved == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "hypot does not support dtypes %s and %s",
            x->dtype->name, y->dtype->name);
        return CNP_ERR_TYPE;
    }
    if (!cnp_can_broadcast(x, y)) {
        cnp_set_error(
            CNP_ERR_BROADCAST, function_name,
            "x and y arrays cannot be broadcast together");
        return CNP_ERR_BROADCAST;
    }
    *result_dtype = resolved;
    return CNP_OK;
}

static CnpArray* hypot_prepare_result(
    const CnpArray *x,
    const CnpArray *y,
    CNP_TYPE result_dtype) {
    const char *function_name = "cnp_hypot";
    int result_ndim = x->ndim > y->ndim ? x->ndim : y->ndim;
    int64_t result_shape[CNP_MAXDIMS];
    for (int axis = 0; axis < result_ndim; ++axis) {
        int x_axis = axis - (result_ndim - x->ndim);
        int y_axis = axis - (result_ndim - y->ndim);
        int64_t x_dimension = x_axis < 0 ? 1 : x->shape[x_axis];
        int64_t y_dimension = y_axis < 0 ? 1 : y->shape[y_axis];
        result_shape[axis] = x_dimension == 1
            ? y_dimension : x_dimension;
    }
    CnpArray *result = cnp_array_new(
        result_ndim, result_shape, result_dtype,
        arithmetic_result_order(x, y));
    if (!result) cnp_relabel_error(function_name);
    return result;
}

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static double hypot_numpy_double(double x, double y);

static float hypot_numpy_float(float x, float y) {
#ifdef _MSC_VER
    return (float)hypot_numpy_double((double)x, (double)y);
#else
    return hypotf(x, y);
#endif
}

static double hypot_numpy_double(double x, double y) {
    if (isinf(x) || isinf(y)) return INFINITY;
    if (isnan(x)) return x + x;
    if (isnan(y)) return y + y;
    x = fabs(x);
    y = fabs(y);
    if (x < y) {
        double temporary = x;
        x = y;
        y = temporary;
    }
    if (x == 0.0) return 0.0;
    if ((x >= DBL_MIN || x == 0.0) &&
            (y >= DBL_MIN || y == 0.0))
        return hypot(x, y);
    volatile double ratio = y / x;
    volatile double square = ratio * ratio;
    volatile double root = sqrt(1.0 + square);
    return x * root;
}

static long double hypot_numpy_longdouble(
    long double x, long double y) {
#ifdef _MSC_VER
    return (long double)hypot_numpy_double((double)x, (double)y);
#else
    if (isinf(x) || isinf(y)) return (long double)INFINITY;
    if (isnan(x)) return x + x;
    if (isnan(y)) return y + y;
    x = fabsl(x);
    y = fabsl(y);
    if (x < y) {
        long double temporary = x;
        x = y;
        y = temporary;
    }
    if (x == 0.0L) return 0.0L;
    if ((x >= LDBL_MIN || x == 0.0L) &&
            (y >= LDBL_MIN || y == 0.0L))
        return hypotl(x, y);
    volatile long double ratio = y / x;
    volatile long double square = ratio * ratio;
    volatile long double root = sqrtl(1.0L + square);
    return x * root;
#endif
}

static CNP_STATUS hypot_element(
    const void *x_source,
    CNP_TYPE x_dtype,
    const void *y_source,
    CNP_TYPE y_dtype,
    void *destination,
    CNP_TYPE result_dtype) {
    const char *function_name = "cnp_hypot";
    CnpArithmeticValue x_value;
    CnpArithmeticValue y_value;
    CNP_STATUS status = arithmetic_read_value(
        x_source, x_dtype, &x_value, function_name);
    if (status != CNP_OK) return status;
    status = arithmetic_read_value(
        y_source, y_dtype, &y_value, function_name);
    if (status != CNP_OK) return status;

    if (result_dtype == CNP_HALF) {
        uint16_t x_bits = cnp_float_to_half(
            arithmetic_value_double(&x_value));
        uint16_t y_bits = cnp_float_to_half(
            arithmetic_value_double(&y_value));
        float x_input = (float)cnp_half_to_float(x_bits);
        float y_input = (float)cnp_half_to_float(y_bits);
        volatile float result = hypot_numpy_float(x_input, y_input);
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float x_input = arithmetic_value_float(&x_value);
        float y_input = arithmetic_value_float(&y_value);
        volatile float result = hypot_numpy_float(x_input, y_input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double x_input = arithmetic_value_double(&x_value);
        double y_input = arithmetic_value_double(&y_value);
        volatile double result = hypot_numpy_double(x_input, y_input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double x_input = arithmetic_value_longdouble(&x_value);
        long double y_input = arithmetic_value_longdouble(&y_value);
        volatile long double result = hypot_numpy_longdouble(
            x_input, y_input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "result dtype %d does not support hypot",
        (int)result_dtype);
    return CNP_ERR_TYPE;
}

static bool hypot_contiguous_typed(
    const CnpArray *x,
    const CnpArray *y,
    CnpArray *result) {
    bool c_contiguous =
        (x->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (y->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_C_CONTIGUOUS);
    bool f_contiguous =
        (x->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (y->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_F_CONTIGUOUS);
    if ((!c_contiguous && !f_contiguous) ||
            !same_shape(x, y) || !same_shape(x, result) ||
            x->dtype->type_num != result->dtype->type_num ||
            y->dtype->type_num != result->dtype->type_num)
        return false;
    if (result->size == 0) return true;

    const char *x_data = (const char*)x->data + x->offset;
    const char *y_data = (const char*)y->data + y->offset;
    char *result_data = (char*)result->data + result->offset;
    if (result->dtype->type_num == CNP_HALF) {
        for (int64_t index = 0; index < result->size; ++index) {
            uint16_t x_bits;
            uint16_t y_bits;
            memcpy(&x_bits, x_data + index * 2, sizeof(x_bits));
            memcpy(&y_bits, y_data + index * 2, sizeof(y_bits));
            float x_input = (float)cnp_half_to_float(x_bits);
            float y_input = (float)cnp_half_to_float(y_bits);
            volatile float value = hypot_numpy_float(x_input, y_input);
            uint16_t result_bits = cnp_float_to_half((double)value);
            memcpy(
                result_data + index * 2,
                &result_bits, sizeof(result_bits));
        }
        return true;
    }
    if (result->dtype->type_num == CNP_FLOAT) {
        const float *x_values = (const float*)x_data;
        const float *y_values = (const float*)y_data;
        float *result_values = (float*)result_data;
        for (int64_t index = 0; index < result->size; ++index) {
            volatile float value = hypot_numpy_float(
                x_values[index], y_values[index]);
            result_values[index] = value;
        }
        return true;
    }
    if (result->dtype->type_num == CNP_DOUBLE) {
        const double *x_values = (const double*)x_data;
        const double *y_values = (const double*)y_data;
        double *result_values = (double*)result_data;
        for (int64_t index = 0; index < result->size; ++index) {
            volatile double value = hypot_numpy_double(
                x_values[index], y_values[index]);
            result_values[index] = value;
        }
        return true;
    }
    if (result->dtype->type_num == CNP_LONGDOUBLE) {
        const long double *x_values = (const long double*)x_data;
        const long double *y_values = (const long double*)y_data;
        long double *result_values = (long double*)result_data;
        for (int64_t index = 0; index < result->size; ++index) {
            volatile long double value = hypot_numpy_longdouble(
                x_values[index], y_values[index]);
            result_values[index] = value;
        }
        return true;
    }
    return false;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static CnpArray* hypot_arrays(
    const CnpArray *x, const CnpArray *y) {
    const char *function_name = "cnp_hypot";
    CNP_TYPE result_dtype = CNP_NOTYPE;
    if (hypot_validate_inputs(x, y, &result_dtype) != CNP_OK)
        return NULL;
    CnpArray *result = hypot_prepare_result(x, y, result_dtype);
    if (!result) return NULL;
    if (hypot_contiguous_typed(x, y, result)) return result;

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t x_offset = arithmetic_broadcast_offset(
            x, coordinates, result->ndim);
        int64_t y_offset = arithmetic_broadcast_offset(
            y, coordinates, result->ndim);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = hypot_element(
            (const char*)x->data + x_offset,
            x->dtype->type_num,
            (const char*)y->data + y_offset,
            y->dtype->type_num,
            (char*)result->data + result_offset,
            result_dtype);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int axis = result->ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < result->shape[axis]) break;
            coordinates[axis] = 0;
        }
    }
    return result;
}

typedef enum {
    CNP_ANGLE_TO_DEGREES = 0,
    CNP_ANGLE_TO_RADIANS = 1
} CnpAngleConversion;

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static float angle_conversion_float(
    float input, CnpAngleConversion conversion) {
    const float factor = conversion == CNP_ANGLE_TO_DEGREES
        ? 57.2957763671875f
        : 0.01745329238474369049f;
    volatile float result = input * factor;
    return result;
}

static double angle_conversion_double(
    double input, CnpAngleConversion conversion) {
    const double factor = conversion == CNP_ANGLE_TO_DEGREES
        ? 180.0 / 3.14159265358979323846
        : 3.14159265358979323846 / 180.0;
    volatile double result = input * factor;
    return result;
}

static long double angle_conversion_longdouble(
    long double input, CnpAngleConversion conversion) {
#ifdef _MSC_VER
    return (long double)angle_conversion_double(
        (double)input, conversion);
#else
    const long double factor = conversion == CNP_ANGLE_TO_DEGREES
        ? 180.0L / 3.141592653589793238462643383279502884L
        : 3.141592653589793238462643383279502884L / 180.0L;
    volatile long double result = input * factor;
    return result;
#endif
}

static CNP_STATUS angle_conversion_element(
    CNP_TYPE source_dtype,
    CNP_TYPE result_dtype,
    const void *source,
    void *destination,
    CnpAngleConversion conversion,
    const char *function_name) {
    CnpArithmeticValue value;
    CNP_STATUS status = arithmetic_read_value(
        source, source_dtype, &value, function_name);
    if (status != CNP_OK) return status;
    if (result_dtype == CNP_HALF) {
        uint16_t input_bits = cnp_float_to_half(
            arithmetic_value_double(&value));
        float input = (float)cnp_half_to_float(input_bits);
        float result = angle_conversion_float(input, conversion);
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float input = arithmetic_value_float(&value);
        float result = angle_conversion_float(input, conversion);
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double input = arithmetic_value_double(&value);
        double result = angle_conversion_double(input, conversion);
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double input = arithmetic_value_longdouble(&value);
        long double result = angle_conversion_longdouble(
            input, conversion);
        memcpy(destination, &result, sizeof(result));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "result dtype %d does not support angle conversion",
        (int)result_dtype);
    return CNP_ERR_TYPE;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static CnpArray* angle_conversion_arrays(
    const CnpArray *source,
    CnpAngleConversion conversion,
    const char *function_name,
    const char *operation_name) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->ndim < 0 || source->ndim > CNP_MAXDIMS ||
            (source->ndim > 0 && (!source->shape || !source->strides))) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source array requires valid shape metadata");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }

    CNP_TYPE source_dtype = source->dtype->type_num;
    CNP_TYPE result_dtype = real_binary_result_dtype(
        source_dtype, source_dtype);
    if (result_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support %s",
            source->dtype->name, operation_name);
        return NULL;
    }

    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);
    if (source->size == 0) return result;

    int source_itemsize = source->dtype->elsize;
    int result_itemsize = result->dtype->elsize;
    if (source->flags & (
            CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) {
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        for (int64_t index = 0; index < source->size; ++index) {
            CNP_STATUS status = angle_conversion_element(
                source_dtype, result_dtype,
                source_data + index * source_itemsize,
                result_data + index * result_itemsize,
                conversion, function_name);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                cnp_relabel_error(function_name);
                return NULL;
            }
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = angle_conversion_element(
            source_dtype, result_dtype,
            (const char*)source->data + source_offset,
            (char*)result->data + result_offset,
            conversion, function_name);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_negative(const CnpArray *a) {
    return unary_sign_arrays(
        a, CNP_UNARY_SIGN_NEGATIVE, "cnp_negative", "negative");
}

CNP_API CnpArray* CNP_CALL cnp_positive(const CnpArray *a) {
    return unary_sign_arrays(
        a, CNP_UNARY_SIGN_POSITIVE, "cnp_positive", "positive");
}

CNP_API CnpArray* CNP_CALL cnp_absolute(const CnpArray *a) {
    if (a && (a->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            a->dtype->type_num == CNP_DOUBLE) {
        CnpArray *result = cnp_array_new(
            a->ndim, a->shape, CNP_DOUBLE, CNP_ORDER_C);
        if (!result) {
            cnp_relabel_error("cnp_absolute");
            return NULL;
        }
        cnp_simd_absolute(
            (const double*)((const char*)a->data + a->offset),
            (double*)result->data, a->size);
        return result;
    }
    return cnp_unary_op_absolute(a, false, "cnp_absolute");
}

CNP_API CnpArray* CNP_CALL cnp_fabs(const CnpArray *a) {
    if (a && (a->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            a->dtype->type_num == CNP_DOUBLE) {
        CnpArray *result = cnp_array_new(
            a->ndim, a->shape, CNP_DOUBLE, CNP_ORDER_C);
        if (!result) {
            cnp_relabel_error("cnp_fabs");
            return NULL;
        }
        cnp_simd_absolute(
            (const double*)((const char*)a->data + a->offset),
            (double*)result->data, a->size);
        return result;
    }
    return cnp_unary_op_absolute(a, true, "cnp_fabs");
}

CNP_API CnpArray* CNP_CALL cnp_sign(const CnpArray *a) {
    return unary_signum_arrays(a);
}

CNP_API CnpArray* CNP_CALL cnp_reciprocal(const CnpArray *a) {
    return unary_reciprocal_arrays(a);
}

CNP_API CnpArray* CNP_CALL cnp_conj(const CnpArray *a) {
    return unary_conjugate_arrays(a, "cnp_conj");
}

CNP_API CnpArray* CNP_CALL cnp_conjugate(const CnpArray *a) {
    return unary_conjugate_arrays(a, "cnp_conjugate");
}

CNP_API CnpArray* CNP_CALL cnp_sqrt(const CnpArray *a) {
    return unary_sqrt_arrays(a);
}

CNP_API CnpArray* CNP_CALL cnp_cbrt(const CnpArray *a) {
    return unary_cbrt_arrays(a);
}

CNP_API CnpArray* CNP_CALL cnp_square(const CnpArray *a) {
    return unary_square_arrays(a);
}

/* =========================================================================
 * Rounding operations
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_ceil(const CnpArray *a) {
    return cnp_unary_op_rounding(
        a, CNP_UNARY_ROUND_CEIL, "cnp_ceil");
}

CNP_API CnpArray* CNP_CALL cnp_floor(const CnpArray *a) {
    if (a && (a->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            a->dtype->type_num == CNP_DOUBLE) {
        CnpArray *result = cnp_array_new(
            a->ndim, a->shape, CNP_DOUBLE, CNP_ORDER_C);
        if (!result) {
            cnp_relabel_error("cnp_floor");
            return NULL;
        }
        cnp_simd_floor(
            (const double*)((const char*)a->data + a->offset),
            (double*)result->data, a->size);
        return result;
    }
    return cnp_unary_op_rounding(
        a, CNP_UNARY_ROUND_FLOOR, "cnp_floor");
}

CNP_API CnpArray* CNP_CALL cnp_trunc(const CnpArray *a) {
    return cnp_unary_op_rounding(
        a, CNP_UNARY_ROUND_TRUNCATE, "cnp_trunc");
}

CNP_API CnpArray* CNP_CALL cnp_rint(const CnpArray *a) {
    return cnp_unary_op_rounding(
        a, CNP_UNARY_ROUND_RINT, "cnp_rint");
}

CNP_API CnpArray* CNP_CALL cnp_round(const CnpArray *a, int decimals) {
    CnpArray *result = cnp_around(a, decimals);
    if (!result) cnp_relabel_error("cnp_round");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_fix(const CnpArray *a) {
    return cnp_unary_op_rounding(
        a, CNP_UNARY_ROUND_TRUNCATE, "cnp_fix");
}

/* =========================================================================
 * Trigonometric functions
 * ========================================================================= */

CNP_API CnpArray* CNP_CALL cnp_sin(const CnpArray *a) {
    return unary_sin_arrays(a);
}
CNP_API CnpArray* CNP_CALL cnp_cos(const CnpArray *a) {
    return unary_cos_arrays(a);
}
CNP_API CnpArray* CNP_CALL cnp_tan(const CnpArray *a) {
    return unary_tan_arrays(a);
}
CNP_API CnpArray* CNP_CALL cnp_arcsin(const CnpArray *a) {
    return unary_arcsin_arrays(a);
}
CNP_API CnpArray* CNP_CALL cnp_arccos(const CnpArray *a) {
    return unary_arccos_arrays(a);
}
CNP_API CnpArray* CNP_CALL cnp_arctan(const CnpArray *a) {
    return unary_arctan_arrays(a);
}

CNP_API CnpArray* CNP_CALL cnp_arctan2(const CnpArray *y, const CnpArray *x) {
    return arctan2_arrays(y, x);
}

CNP_API CnpArray* CNP_CALL cnp_hypot(const CnpArray *x, const CnpArray *y) {
    return hypot_arrays(x, y);
}

CNP_API CnpArray* CNP_CALL cnp_degrees(const CnpArray *a) {
    return angle_conversion_arrays(
        a, CNP_ANGLE_TO_DEGREES, "cnp_degrees", "degrees");
}

CNP_API CnpArray* CNP_CALL cnp_radians(const CnpArray *a) {
    return angle_conversion_arrays(
        a, CNP_ANGLE_TO_RADIANS, "cnp_radians", "radians");
}

CNP_API CnpArray* CNP_CALL cnp_deg2rad(const CnpArray *a) {
    return angle_conversion_arrays(
        a, CNP_ANGLE_TO_RADIANS, "cnp_deg2rad", "deg2rad");
}

CNP_API CnpArray* CNP_CALL cnp_rad2deg(const CnpArray *a) {
    return angle_conversion_arrays(
        a, CNP_ANGLE_TO_DEGREES, "cnp_rad2deg", "rad2deg");
}

/* =========================================================================
 * Hyperbolic functions
 * ========================================================================= */
typedef enum {
    CNP_HYPERBOLIC_SINH = 0,
    CNP_HYPERBOLIC_COSH = 1,
    CNP_HYPERBOLIC_TANH = 2,
    CNP_HYPERBOLIC_ARCSINH = 3,
    CNP_HYPERBOLIC_ARCCOSH = 4,
    CNP_HYPERBOLIC_ARCTANH = 5,
    CNP_TRANSCENDENTAL_EXP = 6,
    CNP_TRANSCENDENTAL_EXP2 = 7,
    CNP_TRANSCENDENTAL_EXPM1 = 8,
    CNP_TRANSCENDENTAL_LOG = 9,
    CNP_TRANSCENDENTAL_LOG2 = 10,
    CNP_TRANSCENDENTAL_LOG10 = 11,
    CNP_TRANSCENDENTAL_LOG1P = 12
} CnpTranscendentalOperation;

static CNP_TYPE unary_transcendental_result_dtype(CNP_TYPE source_dtype) {
    if (source_dtype == CNP_CFLOAT) return CNP_CFLOAT;
    if (source_dtype == CNP_CDOUBLE) return CNP_CDOUBLE;
    if (source_dtype == CNP_CLONGDOUBLE) return CNP_CLONGDOUBLE;
    if (cnp_type_is_complex(source_dtype)) return CNP_NOTYPE;
    return unary_trigonometric_result_dtype(source_dtype);
}

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static float unary_exp_numpy_float(float source) {
    static const float xmax = 88.72283935546875f;
    static const float xmin = -103.97208404541015625f;
    static const float log2e =
        1.442695040888963407359924681001892137f;
    static const float rounding_magic = 12582912.0f;
    static const float cody_waite_high = -6.93145752e-1f;
    static const float cody_waite_low = -1.42860677e-6f;
    static const float p0 = 9.999999999980870924916e-01f;
    static const float p1 = 7.257664613233124478488e-01f;
    static const float p2 = 2.473615434895520810817e-01f;
    static const float p3 = 5.114512081637298353406e-02f;
    static const float p4 = 6.757896990527504603057e-03f;
    static const float p5 = 5.082762527590693718096e-04f;
    static const float q0 = 1.000000000000000000000e+00f;
    static const float q1 = -2.742335390411667452936e-01f;
    static const float q2 = 2.159509375685829852307e-02f;

    if (isnan(source)) {
        uint32_t nan_bits = 0x7fc00000u;
        float result;
        memcpy(&result, &nan_bits, sizeof(result));
        return result;
    }
    if (source >= xmax) return INFINITY;
    if (source <= xmin) return 0.0f;

    /* NumPy 1.25 AVX2/FMA exp range reduction and Remez kernel. */
    volatile float quadrant = source * log2e;
    quadrant = quadrant + rounding_magic;
    quadrant = quadrant - rounding_magic;

    float reduced = fmaf(quadrant, cody_waite_high, source);
    reduced = fmaf(quadrant, cody_waite_low, reduced);

    float numerator = fmaf(p5, reduced, p4);
    numerator = fmaf(numerator, reduced, p3);
    numerator = fmaf(numerator, reduced, p2);
    numerator = fmaf(numerator, reduced, p1);
    numerator = fmaf(numerator, reduced, p0);
    float denominator = fmaf(q2, reduced, q1);
    denominator = fmaf(denominator, reduced, q0);
    volatile float polynomial = numerator / denominator;

    int32_t quadrant_integer = (int32_t)quadrant;
    uint32_t polynomial_bits;
    memcpy(
        &polynomial_bits, (const void*)&polynomial,
        sizeof(polynomial_bits));
    if (quadrant_integer <= -125) {
        uint32_t power_difference =
            1u << (uint32_t)(-quadrant_integer - 125);
        polynomial_bits += (uint32_t)(-125 * 0x00800000);
        memcpy(
            (void*)&polynomial, &polynomial_bits,
            sizeof(polynomial));
        polynomial = polynomial / (float)power_difference;
        return polynomial;
    }

    polynomial_bits +=
        (uint32_t)(quadrant_integer * 0x00800000);
    memcpy((void*)&polynomial, &polynomial_bits, sizeof(polynomial));
    return polynomial;
}

static float unary_log_numpy_float(float source) {
    static const float sqrt_half =
        0.707106781186547524400844362104849039f;
    static const float log_two =
        0.693147180559945309417232121458176568f;
    static const float p0 = 0.000000000000000000000e+00f;
    static const float p1 = 9.999999999999998702752e-01f;
    static const float p2 = 2.112677543073053063722e+00f;
    static const float p3 = 1.480000633576506585156e+00f;
    static const float p4 = 3.808837741388407920751e-01f;
    static const float p5 = 2.589979117907922693523e-02f;
    static const float q0 = 1.000000000000000000000e+00f;
    static const float q1 = 2.612677543073109236779e+00f;
    static const float q2 = 2.453006071784736363091e+00f;
    static const float q3 = 9.864942958519418960339e-01f;
    static const float q4 = 1.546476374983906719538e-01f;
    static const float q5 = 5.875095403124574342950e-03f;

    if (isnan(source)) {
        uint32_t nan_bits = 0x7fc00000u;
        float result;
        memcpy(&result, &nan_bits, sizeof(result));
        return result;
    }
    if (source < 0.0f) {
        uint32_t nan_bits = 0xffc00000u;
        float result;
        memcpy(&result, &nan_bits, sizeof(result));
        return result;
    }
    if (source == 0.0f) return -INFINITY;
    if (isinf(source)) return INFINITY;

    bool denormal = source < FLT_MIN;
    volatile float normalized = source;
    if (denormal) normalized = normalized * 0x1p100f;

    uint32_t normalized_bits;
    memcpy(
        &normalized_bits, (const void*)&normalized,
        sizeof(normalized_bits));
    volatile float exponent =
        (float)((int32_t)(normalized_bits >> 23) - 0x7e);
    if (denormal) exponent = exponent - 100.0f;

    uint32_t mantissa_bits =
        (normalized_bits & 0x007fffffu) | (126u << 23);
    volatile float mantissa;
    memcpy((void*)&mantissa, &mantissa_bits, sizeof(mantissa));
    if (mantissa <= sqrt_half) {
        mantissa = mantissa + mantissa;
        exponent = exponent - 1.0f;
    }
    volatile float reduced = mantissa - 1.0f;

    float numerator = fmaf(p5, reduced, p4);
    numerator = fmaf(numerator, reduced, p3);
    numerator = fmaf(numerator, reduced, p2);
    numerator = fmaf(numerator, reduced, p1);
    numerator = fmaf(numerator, reduced, p0);
    float denominator = fmaf(q5, reduced, q4);
    denominator = fmaf(denominator, reduced, q3);
    denominator = fmaf(denominator, reduced, q2);
    denominator = fmaf(denominator, reduced, q1);
    denominator = fmaf(denominator, reduced, q0);
    volatile float polynomial = numerator / denominator;
    return fmaf(exponent, log_two, polynomial);
}

static cnp_cfloat unary_log_numpy_cfloat(cnp_cfloat source) {
    static const float log_two =
        0.693147180559945309417232121458176568f;
    float absolute_real = fabsf(source.real);
    float absolute_imaginary = fabsf(source.imag);
    volatile float real;

    if (absolute_real > FLT_MAX / 4.0f ||
            absolute_imaginary > FLT_MAX / 4.0f) {
        real = logf(hypot_numpy_float(
            absolute_real / 2.0f,
            absolute_imaginary / 2.0f)) + log_two;
    } else if (absolute_real < FLT_MIN &&
            absolute_imaginary < FLT_MIN) {
        if (absolute_real > 0.0f || absolute_imaginary > 0.0f) {
            volatile float scaled_real =
                ldexpf(absolute_real, FLT_MANT_DIG);
            volatile float scaled_imaginary =
                ldexpf(absolute_imaginary, FLT_MANT_DIG);
            real = logf(hypot_numpy_float(
                scaled_real, scaled_imaginary)) -
                FLT_MANT_DIG * log_two;
        } else {
            real = -1.0f / source.real;
            real = copysignf(real, -1.0f);
            cnp_cfloat result = {
                real, atan2f(source.imag, source.real)};
            return result;
        }
    } else {
        volatile float magnitude =
            hypot_numpy_float(absolute_real, absolute_imaginary);
        if (0.71f <= magnitude && magnitude <= 1.73f) {
            volatile float maximum = absolute_real > absolute_imaginary
                ? absolute_real : absolute_imaginary;
            volatile float minimum = absolute_real > absolute_imaginary
                ? absolute_imaginary : absolute_real;
            volatile float left = (maximum - 1.0f) * (maximum + 1.0f);
            volatile float right = minimum * minimum;
            real = log1pf(left + right) / 2.0f;
        } else {
            real = logf(magnitude);
        }
    }
    cnp_cfloat result = {
        real, atan2f(source.imag, source.real)};
    return result;
}

static cnp_cfloat unary_log1p_numpy_cfloat(cnp_cfloat source) {
    float shifted_real = source.real + 1.0f;
    float magnitude = hypot_numpy_float(shifted_real, source.imag);
    cnp_cfloat result = {
        logf(magnitude), atan2f(source.imag, shifted_real)};
    return result;
}

static cnp_cdouble unary_log_numpy_cdouble(cnp_cdouble source) {
    static const double log_two =
        0.693147180559945309417232121458176568;
    double absolute_real = fabs(source.real);
    double absolute_imaginary = fabs(source.imag);
    volatile double real;

    if (absolute_real > DBL_MAX / 4.0 ||
            absolute_imaginary > DBL_MAX / 4.0) {
        real = log(hypot_numpy_double(
            absolute_real / 2.0,
            absolute_imaginary / 2.0)) + log_two;
    } else if (absolute_real < DBL_MIN &&
            absolute_imaginary < DBL_MIN) {
        if (absolute_real > 0.0 || absolute_imaginary > 0.0) {
            volatile double scaled_real =
                ldexp(absolute_real, DBL_MANT_DIG);
            volatile double scaled_imaginary =
                ldexp(absolute_imaginary, DBL_MANT_DIG);
            real = log(hypot_numpy_double(
                scaled_real, scaled_imaginary)) -
                DBL_MANT_DIG * log_two;
        } else {
            real = -1.0 / source.real;
            real = copysign(real, -1.0);
            cnp_cdouble result = {
                real, atan2(source.imag, source.real)};
            return result;
        }
    } else {
        volatile double magnitude =
            hypot_numpy_double(absolute_real, absolute_imaginary);
        if (0.71 <= magnitude && magnitude <= 1.73) {
            volatile double maximum = absolute_real > absolute_imaginary
                ? absolute_real : absolute_imaginary;
            volatile double minimum = absolute_real > absolute_imaginary
                ? absolute_imaginary : absolute_real;
            volatile double left = (maximum - 1.0) * (maximum + 1.0);
            volatile double right = minimum * minimum;
            real = log1p(left + right) / 2.0;
        } else {
            real = log(magnitude);
        }
    }
    cnp_cdouble result = {
        real, atan2(source.imag, source.real)};
    return result;
}

static cnp_cdouble unary_log1p_numpy_cdouble(cnp_cdouble source) {
    double shifted_real = source.real + 1.0;
    double magnitude = hypot_numpy_double(shifted_real, source.imag);
    cnp_cdouble result = {
        log(magnitude), atan2(source.imag, shifted_real)};
    return result;
}

static cnp_clongdouble unary_log_numpy_clongdouble(
    cnp_clongdouble source) {
    static const long double log_two =
        0.693147180559945309417232121458176568L;
    long double absolute_real = fabsl(source.real);
    long double absolute_imaginary = fabsl(source.imag);
    volatile long double real;

    if (absolute_real > LDBL_MAX / 4.0L ||
            absolute_imaginary > LDBL_MAX / 4.0L) {
        real = logl(hypot_numpy_longdouble(
            absolute_real / 2.0L,
            absolute_imaginary / 2.0L)) + log_two;
    } else if (absolute_real < LDBL_MIN &&
            absolute_imaginary < LDBL_MIN) {
        if (absolute_real > 0.0L || absolute_imaginary > 0.0L) {
            volatile long double scaled_real =
                ldexpl(absolute_real, LDBL_MANT_DIG);
            volatile long double scaled_imaginary =
                ldexpl(absolute_imaginary, LDBL_MANT_DIG);
            real = logl(hypot_numpy_longdouble(
                scaled_real, scaled_imaginary)) -
                LDBL_MANT_DIG * log_two;
        } else {
            real = -1.0L / source.real;
            real = copysignl(real, -1.0L);
            cnp_clongdouble result = {
                real, atan2l(source.imag, source.real)};
            return result;
        }
    } else {
        volatile long double magnitude =
            hypot_numpy_longdouble(absolute_real, absolute_imaginary);
        if (0.71L <= magnitude && magnitude <= 1.73L) {
            volatile long double maximum =
                absolute_real > absolute_imaginary
                ? absolute_real : absolute_imaginary;
            volatile long double minimum =
                absolute_real > absolute_imaginary
                ? absolute_imaginary : absolute_real;
            volatile long double left =
                (maximum - 1.0L) * (maximum + 1.0L);
            volatile long double right = minimum * minimum;
            real = log1pl(left + right) / 2.0L;
        } else {
            real = logl(magnitude);
        }
    }
    cnp_clongdouble result = {
        real, atan2l(source.imag, source.real)};
    return result;
}

static cnp_clongdouble unary_log1p_numpy_clongdouble(
    cnp_clongdouble source) {
    long double shifted_real = source.real + 1.0L;
    long double magnitude = hypot_numpy_longdouble(
        shifted_real, source.imag);
    cnp_clongdouble result = {
        logl(magnitude), atan2l(source.imag, shifted_real)};
    return result;
}

static cnp_cfloat unary_transcendental_cfloat(
    cnp_cfloat source, CnpTranscendentalOperation operation) {
    if (operation == CNP_HYPERBOLIC_ARCSINH)
        return unary_arcsin_numpy_asinh_float(source);
    if (operation == CNP_HYPERBOLIC_ARCTANH)
        return unary_arctan_numpy_atanh_float(source);
    if (operation == CNP_TRANSCENDENTAL_EXPM1) {
        volatile float half_imaginary = source.imag / 2.0f;
        volatile float sine_half = sinf(half_imaginary);
        volatile float real_left =
            expm1f(source.real) * cosf(source.imag);
        volatile float real_right = 2.0f * sine_half * sine_half;
        cnp_cfloat result = {
            real_left - real_right,
            expf(source.real) * sinf(source.imag)};
        return result;
    }
    if (operation == CNP_TRANSCENDENTAL_EXP2) {
        cnp_cfloat scaled = {
            source.real * 0.693147180559945309417232121458176568f,
            source.imag * 0.693147180559945309417232121458176568f};
        return unary_transcendental_cfloat(
            scaled, CNP_TRANSCENDENTAL_EXP);
    }
    if (operation == CNP_TRANSCENDENTAL_EXP) {
#ifdef _MSC_VER
        _Fcomplex native_source;
        _Fcomplex native_result;
        cnp_cfloat result;
        memcpy(&native_source, &source, sizeof(native_source));
        native_result = cexpf(native_source);
        memcpy(&result, &native_result, sizeof(result));
#else
        float complex native_result =
            cexpf(CMPLXF(source.real, source.imag));
        cnp_cfloat result = {
            crealf(native_result), cimagf(native_result)};
#endif
        return result;
    }
    if (operation == CNP_TRANSCENDENTAL_LOG) {
        return unary_log_numpy_cfloat(source);
    }
    if (operation == CNP_TRANSCENDENTAL_LOG2) {
        cnp_cfloat result = unary_log_numpy_cfloat(source);
        result.real *= 1.442695040888963407359924681001892137f;
        result.imag *= 1.442695040888963407359924681001892137f;
        return result;
    }
    if (operation == CNP_TRANSCENDENTAL_LOG10) {
        cnp_cfloat result = unary_log_numpy_cfloat(source);
        result.real *= 0.434294481903251827651128918916605082f;
        result.imag *= 0.434294481903251827651128918916605082f;
        return result;
    }
    if (operation == CNP_TRANSCENDENTAL_LOG1P) {
        return unary_log1p_numpy_cfloat(source);
    }
    if (operation == CNP_HYPERBOLIC_ARCCOSH) {
        cnp_cfloat arccos_value = unary_arccos_numpy_float(source);
        if (isnan(arccos_value.real) && isnan(arccos_value.imag)) {
            cnp_cfloat result = {
                arccos_value.imag, arccos_value.real};
            return result;
        }
        if (isnan(arccos_value.real)) {
            cnp_cfloat result = {
                fabsf(arccos_value.imag), arccos_value.real};
            return result;
        }
        if (isnan(arccos_value.imag)) {
            cnp_cfloat result = {
                arccos_value.imag, arccos_value.imag};
            return result;
        }
        cnp_cfloat result = {
            fabsf(arccos_value.imag),
            copysignf(arccos_value.real, source.imag)};
        return result;
    }
#ifdef _MSC_VER
    _Fcomplex native_source;
    _Fcomplex native_result;
    cnp_cfloat result;
    memcpy(&native_source, &source, sizeof(native_source));
    if (operation == CNP_HYPERBOLIC_SINH)
        native_result = csinhf(native_source);
    else if (operation == CNP_HYPERBOLIC_COSH)
        native_result = ccoshf(native_source);
    else
        native_result = ctanhf(native_source);
    memcpy(&result, &native_result, sizeof(result));
#else
    float complex native_source =
        CMPLXF(source.real, source.imag);
    float complex native_result;
    if (operation == CNP_HYPERBOLIC_SINH)
        native_result = csinhf(native_source);
    else if (operation == CNP_HYPERBOLIC_COSH)
        native_result = ccoshf(native_source);
    else
        native_result = ctanhf(native_source);
    cnp_cfloat result = {
        crealf(native_result), cimagf(native_result)};
#endif
    return result;
}

static cnp_cdouble unary_transcendental_cdouble(
    cnp_cdouble source, CnpTranscendentalOperation operation) {
    if (operation == CNP_HYPERBOLIC_ARCSINH)
        return unary_arcsin_numpy_asinh_double(source);
    if (operation == CNP_HYPERBOLIC_ARCTANH)
        return unary_arctan_numpy_atanh_double(source);
    if (operation == CNP_TRANSCENDENTAL_EXPM1) {
        volatile double half_imaginary = source.imag / 2.0;
        volatile double sine_half = sin(half_imaginary);
        volatile double real_left =
            expm1(source.real) * cos(source.imag);
        volatile double real_right = 2.0 * sine_half * sine_half;
        volatile double exponential = exp(source.real);
        volatile double imaginary_sine = sin(source.imag);
        double imaginary = isnan(exponential) && isnan(imaginary_sine)
            ? imaginary_sine : exponential * imaginary_sine;
        cnp_cdouble result = {
            real_left - real_right,
            imaginary};
        return result;
    }
    if (operation == CNP_TRANSCENDENTAL_EXP2) {
        cnp_cdouble scaled = {
            source.real * 0.693147180559945309417232121458176568,
            source.imag * 0.693147180559945309417232121458176568};
        return unary_transcendental_cdouble(
            scaled, CNP_TRANSCENDENTAL_EXP);
    }
    if (operation == CNP_TRANSCENDENTAL_EXP) {
#ifdef _MSC_VER
        _Dcomplex native_source;
        _Dcomplex native_result;
        cnp_cdouble result;
        memcpy(&native_source, &source, sizeof(native_source));
        native_result = cexp(native_source);
        memcpy(&result, &native_result, sizeof(result));
#else
        double complex native_result =
            cexp(CMPLX(source.real, source.imag));
        cnp_cdouble result = {
            creal(native_result), cimag(native_result)};
#endif
        return result;
    }
    if (operation == CNP_TRANSCENDENTAL_LOG) {
        return unary_log_numpy_cdouble(source);
    }
    if (operation == CNP_TRANSCENDENTAL_LOG2) {
        cnp_cdouble result = unary_log_numpy_cdouble(source);
        result.real *= 1.442695040888963407359924681001892137;
        result.imag *= 1.442695040888963407359924681001892137;
        return result;
    }
    if (operation == CNP_TRANSCENDENTAL_LOG10) {
        cnp_cdouble result = unary_log_numpy_cdouble(source);
        result.real *= 0.434294481903251827651128918916605082;
        result.imag *= 0.434294481903251827651128918916605082;
        return result;
    }
    if (operation == CNP_TRANSCENDENTAL_LOG1P) {
        return unary_log1p_numpy_cdouble(source);
    }
    if (operation == CNP_HYPERBOLIC_ARCCOSH) {
        cnp_cdouble arccos_value = unary_arccos_numpy_double(source);
        if (isnan(arccos_value.real) && isnan(arccos_value.imag)) {
            cnp_cdouble result = {
                arccos_value.imag, arccos_value.real};
            return result;
        }
        if (isnan(arccos_value.real)) {
            cnp_cdouble result = {
                fabs(arccos_value.imag), arccos_value.real};
            return result;
        }
        if (isnan(arccos_value.imag)) {
            cnp_cdouble result = {
                arccos_value.imag, arccos_value.imag};
            return result;
        }
        cnp_cdouble result = {
            fabs(arccos_value.imag),
            copysign(arccos_value.real, source.imag)};
        return result;
    }
#ifdef _MSC_VER
    _Dcomplex native_source;
    _Dcomplex native_result;
    cnp_cdouble result;
    memcpy(&native_source, &source, sizeof(native_source));
    if (operation == CNP_HYPERBOLIC_SINH)
        native_result = csinh(native_source);
    else if (operation == CNP_HYPERBOLIC_COSH)
        native_result = ccosh(native_source);
    else
        native_result = ctanh(native_source);
    memcpy(&result, &native_result, sizeof(result));
#else
    double complex native_source =
        CMPLX(source.real, source.imag);
    double complex native_result;
    if (operation == CNP_HYPERBOLIC_SINH)
        native_result = csinh(native_source);
    else if (operation == CNP_HYPERBOLIC_COSH)
        native_result = ccosh(native_source);
    else
        native_result = ctanh(native_source);
    cnp_cdouble result = {
        creal(native_result), cimag(native_result)};
#endif
    return result;
}

static cnp_clongdouble unary_transcendental_clongdouble(
    cnp_clongdouble source, CnpTranscendentalOperation operation) {
    if (operation == CNP_HYPERBOLIC_ARCSINH) {
#ifdef _MSC_VER
        cnp_cdouble narrowed = {(double)source.real, (double)source.imag};
        cnp_cdouble value = unary_arcsin_numpy_asinh_double(narrowed);
        cnp_clongdouble result = {
            (long double)value.real, (long double)value.imag};
#else
        long double complex native_result =
            casinhl(CMPLXL(source.real, source.imag));
        cnp_clongdouble result = {
            creall(native_result), cimagl(native_result)};
#endif
        return result;
    }
    if (operation == CNP_HYPERBOLIC_ARCTANH) {
#ifdef _MSC_VER
        cnp_cdouble narrowed = {(double)source.real, (double)source.imag};
        cnp_cdouble value = unary_arctan_numpy_atanh_double(narrowed);
        cnp_clongdouble result = {
            (long double)value.real, (long double)value.imag};
#else
        long double complex native_result =
            catanhl(CMPLXL(source.real, source.imag));
        cnp_clongdouble result = {
            creall(native_result), cimagl(native_result)};
#endif
        return result;
    }
    if (operation == CNP_TRANSCENDENTAL_EXPM1) {
        volatile long double half_imaginary = source.imag / 2.0L;
        volatile long double sine_half = sinl(half_imaginary);
        volatile long double real_left =
            expm1l(source.real) * cosl(source.imag);
        volatile long double real_right =
            2.0L * sine_half * sine_half;
        volatile long double exponential = expl(source.real);
        volatile long double imaginary_sine = sinl(source.imag);
        long double imaginary =
            isnan(exponential) && isnan(imaginary_sine)
            ? exponential : exponential * imaginary_sine;
        cnp_clongdouble result = {
            real_left - real_right,
            imaginary};
        return result;
    }
    if (operation == CNP_TRANSCENDENTAL_EXP2) {
        cnp_clongdouble scaled = {
            source.real *
                0.693147180559945309417232121458176568L,
            source.imag *
                0.693147180559945309417232121458176568L};
        return unary_transcendental_clongdouble(
            scaled, CNP_TRANSCENDENTAL_EXP);
    }
    if (operation == CNP_TRANSCENDENTAL_EXP) {
#ifdef _MSC_VER
        cnp_cdouble narrowed = {(double)source.real, (double)source.imag};
        cnp_cdouble value = unary_transcendental_cdouble(
            narrowed, CNP_TRANSCENDENTAL_EXP);
        cnp_clongdouble result = {
            (long double)value.real, (long double)value.imag};
#else
        long double complex native_result =
            cexpl(CMPLXL(source.real, source.imag));
        cnp_clongdouble result = {
            creall(native_result), cimagl(native_result)};
#endif
        return result;
    }
    if (operation == CNP_TRANSCENDENTAL_LOG) {
        return unary_log_numpy_clongdouble(source);
    }
    if (operation == CNP_TRANSCENDENTAL_LOG2) {
        cnp_clongdouble result = unary_log_numpy_clongdouble(source);
        result.real *= 1.442695040888963407359924681001892137L;
        result.imag *= 1.442695040888963407359924681001892137L;
        return result;
    }
    if (operation == CNP_TRANSCENDENTAL_LOG10) {
        cnp_clongdouble result = unary_log_numpy_clongdouble(source);
        result.real *= 0.434294481903251827651128918916605082L;
        result.imag *= 0.434294481903251827651128918916605082L;
        return result;
    }
    if (operation == CNP_TRANSCENDENTAL_LOG1P) {
        return unary_log1p_numpy_clongdouble(source);
    }
    if (operation == CNP_HYPERBOLIC_ARCCOSH) {
#ifdef _MSC_VER
        cnp_cdouble narrowed = {(double)source.real, (double)source.imag};
        cnp_cdouble value = unary_transcendental_cdouble(
            narrowed, CNP_HYPERBOLIC_ARCCOSH);
        cnp_clongdouble result = {
            (long double)value.real, (long double)value.imag};
#else
        cnp_clongdouble arccos_value =
            unary_arccos_clongdouble(source);
        cnp_clongdouble result;
        if (isnan(arccos_value.real) && isnan(arccos_value.imag)) {
            result.real = arccos_value.imag;
            result.imag = arccos_value.real;
        } else if (isnan(arccos_value.real)) {
            result.real = fabsl(arccos_value.imag);
            result.imag = arccos_value.real;
        } else if (isnan(arccos_value.imag)) {
            result.real = arccos_value.imag;
            result.imag = arccos_value.imag;
        } else {
            result.real = fabsl(arccos_value.imag);
            result.imag = copysignl(arccos_value.real, source.imag);
        }
#endif
        return result;
    }
#ifdef _MSC_VER
    _Lcomplex native_source;
    _Lcomplex native_result;
    cnp_clongdouble result;
    memcpy(&native_source, &source, sizeof(native_source));
    if (operation == CNP_HYPERBOLIC_SINH)
        native_result = csinhl(native_source);
    else if (operation == CNP_HYPERBOLIC_COSH)
        native_result = ccoshl(native_source);
    else
        native_result = ctanhl(native_source);
    memcpy(&result, &native_result, sizeof(result));
#else
    long double complex native_source =
        CMPLXL(source.real, source.imag);
    long double complex native_result;
    if (operation == CNP_HYPERBOLIC_SINH)
        native_result = csinhl(native_source);
    else if (operation == CNP_HYPERBOLIC_COSH)
        native_result = ccoshl(native_source);
    else
        native_result = ctanhl(native_source);
    cnp_clongdouble result = {
        creall(native_result), cimagl(native_result)};
#endif
    return result;
}

static CNP_STATUS unary_transcendental_element(
    CNP_TYPE source_dtype,
    CNP_TYPE result_dtype,
    const void *source,
    void *destination,
    CnpTranscendentalOperation operation,
    const char *function_name,
    const char *operation_name) {
    if (operation == CNP_TRANSCENDENTAL_EXP &&
            source_dtype == CNP_HALF && result_dtype == CNP_HALF) {
        uint16_t source_bits;
        memcpy(&source_bits, source, sizeof(source_bits));
        if ((source_bits & 0x7c00u) == 0x7c00u &&
                (source_bits & 0x03ffu) != 0) {
            uint16_t result_bits = source_bits | 0x0200u;
            memcpy(destination, &result_bits, sizeof(result_bits));
            return CNP_OK;
        }
    }
    if (operation == CNP_TRANSCENDENTAL_EXPM1 &&
            source_dtype == CNP_HALF && result_dtype == CNP_HALF) {
        uint16_t source_bits;
        memcpy(&source_bits, source, sizeof(source_bits));
        if ((source_bits & 0x7c00u) == 0x7c00u &&
                (source_bits & 0x03ffu) != 0) {
            memcpy(destination, &source_bits, sizeof(source_bits));
            return CNP_OK;
        }
    }
    if (operation == CNP_TRANSCENDENTAL_LOG1P &&
            source_dtype == CNP_HALF && result_dtype == CNP_HALF) {
        uint16_t source_bits;
        memcpy(&source_bits, source, sizeof(source_bits));
        if ((source_bits & 0x7c00u) == 0x7c00u &&
                (source_bits & 0x03ffu) != 0) {
            memcpy(destination, &source_bits, sizeof(source_bits));
            return CNP_OK;
        }
    }
    if (operation == CNP_TRANSCENDENTAL_EXPM1 &&
            source_dtype == CNP_FLOAT && result_dtype == CNP_FLOAT) {
        uint32_t source_bits;
        memcpy(&source_bits, source, sizeof(source_bits));
        if ((source_bits & 0x7f800000u) == 0x7f800000u &&
                (source_bits & 0x007fffffu) != 0) {
            memcpy(destination, &source_bits, sizeof(source_bits));
            return CNP_OK;
        }
    }
    if (operation == CNP_TRANSCENDENTAL_LOG1P &&
            source_dtype == CNP_FLOAT && result_dtype == CNP_FLOAT) {
        uint32_t source_bits;
        memcpy(&source_bits, source, sizeof(source_bits));
        if ((source_bits & 0x7f800000u) == 0x7f800000u &&
                (source_bits & 0x007fffffu) != 0) {
            memcpy(destination, &source_bits, sizeof(source_bits));
            return CNP_OK;
        }
    }
    if ((operation == CNP_HYPERBOLIC_ARCSINH ||
            operation == CNP_HYPERBOLIC_ARCCOSH ||
            operation == CNP_HYPERBOLIC_ARCTANH) &&
            source_dtype == result_dtype) {
        if (source_dtype == CNP_HALF) {
            uint16_t source_bits;
            memcpy(&source_bits, source, sizeof(source_bits));
            if ((source_bits & 0x7c00u) == 0x7c00u &&
                    (source_bits & 0x03ffu) != 0) {
                memcpy(destination, &source_bits, sizeof(source_bits));
                return CNP_OK;
            }
        }
        else if (source_dtype == CNP_FLOAT) {
            uint32_t source_bits;
            memcpy(&source_bits, source, sizeof(source_bits));
            if ((source_bits & 0x7f800000u) == 0x7f800000u &&
                    (source_bits & 0x007fffffu) != 0) {
                memcpy(destination, &source_bits, sizeof(source_bits));
                return CNP_OK;
            }
        }
        else if (source_dtype == CNP_DOUBLE) {
            uint64_t source_bits;
            memcpy(&source_bits, source, sizeof(source_bits));
            if ((source_bits & 0x7ff0000000000000ull) ==
                    0x7ff0000000000000ull &&
                    (source_bits & 0x000fffffffffffffull) != 0) {
                memcpy(destination, &source_bits, sizeof(source_bits));
                return CNP_OK;
            }
        }
        else if (source_dtype == CNP_LONGDOUBLE) {
            long double input;
            memcpy(&input, source, sizeof(input));
            if (isnan(input)) {
                memcpy(destination, source, sizeof(input));
                return CNP_OK;
            }
        }
    }
    if (source_dtype == CNP_CFLOAT) {
        cnp_cfloat value;
        memcpy(&value, source, sizeof(value));
        value = unary_transcendental_cfloat(value, operation);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    if (source_dtype == CNP_CDOUBLE) {
        cnp_cdouble value;
        memcpy(&value, source, sizeof(value));
        value = unary_transcendental_cdouble(value, operation);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }
    if (source_dtype == CNP_CLONGDOUBLE) {
        cnp_clongdouble value;
        memcpy(&value, source, sizeof(value));
        value = unary_transcendental_clongdouble(value, operation);
        memcpy(destination, &value, sizeof(value));
        return CNP_OK;
    }

    CnpArithmeticValue value;
    CNP_STATUS status = arithmetic_read_value(
        source, source_dtype, &value, function_name);
    if (status != CNP_OK) return status;
    if (result_dtype == CNP_HALF) {
        uint16_t input_bits = cnp_float_to_half(
            arithmetic_value_double(&value));
        float input = (float)cnp_half_to_float(input_bits);
        volatile float result;
        if (operation == CNP_HYPERBOLIC_SINH)
            result = sinhf(input);
        else if (operation == CNP_HYPERBOLIC_COSH)
            result = coshf(input);
        else if (operation == CNP_HYPERBOLIC_TANH)
            result = tanhf(input);
        else if (operation == CNP_HYPERBOLIC_ARCSINH)
            result = asinhf(input);
        else if (operation == CNP_HYPERBOLIC_ARCCOSH)
            result = acoshf(input);
        else if (operation == CNP_HYPERBOLIC_ARCTANH)
            result = atanhf(input);
        else if (operation == CNP_TRANSCENDENTAL_EXP)
            result = unary_exp_numpy_float(input);
        else if (operation == CNP_TRANSCENDENTAL_EXP2)
            result = exp2f(input);
        else if (operation == CNP_TRANSCENDENTAL_EXPM1)
            result = expm1f(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG)
            result = logf(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG2)
            result = log2f(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG10)
            result = log10f(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG1P)
            result = log1pf(input);
        else {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "invalid transcendental operation %d", (int)operation);
            return CNP_ERR_GENERIC;
        }
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float input = arithmetic_value_float(&value);
        volatile float result;
        if (operation == CNP_HYPERBOLIC_SINH)
            result = sinhf(input);
        else if (operation == CNP_HYPERBOLIC_COSH)
            result = coshf(input);
        else if (operation == CNP_HYPERBOLIC_TANH) {
            float dispatched_result;
            cnp_simd_tanh_f32(&input, &dispatched_result, 1);
            result = dispatched_result;
        } else if (operation == CNP_HYPERBOLIC_ARCSINH)
            result = asinhf(input);
        else if (operation == CNP_HYPERBOLIC_ARCCOSH)
            result = acoshf(input);
        else if (operation == CNP_HYPERBOLIC_ARCTANH)
            result = atanhf(input);
        else if (operation == CNP_TRANSCENDENTAL_EXP)
            result = unary_exp_numpy_float(input);
        else if (operation == CNP_TRANSCENDENTAL_EXP2)
            result = exp2f(input);
        else if (operation == CNP_TRANSCENDENTAL_EXPM1)
            result = expm1f(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG)
            result = unary_log_numpy_float(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG2)
            result = log2f(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG10)
            result = log10f(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG1P)
            result = log1pf(input);
        else {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "invalid transcendental operation %d", (int)operation);
            return CNP_ERR_GENERIC;
        }
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double input = arithmetic_value_double(&value);
        volatile double result;
        if (operation == CNP_HYPERBOLIC_SINH)
            result = sinh(input);
        else if (operation == CNP_HYPERBOLIC_COSH)
            result = cosh(input);
        else if (operation == CNP_HYPERBOLIC_TANH) {
            double dispatched_result;
            cnp_simd_tanh_f64(&input, &dispatched_result, 1);
            result = dispatched_result;
        } else if (operation == CNP_HYPERBOLIC_ARCSINH)
            result = asinh(input);
        else if (operation == CNP_HYPERBOLIC_ARCCOSH)
            result = acosh(input);
        else if (operation == CNP_HYPERBOLIC_ARCTANH)
            result = atanh(input);
        else if (operation == CNP_TRANSCENDENTAL_EXP)
            result = exp(input);
        else if (operation == CNP_TRANSCENDENTAL_EXP2)
            result = exp2(input);
        else if (operation == CNP_TRANSCENDENTAL_EXPM1)
            result = expm1(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG)
            result = log(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG2)
            result = log2(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG10)
            result = log10(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG1P)
            result = log1p(input);
        else {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "invalid transcendental operation %d", (int)operation);
            return CNP_ERR_GENERIC;
        }
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double input = arithmetic_value_longdouble(&value);
        volatile long double result;
        if (operation == CNP_HYPERBOLIC_SINH)
            result = sinhl(input);
        else if (operation == CNP_HYPERBOLIC_COSH)
            result = coshl(input);
        else if (operation == CNP_HYPERBOLIC_TANH)
            result = tanhl(input);
        else if (operation == CNP_HYPERBOLIC_ARCSINH)
            result = asinhl(input);
        else if (operation == CNP_HYPERBOLIC_ARCCOSH)
            result = acoshl(input);
        else if (operation == CNP_HYPERBOLIC_ARCTANH)
            result = atanhl(input);
        else if (operation == CNP_TRANSCENDENTAL_EXP)
            result = expl(input);
        else if (operation == CNP_TRANSCENDENTAL_EXP2)
            result = exp2l(input);
        else if (operation == CNP_TRANSCENDENTAL_EXPM1)
            result = expm1l(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG)
            result = logl(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG2)
            result = log2l(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG10)
            result = log10l(input);
        else if (operation == CNP_TRANSCENDENTAL_LOG1P)
            result = log1pl(input);
        else {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "invalid transcendental operation %d", (int)operation);
            return CNP_ERR_GENERIC;
        }
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "result dtype %d does not support %s",
        (int)result_dtype, operation_name);
    return CNP_ERR_TYPE;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static CnpArray* unary_transcendental_arrays(
    const CnpArray *source,
    CnpTranscendentalOperation operation,
    const char *function_name,
    const char *operation_name) {
    if (operation != CNP_HYPERBOLIC_SINH &&
            operation != CNP_HYPERBOLIC_COSH &&
            operation != CNP_HYPERBOLIC_TANH &&
            operation != CNP_HYPERBOLIC_ARCSINH &&
            operation != CNP_HYPERBOLIC_ARCCOSH &&
            operation != CNP_HYPERBOLIC_ARCTANH &&
            operation != CNP_TRANSCENDENTAL_EXP &&
            operation != CNP_TRANSCENDENTAL_EXP2 &&
            operation != CNP_TRANSCENDENTAL_EXPM1 &&
            operation != CNP_TRANSCENDENTAL_LOG &&
            operation != CNP_TRANSCENDENTAL_LOG2 &&
            operation != CNP_TRANSCENDENTAL_LOG10 &&
            operation != CNP_TRANSCENDENTAL_LOG1P) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "invalid transcendental operation %d", (int)operation);
        return NULL;
    }
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array requires a dtype");
        return NULL;
    }
    if (source->ndim < 0 || source->ndim > CNP_MAXDIMS ||
            (source->ndim > 0 && (!source->shape || !source->strides))) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source array requires valid shape metadata");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }

    CNP_TYPE source_dtype = source->dtype->type_num;
    CNP_TYPE result_dtype = unary_transcendental_result_dtype(source_dtype);
    if (result_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support %s",
            source->dtype->name, operation_name);
        return NULL;
    }

    CNP_ORDER order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    unary_set_keep_order_layout(source, result);
    if (source->size == 0) return result;

    int source_itemsize = source->dtype->elsize;
    int result_itemsize = result->dtype->elsize;
    if (source->flags & (
            CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) {
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        if (operation == CNP_TRANSCENDENTAL_EXP &&
                source_dtype == CNP_DOUBLE && result_dtype == CNP_DOUBLE) {
            const double *source_values = (const double*)source_data;
            double *result_values = (double*)result_data;
            int64_t index = 0;
            for (; index + 3 < source->size; index += 4) {
                result_values[index] = exp(source_values[index]);
                result_values[index + 1] = exp(source_values[index + 1]);
                result_values[index + 2] = exp(source_values[index + 2]);
                result_values[index + 3] = exp(source_values[index + 3]);
            }
            for (; index < source->size; ++index)
                result_values[index] = exp(source_values[index]);
            return result;
        }
        if (operation == CNP_TRANSCENDENTAL_EXP2 &&
                source_dtype == CNP_DOUBLE && result_dtype == CNP_DOUBLE) {
            const double *source_values = (const double*)source_data;
            double *result_values = (double*)result_data;
            int64_t index = 0;
            for (; index + 3 < source->size; index += 4) {
                result_values[index] = exp2(source_values[index]);
                result_values[index + 1] = exp2(source_values[index + 1]);
                result_values[index + 2] = exp2(source_values[index + 2]);
                result_values[index + 3] = exp2(source_values[index + 3]);
            }
            for (; index < source->size; ++index)
                result_values[index] = exp2(source_values[index]);
            return result;
        }
        if (operation == CNP_TRANSCENDENTAL_EXPM1 &&
                source_dtype == CNP_DOUBLE && result_dtype == CNP_DOUBLE) {
            const double *source_values = (const double*)source_data;
            double *result_values = (double*)result_data;
            int64_t index = 0;
            for (; index + 3 < source->size; index += 4) {
                result_values[index] = expm1(source_values[index]);
                result_values[index + 1] = expm1(source_values[index + 1]);
                result_values[index + 2] = expm1(source_values[index + 2]);
                result_values[index + 3] = expm1(source_values[index + 3]);
            }
            for (; index < source->size; ++index)
                result_values[index] = expm1(source_values[index]);
            return result;
        }
        if (operation == CNP_TRANSCENDENTAL_LOG &&
                source_dtype == CNP_DOUBLE && result_dtype == CNP_DOUBLE) {
            const double *source_values = (const double*)source_data;
            double *result_values = (double*)result_data;
            int64_t index = 0;
            for (; index + 3 < source->size; index += 4) {
                result_values[index] = log(source_values[index]);
                result_values[index + 1] = log(source_values[index + 1]);
                result_values[index + 2] = log(source_values[index + 2]);
                result_values[index + 3] = log(source_values[index + 3]);
            }
            for (; index < source->size; ++index)
                result_values[index] = log(source_values[index]);
            return result;
        }
        if (operation == CNP_TRANSCENDENTAL_LOG2 &&
                source_dtype == CNP_DOUBLE && result_dtype == CNP_DOUBLE) {
            const double *source_values = (const double*)source_data;
            double *result_values = (double*)result_data;
            int64_t index = 0;
            for (; index + 3 < source->size; index += 4) {
                result_values[index] = log2(source_values[index]);
                result_values[index + 1] = log2(source_values[index + 1]);
                result_values[index + 2] = log2(source_values[index + 2]);
                result_values[index + 3] = log2(source_values[index + 3]);
            }
            for (; index < source->size; ++index)
                result_values[index] = log2(source_values[index]);
            return result;
        }
        if (operation == CNP_TRANSCENDENTAL_LOG10 &&
                source_dtype == CNP_DOUBLE && result_dtype == CNP_DOUBLE) {
            const double *source_values = (const double*)source_data;
            double *result_values = (double*)result_data;
            int64_t index = 0;
            for (; index + 3 < source->size; index += 4) {
                result_values[index] = log10(source_values[index]);
                result_values[index + 1] = log10(source_values[index + 1]);
                result_values[index + 2] = log10(source_values[index + 2]);
                result_values[index + 3] = log10(source_values[index + 3]);
            }
            for (; index < source->size; ++index)
                result_values[index] = log10(source_values[index]);
            return result;
        }
        if (operation == CNP_TRANSCENDENTAL_LOG1P &&
                source_dtype == CNP_DOUBLE && result_dtype == CNP_DOUBLE) {
            const double *source_values = (const double*)source_data;
            double *result_values = (double*)result_data;
            int64_t index = 0;
            for (; index + 3 < source->size; index += 4) {
                result_values[index] = log1p(source_values[index]);
                result_values[index + 1] = log1p(source_values[index + 1]);
                result_values[index + 2] = log1p(source_values[index + 2]);
                result_values[index + 3] = log1p(source_values[index + 3]);
            }
            for (; index < source->size; ++index)
                result_values[index] = log1p(source_values[index]);
            return result;
        }
        if (operation == CNP_HYPERBOLIC_TANH &&
                source_dtype == CNP_FLOAT && result_dtype == CNP_FLOAT) {
            cnp_simd_tanh_f32(
                (const float*)source_data, (float*)result_data, source->size);
            return result;
        }
        if (operation == CNP_HYPERBOLIC_TANH &&
                source_dtype == CNP_DOUBLE && result_dtype == CNP_DOUBLE) {
            cnp_simd_tanh_f64(
                (const double*)source_data,
                (double*)result_data,
                source->size);
            return result;
        }
        for (int64_t index = 0; index < source->size; ++index) {
            CNP_STATUS status = unary_transcendental_element(
                source_dtype, result_dtype,
                source_data + index * source_itemsize,
                result_data + index * result_itemsize,
                operation, function_name, operation_name);
            if (status != CNP_OK) {
                cnp_array_decref(result);
                cnp_relabel_error(function_name);
                return NULL;
            }
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = unary_transcendental_element(
            source_dtype, result_dtype,
            (const char*)source->data + source_offset,
            (char*)result->data + result_offset,
            operation, function_name, operation_name);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int dimension = result->ndim - 1;
                dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_sinh(const CnpArray *a) {
    return unary_transcendental_arrays(
        a, CNP_HYPERBOLIC_SINH, "cnp_sinh", "sinh");
}
CNP_API CnpArray* CNP_CALL cnp_cosh(const CnpArray *a) {
    return unary_transcendental_arrays(
        a, CNP_HYPERBOLIC_COSH, "cnp_cosh", "cosh");
}
CNP_API CnpArray* CNP_CALL cnp_tanh(const CnpArray *a) {
    return unary_transcendental_arrays(
        a, CNP_HYPERBOLIC_TANH, "cnp_tanh", "tanh");
}
CNP_API CnpArray* CNP_CALL cnp_arcsinh(const CnpArray *a) {
    return unary_transcendental_arrays(
        a, CNP_HYPERBOLIC_ARCSINH, "cnp_arcsinh", "arcsinh");
}
CNP_API CnpArray* CNP_CALL cnp_arccosh(const CnpArray *a) {
    return unary_transcendental_arrays(
        a, CNP_HYPERBOLIC_ARCCOSH, "cnp_arccosh", "arccosh");
}
CNP_API CnpArray* CNP_CALL cnp_arctanh(const CnpArray *a) {
    return unary_transcendental_arrays(
        a, CNP_HYPERBOLIC_ARCTANH, "cnp_arctanh", "arctanh");
}

/* =========================================================================
 * Exponential and logarithmic functions
 * ========================================================================= */
typedef enum {
    CNP_LOGADDEXP_NATURAL = 0,
    CNP_LOGADDEXP_BASE2 = 1
} CnpLogaddexpOperation;

static const char* logaddexp_operation_name(
    CnpLogaddexpOperation operation) {
    return operation == CNP_LOGADDEXP_BASE2
        ? "cnp_logaddexp2" : "cnp_logaddexp";
}

static CNP_STATUS logaddexp_validate_inputs(
    const CnpArray *left,
    const CnpArray *right,
    CNP_TYPE *result_dtype,
    CnpLogaddexpOperation operation) {
    const char *function_name = logaddexp_operation_name(operation);
    const char *operation_name = operation == CNP_LOGADDEXP_BASE2
        ? "logaddexp2" : "logaddexp";
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays are required");
        return CNP_ERR_GENERIC;
    }
    if (!left->dtype || !right->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "left and right arrays must have dtypes");
        return CNP_ERR_TYPE;
    }
    if (left->ndim < 0 || left->ndim > CNP_MAXDIMS ||
            right->ndim < 0 || right->ndim > CNP_MAXDIMS ||
            (left->ndim > 0 && (!left->shape || !left->strides)) ||
            (right->ndim > 0 && (!right->shape || !right->strides))) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "left and right arrays require valid shape metadata");
        return CNP_ERR_SHAPE;
    }
    if ((left->size > 0 && !left->data) ||
            (right->size > 0 && !right->data)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays require data buffers");
        return CNP_ERR_GENERIC;
    }
    CNP_TYPE resolved = real_binary_result_dtype(
        left->dtype->type_num, right->dtype->type_num);
    if (resolved == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "%s does not support dtypes %s and %s",
            operation_name, left->dtype->name, right->dtype->name);
        return CNP_ERR_TYPE;
    }
    if (!cnp_can_broadcast(left, right)) {
        cnp_set_error(
            CNP_ERR_BROADCAST, function_name,
            "left and right arrays cannot be broadcast together");
        return CNP_ERR_BROADCAST;
    }
    *result_dtype = resolved;
    return CNP_OK;
}

static CnpArray* logaddexp_prepare_result(
    const CnpArray *left,
    const CnpArray *right,
    CNP_TYPE result_dtype,
    CnpLogaddexpOperation operation) {
    const char *function_name = logaddexp_operation_name(operation);
    int result_ndim = left->ndim > right->ndim
        ? left->ndim : right->ndim;
    int64_t result_shape[CNP_MAXDIMS];
    for (int axis = 0; axis < result_ndim; ++axis) {
        int left_axis = axis - (result_ndim - left->ndim);
        int right_axis = axis - (result_ndim - right->ndim);
        int64_t left_dimension = left_axis < 0
            ? 1 : left->shape[left_axis];
        int64_t right_dimension = right_axis < 0
            ? 1 : right->shape[right_axis];
        result_shape[axis] = left_dimension == 1
            ? right_dimension : left_dimension;
    }
    CnpArray *result = cnp_array_new(
        result_ndim, result_shape, result_dtype,
        arithmetic_result_order(left, right));
    if (!result) cnp_relabel_error(function_name);
    return result;
}

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static float logaddexp_numpy_float(float left, float right) {
    if (left == right)
        return left + 0.693147180559945309417232121458176568f;
    volatile float difference = left - right;
    if (difference > 0.0f)
        return left + log1pf(expf(-difference));
    if (difference <= 0.0f)
        return right + log1pf(expf(difference));
    return difference;
}

static double logaddexp_numpy_double(double left, double right) {
    if (left == right)
        return left + 0.693147180559945309417232121458176568;
    volatile double difference = left - right;
    if (difference > 0.0)
        return left + log1p(exp(-difference));
    if (difference <= 0.0)
        return right + log1p(exp(difference));
    return difference;
}

static long double logaddexp_numpy_longdouble(
    long double left, long double right) {
    if (left == right)
        return left + 0.693147180559945309417232121458176568L;
    volatile long double difference = left - right;
    if (difference > 0.0L)
        return left + log1pl(expl(-difference));
    if (difference <= 0.0L)
        return right + log1pl(expl(difference));
    return difference;
}

static float logaddexp2_numpy_float(float left, float right) {
    if (left == right)
        return left + 1.0f;
    volatile float difference = left - right;
    if (difference > 0.0f)
        return left +
            1.442695040888963407359924681001892137f *
            log1pf(exp2f(-difference));
    if (difference <= 0.0f)
        return right +
            1.442695040888963407359924681001892137f *
            log1pf(exp2f(difference));
    return difference;
}

static double logaddexp2_numpy_double(double left, double right) {
    if (left == right)
        return left + 1.0;
    volatile double difference = left - right;
    if (difference > 0.0)
        return left +
            1.442695040888963407359924681001892137 *
            log1p(exp2(-difference));
    if (difference <= 0.0)
        return right +
            1.442695040888963407359924681001892137 *
            log1p(exp2(difference));
    return difference;
}

static long double logaddexp2_numpy_longdouble(
    long double left, long double right) {
    if (left == right)
        return left + 1.0L;
    volatile long double difference = left - right;
    if (difference > 0.0L)
        return left +
            1.442695040888963407359924681001892137L *
            log1pl(exp2l(-difference));
    if (difference <= 0.0L)
        return right +
            1.442695040888963407359924681001892137L *
            log1pl(exp2l(difference));
    return difference;
}

static CNP_STATUS logaddexp_element(
    const void *left_source,
    CNP_TYPE left_dtype,
    const void *right_source,
    CNP_TYPE right_dtype,
    void *destination,
    CNP_TYPE result_dtype,
    CnpLogaddexpOperation operation) {
    const char *function_name = logaddexp_operation_name(operation);
    CnpArithmeticValue left_value;
    CnpArithmeticValue right_value;
    CNP_STATUS status = arithmetic_read_value(
        left_source, left_dtype, &left_value, function_name);
    if (status != CNP_OK) return status;
    status = arithmetic_read_value(
        right_source, right_dtype, &right_value, function_name);
    if (status != CNP_OK) return status;

    if (result_dtype == CNP_HALF) {
        uint16_t left_bits = cnp_float_to_half(
            arithmetic_value_double(&left_value));
        uint16_t right_bits = cnp_float_to_half(
            arithmetic_value_double(&right_value));
        float left_input = (float)cnp_half_to_float(left_bits);
        float right_input = (float)cnp_half_to_float(right_bits);
        volatile float result = operation == CNP_LOGADDEXP_BASE2
            ? logaddexp2_numpy_float(left_input, right_input)
            : logaddexp_numpy_float(left_input, right_input);
        uint16_t result_bits = cnp_float_to_half((double)result);
        memcpy(destination, &result_bits, sizeof(result_bits));
        return CNP_OK;
    }
    if (result_dtype == CNP_FLOAT) {
        float left_input = arithmetic_value_float(&left_value);
        float right_input = arithmetic_value_float(&right_value);
        volatile float result = operation == CNP_LOGADDEXP_BASE2
            ? logaddexp2_numpy_float(left_input, right_input)
            : logaddexp_numpy_float(left_input, right_input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_DOUBLE) {
        double left_input = arithmetic_value_double(&left_value);
        double right_input = arithmetic_value_double(&right_value);
        volatile double result = operation == CNP_LOGADDEXP_BASE2
            ? logaddexp2_numpy_double(left_input, right_input)
            : logaddexp_numpy_double(left_input, right_input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    if (result_dtype == CNP_LONGDOUBLE) {
        long double left_input = arithmetic_value_longdouble(&left_value);
        long double right_input = arithmetic_value_longdouble(&right_value);
        volatile long double result = operation == CNP_LOGADDEXP_BASE2
            ? logaddexp2_numpy_longdouble(left_input, right_input)
            : logaddexp_numpy_longdouble(left_input, right_input);
        memcpy(destination, (const void*)&result, sizeof(result));
        return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "result dtype %d does not support logaddexp",
        (int)result_dtype);
    return CNP_ERR_TYPE;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static bool logaddexp_contiguous_typed(
    const CnpArray *left,
    const CnpArray *right,
    CnpArray *result,
    CnpLogaddexpOperation operation) {
    bool c_contiguous =
        (left->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (right->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_C_CONTIGUOUS);
    bool f_contiguous =
        (left->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (right->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_F_CONTIGUOUS);
    if ((!c_contiguous && !f_contiguous) ||
            !same_shape(left, right) || !same_shape(left, result) ||
            left->dtype->type_num != CNP_DOUBLE ||
            right->dtype->type_num != CNP_DOUBLE ||
            result->dtype->type_num != CNP_DOUBLE)
        return false;
    if (result->size == 0) return true;

    const double *left_values = (const double*)((const char*)
        left->data + left->offset);
    const double *right_values = (const double*)((const char*)
        right->data + right->offset);
    double *result_values = (double*)((char*)
        result->data + result->offset);
    int64_t index = 0;
    if (operation == CNP_LOGADDEXP_BASE2) {
        for (; index + 3 < result->size; index += 4) {
            result_values[index] = logaddexp2_numpy_double(
                left_values[index], right_values[index]);
            result_values[index + 1] = logaddexp2_numpy_double(
                left_values[index + 1], right_values[index + 1]);
            result_values[index + 2] = logaddexp2_numpy_double(
                left_values[index + 2], right_values[index + 2]);
            result_values[index + 3] = logaddexp2_numpy_double(
                left_values[index + 3], right_values[index + 3]);
        }
        for (; index < result->size; ++index) {
            result_values[index] = logaddexp2_numpy_double(
                left_values[index], right_values[index]);
        }
        return true;
    }
    for (; index + 3 < result->size; index += 4) {
        result_values[index] = logaddexp_numpy_double(
            left_values[index], right_values[index]);
        result_values[index + 1] = logaddexp_numpy_double(
            left_values[index + 1], right_values[index + 1]);
        result_values[index + 2] = logaddexp_numpy_double(
            left_values[index + 2], right_values[index + 2]);
        result_values[index + 3] = logaddexp_numpy_double(
            left_values[index + 3], right_values[index + 3]);
    }
    for (; index < result->size; ++index) {
        result_values[index] = logaddexp_numpy_double(
            left_values[index], right_values[index]);
    }
    return true;
}

static CnpArray* logaddexp_arrays(
    const CnpArray *left,
    const CnpArray *right,
    CnpLogaddexpOperation operation) {
    const char *function_name = logaddexp_operation_name(operation);
    CNP_TYPE result_dtype = CNP_NOTYPE;
    if (logaddexp_validate_inputs(
            left, right, &result_dtype, operation) != CNP_OK)
        return NULL;
    CnpArray *result = logaddexp_prepare_result(
        left, right, result_dtype, operation);
    if (!result) return NULL;
    if (logaddexp_contiguous_typed(
            left, right, result, operation)) return result;

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t left_offset = arithmetic_broadcast_offset(
            left, coordinates, result->ndim);
        int64_t right_offset = arithmetic_broadcast_offset(
            right, coordinates, result->ndim);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CNP_STATUS status = logaddexp_element(
            (const char*)left->data + left_offset,
            left->dtype->type_num,
            (const char*)right->data + right_offset,
            right->dtype->type_num,
            (char*)result->data + result_offset,
            result_dtype,
            operation);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int axis = result->ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < result->shape[axis]) break;
            coordinates[axis] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_exp(const CnpArray *a) {
    return unary_transcendental_arrays(
        a, CNP_TRANSCENDENTAL_EXP, "cnp_exp", "exp");
}
CNP_API CnpArray* CNP_CALL cnp_exp2(const CnpArray *a) {
    return unary_transcendental_arrays(
        a, CNP_TRANSCENDENTAL_EXP2, "cnp_exp2", "exp2");
}
CNP_API CnpArray* CNP_CALL cnp_expm1(const CnpArray *a) {
    return unary_transcendental_arrays(
        a, CNP_TRANSCENDENTAL_EXPM1, "cnp_expm1", "expm1");
}
CNP_API CnpArray* CNP_CALL cnp_log(const CnpArray *a) {
    return unary_transcendental_arrays(
        a, CNP_TRANSCENDENTAL_LOG, "cnp_log", "log");
}
CNP_API CnpArray* CNP_CALL cnp_log2(const CnpArray *a) {
    return unary_transcendental_arrays(
        a, CNP_TRANSCENDENTAL_LOG2, "cnp_log2", "log2");
}
CNP_API CnpArray* CNP_CALL cnp_log10(const CnpArray *a) {
    return unary_transcendental_arrays(
        a, CNP_TRANSCENDENTAL_LOG10, "cnp_log10", "log10");
}
CNP_API CnpArray* CNP_CALL cnp_log1p(const CnpArray *a) {
    return unary_transcendental_arrays(
        a, CNP_TRANSCENDENTAL_LOG1P, "cnp_log1p", "log1p");
}

CNP_API CnpArray* CNP_CALL cnp_logaddexp(const CnpArray *a, const CnpArray *b) {
    return logaddexp_arrays(a, b, CNP_LOGADDEXP_NATURAL);
}

CNP_API CnpArray* CNP_CALL cnp_logaddexp2(const CnpArray *a, const CnpArray *b) {
    return logaddexp_arrays(a, b, CNP_LOGADDEXP_BASE2);
}

/* =========================================================================
 * Comparison operations (element-wise)
 * ========================================================================= */
typedef enum {
    CNP_COMPARISON_EQUAL = 0,
    CNP_COMPARISON_NOT_EQUAL,
    CNP_COMPARISON_LESS,
    CNP_COMPARISON_LESS_EQUAL,
    CNP_COMPARISON_GREATER,
    CNP_COMPARISON_GREATER_EQUAL
} CnpComparisonOperation;

static const char* comparison_operation_name(
    CnpComparisonOperation operation) {
    switch (operation) {
        case CNP_COMPARISON_EQUAL: return "equal";
        case CNP_COMPARISON_NOT_EQUAL: return "not_equal";
        case CNP_COMPARISON_LESS: return "less";
        case CNP_COMPARISON_LESS_EQUAL: return "less_equal";
        case CNP_COMPARISON_GREATER: return "greater";
        case CNP_COMPARISON_GREATER_EQUAL: return "greater_equal";
        default: return "comparison";
    }
}

static CNP_STATUS comparison_from_relation(
    int relation,
    CnpComparisonOperation operation,
    bool *result,
    const char *function_name);

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif
static CNP_STATUS comparison_promoted(
    const CnpArithmeticScalarStorage *left,
    const CnpArithmeticScalarStorage *right,
    CNP_TYPE dtype,
    CnpComparisonOperation operation,
    bool *result,
    const char *function_name) {
#define CNP_COMPARE_PROMOTED_VALUE(c_type) do { \
    c_type left_value; \
    c_type right_value; \
    memcpy(&left_value, left, sizeof(left_value)); \
    memcpy(&right_value, right, sizeof(right_value)); \
    switch (operation) { \
        case CNP_COMPARISON_EQUAL: \
            *result = left_value == right_value; \
            break; \
        case CNP_COMPARISON_NOT_EQUAL: \
            *result = left_value != right_value; \
            break; \
        case CNP_COMPARISON_LESS: \
            *result = left_value < right_value; \
            break; \
        case CNP_COMPARISON_LESS_EQUAL: \
            *result = left_value <= right_value; \
            break; \
        case CNP_COMPARISON_GREATER: \
            *result = left_value > right_value; \
            break; \
        case CNP_COMPARISON_GREATER_EQUAL: \
            *result = left_value >= right_value; \
            break; \
        default: \
            cnp_set_error( \
                CNP_ERR_GENERIC, function_name, \
                "invalid internal comparison operation %d", \
                (int)operation); \
            return CNP_ERR_GENERIC; \
    } \
    return CNP_OK; \
} while (0)
#define CNP_COMPARE_COMPLEX_VALUE(c_type) do { \
    c_type left_value; \
    c_type right_value; \
    memcpy(&left_value, left, sizeof(left_value)); \
    memcpy(&right_value, right, sizeof(right_value)); \
    bool unordered = isnan(left_value.real) || \
        isnan(left_value.imag) || isnan(right_value.real) || \
        isnan(right_value.imag); \
    switch (operation) { \
        case CNP_COMPARISON_EQUAL: \
            *result = left_value.real == right_value.real && \
                left_value.imag == right_value.imag; \
            break; \
        case CNP_COMPARISON_NOT_EQUAL: \
            *result = left_value.real != right_value.real || \
                left_value.imag != right_value.imag; \
            break; \
        case CNP_COMPARISON_LESS: \
            *result = !unordered && \
                (left_value.real < right_value.real || \
                (left_value.real == right_value.real && \
                 left_value.imag < right_value.imag)); \
            break; \
        case CNP_COMPARISON_LESS_EQUAL: \
            *result = !unordered && \
                (left_value.real < right_value.real || \
                (left_value.real == right_value.real && \
                 left_value.imag <= right_value.imag)); \
            break; \
        case CNP_COMPARISON_GREATER: \
            *result = !unordered && \
                (left_value.real > right_value.real || \
                (left_value.real == right_value.real && \
                 left_value.imag > right_value.imag)); \
            break; \
        case CNP_COMPARISON_GREATER_EQUAL: \
            *result = !unordered && \
                (left_value.real > right_value.real || \
                (left_value.real == right_value.real && \
                 left_value.imag >= right_value.imag)); \
            break; \
        default: \
            cnp_set_error( \
                CNP_ERR_GENERIC, function_name, \
                "invalid internal comparison operation %d", \
                (int)operation); \
            return CNP_ERR_GENERIC; \
    } \
    return CNP_OK; \
} while (0)

    switch (dtype) {
        case CNP_BOOL:
        case CNP_BYTE:
            CNP_COMPARE_PROMOTED_VALUE(int8_t);
        case CNP_UBYTE:
            CNP_COMPARE_PROMOTED_VALUE(uint8_t);
        case CNP_SHORT:
            CNP_COMPARE_PROMOTED_VALUE(int16_t);
        case CNP_USHORT:
        case CNP_HALF:
            if (dtype == CNP_HALF) {
                uint16_t left_bits;
                uint16_t right_bits;
                memcpy(&left_bits, left, sizeof(left_bits));
                memcpy(&right_bits, right, sizeof(right_bits));
                volatile float left_value =
                    (float)cnp_half_to_float(left_bits);
                volatile float right_value =
                    (float)cnp_half_to_float(right_bits);
                switch (operation) {
                    case CNP_COMPARISON_EQUAL:
                        *result = left_value == right_value;
                        break;
                    case CNP_COMPARISON_NOT_EQUAL:
                        *result = left_value != right_value;
                        break;
                    case CNP_COMPARISON_LESS:
                        *result = left_value < right_value;
                        break;
                    case CNP_COMPARISON_LESS_EQUAL:
                        *result = left_value <= right_value;
                        break;
                    case CNP_COMPARISON_GREATER:
                        *result = left_value > right_value;
                        break;
                    case CNP_COMPARISON_GREATER_EQUAL:
                        *result = left_value >= right_value;
                        break;
                    default:
                        cnp_set_error(
                            CNP_ERR_GENERIC, function_name,
                            "invalid internal comparison operation %d",
                            (int)operation);
                        return CNP_ERR_GENERIC;
                }
                return CNP_OK;
            }
            CNP_COMPARE_PROMOTED_VALUE(uint16_t);
        case CNP_INT:
            CNP_COMPARE_PROMOTED_VALUE(int32_t);
        case CNP_UINT:
            CNP_COMPARE_PROMOTED_VALUE(uint32_t);
        case CNP_LONG:
        case CNP_LONGLONG:
            CNP_COMPARE_PROMOTED_VALUE(int64_t);
        case CNP_DATETIME:
        case CNP_TIMEDELTA: {
            int64_t left_value;
            int64_t right_value;
            memcpy(&left_value, left, sizeof(left_value));
            memcpy(&right_value, right, sizeof(right_value));
            if (left_value == INT64_MIN || right_value == INT64_MIN) {
                *result = operation == CNP_COMPARISON_NOT_EQUAL;
                return CNP_OK;
            }
            int relation = left_value < right_value ? -1 :
                left_value > right_value ? 1 : 0;
            return comparison_from_relation(
                relation, operation, result, function_name);
        }
        case CNP_ULONG:
        case CNP_ULONGLONG:
            CNP_COMPARE_PROMOTED_VALUE(uint64_t);
        case CNP_FLOAT:
            CNP_COMPARE_PROMOTED_VALUE(float);
        case CNP_DOUBLE:
            CNP_COMPARE_PROMOTED_VALUE(double);
        case CNP_LONGDOUBLE:
            CNP_COMPARE_PROMOTED_VALUE(long double);
        case CNP_CFLOAT:
            CNP_COMPARE_COMPLEX_VALUE(cnp_cfloat);
        case CNP_CDOUBLE:
            CNP_COMPARE_COMPLEX_VALUE(cnp_cdouble);
        case CNP_CLONGDOUBLE:
            CNP_COMPARE_COMPLEX_VALUE(cnp_clongdouble);
        default:
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "promoted dtype %d does not support %s",
                (int)dtype, comparison_operation_name(operation));
            return CNP_ERR_TYPE;
    }
#undef CNP_COMPARE_PROMOTED_VALUE
#undef CNP_COMPARE_COMPLEX_VALUE
}

static CNP_STATUS comparison_from_relation(
    int relation,
    CnpComparisonOperation operation,
    bool *result,
    const char *function_name) {
    switch (operation) {
        case CNP_COMPARISON_EQUAL:
            *result = relation == 0;
            return CNP_OK;
        case CNP_COMPARISON_NOT_EQUAL:
            *result = relation != 0;
            return CNP_OK;
        case CNP_COMPARISON_LESS:
            *result = relation < 0;
            return CNP_OK;
        case CNP_COMPARISON_LESS_EQUAL:
            *result = relation <= 0;
            return CNP_OK;
        case CNP_COMPARISON_GREATER:
            *result = relation > 0;
            return CNP_OK;
        case CNP_COMPARISON_GREATER_EQUAL:
            *result = relation >= 0;
            return CNP_OK;
        default:
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "invalid internal comparison operation %d",
                (int)operation);
            return CNP_ERR_GENERIC;
    }
}

static CNP_STATUS comparison_mixed_integer(
    const void *left_source,
    CNP_TYPE left_dtype,
    const void *right_source,
    CNP_TYPE right_dtype,
    CnpComparisonOperation operation,
    bool *result,
    const char *function_name) {
    CnpArithmeticValue left_value;
    CnpArithmeticValue right_value;
    CNP_STATUS status = arithmetic_read_value(
        left_source, left_dtype, &left_value, function_name);
    if (status != CNP_OK) return status;
    status = arithmetic_read_value(
        right_source, right_dtype, &right_value, function_name);
    if (status != CNP_OK) return status;

    int relation;
    if (left_value.kind == CNP_ARITHMETIC_SIGNED) {
        if (left_value.signed_value < 0) {
            relation = -1;
        } else {
            uint64_t left_unsigned = (uint64_t)left_value.signed_value;
            relation = left_unsigned < right_value.unsigned_value ? -1 :
                left_unsigned > right_value.unsigned_value ? 1 : 0;
        }
    } else {
        if (right_value.signed_value < 0) {
            relation = 1;
        } else {
            uint64_t right_unsigned = (uint64_t)right_value.signed_value;
            relation = left_value.unsigned_value < right_unsigned ? -1 :
                left_value.unsigned_value > right_unsigned ? 1 : 0;
        }
    }
    return comparison_from_relation(
        relation, operation, result, function_name);
}

static CNP_STATUS comparison_element(
    const void *left_source,
    CNP_TYPE left_dtype,
    const void *right_source,
    CNP_TYPE right_dtype,
    CNP_TYPE promoted_dtype,
    CnpComparisonOperation operation,
    bool *result,
    const char *function_name) {
    if (cnp_type_is_integer(left_dtype) &&
            cnp_type_is_integer(right_dtype) &&
            cnp_type_is_unsigned(left_dtype) !=
                cnp_type_is_unsigned(right_dtype)) {
        return comparison_mixed_integer(
            left_source, left_dtype, right_source, right_dtype,
            operation, result, function_name);
    }
    CnpArithmeticScalarStorage left_value = {0};
    CnpArithmeticScalarStorage right_value = {0};
    CNP_STATUS status = cnp_cast_scalar_value(
        left_source, left_dtype, &left_value,
        promoted_dtype, function_name);
    if (status != CNP_OK) return status;
    status = cnp_cast_scalar_value(
        right_source, right_dtype, &right_value,
        promoted_dtype, function_name);
    if (status != CNP_OK) return status;
    return comparison_promoted(
        &left_value, &right_value, promoted_dtype, operation,
        result, function_name);
}

static bool comparison_contiguous_double(
    const CnpArray *left,
    const CnpArray *right,
    CnpArray *result,
    CnpComparisonOperation operation) {
    bool c_contiguous =
        (left->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (right->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_C_CONTIGUOUS);
    bool f_contiguous =
        (left->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (right->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_F_CONTIGUOUS);
    if ((!c_contiguous && !f_contiguous) ||
            !same_shape(left, right) || !same_shape(left, result) ||
            left->dtype->type_num != CNP_DOUBLE ||
            right->dtype->type_num != CNP_DOUBLE) {
        return false;
    }

    const double *left_values = (const double*)(
        (const char*)left->data + left->offset);
    const double *right_values = (const double*)(
        (const char*)right->data + right->offset);
    uint8_t *result_values = (uint8_t*)(
        (char*)result->data + result->offset);
    int64_t index = 0;
    switch (operation) {
        case CNP_COMPARISON_EQUAL:
            for (; index + 3 < result->size; index += 4) {
                result_values[index] =
                    left_values[index] == right_values[index];
                result_values[index + 1] =
                    left_values[index + 1] == right_values[index + 1];
                result_values[index + 2] =
                    left_values[index + 2] == right_values[index + 2];
                result_values[index + 3] =
                    left_values[index + 3] == right_values[index + 3];
            }
            for (; index < result->size; ++index)
                result_values[index] =
                    left_values[index] == right_values[index];
            return true;
        case CNP_COMPARISON_NOT_EQUAL:
            for (; index + 3 < result->size; index += 4) {
                result_values[index] =
                    left_values[index] != right_values[index];
                result_values[index + 1] =
                    left_values[index + 1] != right_values[index + 1];
                result_values[index + 2] =
                    left_values[index + 2] != right_values[index + 2];
                result_values[index + 3] =
                    left_values[index + 3] != right_values[index + 3];
            }
            for (; index < result->size; ++index)
                result_values[index] =
                    left_values[index] != right_values[index];
            return true;
        case CNP_COMPARISON_LESS:
            for (; index + 3 < result->size; index += 4) {
                result_values[index] =
                    left_values[index] < right_values[index];
                result_values[index + 1] =
                    left_values[index + 1] < right_values[index + 1];
                result_values[index + 2] =
                    left_values[index + 2] < right_values[index + 2];
                result_values[index + 3] =
                    left_values[index + 3] < right_values[index + 3];
            }
            for (; index < result->size; ++index)
                result_values[index] =
                    left_values[index] < right_values[index];
            return true;
        case CNP_COMPARISON_LESS_EQUAL:
            for (; index + 3 < result->size; index += 4) {
                result_values[index] =
                    left_values[index] <= right_values[index];
                result_values[index + 1] =
                    left_values[index + 1] <= right_values[index + 1];
                result_values[index + 2] =
                    left_values[index + 2] <= right_values[index + 2];
                result_values[index + 3] =
                    left_values[index + 3] <= right_values[index + 3];
            }
            for (; index < result->size; ++index)
                result_values[index] =
                    left_values[index] <= right_values[index];
            return true;
        case CNP_COMPARISON_GREATER:
            for (; index + 3 < result->size; index += 4) {
                result_values[index] =
                    left_values[index] > right_values[index];
                result_values[index + 1] =
                    left_values[index + 1] > right_values[index + 1];
                result_values[index + 2] =
                    left_values[index + 2] > right_values[index + 2];
                result_values[index + 3] =
                    left_values[index + 3] > right_values[index + 3];
            }
            for (; index < result->size; ++index)
                result_values[index] =
                    left_values[index] > right_values[index];
            return true;
        case CNP_COMPARISON_GREATER_EQUAL:
            for (; index + 3 < result->size; index += 4) {
                result_values[index] =
                    left_values[index] >= right_values[index];
                result_values[index + 1] =
                    left_values[index + 1] >= right_values[index + 1];
                result_values[index + 2] =
                    left_values[index + 2] >= right_values[index + 2];
                result_values[index + 3] =
                    left_values[index + 3] >= right_values[index + 3];
            }
            for (; index < result->size; ++index)
                result_values[index] =
                    left_values[index] >= right_values[index];
            return true;
        default:
            return false;
    }
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

static CNP_TYPE comparison_loop_dtype(
    CNP_TYPE left_dtype, CNP_TYPE right_dtype) {
    if (left_dtype == CNP_DATETIME || right_dtype == CNP_DATETIME) {
        return left_dtype == CNP_DATETIME &&
            right_dtype == CNP_DATETIME
            ? CNP_DATETIME : CNP_NOTYPE;
    }
    if (left_dtype == CNP_TIMEDELTA || right_dtype == CNP_TIMEDELTA) {
        CNP_TYPE other = left_dtype == CNP_TIMEDELTA
            ? right_dtype : left_dtype;
        if (other == CNP_TIMEDELTA || other == CNP_BOOL)
            return CNP_TIMEDELTA;
        if (cnp_type_is_integer(other) &&
                other != CNP_ULONG && other != CNP_ULONGLONG)
            return CNP_TIMEDELTA;
        return CNP_NOTYPE;
    }
    return cnp_promote_type(left_dtype, right_dtype);
}

static CnpArray* comparison_arrays(
    CnpComparisonOperation operation,
    const CnpArray *left,
    const CnpArray *right) {
    const char *operation_name = comparison_operation_name(operation);
    char function_name[32];
    snprintf(function_name, sizeof(function_name), "cnp_%s", operation_name);
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays are required");
        return NULL;
    }
    if (!left->dtype || !right->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "left and right arrays must have dtypes");
        return NULL;
    }
    if ((left->size > 0 && !left->data) ||
            (right->size > 0 && !right->data)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays require data buffers");
        return NULL;
    }
    CNP_TYPE promoted_dtype = comparison_loop_dtype(
        left->dtype->type_num, right->dtype->type_num);
    if (promoted_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtypes %s and %s do not support %s",
            left->dtype->name, right->dtype->name, operation_name);
        return NULL;
    }
    CnpArray *result = arithmetic_prepare_result(
        left, right, CNP_BOOL, function_name);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    if (comparison_contiguous_double(left, right, result, operation))
        return result;

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t left_offset = arithmetic_broadcast_offset(
            left, coordinates, result->ndim);
        int64_t right_offset = arithmetic_broadcast_offset(
            right, coordinates, result->ndim);
        bool element_result = false;
        CNP_STATUS status = comparison_element(
            (const char*)left->data + left_offset,
            left->dtype->type_num,
            (const char*)right->data + right_offset,
            right->dtype->type_num,
            promoted_dtype,
            operation,
            &element_result,
            function_name);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        *((uint8_t*)result->data + result_offset) =
            element_result ? 1 : 0;
        for (int axis = result->ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < result->shape[axis]) break;
            coordinates[axis] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_equal(
    const CnpArray *a, const CnpArray *b) {
    return comparison_arrays(CNP_COMPARISON_EQUAL, a, b);
}

CNP_API CnpArray* CNP_CALL cnp_not_equal(
    const CnpArray *a, const CnpArray *b) {
    return comparison_arrays(CNP_COMPARISON_NOT_EQUAL, a, b);
}

CNP_API CnpArray* CNP_CALL cnp_less(
    const CnpArray *a, const CnpArray *b) {
    return comparison_arrays(CNP_COMPARISON_LESS, a, b);
}

CNP_API CnpArray* CNP_CALL cnp_less_equal(
    const CnpArray *a, const CnpArray *b) {
    return comparison_arrays(CNP_COMPARISON_LESS_EQUAL, a, b);
}

CNP_API CnpArray* CNP_CALL cnp_greater(
    const CnpArray *a, const CnpArray *b) {
    return comparison_arrays(CNP_COMPARISON_GREATER, a, b);
}

CNP_API CnpArray* CNP_CALL cnp_greater_equal(
    const CnpArray *a, const CnpArray *b) {
    return comparison_arrays(CNP_COMPARISON_GREATER_EQUAL, a, b);
}

typedef enum {
    CNP_EXTREMA_MAXIMUM = 0,
    CNP_EXTREMA_MINIMUM,
    CNP_EXTREMA_FMAX,
    CNP_EXTREMA_FMIN
} CnpExtremaOperation;

static const char* extrema_operation_name(CnpExtremaOperation operation) {
    switch (operation) {
        case CNP_EXTREMA_MAXIMUM: return "maximum";
        case CNP_EXTREMA_MINIMUM: return "minimum";
        case CNP_EXTREMA_FMAX: return "fmax";
        case CNP_EXTREMA_FMIN: return "fmin";
        default: return "extrema";
    }
}

static bool extrema_is_maximum(CnpExtremaOperation operation) {
    return operation == CNP_EXTREMA_MAXIMUM ||
        operation == CNP_EXTREMA_FMAX;
}

static bool extrema_ignores_nan(CnpExtremaOperation operation) {
    return operation == CNP_EXTREMA_FMAX ||
        operation == CNP_EXTREMA_FMIN;
}

static CNP_TYPE extrema_loop_dtype(
    CNP_TYPE left_dtype, CNP_TYPE right_dtype) {
    if (left_dtype == CNP_DATETIME || right_dtype == CNP_DATETIME) {
        return left_dtype == CNP_DATETIME &&
            right_dtype == CNP_DATETIME
            ? CNP_DATETIME : CNP_NOTYPE;
    }
    if (left_dtype == CNP_TIMEDELTA || right_dtype == CNP_TIMEDELTA) {
        CNP_TYPE other = left_dtype == CNP_TIMEDELTA
            ? right_dtype : left_dtype;
        if (other == CNP_TIMEDELTA || other == CNP_BOOL)
            return CNP_TIMEDELTA;
        if (cnp_type_is_integer(other) &&
                other != CNP_ULONG && other != CNP_ULONGLONG)
            return CNP_TIMEDELTA;
        return CNP_NOTYPE;
    }
    return cnp_promote_type(left_dtype, right_dtype);
}

static CNP_STATUS extrema_cast_value(
    const void *source,
    CNP_TYPE source_dtype,
    CNP_TYPE result_dtype,
    int result_itemsize,
    CnpArithmeticScalarStorage *value,
    const char *function_name) {
    memset(value, 0, sizeof(*value));
    if (source_dtype == result_dtype) {
        memcpy(value, source, (size_t)result_itemsize);
        return CNP_OK;
    }
    return cnp_cast_scalar_value(
        source, source_dtype, value, result_dtype, function_name);
}

static bool extrema_half_is_nan(uint16_t bits) {
    return (bits & UINT16_C(0x7c00)) == UINT16_C(0x7c00) &&
        (bits & UINT16_C(0x03ff)) != 0;
}

static bool extrema_select_temporal(
    int64_t left,
    int64_t right,
    CnpExtremaOperation operation) {
    bool left_nat = left == INT64_MIN;
    bool right_nat = right == INT64_MIN;
    if (extrema_ignores_nan(operation)) {
        if (left_nat) return right_nat;
        if (right_nat) return true;
    } else {
        if (left_nat) return true;
        if (right_nat) return false;
    }
    return extrema_is_maximum(operation)
        ? left >= right : left <= right;
}

static bool extrema_select_half(
    uint16_t left_bits,
    uint16_t right_bits,
    CnpExtremaOperation operation) {
    bool left_nan = extrema_half_is_nan(left_bits);
    bool right_nan = extrema_half_is_nan(right_bits);
    if (extrema_ignores_nan(operation)) {
        if (left_nan) return right_nan;
        if (right_nan) return true;
    } else {
        if (left_nan) return true;
        if (right_nan) return false;
    }
    float left = (float)cnp_half_to_float(left_bits);
    float right = (float)cnp_half_to_float(right_bits);
    return extrema_is_maximum(operation)
        ? left >= right : left <= right;
}

#define CNP_EXTREMA_SELECT_INTEGER(c_type) do { \
    c_type left; \
    c_type right; \
    memcpy(&left, left_value, sizeof(left)); \
    memcpy(&right, right_value, sizeof(right)); \
    return extrema_is_maximum(operation) \
        ? left >= right : left <= right; \
} while (0)

#define CNP_EXTREMA_SELECT_REAL(c_type) do { \
    c_type left; \
    c_type right; \
    memcpy(&left, left_value, sizeof(left)); \
    memcpy(&right, right_value, sizeof(right)); \
    bool left_nan = isnan(left) != 0; \
    bool right_nan = isnan(right) != 0; \
    if (extrema_ignores_nan(operation)) { \
        if (left_nan) return false; \
        if (right_nan) return true; \
        if (left == (c_type)0 && right == (c_type)0) { \
            if (extrema_is_maximum(operation)) { \
                if (!signbit(left)) return true; \
                if (!signbit(right)) return false; \
            } else { \
                if (signbit(left)) return true; \
                if (signbit(right)) return false; \
            } \
        } \
    } else { \
        if (left_nan) return true; \
        if (right_nan) return false; \
    } \
    return extrema_is_maximum(operation) \
        ? left > right : left < right; \
} while (0)

#define CNP_EXTREMA_SELECT_COMPLEX(c_type) do { \
    c_type left; \
    c_type right; \
    memcpy(&left, left_value, sizeof(left)); \
    memcpy(&right, right_value, sizeof(right)); \
    bool left_nan = isnan(left.real) || isnan(left.imag); \
    bool right_nan = isnan(right.real) || isnan(right.imag); \
    if (extrema_ignores_nan(operation)) { \
        if (left_nan) return right_nan; \
        if (right_nan) return true; \
    } else { \
        if (left_nan) return true; \
        if (right_nan) return false; \
    } \
    int relation = left.real < right.real ? -1 : \
        left.real > right.real ? 1 : \
        left.imag < right.imag ? -1 : \
        left.imag > right.imag ? 1 : 0; \
    return extrema_is_maximum(operation) \
        ? relation >= 0 : relation <= 0; \
} while (0)

static bool extrema_select_left(
    const CnpArithmeticScalarStorage *left_value,
    const CnpArithmeticScalarStorage *right_value,
    CNP_TYPE dtype,
    CnpExtremaOperation operation) {
    switch (dtype) {
        case CNP_BOOL:
        case CNP_BYTE:
            CNP_EXTREMA_SELECT_INTEGER(int8_t);
        case CNP_UBYTE:
            CNP_EXTREMA_SELECT_INTEGER(uint8_t);
        case CNP_SHORT:
            CNP_EXTREMA_SELECT_INTEGER(int16_t);
        case CNP_USHORT:
            CNP_EXTREMA_SELECT_INTEGER(uint16_t);
        case CNP_INT:
            CNP_EXTREMA_SELECT_INTEGER(int32_t);
        case CNP_UINT:
            CNP_EXTREMA_SELECT_INTEGER(uint32_t);
        case CNP_LONG:
        case CNP_LONGLONG:
            CNP_EXTREMA_SELECT_INTEGER(int64_t);
        case CNP_ULONG:
        case CNP_ULONGLONG:
            CNP_EXTREMA_SELECT_INTEGER(uint64_t);
        case CNP_HALF: {
            uint16_t left_bits;
            uint16_t right_bits;
            memcpy(&left_bits, left_value, sizeof(left_bits));
            memcpy(&right_bits, right_value, sizeof(right_bits));
            return extrema_select_half(
                left_bits, right_bits, operation);
        }
        case CNP_FLOAT:
            CNP_EXTREMA_SELECT_REAL(float);
        case CNP_DOUBLE:
            CNP_EXTREMA_SELECT_REAL(double);
        case CNP_LONGDOUBLE:
            CNP_EXTREMA_SELECT_REAL(long double);
        case CNP_CFLOAT:
            CNP_EXTREMA_SELECT_COMPLEX(cnp_cfloat);
        case CNP_CDOUBLE:
            CNP_EXTREMA_SELECT_COMPLEX(cnp_cdouble);
        case CNP_CLONGDOUBLE:
            CNP_EXTREMA_SELECT_COMPLEX(cnp_clongdouble);
        case CNP_DATETIME:
        case CNP_TIMEDELTA: {
            int64_t left;
            int64_t right;
            memcpy(&left, left_value, sizeof(left));
            memcpy(&right, right_value, sizeof(right));
            return extrema_select_temporal(left, right, operation);
        }
        default:
            return false;
    }
}

#undef CNP_EXTREMA_SELECT_INTEGER
#undef CNP_EXTREMA_SELECT_REAL
#undef CNP_EXTREMA_SELECT_COMPLEX

static bool extrema_contiguous_double(
    const CnpArray *left,
    const CnpArray *right,
    CnpArray *result,
    CnpExtremaOperation operation) {
    if (!(left->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            !(right->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            !(result->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            !same_shape(left, right) || !same_shape(left, result) ||
            left->dtype->type_num != CNP_DOUBLE ||
            right->dtype->type_num != CNP_DOUBLE) {
        return false;
    }
    const double *left_values = (const double*)(
        (const char*)left->data + left->offset);
    const double *right_values = (const double*)(
        (const char*)right->data + right->offset);
    double *result_values = (double*)(
        (char*)result->data + result->offset);
    int64_t vector_size = result->size & ~INT64_C(3);
    if (vector_size > 0) {
        switch (operation) {
            case CNP_EXTREMA_MAXIMUM:
                cnp_simd_maximum(
                    left_values, right_values,
                    result_values, vector_size);
                break;
            case CNP_EXTREMA_MINIMUM:
                cnp_simd_minimum(
                    left_values, right_values,
                    result_values, vector_size);
                break;
            case CNP_EXTREMA_FMAX:
                cnp_simd_fmax(
                    left_values, right_values,
                    result_values, vector_size);
                break;
            case CNP_EXTREMA_FMIN:
                cnp_simd_fmin(
                    left_values, right_values,
                    result_values, vector_size);
                break;
            default:
                return false;
        }
    }
    for (int64_t index = vector_size; index < result->size; ++index) {
        CnpArithmeticScalarStorage left_value = {0};
        CnpArithmeticScalarStorage right_value = {0};
        memcpy(&left_value, left_values + index, sizeof(double));
        memcpy(&right_value, right_values + index, sizeof(double));
        const CnpArithmeticScalarStorage *selected =
            extrema_select_left(
                &left_value, &right_value, CNP_DOUBLE, operation)
            ? &left_value : &right_value;
        memcpy(result_values + index, selected, sizeof(double));
    }
    return true;
}

static CnpArray* extrema_arrays(
    const CnpArray *left,
    const CnpArray *right,
    CnpExtremaOperation operation) {
    const char *operation_name = extrema_operation_name(operation);
    char function_name[32];
    snprintf(function_name, sizeof(function_name), "cnp_%s", operation_name);
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays are required");
        return NULL;
    }
    if (!left->dtype || !right->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "left and right arrays must have dtypes");
        return NULL;
    }
    if ((left->size > 0 && !left->data) ||
            (right->size > 0 && !right->data)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays require data buffers");
        return NULL;
    }

    CNP_TYPE result_dtype = extrema_loop_dtype(
        left->dtype->type_num, right->dtype->type_num);
    if (result_dtype == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtypes %s and %s do not support %s",
            left->dtype->name, right->dtype->name, operation_name);
        return NULL;
    }
    CnpArray *result = arithmetic_prepare_result(
        left, right, result_dtype, function_name);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (extrema_contiguous_double(
            left, right, result, operation)) {
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t left_offset = arithmetic_broadcast_offset(
            left, coordinates, result->ndim);
        int64_t right_offset = arithmetic_broadcast_offset(
            right, coordinates, result->ndim);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        CnpArithmeticScalarStorage left_value;
        CnpArithmeticScalarStorage right_value;
        CNP_STATUS status = extrema_cast_value(
            (const char*)left->data + left_offset,
            left->dtype->type_num,
            result_dtype,
            result->dtype->elsize,
            &left_value,
            function_name);
        if (status == CNP_OK) {
            status = extrema_cast_value(
                (const char*)right->data + right_offset,
                right->dtype->type_num,
                result_dtype,
                result->dtype->elsize,
                &right_value,
                function_name);
        }
        if (status != CNP_OK) {
            cnp_array_decref(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        const CnpArithmeticScalarStorage *selected =
            extrema_select_left(
                &left_value, &right_value, result_dtype, operation)
            ? &left_value : &right_value;
        memcpy(
            (char*)result->data + result_offset,
            selected,
            (size_t)result->dtype->elsize);

        for (int axis = result->ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < result->shape[axis]) break;
            coordinates[axis] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_maximum(
    const CnpArray *a, const CnpArray *b) {
    return extrema_arrays(a, b, CNP_EXTREMA_MAXIMUM);
}

CNP_API CnpArray* CNP_CALL cnp_minimum(
    const CnpArray *a, const CnpArray *b) {
    return extrema_arrays(a, b, CNP_EXTREMA_MINIMUM);
}

CNP_API CnpArray* CNP_CALL cnp_fmax(
    const CnpArray *a, const CnpArray *b) {
    return extrema_arrays(a, b, CNP_EXTREMA_FMAX);
}

CNP_API CnpArray* CNP_CALL cnp_fmin(
    const CnpArray *a, const CnpArray *b) {
    return extrema_arrays(a, b, CNP_EXTREMA_FMIN);
}

CNP_API CnpArray* CNP_CALL cnp_clip(const CnpArray *a, double min_val, double max_val) {
    const char *function_name = "cnp_clip";
    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "source array is required");
        return NULL;
    }
    CnpArray *minimum = cnp_array_from_scalar(min_val, CNP_DOUBLE);
    if (!minimum) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpArray *maximum = cnp_array_from_scalar(max_val, CNP_DOUBLE);
    if (!maximum) {
        cnp_array_decref(minimum);
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpArray *result = cnp_clip_array(a, minimum, maximum);
    cnp_array_decref(maximum);
    cnp_array_decref(minimum);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * Logical operations
 * ========================================================================= */
static bool fn_logical_and(bool a, bool b) { return a && b; }
static bool fn_logical_or(bool a, bool b) { return a || b; }
static bool fn_logical_xor(bool a, bool b) { return a != b; }

CNP_API CnpArray* CNP_CALL cnp_logical_and(const CnpArray *a, const CnpArray *b) {
    return cnp_logical_op(
        a, b, fn_logical_and,
        cnp_simd_logical_and, "cnp_logical_and");
}
CNP_API CnpArray* CNP_CALL cnp_logical_or(const CnpArray *a, const CnpArray *b) {
    return cnp_logical_op(
        a, b, fn_logical_or,
        cnp_simd_logical_or, "cnp_logical_or");
}
CNP_API CnpArray* CNP_CALL cnp_logical_xor(const CnpArray *a, const CnpArray *b) {
    return cnp_logical_op(
        a, b, fn_logical_xor,
        cnp_simd_logical_xor, "cnp_logical_xor");
}

CNP_API CnpArray* CNP_CALL cnp_logical_not(const CnpArray *a) {
    const char *function_name = "cnp_logical_not";
    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "source array is required");
        return NULL;
    }
    if (!a->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name, "source array must have a dtype");
        return NULL;
    }
    if (a->size > 0 && !a->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }
    if (cnp_validate_logical_truth_dtype(
            a->dtype->type_num, false, function_name) != CNP_OK)
        return NULL;
    CnpArray *result = cnp_array_new(
        a->ndim, a->shape, CNP_BOOL,
        cnp_logical_result_order(a, a));
    if (!result) return NULL;
    if (result->size == 0) return result;
    if ((a->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            (result->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            a->dtype->type_num == CNP_DOUBLE) {
        cnp_simd_logical_not(
            (const double*)((const char*)a->data + a->offset),
            (uint8_t*)((char*)result->data + result->offset),
            result->size);
        return result;
    }

    int64_t coords[CNP_MAXDIMS] = {0};
    for (int64_t i = 0; i < a->size; i++) {
        int64_t src_offset = a->offset + cnp_multi_to_offset(a->ndim, coords, a->strides);
        bool value_is_true;
        if (cnp_scalar_truth(
                (const char*)a->data + src_offset,
                a->dtype->type_num,
                &value_is_true,
                function_name) != CNP_OK) {
            cnp_array_decref(result);
            return NULL;
        }
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coords, result->strides);
        *((int8_t*)result->data + result_offset) =
            value_is_true ? 0 : 1;
        for (int d = a->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < a->shape[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}

/* =========================================================================
 * Bitwise operations
 * ========================================================================= */
typedef enum {
    CNP_BITWISE_AND,
    CNP_BITWISE_OR,
    CNP_BITWISE_XOR,
    CNP_BITWISE_LEFT_SHIFT,
    CNP_BITWISE_RIGHT_SHIFT,
    CNP_BITWISE_INVERT
} CnpBitwiseOperation;

static bool bitwise_dtype_supported(CNP_TYPE dtype) {
    return dtype == CNP_BOOL || cnp_type_is_integer(dtype);
}

static uint64_t bitwise_mask(CNP_TYPE dtype) {
    int bits = cnp_dtype_itemsize(dtype) * 8;
    return bits == 64
        ? UINT64_MAX
        : (UINT64_C(1) << bits) - UINT64_C(1);
}

static uint64_t bitwise_read_bits(const void *source, CNP_TYPE dtype) {
#define CNP_BITWISE_READ_SIGNED(c_type) do { \
    c_type value; \
    memcpy(&value, source, sizeof(value)); \
    return (uint64_t)(int64_t)value; \
} while (0)
#define CNP_BITWISE_READ_UNSIGNED(c_type) do { \
    c_type value; \
    memcpy(&value, source, sizeof(value)); \
    return (uint64_t)value; \
} while (0)

    switch (dtype) {
        case CNP_BOOL: {
            int8_t value;
            memcpy(&value, source, sizeof(value));
            return value != 0;
        }
        case CNP_BYTE:
            CNP_BITWISE_READ_SIGNED(int8_t);
        case CNP_UBYTE:
            CNP_BITWISE_READ_UNSIGNED(uint8_t);
        case CNP_SHORT:
            CNP_BITWISE_READ_SIGNED(int16_t);
        case CNP_USHORT:
            CNP_BITWISE_READ_UNSIGNED(uint16_t);
        case CNP_INT:
            CNP_BITWISE_READ_SIGNED(int32_t);
        case CNP_UINT:
            CNP_BITWISE_READ_UNSIGNED(uint32_t);
        case CNP_LONG:
        case CNP_LONGLONG:
            CNP_BITWISE_READ_SIGNED(int64_t);
        case CNP_ULONG:
        case CNP_ULONGLONG:
            CNP_BITWISE_READ_UNSIGNED(uint64_t);
        default:
            return 0;
    }

#undef CNP_BITWISE_READ_SIGNED
#undef CNP_BITWISE_READ_UNSIGNED
}

static void bitwise_write_bits(
    void *destination, CNP_TYPE dtype, uint64_t bits) {
    switch (dtype) {
        case CNP_BOOL: {
            uint8_t value = bits != 0;
            memcpy(destination, &value, sizeof(value));
            break;
        }
        case CNP_BYTE:
        case CNP_UBYTE: {
            uint8_t value = (uint8_t)bits;
            memcpy(destination, &value, sizeof(value));
            break;
        }
        case CNP_SHORT:
        case CNP_USHORT: {
            uint16_t value = (uint16_t)bits;
            memcpy(destination, &value, sizeof(value));
            break;
        }
        case CNP_INT:
        case CNP_UINT: {
            uint32_t value = (uint32_t)bits;
            memcpy(destination, &value, sizeof(value));
            break;
        }
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
            memcpy(destination, &bits, sizeof(bits));
            break;
        default:
            break;
    }
}

static const void* bitwise_element_pointer(
    const CnpArray *array,
    const int64_t *result_coordinates,
    int result_ndim) {
    int coordinate_offset = result_ndim - array->ndim;
    int64_t byte_offset = array->offset;
    for (int axis = 0; axis < array->ndim; ++axis) {
        int64_t coordinate =
            result_coordinates[coordinate_offset + axis];
        if (array->shape[axis] == 1) coordinate = 0;
        byte_offset += coordinate * array->strides[axis];
    }
    return (const char*)array->data + byte_offset;
}

static CNP_STATUS bitwise_validate_binary(
    CnpBitwiseOperation operation,
    const CnpArray *left,
    const CnpArray *right,
    const char *function_name,
    const char *operation_name,
    CNP_TYPE *result_dtype) {
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays are required");
        return CNP_ERR_GENERIC;
    }
    if (!left->dtype || !right->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "left and right arrays must have dtypes");
        return CNP_ERR_TYPE;
    }
    if ((left->size > 0 && !left->data) ||
            (right->size > 0 && !right->data)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays require data buffers");
        return CNP_ERR_GENERIC;
    }
    CNP_TYPE left_dtype = left->dtype->type_num;
    CNP_TYPE right_dtype = right->dtype->type_num;
    if (!bitwise_dtype_supported(left_dtype)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtype %d does not support %s",
            (int)left_dtype, operation_name);
        return CNP_ERR_TYPE;
    }
    if (!bitwise_dtype_supported(right_dtype)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtype %d does not support %s",
            (int)right_dtype, operation_name);
        return CNP_ERR_TYPE;
    }
    CNP_TYPE promoted = cnp_promote_type(left_dtype, right_dtype);
    if ((operation == CNP_BITWISE_LEFT_SHIFT ||
            operation == CNP_BITWISE_RIGHT_SHIFT) &&
            promoted == CNP_BOOL)
        promoted = CNP_BYTE;
    if (!bitwise_dtype_supported(promoted)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input dtypes %d and %d do not have a supported %s loop",
            (int)left_dtype, (int)right_dtype, operation_name);
        return CNP_ERR_TYPE;
    }
    if (!cnp_can_broadcast(left, right)) {
        cnp_set_error(
            CNP_ERR_BROADCAST, function_name,
            "left and right arrays cannot be broadcast together");
        return CNP_ERR_BROADCAST;
    }
    *result_dtype = promoted;
    return CNP_OK;
}

static CnpArray* bitwise_binary_result(
    const CnpArray *left,
    const CnpArray *right,
    CNP_TYPE dtype) {
    int result_ndim = left->ndim > right->ndim
        ? left->ndim : right->ndim;
    int64_t result_shape[CNP_MAXDIMS];
    for (int axis = 0; axis < result_ndim; ++axis) {
        int64_t left_dimension = axis < result_ndim - left->ndim
            ? 1 : left->shape[axis - (result_ndim - left->ndim)];
        int64_t right_dimension = axis < result_ndim - right->ndim
            ? 1 : right->shape[axis - (result_ndim - right->ndim)];
        result_shape[axis] = left_dimension == 1
            ? right_dimension : left_dimension;
    }
    return cnp_array_new(
        result_ndim, result_shape, dtype,
        cnp_logical_result_order(left, right));
}

static uint64_t bitwise_apply_shift(
    CnpBitwiseOperation operation,
    uint64_t left_bits,
    uint64_t right_bits,
    uint64_t mask,
    int width,
    bool result_signed) {
    left_bits &= mask;
    right_bits &= mask;
    uint64_t sign_bit = UINT64_C(1) << (width - 1);
    bool negative_count = result_signed &&
        (right_bits & sign_bit) != 0;
    bool negative_left = result_signed &&
        (left_bits & sign_bit) != 0;
    if (negative_count || right_bits >= (uint64_t)width) {
        if (operation == CNP_BITWISE_RIGHT_SHIFT && negative_left)
            return mask;
        return 0;
    }

    unsigned int count = (unsigned int)right_bits;
    if (operation == CNP_BITWISE_LEFT_SHIFT)
        return (left_bits << count) & mask;
    if (count == 0) return left_bits;
    uint64_t result = left_bits >> count;
    if (negative_left)
        result |= mask ^ (mask >> count);
    return result & mask;
}

static uint64_t bitwise_apply_binary(
    CnpBitwiseOperation operation,
    uint64_t left_bits,
    uint64_t right_bits,
    CNP_TYPE result_dtype) {
    uint64_t mask = bitwise_mask(result_dtype);
    left_bits &= mask;
    right_bits &= mask;
    switch (operation) {
        case CNP_BITWISE_AND:
            return left_bits & right_bits;
        case CNP_BITWISE_OR:
            return left_bits | right_bits;
        case CNP_BITWISE_XOR:
            return left_bits ^ right_bits;
        default:
            break;
    }
    int width = cnp_dtype_itemsize(result_dtype) * 8;
    bool result_signed = result_dtype != CNP_BOOL &&
        !cnp_type_is_unsigned(result_dtype);
    return bitwise_apply_shift(
        operation, left_bits, right_bits,
        mask, width, result_signed);
}

static void bitwise_binary_bytes(
    CnpBitwiseOperation operation,
    const uint8_t *left,
    const uint8_t *right,
    uint8_t *result,
    int64_t byte_count) {
    int64_t index = 0;
    for (; index + 7 < byte_count; index += 8) {
        uint64_t left_bits;
        uint64_t right_bits;
        uint64_t result_bits;
        memcpy(&left_bits, left + index, sizeof(left_bits));
        memcpy(&right_bits, right + index, sizeof(right_bits));
        result_bits = operation == CNP_BITWISE_AND
            ? left_bits & right_bits
            : operation == CNP_BITWISE_OR
            ? left_bits | right_bits
            : left_bits ^ right_bits;
        memcpy(result + index, &result_bits, sizeof(result_bits));
    }
    for (; index < byte_count; ++index) {
        result[index] = operation == CNP_BITWISE_AND
            ? left[index] & right[index]
            : operation == CNP_BITWISE_OR
            ? left[index] | right[index]
            : left[index] ^ right[index];
    }
}

static bool bitwise_shift_contiguous(
    CnpBitwiseOperation operation,
    const CnpArray *left,
    const CnpArray *right,
    CnpArray *result) {
    if (operation != CNP_BITWISE_LEFT_SHIFT &&
            operation != CNP_BITWISE_RIGHT_SHIFT)
        return false;
    bool c_contiguous =
        (left->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (right->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_C_CONTIGUOUS);
    bool f_contiguous =
        (left->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (right->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_F_CONTIGUOUS);
    CNP_TYPE dtype = result->dtype->type_num;
    if ((!c_contiguous && !f_contiguous) ||
            !same_shape(left, right) || !same_shape(left, result) ||
            left->dtype->type_num != dtype ||
            right->dtype->type_num != dtype ||
            dtype == CNP_BOOL)
        return false;

    const uint8_t *left_data =
        (const uint8_t*)left->data + left->offset;
    const uint8_t *right_data =
        (const uint8_t*)right->data + right->offset;
    uint8_t *result_data =
        (uint8_t*)result->data + result->offset;
    uint64_t mask = bitwise_mask(dtype);
    int width = result->dtype->elsize * 8;
    bool result_signed = !cnp_type_is_unsigned(dtype);

#define CNP_BITWISE_SHIFT_LOOP(read_type, write_type) do { \
    for (int64_t index = 0; index < result->size; ++index) { \
        read_type left_value; \
        read_type right_value; \
        memcpy(&left_value, \
            left_data + index * sizeof(read_type), sizeof(left_value)); \
        memcpy(&right_value, \
            right_data + index * sizeof(read_type), sizeof(right_value)); \
        uint64_t result_bits = bitwise_apply_shift( \
            operation, (uint64_t)left_value, (uint64_t)right_value, \
            mask, width, result_signed); \
        write_type output = (write_type)result_bits; \
        memcpy(result_data + index * sizeof(write_type), \
            &output, sizeof(output)); \
    } \
    return true; \
} while (0)

    switch (dtype) {
        case CNP_BYTE:
            CNP_BITWISE_SHIFT_LOOP(int8_t, uint8_t);
        case CNP_UBYTE:
            CNP_BITWISE_SHIFT_LOOP(uint8_t, uint8_t);
        case CNP_SHORT:
            CNP_BITWISE_SHIFT_LOOP(int16_t, uint16_t);
        case CNP_USHORT:
            CNP_BITWISE_SHIFT_LOOP(uint16_t, uint16_t);
        case CNP_INT:
            CNP_BITWISE_SHIFT_LOOP(int32_t, uint32_t);
        case CNP_UINT:
            CNP_BITWISE_SHIFT_LOOP(uint32_t, uint32_t);
        case CNP_LONG:
        case CNP_LONGLONG:
            CNP_BITWISE_SHIFT_LOOP(int64_t, uint64_t);
        case CNP_ULONG:
        case CNP_ULONGLONG:
            CNP_BITWISE_SHIFT_LOOP(uint64_t, uint64_t);
        default:
            break;
    }

#undef CNP_BITWISE_SHIFT_LOOP
    return false;
}

static CnpArray* bitwise_binary_arrays(
    CnpBitwiseOperation operation,
    const CnpArray *left,
    const CnpArray *right,
    const char *function_name,
    const char *operation_name) {
    CNP_TYPE result_dtype;
    if (bitwise_validate_binary(
            operation, left, right, function_name,
            operation_name, &result_dtype) != CNP_OK)
        return NULL;
    CnpArray *result = bitwise_binary_result(
        left, right, result_dtype);
    if (!result || result->size == 0) return result;

    if (operation <= CNP_BITWISE_XOR && result_dtype != CNP_BOOL &&
            left->dtype->type_num == result_dtype &&
            right->dtype->type_num == result_dtype &&
            same_shape(left, right) &&
            (left->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            (right->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            (result->flags & CNP_ARRAY_C_CONTIGUOUS)) {
        bitwise_binary_bytes(
            operation,
            (const uint8_t*)left->data + left->offset,
            (const uint8_t*)right->data + right->offset,
            (uint8_t*)result->data + result->offset,
            result->size * result->dtype->elsize);
        return result;
    }
    if (bitwise_shift_contiguous(
            operation, left, right, result))
        return result;

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        uint64_t left_bits = bitwise_read_bits(
            bitwise_element_pointer(left, coordinates, result->ndim),
            left->dtype->type_num);
        uint64_t right_bits = bitwise_read_bits(
            bitwise_element_pointer(right, coordinates, result->ndim),
            right->dtype->type_num);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        bitwise_write_bits(
            (char*)result->data + result_offset,
            result_dtype,
            bitwise_apply_binary(
                operation, left_bits, right_bits, result_dtype));
        for (int axis = result->ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < result->shape[axis]) break;
            coordinates[axis] = 0;
        }
    }
    return result;
}

static CnpArray* bitwise_unary_array(
    const CnpArray *source,
    const char *function_name,
    const char *operation_name) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "source array is required");
        return NULL;
    }
    if (!source->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name, "source array must have a dtype");
        return NULL;
    }
    if (source->size > 0 && !source->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }
    CNP_TYPE dtype = source->dtype->type_num;
    if (!bitwise_dtype_supported(dtype)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtype %d does not support %s",
            (int)dtype, operation_name);
        return NULL;
    }
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, dtype,
        cnp_logical_result_order(source, source));
    if (!result || result->size == 0) return result;

    if (dtype != CNP_BOOL &&
            (source->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            (result->flags & CNP_ARRAY_C_CONTIGUOUS)) {
        const uint8_t *input =
            (const uint8_t*)source->data + source->offset;
        uint8_t *output = (uint8_t*)result->data + result->offset;
        int64_t byte_count = result->size * result->dtype->elsize;
        for (int64_t index = 0; index < byte_count; ++index)
            output[index] = (uint8_t)~input[index];
        return result;
    }

    uint64_t mask = bitwise_mask(dtype);
    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        uint64_t bits = bitwise_read_bits(
            (const char*)source->data + source_offset, dtype);
        uint64_t inverted = dtype == CNP_BOOL ? !bits : (~bits & mask);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        bitwise_write_bits(
            (char*)result->data + result_offset, dtype, inverted);
        for (int axis = source->ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < source->shape[axis]) break;
            coordinates[axis] = 0;
        }
    }
    return result;
}

/* =========================================================================
 * Integer greatest-common-divisor and least-common-multiple operations
 * ========================================================================= */
typedef enum {
    CNP_INTEGER_GCD,
    CNP_INTEGER_LCM
} CnpIntegerGcdLcmOperation;

static CNP_STATUS integer_validate_binary(
    const CnpArray *left,
    const CnpArray *right,
    const char *function_name,
    const char *operation_name,
    CNP_TYPE *result_dtype) {
    if (!left || !right) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays are required");
        return CNP_ERR_GENERIC;
    }
    if (!left->dtype || !right->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "left and right arrays must have dtypes");
        return CNP_ERR_TYPE;
    }
    if ((left->size > 0 && !left->data) ||
            (right->size > 0 && !right->data)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays require data buffers");
        return CNP_ERR_GENERIC;
    }

    CNP_TYPE left_dtype = left->dtype->type_num;
    CNP_TYPE right_dtype = right->dtype->type_num;
    bool left_supported =
        left_dtype == CNP_BOOL || cnp_type_is_integer(left_dtype);
    bool right_supported =
        right_dtype == CNP_BOOL || cnp_type_is_integer(right_dtype);
    if (!left_supported || !right_supported) {
        CNP_TYPE unsupported = left_supported ? right_dtype : left_dtype;
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtype %d does not support %s",
            (int)unsupported, operation_name);
        return CNP_ERR_TYPE;
    }

    CNP_TYPE promoted = cnp_promote_type(left_dtype, right_dtype);
    if (!cnp_type_is_integer(promoted)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input dtypes %d and %d do not have a supported %s loop",
            (int)left_dtype, (int)right_dtype, operation_name);
        return CNP_ERR_TYPE;
    }
    if (!cnp_can_broadcast(left, right)) {
        cnp_set_error(
            CNP_ERR_BROADCAST, function_name,
            "left and right arrays cannot be broadcast together");
        return CNP_ERR_BROADCAST;
    }
    *result_dtype = promoted;
    return CNP_OK;
}

static uint64_t integer_magnitude_bits(
    uint64_t bits,
    uint64_t mask,
    uint64_t sign_bit,
    bool result_unsigned) {
    bits &= mask;
    if (result_unsigned) return bits;
    return (bits & sign_bit) == 0
        ? bits : ((~bits + UINT64_C(1)) & mask);
}

static uint64_t integer_gcd_bits(
    uint64_t left_magnitude, uint64_t right_magnitude) {
    while (right_magnitude != 0) {
        uint64_t remainder = left_magnitude % right_magnitude;
        left_magnitude = right_magnitude;
        right_magnitude = remainder;
    }
    return left_magnitude;
}

static uint64_t integer_gcd_lcm_bits(
    CnpIntegerGcdLcmOperation operation,
    uint64_t left_bits,
    uint64_t right_bits,
    uint64_t mask,
    uint64_t sign_bit,
    bool result_unsigned) {
    uint64_t left_magnitude = integer_magnitude_bits(
        left_bits, mask, sign_bit, result_unsigned);
    uint64_t right_magnitude = integer_magnitude_bits(
        right_bits, mask, sign_bit, result_unsigned);
    uint64_t greatest_common_divisor = integer_gcd_bits(
        left_magnitude, right_magnitude);
    if (operation == CNP_INTEGER_GCD)
        return greatest_common_divisor & mask;
    if (greatest_common_divisor == 0) return 0;
    return ((left_magnitude / greatest_common_divisor) *
        right_magnitude) & mask;
}

static bool integer_gcd_lcm_contiguous(
    CnpIntegerGcdLcmOperation operation,
    const CnpArray *left,
    const CnpArray *right,
    CnpArray *result) {
    if (result->size == 0) return true;
    bool c_contiguous =
        (left->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (right->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_C_CONTIGUOUS);
    bool f_contiguous =
        (left->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (right->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_F_CONTIGUOUS);
    CNP_TYPE dtype = result->dtype->type_num;
    if ((!c_contiguous && !f_contiguous) ||
            !same_shape(left, right) || !same_shape(left, result) ||
            left->dtype->type_num != dtype ||
            right->dtype->type_num != dtype)
        return false;

    uint64_t mask = bitwise_mask(dtype);
    uint64_t sign_bit = (mask >> 1) + UINT64_C(1);
    bool result_unsigned = cnp_type_is_unsigned(dtype);

    const uint8_t *left_data =
        (const uint8_t*)left->data + left->offset;
    const uint8_t *right_data =
        (const uint8_t*)right->data + right->offset;
    uint8_t *result_data =
        (uint8_t*)result->data + result->offset;

#define CNP_INTEGER_GCD_LCM_LOOP(read_type, write_type) do { \
    for (int64_t index = 0; index < result->size; ++index) { \
        read_type left_value; \
        read_type right_value; \
        memcpy(&left_value, \
            left_data + index * sizeof(read_type), sizeof(left_value)); \
        memcpy(&right_value, \
            right_data + index * sizeof(read_type), sizeof(right_value)); \
        uint64_t result_bits = integer_gcd_lcm_bits( \
            operation, (uint64_t)left_value, (uint64_t)right_value, \
            mask, sign_bit, result_unsigned); \
        write_type output = (write_type)result_bits; \
        memcpy(result_data + index * sizeof(write_type), \
            &output, sizeof(output)); \
    } \
    return true; \
} while (0)

    switch (dtype) {
        case CNP_BYTE:
            CNP_INTEGER_GCD_LCM_LOOP(int8_t, uint8_t);
        case CNP_UBYTE:
            CNP_INTEGER_GCD_LCM_LOOP(uint8_t, uint8_t);
        case CNP_SHORT:
            CNP_INTEGER_GCD_LCM_LOOP(int16_t, uint16_t);
        case CNP_USHORT:
            CNP_INTEGER_GCD_LCM_LOOP(uint16_t, uint16_t);
        case CNP_INT:
            CNP_INTEGER_GCD_LCM_LOOP(int32_t, uint32_t);
        case CNP_UINT:
            CNP_INTEGER_GCD_LCM_LOOP(uint32_t, uint32_t);
        case CNP_LONG:
        case CNP_LONGLONG:
            CNP_INTEGER_GCD_LCM_LOOP(int64_t, uint64_t);
        case CNP_ULONG:
        case CNP_ULONGLONG:
            CNP_INTEGER_GCD_LCM_LOOP(uint64_t, uint64_t);
        default:
            break;
    }

#undef CNP_INTEGER_GCD_LCM_LOOP
    return false;
}

static CnpArray* integer_gcd_lcm_arrays(
    CnpIntegerGcdLcmOperation operation,
    const CnpArray *left,
    const CnpArray *right,
    const char *function_name,
    const char *operation_name) {
    CNP_TYPE result_dtype;
    if (integer_validate_binary(
            left, right, function_name,
            operation_name, &result_dtype) != CNP_OK)
        return NULL;

    CnpArray *result = bitwise_binary_result(
        left, right, result_dtype);
    if (!result) return NULL;
    if (integer_gcd_lcm_contiguous(
            operation, left, right, result))
        return result;

    uint64_t mask = bitwise_mask(result_dtype);
    uint64_t sign_bit = (mask >> 1) + UINT64_C(1);
    bool result_unsigned = cnp_type_is_unsigned(result_dtype);
    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        uint64_t left_bits = bitwise_read_bits(
            bitwise_element_pointer(left, coordinates, result->ndim),
            left->dtype->type_num);
        uint64_t right_bits = bitwise_read_bits(
            bitwise_element_pointer(right, coordinates, result->ndim),
            right->dtype->type_num);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        bitwise_write_bits(
            (char*)result->data + result_offset,
            result_dtype,
            integer_gcd_lcm_bits(
                operation, left_bits, right_bits,
                mask, sign_bit, result_unsigned));
        for (int axis = result->ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < result->shape[axis]) break;
            coordinates[axis] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_gcd(
    const CnpArray *x1, const CnpArray *x2) {
    return integer_gcd_lcm_arrays(
        CNP_INTEGER_GCD, x1, x2, "cnp_gcd", "gcd");
}

CNP_API CnpArray* CNP_CALL cnp_lcm(
    const CnpArray *x1, const CnpArray *x2) {
    return integer_gcd_lcm_arrays(
        CNP_INTEGER_LCM, x1, x2, "cnp_lcm", "lcm");
}

CNP_API CnpArray* CNP_CALL cnp_bitwise_and(
    const CnpArray *a, const CnpArray *b) {
    return bitwise_binary_arrays(
        CNP_BITWISE_AND, a, b,
        "cnp_bitwise_and", "bitwise_and");
}

CNP_API CnpArray* CNP_CALL cnp_bitwise_or(
    const CnpArray *a, const CnpArray *b) {
    return bitwise_binary_arrays(
        CNP_BITWISE_OR, a, b,
        "cnp_bitwise_or", "bitwise_or");
}

CNP_API CnpArray* CNP_CALL cnp_bitwise_xor(
    const CnpArray *a, const CnpArray *b) {
    return bitwise_binary_arrays(
        CNP_BITWISE_XOR, a, b,
        "cnp_bitwise_xor", "bitwise_xor");
}

CNP_API CnpArray* CNP_CALL cnp_left_shift(
    const CnpArray *a, const CnpArray *b) {
    return bitwise_binary_arrays(
        CNP_BITWISE_LEFT_SHIFT, a, b,
        "cnp_left_shift", "left_shift");
}

CNP_API CnpArray* CNP_CALL cnp_right_shift(
    const CnpArray *a, const CnpArray *b) {
    return bitwise_binary_arrays(
        CNP_BITWISE_RIGHT_SHIFT, a, b,
        "cnp_right_shift", "right_shift");
}

CNP_API CnpArray* CNP_CALL cnp_invert(const CnpArray *a) {
    return bitwise_unary_array(a, "cnp_invert", "invert");
}

CNP_API CnpArray* CNP_CALL cnp_bitwise_not(const CnpArray *a) {
    return bitwise_unary_array(
        a, "cnp_bitwise_not", "bitwise_not");
}

/* =========================================================================
 * Special functions (nan/inf handling)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_nan_to_num(const CnpArray *a, double nan_val, double posinf_val, double neginf_val) {
    const char *function_name = "cnp_nan_to_num";
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            a, function_name, &ignored_nbytes)) return NULL;
    if (a->ndim < 0 || a->ndim > CNP_MAXDIMS ||
            (a->ndim > 0 && (!a->shape || !a->strides)) ||
            (a->size > 0 && !a->data)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source array shape metadata and data buffer must be valid");
        return NULL;
    }
    CNP_TYPE source_type = a->dtype->type_num;
    if (!(source_type == CNP_BOOL ||
          cnp_type_is_integer(source_type) ||
          cnp_type_is_float(source_type) ||
          cnp_type_is_complex(source_type))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array must have a represented numeric dtype");
        return NULL;
    }
    CnpArray *result = cnp_array_copy(a);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    if (source_type == CNP_BOOL || cnp_type_is_integer(source_type))
        return result;

    int itemsize = result->dtype->elsize;
    for (int64_t index = 0; index < result->size; ++index) {
        char *element = (char*)result->data + index * itemsize;
        cnp_clongdouble value = {0};
        CNP_STATUS status = cnp_cast_scalar_value(
            element, source_type,
            &value, CNP_CLONGDOUBLE, function_name);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            return NULL;
        }

        bool changed = false;
        if (isnan(value.real)) {
            value.real = nan_val;
            changed = true;
        } else if (isinf(value.real)) {
            value.real = value.real > 0.0L ? posinf_val : neginf_val;
            changed = true;
        }
        if (cnp_type_is_complex(source_type)) {
            if (isnan(value.imag)) {
                value.imag = nan_val;
                changed = true;
            } else if (isinf(value.imag)) {
                value.imag = value.imag > 0.0L ? posinf_val : neginf_val;
                changed = true;
            }
        }
        if (!changed) continue;

        status = cnp_cast_scalar_value(
            &value, CNP_CLONGDOUBLE,
            element, source_type, function_name);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            return NULL;
        }
    }
    return result;
}

typedef enum {
    CNP_PREDICATE_ISNAN,
    CNP_PREDICATE_ISINF,
    CNP_PREDICATE_ISFINITE,
    CNP_PREDICATE_SIGNBIT
} CnpArrayPredicateOperation;

static const char *predicate_operation_name(
    CnpArrayPredicateOperation operation) {
    switch (operation) {
        case CNP_PREDICATE_ISNAN: return "isnan";
        case CNP_PREDICATE_ISINF: return "isinf";
        case CNP_PREDICATE_ISFINITE: return "isfinite";
        case CNP_PREDICATE_SIGNBIT: return "signbit";
    }
    return "unknown predicate";
}

static CNP_STATUS predicate_element(
    CnpArrayPredicateOperation operation,
    const void *source,
    CNP_TYPE source_dtype,
    uint8_t *result,
    const char *function_name) {
    if (source_dtype == CNP_DATETIME || source_dtype == CNP_TIMEDELTA) {
        int64_t value;
        memcpy(&value, source, sizeof(value));
        bool is_nat = value == INT64_MIN;
        *result = operation == CNP_PREDICATE_ISNAN
            ? (uint8_t)is_nat
            : operation == CNP_PREDICATE_ISINF
            ? 0
            : (uint8_t)!is_nat;
        return CNP_OK;
    }

    CnpArithmeticValue value;
    CNP_STATUS status = arithmetic_read_value(
        source, source_dtype, &value, function_name);
    if (status != CNP_OK) return status;
    switch (operation) {
        case CNP_PREDICATE_ISNAN:
            *result = value.kind == CNP_ARITHMETIC_COMPLEX
                ? (uint8_t)(isnan(value.real) || isnan(value.imaginary))
                : value.kind == CNP_ARITHMETIC_FLOATING
                ? (uint8_t)isnan(value.real)
                : 0;
            return CNP_OK;
        case CNP_PREDICATE_ISINF:
            *result = value.kind == CNP_ARITHMETIC_COMPLEX
                ? (uint8_t)(isinf(value.real) || isinf(value.imaginary))
                : value.kind == CNP_ARITHMETIC_FLOATING
                ? (uint8_t)isinf(value.real)
                : 0;
            return CNP_OK;
        case CNP_PREDICATE_ISFINITE:
            *result = value.kind == CNP_ARITHMETIC_COMPLEX
                ? (uint8_t)(isfinite(value.real) &&
                    isfinite(value.imaginary))
                : value.kind == CNP_ARITHMETIC_FLOATING
                ? (uint8_t)isfinite(value.real)
                : 1;
            return CNP_OK;
        case CNP_PREDICATE_SIGNBIT:
            *result = value.kind == CNP_ARITHMETIC_SIGNED
                ? (uint8_t)(value.signed_value < 0)
                : value.kind == CNP_ARITHMETIC_FLOATING
                ? (uint8_t)signbit(value.real)
                : 0;
            return CNP_OK;
    }
    cnp_set_error(
        CNP_ERR_GENERIC, function_name,
        "invalid internal predicate operation %d", (int)operation);
    return CNP_ERR_GENERIC;
}

static uint64_t predicate_double_bits(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static bool predicate_contiguous_double(
    const CnpArray *source,
    CnpArray *result,
    CnpArrayPredicateOperation operation) {
    if (source->dtype->type_num != CNP_DOUBLE) return false;
    bool c_contiguous =
        (source->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_C_CONTIGUOUS);
    bool f_contiguous =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_F_CONTIGUOUS);
    if (!c_contiguous && !f_contiguous) return false;
    if (result->size == 0) return true;

    const double *input = (const double*)(
        (const char*)source->data + source->offset);
    uint8_t *output = (uint8_t*)result->data + result->offset;
    const uint64_t exponent_mask = UINT64_C(0x7ff0000000000000);
    const uint64_t fraction_mask = UINT64_C(0x000fffffffffffff);
    int64_t index = 0;

    switch (operation) {
        case CNP_PREDICATE_ISNAN:
            for (; index + 3 < result->size; index += 4) {
                uint64_t bits0 = predicate_double_bits(input[index]);
                uint64_t bits1 = predicate_double_bits(input[index + 1]);
                uint64_t bits2 = predicate_double_bits(input[index + 2]);
                uint64_t bits3 = predicate_double_bits(input[index + 3]);
                output[index] = (uint8_t)(
                    (bits0 & exponent_mask) == exponent_mask &&
                    (bits0 & fraction_mask) != 0);
                output[index + 1] = (uint8_t)(
                    (bits1 & exponent_mask) == exponent_mask &&
                    (bits1 & fraction_mask) != 0);
                output[index + 2] = (uint8_t)(
                    (bits2 & exponent_mask) == exponent_mask &&
                    (bits2 & fraction_mask) != 0);
                output[index + 3] = (uint8_t)(
                    (bits3 & exponent_mask) == exponent_mask &&
                    (bits3 & fraction_mask) != 0);
            }
            for (; index < result->size; ++index) {
                uint64_t bits = predicate_double_bits(input[index]);
                output[index] = (uint8_t)(
                    (bits & exponent_mask) == exponent_mask &&
                    (bits & fraction_mask) != 0);
            }
            return true;
        case CNP_PREDICATE_ISINF:
            for (; index + 3 < result->size; index += 4) {
                uint64_t bits0 = predicate_double_bits(input[index]);
                uint64_t bits1 = predicate_double_bits(input[index + 1]);
                uint64_t bits2 = predicate_double_bits(input[index + 2]);
                uint64_t bits3 = predicate_double_bits(input[index + 3]);
                output[index] = (uint8_t)(
                    (bits0 & (exponent_mask | fraction_mask)) ==
                    exponent_mask);
                output[index + 1] = (uint8_t)(
                    (bits1 & (exponent_mask | fraction_mask)) ==
                    exponent_mask);
                output[index + 2] = (uint8_t)(
                    (bits2 & (exponent_mask | fraction_mask)) ==
                    exponent_mask);
                output[index + 3] = (uint8_t)(
                    (bits3 & (exponent_mask | fraction_mask)) ==
                    exponent_mask);
            }
            for (; index < result->size; ++index) {
                uint64_t bits = predicate_double_bits(input[index]);
                output[index] = (uint8_t)(
                    (bits & (exponent_mask | fraction_mask)) ==
                    exponent_mask);
            }
            return true;
        case CNP_PREDICATE_ISFINITE:
            for (; index + 3 < result->size; index += 4) {
                output[index] = (uint8_t)(
                    (predicate_double_bits(input[index]) & exponent_mask) !=
                    exponent_mask);
                output[index + 1] = (uint8_t)(
                    (predicate_double_bits(input[index + 1]) &
                        exponent_mask) != exponent_mask);
                output[index + 2] = (uint8_t)(
                    (predicate_double_bits(input[index + 2]) &
                        exponent_mask) != exponent_mask);
                output[index + 3] = (uint8_t)(
                    (predicate_double_bits(input[index + 3]) &
                        exponent_mask) != exponent_mask);
            }
            for (; index < result->size; ++index) {
                output[index] = (uint8_t)(
                    (predicate_double_bits(input[index]) & exponent_mask) !=
                    exponent_mask);
            }
            return true;
        case CNP_PREDICATE_SIGNBIT:
            for (; index + 3 < result->size; index += 4) {
                output[index] = (uint8_t)(
                    predicate_double_bits(input[index]) >> 63);
                output[index + 1] = (uint8_t)(
                    predicate_double_bits(input[index + 1]) >> 63);
                output[index + 2] = (uint8_t)(
                    predicate_double_bits(input[index + 2]) >> 63);
                output[index + 3] = (uint8_t)(
                    predicate_double_bits(input[index + 3]) >> 63);
            }
            for (; index < result->size; ++index) {
                output[index] = (uint8_t)(
                    predicate_double_bits(input[index]) >> 63);
            }
            return true;
    }
    return false;
}

static CnpArray *predicate_arrays(
    const CnpArray *a,
    CnpArrayPredicateOperation operation,
    const char *function_name) {
    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!a->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array must have a dtype");
        return NULL;
    }
    CNP_TYPE source_dtype = a->dtype->type_num;
    bool supported = operation == CNP_PREDICATE_SIGNBIT
        ? source_dtype == CNP_BOOL || cnp_type_is_integer(source_dtype) ||
            cnp_type_is_float(source_dtype)
        : arithmetic_dtype_supported(source_dtype) ||
            source_dtype == CNP_DATETIME ||
            source_dtype == CNP_TIMEDELTA;
    if (!supported) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtype %s does not support %s",
            a->dtype->name, predicate_operation_name(operation));
        return NULL;
    }
    if (a->size > 0 && !a->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array requires a data buffer");
        return NULL;
    }
    CnpArray *result = cnp_array_new(
        a->ndim, a->shape, CNP_BOOL,
        arithmetic_result_order(a, a));
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (predicate_contiguous_double(a, result, operation)) return result;
    int64_t coords[CNP_MAXDIMS] = {0};
    for (int64_t i = 0; i < a->size; i++) {
        int64_t src_offset = a->offset + cnp_multi_to_offset(
            a->ndim, coords, a->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coords, result->strides);
        CNP_STATUS status = predicate_element(
            operation,
            (const char*)a->data + src_offset,
            source_dtype,
            (uint8_t*)result->data + result_offset,
            function_name);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        for (int d = a->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < a->shape[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_isnan(const CnpArray *a) {
    return predicate_arrays(a, CNP_PREDICATE_ISNAN, "cnp_isnan");
}

CNP_API CnpArray* CNP_CALL cnp_isinf(const CnpArray *a) {
    return predicate_arrays(a, CNP_PREDICATE_ISINF, "cnp_isinf");
}

CNP_API CnpArray* CNP_CALL cnp_isfinite(const CnpArray *a) {
    return predicate_arrays(a, CNP_PREDICATE_ISFINITE, "cnp_isfinite");
}

CNP_API CnpArray* CNP_CALL cnp_isnan_arr(const CnpArray *a) {
    return predicate_arrays(a, CNP_PREDICATE_ISNAN, "cnp_isnan_arr");
}

CNP_API CnpArray* CNP_CALL cnp_isinf_arr(const CnpArray *a) {
    return predicate_arrays(a, CNP_PREDICATE_ISINF, "cnp_isinf_arr");
}

CNP_API CnpArray* CNP_CALL cnp_isfinite_arr(const CnpArray *a) {
    return predicate_arrays(a, CNP_PREDICATE_ISFINITE, "cnp_isfinite_arr");
}

CNP_API CnpArray* CNP_CALL cnp_signbit(const CnpArray *a) {
    return predicate_arrays(a, CNP_PREDICATE_SIGNBIT, "cnp_signbit");
}
