/**
 * cnumpy type conversion, casting, and additional array operations
 * Corresponds to numpy: can_cast, promote_types, result_type, copyto,
 *   where, matrix_power, matrix_rank, nanmedian, nanpercentile,
 *   finfo, iinfo, nextafter, spacing, copysign, frexp, ldexp, modf
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>
#include <float.h>
#include <emmintrin.h>

static double cnp_typeconv_positive_infinity(void) {
    uint64_t bits = UINT64_C(0x7ff0000000000000);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float cnp_typeconv_positive_infinity_float(void) {
    uint32_t bits = UINT32_C(0x7f800000);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float cnp_typeconv_quiet_nan_float(void) {
    uint32_t bits = UINT32_C(0x7fc00000);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static double cnp_typeconv_quiet_nan(void) {
    uint64_t bits = UINT64_C(0x7ff8000000000000);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static CNP_TYPE cnp_finfo_component_type(
    CNP_TYPE dtype,
    const char *function
) {
    switch (dtype) {
        case CNP_HALF:
            return CNP_HALF;
        case CNP_FLOAT:
        case CNP_CFLOAT:
            return CNP_FLOAT;
        case CNP_DOUBLE:
        case CNP_LONGDOUBLE:
        case CNP_CDOUBLE:
        case CNP_CLONGDOUBLE:
            return CNP_DOUBLE;
        default:
            cnp_set_error(
                CNP_ERR_TYPE, function,
                "dtype %d is not an inexact dtype", (int)dtype);
            return CNP_NOTYPE;
    }
}

/* =========================================================================
 * cnp_can_cast - Check if one type can be cast to another safely
 * Note: cnp_dtype_can_cast already exists in core.c with same functionality
 * This is an alias with numpy-compatible name
 * ========================================================================= */

static bool typeconv_valid_type(CNP_TYPE type) {
    return type > CNP_NOTYPE && type < CNP_NTYPES;
}

static int typeconv_numeric_category(CNP_TYPE type) {
    if (type == CNP_BOOL)
        return 0;
    if (type >= CNP_BYTE && type <= CNP_ULONGLONG)
        return 1;
    if (type == CNP_HALF || type == CNP_FLOAT ||
            type == CNP_DOUBLE || type == CNP_LONGDOUBLE)
        return 2;
    if (type == CNP_CFLOAT || type == CNP_CDOUBLE ||
            type == CNP_CLONGDOUBLE)
        return 3;
    return -1;
}

static bool typeconv_unsigned_integer(CNP_TYPE type) {
    return type == CNP_UBYTE || type == CNP_USHORT ||
        type == CNP_UINT || type == CNP_ULONG ||
        type == CNP_ULONGLONG;
}

static int typeconv_integer_bits(CNP_TYPE type) {
    switch (type) {
        case CNP_BYTE:
        case CNP_UBYTE:
            return 8;
        case CNP_SHORT:
        case CNP_USHORT:
            return 16;
        case CNP_INT:
        case CNP_UINT:
            return 32;
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
            return 64;
        default:
            return 0;
    }
}

static int typeconv_float_rank(CNP_TYPE type) {
    switch (type) {
        case CNP_HALF:
            return 0;
        case CNP_FLOAT:
            return 1;
        case CNP_DOUBLE:
            return 2;
        case CNP_LONGDOUBLE:
            return 3;
        case CNP_CFLOAT:
            return 1;
        case CNP_CDOUBLE:
            return 2;
        case CNP_CLONGDOUBLE:
            return 3;
        default:
            return -1;
    }
}

static bool typeconv_integer_safe_to_float(
    CNP_TYPE source, CNP_TYPE destination) {
    int bits = typeconv_integer_bits(source);
    switch (destination) {
        case CNP_HALF:
            return bits <= 8;
        case CNP_FLOAT:
            return bits <= 16;
        case CNP_DOUBLE:
        case CNP_LONGDOUBLE:
            return bits <= 64;
        default:
            return false;
    }
}

static bool typeconv_numeric_safe_cast(
    CNP_TYPE source, CNP_TYPE destination) {
    if (source == destination)
        return true;
    int source_category = typeconv_numeric_category(source);
    int destination_category = typeconv_numeric_category(destination);
    if (source_category < 0 || destination_category < 0)
        return false;
    if (source == CNP_BOOL)
        return true;
    if (destination == CNP_BOOL)
        return false;

    if (source_category == 1) {
        if (destination_category == 1) {
            int source_bits = typeconv_integer_bits(source);
            int destination_bits = typeconv_integer_bits(destination);
            bool source_unsigned = typeconv_unsigned_integer(source);
            bool destination_unsigned =
                typeconv_unsigned_integer(destination);
            if (source_unsigned == destination_unsigned)
                return destination_bits >= source_bits;
            if (!source_unsigned)
                return false;
            return destination_bits > source_bits;
        }
        if (destination_category == 2)
            return typeconv_integer_safe_to_float(source, destination);
        if (destination_category == 3) {
            CNP_TYPE component = destination == CNP_CFLOAT
                ? CNP_FLOAT
                : destination == CNP_CDOUBLE
                    ? CNP_DOUBLE
                    : CNP_LONGDOUBLE;
            return typeconv_integer_safe_to_float(source, component);
        }
        return false;
    }
    if (source_category == 2) {
        if (destination_category == 2)
            return typeconv_float_rank(destination) >=
                typeconv_float_rank(source);
        if (destination_category == 3)
            return typeconv_float_rank(destination) >=
                typeconv_float_rank(source);
        return false;
    }
    if (source_category == 3 && destination_category == 3)
        return typeconv_float_rank(destination) >=
            typeconv_float_rank(source);
    return false;
}

static CNP_TYPE typeconv_min_scalar_integer_signed(int64_t value) {
    if (value >= 0) {
        uint64_t unsigned_value = (uint64_t)value;
        if (unsigned_value <= UINT8_MAX) return CNP_UBYTE;
        if (unsigned_value <= UINT16_MAX) return CNP_USHORT;
        if (unsigned_value <= UINT32_MAX) return CNP_UINT;
        return CNP_ULONGLONG;
    }
    if (value >= INT8_MIN) return CNP_BYTE;
    if (value >= INT16_MIN) return CNP_SHORT;
    if (value >= INT32_MIN) return CNP_INT;
    return CNP_LONGLONG;
}

static CNP_TYPE typeconv_min_scalar_integer_unsigned(uint64_t value) {
    if (value <= UINT8_MAX) return CNP_UBYTE;
    if (value <= UINT16_MAX) return CNP_USHORT;
    if (value <= UINT32_MAX) return CNP_UINT;
    return CNP_ULONGLONG;
}

static CNP_TYPE typeconv_min_scalar_float(long double value) {
    if (!isfinite(value) || fabsl(value) < 65000.0L)
        return CNP_HALF;
    if (fabsl(value) < 3.4e38L)
        return CNP_FLOAT;
    if (fabsl(value) <= (long double)DBL_MAX)
        return CNP_DOUBLE;
    return CNP_LONGDOUBLE;
}

static CNP_TYPE typeconv_min_scalar_type(const CnpArray *array) {
    const char *source = (const char*)array->data + array->offset;
    switch (array->dtype->type_num) {
        case CNP_BOOL:
            return CNP_BOOL;
        case CNP_BYTE:
            return typeconv_min_scalar_integer_signed(
                (int64_t)*(const int8_t*)source);
        case CNP_UBYTE:
            return typeconv_min_scalar_integer_unsigned(
                (uint64_t)*(const uint8_t*)source);
        case CNP_SHORT:
            return typeconv_min_scalar_integer_signed(
                (int64_t)*(const int16_t*)source);
        case CNP_USHORT:
            return typeconv_min_scalar_integer_unsigned(
                (uint64_t)*(const uint16_t*)source);
        case CNP_INT:
            return typeconv_min_scalar_integer_signed(
                (int64_t)*(const int32_t*)source);
        case CNP_UINT:
            return typeconv_min_scalar_integer_unsigned(
                (uint64_t)*(const uint32_t*)source);
        case CNP_LONG:
        case CNP_LONGLONG:
            return typeconv_min_scalar_integer_signed(
                *(const int64_t*)source);
        case CNP_ULONG:
        case CNP_ULONGLONG:
            return typeconv_min_scalar_integer_unsigned(
                *(const uint64_t*)source);
        case CNP_HALF:
            return typeconv_min_scalar_float((long double)
                cnp_half_to_float(*(const uint16_t*)source));
        case CNP_FLOAT:
            return typeconv_min_scalar_float(
                (long double)*(const float*)source);
        case CNP_DOUBLE:
            return typeconv_min_scalar_float(
                (long double)*(const double*)source);
        case CNP_LONGDOUBLE:
            return typeconv_min_scalar_float(
                *(const long double*)source);
        case CNP_CFLOAT: {
            const cnp_cfloat *value = (const cnp_cfloat*)source;
            return !isfinite(value->real) || !isfinite(value->imag) ||
                (fabsf(value->real) < 3.4e38F &&
                 fabsf(value->imag) < 3.4e38F)
                ? CNP_CFLOAT : CNP_CDOUBLE;
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *value = (const cnp_cdouble*)source;
            return !isfinite(value->real) || !isfinite(value->imag) ||
                (fabs(value->real) < 3.4e38 &&
                 fabs(value->imag) < 3.4e38)
                ? CNP_CFLOAT : CNP_CDOUBLE;
        }
        case CNP_CLONGDOUBLE: {
            const cnp_clongdouble *value =
                (const cnp_clongdouble*)source;
            if (!isfinite(value->real) || !isfinite(value->imag) ||
                    (fabsl(value->real) < 3.4e38L &&
                     fabsl(value->imag) < 3.4e38L))
                return CNP_CFLOAT;
            if (fabsl(value->real) <= DBL_MAX &&
                    fabsl(value->imag) <= DBL_MAX)
                return CNP_CDOUBLE;
            return CNP_CLONGDOUBLE;
        }
        default:
            return array->dtype->type_num;
    }
}

static CNP_TYPE typeconv_result_input_type(
    const CnpArray *array, int maximum_array_category) {
    CNP_TYPE type = array->dtype->type_num;
    int category = typeconv_numeric_category(type);
    if (array->ndim == 0 && maximum_array_category >= category &&
            category >= 0)
        return typeconv_min_scalar_type(array);
    return type;
}

static bool typeconv_scalar_integer_fits(
    const CnpArray *array, CNP_TYPE destination) {
    int destination_bits = typeconv_integer_bits(destination);
    if (destination_bits == 0) return false;

    const char *source = (const char*)array->data + array->offset;
    bool source_unsigned = typeconv_unsigned_integer(
        array->dtype->type_num);
    int64_t signed_value = 0;
    uint64_t unsigned_value = 0;
    switch (array->dtype->type_num) {
        case CNP_BYTE:
            signed_value = *(const int8_t*)source;
            break;
        case CNP_SHORT:
            signed_value = *(const int16_t*)source;
            break;
        case CNP_INT:
            signed_value = *(const int32_t*)source;
            break;
        case CNP_LONG:
        case CNP_LONGLONG:
            signed_value = *(const int64_t*)source;
            break;
        case CNP_UBYTE:
            unsigned_value = *(const uint8_t*)source;
            break;
        case CNP_USHORT:
            unsigned_value = *(const uint16_t*)source;
            break;
        case CNP_UINT:
            unsigned_value = *(const uint32_t*)source;
            break;
        case CNP_ULONG:
        case CNP_ULONGLONG:
            unsigned_value = *(const uint64_t*)source;
            break;
        default:
            return false;
    }

    if (typeconv_unsigned_integer(destination)) {
        uint64_t maximum = destination_bits == 64
            ? UINT64_MAX
            : (UINT64_C(1) << destination_bits) - UINT64_C(1);
        if (source_unsigned) return unsigned_value <= maximum;
        return signed_value >= 0 &&
            (uint64_t)signed_value <= maximum;
    }

    int64_t minimum = destination_bits == 64
        ? INT64_MIN
        : -(INT64_C(1) << (destination_bits - 1));
    int64_t maximum = destination_bits == 64
        ? INT64_MAX
        : (INT64_C(1) << (destination_bits - 1)) - INT64_C(1);
    if (source_unsigned) return unsigned_value <= (uint64_t)maximum;
    return signed_value >= minimum && signed_value <= maximum;
}

static bool typeconv_result_input_safe_cast(
    const CnpArray *array, int maximum_array_category,
    CNP_TYPE destination) {
    CNP_TYPE source_type = array->dtype->type_num;
    int source_category = typeconv_numeric_category(source_type);
    if (array->ndim == 0 && source_category == 1 &&
            maximum_array_category >= source_category &&
            typeconv_numeric_category(destination) == 1)
        return typeconv_scalar_integer_fits(array, destination);
    return typeconv_numeric_safe_cast(
        typeconv_result_input_type(array, maximum_array_category),
        destination);
}

static CNP_TYPE typeconv_numeric_common_type(
    int narrays, const CnpArray **arrays, int maximum_array_category) {
    static const CNP_TYPE candidates[][8] = {
        {CNP_BOOL, CNP_NOTYPE},
        {CNP_BYTE, CNP_UBYTE, CNP_SHORT, CNP_USHORT,
         CNP_INT, CNP_UINT, CNP_LONGLONG, CNP_ULONGLONG},
        {CNP_HALF, CNP_FLOAT, CNP_DOUBLE, CNP_LONGDOUBLE,
         CNP_NOTYPE},
        {CNP_CFLOAT, CNP_CDOUBLE, CNP_CLONGDOUBLE, CNP_NOTYPE}
    };
    int maximum_category = 0;
    CNP_TYPE first_type = typeconv_result_input_type(
        arrays[0], maximum_array_category);
    bool all_same = true;
    for (int index = 0; index < narrays; ++index) {
        CNP_TYPE source_type = typeconv_result_input_type(
            arrays[index], maximum_array_category);
        int category = typeconv_numeric_category(source_type);
        if (category > maximum_category)
            maximum_category = category;
        if (source_type != first_type)
            all_same = false;
    }
    if (all_same)
        return first_type;

    for (int category = maximum_category; category <= 3; ++category) {
        for (int candidate_index = 0; candidate_index < 8;
                ++candidate_index) {
            CNP_TYPE candidate = candidates[category][candidate_index];
            if (candidate == CNP_NOTYPE)
                break;
            bool accepted = true;
            for (int input_index = 0; input_index < narrays;
                    ++input_index) {
                if (!typeconv_result_input_safe_cast(
                        arrays[input_index], maximum_array_category,
                        candidate)) {
                    accepted = false;
                    break;
                }
            }
            if (accepted)
                return candidate;
        }
    }
    return CNP_NOTYPE;
}

/* =========================================================================
 * cnp_promote_types_public - Public API for type promotion
 * numpy.promote_types(type1, type2)
 * ========================================================================= */
CNP_API CNP_TYPE CNP_CALL cnp_promote_types_public(CNP_TYPE type1, CNP_TYPE type2) {
    const char *function_name = "cnp_promote_types_public";
    if (!typeconv_valid_type(type1) || !typeconv_valid_type(type2)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "types %d and %d must be valid CNP_TYPE values",
            (int)type1, (int)type2);
        return CNP_NOTYPE;
    }
    CNP_TYPE result = cnp_promote_type_full(type1, type2);
    if (result == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "types %d and %d do not have a common dtype",
            (int)type1, (int)type2);
    }
    return result;
}

/* =========================================================================
 * cnp_result_type - Determine result type of arrays/scalars
 * numpy.result_type(*arrays_and_dtypes)
 * ========================================================================= */
CNP_API CNP_TYPE CNP_CALL cnp_result_type(int narrays, const CnpArray **arrays) {
    const char *function_name = "cnp_result_type";
    if (narrays <= 0) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "at least one array is required");
        return CNP_NOTYPE;
    }
    if (!arrays) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "arrays must not be NULL");
        return CNP_NOTYPE;
    }

    bool all_numeric = true;
    int maximum_array_category = -1;
    for (int index = 0; index < narrays; ++index) {
        const CnpArray *array = arrays[index];
        if (!array) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "arrays[%d] must not be NULL", index);
            return CNP_NOTYPE;
        }
        CNP_TYPE type = array->dtype->type_num;
        if (!typeconv_valid_type(type)) {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "arrays[%d] has invalid dtype %d", index, (int)type);
            return CNP_NOTYPE;
        }
        int category = typeconv_numeric_category(type);
        if (category < 0) {
            all_numeric = false;
        } else if (array->ndim != 0 &&
                category > maximum_array_category) {
            maximum_array_category = category;
        }
    }

    CNP_TYPE result;
    if (all_numeric) {
        result = typeconv_numeric_common_type(
            narrays, arrays, maximum_array_category);
    } else {
        result = arrays[0]->dtype->type_num;
        for (int index = 1; index < narrays &&
                result != CNP_NOTYPE; ++index) {
            result = cnp_promote_type_full(
                result, arrays[index]->dtype->type_num);
        }
    }
    if (result == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "array dtypes do not have a common dtype");
    }
    return result;
}

/* =========================================================================
 * cnp_copyto - Copy values from source to destination (with broadcasting)
 * numpy.copyto(dst, src, casting='same_kind', where=True)
 * ========================================================================= */

typedef enum {
    CNP_COPY_VALUE_SIGNED,
    CNP_COPY_VALUE_UNSIGNED,
    CNP_COPY_VALUE_FLOATING,
    CNP_COPY_VALUE_COMPLEX
} CnpCopyValueKind;

typedef struct {
    CnpCopyValueKind kind;
    int64_t signed_value;
    uint64_t unsigned_value;
    long double real;
    long double imaginary;
} CnpCopyValue;

static bool typeconv_copy_scalar_dtype(CNP_TYPE type) {
    return typeconv_numeric_category(type) >= 0 ||
        type == CNP_DATETIME || type == CNP_TIMEDELTA;
}

static bool typeconv_copy_broadcastable(
    const CnpArray *source, const CnpArray *destination) {
    int destination_axis = destination->ndim - 1;
    for (int source_axis = source->ndim - 1;
         source_axis >= 0; --source_axis, --destination_axis) {
        int64_t source_dimension = source->shape[source_axis];
        if (destination_axis < 0) {
            if (source_dimension != 1) return false;
            continue;
        }
        int64_t destination_dimension =
            destination->shape[destination_axis];
        if (source_dimension != 1 &&
                source_dimension != destination_dimension)
            return false;
    }
    return true;
}

static int64_t typeconv_copy_source_offset(
    const CnpArray *source, const CnpArray *destination,
    const int64_t *destination_coordinates) {
    int destination_axis = destination->ndim - source->ndim;
    int64_t offset = source->offset;
    for (int source_axis = 0; source_axis < source->ndim;
         ++source_axis, ++destination_axis) {
        int64_t coordinate = 0;
        if (destination_axis >= 0 && source->shape[source_axis] != 1)
            coordinate = destination_coordinates[destination_axis];
        offset += coordinate * source->strides[source_axis];
    }
    return offset;
}

static CNP_STATUS typeconv_copy_read_value(
    const CnpArray *source, int64_t source_offset,
    CnpCopyValue *value) {
    const char *pointer = (const char*)source->data + source_offset;
    value->signed_value = 0;
    value->unsigned_value = 0;
    value->real = 0.0L;
    value->imaginary = 0.0L;
    switch (source->dtype->type_num) {
        case CNP_BOOL:
            value->kind = CNP_COPY_VALUE_SIGNED;
            value->signed_value = *(const int8_t*)pointer != 0;
            return CNP_OK;
        case CNP_BYTE:
            value->kind = CNP_COPY_VALUE_SIGNED;
            value->signed_value = *(const int8_t*)pointer;
            return CNP_OK;
        case CNP_SHORT:
            value->kind = CNP_COPY_VALUE_SIGNED;
            value->signed_value = *(const int16_t*)pointer;
            return CNP_OK;
        case CNP_INT:
            value->kind = CNP_COPY_VALUE_SIGNED;
            value->signed_value = *(const int32_t*)pointer;
            return CNP_OK;
        case CNP_LONG:
        case CNP_LONGLONG:
        case CNP_DATETIME:
        case CNP_TIMEDELTA:
            value->kind = CNP_COPY_VALUE_SIGNED;
            value->signed_value = *(const int64_t*)pointer;
            return CNP_OK;
        case CNP_UBYTE:
            value->kind = CNP_COPY_VALUE_UNSIGNED;
            value->unsigned_value = *(const uint8_t*)pointer;
            return CNP_OK;
        case CNP_USHORT:
            value->kind = CNP_COPY_VALUE_UNSIGNED;
            value->unsigned_value = *(const uint16_t*)pointer;
            return CNP_OK;
        case CNP_UINT:
            value->kind = CNP_COPY_VALUE_UNSIGNED;
            value->unsigned_value = *(const uint32_t*)pointer;
            return CNP_OK;
        case CNP_ULONG:
        case CNP_ULONGLONG:
            value->kind = CNP_COPY_VALUE_UNSIGNED;
            value->unsigned_value = *(const uint64_t*)pointer;
            return CNP_OK;
        case CNP_HALF:
            value->kind = CNP_COPY_VALUE_FLOATING;
            value->real = (long double)cnp_half_to_float(
                *(const uint16_t*)pointer);
            return CNP_OK;
        case CNP_FLOAT:
            value->kind = CNP_COPY_VALUE_FLOATING;
            value->real = (long double)*(const float*)pointer;
            return CNP_OK;
        case CNP_DOUBLE:
            value->kind = CNP_COPY_VALUE_FLOATING;
            value->real = (long double)*(const double*)pointer;
            return CNP_OK;
        case CNP_LONGDOUBLE:
            value->kind = CNP_COPY_VALUE_FLOATING;
            value->real = *(const long double*)pointer;
            return CNP_OK;
        case CNP_CFLOAT: {
            const cnp_cfloat *complex_value =
                (const cnp_cfloat*)pointer;
            value->kind = CNP_COPY_VALUE_COMPLEX;
            value->real = (long double)complex_value->real;
            value->imaginary = (long double)complex_value->imag;
            return CNP_OK;
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *complex_value =
                (const cnp_cdouble*)pointer;
            value->kind = CNP_COPY_VALUE_COMPLEX;
            value->real = (long double)complex_value->real;
            value->imaginary = (long double)complex_value->imag;
            return CNP_OK;
        }
        case CNP_CLONGDOUBLE: {
            const cnp_clongdouble *complex_value =
                (const cnp_clongdouble*)pointer;
            value->kind = CNP_COPY_VALUE_COMPLEX;
            value->real = complex_value->real;
            value->imaginary = complex_value->imag;
            return CNP_OK;
        }
        default:
            cnp_set_error(
                CNP_ERR_TYPE, "cnp_copyto",
                "source dtype %d is not supported for value conversion",
                (int)source->dtype->type_num);
            return CNP_ERR_TYPE;
    }
}

static uint64_t typeconv_copy_integer_bits(const CnpCopyValue *value) {
    return value->kind == CNP_COPY_VALUE_SIGNED
        ? (uint64_t)value->signed_value
        : value->unsigned_value;
}

static long double typeconv_copy_real(const CnpCopyValue *value) {
    if (value->kind == CNP_COPY_VALUE_SIGNED)
        return (long double)value->signed_value;
    if (value->kind == CNP_COPY_VALUE_UNSIGNED)
        return (long double)value->unsigned_value;
    return value->real;
}

static bool typeconv_copy_truth(const CnpCopyValue *value) {
    if (value->kind == CNP_COPY_VALUE_SIGNED)
        return value->signed_value != 0;
    if (value->kind == CNP_COPY_VALUE_UNSIGNED)
        return value->unsigned_value != 0;
    return value->real != 0.0L ||
        (value->kind == CNP_COPY_VALUE_COMPLEX &&
         value->imaginary != 0.0L);
}

static CNP_STATUS typeconv_copy_write_integer(
    char *target, CNP_TYPE destination_type,
    const CnpCopyValue *value) {
    bool integer_source = value->kind == CNP_COPY_VALUE_SIGNED ||
        value->kind == CNP_COPY_VALUE_UNSIGNED;
    uint64_t bits = integer_source
        ? typeconv_copy_integer_bits(value)
        : 0;
    long double real = typeconv_copy_real(value);
    switch (destination_type) {
        case CNP_BYTE: {
            int8_t converted = integer_source
                ? (int8_t)bits : (int8_t)real;
            memcpy(target, &converted, sizeof(converted));
            return CNP_OK;
        }
        case CNP_UBYTE: {
            uint8_t converted = integer_source
                ? (uint8_t)bits : (uint8_t)real;
            memcpy(target, &converted, sizeof(converted));
            return CNP_OK;
        }
        case CNP_SHORT: {
            int16_t converted = integer_source
                ? (int16_t)bits : (int16_t)real;
            memcpy(target, &converted, sizeof(converted));
            return CNP_OK;
        }
        case CNP_USHORT: {
            uint16_t converted = integer_source
                ? (uint16_t)bits : (uint16_t)real;
            memcpy(target, &converted, sizeof(converted));
            return CNP_OK;
        }
        case CNP_INT: {
            int32_t converted = integer_source
                ? (int32_t)bits : (int32_t)real;
            memcpy(target, &converted, sizeof(converted));
            return CNP_OK;
        }
        case CNP_UINT: {
            uint32_t converted = integer_source
                ? (uint32_t)bits : (uint32_t)real;
            memcpy(target, &converted, sizeof(converted));
            return CNP_OK;
        }
        case CNP_LONG:
        case CNP_LONGLONG:
        case CNP_DATETIME:
        case CNP_TIMEDELTA: {
            int64_t converted = integer_source
                ? (int64_t)bits : (int64_t)real;
            memcpy(target, &converted, sizeof(converted));
            return CNP_OK;
        }
        case CNP_ULONG:
        case CNP_ULONGLONG: {
            uint64_t converted = integer_source
                ? bits : (uint64_t)real;
            memcpy(target, &converted, sizeof(converted));
            return CNP_OK;
        }
        default:
            cnp_set_error(
                CNP_ERR_TYPE, "cnp_copyto",
                "destination dtype %d is not supported for integer conversion",
                (int)destination_type);
            return CNP_ERR_TYPE;
    }
}

static CNP_STATUS typeconv_copy_write_value(
    CnpArray *destination, int64_t destination_offset,
    const CnpCopyValue *value) {
    char *target = (char*)destination->data + destination_offset;
    CNP_TYPE destination_type = destination->dtype->type_num;
    if (destination_type == CNP_BOOL) {
        int8_t converted = typeconv_copy_truth(value) ? 1 : 0;
        memcpy(target, &converted, sizeof(converted));
        return CNP_OK;
    }
    if (cnp_type_is_integer(destination_type) ||
            destination_type == CNP_DATETIME ||
            destination_type == CNP_TIMEDELTA) {
        return typeconv_copy_write_integer(
            target, destination_type, value);
    }
    long double real = typeconv_copy_real(value);
    switch (destination_type) {
        case CNP_HALF: {
            uint16_t converted = cnp_float_to_half((double)real);
            memcpy(target, &converted, sizeof(converted));
            return CNP_OK;
        }
        case CNP_FLOAT: {
            float converted = (float)real;
            memcpy(target, &converted, sizeof(converted));
            return CNP_OK;
        }
        case CNP_DOUBLE: {
            double converted = (double)real;
            memcpy(target, &converted, sizeof(converted));
            return CNP_OK;
        }
        case CNP_LONGDOUBLE:
            memcpy(target, &real, sizeof(real));
            return CNP_OK;
        case CNP_CFLOAT: {
            cnp_cfloat converted = {
                (float)real,
                value->kind == CNP_COPY_VALUE_COMPLEX
                    ? (float)value->imaginary : 0.0f
            };
            memcpy(target, &converted, sizeof(converted));
            return CNP_OK;
        }
        case CNP_CDOUBLE: {
            cnp_cdouble converted = {
                (double)real,
                value->kind == CNP_COPY_VALUE_COMPLEX
                    ? (double)value->imaginary : 0.0
            };
            memcpy(target, &converted, sizeof(converted));
            return CNP_OK;
        }
        case CNP_CLONGDOUBLE: {
            cnp_clongdouble converted = {
                real,
                value->kind == CNP_COPY_VALUE_COMPLEX
                    ? value->imaginary : 0.0L
            };
            memcpy(target, &converted, sizeof(converted));
            return CNP_OK;
        }
        default:
            cnp_set_error(
                CNP_ERR_TYPE, "cnp_copyto",
                "destination dtype %d is not supported for value conversion",
                (int)destination_type);
            return CNP_ERR_TYPE;
    }
}

CNP_STATUS cnp_cast_scalar_value(
    const void *source,
    CNP_TYPE source_type,
    void *destination,
    CNP_TYPE destination_type,
    const char *function_name) {
    if (!source || !destination) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source and destination are required");
        return CNP_ERR_GENERIC;
    }

    CnpDtype source_descriptor = {0};
    CnpDtype destination_descriptor = {0};
    CnpArray source_array = {0};
    CnpArray destination_array = {0};
    source_descriptor.type_num = source_type;
    destination_descriptor.type_num = destination_type;
    source_array.data = (void*)source;
    source_array.dtype = &source_descriptor;
    destination_array.data = destination;
    destination_array.dtype = &destination_descriptor;

    CnpCopyValue value;
    CNP_STATUS status = typeconv_copy_read_value(
        &source_array, 0, &value);
    if (status == CNP_OK) {
        status = typeconv_copy_write_value(
            &destination_array, 0, &value);
    }
    if (status != CNP_OK) cnp_relabel_error(function_name);
    return status;
}

CNP_API CNP_STATUS CNP_CALL cnp_copyto(CnpArray *dst, const CnpArray *src, CNP_CASTING casting) {
    const char *function_name = "cnp_copyto";
    if (!dst || !src) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "destination and source arrays are required");
        return CNP_ERR_GENERIC;
    }
    if (!(dst->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "destination array is not writeable");
        return CNP_ERR_GENERIC;
    }
    if (casting < CNP_CAST_NO || casting > CNP_CAST_UNSAFE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "casting %d must be a valid CNP_CASTING value",
            (int)casting);
        return CNP_ERR_TYPE;
    }
    if (!cnp_dtype_can_cast(src->dtype->type_num, dst->dtype->type_num, casting)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "cannot cast dtype %d to dtype %d under casting mode %d",
            (int)src->dtype->type_num, (int)dst->dtype->type_num,
            (int)casting);
        return CNP_ERR_TYPE;
    }
    if (!typeconv_copy_broadcastable(src, dst)) {
        cnp_set_error(
            CNP_ERR_BROADCAST, function_name,
            "source shape cannot broadcast to destination shape");
        return CNP_ERR_BROADCAST;
    }

    bool equivalent = cnp_dtype_can_cast(
        src->dtype->type_num, dst->dtype->type_num, CNP_CAST_NO);
    if (!equivalent &&
            (!typeconv_copy_scalar_dtype(src->dtype->type_num) ||
             !typeconv_copy_scalar_dtype(dst->dtype->type_num))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "copy conversion from dtype %d to dtype %d is not supported",
            (int)src->dtype->type_num, (int)dst->dtype->type_num);
        return CNP_ERR_TYPE;
    }
    if (dst->size == 0 || dst == src) return CNP_OK;

    if (equivalent && dst->size == src->size &&
            (dst->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            (src->flags & CNP_ARRAY_C_CONTIGUOUS)) {
        memmove(
            (char*)dst->data + dst->offset,
            (const char*)src->data + src->offset,
            (size_t)dst->size * (size_t)dst->dtype->elsize);
        return CNP_OK;
    }

    const CnpArray *copy_source = src;
    CnpArray *source_snapshot = NULL;
    if (dst->data == src->data) {
        source_snapshot = cnp_array_copy(src);
        if (!source_snapshot) {
            cnp_relabel_error(function_name);
            return cnp_get_error(NULL);
        }
        copy_source = source_snapshot;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < dst->size; ++index) {
        int64_t source_offset = typeconv_copy_source_offset(
            copy_source, dst, coordinates);
        int64_t destination_offset = dst->offset +
            cnp_multi_to_offset(dst->ndim, coordinates, dst->strides);
        if (equivalent) {
            memcpy(
                (char*)dst->data + destination_offset,
                (const char*)copy_source->data + source_offset,
                (size_t)dst->dtype->elsize);
        } else {
            CnpCopyValue value;
            CNP_STATUS conversion_status = typeconv_copy_read_value(
                copy_source, source_offset, &value);
            if (conversion_status == CNP_OK)
                conversion_status = typeconv_copy_write_value(
                    dst, destination_offset, &value);
            if (conversion_status != CNP_OK) {
                if (source_snapshot) cnp_array_free(source_snapshot);
                return conversion_status;
            }
        }
        for (int dimension = dst->ndim - 1;
             dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < dst->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    if (source_snapshot) cnp_array_free(source_snapshot);
    return CNP_OK;
}

/* =========================================================================
 * cnp_where - Conditional selection
 * numpy.where(condition, x, y)
 * ========================================================================= */
static void typeconv_release_arrays(CnpArray **arrays, int count) {
    if (!arrays) return;
    for (int index = 0; index < count; ++index) {
        if (arrays[index]) {
            cnp_array_free(arrays[index]);
            arrays[index] = NULL;
        }
    }
}

CNP_API CnpArray* CNP_CALL cnp_where(
    const CnpArray *condition, const CnpArray *x, const CnpArray *y) {
    const char *function_name = "cnp_where";
    if (!condition || !x || !y) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "condition, x, and y arrays are required");
        return NULL;
    }
    if (!typeconv_copy_scalar_dtype(condition->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "condition dtype %d is not supported",
            (int)condition->dtype->type_num);
        return NULL;
    }

    const CnpArray *choices[2] = {x, y};
    CNP_TYPE result_type = cnp_result_type(2, choices);
    if (result_type == CNP_NOTYPE) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (!typeconv_copy_scalar_dtype(x->dtype->type_num) ||
            !typeconv_copy_scalar_dtype(y->dtype->type_num) ||
            !typeconv_copy_scalar_dtype(result_type)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "choice dtypes %d and %d cannot be selected into dtype %d",
            (int)x->dtype->type_num, (int)y->dtype->type_num,
            (int)result_type);
        return NULL;
    }

    CnpArray *inputs[3] = {
        (CnpArray*)condition, (CnpArray*)x, (CnpArray*)y
    };
    CnpArray *broadcasted[3] = {NULL, NULL, NULL};
    CNP_STATUS status = cnp_broadcast_arrays_v2(
        3, inputs, broadcasted, 3);
    if (status != CNP_OK) {
        if (status == CNP_ERR_BROADCAST) {
            char detail[256];
            strncpy(detail, cnp_get_error_message(), sizeof(detail) - 1);
            detail[sizeof(detail) - 1] = '\0';
            cnp_set_error(
                status, function_name,
                "condition, x, and y cannot broadcast to a common shape: %s",
                detail);
        } else {
            cnp_relabel_error(function_name);
        }
        return NULL;
    }

    CnpArray *result = cnp_array_new(
        broadcasted[0]->ndim, broadcasted[0]->shape,
        result_type, CNP_ORDER_C);
    if (!result) {
        typeconv_release_arrays(broadcasted, 3);
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t condition_offset = broadcasted[0]->offset +
            cnp_multi_to_offset(
                broadcasted[0]->ndim, coordinates,
                broadcasted[0]->strides);
        CnpCopyValue condition_value;
        status = typeconv_copy_read_value(
            broadcasted[0], condition_offset, &condition_value);
        if (status != CNP_OK) break;

        const CnpArray *selected = typeconv_copy_truth(&condition_value)
            ? broadcasted[1] : broadcasted[2];
        int64_t selected_offset = selected->offset +
            cnp_multi_to_offset(
                selected->ndim, coordinates, selected->strides);
        CnpCopyValue selected_value;
        status = typeconv_copy_read_value(
            selected, selected_offset, &selected_value);
        if (status != CNP_OK) break;

        int64_t result_offset = result->offset +
            cnp_multi_to_offset(
                result->ndim, coordinates, result->strides);
        status = typeconv_copy_write_value(
            result, result_offset, &selected_value);
        if (status != CNP_OK) break;

        for (int dimension = result->ndim - 1;
             dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }

    typeconv_release_arrays(broadcasted, 3);
    if (status != CNP_OK) {
        cnp_array_free(result);
        cnp_relabel_error(function_name);
        return NULL;
    }
    return result;
}

/* =========================================================================
 * cnp_where_indices - Return indices where condition is true
 * numpy.where(condition) - returns tuple of arrays
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_where_indices(const CnpArray *condition) {
    if (!condition) return NULL;
    return cnp_flatnonzero(condition);
}

CNP_API CNP_STATUS CNP_CALL cnp_where_indices_v2(
    const CnpArray *condition,
    CnpArray **results, int result_capacity) {
    const char *function_name = "cnp_where_indices_v2";
    if (result_capacity < 0) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "result capacity must be non-negative");
        return CNP_ERR_GENERIC;
    }
    if (results) {
        for (int index = 0; index < result_capacity; ++index)
            results[index] = NULL;
    }
    if (!condition) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "condition array is required");
        return CNP_ERR_GENERIC;
    }
    int result_count = condition->ndim > 0 ? condition->ndim : 1;
    if (result_capacity < result_count) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "result capacity %d is smaller than required count %d",
            result_capacity, result_count);
        return CNP_ERR_GENERIC;
    }
    if (!results) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "result array is required");
        return CNP_ERR_GENERIC;
    }
    if (!typeconv_copy_scalar_dtype(condition->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "condition dtype %d is not supported",
            (int)condition->dtype->type_num);
        return CNP_ERR_TYPE;
    }

    int64_t nonzero_count = 0;
    int64_t coordinates[CNP_MAXDIMS] = {0};
    CNP_STATUS status = CNP_OK;
    for (int64_t index = 0; index < condition->size; ++index) {
        int64_t offset = condition->offset +
            cnp_multi_to_offset(
                condition->ndim, coordinates, condition->strides);
        CnpCopyValue value;
        status = typeconv_copy_read_value(condition, offset, &value);
        if (status != CNP_OK) break;
        if (typeconv_copy_truth(&value)) ++nonzero_count;
        for (int dimension = condition->ndim - 1;
             dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < condition->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    if (status != CNP_OK) {
        cnp_relabel_error(function_name);
        return status;
    }

    int64_t result_shape[1] = {nonzero_count};
    for (int axis = 0; axis < result_count; ++axis) {
        results[axis] = cnp_array_new(
            1, result_shape, CNP_LONGLONG, CNP_ORDER_C);
        if (!results[axis]) {
            typeconv_release_arrays(results, result_count);
            cnp_relabel_error(function_name);
            return cnp_get_error(NULL);
        }
    }

    memset(coordinates, 0, sizeof(coordinates));
    int64_t output_index = 0;
    for (int64_t index = 0; index < condition->size; ++index) {
        int64_t offset = condition->offset +
            cnp_multi_to_offset(
                condition->ndim, coordinates, condition->strides);
        CnpCopyValue value;
        status = typeconv_copy_read_value(condition, offset, &value);
        if (status != CNP_OK) break;
        if (typeconv_copy_truth(&value)) {
            for (int axis = 0; axis < result_count; ++axis) {
                int64_t coordinate = condition->ndim == 0
                    ? 0 : coordinates[axis];
                ((int64_t*)results[axis]->data)[output_index] = coordinate;
            }
            ++output_index;
        }
        for (int dimension = condition->ndim - 1;
             dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < condition->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    if (status != CNP_OK) {
        typeconv_release_arrays(results, result_count);
        cnp_relabel_error(function_name);
        return status;
    }
    return CNP_OK;
}

/* =========================================================================
 * cnp_matrix_power - Raise a square matrix to integer power
 * ========================================================================= */
typedef struct {
    long double real;
    long double imaginary;
} CnpMatrixValue;

static bool typeconv_matrix_batch_count(
    const CnpArray *array, int64_t *batch_count,
    const char *function_name) {
    int64_t count = 1;
    for (int axis = 0; axis < array->ndim - 2; ++axis) {
        int64_t dimension = array->shape[axis];
        if (dimension < 0 ||
                (dimension > 0 && count > INT64_MAX / dimension)) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "batch dimensions are invalid or overflow int64");
            return false;
        }
        count *= dimension;
    }
    *batch_count = count;
    return true;
}

static int64_t typeconv_matrix_element_offset(
    const CnpArray *array, int64_t batch,
    int64_t row, int64_t column) {
    int64_t offset = array->offset +
        row * array->strides[array->ndim - 2] +
        column * array->strides[array->ndim - 1];
    for (int axis = array->ndim - 3; axis >= 0; --axis) {
        int64_t dimension = array->shape[axis];
        int64_t coordinate = dimension == 0 ? 0 : batch % dimension;
        if (dimension != 0) batch /= dimension;
        offset += coordinate * array->strides[axis];
    }
    return offset;
}

static CnpMatrixValue typeconv_matrix_value_from_copy(
    const CnpCopyValue *value) {
    CnpMatrixValue converted = {
        typeconv_copy_real(value),
        value->kind == CNP_COPY_VALUE_COMPLEX
            ? value->imaginary : 0.0L
    };
    return converted;
}

static CNP_STATUS typeconv_matrix_read(
    const CnpArray *array, int64_t batch,
    int64_t row, int64_t column,
    CnpCopyValue *value) {
    return typeconv_copy_read_value(
        array,
        typeconv_matrix_element_offset(array, batch, row, column),
        value);
}

static CNP_STATUS typeconv_matrix_write(
    CnpArray *array, int64_t batch,
    int64_t row, int64_t column,
    const CnpCopyValue *value) {
    return typeconv_copy_write_value(
        array,
        typeconv_matrix_element_offset(array, batch, row, column),
        value);
}

static CnpMatrixValue typeconv_matrix_accumulate_product(
    CnpMatrixValue sum,
    CnpMatrixValue left, CnpMatrixValue right,
    CNP_TYPE dtype) {
    if (dtype == CNP_HALF || dtype == CNP_FLOAT) {
        float updated = (float)sum.real +
            (float)left.real * (float)right.real;
        sum.real = (long double)updated;
        return sum;
    }
    if (dtype == CNP_DOUBLE) {
        double updated = (double)sum.real +
            (double)left.real * (double)right.real;
        sum.real = (long double)updated;
        return sum;
    }
    if (dtype == CNP_LONGDOUBLE) {
        sum.real += left.real * right.real;
        return sum;
    }
    if (dtype == CNP_CFLOAT) {
        float sum_real = (float)sum.real;
        float sum_imaginary = (float)sum.imaginary;
        float left_real = (float)left.real;
        float left_imaginary = (float)left.imaginary;
        float right_real = (float)right.real;
        float right_imaginary = (float)right.imaginary;
        sum_real += left_real * right_real -
            left_imaginary * right_imaginary;
        sum_imaginary += left_real * right_imaginary +
            left_imaginary * right_real;
        sum.real = (long double)sum_real;
        sum.imaginary = (long double)sum_imaginary;
        return sum;
    }
    if (dtype == CNP_CDOUBLE) {
        double sum_real = (double)sum.real;
        double sum_imaginary = (double)sum.imaginary;
        double left_real = (double)left.real;
        double left_imaginary = (double)left.imaginary;
        double right_real = (double)right.real;
        double right_imaginary = (double)right.imaginary;
        sum_real += left_real * right_real -
            left_imaginary * right_imaginary;
        sum_imaginary += left_real * right_imaginary +
            left_imaginary * right_real;
        sum.real = (long double)sum_real;
        sum.imaginary = (long double)sum_imaginary;
        return sum;
    }
    sum.real += left.real * right.real -
        left.imaginary * right.imaginary;
    sum.imaginary += left.real * right.imaginary +
        left.imaginary * right.real;
    return sum;
}

static CnpMatrixValue typeconv_matrix_quantize(
    CnpMatrixValue value, CNP_TYPE dtype) {
    if (dtype == CNP_FLOAT) {
        value.real = (long double)(float)value.real;
    } else if (dtype == CNP_DOUBLE) {
        value.real = (long double)(double)value.real;
    } else if (dtype == CNP_CFLOAT) {
        value.real = (long double)(float)value.real;
        value.imaginary = (long double)(float)value.imaginary;
    } else if (dtype == CNP_CDOUBLE) {
        value.real = (long double)(double)value.real;
        value.imaginary = (long double)(double)value.imaginary;
    }
    return value;
}

static bool typeconv_matrix_is_nan(long double value) {
    double converted = (double)value;
    uint64_t bits;
    memcpy(&bits, &converted, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) ==
            UINT64_C(0x7ff0000000000000) &&
        (bits & UINT64_C(0x000fffffffffffff)) != 0;
}

static CnpMatrixValue typeconv_matrix_divide(
    CnpMatrixValue numerator, CnpMatrixValue denominator,
    CNP_TYPE dtype, bool complex_values) {
    CnpMatrixValue result = {0.0L, 0.0L};
    if (!complex_values) {
        result.real = numerator.real / denominator.real;
        return typeconv_matrix_quantize(result, dtype);
    }
    long double real_abs = fabsl(denominator.real);
    long double imaginary_abs = fabsl(denominator.imaginary);
    if (real_abs >= imaginary_abs) {
        long double ratio = denominator.imaginary / denominator.real;
        long double divisor = denominator.real +
            denominator.imaginary * ratio;
        result.real = (numerator.real +
            numerator.imaginary * ratio) / divisor;
        result.imaginary = (numerator.imaginary -
            numerator.real * ratio) / divisor;
    } else {
        long double ratio = denominator.real / denominator.imaginary;
        long double divisor = denominator.imaginary +
            denominator.real * ratio;
        result.real = (numerator.real * ratio +
            numerator.imaginary) / divisor;
        result.imaginary = (numerator.imaginary * ratio -
            numerator.real) / divisor;
    }
    return typeconv_matrix_quantize(result, dtype);
}

static CnpMatrixValue typeconv_matrix_subtract_product(
    CnpMatrixValue value,
    CnpMatrixValue left, CnpMatrixValue right,
    CNP_TYPE dtype, bool complex_values) {
    CnpMatrixValue product = {
        left.real * right.real - left.imaginary * right.imaginary,
        left.real * right.imaginary + left.imaginary * right.real
    };
    value.real -= product.real;
    if (complex_values) value.imaginary -= product.imaginary;
    return typeconv_matrix_quantize(value, dtype);
}

static CnpArray *typeconv_matrix_identity(
    const CnpArray *source, CNP_TYPE dtype,
    int64_t batch_count, int64_t dimension,
    const char *function_name) {
    CnpArray *identity = cnp_array_zeros(
        source->ndim, source->shape, dtype, CNP_ORDER_C);
    if (!identity) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpCopyValue one = {0};
    if (cnp_type_is_complex(dtype)) {
        one.kind = CNP_COPY_VALUE_COMPLEX;
        one.real = 1.0L;
    } else if (cnp_type_is_float(dtype) || dtype == CNP_HALF) {
        one.kind = CNP_COPY_VALUE_FLOATING;
        one.real = 1.0L;
    } else {
        one.kind = CNP_COPY_VALUE_SIGNED;
        one.signed_value = 1;
    }
    for (int64_t batch = 0; batch < batch_count; ++batch) {
        for (int64_t diagonal = 0; diagonal < dimension; ++diagonal) {
            CNP_STATUS status = typeconv_matrix_write(
                identity, batch, diagonal, diagonal, &one);
            if (status != CNP_OK) {
                cnp_array_free(identity);
                cnp_relabel_error(function_name);
                return NULL;
            }
        }
    }
    return identity;
}

static CnpArray *typeconv_matrix_multiply(
    const CnpArray *left, const CnpArray *right,
    int64_t batch_count, int64_t dimension,
    const char *function_name) {
    CNP_TYPE dtype = left->dtype->type_num;
    if (left->ndim == 2 && dtype == CNP_DOUBLE &&
            right->dtype->type_num == CNP_DOUBLE &&
            (left->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            (right->flags & CNP_ARRAY_C_CONTIGUOUS)) {
        CnpArray *result = cnp_matmul(left, right);
        if (!result) cnp_relabel_error(function_name);
        return result;
    }
    int category = typeconv_numeric_category(dtype);
    CnpArray *result = cnp_array_new(
        left->ndim, left->shape, dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int64_t batch = 0; batch < batch_count; ++batch) {
        for (int64_t row = 0; row < dimension; ++row) {
            for (int64_t column = 0; column < dimension; ++column) {
                CnpCopyValue output = {0};
                CNP_STATUS status = CNP_OK;
                if (category == 0) {
                    bool truth = false;
                    for (int64_t inner = 0; inner < dimension; ++inner) {
                        CnpCopyValue left_value;
                        CnpCopyValue right_value;
                        status = typeconv_matrix_read(
                            left, batch, row, inner, &left_value);
                        if (status == CNP_OK)
                            status = typeconv_matrix_read(
                                right, batch, inner, column, &right_value);
                        if (status != CNP_OK) break;
                        if (typeconv_copy_truth(&left_value) &&
                                typeconv_copy_truth(&right_value)) {
                            truth = true;
                            break;
                        }
                    }
                    output.kind = CNP_COPY_VALUE_SIGNED;
                    output.signed_value = truth ? 1 : 0;
                } else if (category == 1) {
                    uint64_t sum = 0;
                    for (int64_t inner = 0; inner < dimension; ++inner) {
                        CnpCopyValue left_value;
                        CnpCopyValue right_value;
                        status = typeconv_matrix_read(
                            left, batch, row, inner, &left_value);
                        if (status == CNP_OK)
                            status = typeconv_matrix_read(
                                right, batch, inner, column, &right_value);
                        if (status != CNP_OK) break;
                        sum += typeconv_copy_integer_bits(&left_value) *
                            typeconv_copy_integer_bits(&right_value);
                    }
                    output.kind = CNP_COPY_VALUE_UNSIGNED;
                    output.unsigned_value = sum;
                } else {
                    CnpMatrixValue sum = {0.0L, 0.0L};
                    for (int64_t inner = 0; inner < dimension; ++inner) {
                        CnpCopyValue left_value;
                        CnpCopyValue right_value;
                        status = typeconv_matrix_read(
                            left, batch, row, inner, &left_value);
                        if (status == CNP_OK)
                            status = typeconv_matrix_read(
                                right, batch, inner, column, &right_value);
                        if (status != CNP_OK) break;
                        sum = typeconv_matrix_accumulate_product(
                            sum,
                            typeconv_matrix_value_from_copy(&left_value),
                            typeconv_matrix_value_from_copy(&right_value),
                            dtype);
                    }
                    output.kind = category == 3
                        ? CNP_COPY_VALUE_COMPLEX
                        : CNP_COPY_VALUE_FLOATING;
                    output.real = sum.real;
                    output.imaginary = sum.imaginary;
                }
                if (status == CNP_OK)
                    status = typeconv_matrix_write(
                        result, batch, row, column, &output);
                if (status != CNP_OK) {
                    cnp_array_free(result);
                    cnp_relabel_error(function_name);
                    return NULL;
                }
            }
        }
    }
    return result;
}

static CnpArray *typeconv_matrix_cast_copy(
    const CnpArray *source, CNP_TYPE dtype,
    const char *function_name) {
    if (source->dtype->type_num == dtype) {
        CnpArray *copy = cnp_array_copy(source);
        if (!copy) cnp_relabel_error(function_name);
        return copy;
    }
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < source->size; ++index) {
        int64_t source_offset = source->offset +
            cnp_multi_to_offset(
                source->ndim, coordinates, source->strides);
        CnpCopyValue value;
        CNP_STATUS status = typeconv_copy_read_value(
            source, source_offset, &value);
        if (status == CNP_OK)
            status = typeconv_copy_write_value(
                result, result->offset +
                    index * result->dtype->elsize,
                &value);
        if (status != CNP_OK) {
            cnp_array_free(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        for (int axis = source->ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < source->shape[axis]) break;
            coordinates[axis] = 0;
        }
    }
    return result;
}

static CnpArray *typeconv_matrix_inverse(
    const CnpArray *source,
    int64_t batch_count, int64_t dimension,
    const char *function_name) {
    CNP_TYPE dtype = source->dtype->type_num;
    bool complex_values = cnp_type_is_complex(dtype);
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (batch_count == 0 || dimension == 0) return result;

    uint64_t dimension_unsigned = (uint64_t)dimension;
    if (dimension_unsigned > (uint64_t)SIZE_MAX / 2) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "inverse workspace dimensions overflow size_t");
        return NULL;
    }
    size_t row_width = (size_t)dimension_unsigned * 2;
    if ((size_t)dimension_unsigned > SIZE_MAX / row_width ||
            (size_t)dimension_unsigned * row_width >
                SIZE_MAX / sizeof(CnpMatrixValue)) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "inverse workspace size overflows size_t");
        return NULL;
    }
    size_t workspace_count =
        (size_t)dimension_unsigned * row_width;
    size_t workspace_bytes =
        workspace_count * sizeof(CnpMatrixValue);
    CnpMatrixValue *workspace =
        (CnpMatrixValue*)cnp_malloc(workspace_bytes);
    if (!workspace) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate inverse workspace");
        return NULL;
    }

    for (int64_t batch = 0; batch < batch_count; ++batch) {
        for (int64_t row = 0; row < dimension; ++row) {
            for (int64_t column = 0; column < dimension; ++column) {
                CnpCopyValue input;
                CNP_STATUS status = typeconv_matrix_read(
                    source, batch, row, column, &input);
                if (status != CNP_OK) {
                    cnp_free(workspace, workspace_bytes);
                    cnp_array_free(result);
                    cnp_relabel_error(function_name);
                    return NULL;
                }
                workspace[(size_t)row * row_width +
                    (size_t)column] =
                    typeconv_matrix_value_from_copy(&input);
                CnpMatrixValue identity_value = {
                    row == column ? 1.0L : 0.0L,
                    0.0L
                };
                workspace[(size_t)row * row_width +
                    (size_t)dimension + (size_t)column] =
                    identity_value;
            }
        }

        for (int64_t pivot_column = 0;
             pivot_column < dimension; ++pivot_column) {
            int64_t pivot_row = pivot_column;
            long double pivot_magnitude = 0.0L;
            bool pivot_is_nan = false;
            for (int64_t row = pivot_column; row < dimension; ++row) {
                CnpMatrixValue candidate = workspace[
                    (size_t)row * row_width +
                    (size_t)pivot_column];
                bool candidate_is_nan =
                    typeconv_matrix_is_nan(candidate.real) ||
                    typeconv_matrix_is_nan(candidate.imaginary);
                if (candidate_is_nan) {
                    pivot_row = row;
                    pivot_is_nan = true;
                    break;
                }
                long double magnitude = complex_values
                    ? hypotl(candidate.real, candidate.imaginary)
                    : fabsl(candidate.real);
                if (row == pivot_column || magnitude > pivot_magnitude) {
                    pivot_row = row;
                    pivot_magnitude = magnitude;
                }
            }
            if (!pivot_is_nan && pivot_magnitude == 0.0L) {
                cnp_free(workspace, workspace_bytes);
                cnp_array_free(result);
                cnp_set_error(
                    CNP_ERR_SINGULAR, function_name,
                    "matrix is singular");
                return NULL;
            }
            if (pivot_row != pivot_column) {
                for (size_t column = 0; column < row_width; ++column) {
                    CnpMatrixValue temporary = workspace[
                        (size_t)pivot_column * row_width + column];
                    workspace[(size_t)pivot_column * row_width + column] =
                        workspace[(size_t)pivot_row * row_width + column];
                    workspace[(size_t)pivot_row * row_width + column] =
                        temporary;
                }
            }

            CnpMatrixValue pivot = workspace[
                (size_t)pivot_column * row_width +
                (size_t)pivot_column];
            for (size_t column = 0; column < row_width; ++column) {
                size_t index =
                    (size_t)pivot_column * row_width + column;
                workspace[index] = typeconv_matrix_divide(
                    workspace[index], pivot, dtype, complex_values);
            }
            for (int64_t row = 0; row < dimension; ++row) {
                if (row == pivot_column) continue;
                CnpMatrixValue factor = workspace[
                    (size_t)row * row_width +
                    (size_t)pivot_column];
                if (factor.real == 0.0L &&
                        factor.imaginary == 0.0L) continue;
                for (size_t column = 0; column < row_width; ++column) {
                    size_t destination_index =
                        (size_t)row * row_width + column;
                    size_t pivot_index =
                        (size_t)pivot_column * row_width + column;
                    workspace[destination_index] =
                        typeconv_matrix_subtract_product(
                            workspace[destination_index], factor,
                            workspace[pivot_index],
                            dtype, complex_values);
                }
            }
        }

        for (int64_t row = 0; row < dimension; ++row) {
            for (int64_t column = 0; column < dimension; ++column) {
                CnpMatrixValue value = workspace[
                    (size_t)row * row_width +
                    (size_t)dimension + (size_t)column];
                CnpCopyValue output = {0};
                output.kind = complex_values
                    ? CNP_COPY_VALUE_COMPLEX
                    : CNP_COPY_VALUE_FLOATING;
                output.real = value.real;
                output.imaginary = value.imaginary;
                CNP_STATUS status = typeconv_matrix_write(
                    result, batch, row, column, &output);
                if (status != CNP_OK) {
                    cnp_free(workspace, workspace_bytes);
                    cnp_array_free(result);
                    cnp_relabel_error(function_name);
                    return NULL;
                }
            }
        }
    }

    cnp_free(workspace, workspace_bytes);
    return result;
}

CnpArray* cnp_matrix_power_impl(
    const CnpArray *a, int64_t exponent,
    const char *function_name) {
    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array is required");
        return NULL;
    }
    if (a->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input must be at least two-dimensional");
        return NULL;
    }
    int64_t rows = a->shape[a->ndim - 2];
    int64_t columns = a->shape[a->ndim - 1];
    if (rows != columns) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "last two dimensions must be square");
        return NULL;
    }
    CNP_TYPE input_dtype = a->dtype->type_num;
    int category = typeconv_numeric_category(input_dtype);
    if (category < 0) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtype %d is not supported for matrix power",
            (int)input_dtype);
        return NULL;
    }
    int64_t batch_count;
    if (!typeconv_matrix_batch_count(
            a, &batch_count, function_name)) return NULL;

    if (exponent == 0) {
        return typeconv_matrix_identity(
            a, input_dtype, batch_count, rows, function_name);
    }

    const CnpArray *power_source = a;
    CnpArray *owned_source = NULL;
    uint64_t magnitude;
    if (exponent < 0) {
        CNP_TYPE inverse_dtype;
        if (category <= 1) {
            inverse_dtype = CNP_DOUBLE;
        } else if (input_dtype == CNP_HALF) {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "float16 is unsupported for negative matrix powers");
            return NULL;
        } else if (input_dtype == CNP_LONGDOUBLE ||
                   input_dtype == CNP_CLONGDOUBLE) {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "long double dtypes are unsupported for negative matrix powers");
            return NULL;
        } else {
            inverse_dtype = input_dtype;
        }
        CnpArray *converted = typeconv_matrix_cast_copy(
            a, inverse_dtype, function_name);
        if (!converted) return NULL;
        owned_source = typeconv_matrix_inverse(
            converted, batch_count, rows, function_name);
        cnp_array_free(converted);
        if (!owned_source) return NULL;
        power_source = owned_source;
        magnitude = (uint64_t)(-(exponent + 1)) + UINT64_C(1);
    } else {
        magnitude = (uint64_t)exponent;
    }

    if (magnitude == 1) {
        if (owned_source) return owned_source;
        cnp_array_incref((CnpArray*)a);
        return (CnpArray*)a;
    }
    if (magnitude == 2) {
        CnpArray *result = typeconv_matrix_multiply(
            power_source, power_source,
            batch_count, rows, function_name);
        if (owned_source) cnp_array_free(owned_source);
        return result;
    }
    if (magnitude == 3) {
        CnpArray *square = typeconv_matrix_multiply(
            power_source, power_source,
            batch_count, rows, function_name);
        if (!square) {
            if (owned_source) cnp_array_free(owned_source);
            return NULL;
        }
        CnpArray *result = typeconv_matrix_multiply(
            square, power_source,
            batch_count, rows, function_name);
        cnp_array_free(square);
        if (owned_source) cnp_array_free(owned_source);
        return result;
    }

    CnpArray *power = cnp_array_copy(power_source);
    if (!power) {
        if (owned_source) cnp_array_free(owned_source);
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpArray *result = NULL;
    bool failed = false;
    uint64_t remaining = magnitude;
    while (remaining > 0) {
        uint64_t bit = remaining & UINT64_C(1);
        remaining >>= 1;
        if (bit) {
            if (!result) {
                result = cnp_array_copy(power);
                if (!result) cnp_relabel_error(function_name);
            } else {
                CnpArray *next = typeconv_matrix_multiply(
                    result, power,
                    batch_count, rows, function_name);
                cnp_array_free(result);
                result = next;
            }
            if (!result) break;
        }
        if (remaining > 0) {
            CnpArray *next_power = typeconv_matrix_multiply(
                power, power,
                batch_count, rows, function_name);
            cnp_array_free(power);
            power = next_power;
            if (!power) {
                failed = true;
                break;
            }
        }
    }
    if (power) cnp_array_free(power);
    if (owned_source) cnp_array_free(owned_source);
    if (failed) {
        if (result) cnp_array_free(result);
        return NULL;
    }
    if (!result) return NULL;
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_matrix_power(const CnpArray *a, int n) {
    return cnp_matrix_power_impl(a, (int64_t)n, "cnp_matrix_power");
}

/* =========================================================================
 * cnp_matrix_rank - Matrix rank via SVD
 * numpy.linalg.matrix_rank(a, tol=None)
 * ========================================================================= */
CNP_API int CNP_CALL cnp_matrix_rank(const CnpArray *a, double tol) {
    const char *function_name = "cnp_matrix_rank";
    CnpArray *result;
    int64_t rank;

    if (a && a->ndim > 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "legacy scalar matrix rank cannot represent batch results");
        return 0;
    }
    result = cnp_linalg_matrix_rank(a, tol);
    if (!result) {
        cnp_relabel_error(function_name);
        return 0;
    }
    rank = cnp_array_get_int(result, NULL);
    cnp_array_free(result);
    if (rank > INT_MAX) {
        cnp_set_error(
            CNP_ERR_CONVERSION, function_name,
            "matrix rank does not fit the legacy int return type");
        return 0;
    }
    return (int)rank;
}

/* =========================================================================
 * cnp_nanmedian - Median ignoring NaN
 * numpy.nanmedian(a, axis=None)
 * ========================================================================= */
static double cnp_legacy_nan_statistic_value(
    CnpArray *result, const char *function_name) {
    if (!result) {
        cnp_relabel_error(function_name);
        return NAN;
    }
    if (result->ndim != 0) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "legacy scalar return cannot represent an array result");
        return NAN;
    }
    double value = cnp_array_get_double(result, NULL);
    cnp_array_free(result);
    return value;
}

CNP_API double CNP_CALL cnp_nanmedian(const CnpArray *arr, int axis) {
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_nanmedian",
            "source array is required");
        return NAN;
    }
    return cnp_legacy_nan_statistic_value(
        cnp_nanmedian_v2(arr, axis, axis == CNP_AXIS_NONE),
        "cnp_nanmedian");
}

/* =========================================================================
 * cnp_nanpercentile - Percentile ignoring NaN
 * numpy.nanpercentile(a, q, axis=None)
 * ========================================================================= */
CNP_API double CNP_CALL cnp_nanpercentile(const CnpArray *arr, double q, int axis) {
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_nanpercentile",
            "source array is required");
        return NAN;
    }
    return cnp_legacy_nan_statistic_value(
        cnp_nanpercentile_v2(
            arr, q, axis, axis == CNP_AXIS_NONE),
        "cnp_nanpercentile");
}

static int cnp_reduction_compare_double(
    const void *left_pointer, const void *right_pointer) {
    double left = *(const double*)left_pointer;
    double right = *(const double*)right_pointer;
    return (left > right) - (left < right);
}

static CnpArray *cnp_percentile_family_v2(
    const CnpArray *arr, double q, int axis, bool axis_none,
    bool ignore_nan, bool median, const char *function_name) {
    if (isnan(q) || q < 0.0 || q > 100.0) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "q must be in [0, 100]");
        return NULL;
    }
    int resolved_axis;
    if (!cnp_reduction_resolve_axis_strict_scalar(
            arr, axis, axis_none,
            function_name, &resolved_axis)) return NULL;

    CnpReductionTraversal traversal;
    cnp_reduction_traversal_init(arr, resolved_axis, &traversal);
    CNP_TYPE out_dtype = median &&
        arr->dtype->type_num == CNP_FLOAT ? CNP_FLOAT : CNP_DOUBLE;
    CnpArray *result = cnp_array_new(
        traversal.result_ndim, traversal.result_shape,
        out_dtype, CNP_ORDER_C);
    if (!result) return NULL;

    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            int64_t value_count = 0;
            bool encountered_nan = false;
            double *values = NULL;
            if (traversal.axis_length != 0) {
                values = (double*)cnp_malloc(
                    traversal.axis_length * sizeof(double));
                if (!values) {
                    cnp_array_free(result);
                    cnp_set_error(CNP_ERR_MEMORY, function_name,
                                  "failed to allocate slice buffer");
                    return NULL;
                }
            }
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item) {
                int64_t source_offset = cnp_reduction_source_offset(
                    &traversal, outer, inner, item);
                double value = cnp_get_element_double(
                    arr->data, source_offset,
                    arr->dtype->type_num);
                if (isnan(value)) {
                    encountered_nan = true;
                    if (ignore_nan) continue;
                }
                values[value_count++] = value;
            }

            double output_value;
            if ((!ignore_nan && encountered_nan) || value_count == 0) {
                if (!median && !ignore_nan && value_count == 0) {
                    if (values)
                        cnp_free(values,
                                 traversal.axis_length * sizeof(double));
                    cnp_array_free(result);
                    cnp_set_error(CNP_ERR_GENERIC, function_name,
                                  "cannot percentile an empty slice");
                    return NULL;
                }
                output_value = NAN;
            } else {
                qsort(values, (size_t)value_count,
                      sizeof(double), cnp_reduction_compare_double);
                double rank = (q / 100.0) * (double)(value_count - 1);
                int64_t lower;
                int64_t upper;
                double fraction;
                if (rank >= (double)(value_count - 1)) {
                    lower = value_count - 1;
                    upper = value_count - 1;
                    fraction = rank + 1.0;
                } else {
                    lower = (int64_t)floor(rank);
                    upper = lower + 1;
                    fraction = rank - (double)lower;
                }
                double difference = values[upper] - values[lower];
                output_value = fraction < 0.5
                    ? values[lower] + difference * fraction
                    : values[upper] - difference * (1.0 - fraction);
            }
            if (values)
                cnp_free(values,
                         traversal.axis_length * sizeof(double));
            int64_t output_index = outer * traversal.inner + inner;
            cnp_set_element_double(
                result->data, output_index * result->dtype->elsize,
                out_dtype, output_value);
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_median_v2(
    const CnpArray *arr, int axis, bool axis_none) {
    return cnp_percentile_family_v2(
        arr, 50.0, axis, axis_none,
        false, true, "cnp_median_v2");
}

CNP_API CnpArray* CNP_CALL cnp_percentile_v2(
    const CnpArray *arr, double q, int axis, bool axis_none) {
    return cnp_percentile_family_v2(
        arr, q, axis, axis_none,
        false, false, "cnp_percentile_v2");
}

CNP_API CnpArray* CNP_CALL cnp_nanmedian_v2(
    const CnpArray *arr, int axis, bool axis_none) {
    return cnp_percentile_family_v2(
        arr, 50.0, axis, axis_none,
        true, true, "cnp_nanmedian_v2");
}

CNP_API CnpArray* CNP_CALL cnp_nanpercentile_v2(
    const CnpArray *arr, double q, int axis, bool axis_none) {
    return cnp_percentile_family_v2(
        arr, q, axis, axis_none,
        true, false, "cnp_nanpercentile_v2");
}

CNP_API CnpArray* CNP_CALL cnp_quantile_v2(
    const CnpArray *arr, double q, int axis, bool axis_none) {
    if (isnan(q) || q < 0.0 || q > 1.0) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_quantile_v2",
                      "q must be in [0, 1]");
        return NULL;
    }
    return cnp_percentile_family_v2(
        arr, q * 100.0, axis, axis_none,
        false, false, "cnp_quantile_v2");
}

CNP_API CnpArray* CNP_CALL cnp_nanquantile_v2(
    const CnpArray *arr, double q, int axis, bool axis_none) {
    if (isnan(q) || q < 0.0 || q > 1.0) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_nanquantile_v2",
                      "q must be in [0, 1]");
        return NULL;
    }
    return cnp_percentile_family_v2(
        arr, q * 100.0, axis, axis_none,
        true, false, "cnp_nanquantile_v2");
}

/* =========================================================================
 * cnp_nextafter - Next representable floating point value
 * numpy.nextafter(x1, x2)
 * ========================================================================= */
static int typeconv_real_ufunc_loop_rank(CNP_TYPE source_type) {
    switch (source_type) {
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

static CNP_TYPE typeconv_real_binary_result_type(
    CNP_TYPE left_type, CNP_TYPE right_type) {
    int left_rank = typeconv_real_ufunc_loop_rank(left_type);
    int right_rank = typeconv_real_ufunc_loop_rank(right_type);
    if (left_rank < 0 || right_rank < 0)
        return CNP_NOTYPE;
    int result_rank = left_rank > right_rank ? left_rank : right_rank;
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

static bool typeconv_binary_broadcast_shape(
    const CnpArray *left,
    const CnpArray *right,
    int *result_ndim,
    int64_t *result_shape,
    const char *function_name
) {
    int ndim = left->ndim > right->ndim ? left->ndim : right->ndim;
    for (int axis = 0; axis < ndim; ++axis) {
        int left_axis = axis - (ndim - left->ndim);
        int right_axis = axis - (ndim - right->ndim);
        int64_t left_dimension =
            left_axis < 0 ? 1 : left->shape[left_axis];
        int64_t right_dimension =
            right_axis < 0 ? 1 : right->shape[right_axis];
        if (left_dimension == right_dimension) {
            result_shape[axis] = left_dimension;
        } else if (left_dimension == 1) {
            result_shape[axis] = right_dimension;
        } else if (right_dimension == 1) {
            result_shape[axis] = left_dimension;
        } else {
            cnp_set_error(
                CNP_ERR_BROADCAST, function_name,
                "cannot broadcast result axis %d with dimensions %lld and %lld",
                axis,
                (long long)left_dimension,
                (long long)right_dimension);
            return false;
        }
    }
    *result_ndim = ndim;
    return true;
}

static CNP_ORDER typeconv_result_order(
    int input_count, const CnpArray *const *inputs) {
    bool prefer_fortran = false;
    for (int input_index = 0; input_index < input_count; ++input_index) {
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

static CNP_ORDER typeconv_binary_result_order(
    const CnpArray *left, const CnpArray *right) {
    const CnpArray *inputs[2] = {left, right};
    return typeconv_result_order(2, inputs);
}

static int64_t typeconv_binary_broadcast_offset(
    const CnpArray *array,
    const int64_t *result_coordinates,
    int result_ndim
) {
    int result_axis = result_ndim - array->ndim;
    int64_t offset = array->offset;
    for (int axis = 0; axis < array->ndim; ++axis) {
        int64_t coordinate = array->shape[axis] == 1
            ? 0
            : result_coordinates[result_axis + axis];
        offset += coordinate * array->strides[axis];
    }
    return offset;
}

static uint16_t typeconv_read_half(
    const CnpArray *array, int64_t offset) {
    if (array->dtype->type_num == CNP_HALF)
        return *(const uint16_t*)((const char*)array->data + offset);
    return cnp_float_to_half(cnp_get_element_double(
        array->data, offset, array->dtype->type_num));
}

static float typeconv_read_float(
    const CnpArray *array, int64_t offset) {
    const char *source = (const char*)array->data + offset;
    if (array->dtype->type_num == CNP_FLOAT)
        return *(const float*)source;
    if (array->dtype->type_num == CNP_HALF) {
        uint16_t bits = *(const uint16_t*)source;
        return (float)cnp_half_to_float(bits);
    }
    return (float)cnp_get_element_double(
        array->data, offset, array->dtype->type_num);
}

static uint32_t typeconv_half_to_float_bits(uint16_t half_bits) {
    uint32_t sign = ((uint32_t)half_bits & UINT32_C(0x8000)) << 16;
    uint32_t exponent = ((uint32_t)half_bits >> 10) & UINT32_C(0x1f);
    uint32_t fraction = (uint32_t)half_bits & UINT32_C(0x03ff);
    if (exponent == 0) {
        if (fraction == 0)
            return sign;
        int unbiased_exponent = -14;
        while ((fraction & UINT32_C(0x0400)) == 0) {
            fraction <<= 1;
            --unbiased_exponent;
        }
        fraction &= UINT32_C(0x03ff);
        return sign |
            ((uint32_t)(unbiased_exponent + 127) << 23) |
            (fraction << 13);
    }
    if (exponent == UINT32_C(0x1f))
        return sign | UINT32_C(0x7f800000) | (fraction << 13);
    return sign | ((exponent + UINT32_C(112)) << 23) |
        (fraction << 13);
}

static uint32_t typeconv_read_float_bits(
    const CnpArray *array, int64_t offset) {
    uint32_t bits;
    if (array->dtype->type_num == CNP_FLOAT) {
        memcpy(
            &bits, (const char*)array->data + offset,
            sizeof(bits));
        return bits;
    }
    if (array->dtype->type_num == CNP_HALF) {
        uint16_t half_bits;
        memcpy(
            &half_bits, (const char*)array->data + offset,
            sizeof(half_bits));
        return typeconv_half_to_float_bits(half_bits);
    }
    float value = typeconv_read_float(array, offset);
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static double typeconv_read_double(
    const CnpArray *array, int64_t offset) {
    const char *source = (const char*)array->data + offset;
    if (array->dtype->type_num == CNP_DOUBLE)
        return *(const double*)source;
    if (array->dtype->type_num == CNP_FLOAT)
        return (double)*(const float*)source;
    if (array->dtype->type_num == CNP_HALF) {
        uint16_t bits = *(const uint16_t*)source;
        return cnp_half_to_float(bits);
    }
    return cnp_get_element_double(
        array->data, offset, array->dtype->type_num);
}

static uint64_t typeconv_read_double_bits(
    const CnpArray *array, int64_t offset) {
    uint64_t bits;
    if (array->dtype->type_num == CNP_DOUBLE) {
        memcpy(
            &bits, (const char*)array->data + offset,
            sizeof(bits));
        return bits;
    }
    double value = typeconv_read_double(array, offset);
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static long double typeconv_read_longdouble(
    const CnpArray *array, int64_t offset) {
    if (array->dtype->type_num == CNP_LONGDOUBLE) {
        return *(const long double*)(
            (const char*)array->data + offset);
    }
    return (long double)typeconv_read_double(array, offset);
}

static uint64_t typeconv_read_longdouble_bits(
    const CnpArray *array, int64_t offset) {
    uint64_t bits;
    if (array->dtype->type_num == CNP_LONGDOUBLE) {
        memcpy(
            &bits, (const char*)array->data + offset,
            sizeof(bits));
        return bits;
    }
    long double value = typeconv_read_longdouble(array, offset);
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint16_t typeconv_nextafter_half_bits(
    uint16_t left_bits, uint16_t right_bits) {
    uint16_t left_magnitude = left_bits & UINT16_C(0x7fff);
    uint16_t right_magnitude = right_bits & UINT16_C(0x7fff);
    if (left_magnitude > UINT16_C(0x7c00) ||
            right_magnitude > UINT16_C(0x7c00))
        return UINT16_C(0x7e00);

    double left = cnp_half_to_float(left_bits);
    double right = cnp_half_to_float(right_bits);
    if (left == right)
        return left_bits;
    if (left_magnitude == 0)
        return (right_bits & UINT16_C(0x8000)) | UINT16_C(1);

    bool move_up = left < right;
    if (left_bits & UINT16_C(0x8000))
        return move_up ? left_bits - UINT16_C(1)
                       : left_bits + UINT16_C(1);
    return move_up ? left_bits + UINT16_C(1)
                   : left_bits - UINT16_C(1);
}

static uint32_t typeconv_nextafter_float_bits(
    uint32_t left_bits, uint32_t right_bits) {
    if ((left_bits & UINT32_C(0x7f800000)) ==
            UINT32_C(0x7f800000) &&
            (left_bits & UINT32_C(0x007fffff)) != 0)
        return left_bits;
    if ((right_bits & UINT32_C(0x7f800000)) ==
            UINT32_C(0x7f800000) &&
            (right_bits & UINT32_C(0x007fffff)) != 0) {
        right_bits |= UINT32_C(0x00400000);
        return right_bits;
    }
    float left;
    float right;
    float result;
    uint32_t result_bits;
    memcpy(&left, &left_bits, sizeof(left));
    memcpy(&right, &right_bits, sizeof(right));
    result = nextafterf(left, right);
    memcpy(&result_bits, &result, sizeof(result_bits));
    return result_bits;
}

CNP_API CnpArray* CNP_CALL cnp_nextafter(const CnpArray *x1, const CnpArray *x2) {
    const char *function_name = "cnp_nextafter";
    if (!x1) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "x1 must not be NULL");
        return NULL;
    }
    if (!x2) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "x2 must not be NULL");
        return NULL;
    }

    CNP_TYPE result_type = typeconv_real_binary_result_type(
        x1->dtype->type_num, x2->dtype->type_num);
    if (result_type == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "nextafter requires real numeric dtypes, got %d and %d",
            (int)x1->dtype->type_num,
            (int)x2->dtype->type_num);
        return NULL;
    }

    int result_ndim = 0;
    int64_t result_shape[CNP_MAXDIMS] = {0};
    if (!typeconv_binary_broadcast_shape(
            x1, x2, &result_ndim, result_shape, function_name))
        return NULL;

    CnpArray *result = cnp_array_new(
        result_ndim,
        result_ndim == 0 ? NULL : result_shape,
        result_type,
        typeconv_binary_result_order(x1, x2));
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t left_offset = typeconv_binary_broadcast_offset(
            x1, coordinates, result_ndim);
        int64_t right_offset = typeconv_binary_broadcast_offset(
            x2, coordinates, result_ndim);
        int64_t result_offset = result->offset;
        for (int axis = 0; axis < result_ndim; ++axis)
            result_offset += coordinates[axis] * result->strides[axis];
        char *destination = (char*)result->data + result_offset;

        switch (result_type) {
            case CNP_HALF: {
                uint16_t left_bits = typeconv_read_half(
                    x1, left_offset);
                uint16_t right_bits = typeconv_read_half(
                    x2, right_offset);
                *(uint16_t*)destination = typeconv_nextafter_half_bits(
                    left_bits, right_bits);
                break;
            }
            case CNP_FLOAT: {
                uint32_t left_bits = typeconv_read_float_bits(
                    x1, left_offset);
                uint32_t right_bits = typeconv_read_float_bits(
                    x2, right_offset);
                *(uint32_t*)destination = typeconv_nextafter_float_bits(
                    left_bits, right_bits);
                break;
            }
            case CNP_DOUBLE: {
                double left = typeconv_read_double(x1, left_offset);
                double right = typeconv_read_double(x2, right_offset);
                *(double*)destination = nextafter(left, right);
                break;
            }
            case CNP_LONGDOUBLE: {
                long double left = typeconv_read_longdouble(
                    x1, left_offset);
                long double right = typeconv_read_longdouble(
                    x2, right_offset);
                *(long double*)destination = nextafterl(left, right);
                break;
            }
            default: {
                CNP_TYPE unexpected_type = result_type;
                cnp_array_free(result);
                cnp_set_error(
                    CNP_ERR_TYPE, function_name,
                    "internal nextafter result dtype %d is unsupported",
                    (int)unexpected_type);
                return NULL;
            }
        }

        for (int axis = result_ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < result_shape[axis])
                break;
            coordinates[axis] = 0;
        }
    }
    return result;
}

static int64_t typeconv_flat_offset(
    const CnpArray *array, int64_t flat_index) {
    int64_t coordinates[CNP_MAXDIMS] = {0};
    int64_t remaining = flat_index;
    for (int dimension = array->ndim - 1; dimension >= 0; --dimension) {
        coordinates[dimension] = remaining % array->shape[dimension];
        remaining /= array->shape[dimension];
    }
    return array->offset + cnp_multi_to_offset(
        array->ndim, coordinates, array->strides);
}

static CNP_TYPE typeconv_spacing_result_type(CNP_TYPE source_type) {
    switch (source_type) {
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

static uint16_t typeconv_spacing_half_bits(uint16_t value_bits) {
    uint16_t magnitude = value_bits & UINT16_C(0x7fff);
    if (magnitude >= UINT16_C(0x7c00))
        return UINT16_C(0x7e00);

    uint16_t adjacent_bits;
    if (magnitude == 0) {
        adjacent_bits = UINT16_C(1);
    } else if (value_bits & UINT16_C(0x8000)) {
        adjacent_bits = value_bits - UINT16_C(1);
    } else {
        adjacent_bits = value_bits + UINT16_C(1);
    }
    float value = (float)cnp_half_to_float(value_bits);
    float adjacent = (float)cnp_half_to_float(adjacent_bits);
    return cnp_float_to_half((double)(adjacent - value));
}

static float typeconv_spacing_float_value(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
        if ((bits & UINT32_C(0x007fffff)) == 0)
            return cnp_typeconv_quiet_nan_float();
        bits |= UINT32_C(0x00400000);
        memcpy(&value, &bits, sizeof(value));
        return value;
    }
    float direction = value < 0.0f
        ? -cnp_typeconv_positive_infinity_float()
        : cnp_typeconv_positive_infinity_float();
    return nextafterf(value, direction) - value;
}

static double typeconv_spacing_double_value(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT64_C(0x7ff0000000000000)) ==
            UINT64_C(0x7ff0000000000000)) {
        if ((bits & UINT64_C(0x000fffffffffffff)) == 0)
            return cnp_typeconv_quiet_nan();
        bits |= UINT64_C(0x0008000000000000);
        memcpy(&value, &bits, sizeof(value));
        return value;
    }
    double direction = value < 0.0
        ? -cnp_typeconv_positive_infinity()
        : cnp_typeconv_positive_infinity();
    return nextafter(value, direction) - value;
}

static long double typeconv_spacing_longdouble_value(long double value) {
    if (sizeof(long double) == sizeof(double)) {
        uint64_t bits;
        memcpy(&bits, &value, sizeof(bits));
        if ((bits & UINT64_C(0x7ff0000000000000)) ==
                UINT64_C(0x7ff0000000000000)) {
            uint64_t fraction =
                bits & UINT64_C(0x000fffffffffffff);
            if (fraction == 0) {
                double quiet_nan = cnp_typeconv_quiet_nan();
                memcpy(&value, &quiet_nan, sizeof(quiet_nan));
                return value;
            }
            bits |= UINT64_C(0x0008000000000000);
            if ((bits & UINT64_C(0x000fffffffffffff)) !=
                    UINT64_C(0x000fffffffffffff))
                bits += UINT64_C(1);
            memcpy(&value, &bits, sizeof(bits));
            return value;
        }
    } else if (isinf(value)) {
        return (long double)cnp_typeconv_quiet_nan();
    } else if (isnan(value)) {
        return value + value;
    }
    long double direction = value < 0.0L
        ? -(long double)cnp_typeconv_positive_infinity()
        : (long double)cnp_typeconv_positive_infinity();
    return nextafterl(value, direction) - value;
}

/* =========================================================================
 * cnp_spacing - Distance between adjacent floating point values
 * numpy.spacing(x)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_spacing(const CnpArray *x) {
    const char *function_name = "cnp_spacing";
    if (!x) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array must not be NULL");
        return NULL;
    }

    CNP_TYPE result_type = typeconv_spacing_result_type(
        x->dtype->type_num);
    if (result_type == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "spacing requires a real numeric dtype, got %d",
            (int)x->dtype->type_num);
        return NULL;
    }
    CNP_ORDER result_order =
        (x->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(x->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        x->ndim, x->shape, result_type, result_order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    for (int64_t index = 0; index < x->size; ++index) {
        int64_t source_offset = typeconv_flat_offset(x, index);
        int64_t result_offset = typeconv_flat_offset(result, index);
        const char *source = (const char*)x->data + source_offset;
        char *destination = (char*)result->data + result_offset;
        switch (result_type) {
            case CNP_HALF: {
                uint16_t value_bits = x->dtype->type_num == CNP_HALF
                    ? *(const uint16_t*)source
                    : cnp_float_to_half(cnp_get_element_double(
                        x->data, source_offset, x->dtype->type_num));
                *(uint16_t*)destination =
                    typeconv_spacing_half_bits(value_bits);
                break;
            }
            case CNP_FLOAT: {
                float value = (float)cnp_get_element_double(
                    x->data, source_offset, x->dtype->type_num);
                *(float*)destination =
                    typeconv_spacing_float_value(value);
                break;
            }
            case CNP_DOUBLE: {
                double value = cnp_get_element_double(
                    x->data, source_offset, x->dtype->type_num);
                *(double*)destination =
                    typeconv_spacing_double_value(value);
                break;
            }
            case CNP_LONGDOUBLE: {
                long double value = *(const long double*)source;
                *(long double*)destination =
                    typeconv_spacing_longdouble_value(value);
                break;
            }
            default:
                break;
        }
    }
    return result;
}

/* =========================================================================
 * cnp_copysign - Copy sign of one number to another
 * numpy.copysign(x1, x2)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_copysign(const CnpArray *x1, const CnpArray *x2) {
    const char *function_name = "cnp_copysign";
    if (!x1) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "x1 must not be NULL");
        return NULL;
    }
    if (!x2) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "x2 must not be NULL");
        return NULL;
    }

    CNP_TYPE result_type = typeconv_real_binary_result_type(
        x1->dtype->type_num, x2->dtype->type_num);
    if (result_type == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "copysign requires real numeric dtypes, got %d and %d",
            (int)x1->dtype->type_num,
            (int)x2->dtype->type_num);
        return NULL;
    }

    int result_ndim = 0;
    int64_t result_shape[CNP_MAXDIMS] = {0};
    if (!typeconv_binary_broadcast_shape(
            x1, x2, &result_ndim, result_shape, function_name))
        return NULL;

    CnpArray *result = cnp_array_new(
        result_ndim,
        result_ndim == 0 ? NULL : result_shape,
        result_type,
        typeconv_binary_result_order(x1, x2));
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t left_offset = typeconv_binary_broadcast_offset(
            x1, coordinates, result_ndim);
        int64_t right_offset = typeconv_binary_broadcast_offset(
            x2, coordinates, result_ndim);
        int64_t result_offset = result->offset;
        for (int axis = 0; axis < result_ndim; ++axis)
            result_offset += coordinates[axis] * result->strides[axis];
        char *destination = (char*)result->data + result_offset;

        switch (result_type) {
            case CNP_HALF: {
                uint16_t left_bits = typeconv_read_half(
                    x1, left_offset);
                uint16_t right_bits = typeconv_read_half(
                    x2, right_offset);
                *(uint16_t*)destination =
                    (left_bits & UINT16_C(0x7fff)) |
                    (right_bits & UINT16_C(0x8000));
                break;
            }
            case CNP_FLOAT: {
                uint32_t left_bits = typeconv_read_float_bits(
                    x1, left_offset);
                uint32_t right_bits = typeconv_read_float_bits(
                    x2, right_offset);
                *(uint32_t*)destination =
                    (left_bits & UINT32_C(0x7fffffff)) |
                    (right_bits & UINT32_C(0x80000000));
                break;
            }
            case CNP_DOUBLE: {
                uint64_t left_bits = typeconv_read_double_bits(
                    x1, left_offset);
                uint64_t right_bits = typeconv_read_double_bits(
                    x2, right_offset);
                *(uint64_t*)destination =
                    (left_bits & UINT64_C(0x7fffffffffffffff)) |
                    (right_bits & UINT64_C(0x8000000000000000));
                break;
            }
            case CNP_LONGDOUBLE: {
                if (sizeof(long double) == sizeof(uint64_t)) {
                    uint64_t left_bits = typeconv_read_longdouble_bits(
                        x1, left_offset);
                    uint64_t right_bits = typeconv_read_longdouble_bits(
                        x2, right_offset);
                    *(uint64_t*)destination =
                        (left_bits & UINT64_C(0x7fffffffffffffff)) |
                        (right_bits & UINT64_C(0x8000000000000000));
                } else {
                    long double left = typeconv_read_longdouble(
                        x1, left_offset);
                    long double right = typeconv_read_longdouble(
                        x2, right_offset);
                    *(long double*)destination = copysignl(left, right);
                }
                break;
            }
            default: {
                CNP_TYPE unexpected_type = result_type;
                cnp_array_free(result);
                cnp_set_error(
                    CNP_ERR_TYPE, function_name,
                    "internal copysign result dtype %d is unsupported",
                    (int)unexpected_type);
                return NULL;
            }
        }

        for (int axis = result_ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < result_shape[axis])
                break;
            coordinates[axis] = 0;
        }
    }
    return result;
}

/* =========================================================================
 * cnp_frexp - Decompose into mantissa and exponent
 * numpy.frexp(x) -> (mantissa, exponent)
 * ========================================================================= */
static uint16_t typeconv_frexp_half_bits(
    uint16_t source_bits, int32_t *exponent_out) {
    uint16_t exponent =
        (source_bits >> 10) & UINT16_C(0x001f);
    uint16_t fraction = source_bits & UINT16_C(0x03ff);
    if (exponent == UINT16_C(0x001f)) {
        *exponent_out = -1;
        if (fraction != 0)
            source_bits |= UINT16_C(0x0200);
        return source_bits;
    }
    if (exponent != 0) {
        *exponent_out = (int32_t)exponent - 14;
        return (source_bits & UINT16_C(0x83ff)) |
            UINT16_C(0x3800);
    }
    if (fraction == 0) {
        *exponent_out = 0;
        return source_bits;
    }

    int32_t shift = 0;
    while ((fraction & UINT16_C(0x0400)) == 0) {
        fraction <<= 1;
        ++shift;
    }
    *exponent_out = -13 - shift;
    return (source_bits & UINT16_C(0x8000)) |
        UINT16_C(0x3800) |
        (fraction & UINT16_C(0x03ff));
}

static uint32_t typeconv_frexp_float_bits(
    uint32_t source_bits, int32_t *exponent_out) {
    uint32_t exponent =
        (source_bits >> 23) & UINT32_C(0x00ff);
    uint32_t fraction = source_bits & UINT32_C(0x007fffff);
    if (exponent == UINT32_C(0x00ff)) {
        *exponent_out = -1;
        if (fraction != 0)
            source_bits |= UINT32_C(0x00400000);
        return source_bits;
    }
    if (exponent != 0) {
        *exponent_out = (int32_t)exponent - 126;
        return (source_bits & UINT32_C(0x807fffff)) |
            UINT32_C(0x3f000000);
    }
    if (fraction == 0) {
        *exponent_out = 0;
        return source_bits;
    }

    int32_t shift = 0;
    while ((fraction & UINT32_C(0x00800000)) == 0) {
        fraction <<= 1;
        ++shift;
    }
    *exponent_out = -125 - shift;
    return (source_bits & UINT32_C(0x80000000)) |
        UINT32_C(0x3f000000) |
        (fraction & UINT32_C(0x007fffff));
}

static uint64_t typeconv_frexp_double_bits(
    uint64_t source_bits, int32_t *exponent_out) {
    uint64_t exponent =
        (source_bits >> 52) & UINT64_C(0x07ff);
    uint64_t fraction =
        source_bits & UINT64_C(0x000fffffffffffff);
    if (exponent == UINT64_C(0x07ff)) {
        *exponent_out = -1;
        if (fraction != 0)
            source_bits |= UINT64_C(0x0008000000000000);
        return source_bits;
    }
    if (exponent != 0) {
        *exponent_out = (int32_t)exponent - 1022;
        return (source_bits & UINT64_C(0x800fffffffffffff)) |
            UINT64_C(0x3fe0000000000000);
    }
    if (fraction == 0) {
        *exponent_out = 0;
        return source_bits;
    }

    int32_t shift = 0;
    while ((fraction & UINT64_C(0x0010000000000000)) == 0) {
        fraction <<= 1;
        ++shift;
    }
    *exponent_out = -1021 - shift;
    return (source_bits & UINT64_C(0x8000000000000000)) |
        UINT64_C(0x3fe0000000000000) |
        (fraction & UINT64_C(0x000fffffffffffff));
}

CNP_API CNP_STATUS CNP_CALL cnp_frexp(const CnpArray *x, CnpArray **mantissa, CnpArray **exponent) {
    const char *function_name = "cnp_frexp";
    if (mantissa)
        *mantissa = NULL;
    if (exponent)
        *exponent = NULL;
    if (!x) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "x must not be NULL");
        return CNP_ERR_GENERIC;
    }
    if (!mantissa) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "mantissa output must not be NULL");
        return CNP_ERR_GENERIC;
    }
    if (!exponent) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "exponent output must not be NULL");
        return CNP_ERR_GENERIC;
    }

    CNP_TYPE mantissa_type = typeconv_real_binary_result_type(
        x->dtype->type_num, x->dtype->type_num);
    if (mantissa_type == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "frexp requires a real numeric dtype, got %d",
            (int)x->dtype->type_num);
        return CNP_ERR_TYPE;
    }

    CNP_ORDER result_order = typeconv_binary_result_order(x, x);
    CnpArray *mantissa_result = cnp_array_new(
        x->ndim, x->ndim == 0 ? NULL : x->shape,
        mantissa_type, result_order);
    if (!mantissa_result) {
        cnp_relabel_error(function_name);
        return CNP_ERR_MEMORY;
    }
    CnpArray *exponent_result = cnp_array_new(
        x->ndim, x->ndim == 0 ? NULL : x->shape,
        CNP_INT, result_order);
    if (!exponent_result) {
        cnp_array_free(mantissa_result);
        cnp_relabel_error(function_name);
        return CNP_ERR_MEMORY;
    }

    for (int64_t index = 0; index < x->size; ++index) {
        int64_t source_offset = typeconv_flat_offset(x, index);
        int64_t mantissa_offset = typeconv_flat_offset(
            mantissa_result, index);
        int64_t exponent_offset = typeconv_flat_offset(
            exponent_result, index);
        char *mantissa_destination =
            (char*)mantissa_result->data + mantissa_offset;
        int32_t *exponent_destination = (int32_t*)(
            (char*)exponent_result->data + exponent_offset);

        switch (mantissa_type) {
            case CNP_HALF: {
                uint16_t source_bits = typeconv_read_half(
                    x, source_offset);
                *(uint16_t*)mantissa_destination =
                    typeconv_frexp_half_bits(
                        source_bits, exponent_destination);
                break;
            }
            case CNP_FLOAT: {
                uint32_t source_bits = typeconv_read_float_bits(
                    x, source_offset);
                *(uint32_t*)mantissa_destination =
                    typeconv_frexp_float_bits(
                        source_bits, exponent_destination);
                break;
            }
            case CNP_DOUBLE: {
                uint64_t source_bits = typeconv_read_double_bits(
                    x, source_offset);
                *(uint64_t*)mantissa_destination =
                    typeconv_frexp_double_bits(
                        source_bits, exponent_destination);
                break;
            }
            case CNP_LONGDOUBLE: {
                if (sizeof(long double) == sizeof(uint64_t)) {
                    uint64_t source_bits = typeconv_read_longdouble_bits(
                        x, source_offset);
                    *(uint64_t*)mantissa_destination =
                        typeconv_frexp_double_bits(
                            source_bits, exponent_destination);
                } else {
                    long double value = typeconv_read_longdouble(
                        x, source_offset);
                    int native_exponent = 0;
                    *(long double*)mantissa_destination = frexpl(
                        value, &native_exponent);
                    *exponent_destination = (int32_t)native_exponent;
                }
                break;
            }
            default: {
                CNP_TYPE unexpected_type = mantissa_type;
                cnp_array_free(mantissa_result);
                cnp_array_free(exponent_result);
                cnp_set_error(
                    CNP_ERR_TYPE, function_name,
                    "internal frexp mantissa dtype %d is unsupported",
                    (int)unexpected_type);
                return CNP_ERR_TYPE;
            }
        }
    }

    *mantissa = mantissa_result;
    *exponent = exponent_result;
    return CNP_OK;
}

/* =========================================================================
 * cnp_ldexp - Multiply by power of 2
 * numpy.ldexp(x1, x2) = x1 * 2**x2
 * ========================================================================= */
static bool typeconv_ldexp_exponent_type_supported(CNP_TYPE dtype) {
    switch (dtype) {
        case CNP_BOOL:
        case CNP_BYTE:
        case CNP_UBYTE:
        case CNP_SHORT:
        case CNP_USHORT:
        case CNP_INT:
            return true;
        default:
            return false;
    }
}

static bool typeconv_ldexp_read_exponent(
    const CnpArray *array, int64_t offset,
    int32_t *exponent_out, const char *function_name) {
    const char *source = (const char*)array->data + offset;
    switch (array->dtype->type_num) {
        case CNP_BOOL:
        case CNP_BYTE:
            *exponent_out = (int32_t)*(const int8_t*)source;
            return true;
        case CNP_UBYTE:
            *exponent_out = (int32_t)*(const uint8_t*)source;
            return true;
        case CNP_SHORT:
            *exponent_out = (int32_t)*(const int16_t*)source;
            return true;
        case CNP_USHORT:
            *exponent_out = (int32_t)*(const uint16_t*)source;
            return true;
        case CNP_INT:
            *exponent_out = *(const int32_t*)source;
            return true;
        default:
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "internal ldexp exponent dtype %d is unsupported",
                (int)array->dtype->type_num);
            return false;
    }
}

static uint16_t typeconv_ldexp_half_bits(
    uint16_t source_bits, int exponent) {
    uint16_t source_exponent =
        source_bits & UINT16_C(0x7c00);
    uint16_t fraction = source_bits & UINT16_C(0x03ff);
    if (source_exponent == UINT16_C(0x7c00)) {
        if (fraction != 0)
            source_bits |= UINT16_C(0x0200);
        return source_bits;
    }
    double value = cnp_half_to_float(source_bits);
    return cnp_float_to_half(ldexp(value, exponent));
}

static uint32_t typeconv_ldexp_float_bits(
    uint32_t source_bits, int exponent) {
    uint32_t source_exponent =
        source_bits & UINT32_C(0x7f800000);
    uint32_t fraction = source_bits & UINT32_C(0x007fffff);
    if (source_exponent == UINT32_C(0x7f800000)) {
        if (fraction != 0)
            source_bits |= UINT32_C(0x00400000);
        return source_bits;
    }
    float value;
    float result;
    uint32_t result_bits;
    memcpy(&value, &source_bits, sizeof(value));
    result = ldexpf(value, exponent);
    memcpy(&result_bits, &result, sizeof(result_bits));
    return result_bits;
}

static uint64_t typeconv_ldexp_double_bits(
    uint64_t source_bits, int exponent) {
    uint64_t source_exponent =
        source_bits & UINT64_C(0x7ff0000000000000);
    uint64_t fraction =
        source_bits & UINT64_C(0x000fffffffffffff);
    if (source_exponent == UINT64_C(0x7ff0000000000000)) {
        if (fraction != 0)
            source_bits |= UINT64_C(0x0008000000000000);
        return source_bits;
    }
    double value;
    double result;
    uint64_t result_bits;
    memcpy(&value, &source_bits, sizeof(value));
    result = ldexp(value, exponent);
    memcpy(&result_bits, &result, sizeof(result_bits));
    return result_bits;
}

CNP_API CnpArray* CNP_CALL cnp_ldexp(const CnpArray *x1, const CnpArray *x2) {
    const char *function_name = "cnp_ldexp";
    if (!x1) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "x1 must not be NULL");
        return NULL;
    }
    if (!x2) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "x2 must not be NULL");
        return NULL;
    }

    CNP_TYPE result_type = typeconv_real_binary_result_type(
        x1->dtype->type_num, x1->dtype->type_num);
    if (result_type == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "ldexp requires a real numeric dtype, got %d",
            (int)x1->dtype->type_num);
        return NULL;
    }
    if (!typeconv_ldexp_exponent_type_supported(
            x2->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "ldexp requires an integer exponent dtype safely castable to int32, got %d",
            (int)x2->dtype->type_num);
        return NULL;
    }

    int result_ndim = 0;
    int64_t result_shape[CNP_MAXDIMS] = {0};
    if (!typeconv_binary_broadcast_shape(
            x1, x2, &result_ndim, result_shape, function_name))
        return NULL;

    CnpArray *result = cnp_array_new(
        result_ndim,
        result_ndim == 0 ? NULL : result_shape,
        result_type,
        typeconv_binary_result_order(x1, x2));
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t value_offset = typeconv_binary_broadcast_offset(
            x1, coordinates, result_ndim);
        int64_t exponent_offset = typeconv_binary_broadcast_offset(
            x2, coordinates, result_ndim);
        int64_t result_offset = result->offset;
        for (int axis = 0; axis < result_ndim; ++axis)
            result_offset += coordinates[axis] * result->strides[axis];
        char *destination = (char*)result->data + result_offset;
        int32_t exponent = 0;
        if (!typeconv_ldexp_read_exponent(
                x2, exponent_offset, &exponent, function_name)) {
            cnp_array_free(result);
            return NULL;
        }

        switch (result_type) {
            case CNP_HALF: {
                uint16_t source_bits = typeconv_read_half(
                    x1, value_offset);
                *(uint16_t*)destination = typeconv_ldexp_half_bits(
                    source_bits, (int)exponent);
                break;
            }
            case CNP_FLOAT: {
                uint32_t source_bits = typeconv_read_float_bits(
                    x1, value_offset);
                *(uint32_t*)destination = typeconv_ldexp_float_bits(
                    source_bits, (int)exponent);
                break;
            }
            case CNP_DOUBLE: {
                uint64_t source_bits = typeconv_read_double_bits(
                    x1, value_offset);
                *(uint64_t*)destination = typeconv_ldexp_double_bits(
                    source_bits, (int)exponent);
                break;
            }
            case CNP_LONGDOUBLE: {
                if (sizeof(long double) == sizeof(uint64_t)) {
                    uint64_t source_bits = typeconv_read_longdouble_bits(
                        x1, value_offset);
                    *(uint64_t*)destination = typeconv_ldexp_double_bits(
                        source_bits, (int)exponent);
                } else {
                    long double value = typeconv_read_longdouble(
                        x1, value_offset);
                    *(long double*)destination = ldexpl(
                        value, (int)exponent);
                }
                break;
            }
            default: {
                CNP_TYPE unexpected_type = result_type;
                cnp_array_free(result);
                cnp_set_error(
                    CNP_ERR_TYPE, function_name,
                    "internal ldexp result dtype %d is unsupported",
                    (int)unexpected_type);
                return NULL;
            }
        }

        for (int axis = result_ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < result_shape[axis])
                break;
            coordinates[axis] = 0;
        }
    }
    return result;
}

/* =========================================================================
 * cnp_modf - Split into fractional and integer parts
 * numpy.modf(x) -> (fractional, integer)
 * ========================================================================= */
static void typeconv_modf_half_bits(
    uint16_t source_bits,
    uint16_t *fractional_bits,
    uint16_t *integral_bits
) {
    uint16_t sign = source_bits & UINT16_C(0x8000);
    uint16_t exponent =
        (source_bits >> 10) & UINT16_C(0x001f);
    uint16_t fraction = source_bits & UINT16_C(0x03ff);
    if (exponent == UINT16_C(0x001f)) {
        if (fraction == 0) {
            *fractional_bits = sign;
            *integral_bits = source_bits;
        } else {
            uint16_t quiet_bits = source_bits | UINT16_C(0x0200);
            *fractional_bits = quiet_bits;
            *integral_bits = source_bits;
        }
        return;
    }
    if (exponent < UINT16_C(15)) {
        *fractional_bits = source_bits;
        *integral_bits = sign;
        return;
    }

    int integral_fraction_bits = (int)exponent - 15;
    if (integral_fraction_bits >= 10) {
        *fractional_bits = sign;
        *integral_bits = source_bits;
        return;
    }
    uint16_t fractional_mask = (uint16_t)(
        (UINT16_C(1) << (10 - integral_fraction_bits)) - UINT16_C(1));
    if ((source_bits & fractional_mask) == 0) {
        *fractional_bits = sign;
        *integral_bits = source_bits;
        return;
    }

    *integral_bits = source_bits & (uint16_t)~fractional_mask;
    double source_value = cnp_half_to_float(source_bits);
    double integral_value = cnp_half_to_float(*integral_bits);
    *fractional_bits = cnp_float_to_half(source_value - integral_value);
}

static void typeconv_modf_float_bits(
    uint32_t source_bits,
    uint32_t *fractional_bits,
    uint32_t *integral_bits
) {
    uint32_t sign = source_bits & UINT32_C(0x80000000);
    uint32_t exponent =
        (source_bits >> 23) & UINT32_C(0x000000ff);
    uint32_t fraction = source_bits & UINT32_C(0x007fffff);
    if (exponent == UINT32_C(0x000000ff)) {
        if (fraction == 0) {
            *fractional_bits = sign;
            *integral_bits = source_bits;
        } else {
            uint32_t quiet_bits = source_bits | UINT32_C(0x00400000);
            *fractional_bits = quiet_bits;
            *integral_bits = source_bits;
        }
        return;
    }
    if (exponent < UINT32_C(127)) {
        *fractional_bits = source_bits;
        *integral_bits = sign;
        return;
    }

    int integral_fraction_bits = (int)exponent - 127;
    if (integral_fraction_bits >= 23) {
        *fractional_bits = sign;
        *integral_bits = source_bits;
        return;
    }
    uint32_t fractional_mask =
        (UINT32_C(1) << (23 - integral_fraction_bits)) - UINT32_C(1);
    if ((source_bits & fractional_mask) == 0) {
        *fractional_bits = sign;
        *integral_bits = source_bits;
        return;
    }

    *integral_bits = source_bits & ~fractional_mask;
    float source_value;
    float integral_value;
    float fractional_value;
    memcpy(&source_value, &source_bits, sizeof(source_value));
    memcpy(&integral_value, integral_bits, sizeof(integral_value));
    fractional_value = source_value - integral_value;
    memcpy(fractional_bits, &fractional_value, sizeof(*fractional_bits));
}

static void typeconv_modf_double_bits(
    uint64_t source_bits,
    uint64_t *fractional_bits,
    uint64_t *integral_bits
) {
    uint64_t sign = source_bits & UINT64_C(0x8000000000000000);
    uint64_t exponent =
        (source_bits >> 52) & UINT64_C(0x00000000000007ff);
    uint64_t fraction = source_bits & UINT64_C(0x000fffffffffffff);
    if (exponent == UINT64_C(0x00000000000007ff)) {
        if (fraction == 0) {
            *fractional_bits = sign;
            *integral_bits = source_bits;
        } else {
            uint64_t quiet_bits =
                source_bits | UINT64_C(0x0008000000000000);
            *fractional_bits = quiet_bits;
            *integral_bits = source_bits;
        }
        return;
    }
    if (exponent < UINT64_C(1023)) {
        *fractional_bits = source_bits;
        *integral_bits = sign;
        return;
    }

    int integral_fraction_bits = (int)exponent - 1023;
    if (integral_fraction_bits >= 52) {
        *fractional_bits = sign;
        *integral_bits = source_bits;
        return;
    }
    uint64_t fractional_mask =
        (UINT64_C(1) << (52 - integral_fraction_bits)) - UINT64_C(1);
    if ((source_bits & fractional_mask) == 0) {
        *fractional_bits = sign;
        *integral_bits = source_bits;
        return;
    }

    *integral_bits = source_bits & ~fractional_mask;
    double source_value;
    double integral_value;
    double fractional_value;
    memcpy(&source_value, &source_bits, sizeof(source_value));
    memcpy(&integral_value, integral_bits, sizeof(integral_value));
    fractional_value = source_value - integral_value;
    memcpy(fractional_bits, &fractional_value, sizeof(*fractional_bits));
}

CNP_API CNP_STATUS CNP_CALL cnp_modf(const CnpArray *x, CnpArray **frac, CnpArray **integ) {
    const char *function_name = "cnp_modf";
    if (frac)
        *frac = NULL;
    if (integ)
        *integ = NULL;
    if (!x) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "x must not be NULL");
        return CNP_ERR_GENERIC;
    }
    if (!frac) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "fractional output must not be NULL");
        return CNP_ERR_GENERIC;
    }
    if (!integ) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "integral output must not be NULL");
        return CNP_ERR_GENERIC;
    }

    CNP_TYPE result_type = typeconv_real_binary_result_type(
        x->dtype->type_num, x->dtype->type_num);
    if (result_type == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "modf requires a real numeric dtype, got %d",
            (int)x->dtype->type_num);
        return CNP_ERR_TYPE;
    }

    CNP_ORDER result_order = typeconv_binary_result_order(x, x);
    CnpArray *fractional_result = cnp_array_new(
        x->ndim, x->ndim == 0 ? NULL : x->shape,
        result_type, result_order);
    if (!fractional_result) {
        cnp_relabel_error(function_name);
        return CNP_ERR_MEMORY;
    }
    CnpArray *integral_result = cnp_array_new(
        x->ndim, x->ndim == 0 ? NULL : x->shape,
        result_type, result_order);
    if (!integral_result) {
        cnp_array_free(fractional_result);
        cnp_relabel_error(function_name);
        return CNP_ERR_MEMORY;
    }

    for (int64_t index = 0; index < x->size; ++index) {
        int64_t source_offset = typeconv_flat_offset(x, index);
        int64_t fractional_offset = typeconv_flat_offset(
            fractional_result, index);
        int64_t integral_offset = typeconv_flat_offset(
            integral_result, index);
        char *fractional_destination =
            (char*)fractional_result->data + fractional_offset;
        char *integral_destination =
            (char*)integral_result->data + integral_offset;

        switch (result_type) {
            case CNP_HALF: {
                uint16_t source_bits = typeconv_read_half(
                    x, source_offset);
                typeconv_modf_half_bits(
                    source_bits,
                    (uint16_t*)fractional_destination,
                    (uint16_t*)integral_destination);
                break;
            }
            case CNP_FLOAT: {
                uint32_t source_bits = typeconv_read_float_bits(
                    x, source_offset);
                typeconv_modf_float_bits(
                    source_bits,
                    (uint32_t*)fractional_destination,
                    (uint32_t*)integral_destination);
                break;
            }
            case CNP_DOUBLE: {
                uint64_t source_bits = typeconv_read_double_bits(
                    x, source_offset);
                typeconv_modf_double_bits(
                    source_bits,
                    (uint64_t*)fractional_destination,
                    (uint64_t*)integral_destination);
                break;
            }
            case CNP_LONGDOUBLE: {
                if (sizeof(long double) == sizeof(uint64_t)) {
                    uint64_t source_bits = typeconv_read_longdouble_bits(
                        x, source_offset);
                    typeconv_modf_double_bits(
                        source_bits,
                        (uint64_t*)fractional_destination,
                        (uint64_t*)integral_destination);
                } else {
                    long double value = typeconv_read_longdouble(
                        x, source_offset);
                    long double integral_value;
                    *(long double*)fractional_destination = modfl(
                        value, &integral_value);
                    *(long double*)integral_destination = integral_value;
                }
                break;
            }
            default: {
                CNP_TYPE unexpected_type = result_type;
                cnp_array_free(fractional_result);
                cnp_array_free(integral_result);
                cnp_set_error(
                    CNP_ERR_TYPE, function_name,
                    "internal modf result dtype %d is unsupported",
                    (int)unexpected_type);
                return CNP_ERR_TYPE;
            }
        }
    }

    *frac = fractional_result;
    *integ = integral_result;
    return CNP_OK;
}

/* =========================================================================
 * cnp_finfo_eps - Machine epsilon for float type
 * ========================================================================= */
CNP_API double CNP_CALL cnp_finfo_eps(CNP_TYPE dtype) {
    CNP_TYPE component = cnp_finfo_component_type(dtype, "cnp_finfo_eps");
    switch (component) {
        case CNP_HALF: return 0x1p-10;
        case CNP_FLOAT: return FLT_EPSILON;
        case CNP_DOUBLE: return DBL_EPSILON;
        default: return cnp_typeconv_quiet_nan();
    }
}

/* =========================================================================
 * cnp_finfo_max - Maximum value for float type
 * ========================================================================= */
CNP_API double CNP_CALL cnp_finfo_max(CNP_TYPE dtype) {
    CNP_TYPE component = cnp_finfo_component_type(dtype, "cnp_finfo_max");
    switch (component) {
        case CNP_HALF: return 65504.0;
        case CNP_FLOAT: return FLT_MAX;
        case CNP_DOUBLE: return DBL_MAX;
        default: return cnp_typeconv_quiet_nan();
    }
}

/* =========================================================================
 * cnp_finfo_min - Minimum finite value for float type
 * ========================================================================= */
CNP_API double CNP_CALL cnp_finfo_min(CNP_TYPE dtype) {
    CNP_TYPE component = cnp_finfo_component_type(dtype, "cnp_finfo_min");
    switch (component) {
        case CNP_HALF: return -65504.0;
        case CNP_FLOAT: return -FLT_MAX;
        case CNP_DOUBLE: return -DBL_MAX;
        default: return cnp_typeconv_quiet_nan();
    }
}

static bool cnp_iinfo_bounds(
    CNP_TYPE dtype, int64_t *minimum, uint64_t *maximum) {
    switch (dtype) {
        case CNP_BYTE:
            *minimum = INT8_MIN;
            *maximum = INT8_MAX;
            return true;
        case CNP_UBYTE:
            *minimum = 0;
            *maximum = UINT8_MAX;
            return true;
        case CNP_SHORT:
            *minimum = INT16_MIN;
            *maximum = INT16_MAX;
            return true;
        case CNP_USHORT:
            *minimum = 0;
            *maximum = UINT16_MAX;
            return true;
        case CNP_INT:
            *minimum = INT32_MIN;
            *maximum = INT32_MAX;
            return true;
        case CNP_UINT:
            *minimum = 0;
            *maximum = UINT32_MAX;
            return true;
        case CNP_LONG:
        case CNP_LONGLONG:
            *minimum = INT64_MIN;
            *maximum = INT64_MAX;
            return true;
        case CNP_ULONG:
        case CNP_ULONGLONG:
            *minimum = 0;
            *maximum = UINT64_MAX;
            return true;
        default:
            return false;
    }
}

/* =========================================================================
 * cnp_iinfo_v2 - Exact bounds for an integer dtype
 * ========================================================================= */
CNP_API CNP_STATUS CNP_CALL cnp_iinfo_v2(
    CNP_TYPE dtype, int64_t *minimum, uint64_t *maximum) {
    const char *function_name = "cnp_iinfo_v2";
    if (!minimum || !maximum) {
        if (minimum) *minimum = 0;
        if (maximum) *maximum = 0;
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "minimum and maximum outputs must not be NULL");
        return CNP_ERR_GENERIC;
    }
    *minimum = 0;
    *maximum = 0;
    if (!cnp_iinfo_bounds(dtype, minimum, maximum)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtype %d is not an integer dtype", (int)dtype);
        return CNP_ERR_TYPE;
    }
    return CNP_OK;
}

/* =========================================================================
 * Legacy signed-return iinfo accessors
 * ========================================================================= */
CNP_API int64_t CNP_CALL cnp_iinfo_max(CNP_TYPE dtype) {
    int64_t minimum = 0;
    uint64_t maximum = 0;
    CNP_STATUS status = cnp_iinfo_v2(dtype, &minimum, &maximum);
    if (status != CNP_OK) {
        cnp_relabel_error("cnp_iinfo_max");
        return 0;
    }
    if (maximum > INT64_MAX) {
        cnp_set_error(
            CNP_ERR_CONVERSION, "cnp_iinfo_max",
            "signed legacy return cannot represent uint64 maximum; "
            "use cnp_iinfo_v2");
        return -1;
    }
    return (int64_t)maximum;
}

CNP_API int64_t CNP_CALL cnp_iinfo_min(CNP_TYPE dtype) {
    int64_t minimum = 0;
    uint64_t maximum = 0;
    CNP_STATUS status = cnp_iinfo_v2(dtype, &minimum, &maximum);
    if (status != CNP_OK) {
        cnp_relabel_error("cnp_iinfo_min");
        return 0;
    }
    return minimum;
}

/* =========================================================================
 * cnp_clip_array - Clip array values to range (array version)
 * numpy.clip(a, a_min, a_max)
 * Already exists in math_ops as cnp_clip with double params
 * This version takes array bounds
 * ========================================================================= */
static bool typeconv_clip_value_less(
    const CnpCopyValue *left, const CnpCopyValue *right) {
    if (left->kind == CNP_COPY_VALUE_COMPLEX ||
            right->kind == CNP_COPY_VALUE_COMPLEX) {
        long double left_real = typeconv_copy_real(left);
        long double right_real = typeconv_copy_real(right);
        if (left_real < right_real) return true;
        if (right_real < left_real) return false;
        long double left_imaginary =
            left->kind == CNP_COPY_VALUE_COMPLEX
            ? left->imaginary : 0.0L;
        long double right_imaginary =
            right->kind == CNP_COPY_VALUE_COMPLEX
            ? right->imaginary : 0.0L;
        return left_imaginary < right_imaginary;
    }
    bool left_integer = left->kind == CNP_COPY_VALUE_SIGNED ||
        left->kind == CNP_COPY_VALUE_UNSIGNED;
    bool right_integer = right->kind == CNP_COPY_VALUE_SIGNED ||
        right->kind == CNP_COPY_VALUE_UNSIGNED;
    if (!left_integer || !right_integer)
        return typeconv_copy_real(left) < typeconv_copy_real(right);
    if (left->kind == CNP_COPY_VALUE_SIGNED &&
            right->kind == CNP_COPY_VALUE_SIGNED)
        return left->signed_value < right->signed_value;
    if (left->kind == CNP_COPY_VALUE_UNSIGNED &&
            right->kind == CNP_COPY_VALUE_UNSIGNED)
        return left->unsigned_value < right->unsigned_value;
    if (left->kind == CNP_COPY_VALUE_SIGNED) {
        if (left->signed_value < 0) return true;
        return (uint64_t)left->signed_value < right->unsigned_value;
    }
    if (right->signed_value < 0) return false;
    return left->unsigned_value < (uint64_t)right->signed_value;
}

static bool typeconv_clip_value_nan(const CnpCopyValue *value) {
    if (value->kind != CNP_COPY_VALUE_FLOATING &&
            value->kind != CNP_COPY_VALUE_COMPLEX)
        return false;
    return isnan(value->real) ||
        (value->kind == CNP_COPY_VALUE_COMPLEX &&
         isnan(value->imaginary));
}

static void typeconv_clip_apply_bounds(
    CnpCopyValue *selected_value,
    const CnpCopyValue *minimum_value,
    const CnpCopyValue *maximum_value) {
    if (minimum_value &&
            !typeconv_clip_value_nan(selected_value) &&
            (typeconv_clip_value_nan(minimum_value) ||
             !typeconv_clip_value_less(
                 minimum_value, selected_value)))
        *selected_value = *minimum_value;
    if (maximum_value &&
            !typeconv_clip_value_nan(selected_value) &&
            (typeconv_clip_value_nan(maximum_value) ||
             !typeconv_clip_value_less(
                 selected_value, maximum_value)))
        *selected_value = *maximum_value;
}

static bool typeconv_clip_same_shape(
    const CnpArray *left, const CnpArray *right) {
    if (left->ndim != right->ndim) return false;
    for (int axis = 0; axis < left->ndim; ++axis) {
        if (left->shape[axis] != right->shape[axis]) return false;
    }
    return true;
}

static bool typeconv_clip_matching_contiguous_order(
    const CnpArray *source, const CnpArray *result) {
    bool c_order = (source->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_C_CONTIGUOUS);
    bool f_order = (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        (result->flags & CNP_ARRAY_F_CONTIGUOUS);
    return c_order || f_order;
}

static bool typeconv_clip_double_nan_bits(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) ==
            UINT64_C(0x7ff0000000000000) &&
        (bits & UINT64_C(0x000fffffffffffff)) != 0;
}

static bool typeconv_clip_float_nan_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000) &&
        (bits & UINT32_C(0x007fffff)) != 0;
}

static bool typeconv_clip_contiguous_scalar_bounds(
    const CnpArray *source,
    const CnpArray *a_min,
    const CnpArray *a_max,
    CnpArray *result,
    CNP_STATUS *status) {
    if (!typeconv_clip_same_shape(source, result) ||
            !typeconv_clip_matching_contiguous_order(source, result) ||
            (a_min && a_min->size != 1) ||
            (a_max && a_max->size != 1))
        return false;

    CnpCopyValue minimum_value;
    CnpCopyValue maximum_value;
    CnpCopyValue *minimum_pointer = NULL;
    CnpCopyValue *maximum_pointer = NULL;
    if (a_min) {
        *status = typeconv_copy_read_value(
            a_min, a_min->offset, &minimum_value);
        if (*status != CNP_OK) return true;
        minimum_pointer = &minimum_value;
    }
    if (a_max) {
        *status = typeconv_copy_read_value(
            a_max, a_max->offset, &maximum_value);
        if (*status != CNP_OK) return true;
        maximum_pointer = &maximum_value;
    }

    if (source->dtype->type_num == CNP_DOUBLE &&
            result->dtype->type_num == CNP_DOUBLE) {
        const double *input = (const double*)(
            (const char*)source->data + source->offset);
        double *output = (double*)((char*)result->data + result->offset);
        double minimum = minimum_pointer
            ? (double)typeconv_copy_real(minimum_pointer) : 0.0;
        double maximum = maximum_pointer
            ? (double)typeconv_copy_real(maximum_pointer) : 0.0;
        bool simple_minimum = !minimum_pointer ||
            (!isnan(minimum) && minimum != 0.0);
        bool simple_maximum = !maximum_pointer ||
            (!isnan(maximum) && maximum != 0.0);
        if (simple_minimum && simple_maximum) {
            int64_t index = 0;
            __m128d minimum_vector = _mm_set1_pd(minimum);
            __m128d maximum_vector = _mm_set1_pd(maximum);
            for (; index + 2 <= result->size; index += 2) {
                __m128d value = _mm_loadu_pd(input + index);
                if (minimum_pointer) {
                    __m128d mask = _mm_cmplt_pd(value, minimum_vector);
                    value = _mm_or_pd(
                        _mm_and_pd(mask, minimum_vector),
                        _mm_andnot_pd(mask, value));
                }
                if (maximum_pointer) {
                    __m128d mask = _mm_cmplt_pd(maximum_vector, value);
                    value = _mm_or_pd(
                        _mm_and_pd(mask, maximum_vector),
                        _mm_andnot_pd(mask, value));
                }
                _mm_storeu_pd(output + index, value);
            }
            for (; index < result->size; ++index) {
                double value = input[index];
                if (!typeconv_clip_double_nan_bits(value)) {
                    if (minimum_pointer && value < minimum) value = minimum;
                    if (maximum_pointer && value > maximum) value = maximum;
                }
                output[index] = value;
            }
            *status = CNP_OK;
            return true;
        }
        for (int64_t index = 0; index < result->size; ++index) {
            double value = input[index];
            if (!isnan(value) && minimum_pointer &&
                    (isnan(minimum) || !(minimum < value)))
                value = minimum;
            if (!isnan(value) && maximum_pointer &&
                    (isnan(maximum) || !(value < maximum)))
                value = maximum;
            output[index] = value;
        }
        *status = CNP_OK;
        return true;
    }
    if (source->dtype->type_num == CNP_FLOAT &&
            result->dtype->type_num == CNP_FLOAT) {
        const float *input = (const float*)(
            (const char*)source->data + source->offset);
        float *output = (float*)((char*)result->data + result->offset);
        float minimum = minimum_pointer
            ? (float)typeconv_copy_real(minimum_pointer) : 0.0f;
        float maximum = maximum_pointer
            ? (float)typeconv_copy_real(maximum_pointer) : 0.0f;
        bool simple_minimum = !minimum_pointer ||
            (!isnan(minimum) && minimum != 0.0f);
        bool simple_maximum = !maximum_pointer ||
            (!isnan(maximum) && maximum != 0.0f);
        if (simple_minimum && simple_maximum) {
            int64_t index = 0;
            __m128 minimum_vector = _mm_set1_ps(minimum);
            __m128 maximum_vector = _mm_set1_ps(maximum);
            for (; index + 4 <= result->size; index += 4) {
                __m128 value = _mm_loadu_ps(input + index);
                if (minimum_pointer) {
                    __m128 mask = _mm_cmplt_ps(value, minimum_vector);
                    value = _mm_or_ps(
                        _mm_and_ps(mask, minimum_vector),
                        _mm_andnot_ps(mask, value));
                }
                if (maximum_pointer) {
                    __m128 mask = _mm_cmplt_ps(maximum_vector, value);
                    value = _mm_or_ps(
                        _mm_and_ps(mask, maximum_vector),
                        _mm_andnot_ps(mask, value));
                }
                _mm_storeu_ps(output + index, value);
            }
            for (; index < result->size; ++index) {
                float value = input[index];
                if (!typeconv_clip_float_nan_bits(value)) {
                    if (minimum_pointer && value < minimum) value = minimum;
                    if (maximum_pointer && value > maximum) value = maximum;
                }
                output[index] = value;
            }
            *status = CNP_OK;
            return true;
        }
        for (int64_t index = 0; index < result->size; ++index) {
            float value = input[index];
            if (!isnan(value) && minimum_pointer &&
                    (isnan(minimum) || !(minimum < value)))
                value = minimum;
            if (!isnan(value) && maximum_pointer &&
                    (isnan(maximum) || !(value < maximum)))
                value = maximum;
            output[index] = value;
        }
        *status = CNP_OK;
        return true;
    }

    for (int64_t index = 0; index < result->size; ++index) {
        CnpCopyValue selected_value;
        *status = typeconv_copy_read_value(
            source, source->offset + index * source->dtype->elsize,
            &selected_value);
        if (*status != CNP_OK) return true;
        typeconv_clip_apply_bounds(
            &selected_value, minimum_pointer, maximum_pointer);
        *status = typeconv_copy_write_value(
            result, result->offset + index * result->dtype->elsize,
            &selected_value);
        if (*status != CNP_OK) return true;
    }
    *status = CNP_OK;
    return true;
}

CNP_API CnpArray* CNP_CALL cnp_clip_array(const CnpArray *arr, const CnpArray *a_min, const CnpArray *a_max) {
    const char *function_name = "cnp_clip_array";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!a_min && !a_max) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "one of a_min or a_max is required");
        return NULL;
    }

    const CnpArray *operands[3] = {arr, NULL, NULL};
    CnpArray *inputs[3] = {(CnpArray*)arr, NULL, NULL};
    int operand_count = 1;
    int minimum_index = -1;
    int maximum_index = -1;
    if (a_min) {
        minimum_index = operand_count;
        operands[operand_count] = a_min;
        inputs[operand_count++] = (CnpArray*)a_min;
    }
    if (a_max) {
        maximum_index = operand_count;
        operands[operand_count] = a_max;
        inputs[operand_count++] = (CnpArray*)a_max;
    }

    CNP_TYPE result_type = cnp_result_type(operand_count, operands);
    if (result_type == CNP_NOTYPE) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    CnpArray *broadcasted[3] = {NULL, NULL, NULL};
    CNP_STATUS status = cnp_broadcast_arrays_v2(
        operand_count, inputs, broadcasted, operand_count);
    if (status != CNP_OK) {
        if (status == CNP_ERR_BROADCAST) {
            char detail[256];
            strncpy(detail, cnp_get_error_message(), sizeof(detail) - 1);
            detail[sizeof(detail) - 1] = '\0';
            cnp_set_error(
                status, function_name,
                "clip operands cannot broadcast to a common shape: %s",
                detail);
        } else {
            cnp_relabel_error(function_name);
        }
        return NULL;
    }

    CnpArray *result = cnp_array_new(
        broadcasted[0]->ndim, broadcasted[0]->shape,
        result_type, typeconv_result_order(operand_count, operands));
    if (!result) {
        typeconv_release_arrays(broadcasted, operand_count);
        cnp_relabel_error(function_name);
        return NULL;
    }

    if (typeconv_clip_contiguous_scalar_bounds(
            arr, a_min, a_max, result, &status)) {
        typeconv_release_arrays(broadcasted, operand_count);
        if (status != CNP_OK) {
            cnp_array_free(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        CnpCopyValue selected_value;
        int64_t source_offset = broadcasted[0]->offset +
            cnp_multi_to_offset(
                broadcasted[0]->ndim, coordinates,
                broadcasted[0]->strides);
        status = typeconv_copy_read_value(
            broadcasted[0], source_offset, &selected_value);
        if (status != CNP_OK) break;

        if (minimum_index >= 0) {
            CnpCopyValue minimum_value;
            CnpArray *minimum = broadcasted[minimum_index];
            int64_t minimum_offset = minimum->offset +
                cnp_multi_to_offset(
                    minimum->ndim, coordinates, minimum->strides);
            status = typeconv_copy_read_value(
                minimum, minimum_offset, &minimum_value);
            if (status != CNP_OK) break;
            typeconv_clip_apply_bounds(
                &selected_value, &minimum_value, NULL);
        }
        if (maximum_index >= 0) {
            CnpCopyValue maximum_value;
            CnpArray *maximum = broadcasted[maximum_index];
            int64_t maximum_offset = maximum->offset +
                cnp_multi_to_offset(
                    maximum->ndim, coordinates, maximum->strides);
            status = typeconv_copy_read_value(
                maximum, maximum_offset, &maximum_value);
            if (status != CNP_OK) break;
            typeconv_clip_apply_bounds(
                &selected_value, NULL, &maximum_value);
        }

        int64_t result_offset = result->offset +
            cnp_multi_to_offset(
                result->ndim, coordinates, result->strides);
        status = typeconv_copy_write_value(
            result, result_offset, &selected_value);
        if (status != CNP_OK) break;

        for (int dimension = result->ndim - 1;
             dimension >= 0; --dimension) {
            ++coordinates[dimension];
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }

    typeconv_release_arrays(broadcasted, operand_count);
    if (status != CNP_OK) {
        cnp_array_free(result);
        cnp_relabel_error(function_name);
        return NULL;
    }
    return result;
}

/* =========================================================================
 * cnp_around - Round to given number of decimals
 * numpy.around(a, decimals=0)
 * ========================================================================= */
static double typeconv_around_factor(int decimals) {
    uint32_t magnitude = decimals < 0
        ? (uint32_t)(-(int64_t)decimals)
        : (uint32_t)decimals;
    if (magnitude > 308u)
        return cnp_typeconv_positive_infinity();
    return pow(10.0, (double)magnitude);
}

static double typeconv_around_negative_nan(void) {
    uint64_t bits = UINT64_C(0xfff8000000000000);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float typeconv_around_negative_nan_float(void) {
    uint32_t bits = UINT32_C(0xffc00000);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool typeconv_around_double_special(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7fffffffffffffff)) == 0 ||
        (bits & UINT64_C(0x7ff0000000000000)) ==
            UINT64_C(0x7ff0000000000000);
}

static bool typeconv_around_float_special(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7fffffff)) == 0 ||
        (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000);
}

static double typeconv_around_round_even(double value) {
    uint64_t value_bits;
    memcpy(&value_bits, &value, sizeof(value_bits));
    uint64_t magnitude_bits =
        value_bits & UINT64_C(0x7fffffffffffffff);
    if (magnitude_bits == 0 ||
            (magnitude_bits & UINT64_C(0x7ff0000000000000)) ==
                UINT64_C(0x7ff0000000000000) ||
            magnitude_bits >= UINT64_C(0x4330000000000000))
        return value;

    double magnitude;
    memcpy(&magnitude, &magnitude_bits, sizeof(magnitude));
    double lower = floor(magnitude);
    double fraction = magnitude - lower;
    uint64_t lower_integer = (uint64_t)lower;
    if (fraction > 0.5 ||
            (fraction == 0.5 && (lower_integer & UINT64_C(1))))
        lower += 1.0;

    uint64_t result_bits;
    memcpy(&result_bits, &lower, sizeof(result_bits));
    result_bits |= value_bits & UINT64_C(0x8000000000000000);
    memcpy(&lower, &result_bits, sizeof(lower));
    return lower;
}

static float typeconv_around_multiply_float(float left, float right) {
    volatile float result = left * right;
    return result;
}

static float typeconv_around_divide_float(float left, float right) {
    volatile float result = left / right;
    return result;
}

static double typeconv_around_multiply_double(
    double left, double right) {
    volatile double result = left * right;
    return result;
}

static double typeconv_around_divide_double(double left, double right) {
    volatile double result = left / right;
    return result;
}

static double typeconv_around_scale_double(
    double value, double factor, bool multiply) {
    if (typeconv_around_double_special(value)) return value;
    return multiply
        ? typeconv_around_multiply_double(value, factor)
        : typeconv_around_divide_double(value, factor);
}

static float typeconv_around_scale_float(
    float value, double factor, bool multiply) {
    if (typeconv_around_float_special(value)) return value;
    if (factor <= FLT_MAX) {
        float float_factor = (float)factor;
        return multiply
            ? typeconv_around_multiply_float(value, float_factor)
            : typeconv_around_divide_float(value, float_factor);
    }
    double scaled = multiply
        ? typeconv_around_multiply_double((double)value, factor)
        : typeconv_around_divide_double((double)value, factor);
    return (float)scaled;
}

static uint16_t typeconv_around_scale_half(
    uint16_t value_bits, double factor, bool multiply) {
    uint16_t magnitude = value_bits & UINT16_C(0x7fff);
    if (magnitude == 0 || (magnitude & UINT16_C(0x7c00)) ==
            UINT16_C(0x7c00))
        return value_bits;

    double value = cnp_half_to_float(value_bits);
    double scaled;
    if (factor <= 65504.0) {
        float left = (float)value;
        float right = (float)cnp_half_to_float(
            cnp_float_to_half(factor));
        scaled = (double)(multiply
            ? typeconv_around_multiply_float(left, right)
            : typeconv_around_divide_float(left, right));
    } else if (factor <= FLT_MAX) {
        float left = (float)value;
        float right = (float)factor;
        scaled = (double)(multiply
            ? typeconv_around_multiply_float(left, right)
            : typeconv_around_divide_float(left, right));
    } else {
        scaled = multiply
            ? typeconv_around_multiply_double(value, factor)
            : typeconv_around_divide_double(value, factor);
    }
    return cnp_float_to_half(scaled);
}

static uint16_t typeconv_around_half_component(
    uint16_t value_bits, double factor, int decimals) {
    if (!isfinite(factor)) {
        if ((value_bits & UINT16_C(0x7c00)) == UINT16_C(0x7c00) &&
                (value_bits & UINT16_C(0x03ff)) != 0)
            return value_bits | UINT16_C(0x0200);
        return UINT16_C(0xfe00);
    }
    if (decimals == 0) {
        double rounded = typeconv_around_round_even(
            cnp_half_to_float(value_bits));
        return cnp_float_to_half(rounded);
    }

    bool first_multiply = decimals > 0;
    uint16_t stage = typeconv_around_scale_half(
        value_bits, factor, first_multiply);
    stage = cnp_float_to_half(typeconv_around_round_even(
        cnp_half_to_float(stage)));
    return typeconv_around_scale_half(
        stage, factor, !first_multiply);
}

static float typeconv_around_float_component(
    float value, double factor, int decimals) {
    if (!isfinite(factor)) {
        if (typeconv_clip_float_nan_bits(value)) return value;
        return typeconv_around_negative_nan_float();
    }
    if (decimals == 0)
        return (float)typeconv_around_round_even((double)value);

    bool first_multiply = decimals > 0;
    float stage = typeconv_around_scale_float(
        value, factor, first_multiply);
    stage = (float)typeconv_around_round_even((double)stage);
    return typeconv_around_scale_float(
        stage, factor, !first_multiply);
}

static double typeconv_around_double_component(
    double value, double factor, int decimals) {
    if (!isfinite(factor)) {
        if (typeconv_clip_double_nan_bits(value)) return value;
        return typeconv_around_negative_nan();
    }
    if (decimals == 0) return typeconv_around_round_even(value);

    bool first_multiply = decimals > 0;
    double stage = typeconv_around_scale_double(
        value, factor, first_multiply);
    stage = typeconv_around_round_even(stage);
    return typeconv_around_scale_double(
        stage, factor, !first_multiply);
}

static long double typeconv_around_long_double_component(
    long double value, double factor, int decimals) {
    if (sizeof(long double) == sizeof(double)) {
        double double_value;
        memcpy(&double_value, &value, sizeof(double_value));
        double_value = typeconv_around_double_component(
            double_value, factor, decimals);
        memcpy(&value, &double_value, sizeof(double_value));
        return value;
    }
    if (!isfinite(factor)) {
        if (isnan(value)) return value;
        return -(long double)NAN;
    }
    if (value == 0.0L || isnan(value) || isinf(value)) return value;
    long double long_factor = (long double)factor;
    if (decimals > 0)
        value *= long_factor;
    else if (decimals < 0)
        value /= long_factor;
    value = rintl(value);
    if (decimals > 0)
        value /= long_factor;
    else if (decimals < 0)
        value *= long_factor;
    return value;
}

static double typeconv_around_integer_value(
    const void *source, CNP_TYPE dtype) {
    switch (dtype) {
        case CNP_BYTE: return (double)*(const int8_t*)source;
        case CNP_UBYTE: return (double)*(const uint8_t*)source;
        case CNP_SHORT: return (double)*(const int16_t*)source;
        case CNP_USHORT: return (double)*(const uint16_t*)source;
        case CNP_INT: return (double)*(const int32_t*)source;
        case CNP_UINT: return (double)*(const uint32_t*)source;
        case CNP_LONG:
        case CNP_LONGLONG:
            return (double)*(const int64_t*)source;
        case CNP_ULONG:
        case CNP_ULONGLONG:
            return (double)*(const uint64_t*)source;
        default: return 0.0;
    }
}

static void typeconv_around_write_integer(
    void *destination, CNP_TYPE dtype, double value) {
    if (dtype == CNP_BYTE || dtype == CNP_SHORT) {
        int32_t converted = isfinite(value) &&
                value >= (double)INT32_MIN && value < 2147483648.0
            ? (int32_t)value : INT32_MIN;
        if (dtype == CNP_BYTE) {
            uint8_t raw = (uint8_t)(uint32_t)converted;
            memcpy(destination, &raw, sizeof(raw));
        } else {
            uint16_t raw = (uint16_t)(uint32_t)converted;
            memcpy(destination, &raw, sizeof(raw));
        }
        return;
    }
    if (dtype == CNP_UBYTE || dtype == CNP_USHORT) {
        uint64_t converted = isfinite(value) && value >= 0.0 &&
                value < 18446744073709551616.0
            ? (uint64_t)value : UINT64_C(0);
        if (dtype == CNP_UBYTE) {
            uint8_t raw = (uint8_t)converted;
            memcpy(destination, &raw, sizeof(raw));
        } else {
            uint16_t raw = (uint16_t)converted;
            memcpy(destination, &raw, sizeof(raw));
        }
        return;
    }
    if (dtype == CNP_INT) {
        int32_t converted = isfinite(value) &&
                value >= (double)INT32_MIN && value < 2147483648.0
            ? (int32_t)value : INT32_MIN;
        memcpy(destination, &converted, sizeof(converted));
        return;
    }
    if (dtype == CNP_UINT) {
        uint32_t converted = 0;
        if (isfinite(value) && value >= 0.0 &&
                value < 18446744073709551616.0)
            converted = (uint32_t)(uint64_t)value;
        memcpy(destination, &converted, sizeof(converted));
        return;
    }
    if (dtype == CNP_LONG || dtype == CNP_LONGLONG) {
        int64_t converted = isfinite(value) &&
                value >= -9223372036854775808.0 &&
                value < 9223372036854775808.0
            ? (int64_t)value : INT64_MIN;
        memcpy(destination, &converted, sizeof(converted));
        return;
    }
    uint64_t converted = isfinite(value) && value >= 0.0 &&
            value < 18446744073709551616.0
        ? (uint64_t)value : (UINT64_C(1) << 63);
    memcpy(destination, &converted, sizeof(converted));
}

static void typeconv_around_advance_coordinates(
    int ndim, const int64_t *shape, int64_t *coordinates) {
    for (int dimension = ndim - 1; dimension >= 0; --dimension) {
        ++coordinates[dimension];
        if (coordinates[dimension] < shape[dimension]) return;
        coordinates[dimension] = 0;
    }
}

static bool typeconv_around_contiguous(
    const CnpArray *source,
    CnpArray *result,
    double factor,
    int decimals) {
    if (!typeconv_clip_same_shape(source, result) ||
            !typeconv_clip_matching_contiguous_order(source, result))
        return false;

    const void *source_data =
        (const char*)source->data + source->offset;
    void *result_data = (char*)result->data + result->offset;
    CNP_TYPE source_type = source->dtype->type_num;
    if (source_type == CNP_BOOL) {
        const uint8_t *input = (const uint8_t*)source_data;
        uint16_t *output = (uint16_t*)result_data;
        for (int64_t index = 0; index < result->size; ++index)
            output[index] = input[index]
                ? UINT16_C(0x3c00) : UINT16_C(0);
        return true;
    }
    if (cnp_type_is_integer(source_type) && decimals >= 0) {
        if (result->size > 0)
            memcpy(
                result_data, source_data,
                (size_t)result->size * (size_t)result->dtype->elsize);
        return true;
    }
    if (source_type == CNP_HALF) {
        const uint16_t *input = (const uint16_t*)source_data;
        uint16_t *output = (uint16_t*)result_data;
        for (int64_t index = 0; index < result->size; ++index)
            output[index] = typeconv_around_half_component(
                input[index], factor, decimals);
        return true;
    }
    if (source_type == CNP_FLOAT) {
        const float *input = (const float*)source_data;
        float *output = (float*)result_data;
        for (int64_t index = 0; index < result->size; ++index)
            output[index] = typeconv_around_float_component(
                input[index], factor, decimals);
        return true;
    }
    if (source_type == CNP_DOUBLE) {
        const double *input = (const double*)source_data;
        double *output = (double*)result_data;
        for (int64_t index = 0; index < result->size; ++index)
            output[index] = typeconv_around_double_component(
                input[index], factor, decimals);
        return true;
    }
    if (source_type == CNP_LONGDOUBLE) {
        const long double *input = (const long double*)source_data;
        long double *output = (long double*)result_data;
        for (int64_t index = 0; index < result->size; ++index)
            output[index] = typeconv_around_long_double_component(
                input[index], factor, decimals);
        return true;
    }
    if (source_type == CNP_CFLOAT) {
        const cnp_cfloat *input = (const cnp_cfloat*)source_data;
        cnp_cfloat *output = (cnp_cfloat*)result_data;
        for (int64_t index = 0; index < result->size; ++index) {
            output[index].real = typeconv_around_float_component(
                input[index].real, factor, decimals);
            output[index].imag = typeconv_around_float_component(
                input[index].imag, factor, decimals);
        }
        return true;
    }
    if (source_type == CNP_CDOUBLE) {
        const cnp_cdouble *input = (const cnp_cdouble*)source_data;
        cnp_cdouble *output = (cnp_cdouble*)result_data;
        for (int64_t index = 0; index < result->size; ++index) {
            output[index].real = typeconv_around_double_component(
                input[index].real, factor, decimals);
            output[index].imag = typeconv_around_double_component(
                input[index].imag, factor, decimals);
        }
        return true;
    }
    if (source_type == CNP_CLONGDOUBLE) {
        const cnp_clongdouble *input =
            (const cnp_clongdouble*)source_data;
        cnp_clongdouble *output = (cnp_clongdouble*)result_data;
        for (int64_t index = 0; index < result->size; ++index) {
            output[index].real = typeconv_around_long_double_component(
                input[index].real, factor, decimals);
            output[index].imag = typeconv_around_long_double_component(
                input[index].imag, factor, decimals);
        }
        return true;
    }
    return false;
}

static CNP_TYPE typeconv_real_floating_result_type(
    CNP_TYPE source_type) {
    switch (source_type) {
        case CNP_BOOL:
        case CNP_BYTE:
        case CNP_UBYTE:
            return CNP_HALF;
        case CNP_SHORT:
        case CNP_USHORT:
            return CNP_FLOAT;
        case CNP_INT:
        case CNP_UINT:
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
            return CNP_DOUBLE;
        case CNP_HALF:
        case CNP_FLOAT:
        case CNP_DOUBLE:
        case CNP_LONGDOUBLE:
            return source_type;
        default:
            return CNP_NOTYPE;
    }
}

static CNP_TYPE typeconv_absolute_result_type(
    CNP_TYPE source_type, bool floating_result) {
    if (floating_result)
        return typeconv_real_floating_result_type(source_type);
    if (source_type == CNP_CFLOAT) return CNP_FLOAT;
    if (source_type == CNP_CDOUBLE) return CNP_DOUBLE;
    if (source_type == CNP_CLONGDOUBLE) return CNP_LONGDOUBLE;
    if (source_type == CNP_BOOL || cnp_type_is_integer(source_type) ||
            cnp_type_is_float(source_type))
        return source_type;
    return CNP_NOTYPE;
}

static void typeconv_absolute_signed_integer(
    const void *source, CNP_TYPE source_type, void *destination) {
    if (source_type == CNP_BYTE) {
        int8_t value;
        memcpy(&value, source, sizeof(value));
        uint8_t magnitude = value < 0
            ? (uint8_t)(0 - (uint8_t)value) : (uint8_t)value;
        memcpy(destination, &magnitude, sizeof(magnitude));
        return;
    }
    if (source_type == CNP_SHORT) {
        int16_t value;
        memcpy(&value, source, sizeof(value));
        uint16_t magnitude = value < 0
            ? (uint16_t)(0 - (uint16_t)value) : (uint16_t)value;
        memcpy(destination, &magnitude, sizeof(magnitude));
        return;
    }
    if (source_type == CNP_INT) {
        int32_t value;
        memcpy(&value, source, sizeof(value));
        uint32_t magnitude = value < 0
            ? UINT32_C(0) - (uint32_t)value : (uint32_t)value;
        memcpy(destination, &magnitude, sizeof(magnitude));
        return;
    }

    int64_t value;
    memcpy(&value, source, sizeof(value));
    uint64_t magnitude = value < 0
        ? UINT64_C(0) - (uint64_t)value : (uint64_t)value;
    memcpy(destination, &magnitude, sizeof(magnitude));
}

static double typeconv_absolute_complex_magnitude_double(
    double real, double imaginary) {
    double real_magnitude = fabs(real);
    double imaginary_magnitude = fabs(imaginary);
    if (isinf(real_magnitude) || isinf(imaginary_magnitude))
        return cnp_typeconv_positive_infinity();
    if (isnan(real_magnitude) || isnan(imaginary_magnitude))
        return NAN;
    if (real_magnitude < imaginary_magnitude) {
        double temporary = real_magnitude;
        real_magnitude = imaginary_magnitude;
        imaginary_magnitude = temporary;
    }
    if (real_magnitude == 0.0) return 0.0;
    volatile double ratio = imaginary_magnitude / real_magnitude;
    volatile double ratio_squared = ratio * ratio;
    volatile double scaled = sqrt(1.0 + ratio_squared);
    return real_magnitude * scaled;
}

static long double typeconv_absolute_complex_magnitude_long_double(
    long double real, long double imaginary) {
    long double real_magnitude = fabsl(real);
    long double imaginary_magnitude = fabsl(imaginary);
    if (isinf(real_magnitude) || isinf(imaginary_magnitude))
        return (long double)cnp_typeconv_positive_infinity();
    if (isnan(real_magnitude) || isnan(imaginary_magnitude))
        return (long double)NAN;
    if (real_magnitude < imaginary_magnitude) {
        long double temporary = real_magnitude;
        real_magnitude = imaginary_magnitude;
        imaginary_magnitude = temporary;
    }
    if (real_magnitude == 0.0L) return 0.0L;
    volatile long double ratio = imaginary_magnitude / real_magnitude;
    volatile long double ratio_squared = ratio * ratio;
    volatile long double scaled = sqrtl(1.0L + ratio_squared);
    return real_magnitude * scaled;
}

static void typeconv_absolute_apply(
    const void *source,
    CNP_TYPE source_type,
    int source_itemsize,
    void *destination,
    CNP_TYPE result_type,
    bool floating_result) {
    if (floating_result &&
            (source_type == CNP_BOOL || cnp_type_is_integer(source_type))) {
        double value = source_type == CNP_BOOL
            ? (double)*(const uint8_t*)source
            : typeconv_around_integer_value(source, source_type);
        value = fabs(value);
        if (result_type == CNP_HALF) {
            uint16_t converted = cnp_float_to_half(value);
            memcpy(destination, &converted, sizeof(converted));
        } else if (result_type == CNP_FLOAT) {
            float converted = (float)value;
            memcpy(destination, &converted, sizeof(converted));
        } else {
            memcpy(destination, &value, sizeof(value));
        }
        return;
    }
    if (source_type == CNP_BOOL || cnp_type_is_unsigned(source_type)) {
        memcpy(destination, source, (size_t)source_itemsize);
        return;
    }
    if (cnp_type_is_integer(source_type)) {
        typeconv_absolute_signed_integer(
            source, source_type, destination);
        return;
    }
    if (source_type == CNP_HALF) {
        uint16_t bits;
        memcpy(&bits, source, sizeof(bits));
        bits &= UINT16_C(0x7fff);
        memcpy(destination, &bits, sizeof(bits));
        return;
    }
    if (source_type == CNP_FLOAT) {
        uint32_t bits;
        memcpy(&bits, source, sizeof(bits));
        bits &= UINT32_C(0x7fffffff);
        memcpy(destination, &bits, sizeof(bits));
        return;
    }
    if (source_type == CNP_DOUBLE) {
        uint64_t bits;
        memcpy(&bits, source, sizeof(bits));
        bits &= UINT64_C(0x7fffffffffffffff);
        memcpy(destination, &bits, sizeof(bits));
        return;
    }
    if (source_type == CNP_LONGDOUBLE) {
        long double value;
        memcpy(&value, source, sizeof(value));
        value = fabsl(value);
        memcpy(destination, &value, sizeof(value));
        return;
    }
    if (source_type == CNP_CFLOAT) {
        cnp_cfloat value;
        memcpy(&value, source, sizeof(value));
        float magnitude = (float)typeconv_absolute_complex_magnitude_double(
            (double)value.real, (double)value.imag);
        memcpy(destination, &magnitude, sizeof(magnitude));
        return;
    }
    if (source_type == CNP_CDOUBLE) {
        cnp_cdouble value;
        memcpy(&value, source, sizeof(value));
        double magnitude = typeconv_absolute_complex_magnitude_double(
            value.real, value.imag);
        memcpy(destination, &magnitude, sizeof(magnitude));
        return;
    }

    cnp_clongdouble value;
    memcpy(&value, source, sizeof(value));
    long double magnitude = typeconv_absolute_complex_magnitude_long_double(
        value.real, value.imag);
    memcpy(destination, &magnitude, sizeof(magnitude));
}

CnpArray* cnp_unary_op_absolute(
    const CnpArray *source,
    bool floating_result,
    const char *function_name) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }

    CNP_TYPE source_type = source->dtype->type_num;
    CNP_TYPE result_type = typeconv_absolute_result_type(
        source_type, floating_result);
    if (result_type == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support %s",
            source->dtype->name,
            floating_result ? "fabs" : "absolute");
        return NULL;
    }

    const CnpArray *inputs[1] = {source};
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_type,
        typeconv_result_order(1, inputs));
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    if (typeconv_clip_matching_contiguous_order(source, result)) {
        if (source_type == CNP_DOUBLE && result_type == CNP_DOUBLE) {
            cnp_simd_absolute(
                (const double*)((const char*)source->data + source->offset),
                (double*)((char*)result->data + result->offset),
                result->size);
            return result;
        }
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        int source_itemsize = source->dtype->elsize;
        int result_itemsize = result->dtype->elsize;
        for (int64_t index = 0; index < result->size; ++index) {
            typeconv_absolute_apply(
                source_data + index * source_itemsize,
                source_type,
                source_itemsize,
                result_data + index * result_itemsize,
                result_type,
                floating_result);
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        typeconv_absolute_apply(
            (const char*)source->data + source_offset,
            source_type,
            source->dtype->elsize,
            (char*)result->data + result_offset,
            result_type,
            floating_result);
        typeconv_around_advance_coordinates(
            result->ndim, result->shape, coordinates);
    }
    return result;
}

static CNP_TYPE typeconv_unary_rounding_result_type(
    CNP_TYPE source_type) {
    CNP_TYPE real_result =
        typeconv_real_floating_result_type(source_type);
    if (real_result != CNP_NOTYPE) return real_result;
    return cnp_type_is_complex(source_type)
        ? source_type : CNP_NOTYPE;
}

static double typeconv_unary_rounding_double(
    double value, CnpUnaryRoundingMode mode) {
    if (mode == CNP_UNARY_ROUND_RINT)
        return typeconv_around_round_even(value);
    if (mode == CNP_UNARY_ROUND_TRUNCATE) return trunc(value);
    if (mode == CNP_UNARY_ROUND_FLOOR) return floor(value);
    double rounded = ceil(value);
    return rounded == 0.0 && signbit(value)
        ? -0.0 : rounded;
}

static long double typeconv_unary_rounding_long_double(
    long double value, CnpUnaryRoundingMode mode) {
    if (mode == CNP_UNARY_ROUND_RINT)
        return typeconv_around_long_double_component(value, 1.0, 0);
    if (mode == CNP_UNARY_ROUND_TRUNCATE) return truncl(value);
    if (mode == CNP_UNARY_ROUND_FLOOR) return floorl(value);
    long double rounded = ceill(value);
    return rounded == 0.0L && signbit(value)
        ? -0.0L : rounded;
}

static void typeconv_unary_rounding_write_promoted_integer(
    void *destination, CNP_TYPE result_type, double value) {
    if (result_type == CNP_HALF) {
        uint16_t converted = cnp_float_to_half(value);
        memcpy(destination, &converted, sizeof(converted));
    } else if (result_type == CNP_FLOAT) {
        float converted = (float)value;
        memcpy(destination, &converted, sizeof(converted));
    } else {
        memcpy(destination, &value, sizeof(value));
    }
}

static void typeconv_unary_rounding_apply(
    const void *source,
    CNP_TYPE source_type,
    void *destination,
    CNP_TYPE result_type,
    CnpUnaryRoundingMode mode) {
    if (source_type == CNP_BOOL || cnp_type_is_integer(source_type)) {
        double value = source_type == CNP_BOOL
            ? (double)*(const uint8_t*)source
            : typeconv_around_integer_value(source, source_type);
        typeconv_unary_rounding_write_promoted_integer(
            destination, result_type, value);
        return;
    }

    if (source_type == CNP_HALF) {
        uint16_t bits;
        memcpy(&bits, source, sizeof(bits));
        double value = cnp_half_to_float(bits);
        bits = cnp_float_to_half(
            typeconv_unary_rounding_double(value, mode));
        memcpy(destination, &bits, sizeof(bits));
        return;
    }
    if (source_type == CNP_FLOAT) {
        float value;
        memcpy(&value, source, sizeof(value));
        value = mode == CNP_UNARY_ROUND_RINT
            ? typeconv_around_float_component(value, 1.0, 0)
            : (float)typeconv_unary_rounding_double((double)value, mode);
        memcpy(destination, &value, sizeof(value));
        return;
    }
    if (source_type == CNP_DOUBLE) {
        double value;
        memcpy(&value, source, sizeof(value));
        value = typeconv_unary_rounding_double(value, mode);
        memcpy(destination, &value, sizeof(value));
        return;
    }
    if (source_type == CNP_LONGDOUBLE) {
        long double value;
        memcpy(&value, source, sizeof(value));
        value = typeconv_unary_rounding_long_double(value, mode);
        memcpy(destination, &value, sizeof(value));
        return;
    }
    if (source_type == CNP_CFLOAT) {
        cnp_cfloat value;
        memcpy(&value, source, sizeof(value));
        value.real = typeconv_around_float_component(value.real, 1.0, 0);
        value.imag = typeconv_around_float_component(value.imag, 1.0, 0);
        memcpy(destination, &value, sizeof(value));
        return;
    }
    if (source_type == CNP_CDOUBLE) {
        cnp_cdouble value;
        memcpy(&value, source, sizeof(value));
        value.real = typeconv_around_double_component(value.real, 1.0, 0);
        value.imag = typeconv_around_double_component(value.imag, 1.0, 0);
        memcpy(destination, &value, sizeof(value));
        return;
    }

    cnp_clongdouble value;
    memcpy(&value, source, sizeof(value));
    value.real = typeconv_around_long_double_component(value.real, 1.0, 0);
    value.imag = typeconv_around_long_double_component(value.imag, 1.0, 0);
    memcpy(destination, &value, sizeof(value));
}

static void typeconv_unary_rounding_contiguous_double(
    const CnpArray *source,
    CnpArray *result,
    CnpUnaryRoundingMode mode) {
    const double *input = (const double*)(
        (const char*)source->data + source->offset);
    double *output = (double*)((char*)result->data + result->offset);
    int64_t index = 0;

    if (mode == CNP_UNARY_ROUND_RINT) {
        for (; index + 3 < result->size; index += 4) {
            output[index] = typeconv_around_round_even(input[index]);
            output[index + 1] = typeconv_around_round_even(input[index + 1]);
            output[index + 2] = typeconv_around_round_even(input[index + 2]);
            output[index + 3] = typeconv_around_round_even(input[index + 3]);
        }
        for (; index < result->size; ++index)
            output[index] = typeconv_around_round_even(input[index]);
        return;
    }
    if (mode == CNP_UNARY_ROUND_TRUNCATE) {
        for (; index + 3 < result->size; index += 4) {
            output[index] = trunc(input[index]);
            output[index + 1] = trunc(input[index + 1]);
            output[index + 2] = trunc(input[index + 2]);
            output[index + 3] = trunc(input[index + 3]);
        }
        for (; index < result->size; ++index)
            output[index] = trunc(input[index]);
        return;
    }
    if (mode == CNP_UNARY_ROUND_FLOOR) {
        cnp_simd_floor(input, output, result->size);
        return;
    }

    for (; index + 3 < result->size; index += 4) {
        for (int lane = 0; lane < 4; ++lane) {
            double value = input[index + lane];
            double rounded = ceil(value);
            output[index + lane] = rounded == 0.0 && signbit(value)
                ? -0.0 : rounded;
        }
    }
    for (; index < result->size; ++index) {
        double value = input[index];
        double rounded = ceil(value);
        output[index] = rounded == 0.0 && signbit(value)
            ? -0.0 : rounded;
    }
}

CnpArray* cnp_unary_op_rounding(
    const CnpArray *source,
    CnpUnaryRoundingMode mode,
    const char *function_name) {
    if (mode < CNP_UNARY_ROUND_RINT ||
            mode > CNP_UNARY_ROUND_CEIL) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "invalid unary rounding mode %d", (int)mode);
        return NULL;
    }
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }

    CNP_TYPE source_type = source->dtype->type_num;
    CNP_TYPE result_type = typeconv_unary_rounding_result_type(source_type);
    if (result_type == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support this rounding operation",
            source->dtype->name);
        return NULL;
    }
    if (cnp_type_is_complex(source_type) &&
            mode != CNP_UNARY_ROUND_RINT) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support this rounding operation",
            source->dtype->name);
        return NULL;
    }

    const CnpArray *inputs[1] = {source};
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_type,
        typeconv_result_order(1, inputs));
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    if (typeconv_clip_matching_contiguous_order(source, result)) {
        if (source_type == CNP_DOUBLE && result_type == CNP_DOUBLE) {
            typeconv_unary_rounding_contiguous_double(
                source, result, mode);
            return result;
        }
        const char *source_data =
            (const char*)source->data + source->offset;
        char *result_data = (char*)result->data + result->offset;
        int source_itemsize = source->dtype->elsize;
        int result_itemsize = result->dtype->elsize;
        for (int64_t index = 0; index < result->size; ++index) {
            typeconv_unary_rounding_apply(
                source_data + index * source_itemsize,
                source_type,
                result_data + index * result_itemsize,
                result_type,
                mode);
        }
        return result;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, coordinates, source->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        typeconv_unary_rounding_apply(
            (const char*)source->data + source_offset,
            source_type,
            (char*)result->data + result_offset,
            result_type,
            mode);
        typeconv_around_advance_coordinates(
            result->ndim, result->shape, coordinates);
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_around(const CnpArray *arr, int decimals) {
    const char *function_name = "cnp_around";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }

    CNP_TYPE source_type = arr->dtype->type_num;
    bool supported = source_type == CNP_BOOL ||
        cnp_type_is_integer(source_type) ||
        cnp_type_is_float(source_type) ||
        cnp_type_is_complex(source_type);
    if (!supported) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source dtype %s does not support around",
            arr->dtype->name);
        return NULL;
    }
    if (source_type == CNP_BOOL && decimals != 0) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "bool around requires decimals equal to zero");
        return NULL;
    }

    CNP_TYPE result_type = source_type == CNP_BOOL
        ? CNP_HALF : source_type;
    const CnpArray *inputs[1] = {arr};
    CnpArray *result = cnp_array_new(
        arr->ndim, arr->shape, result_type,
        typeconv_result_order(1, inputs));
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    double factor = typeconv_around_factor(decimals);
    if (typeconv_around_contiguous(
            arr, result, factor, decimals))
        return result;

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t source_offset = arr->offset + cnp_multi_to_offset(
            arr->ndim, coordinates, arr->strides);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coordinates, result->strides);
        const void *source = (const char*)arr->data + source_offset;
        void *destination = (char*)result->data + result_offset;

        if (source_type == CNP_BOOL) {
            uint16_t half = *(const uint8_t*)source
                ? UINT16_C(0x3c00) : UINT16_C(0);
            memcpy(destination, &half, sizeof(half));
        } else if (cnp_type_is_integer(source_type)) {
            if (decimals >= 0) {
                memcpy(destination, source, arr->dtype->elsize);
            } else {
                double value = typeconv_around_integer_value(
                    source, source_type);
                double rounded;
                if (isfinite(factor)) {
                    rounded = typeconv_around_divide_double(
                        value, factor);
                    rounded = typeconv_around_round_even(rounded);
                    rounded = typeconv_around_multiply_double(
                        rounded, factor);
                } else {
                    rounded = typeconv_around_negative_nan();
                }
                typeconv_around_write_integer(
                    destination, source_type, rounded);
            }
        } else if (source_type == CNP_HALF) {
            uint16_t value;
            memcpy(&value, source, sizeof(value));
            value = typeconv_around_half_component(
                value, factor, decimals);
            memcpy(destination, &value, sizeof(value));
        } else if (source_type == CNP_FLOAT) {
            float value;
            memcpy(&value, source, sizeof(value));
            value = typeconv_around_float_component(
                value, factor, decimals);
            memcpy(destination, &value, sizeof(value));
        } else if (source_type == CNP_DOUBLE) {
            double value;
            memcpy(&value, source, sizeof(value));
            value = typeconv_around_double_component(
                value, factor, decimals);
            memcpy(destination, &value, sizeof(value));
        } else if (source_type == CNP_LONGDOUBLE) {
            long double value;
            memcpy(&value, source, sizeof(value));
            value = typeconv_around_long_double_component(
                value, factor, decimals);
            memcpy(destination, &value, sizeof(value));
        } else if (source_type == CNP_CFLOAT) {
            cnp_cfloat value;
            memcpy(&value, source, sizeof(value));
            value.real = typeconv_around_float_component(
                value.real, factor, decimals);
            value.imag = typeconv_around_float_component(
                value.imag, factor, decimals);
            memcpy(destination, &value, sizeof(value));
        } else if (source_type == CNP_CDOUBLE) {
            cnp_cdouble value;
            memcpy(&value, source, sizeof(value));
            value.real = typeconv_around_double_component(
                value.real, factor, decimals);
            value.imag = typeconv_around_double_component(
                value.imag, factor, decimals);
            memcpy(destination, &value, sizeof(value));
        } else {
            cnp_clongdouble value;
            memcpy(&value, source, sizeof(value));
            value.real = typeconv_around_long_double_component(
                value.real, factor, decimals);
            value.imag = typeconv_around_long_double_component(
                value.imag, factor, decimals);
            memcpy(destination, &value, sizeof(value));
        }

        typeconv_around_advance_coordinates(
            result->ndim, result->shape, coordinates);
    }
    return result;
}

/* cnp_rint already in math_ops.c */
/* cnp_fix already in math_ops.c */
