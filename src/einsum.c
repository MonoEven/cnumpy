/**
 * Parsed NumPy-style einsum execution for real and complex numeric arrays.
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <ctype.h>
#include <limits.h>

#define CNP_EINSUM_NAMED_LABELS 52
#define CNP_EINSUM_ELLIPSIS_BASE CNP_EINSUM_NAMED_LABELS
#define CNP_EINSUM_LABEL_CAPACITY \
    (CNP_EINSUM_NAMED_LABELS + CNP_MAXDIMS)

typedef struct {
    int start;
    int end;
    bool has_ellipsis;
    int named_count;
    int ellipsis_ndim;
} CnpEinsumTerm;

typedef struct {
    const CnpArray *array;
    int axis_labels[CNP_MAXDIMS];
} CnpEinsumOperandPlan;

typedef struct {
    int narrays;
    CnpEinsumOperandPlan operands[CNP_MAXARGS];
    int max_ellipsis_ndim;
    bool label_present[CNP_EINSUM_LABEL_CAPACITY];
    int label_occurrences[CNP_EINSUM_LABEL_CAPACITY];
    int64_t label_lengths[CNP_EINSUM_LABEL_CAPACITY];
    int output_ndim;
    int output_labels[CNP_MAXDIMS];
    int64_t output_shape[CNP_MAXDIMS];
    int contraction_count;
    int contraction_labels[CNP_EINSUM_LABEL_CAPACITY];
    CNP_TYPE output_type;
} CnpEinsumPlan;

typedef union {
    uint64_t integer;
    float float32;
    double float64;
    cnp_cfloat complex64;
    cnp_cdouble complex128;
} CnpEinsumValue;

static int einsum_label_id(char value) {
    if (value >= 'A' && value <= 'Z') return value - 'A';
    if (value >= 'a' && value <= 'z') return 26 + value - 'a';
    return -1;
}

static char einsum_label_character(int label) {
    if (label < 26) return (char)('A' + label);
    return (char)('a' + label - 26);
}

static bool einsum_is_ellipsis(
    const char *expression, int position, int end) {
    return position + 2 < end && expression[position] == '.' &&
        expression[position + 1] == '.' && expression[position + 2] == '.';
}

static bool einsum_supported_dtype(CNP_TYPE type) {
    return type == CNP_BOOL || cnp_type_is_integer(type) ||
        type == CNP_HALF || type == CNP_FLOAT || type == CNP_DOUBLE ||
        type == CNP_CFLOAT || type == CNP_CDOUBLE;
}

static CNP_TYPE einsum_signed_type(int bits) {
    if (bits <= 8) return CNP_BYTE;
    if (bits <= 16) return CNP_SHORT;
    if (bits <= 32) return CNP_INT;
    return CNP_LONGLONG;
}

static CNP_TYPE einsum_unsigned_type(int bits) {
    if (bits <= 8) return CNP_UBYTE;
    if (bits <= 16) return CNP_USHORT;
    if (bits <= 32) return CNP_UINT;
    return CNP_ULONGLONG;
}

static CNP_TYPE einsum_promote_integer_pair(CNP_TYPE left, CNP_TYPE right) {
    int left_bits = cnp_dtype_itemsize(left) * 8;
    int right_bits = cnp_dtype_itemsize(right) * 8;
    bool left_unsigned = cnp_type_is_unsigned(left);
    bool right_unsigned = cnp_type_is_unsigned(right);
    if (left_unsigned == right_unsigned) {
        int bits = left_bits > right_bits ? left_bits : right_bits;
        return left_unsigned
            ? einsum_unsigned_type(bits) : einsum_signed_type(bits);
    }

    int signed_bits = left_unsigned ? right_bits : left_bits;
    int unsigned_bits = left_unsigned ? left_bits : right_bits;
    if (signed_bits > unsigned_bits)
        return einsum_signed_type(signed_bits);
    if (unsigned_bits < 64)
        return einsum_signed_type(unsigned_bits * 2);
    return CNP_DOUBLE;
}

static bool einsum_float32_compatible(CNP_TYPE type) {
    if (type == CNP_BOOL || type == CNP_HALF ||
            type == CNP_FLOAT || type == CNP_CFLOAT)
        return true;
    return cnp_type_is_integer(type) && cnp_dtype_itemsize(type) <= 2;
}

static CNP_TYPE einsum_promote_pair(CNP_TYPE left, CNP_TYPE right) {
    if (left == right) return left;
    if (left == CNP_BOOL) return right;
    if (right == CNP_BOOL) return left;

    if (cnp_type_is_complex(left) || cnp_type_is_complex(right)) {
        if (left == CNP_CDOUBLE || right == CNP_CDOUBLE)
            return CNP_CDOUBLE;
        return einsum_float32_compatible(left) &&
            einsum_float32_compatible(right)
            ? CNP_CFLOAT : CNP_CDOUBLE;
    }
    if (cnp_type_is_float(left) || cnp_type_is_float(right)) {
        if (left == CNP_DOUBLE || right == CNP_DOUBLE)
            return CNP_DOUBLE;
        if (left == CNP_HALF || right == CNP_HALF) {
            CNP_TYPE other = left == CNP_HALF ? right : left;
            if (cnp_type_is_integer(other)) {
                int integer_bits = cnp_dtype_itemsize(other) * 8;
                if (integer_bits <= 8) return CNP_HALF;
                if (integer_bits <= 16) return CNP_FLOAT;
                return CNP_DOUBLE;
            }
        }
        return einsum_float32_compatible(left) &&
            einsum_float32_compatible(right)
            ? CNP_FLOAT : CNP_DOUBLE;
    }
    return einsum_promote_integer_pair(left, right);
}

static bool einsum_result_type(
    int narrays, const CnpArray *const *arrays, CNP_TYPE *result_type) {
    CNP_TYPE current = CNP_BOOL;
    for (int index = 0; index < narrays; ++index) {
        CNP_TYPE type = arrays[index]->dtype->type_num;
        if (!einsum_supported_dtype(type)) {
            cnp_set_error(CNP_ERR_TYPE, "cnp_einsum",
                          "operand %d has unsupported dtype %s",
                          index, arrays[index]->dtype->name);
            return false;
        }
        current = index == 0 ? type : einsum_promote_pair(current, type);
    }
    *result_type = current;
    return true;
}

static bool einsum_combine_length(
    CnpEinsumPlan *plan, int label, int64_t length) {
    if (!plan->label_present[label]) {
        plan->label_present[label] = true;
        plan->label_lengths[label] = length;
        return true;
    }
    int64_t current = plan->label_lengths[label];
    if (current == length || length == 1) return true;
    if (current == 1) {
        plan->label_lengths[label] = length;
        return true;
    }
    if (label < CNP_EINSUM_NAMED_LABELS) {
        cnp_set_error(CNP_ERR_BROADCAST, "cnp_einsum",
                      "label '%c' dimensions %lld and %lld cannot broadcast",
                      einsum_label_character(label),
                      (long long)current, (long long)length);
    } else {
        cnp_set_error(CNP_ERR_BROADCAST, "cnp_einsum",
                      "ellipsis dimensions %lld and %lld cannot broadcast",
                      (long long)current, (long long)length);
    }
    return false;
}

static bool einsum_scan_term(
    const char *expression, int start, int end, const CnpArray *array,
    int operand_index, CnpEinsumTerm *term) {
    memset(term, 0, sizeof(*term));
    term->start = start;
    term->end = end;
    for (int position = start; position < end;) {
        int label = einsum_label_id(expression[position]);
        if (label >= 0) {
            ++term->named_count;
            ++position;
            continue;
        }
        if (expression[position] == '.') {
            if (!einsum_is_ellipsis(expression, position, end)) {
                cnp_set_error(CNP_ERR_GENERIC, "cnp_einsum",
                              "operand %d has a malformed ellipsis",
                              operand_index);
                return false;
            }
            if (term->has_ellipsis) {
                cnp_set_error(CNP_ERR_GENERIC, "cnp_einsum",
                              "operand %d contains more than one ellipsis",
                              operand_index);
                return false;
            }
            term->has_ellipsis = true;
            position += 3;
            continue;
        }
        cnp_set_error(CNP_ERR_GENERIC, "cnp_einsum",
                      "invalid subscript character '%c' in operand %d",
                      expression[position], operand_index);
        return false;
    }

    if (!term->has_ellipsis && array->ndim != term->named_count) {
        cnp_set_error(CNP_ERR_SHAPE, "cnp_einsum",
                      "operand %d has %d dimensions but %d subscripts",
                      operand_index, array->ndim, term->named_count);
        return false;
    }
    if (term->has_ellipsis && array->ndim < term->named_count) {
        cnp_set_error(CNP_ERR_SHAPE, "cnp_einsum",
                      "operand %d has fewer dimensions than named subscripts",
                      operand_index);
        return false;
    }
    term->ellipsis_ndim = term->has_ellipsis
        ? array->ndim - term->named_count : 0;
    return true;
}

static bool einsum_assign_operand(
    const char *expression, const CnpEinsumTerm *term,
    int operand_index, CnpEinsumPlan *plan) {
    CnpEinsumOperandPlan *operand = &plan->operands[operand_index];
    const CnpArray *array = operand->array;
    int64_t repeated_lengths[CNP_EINSUM_NAMED_LABELS];
    bool repeated_seen[CNP_EINSUM_NAMED_LABELS];
    memset(repeated_seen, 0, sizeof(repeated_seen));

    int axis = 0;
    for (int position = term->start; position < term->end;) {
        int label = einsum_label_id(expression[position]);
        if (label >= 0) {
            int64_t length = array->shape[axis];
            if (repeated_seen[label] && repeated_lengths[label] != length) {
                cnp_set_error(CNP_ERR_SHAPE, "cnp_einsum",
                              "repeated label '%c' has dimensions %lld and %lld in operand %d",
                              expression[position],
                              (long long)repeated_lengths[label],
                              (long long)length, operand_index);
                return false;
            }
            repeated_seen[label] = true;
            repeated_lengths[label] = length;
            operand->axis_labels[axis++] = label;
            ++plan->label_occurrences[label];
            if (!einsum_combine_length(plan, label, length)) return false;
            ++position;
            continue;
        }

        int first_ellipsis = plan->max_ellipsis_ndim - term->ellipsis_ndim;
        for (int ellipsis_axis = 0;
             ellipsis_axis < term->ellipsis_ndim; ++ellipsis_axis) {
            int ellipsis_label = CNP_EINSUM_ELLIPSIS_BASE +
                first_ellipsis + ellipsis_axis;
            int64_t length = array->shape[axis];
            operand->axis_labels[axis++] = ellipsis_label;
            if (!einsum_combine_length(plan, ellipsis_label, length))
                return false;
        }
        position += 3;
    }
    return true;
}

static bool einsum_append_output_label(
    CnpEinsumPlan *plan, bool *output_seen, int label) {
    if (output_seen[label]) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_einsum",
                      "output contains repeated label '%c'",
                      label < CNP_EINSUM_NAMED_LABELS
                          ? einsum_label_character(label) : '.');
        return false;
    }
    if (plan->output_ndim >= CNP_MAXDIMS) {
        cnp_set_error(CNP_ERR_SHAPE, "cnp_einsum",
                      "output exceeds CNP_MAXDIMS");
        return false;
    }
    output_seen[label] = true;
    plan->output_labels[plan->output_ndim++] = label;
    return true;
}

static bool einsum_parse_explicit_output(
    const char *expression, int start, int end, CnpEinsumPlan *plan,
    bool *output_seen, bool *has_ellipsis) {
    *has_ellipsis = false;
    for (int position = start; position < end;) {
        int label = einsum_label_id(expression[position]);
        if (label >= 0) {
            if (!plan->label_present[label]) {
                cnp_set_error(CNP_ERR_GENERIC, "cnp_einsum",
                              "output label '%c' does not appear in any operand",
                              expression[position]);
                return false;
            }
            if (!einsum_append_output_label(plan, output_seen, label))
                return false;
            ++position;
            continue;
        }
        if (expression[position] == '.') {
            if (!einsum_is_ellipsis(expression, position, end)) {
                cnp_set_error(CNP_ERR_GENERIC, "cnp_einsum",
                              "output has a malformed ellipsis");
                return false;
            }
            if (*has_ellipsis) {
                cnp_set_error(CNP_ERR_GENERIC, "cnp_einsum",
                              "output contains more than one ellipsis");
                return false;
            }
            *has_ellipsis = true;
            for (int ellipsis_axis = 0;
                 ellipsis_axis < plan->max_ellipsis_ndim;
                 ++ellipsis_axis) {
                if (!einsum_append_output_label(
                        plan, output_seen,
                        CNP_EINSUM_ELLIPSIS_BASE + ellipsis_axis))
                    return false;
            }
            position += 3;
            continue;
        }
        cnp_set_error(CNP_ERR_GENERIC, "cnp_einsum",
                      "invalid output subscript character '%c'",
                      expression[position]);
        return false;
    }
    return true;
}

static bool einsum_finalize_labels(
    CnpEinsumPlan *plan, bool explicit_output,
    const char *expression, int output_start, int output_end) {
    bool output_seen[CNP_EINSUM_LABEL_CAPACITY];
    memset(output_seen, 0, sizeof(output_seen));
    if (explicit_output) {
        bool has_output_ellipsis;
        if (!einsum_parse_explicit_output(
                expression, output_start, output_end,
                plan, output_seen, &has_output_ellipsis))
            return false;
        if (plan->max_ellipsis_ndim > 0 && !has_output_ellipsis) {
            cnp_set_error(CNP_ERR_GENERIC, "cnp_einsum",
                          "explicit output must contain an ellipsis when operands have extra dimensions");
            return false;
        }
    } else {
        for (int ellipsis_axis = 0;
             ellipsis_axis < plan->max_ellipsis_ndim; ++ellipsis_axis) {
            if (!einsum_append_output_label(
                    plan, output_seen,
                    CNP_EINSUM_ELLIPSIS_BASE + ellipsis_axis))
                return false;
        }
        for (int label = 0; label < CNP_EINSUM_NAMED_LABELS; ++label) {
            if (plan->label_occurrences[label] == 1 &&
                    !einsum_append_output_label(
                        plan, output_seen, label))
                return false;
        }
    }

    for (int dimension = 0; dimension < plan->output_ndim; ++dimension)
        plan->output_shape[dimension] =
            plan->label_lengths[plan->output_labels[dimension]];
    if (plan->output_ndim == 0) plan->output_shape[0] = 1;

    for (int ellipsis_axis = 0;
         ellipsis_axis < plan->max_ellipsis_ndim; ++ellipsis_axis) {
        int label = CNP_EINSUM_ELLIPSIS_BASE + ellipsis_axis;
        if (plan->label_present[label] && !output_seen[label])
            plan->contraction_labels[plan->contraction_count++] = label;
    }
    for (int label = 0; label < CNP_EINSUM_NAMED_LABELS; ++label) {
        if (plan->label_present[label] && !output_seen[label])
            plan->contraction_labels[plan->contraction_count++] = label;
    }
    return true;
}

static bool einsum_parse(
    const char *expression, int expression_length,
    int narrays, const CnpArray *const *arrays, CnpEinsumPlan *plan) {
    memset(plan, 0, sizeof(*plan));
    plan->narrays = narrays;
    for (int index = 0; index < narrays; ++index)
        plan->operands[index].array = arrays[index];

    int arrow = -1;
    for (int position = 0; position + 1 < expression_length; ++position) {
        if (expression[position] == '-' && expression[position + 1] == '>') {
            if (arrow >= 0) {
                cnp_set_error(CNP_ERR_GENERIC, "cnp_einsum",
                              "subscripts contain more than one output arrow");
                return false;
            }
            arrow = position;
            ++position;
        }
    }
    int input_end = arrow >= 0 ? arrow : expression_length;
    int term_count = 1;
    for (int position = 0; position < input_end; ++position)
        if (expression[position] == ',') ++term_count;
    if (term_count != narrays) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_einsum",
                      "subscripts describe %d operands but %d were provided",
                      term_count, narrays);
        return false;
    }

    CnpEinsumTerm terms[CNP_MAXARGS];
    int term_index = 0;
    int term_start = 0;
    for (int position = 0; position <= input_end; ++position) {
        if (position != input_end && expression[position] != ',') continue;
        if (!einsum_scan_term(
                expression, term_start, position, arrays[term_index],
                term_index, &terms[term_index]))
            return false;
        if (terms[term_index].ellipsis_ndim > plan->max_ellipsis_ndim)
            plan->max_ellipsis_ndim = terms[term_index].ellipsis_ndim;
        ++term_index;
        term_start = position + 1;
    }

    for (int index = 0; index < narrays; ++index) {
        if (!einsum_assign_operand(expression, &terms[index], index, plan))
            return false;
    }
    if (!einsum_finalize_labels(
            plan, arrow >= 0, expression,
            arrow >= 0 ? arrow + 2 : expression_length,
            expression_length))
        return false;
    return einsum_result_type(narrays, arrays, &plan->output_type);
}

static uint64_t einsum_integer_mask(CNP_TYPE type) {
    int bits = cnp_dtype_itemsize(type) * 8;
    return bits >= 64 ? UINT64_MAX : ((UINT64_C(1) << bits) - 1);
}

static uint64_t einsum_read_integer(
    const char *pointer, CNP_TYPE source_type) {
    switch (source_type) {
        case CNP_BOOL: return *(const int8_t*)pointer != 0;
        case CNP_BYTE: return (uint64_t)*(const int8_t*)pointer;
        case CNP_UBYTE: return *(const uint8_t*)pointer;
        case CNP_SHORT: return (uint64_t)*(const int16_t*)pointer;
        case CNP_USHORT: return *(const uint16_t*)pointer;
        case CNP_INT: return (uint64_t)*(const int32_t*)pointer;
        case CNP_UINT: return *(const uint32_t*)pointer;
        case CNP_LONG:
        case CNP_LONGLONG: return (uint64_t)*(const int64_t*)pointer;
        case CNP_ULONG:
        case CNP_ULONGLONG: return *(const uint64_t*)pointer;
        default: return 0;
    }
}

static CnpEinsumValue einsum_read_value(
    const CnpArray *array, int64_t offset, CNP_TYPE output_type) {
    CnpEinsumValue result;
    memset(&result, 0, sizeof(result));
    const char *pointer = (const char*)array->data + offset;
    CNP_TYPE source_type = array->dtype->type_num;
    if (output_type == CNP_BOOL || cnp_type_is_integer(output_type)) {
        result.integer = einsum_read_integer(pointer, source_type) &
            einsum_integer_mask(output_type);
    } else if (output_type == CNP_HALF || output_type == CNP_FLOAT) {
        result.float32 = (float)cnp_get_element_double(
            array->data, offset, source_type);
    } else if (output_type == CNP_DOUBLE) {
        result.float64 = cnp_get_element_double(
            array->data, offset, source_type);
    } else if (output_type == CNP_CFLOAT) {
        if (source_type == CNP_CFLOAT) {
            result.complex64 = *(const cnp_cfloat*)pointer;
        } else if (source_type == CNP_CDOUBLE) {
            const cnp_cdouble value = *(const cnp_cdouble*)pointer;
            result.complex64.real = (float)value.real;
            result.complex64.imag = (float)value.imag;
        } else {
            result.complex64.real = (float)cnp_get_element_double(
                array->data, offset, source_type);
            result.complex64.imag = 0.0f;
        }
    } else {
        if (source_type == CNP_CDOUBLE) {
            result.complex128 = *(const cnp_cdouble*)pointer;
        } else if (source_type == CNP_CFLOAT) {
            const cnp_cfloat value = *(const cnp_cfloat*)pointer;
            result.complex128.real = value.real;
            result.complex128.imag = value.imag;
        } else {
            result.complex128.real = cnp_get_element_double(
                array->data, offset, source_type);
            result.complex128.imag = 0.0;
        }
    }
    return result;
}

static CnpEinsumValue einsum_zero(void) {
    CnpEinsumValue result;
    memset(&result, 0, sizeof(result));
    return result;
}

static CnpEinsumValue einsum_one(CNP_TYPE type) {
    CnpEinsumValue result = einsum_zero();
    if (type == CNP_BOOL || cnp_type_is_integer(type)) result.integer = 1;
    else if (type == CNP_HALF || type == CNP_FLOAT)
        result.float32 = 1.0f;
    else if (type == CNP_DOUBLE) result.float64 = 1.0;
    else if (type == CNP_CFLOAT) result.complex64.real = 1.0f;
    else result.complex128.real = 1.0;
    return result;
}

static CnpEinsumValue einsum_multiply(
    CnpEinsumValue left, CnpEinsumValue right, CNP_TYPE type) {
    CnpEinsumValue result = einsum_zero();
    if (type == CNP_BOOL) {
        result.integer = left.integer != 0 && right.integer != 0;
    } else if (cnp_type_is_integer(type)) {
        result.integer = left.integer * right.integer &
            einsum_integer_mask(type);
    } else if (type == CNP_HALF || type == CNP_FLOAT) {
        result.float32 = left.float32 * right.float32;
    } else if (type == CNP_DOUBLE) {
        result.float64 = left.float64 * right.float64;
    } else if (type == CNP_CFLOAT) {
        result.complex64.real = left.complex64.real * right.complex64.real -
            left.complex64.imag * right.complex64.imag;
        result.complex64.imag = left.complex64.real * right.complex64.imag +
            left.complex64.imag * right.complex64.real;
    } else {
        result.complex128.real =
            left.complex128.real * right.complex128.real -
            left.complex128.imag * right.complex128.imag;
        result.complex128.imag =
            left.complex128.real * right.complex128.imag +
            left.complex128.imag * right.complex128.real;
    }
    return result;
}

static CnpEinsumValue einsum_add(
    CnpEinsumValue left, CnpEinsumValue right, CNP_TYPE type) {
    CnpEinsumValue result = einsum_zero();
    if (type == CNP_BOOL) {
        result.integer = left.integer != 0 || right.integer != 0;
    } else if (cnp_type_is_integer(type)) {
        result.integer = (left.integer + right.integer) &
            einsum_integer_mask(type);
    } else if (type == CNP_HALF || type == CNP_FLOAT) {
        result.float32 = left.float32 + right.float32;
    } else if (type == CNP_DOUBLE) {
        result.float64 = left.float64 + right.float64;
    } else if (type == CNP_CFLOAT) {
        result.complex64.real = left.complex64.real + right.complex64.real;
        result.complex64.imag = left.complex64.imag + right.complex64.imag;
    } else {
        result.complex128.real =
            left.complex128.real + right.complex128.real;
        result.complex128.imag =
            left.complex128.imag + right.complex128.imag;
    }
    return result;
}

static void einsum_store_value(
    CnpArray *result, int64_t flat_index, CnpEinsumValue value) {
    char *pointer = (char*)result->data +
        flat_index * result->dtype->elsize;
    switch (result->dtype->type_num) {
        case CNP_BOOL: *(int8_t*)pointer = value.integer != 0; break;
        case CNP_BYTE:
        case CNP_UBYTE: {
            uint8_t bits = (uint8_t)value.integer;
            memcpy(pointer, &bits, sizeof(bits));
            break;
        }
        case CNP_SHORT:
        case CNP_USHORT: {
            uint16_t bits = (uint16_t)value.integer;
            memcpy(pointer, &bits, sizeof(bits));
            break;
        }
        case CNP_INT:
        case CNP_UINT: {
            uint32_t bits = (uint32_t)value.integer;
            memcpy(pointer, &bits, sizeof(bits));
            break;
        }
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
            memcpy(pointer, &value.integer, sizeof(value.integer));
            break;
        case CNP_HALF:
            *(uint16_t*)pointer = cnp_float_to_half((double)value.float32);
            break;
        case CNP_FLOAT: *(float*)pointer = value.float32; break;
        case CNP_DOUBLE: *(double*)pointer = value.float64; break;
        case CNP_CFLOAT: *(cnp_cfloat*)pointer = value.complex64; break;
        case CNP_CDOUBLE: *(cnp_cdouble*)pointer = value.complex128; break;
        default: break;
    }
}

CNP_STATUS cnp_multiply_scalar_values(
        const void *left, CNP_TYPE left_type,
        const void *right, CNP_TYPE right_type,
        void *output, CNP_TYPE output_type,
        const char *function_name) {
    if (!left || !right || !output) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "scalar product pointers must not be null");
        return CNP_ERR_GENERIC;
    }
    if (!einsum_supported_dtype(left_type) ||
            !einsum_supported_dtype(right_type) ||
            !einsum_supported_dtype(output_type)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "scalar product requires represented numeric dtypes");
        return CNP_ERR_TYPE;
    }

    CnpDtype left_dtype = {0};
    CnpDtype right_dtype = {0};
    CnpDtype output_dtype = {0};
    CnpArray left_array = {0};
    CnpArray right_array = {0};
    CnpArray output_array = {0};
    left_dtype.type_num = left_type;
    right_dtype.type_num = right_type;
    output_dtype.type_num = output_type;
    output_dtype.elsize = cnp_dtype_itemsize(output_type);
    left_array.data = (void*)left;
    left_array.dtype = &left_dtype;
    right_array.data = (void*)right;
    right_array.dtype = &right_dtype;
    output_array.data = output;
    output_array.dtype = &output_dtype;

    CnpEinsumValue product = einsum_multiply(
        einsum_read_value(&left_array, 0, output_type),
        einsum_read_value(&right_array, 0, output_type),
        output_type);
    einsum_store_value(&output_array, 0, product);
    return CNP_OK;
}

static void einsum_decode_output_coordinates(
    const CnpEinsumPlan *plan, int64_t flat_index,
    int64_t *label_coordinates) {
    for (int dimension = plan->output_ndim - 1;
         dimension >= 0; --dimension) {
        int label = plan->output_labels[dimension];
        int64_t length = plan->label_lengths[label];
        label_coordinates[label] = length > 0 ? flat_index % length : 0;
        if (length > 0) flat_index /= length;
    }
}

static int64_t einsum_operand_offset(
    const CnpEinsumOperandPlan *operand,
    const int64_t *label_coordinates) {
    const CnpArray *array = operand->array;
    int64_t offset = array->offset;
    for (int axis = 0; axis < array->ndim; ++axis) {
        int label = operand->axis_labels[axis];
        int64_t coordinate = array->shape[axis] == 1
            ? 0 : label_coordinates[label];
        offset += coordinate * array->strides[axis];
    }
    return offset;
}

static bool einsum_increment_contraction(
    const CnpEinsumPlan *plan, int64_t *label_coordinates) {
    for (int index = plan->contraction_count - 1;
         index >= 0; --index) {
        int label = plan->contraction_labels[index];
        ++label_coordinates[label];
        if (label_coordinates[label] < plan->label_lengths[label])
            return true;
        label_coordinates[label] = 0;
    }
    return false;
}

static CnpArray *einsum_execute_generic(const CnpEinsumPlan *plan) {
    CnpArray *result = cnp_array_new(
        plan->output_ndim, plan->output_shape,
        plan->output_type, CNP_ORDER_C);
    if (!result) return NULL;

    bool empty_contraction = false;
    for (int index = 0; index < plan->contraction_count; ++index) {
        if (plan->label_lengths[plan->contraction_labels[index]] == 0) {
            empty_contraction = true;
            break;
        }
    }

    int64_t label_coordinates[CNP_EINSUM_LABEL_CAPACITY];
    for (int64_t output_index = 0;
         output_index < result->size; ++output_index) {
        memset(label_coordinates, 0, sizeof(label_coordinates));
        einsum_decode_output_coordinates(
            plan, output_index, label_coordinates);
        CnpEinsumValue total = einsum_zero();
        if (!empty_contraction) {
            bool more;
            do {
                CnpEinsumValue product = einsum_one(plan->output_type);
                for (int operand_index = 0;
                     operand_index < plan->narrays; ++operand_index) {
                    const CnpEinsumOperandPlan *operand =
                        &plan->operands[operand_index];
                    int64_t offset = einsum_operand_offset(
                        operand, label_coordinates);
                    product = einsum_multiply(
                        product,
                        einsum_read_value(
                            operand->array, offset, plan->output_type),
                        plan->output_type);
                }
                total = einsum_add(total, product, plan->output_type);
                more = plan->contraction_count > 0 &&
                    einsum_increment_contraction(
                        plan, label_coordinates);
            } while (more);
        }
        einsum_store_value(result, output_index, total);
    }
    return result;
}

static bool einsum_operand_is_vector(
    const CnpEinsumOperandPlan *operand, int label) {
    return operand->array->ndim == 1 && operand->axis_labels[0] == label;
}

static bool einsum_operand_is_matrix(
    const CnpEinsumOperandPlan *operand, int first_label, int second_label) {
    return operand->array->ndim == 2 &&
        ((operand->axis_labels[0] == first_label &&
          operand->axis_labels[1] == second_label) ||
         (operand->axis_labels[0] == second_label &&
          operand->axis_labels[1] == first_label));
}

static int64_t einsum_operand_label_length(
    const CnpEinsumOperandPlan *operand, int label) {
    for (int axis = 0; axis < operand->array->ndim; axis++) {
        if (operand->axis_labels[axis] == label)
            return operand->array->shape[axis];
    }
    return -1;
}

static const CnpArray *einsum_orient_matrix(
    const CnpEinsumOperandPlan *operand,
    int first_label, int second_label,
    CnpArray **temporary) {
    *temporary = NULL;
    if (operand->axis_labels[0] == first_label &&
            operand->axis_labels[1] == second_label)
        return operand->array;
    *temporary = cnp_transpose(operand->array, NULL);
    return *temporary;
}

static CnpArray *einsum_execute_fast(
    const CnpEinsumPlan *plan, bool *handled) {
    *handled = false;
    if (plan->output_type != CNP_DOUBLE) return NULL;

    CnpArray *result = NULL;
    if (plan->narrays == 2 && plan->output_ndim == 2 &&
            plan->contraction_count == 1) {
        int row_label = plan->output_labels[0];
        int column_label = plan->output_labels[1];
        int contraction_label = plan->contraction_labels[0];
        const CnpEinsumOperandPlan *left = &plan->operands[0];
        const CnpEinsumOperandPlan *right = &plan->operands[1];
        if (row_label != column_label && row_label != contraction_label &&
                column_label != contraction_label &&
                einsum_operand_is_matrix(
                    left, row_label, contraction_label) &&
                einsum_operand_is_matrix(
                    right, contraction_label, column_label) &&
                einsum_operand_label_length(left, contraction_label) ==
                    einsum_operand_label_length(right, contraction_label)) {
            CnpArray *left_temporary = NULL;
            CnpArray *right_temporary = NULL;
            const CnpArray *left_matrix;
            const CnpArray *right_matrix;
            *handled = true;
            left_matrix = einsum_orient_matrix(
                left, row_label, contraction_label, &left_temporary);
            right_matrix = einsum_orient_matrix(
                right, contraction_label, column_label, &right_temporary);
            if (left_matrix && right_matrix)
                result = cnp_matmul(left_matrix, right_matrix);
            if (left_temporary) cnp_array_free(left_temporary);
            if (right_temporary) cnp_array_free(right_temporary);
        }
    }
    if (!*handled && plan->narrays == 2 && plan->output_ndim == 0 &&
            plan->contraction_count == 1) {
        int contraction_label = plan->contraction_labels[0];
        const CnpEinsumOperandPlan *left = &plan->operands[0];
        const CnpEinsumOperandPlan *right = &plan->operands[1];
        if (einsum_operand_is_vector(left, contraction_label) &&
                einsum_operand_is_vector(right, contraction_label) &&
                left->array->shape[0] == right->array->shape[0]) {
            *handled = true;
            result = cnp_dot(left->array, right->array);
        }
    }
    if (!*handled && plan->narrays == 2 && plan->output_ndim == 2 &&
            plan->contraction_count == 0) {
        const CnpEinsumOperandPlan *left = &plan->operands[0];
        const CnpEinsumOperandPlan *right = &plan->operands[1];
        int left_label = left->array->ndim == 1
            ? left->axis_labels[0] : -1;
        int right_label = right->array->ndim == 1
            ? right->axis_labels[0] : -1;
        if (left_label >= 0 && right_label >= 0 &&
                left_label != right_label &&
                ((plan->output_labels[0] == left_label &&
                  plan->output_labels[1] == right_label) ||
                 (plan->output_labels[0] == right_label &&
                  plan->output_labels[1] == left_label))) {
            *handled = true;
            result = plan->output_labels[0] == left_label
                ? cnp_outer(left->array, right->array)
                : cnp_outer(right->array, left->array);
        }
    }
    if (!*handled && plan->narrays == 2 && plan->output_ndim == 1 &&
            plan->contraction_count == 1) {
        int output_label = plan->output_labels[0];
        int contraction_label = plan->contraction_labels[0];
        const CnpEinsumOperandPlan *left = &plan->operands[0];
        const CnpEinsumOperandPlan *right = &plan->operands[1];
        if (einsum_operand_is_matrix(
                left, output_label, contraction_label) &&
                einsum_operand_is_vector(right, contraction_label) &&
                einsum_operand_label_length(left, contraction_label) ==
                    right->array->shape[0]) {
            CnpArray *matrix_temporary = NULL;
            const CnpArray *matrix;
            *handled = true;
            matrix = einsum_orient_matrix(
                left, output_label, contraction_label, &matrix_temporary);
            if (matrix) result = cnp_dot(matrix, right->array);
            if (matrix_temporary) cnp_array_free(matrix_temporary);
        } else if (einsum_operand_is_vector(left, contraction_label) &&
                einsum_operand_is_matrix(
                    right, contraction_label, output_label) &&
                left->array->shape[0] ==
                    einsum_operand_label_length(
                        right, contraction_label)) {
            CnpArray *matrix_temporary = NULL;
            const CnpArray *matrix;
            *handled = true;
            matrix = einsum_orient_matrix(
                right, contraction_label, output_label, &matrix_temporary);
            if (matrix) result = cnp_dot(left->array, matrix);
            if (matrix_temporary) cnp_array_free(matrix_temporary);
        }
    }
    if (!*handled && plan->narrays == 1 && plan->output_ndim == 0 &&
            plan->contraction_count == 1) {
        int contraction_label = plan->contraction_labels[0];
        const CnpEinsumOperandPlan *operand = &plan->operands[0];
        if (operand->array->ndim == 2 &&
                operand->axis_labels[0] == contraction_label &&
                operand->axis_labels[1] == contraction_label) {
            *handled = true;
            result = cnp_array_from_scalar(
                cnp_trace_ext(operand->array, 0), CNP_DOUBLE);
        }
    }
    if (*handled && !result) cnp_relabel_error("cnp_einsum");
    return result;
}

static CnpArray *einsum_execute_expression(
    const char *subscripts, int narrays,
    const CnpArray *const *arrays,
    bool allow_fast, const char *function_name) {
    if (!subscripts) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "subscripts must not be null");
        return NULL;
    }
    if (narrays <= 0 || narrays > CNP_MAXARGS) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "operand count must be in [1, %d]", CNP_MAXARGS);
        return NULL;
    }
    if (!arrays) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "operand pointer array must not be null");
        return NULL;
    }
    for (int index = 0; index < narrays; ++index) {
        if (!arrays[index]) {
            cnp_set_error(CNP_ERR_GENERIC, function_name,
                          "operand %d must not be null", index);
            return NULL;
        }
    }

    size_t source_length = strlen(subscripts);
    if (source_length >= (size_t)INT_MAX) {
        cnp_set_error(CNP_ERR_VALUE, function_name,
                      "subscripts are too long to parse");
        return NULL;
    }
    char *expression = (char*)cnp_malloc(source_length + 1);
    if (!expression) {
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "failed to allocate parsed subscript storage");
        return NULL;
    }
    int expression_length = 0;
    for (size_t index = 0; index < source_length; ++index) {
        unsigned char value = (unsigned char)subscripts[index];
        if (!isspace(value)) expression[expression_length++] = (char)value;
    }
    expression[expression_length] = '\0';

    CnpEinsumPlan plan;
    CnpArray *result = NULL;
    if (einsum_parse(
            expression, expression_length, narrays, arrays, &plan)) {
        bool handled = false;
        if (allow_fast) result = einsum_execute_fast(&plan, &handled);
        if (!handled) result = einsum_execute_generic(&plan);
    }
    cnp_free(expression, source_length + 1);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_einsum(
    const char *subscripts, int narrays,
    const CnpArray *const *arrays) {
    return einsum_execute_expression(
        subscripts, narrays, arrays, true, "cnp_einsum");
}

CnpArray *cnp_einsum_generic(
    const char *subscripts, int narrays,
    const CnpArray *const *arrays, const char *function_name) {
    return einsum_execute_expression(
        subscripts, narrays, arrays, false, function_name);
}
