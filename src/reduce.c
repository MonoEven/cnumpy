/**
 * cnumpy reduction operations - sum, prod, mean, std, var, max, min, etc.
 */
#include "../include/cnumpy/cnumpy_internal.h"

/* =========================================================================
 * Public reduction API
 * ========================================================================= */
static bool reduction_resolve_axis(
    const CnpArray *arr, int axis, bool axis_none,
    const char *function_name, int *resolved_axis) {
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "source array must not be null");
        return false;
    }
    if (axis_none) {
        *resolved_axis = CNP_AXIS_NONE;
        return true;
    }
    if (arr->ndim == 0) {
        if (axis == 0 || axis == -1) {
            *resolved_axis = CNP_AXIS_NONE;
            return true;
        }
        cnp_set_error(CNP_ERR_AXIS, function_name,
                      "axis %d is out of bounds for array of dimension 0",
                      axis);
        return false;
    }
    int normalized = axis;
    if (normalized < 0) normalized += arr->ndim;
    if (normalized < 0 || normalized >= arr->ndim) {
        cnp_set_error(CNP_ERR_AXIS, function_name,
                      "axis %d is out of bounds for array of dimension %d",
                      axis, arr->ndim);
        return false;
    }
    *resolved_axis = normalized;
    return true;
}

static void reduction_traversal_init(
    const CnpArray *arr, int resolved_axis,
    CnpReductionTraversal *traversal) {
    traversal->array = arr;
    traversal->axis_none = resolved_axis == CNP_AXIS_NONE;
    traversal->outer = 1;
    traversal->inner = 1;
    if (traversal->axis_none) {
        traversal->axis = 0;
        traversal->axis_length = arr->size;
        traversal->axis_stride = arr->dtype->elsize;
        traversal->result_ndim = 0;
        traversal->result_shape[0] = 1;
        return;
    }
    traversal->axis = resolved_axis;
    traversal->axis_length = arr->shape[resolved_axis];
    traversal->axis_stride = arr->strides[resolved_axis];
    traversal->result_ndim = arr->ndim - 1;
    int result_dimension = 0;
    for (int dimension = 0; dimension < arr->ndim; ++dimension) {
        if (dimension < resolved_axis)
            traversal->outer *= arr->shape[dimension];
        else if (dimension > resolved_axis)
            traversal->inner *= arr->shape[dimension];
        if (dimension != resolved_axis)
            traversal->result_shape[result_dimension++] = arr->shape[dimension];
    }
    if (traversal->result_ndim == 0)
        traversal->result_shape[0] = 1;
}

static int64_t reduction_flat_offset(
    const CnpArray *arr, int64_t flat) {
    int64_t offset = arr->offset;
    for (int dimension = arr->ndim - 1; dimension >= 0; --dimension) {
        int64_t coordinate = flat % arr->shape[dimension];
        flat /= arr->shape[dimension];
        offset += coordinate * arr->strides[dimension];
    }
    return offset;
}

static int64_t reduction_slice_base(
    const CnpReductionTraversal *traversal,
    int64_t outer, int64_t inner) {
    const CnpArray *arr = traversal->array;
    int64_t offset = arr->offset;
    for (int dimension = traversal->axis - 1;
         dimension >= 0; --dimension) {
        int64_t coordinate = outer % arr->shape[dimension];
        outer /= arr->shape[dimension];
        offset += coordinate * arr->strides[dimension];
    }
    for (int dimension = arr->ndim - 1;
         dimension > traversal->axis; --dimension) {
        int64_t coordinate = inner % arr->shape[dimension];
        inner /= arr->shape[dimension];
        offset += coordinate * arr->strides[dimension];
    }
    return offset;
}

static int64_t reduction_source_offset(
    const CnpReductionTraversal *traversal,
    int64_t outer, int64_t inner, int64_t item) {
    if (traversal->axis_none) {
        const CnpArray *arr = traversal->array;
        if ((arr->flags & CNP_ARRAY_C_CONTIGUOUS) != 0)
            return arr->offset + item * arr->dtype->elsize;
        return reduction_flat_offset(arr, item);
    }
    return reduction_slice_base(traversal, outer, inner) +
        item * traversal->axis_stride;
}

bool cnp_reduction_resolve_axis(
    const CnpArray *arr, int axis, bool axis_none,
    const char *function_name, int *resolved_axis) {
    return reduction_resolve_axis(
        arr, axis, axis_none, function_name, resolved_axis);
}

bool cnp_reduction_resolve_axis_strict_scalar(
    const CnpArray *arr, int axis, bool axis_none,
    const char *function_name, int *resolved_axis) {
    if (arr && arr->ndim == 0 && !axis_none) {
        cnp_set_error(CNP_ERR_AXIS, function_name,
                      "axis %d is out of bounds for array of dimension 0",
                      axis);
        return false;
    }
    return reduction_resolve_axis(
        arr, axis, axis_none, function_name, resolved_axis);
}

void cnp_reduction_traversal_init(
    const CnpArray *arr, int resolved_axis,
    CnpReductionTraversal *traversal) {
    reduction_traversal_init(arr, resolved_axis, traversal);
}

int64_t cnp_reduction_source_offset(
    const CnpReductionTraversal *traversal,
    int64_t outer, int64_t inner, int64_t item) {
    return reduction_source_offset(traversal, outer, inner, item);
}

static int reduction_compare_offsets(
    const CnpArray *arr, int64_t left_offset, int64_t right_offset) {
    const char *left = (const char*)arr->data + left_offset;
    const char *right = (const char*)arr->data + right_offset;
#define CNP_COMPARE_TYPED(type) \
    do { \
        type left_value = *(const type*)left; \
        type right_value = *(const type*)right; \
        return (left_value > right_value) - (left_value < right_value); \
    } while (0)
    switch (arr->dtype->type_num) {
        case CNP_BOOL:
        case CNP_BYTE: CNP_COMPARE_TYPED(int8_t);
        case CNP_UBYTE: CNP_COMPARE_TYPED(uint8_t);
        case CNP_SHORT: CNP_COMPARE_TYPED(int16_t);
        case CNP_USHORT: CNP_COMPARE_TYPED(uint16_t);
        case CNP_INT: CNP_COMPARE_TYPED(int32_t);
        case CNP_UINT: CNP_COMPARE_TYPED(uint32_t);
        case CNP_LONG:
        case CNP_LONGLONG: CNP_COMPARE_TYPED(int64_t);
        case CNP_ULONG:
        case CNP_ULONGLONG: CNP_COMPARE_TYPED(uint64_t);
        case CNP_FLOAT: CNP_COMPARE_TYPED(float);
        case CNP_DOUBLE: CNP_COMPARE_TYPED(double);
        default: {
            double left_value = cnp_get_element_double(
                arr->data, left_offset, arr->dtype->type_num);
            double right_value = cnp_get_element_double(
                arr->data, right_offset, arr->dtype->type_num);
            return (left_value > right_value) -
                (left_value < right_value);
        }
    }
#undef CNP_COMPARE_TYPED
}

static double reduction_canonical_nan_double(void) {
    uint64_t bits = UINT64_C(0x7ff8000000000000);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static CnpArray *reduction_extrema_contiguous_double(
    const CnpArray *arr, CnpArray *result,
    const CnpReductionTraversal *traversal, bool maximum) {
    const double *values = (const double*)(
        (const char*)arr->data + arr->offset);
    for (int64_t output_index = 0;
         output_index < result->size; ++output_index) {
        const double *slice = values +
            (traversal->axis_none
                ? 0 : output_index * traversal->axis_length);
        double selected = slice[0];
        int64_t selected_index = 0;
        int64_t item = 1;
        if (traversal->axis_length >= 5) {
            int64_t vector_end = 1 +
                ((traversal->axis_length - 1) / 4) * 4;
            double lanes[4] = {selected, selected, selected, selected};
            int64_t lane_indices[4] = {0, 0, 0, 0};
            bool vector_nan = isnan(selected);
            for (; item < vector_end; item += 4) {
                for (int lane = 0; lane < 4; ++lane) {
                    double source = slice[item + lane];
                    if (isnan(source)) {
                        vector_nan = true;
                        continue;
                    }
                    if (isnan(lanes[lane]) ||
                        (maximum
                            ? source >= lanes[lane]
                            : source <= lanes[lane])) {
                        lanes[lane] = source;
                        lane_indices[lane] = item + lane;
                    }
                }
            }
            if (vector_nan) {
                selected = reduction_canonical_nan_double();
                selected_index = -1;
            } else {
                selected = lanes[0];
                selected_index = lane_indices[0];
                for (int lane = 1; lane < 4; ++lane) {
                    bool select_lane = maximum
                        ? lanes[lane] > selected
                        : lanes[lane] < selected;
                    if (!select_lane && lanes[lane] == selected)
                        select_lane =
                            lane_indices[lane] > selected_index;
                    if (select_lane) {
                        selected = lanes[lane];
                        selected_index = lane_indices[lane];
                    }
                }
            }
        }
        if (!traversal->axis_none && arr->ndim > 1 &&
            traversal->axis_length > 1 &&
            traversal->axis_length < 5 && isnan(selected)) {
            selected = reduction_canonical_nan_double();
            selected_index = -1;
        }
        bool initial_nan = traversal->axis_length < 5 && isnan(selected);
        for (; item < traversal->axis_length; ++item) {
            double source = slice[item];
            bool select_source;
            if (isnan(selected)) {
                if (initial_nan && isnan(source)) {
                    selected = reduction_canonical_nan_double();
                    selected_index = -1;
                    initial_nan = false;
                }
                select_source = false;
            } else if (isnan(source)) {
                select_source = true;
            } else {
                select_source = maximum
                    ? source >= selected : source <= selected;
            }
            if (select_source) {
                selected = source;
                selected_index = item;
            }
        }
        if (selected_index >= 0) {
            memcpy((char*)result->data + output_index * sizeof(double),
                   slice + selected_index, sizeof(double));
        } else {
            memcpy((char*)result->data + output_index * sizeof(double),
                   &selected, sizeof(double));
        }
    }
    return result;
}

static CnpArray *reduction_arg_extrema_contiguous_double(
    const CnpArray *arr, CnpArray *result,
    const CnpReductionTraversal *traversal, bool maximum) {
    const double *values = (const double*)(
        (const char*)arr->data + arr->offset);
    for (int64_t output_index = 0;
         output_index < result->size; ++output_index) {
        const double *slice = values +
            (traversal->axis_none
                ? 0 : output_index * traversal->axis_length);
        double selected = slice[0];
        int64_t selected_item = 0;
        for (int64_t item = 1;
             item < traversal->axis_length; ++item) {
            double source = slice[item];
            bool select_source = !isnan(selected) &&
                (isnan(source) ||
                 (maximum ? source > selected : source < selected));
            if (select_source) {
                selected = source;
                selected_item = item;
            }
        }
        ((int64_t*)result->data)[output_index] = selected_item;
    }
    return result;
}

static CnpArray *reduction_extrema(
    const CnpArray *arr, int resolved_axis, bool maximum,
    const char *function_name) {
    CnpReductionTraversal traversal;
    reduction_traversal_init(arr, resolved_axis, &traversal);
    CnpArray *result = cnp_array_new(
        traversal.result_ndim, traversal.result_shape,
        arr->dtype->type_num, CNP_ORDER_C);
    if (!result) return NULL;
    if (result->size != 0 && traversal.axis_length == 0) {
        cnp_array_free(result);
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "zero-size reduction has no identity");
        return NULL;
    }
    if ((traversal.axis_none ||
         traversal.axis == arr->ndim - 1) &&
        (arr->flags & CNP_ARRAY_C_CONTIGUOUS) != 0 &&
        arr->dtype->type_num == CNP_DOUBLE)
        return reduction_extrema_contiguous_double(
            arr, result, &traversal, maximum);
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            int64_t selected_offset = reduction_source_offset(
                &traversal, outer, inner, 0);
            for (int64_t item = 1;
                 item < traversal.axis_length; ++item) {
                int64_t source_offset = reduction_source_offset(
                    &traversal, outer, inner, item);
                double selected_value = cnp_get_element_double(
                    arr->data, selected_offset, arr->dtype->type_num);
                double source_value = cnp_get_element_double(
                    arr->data, source_offset, arr->dtype->type_num);
                bool select_source;
                if (isnan(selected_value)) {
                    select_source = false;
                } else if (isnan(source_value)) {
                    select_source = true;
                } else {
                    int comparison = reduction_compare_offsets(
                        arr, source_offset, selected_offset);
                    select_source = maximum
                        ? comparison >= 0 : comparison <= 0;
                }
                if (select_source) selected_offset = source_offset;
            }
            int64_t output_index = outer * traversal.inner + inner;
            memcpy((char*)result->data +
                       output_index * result->dtype->elsize,
                   (const char*)arr->data + selected_offset,
                   result->dtype->elsize);
        }
    }
    return result;
}

static CnpArray *reduction_arg_extrema(
    const CnpArray *arr, int resolved_axis, bool maximum,
    const char *function_name) {
    CnpReductionTraversal traversal;
    reduction_traversal_init(arr, resolved_axis, &traversal);
    CnpArray *result = cnp_array_new(
        traversal.result_ndim, traversal.result_shape,
        CNP_LONGLONG, CNP_ORDER_C);
    if (!result) return NULL;
    if (result->size != 0 && traversal.axis_length == 0) {
        cnp_array_free(result);
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "attempt to get arg reduction of an empty sequence");
        return NULL;
    }
    if ((traversal.axis_none ||
         traversal.axis == arr->ndim - 1) &&
        (arr->flags & CNP_ARRAY_C_CONTIGUOUS) != 0 &&
        arr->dtype->type_num == CNP_DOUBLE)
        return reduction_arg_extrema_contiguous_double(
            arr, result, &traversal, maximum);
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            int64_t selected_item = 0;
            int64_t selected_offset = reduction_source_offset(
                &traversal, outer, inner, 0);
            for (int64_t item = 1;
                 item < traversal.axis_length; ++item) {
                int64_t source_offset = reduction_source_offset(
                    &traversal, outer, inner, item);
                double selected_value = cnp_get_element_double(
                    arr->data, selected_offset, arr->dtype->type_num);
                double source_value = cnp_get_element_double(
                    arr->data, source_offset, arr->dtype->type_num);
                bool select_source = !isnan(selected_value) &&
                    (isnan(source_value) ||
                     (maximum
                         ? reduction_compare_offsets(
                               arr, source_offset, selected_offset) > 0
                         : reduction_compare_offsets(
                               arr, source_offset, selected_offset) < 0));
                if (select_source) {
                    selected_item = item;
                    selected_offset = source_offset;
                }
            }
            ((int64_t*)result->data)[
                outer * traversal.inner + inner] = selected_item;
        }
    }
    return result;
}

static double reduction_positive_infinity(void) {
    uint64_t bits = UINT64_C(0x7ff0000000000000);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/*
 * NumPy's add.reduce combines the pairwise partial with a +0 identity.
 * Release uses /fp:fast, which may fold that arithmetic identity away, so
 * enforce its only observable effect by clearing the sign bit of exact zero.
 */
static float reduction_apply_sum_identity_float(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT32_C(0x7fffffff)) == 0)
        bits = 0;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static double reduction_apply_sum_identity_double(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT64_C(0x7fffffffffffffff)) == 0)
        bits = 0;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

typedef enum {
    CNP_REDUCTION_VALUE,
    CNP_REDUCTION_NAN_TO_ZERO,
    CNP_REDUCTION_SQUARED_DEVIATION,
    CNP_REDUCTION_NAN_SQUARED_DEVIATION
} CnpReductionValueMode;

#if defined(_MSC_VER)
#pragma float_control(precise, on, push)
#endif
static float reduction_divide_float(
    float numerator, int64_t denominator) {
    return numerator / (float)denominator;
}

static double reduction_divide_double(
    double numerator, int64_t denominator) {
    return numerator / (double)denominator;
}

static float reduction_squared_deviation_float(
    float value, float center) {
    float difference = value - center;
    return difference * difference;
}

static double reduction_squared_deviation_double(
    double value, double center) {
    double difference = value - center;
    return difference * difference;
}

static float reduction_squared_double_to_float(
    double value, double center) {
    return (float)reduction_squared_deviation_double(value, center);
}

static double reduction_nan_squared_float_to_double(
    float value, double center) {
    float difference = (float)((double)value - center);
    return (double)(difference * difference);
}
#if defined(_MSC_VER)
#pragma float_control(pop)
#endif

static float reduction_source_float(
    const CnpReductionTraversal *traversal,
    int64_t outer, int64_t inner, int64_t item,
    CnpReductionValueMode mode, float center);

static double reduction_source_double(
    const CnpReductionTraversal *traversal,
    int64_t outer, int64_t inner, int64_t item,
    CnpReductionValueMode mode, double center);

static float reduction_pairwise_sum_float(
    const CnpReductionTraversal *traversal,
    int64_t outer, int64_t inner,
    int64_t first, int64_t length,
    CnpReductionValueMode mode, float center);

static double reduction_pairwise_sum_double(
    const CnpReductionTraversal *traversal,
    int64_t outer, int64_t inner,
    int64_t first, int64_t length,
    CnpReductionValueMode mode, double center);

static double reduction_pairwise_sum_contiguous_double(
    const double *values,
    int64_t first, int64_t length,
    CnpReductionValueMode mode, double center);

static CnpArray *reduction_variance(
    const CnpArray *arr, int resolved_axis, int ddof,
    CNP_TYPE dtype, bool standard_deviation) {
    CnpReductionTraversal traversal;
    reduction_traversal_init(arr, resolved_axis, &traversal);
    CNP_TYPE out_dtype = dtype != CNP_NOTYPE ? dtype :
        (arr->dtype->type_num == CNP_FLOAT ? CNP_FLOAT : CNP_DOUBLE);
    CnpArray *result = cnp_array_new(
        traversal.result_ndim, traversal.result_shape,
        out_dtype, CNP_ORDER_C);
    if (!result) return NULL;
    bool contiguous_double_variance = traversal.axis_none &&
        (arr->flags & CNP_ARRAY_C_CONTIGUOUS) != 0 &&
        arr->dtype->type_num == CNP_DOUBLE &&
        out_dtype == CNP_DOUBLE;
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            int64_t divisor = traversal.axis_length - ddof;
            int64_t output_index = outer * traversal.inner + inner;
            if (out_dtype == CNP_FLOAT) {
                float total = 0.0f;
                if (traversal.axis_length != 0)
                    total += reduction_pairwise_sum_float(
                        &traversal, outer, inner, 0,
                        traversal.axis_length,
                        CNP_REDUCTION_VALUE, 0.0f);
                total = reduction_apply_sum_identity_float(total);
                float mean = reduction_divide_float(
                    total, traversal.axis_length);
                float squared_deviation = 0.0f;
                if (traversal.axis_length != 0)
                    squared_deviation += reduction_pairwise_sum_float(
                        &traversal, outer, inner, 0,
                        traversal.axis_length,
                        CNP_REDUCTION_SQUARED_DEVIATION, mean);
                squared_deviation = reduction_apply_sum_identity_float(
                    squared_deviation);
                float value = divisor > 0
                    ? reduction_divide_float(
                        squared_deviation, divisor)
                    : (squared_deviation == 0.0f ||
                       isnan(squared_deviation)
                        ? NAN : (float)reduction_positive_infinity());
                if (standard_deviation) value = sqrtf(value);
                ((float*)result->data)[output_index] = value;
                continue;
            }
            if (out_dtype == CNP_DOUBLE) {
                const double *values = (const double*)(
                    (const char*)arr->data + arr->offset);
                double total = 0.0;
                if (traversal.axis_length != 0)
                    total += contiguous_double_variance
                        ? reduction_pairwise_sum_contiguous_double(
                              values, 0, traversal.axis_length,
                              CNP_REDUCTION_VALUE, 0.0)
                        : reduction_pairwise_sum_double(
                              &traversal, outer, inner, 0,
                              traversal.axis_length,
                              CNP_REDUCTION_VALUE, 0.0);
                total = reduction_apply_sum_identity_double(total);
                double mean = reduction_divide_double(
                    total, traversal.axis_length);
                double squared_deviation = 0.0;
                if (traversal.axis_length != 0)
                    squared_deviation += contiguous_double_variance
                        ? reduction_pairwise_sum_contiguous_double(
                              values, 0, traversal.axis_length,
                              CNP_REDUCTION_SQUARED_DEVIATION, mean)
                        : reduction_pairwise_sum_double(
                              &traversal, outer, inner, 0,
                              traversal.axis_length,
                              CNP_REDUCTION_SQUARED_DEVIATION, mean);
                squared_deviation = reduction_apply_sum_identity_double(
                    squared_deviation);
                double value = divisor > 0
                    ? reduction_divide_double(
                        squared_deviation, divisor)
                    : (squared_deviation == 0.0 ||
                       isnan(squared_deviation)
                        ? NAN : reduction_positive_infinity());
                if (standard_deviation) value = sqrt(value);
                ((double*)result->data)[output_index] = value;
                continue;
            }

            double mean = 0.0;
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item)
                mean += reduction_source_double(
                    &traversal, outer, inner, item,
                    CNP_REDUCTION_VALUE, 0.0);
            mean /= (double)traversal.axis_length;
            double squared_deviation = 0.0;
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item)
                squared_deviation += reduction_source_double(
                    &traversal, outer, inner, item,
                    CNP_REDUCTION_SQUARED_DEVIATION, mean);
            double value = divisor > 0
                ? reduction_divide_double(
                    squared_deviation, divisor)
                : (squared_deviation == 0.0 || isnan(squared_deviation)
                    ? NAN : reduction_positive_infinity());
            if (standard_deviation) value = sqrt(value);
            cnp_set_element_double(
                result->data, output_index * result->dtype->elsize,
                out_dtype, value);
        }
    }
    return result;
}

static bool reduction_type_is_integer(CNP_TYPE type) {
    return type == CNP_BOOL || cnp_type_is_integer(type);
}

static CNP_TYPE reduction_default_sum_dtype(CNP_TYPE type) {
    switch (type) {
        case CNP_BOOL:
        case CNP_BYTE:
        case CNP_SHORT:
        case CNP_INT:
            return CNP_INT;
        case CNP_UBYTE:
        case CNP_USHORT:
        case CNP_UINT:
            return CNP_UINT;
        default:
            return type;
    }
}

static uint64_t reduction_read_integer_bits(
    const CnpArray *arr, int64_t offset) {
    const char *pointer = (const char*)arr->data + offset;
    switch (arr->dtype->type_num) {
        case CNP_BOOL: return *(const int8_t*)pointer != 0;
        case CNP_BYTE: return (uint64_t)(int64_t)*(const int8_t*)pointer;
        case CNP_UBYTE: return *(const uint8_t*)pointer;
        case CNP_SHORT: return (uint64_t)(int64_t)*(const int16_t*)pointer;
        case CNP_USHORT: return *(const uint16_t*)pointer;
        case CNP_INT: return (uint64_t)(int64_t)*(const int32_t*)pointer;
        case CNP_UINT: return *(const uint32_t*)pointer;
        case CNP_LONG:
        case CNP_LONGLONG: {
            uint64_t bits;
            memcpy(&bits, pointer, sizeof(bits));
            return bits;
        }
        case CNP_ULONG:
        case CNP_ULONGLONG: return *(const uint64_t*)pointer;
        default: return 0;
    }
}

static void reduction_store_integer_bits(
    CnpArray *result, int64_t output_index, uint64_t bits) {
    char *target = (char*)result->data +
        output_index * result->dtype->elsize;
    switch (result->dtype->type_num) {
        case CNP_BOOL:
            *(int8_t*)target = bits != 0;
            break;
        case CNP_BYTE: {
            uint8_t narrowed = (uint8_t)bits;
            memcpy(target, &narrowed, sizeof(narrowed));
            break;
        }
        case CNP_UBYTE:
            *(uint8_t*)target = (uint8_t)bits;
            break;
        case CNP_SHORT: {
            uint16_t narrowed = (uint16_t)bits;
            memcpy(target, &narrowed, sizeof(narrowed));
            break;
        }
        case CNP_USHORT:
            *(uint16_t*)target = (uint16_t)bits;
            break;
        case CNP_INT: {
            uint32_t narrowed = (uint32_t)bits;
            memcpy(target, &narrowed, sizeof(narrowed));
            break;
        }
        case CNP_UINT:
            *(uint32_t*)target = (uint32_t)bits;
            break;
        case CNP_LONG:
        case CNP_LONGLONG:
            memcpy(target, &bits, sizeof(bits));
            break;
        case CNP_ULONG:
        case CNP_ULONGLONG:
            *(uint64_t*)target = bits;
            break;
        default:
            break;
    }
}

static CnpArray *reduction_sum_integer(
    const CnpArray *arr, int resolved_axis, CNP_TYPE out_dtype) {
    CnpReductionTraversal traversal;
    reduction_traversal_init(arr, resolved_axis, &traversal);
    CnpArray *result = cnp_array_new(
        traversal.result_ndim, traversal.result_shape,
        out_dtype, CNP_ORDER_C);
    if (!result) return NULL;
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            uint64_t accumulator = 0;
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item) {
                int64_t source_offset = traversal.axis_none
                    ? reduction_flat_offset(arr, item)
                    : reduction_slice_base(&traversal, outer, inner) +
                      item * traversal.axis_stride;
                uint64_t value = reduction_read_integer_bits(
                    arr, source_offset);
                if (out_dtype == CNP_BOOL)
                    accumulator = accumulator != 0 || value != 0;
                else
                    accumulator += value;
            }
            reduction_store_integer_bits(
                result, outer * traversal.inner + inner, accumulator);
        }
    }
    return result;
}

static CnpArray *reduction_prod_integer(
    const CnpArray *arr, int resolved_axis, CNP_TYPE out_dtype) {
    CnpReductionTraversal traversal;
    reduction_traversal_init(arr, resolved_axis, &traversal);
    CnpArray *result = cnp_array_new(
        traversal.result_ndim, traversal.result_shape,
        out_dtype, CNP_ORDER_C);
    if (!result) return NULL;
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            uint64_t accumulator = 1;
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item) {
                int64_t source_offset = traversal.axis_none
                    ? reduction_flat_offset(arr, item)
                    : reduction_slice_base(&traversal, outer, inner) +
                      item * traversal.axis_stride;
                uint64_t value = reduction_read_integer_bits(
                    arr, source_offset);
                if (out_dtype == CNP_BOOL)
                    accumulator = accumulator != 0 && value != 0;
                else
                    accumulator *= value;
            }
            reduction_store_integer_bits(
                result, outer * traversal.inner + inner, accumulator);
        }
    }
    return result;
}

#define CNP_REDUCTION_PAIRWISE_BLOCK 128

static float reduction_source_float(
    const CnpReductionTraversal *traversal,
    int64_t outer, int64_t inner, int64_t item,
    CnpReductionValueMode mode, float center) {
    int64_t source_offset = reduction_source_offset(
        traversal, outer, inner, item);
    double source_value = cnp_get_element_double(
        traversal->array->data, source_offset,
        traversal->array->dtype->type_num);
    if ((mode == CNP_REDUCTION_NAN_TO_ZERO ||
         mode == CNP_REDUCTION_NAN_SQUARED_DEVIATION) &&
        isnan(source_value))
        return 0.0f;
    if (mode == CNP_REDUCTION_SQUARED_DEVIATION ||
        mode == CNP_REDUCTION_NAN_SQUARED_DEVIATION) {
        if (traversal->array->dtype->type_num == CNP_DOUBLE)
            return reduction_squared_double_to_float(
                source_value, (double)center);
        return reduction_squared_deviation_float(
            (float)source_value, center);
    }
    return (float)source_value;
}

static double reduction_source_double(
    const CnpReductionTraversal *traversal,
    int64_t outer, int64_t inner, int64_t item,
    CnpReductionValueMode mode, double center) {
    int64_t source_offset = reduction_source_offset(
        traversal, outer, inner, item);
    double value = cnp_get_element_double(
        traversal->array->data, source_offset,
        traversal->array->dtype->type_num);
    if ((mode == CNP_REDUCTION_NAN_TO_ZERO ||
         mode == CNP_REDUCTION_NAN_SQUARED_DEVIATION) &&
        isnan(value))
        return 0.0;
    if (mode == CNP_REDUCTION_SQUARED_DEVIATION ||
        mode == CNP_REDUCTION_NAN_SQUARED_DEVIATION) {
        if (mode == CNP_REDUCTION_NAN_SQUARED_DEVIATION &&
            traversal->array->dtype->type_num == CNP_FLOAT)
            return reduction_nan_squared_float_to_double(
                (float)value, center);
        return reduction_squared_deviation_double(value, center);
    }
    return value;
}

/* Match NumPy 1.25 loops_utils.h pairwise summation order exactly. */
static float reduction_pairwise_sum_float(
    const CnpReductionTraversal *traversal,
    int64_t outer, int64_t inner,
    int64_t first, int64_t length,
    CnpReductionValueMode mode, float center) {
    if (length < 8) {
        float result = -0.0f;
        for (int64_t item = 0; item < length; ++item)
            result += reduction_source_float(
                traversal, outer, inner, first + item,
                mode, center);
        return result;
    }
    if (length <= CNP_REDUCTION_PAIRWISE_BLOCK) {
        float partial[8];
        for (int lane = 0; lane < 8; ++lane)
            partial[lane] = reduction_source_float(
                traversal, outer, inner, first + lane,
                mode, center);
        int64_t item;
        for (item = 8; item < length - (length % 8); item += 8) {
            for (int lane = 0; lane < 8; ++lane)
                partial[lane] += reduction_source_float(
                    traversal, outer, inner, first + item + lane,
                    mode, center);
        }
        float result =
            ((partial[0] + partial[1]) +
             (partial[2] + partial[3])) +
            ((partial[4] + partial[5]) +
             (partial[6] + partial[7]));
        for (; item < length; ++item)
            result += reduction_source_float(
                traversal, outer, inner, first + item,
                mode, center);
        return result;
    }
    int64_t left_length = length / 2;
    left_length -= left_length % 8;
    return reduction_pairwise_sum_float(
               traversal, outer, inner, first, left_length,
               mode, center) +
           reduction_pairwise_sum_float(
               traversal, outer, inner,
               first + left_length, length - left_length,
               mode, center);
}

static double reduction_pairwise_sum_double(
    const CnpReductionTraversal *traversal,
    int64_t outer, int64_t inner,
    int64_t first, int64_t length,
    CnpReductionValueMode mode, double center) {
    if (length < 8) {
        double result = -0.0;
        for (int64_t item = 0; item < length; ++item)
            result += reduction_source_double(
                traversal, outer, inner, first + item,
                mode, center);
        return result;
    }
    if (length <= CNP_REDUCTION_PAIRWISE_BLOCK) {
        double partial[8];
        for (int lane = 0; lane < 8; ++lane)
            partial[lane] = reduction_source_double(
                traversal, outer, inner, first + lane,
                mode, center);
        int64_t item;
        for (item = 8; item < length - (length % 8); item += 8) {
            for (int lane = 0; lane < 8; ++lane)
                partial[lane] += reduction_source_double(
                    traversal, outer, inner, first + item + lane,
                    mode, center);
        }
        double result =
            ((partial[0] + partial[1]) +
             (partial[2] + partial[3])) +
            ((partial[4] + partial[5]) +
             (partial[6] + partial[7]));
        for (; item < length; ++item)
            result += reduction_source_double(
                traversal, outer, inner, first + item,
                mode, center);
        return result;
    }
    int64_t left_length = length / 2;
    left_length -= left_length % 8;
    return reduction_pairwise_sum_double(
               traversal, outer, inner, first, left_length,
               mode, center) +
           reduction_pairwise_sum_double(
               traversal, outer, inner,
               first + left_length, length - left_length,
               mode, center);
}

static double reduction_contiguous_double_value(
    const double *values, int64_t item,
    CnpReductionValueMode mode, double center) {
    double value = values[item];
    if (mode == CNP_REDUCTION_SQUARED_DEVIATION)
        return reduction_squared_deviation_double(value, center);
    return value;
}

static double reduction_pairwise_sum_contiguous_double(
    const double *values,
    int64_t first, int64_t length,
    CnpReductionValueMode mode, double center) {
    if (length < 8) {
        double result = -0.0;
        for (int64_t item = 0; item < length; ++item)
            result += reduction_contiguous_double_value(
                values, first + item, mode, center);
        return result;
    }
    if (length <= CNP_REDUCTION_PAIRWISE_BLOCK) {
        double partial[8];
        for (int lane = 0; lane < 8; ++lane)
            partial[lane] = reduction_contiguous_double_value(
                values, first + lane, mode, center);
        int64_t item;
        for (item = 8; item < length - (length % 8); item += 8) {
            for (int lane = 0; lane < 8; ++lane)
                partial[lane] += reduction_contiguous_double_value(
                    values, first + item + lane, mode, center);
        }
        double result =
            ((partial[0] + partial[1]) +
             (partial[2] + partial[3])) +
            ((partial[4] + partial[5]) +
             (partial[6] + partial[7]));
        for (; item < length; ++item)
            result += reduction_contiguous_double_value(
                values, first + item, mode, center);
        return result;
    }
    int64_t left_length = length / 2;
    left_length -= left_length % 8;
    return reduction_pairwise_sum_contiguous_double(
               values, first, left_length, mode, center) +
           reduction_pairwise_sum_contiguous_double(
               values, first + left_length, length - left_length,
               mode, center);
}

double cnp_reduction_sum_contiguous_double(
    const double *values, int64_t length) {
    double result = 0.0;
    if (length != 0)
        result += reduction_pairwise_sum_contiguous_double(
            values, 0, length, CNP_REDUCTION_VALUE, 0.0);
    return reduction_apply_sum_identity_double(result);
}

static double reduction_product_contiguous_double(
    const double *values, int64_t length) {
    double accumulator = 1.0;
    for (int64_t item = 0; item < length; ++item)
        accumulator *= values[item];
    return accumulator;
}

static CnpArray *reduction_sumprod_real(
    const CnpArray *arr, int resolved_axis,
    CNP_TYPE out_dtype, bool product) {
    CnpReductionTraversal traversal;
    reduction_traversal_init(arr, resolved_axis, &traversal);
    CnpArray *result = cnp_array_new(
        traversal.result_ndim, traversal.result_shape,
        out_dtype, CNP_ORDER_C);
    if (!result) return NULL;
    bool contiguous_double_slices =
        (arr->flags & CNP_ARRAY_C_CONTIGUOUS) != 0 &&
        arr->dtype->type_num == CNP_DOUBLE &&
        out_dtype == CNP_DOUBLE &&
        (traversal.axis_none ||
         traversal.axis == arr->ndim - 1);
    const double *contiguous_values = (const double*)(
        (const char*)arr->data + arr->offset);
    if (contiguous_double_slices) {
        for (int64_t output_index = 0;
             output_index < result->size; ++output_index) {
            int64_t first = output_index * traversal.axis_length;
            double value;
            if (product) {
                value = reduction_product_contiguous_double(
                    contiguous_values + first,
                    traversal.axis_length);
            } else {
                value = 0.0;
                if (traversal.axis_length != 0)
                    value += reduction_pairwise_sum_contiguous_double(
                        contiguous_values, first,
                        traversal.axis_length,
                        CNP_REDUCTION_VALUE, 0.0);
                value = reduction_apply_sum_identity_double(value);
            }
            ((double*)result->data)[output_index] = value;
        }
        return result;
    }
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            int64_t output_index = outer * traversal.inner + inner;
            if (!product && out_dtype == CNP_FLOAT) {
                float value = 0.0f;
                if (traversal.axis_length != 0)
                    value += reduction_pairwise_sum_float(
                        &traversal, outer, inner, 0,
                        traversal.axis_length,
                        CNP_REDUCTION_VALUE, 0.0f);
                value = reduction_apply_sum_identity_float(value);
                ((float*)result->data)[output_index] = value;
                continue;
            }
            if (!product && out_dtype == CNP_DOUBLE) {
                double value = 0.0;
                if (traversal.axis_length != 0)
                    value += reduction_pairwise_sum_double(
                        &traversal, outer, inner, 0,
                        traversal.axis_length,
                        CNP_REDUCTION_VALUE, 0.0);
                value = reduction_apply_sum_identity_double(value);
                ((double*)result->data)[output_index] = value;
                continue;
            }
            if (product && out_dtype == CNP_FLOAT) {
                float accumulator = 1.0f;
                for (int64_t item = 0;
                     item < traversal.axis_length; ++item)
                    accumulator *= reduction_source_float(
                        &traversal, outer, inner, item,
                        CNP_REDUCTION_VALUE, 0.0f);
                ((float*)result->data)[output_index] = accumulator;
                continue;
            }
            if (product && out_dtype == CNP_DOUBLE) {
                double accumulator = 1.0;
                for (int64_t item = 0;
                     item < traversal.axis_length; ++item)
                    accumulator *= reduction_source_double(
                        &traversal, outer, inner, item,
                        CNP_REDUCTION_VALUE, 0.0);
                ((double*)result->data)[output_index] = accumulator;
                continue;
            }

            double accumulator = product ? 1.0 : 0.0;
            int64_t output_offset =
                output_index * result->dtype->elsize;
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item) {
                double value = reduction_source_double(
                    &traversal, outer, inner, item,
                    CNP_REDUCTION_VALUE, 0.0);
                accumulator = product
                    ? accumulator * value : accumulator + value;
                cnp_set_element_double(
                    result->data, output_offset,
                    out_dtype, accumulator);
                accumulator = cnp_get_element_double(
                    result->data, output_offset, out_dtype);
            }
            if (traversal.axis_length == 0)
                cnp_set_element_double(
                    result->data, output_offset,
                    out_dtype, accumulator);
        }
    }
    return result;
}

static CnpArray *reduction_mean(
    const CnpArray *arr, int resolved_axis, CNP_TYPE dtype) {
    CnpReductionTraversal traversal;
    reduction_traversal_init(arr, resolved_axis, &traversal);
    CNP_TYPE out_dtype = dtype != CNP_NOTYPE ? dtype :
        (arr->dtype->type_num == CNP_FLOAT ? CNP_FLOAT : CNP_DOUBLE);
    CnpArray *result = reduction_sumprod_real(
        arr, resolved_axis, out_dtype, false);
    if (!result) return NULL;
    for (int64_t index = 0; index < result->size; ++index) {
        if (out_dtype == CNP_FLOAT) {
            ((float*)result->data)[index] = reduction_divide_float(
                ((float*)result->data)[index], traversal.axis_length);
            continue;
        }
        if (out_dtype == CNP_DOUBLE) {
            ((double*)result->data)[index] = reduction_divide_double(
                ((double*)result->data)[index], traversal.axis_length);
            continue;
        }
        int64_t offset = index * result->dtype->elsize;
        double value = cnp_get_element_double(
            result->data, offset, out_dtype);
        cnp_set_element_double(
            result->data, offset, out_dtype,
            reduction_divide_double(value, traversal.axis_length));
    }
    return result;
}

static CnpArray *reduction_cumulative_real(
    const CnpArray *arr, int resolved_axis,
    CNP_TYPE out_dtype, bool product) {
    CnpReductionTraversal traversal;
    reduction_traversal_init(arr, resolved_axis, &traversal);
    int result_ndim;
    int64_t result_shape[CNP_MAXDIMS];
    if (traversal.axis_none) {
        result_ndim = 1;
        result_shape[0] = arr->size;
    } else {
        result_ndim = arr->ndim;
        for (int dimension = 0; dimension < arr->ndim; ++dimension)
            result_shape[dimension] = arr->shape[dimension];
    }
    CnpArray *result = cnp_array_new(
        result_ndim, result_shape, out_dtype, CNP_ORDER_C);
    if (!result) return NULL;
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            double accumulator = product ? 1.0 : 0.0;
            int64_t slice_base = traversal.axis_none
                ? arr->offset
                : reduction_slice_base(&traversal, outer, inner);
            if (arr->dtype->type_num == CNP_DOUBLE &&
                out_dtype == CNP_DOUBLE) {
                for (int64_t item = 0;
                     item < traversal.axis_length; ++item) {
                    int64_t source_offset = traversal.axis_none
                        ? reduction_flat_offset(arr, item)
                        : slice_base + item * traversal.axis_stride;
                    double value = *(const double*)
                        ((const char*)arr->data + source_offset);
                    accumulator = product
                        ? accumulator * value : accumulator + value;
                    int64_t destination_index = traversal.axis_none
                        ? item
                        : (outer * traversal.axis_length + item) *
                          traversal.inner + inner;
                    ((double*)result->data)[destination_index] = accumulator;
                }
                continue;
            }
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item) {
                int64_t source_offset = traversal.axis_none
                    ? reduction_flat_offset(arr, item)
                    : slice_base + item * traversal.axis_stride;
                double value = cnp_get_element_double(
                    arr->data, source_offset, arr->dtype->type_num);
                accumulator = product
                    ? accumulator * value : accumulator + value;
                int64_t destination_index = traversal.axis_none
                    ? item
                    : (outer * traversal.axis_length + item) *
                      traversal.inner + inner;
                int64_t destination_offset =
                    destination_index * result->dtype->elsize;
                cnp_set_element_double(
                    result->data, destination_offset,
                    out_dtype, accumulator);
                accumulator = cnp_get_element_double(
                    result->data, destination_offset, out_dtype);
            }
        }
    }
    return result;
}

static CnpArray *reduction_cumulative_integer(
    const CnpArray *arr, int resolved_axis, CNP_TYPE out_dtype,
    bool product) {
    CnpReductionTraversal traversal;
    reduction_traversal_init(arr, resolved_axis, &traversal);
    int result_ndim;
    int64_t result_shape[CNP_MAXDIMS];
    if (traversal.axis_none) {
        result_ndim = 1;
        result_shape[0] = arr->size;
    } else {
        result_ndim = arr->ndim;
        for (int dimension = 0; dimension < arr->ndim; ++dimension)
            result_shape[dimension] = arr->shape[dimension];
    }
    CnpArray *result = cnp_array_new(
        result_ndim, result_shape, out_dtype, CNP_ORDER_C);
    if (!result) return NULL;
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            uint64_t accumulator = product ? 1 : 0;
            int64_t slice_base = traversal.axis_none
                ? arr->offset
                : reduction_slice_base(&traversal, outer, inner);
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item) {
                int64_t source_offset = traversal.axis_none
                    ? reduction_flat_offset(arr, item)
                    : slice_base + item * traversal.axis_stride;
                uint64_t value = reduction_read_integer_bits(
                    arr, source_offset);
                if (out_dtype == CNP_BOOL) {
                    accumulator = product
                        ? (accumulator != 0 && value != 0)
                        : (accumulator != 0 || value != 0);
                } else if (product) {
                    accumulator *= value;
                } else {
                    accumulator += value;
                }
                int64_t destination_index = traversal.axis_none
                    ? item
                    : (outer * traversal.axis_length + item) *
                      traversal.inner + inner;
                reduction_store_integer_bits(
                    result, destination_index, accumulator);
            }
        }
    }
    return result;
}

static CnpArray *reduction_nan_sumprod(
    const CnpArray *arr, int resolved_axis, CNP_TYPE out_dtype,
    bool product) {
    CnpReductionTraversal traversal;
    reduction_traversal_init(arr, resolved_axis, &traversal);
    CnpArray *result = cnp_array_new(
        traversal.result_ndim, traversal.result_shape,
        out_dtype, CNP_ORDER_C);
    if (!result) return NULL;
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            int64_t output_index = outer * traversal.inner + inner;
            if (!product && out_dtype == CNP_FLOAT) {
                float value = 0.0f;
                if (traversal.axis_length != 0)
                    value += reduction_pairwise_sum_float(
                        &traversal, outer, inner, 0,
                        traversal.axis_length,
                        CNP_REDUCTION_NAN_TO_ZERO, 0.0f);
                value = reduction_apply_sum_identity_float(value);
                ((float*)result->data)[output_index] = value;
                continue;
            }
            if (!product && out_dtype == CNP_DOUBLE) {
                double value = 0.0;
                if (traversal.axis_length != 0)
                    value += reduction_pairwise_sum_double(
                        &traversal, outer, inner, 0,
                        traversal.axis_length,
                        CNP_REDUCTION_NAN_TO_ZERO, 0.0);
                value = reduction_apply_sum_identity_double(value);
                ((double*)result->data)[output_index] = value;
                continue;
            }
            if (product && out_dtype == CNP_FLOAT) {
                float accumulator = 1.0f;
                for (int64_t item = 0;
                     item < traversal.axis_length; ++item) {
                    float value = reduction_source_float(
                        &traversal, outer, inner, item,
                        CNP_REDUCTION_VALUE, 0.0f);
                    if (!isnan(value)) accumulator *= value;
                }
                ((float*)result->data)[output_index] = accumulator;
                continue;
            }
            if (product && out_dtype == CNP_DOUBLE) {
                double accumulator = 1.0;
                for (int64_t item = 0;
                     item < traversal.axis_length; ++item) {
                    double value = reduction_source_double(
                        &traversal, outer, inner, item,
                        CNP_REDUCTION_VALUE, 0.0);
                    if (!isnan(value)) accumulator *= value;
                }
                ((double*)result->data)[output_index] = accumulator;
                continue;
            }

            double accumulator = product ? 1.0 : 0.0;
            int64_t output_offset =
                output_index * result->dtype->elsize;
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item) {
                double value = reduction_source_double(
                    &traversal, outer, inner, item,
                    CNP_REDUCTION_VALUE, 0.0);
                if (!isnan(value))
                    accumulator = product
                        ? accumulator * value : accumulator + value;
                cnp_set_element_double(
                    result->data, output_offset,
                    out_dtype, accumulator);
                accumulator = cnp_get_element_double(
                    result->data, output_offset, out_dtype);
            }
            if (traversal.axis_length == 0)
                cnp_set_element_double(
                    result->data, output_offset,
                    out_dtype, accumulator);
        }
    }
    return result;
}

static CnpArray *reduction_any_all(
    const CnpArray *arr, int resolved_axis, bool all) {
    CnpReductionTraversal traversal;
    reduction_traversal_init(arr, resolved_axis, &traversal);
    CnpArray *result = cnp_array_new(
        traversal.result_ndim, traversal.result_shape,
        CNP_BOOL, CNP_ORDER_C);
    if (!result) return NULL;
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            bool accumulator = all;
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item) {
                int64_t source_offset = reduction_source_offset(
                    &traversal, outer, inner, item);
                bool value = cnp_get_element_double(
                    arr->data, source_offset,
                    arr->dtype->type_num) != 0.0;
                accumulator = all
                    ? accumulator && value : accumulator || value;
            }
            ((int8_t*)result->data)[
                outer * traversal.inner + inner] = accumulator;
        }
    }
    return result;
}

static CnpArray *reduction_nan_cumulative(
    const CnpArray *arr, int resolved_axis, CNP_TYPE out_dtype,
    bool product) {
    CnpReductionTraversal traversal;
    reduction_traversal_init(arr, resolved_axis, &traversal);
    int result_ndim = traversal.axis_none ? 1 : arr->ndim;
    int64_t result_shape[CNP_MAXDIMS];
    if (traversal.axis_none) {
        result_shape[0] = arr->size;
    } else {
        for (int dimension = 0; dimension < arr->ndim; ++dimension)
            result_shape[dimension] = arr->shape[dimension];
    }
    CnpArray *result = cnp_array_new(
        result_ndim, result_shape, out_dtype, CNP_ORDER_C);
    if (!result) return NULL;
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            double accumulator = product ? 1.0 : 0.0;
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item) {
                int64_t source_offset = reduction_source_offset(
                    &traversal, outer, inner, item);
                double value = cnp_get_element_double(
                    arr->data, source_offset, arr->dtype->type_num);
                if (!isnan(value))
                    accumulator = product
                        ? accumulator * value : accumulator + value;
                int64_t destination_index = traversal.axis_none
                    ? item
                    : (outer * traversal.axis_length + item) *
                      traversal.inner + inner;
                cnp_set_element_double(
                    result->data,
                    destination_index * result->dtype->elsize,
                    out_dtype, accumulator);
                if (out_dtype != CNP_DOUBLE)
                    accumulator = cnp_get_element_double(
                        result->data,
                        destination_index * result->dtype->elsize,
                        out_dtype);
            }
        }
    }
    return result;
}

static CnpArray *reduction_nan_mean_or_deviation(
    const CnpArray *arr, int resolved_axis, int ddof,
    CNP_TYPE dtype, bool deviation, bool standard_deviation) {
    CnpReductionTraversal traversal;
    reduction_traversal_init(arr, resolved_axis, &traversal);
    CNP_TYPE out_dtype = dtype != CNP_NOTYPE ? dtype :
        (arr->dtype->type_num == CNP_FLOAT ? CNP_FLOAT : CNP_DOUBLE);
    CnpArray *result = cnp_array_new(
        traversal.result_ndim, traversal.result_shape,
        out_dtype, CNP_ORDER_C);
    if (!result) return NULL;
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            int64_t count = 0;
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item) {
                double value = reduction_source_double(
                    &traversal, outer, inner, item,
                    CNP_REDUCTION_VALUE, 0.0);
                if (!isnan(value)) ++count;
            }
            int64_t output_index = outer * traversal.inner + inner;
            if (out_dtype == CNP_FLOAT) {
                float total = 0.0f;
                if (traversal.axis_length != 0)
                    total += reduction_pairwise_sum_float(
                        &traversal, outer, inner, 0,
                        traversal.axis_length,
                        CNP_REDUCTION_NAN_TO_ZERO, 0.0f);
                total = reduction_apply_sum_identity_float(total);
                float value = count == 0
                    ? NAN : reduction_divide_float(total, count);
                if (deviation) {
                    float squared_deviation = 0.0f;
                    if (traversal.axis_length != 0)
                        squared_deviation += reduction_pairwise_sum_float(
                            &traversal, outer, inner, 0,
                            traversal.axis_length,
                            CNP_REDUCTION_NAN_SQUARED_DEVIATION,
                            value);
                    squared_deviation = reduction_apply_sum_identity_float(
                        squared_deviation);
                    value = count - ddof > 0
                        ? reduction_divide_float(
                            squared_deviation, count - ddof)
                        : NAN;
                    if (standard_deviation) value = sqrtf(value);
                }
                ((float*)result->data)[output_index] = value;
                continue;
            }
            if (out_dtype == CNP_DOUBLE) {
                double total = 0.0;
                if (traversal.axis_length != 0)
                    total += reduction_pairwise_sum_double(
                        &traversal, outer, inner, 0,
                        traversal.axis_length,
                        CNP_REDUCTION_NAN_TO_ZERO, 0.0);
                total = reduction_apply_sum_identity_double(total);
                double value = count == 0
                    ? NAN : reduction_divide_double(total, count);
                if (deviation) {
                    double squared_deviation = 0.0;
                    if (traversal.axis_length != 0)
                        squared_deviation += reduction_pairwise_sum_double(
                            &traversal, outer, inner, 0,
                            traversal.axis_length,
                            CNP_REDUCTION_NAN_SQUARED_DEVIATION,
                            value);
                    squared_deviation = reduction_apply_sum_identity_double(
                        squared_deviation);
                    value = count - ddof > 0
                        ? reduction_divide_double(
                            squared_deviation, count - ddof)
                        : NAN;
                    if (standard_deviation) value = sqrt(value);
                }
                ((double*)result->data)[output_index] = value;
                continue;
            }

            double total = 0.0;
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item) {
                double item_value = reduction_source_double(
                    &traversal, outer, inner, item,
                    CNP_REDUCTION_VALUE, 0.0);
                if (!isnan(item_value)) total += item_value;
            }
            double value = count == 0
                ? NAN : reduction_divide_double(total, count);
            if (deviation) {
                double squared_deviation = 0.0;
                for (int64_t item = 0;
                     item < traversal.axis_length; ++item) {
                    double item_value = reduction_source_double(
                        &traversal, outer, inner, item,
                        CNP_REDUCTION_VALUE, 0.0);
                    if (!isnan(item_value)) {
                        double difference = item_value - value;
                        squared_deviation += difference * difference;
                    }
                }
                value = count - ddof > 0
                    ? reduction_divide_double(
                        squared_deviation, count - ddof) : NAN;
                if (standard_deviation) value = sqrt(value);
            }
            cnp_set_element_double(
                result->data, output_index * result->dtype->elsize,
                out_dtype, value);
        }
    }
    return result;
}

static bool reduction_offset_signbit(
    const CnpArray *arr, int64_t offset) {
    const char *source = (const char*)arr->data + offset;
    if (arr->dtype->type_num == CNP_FLOAT) {
        uint32_t bits;
        memcpy(&bits, source, sizeof(bits));
        return (bits >> 31) != 0;
    }
    if (arr->dtype->type_num == CNP_DOUBLE) {
        uint64_t bits;
        memcpy(&bits, source, sizeof(bits));
        return (bits >> 63) != 0;
    }
    return signbit(cnp_get_element_double(
        arr->data, offset, arr->dtype->type_num)) != 0;
}

static int64_t reduction_nan_extrema_scalar_select(
    const CnpArray *arr,
    int64_t left_offset, int64_t right_offset,
    bool maximum) {
    double left = cnp_get_element_double(
        arr->data, left_offset, arr->dtype->type_num);
    double right = cnp_get_element_double(
        arr->data, right_offset, arr->dtype->type_num);
    if (isnan(left)) return right_offset;
    if (isnan(right)) return left_offset;
    if (maximum) {
        if (left > right) return left_offset;
        if (right > left) return right_offset;
        if (left == 0.0 && right == 0.0) {
            bool left_negative = reduction_offset_signbit(
                arr, left_offset);
            bool right_negative = reduction_offset_signbit(
                arr, right_offset);
            return left_negative && !right_negative
                ? right_offset : left_offset;
        }
    } else {
        if (left < right) return left_offset;
        if (right < left) return right_offset;
        if (left == 0.0 && right == 0.0) {
            bool left_negative = reduction_offset_signbit(
                arr, left_offset);
            bool right_negative = reduction_offset_signbit(
                arr, right_offset);
            return !left_negative && right_negative
                ? right_offset : left_offset;
        }
    }
    return left_offset;
}

static int64_t reduction_nan_extrema_vector_select(
    const CnpArray *arr,
    int64_t left_offset, int64_t right_offset,
    bool maximum) {
    double left = cnp_get_element_double(
        arr->data, left_offset, arr->dtype->type_num);
    double right = cnp_get_element_double(
        arr->data, right_offset, arr->dtype->type_num);
    if (isnan(left)) return right_offset;
    if (isnan(right)) return left_offset;
    if (maximum)
        return left > right ? left_offset : right_offset;
    return left < right ? left_offset : right_offset;
}

static int64_t reduction_nan_extrema_vector_reduce(
    const CnpArray *arr, const int64_t lane_offsets[8],
    int lanes, bool maximum) {
    if (lanes == 8) {
        int64_t pair_04 = reduction_nan_extrema_vector_select(
            arr, lane_offsets[0], lane_offsets[4], maximum);
        int64_t pair_26 = reduction_nan_extrema_vector_select(
            arr, lane_offsets[2], lane_offsets[6], maximum);
        int64_t pair_15 = reduction_nan_extrema_vector_select(
            arr, lane_offsets[1], lane_offsets[5], maximum);
        int64_t pair_37 = reduction_nan_extrema_vector_select(
            arr, lane_offsets[3], lane_offsets[7], maximum);
        int64_t even = reduction_nan_extrema_vector_select(
            arr, pair_04, pair_26, maximum);
        int64_t odd = reduction_nan_extrema_vector_select(
            arr, pair_15, pair_37, maximum);
        return reduction_nan_extrema_vector_select(
            arr, even, odd, maximum);
    }
    int64_t even = reduction_nan_extrema_vector_select(
        arr, lane_offsets[0], lane_offsets[2], maximum);
    int64_t odd = reduction_nan_extrema_vector_select(
        arr, lane_offsets[1], lane_offsets[3], maximum);
    return reduction_nan_extrema_vector_select(
        arr, even, odd, maximum);
}

typedef struct {
    const CnpReductionTraversal *traversal;
    int dimension_count;
    int dimensions[CNP_MAXDIMS];
    int reduced_position;
    int64_t horizontal_inner_stride;
} CnpReductionIteratorRun;

static int reduction_keeporder_compare_dimensions(
    const CnpArray *arr,
    int left_dimension, int right_dimension) {
    int64_t left_stride = llabs(arr->strides[left_dimension]);
    int64_t right_stride = llabs(arr->strides[right_dimension]);
    if (left_stride == 0 || right_stride == 0 ||
        left_stride == right_stride)
        return 0;
    return left_stride < right_stride ? -1 : 1;
}

static int reduction_keeporder_dimensions(
    const CnpArray *arr, int excluded_dimension,
    int dimensions[CNP_MAXDIMS]) {
    int dimension_count = 0;
    for (int dimension = arr->ndim - 1;
         dimension >= 0; --dimension) {
        if (dimension != excluded_dimension &&
            arr->shape[dimension] > 1)
            dimensions[dimension_count++] = dimension;
    }
    for (int left = 1; left < dimension_count; ++left) {
        int dimension = dimensions[left];
        int insert = left;
        for (int previous = left - 1;
             previous >= 0; --previous) {
            int comparison = reduction_keeporder_compare_dimensions(
                arr, dimension, dimensions[previous]);
            if (comparison < 0)
                insert = previous;
            else if (comparison > 0)
                break;
        }
        if (insert != left) {
            for (int position = left;
                 position > insert; --position)
                dimensions[position] = dimensions[position - 1];
            dimensions[insert] = dimension;
        }
    }
    return dimension_count;
}

static void reduction_iterator_run_init(
    const CnpReductionTraversal *traversal,
    CnpReductionIteratorRun *run) {
    const CnpArray *arr = traversal->array;
    run->traversal = traversal;
    run->dimension_count = reduction_keeporder_dimensions(
        arr, -1, run->dimensions);
    run->reduced_position = -1;
    run->horizontal_inner_stride = 0;

    if (!traversal->axis_none) {
        for (int position = 0;
             position < run->dimension_count; ++position) {
            if (run->dimensions[position] == traversal->axis) {
                run->reduced_position = position;
                break;
            }
        }
        run->horizontal_inner_stride = traversal->axis_stride;
        if (run->reduced_position == 0 &&
            run->dimension_count > 2) {
            int fastest_output = run->dimensions[1];
            int64_t coalesced_size = arr->shape[fastest_output];
            for (int position = 2;
                 position < run->dimension_count; ++position) {
                int dimension = run->dimensions[position];
                if (llabs(arr->strides[dimension]) !=
                    llabs(arr->strides[fastest_output]) * coalesced_size) {
                    run->horizontal_inner_stride = arr->dtype->elsize;
                    break;
                }
                coalesced_size *= arr->shape[dimension];
            }
        }
        return;
    }

    if (run->dimension_count == 0) return;
    int fastest = run->dimensions[0];
    int64_t coalesced_size = arr->shape[fastest];
    for (int position = 1;
         position < run->dimension_count; ++position) {
        int dimension = run->dimensions[position];
        if (arr->strides[dimension] !=
            arr->strides[fastest] * coalesced_size) {
            run->horizontal_inner_stride = arr->dtype->elsize;
            return;
        }
        coalesced_size *= arr->shape[dimension];
    }
    run->horizontal_inner_stride = arr->strides[fastest];
}

static int64_t reduction_iterator_binary_inner_stride(
    const CnpReductionIteratorRun *run) {
    const CnpArray *arr = run->traversal->array;
    if (run->reduced_position <= 0)
        return arr->dtype->elsize;
    int fastest = run->dimensions[0];
    int64_t inner_stride = arr->strides[fastest];
    int64_t inner_shape = arr->shape[fastest];
    for (int position = 1;
         position < run->reduced_position; ++position) {
        int dimension = run->dimensions[position];
        if (inner_stride * inner_shape != arr->strides[dimension])
            return arr->dtype->elsize;
        inner_shape *= arr->shape[dimension];
    }
    return inner_stride;
}

static int64_t reduction_keeporder_flat_offset(
    const CnpReductionIteratorRun *run, int64_t flat) {
    const CnpArray *arr = run->traversal->array;
    int64_t offset = arr->offset;
    for (int position = 0;
         position < run->dimension_count; ++position) {
        int dimension = run->dimensions[position];
        int64_t coordinate = flat % arr->shape[dimension];
        flat /= arr->shape[dimension];
        offset += coordinate * arr->strides[dimension];
    }
    return offset;
}

/* NumPy 1.25 reduction NpyIter uses an 8192-element external-loop buffer. */
#define CNP_REDUCTION_ITERATOR_BUFFER_SIZE 8192

static int64_t reduction_iterator_horizontal_chunk_end(
    int64_t first_item, int64_t item_count) {
    int64_t chunk_end =
        (first_item / CNP_REDUCTION_ITERATOR_BUFFER_SIZE + 1) *
        CNP_REDUCTION_ITERATOR_BUFFER_SIZE;
    return item_count < chunk_end ? item_count : chunk_end;
}

static int64_t reduction_nan_extrema_item_offset(
    const CnpReductionIteratorRun *run,
    int64_t outer, int64_t inner, int64_t item) {
    const CnpReductionTraversal *traversal = run->traversal;
    if (traversal->axis_none)
        return reduction_keeporder_flat_offset(run, item);
    return reduction_source_offset(
        traversal, outer, inner, item);
}

static int64_t reduction_nan_extrema_vector_run(
    const CnpReductionIteratorRun *run,
    int64_t outer, int64_t inner,
    int64_t first_item, int64_t end_item,
    int64_t selected_offset, bool maximum) {
    const CnpReductionTraversal *traversal = run->traversal;
    const CnpArray *arr = traversal->array;
    int lanes = arr->dtype->type_num == CNP_FLOAT ? 8 : 4;
    int64_t lane_offsets[8];
    for (int lane = 0; lane < lanes; ++lane)
        lane_offsets[lane] = selected_offset;
    int64_t item = first_item;
    for (; item + lanes <= end_item; item += lanes) {
        for (int lane = 0; lane < lanes; ++lane) {
            int64_t source_offset = reduction_nan_extrema_item_offset(
                run, outer, inner, item + lane);
            lane_offsets[lane] = reduction_nan_extrema_vector_select(
                arr, lane_offsets[lane], source_offset, maximum);
        }
    }
    selected_offset = reduction_nan_extrema_vector_reduce(
        arr, lane_offsets, lanes, maximum);
    for (; item < end_item; ++item) {
        int64_t source_offset = reduction_nan_extrema_item_offset(
            run, outer, inner, item);
        selected_offset = reduction_nan_extrema_scalar_select(
            arr, selected_offset, source_offset, maximum);
    }
    return selected_offset;
}

static int64_t reduction_nan_extrema_scalar_unrolled_run(
    const CnpReductionIteratorRun *run,
    int64_t outer, int64_t inner,
    int64_t first_item, int64_t end_item,
    int64_t selected_offset, bool maximum) {
    const CnpArray *arr = run->traversal->array;
    int64_t lane_offsets[8];
    for (int lane = 0; lane < 8; ++lane)
        lane_offsets[lane] = selected_offset;
    int64_t item = first_item;
    for (; item + 8 <= end_item; item += 8) {
        for (int lane = 0; lane < 8; ++lane) {
            int64_t source_offset = reduction_nan_extrema_item_offset(
                run, outer, inner, item + lane);
            lane_offsets[lane] = reduction_nan_extrema_scalar_select(
                arr, lane_offsets[lane], source_offset, maximum);
        }
    }
    selected_offset = lane_offsets[0];
    for (int lane = 1; lane < 8; ++lane)
        selected_offset = reduction_nan_extrema_scalar_select(
            arr, selected_offset, lane_offsets[lane], maximum);
    for (; item < end_item; ++item) {
        int64_t source_offset = reduction_nan_extrema_item_offset(
            run, outer, inner, item);
        selected_offset = reduction_nan_extrema_scalar_select(
            arr, selected_offset, source_offset, maximum);
    }
    return selected_offset;
}

static void reduction_buffered_output_run_position(
    const CnpReductionIteratorRun *iterator_run,
    int64_t outer, int64_t inner,
    int64_t *run_position, int64_t *run_length) {
    const CnpReductionTraversal *traversal =
        iterator_run->traversal;
    const CnpArray *arr = traversal->array;
    int64_t coordinates[CNP_MAXDIMS] = {0};
    int64_t outer_index = outer;
    for (int dimension = traversal->axis - 1;
         dimension >= 0; --dimension) {
        coordinates[dimension] =
            outer_index % arr->shape[dimension];
        outer_index /= arr->shape[dimension];
    }
    int64_t inner_index = inner;
    for (int dimension = arr->ndim - 1;
         dimension > traversal->axis; --dimension) {
        coordinates[dimension] =
            inner_index % arr->shape[dimension];
        inner_index /= arr->shape[dimension];
    }

    *run_position = 0;
    *run_length = 1;
    for (int position = 0;
         position < iterator_run->reduced_position; ++position) {
        int dimension = iterator_run->dimensions[position];
        *run_position += coordinates[dimension] * *run_length;
        *run_length *= arr->shape[dimension];
    }
}

static int64_t reduction_nan_extrema_slice(
    const CnpReductionIteratorRun *iterator_run,
    int64_t outer, int64_t inner, bool maximum) {
    const CnpReductionTraversal *traversal =
        iterator_run->traversal;
    const CnpArray *arr = traversal->array;
    int64_t first_offset = reduction_nan_extrema_item_offset(
        iterator_run, outer, inner, 0);
    bool horizontal_run = traversal->axis_none ||
        iterator_run->reduced_position == 0;
    if (!horizontal_run) {
        int lanes = arr->dtype->type_num == CNP_FLOAT ? 8 : 4;
        int64_t run_position;
        int64_t run_length;
        reduction_buffered_output_run_position(
            iterator_run, outer, inner,
            &run_position, &run_length);
        int64_t chunk_position =
            run_position % CNP_REDUCTION_ITERATOR_BUFFER_SIZE;
        int64_t chunk_start = run_position - chunk_position;
        int64_t chunk_length = run_length - chunk_start;
        if (chunk_length > CNP_REDUCTION_ITERATOR_BUFFER_SIZE)
            chunk_length = CNP_REDUCTION_ITERATOR_BUFFER_SIZE;
        int64_t vectorized_chunk_length =
            chunk_length - chunk_length % lanes;
        int64_t binary_inner_stride =
            reduction_iterator_binary_inner_stride(iterator_run);
        bool float32_negative_inner_stride =
            arr->dtype->type_num == CNP_FLOAT &&
            binary_inner_stride < 0;
        bool vector_output_order =
            (arr->dtype->type_num == CNP_FLOAT ||
             arr->dtype->type_num == CNP_DOUBLE) &&
            !float32_negative_inner_stride &&
            chunk_position < vectorized_chunk_length;
        int64_t selected_offset = first_offset;
        for (int64_t item = 1;
             item < traversal->axis_length; ++item) {
            int64_t source_offset = reduction_nan_extrema_item_offset(
                iterator_run, outer, inner, item);
            selected_offset = vector_output_order
                ? reduction_nan_extrema_vector_select(
                    arr, selected_offset, source_offset, maximum)
                : reduction_nan_extrema_scalar_select(
                    arr, selected_offset, source_offset, maximum);
        }
        return selected_offset;
    }

    bool vector_horizontal_order =
        (arr->dtype->type_num == CNP_FLOAT ||
         arr->dtype->type_num == CNP_DOUBLE) &&
        iterator_run->horizontal_inner_stride == arr->dtype->elsize;
    int64_t selected_offset = first_offset;
    int64_t first_item = 1;
    do {
        int64_t end_item = reduction_iterator_horizontal_chunk_end(
            first_item, traversal->axis_length);
        selected_offset = vector_horizontal_order
            ? reduction_nan_extrema_vector_run(
                iterator_run, outer, inner, first_item, end_item,
                selected_offset, maximum)
            : reduction_nan_extrema_scalar_unrolled_run(
                iterator_run, outer, inner, first_item, end_item,
                selected_offset, maximum);
        first_item = end_item;
    } while (first_item < traversal->axis_length);
    return selected_offset;
}

static CnpArray *reduction_nan_extrema(
    const CnpArray *arr, int resolved_axis, bool maximum,
    const char *function_name) {
    CnpReductionTraversal traversal;
    reduction_traversal_init(arr, resolved_axis, &traversal);
    CnpArray *result = cnp_array_new(
        traversal.result_ndim, traversal.result_shape,
        arr->dtype->type_num, CNP_ORDER_C);
    if (!result) return NULL;
    if (result->size != 0 && traversal.axis_length == 0) {
        cnp_array_free(result);
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "zero-size reduction has no identity");
        return NULL;
    }
    CnpReductionIteratorRun iterator_run;
    reduction_iterator_run_init(&traversal, &iterator_run);
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            int64_t output_index = outer * traversal.inner + inner;
            int64_t selected_offset = reduction_nan_extrema_slice(
                &iterator_run, outer, inner, maximum);
            memcpy(
                (char*)result->data +
                    output_index * result->dtype->elsize,
                (const char*)arr->data + selected_offset,
                result->dtype->elsize);
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_sum(const CnpArray *arr, int axis, CNP_TYPE dtype) {
    CnpArray *result = cnp_sum_v2(
        arr, axis, axis == CNP_AXIS_NONE, dtype);
    if (!result) cnp_relabel_error("cnp_sum");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_sum_v2(
    const CnpArray *arr, int axis, bool axis_none, CNP_TYPE dtype) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_sum_v2", &resolved_axis)) return NULL;
    CNP_TYPE out_dtype = dtype == CNP_NOTYPE
        ? reduction_default_sum_dtype(arr->dtype->type_num) : dtype;
    if (reduction_type_is_integer(arr->dtype->type_num) &&
        reduction_type_is_integer(out_dtype))
        return reduction_sum_integer(arr, resolved_axis, out_dtype);
    return reduction_sumprod_real(
        arr, resolved_axis, out_dtype, false);
}

CNP_API CnpArray* CNP_CALL cnp_prod(const CnpArray *arr, int axis, CNP_TYPE dtype) {
    CnpArray *result = cnp_prod_v2(
        arr, axis, axis == CNP_AXIS_NONE, dtype);
    if (!result) cnp_relabel_error("cnp_prod");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_prod_v2(
    const CnpArray *arr, int axis, bool axis_none, CNP_TYPE dtype) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_prod_v2", &resolved_axis)) return NULL;
    CNP_TYPE out_dtype = dtype == CNP_NOTYPE
        ? reduction_default_sum_dtype(arr->dtype->type_num) : dtype;
    if (reduction_type_is_integer(arr->dtype->type_num) &&
        reduction_type_is_integer(out_dtype))
        return reduction_prod_integer(arr, resolved_axis, out_dtype);
    return reduction_sumprod_real(
        arr, resolved_axis, out_dtype, true);
}

CNP_API CnpArray* CNP_CALL cnp_max(const CnpArray *arr, int axis) {
    CnpArray *result = cnp_max_v2(arr, axis, axis == CNP_AXIS_NONE);
    if (!result) cnp_relabel_error("cnp_max");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_min(const CnpArray *arr, int axis) {
    CnpArray *result = cnp_min_v2(arr, axis, axis == CNP_AXIS_NONE);
    if (!result) cnp_relabel_error("cnp_min");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_amax(const CnpArray *arr, int axis) {
    CnpArray *result = cnp_max(arr, axis);
    if (!result) cnp_relabel_error("cnp_amax");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_amin(const CnpArray *arr, int axis) {
    CnpArray *result = cnp_min(arr, axis);
    if (!result) cnp_relabel_error("cnp_amin");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_any(const CnpArray *arr, int axis) {
    CnpArray *result = cnp_any_v2(arr, axis, axis == CNP_AXIS_NONE);
    if (!result) cnp_relabel_error("cnp_any");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_all(const CnpArray *arr, int axis) {
    CnpArray *result = cnp_all_v2(arr, axis, axis == CNP_AXIS_NONE);
    if (!result) cnp_relabel_error("cnp_all");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_any_v2(
    const CnpArray *arr, int axis, bool axis_none) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_any_v2", &resolved_axis)) return NULL;
    return reduction_any_all(arr, resolved_axis, false);
}

CNP_API CnpArray* CNP_CALL cnp_all_v2(
    const CnpArray *arr, int axis, bool axis_none) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_all_v2", &resolved_axis)) return NULL;
    return reduction_any_all(arr, resolved_axis, true);
}

/* Mean */
CNP_API CnpArray* CNP_CALL cnp_mean(const CnpArray *arr, int axis, CNP_TYPE dtype) {
    CnpArray *result = cnp_mean_v2(
        arr, axis, axis == CNP_AXIS_NONE, dtype);
    if (!result) cnp_relabel_error("cnp_mean");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_max_v2(
    const CnpArray *arr, int axis, bool axis_none) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_max_v2", &resolved_axis)) return NULL;
    return reduction_extrema(arr, resolved_axis, true, "cnp_max_v2");
}

CNP_API CnpArray* CNP_CALL cnp_min_v2(
    const CnpArray *arr, int axis, bool axis_none) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_min_v2", &resolved_axis)) return NULL;
    return reduction_extrema(arr, resolved_axis, false, "cnp_min_v2");
}

CNP_API CnpArray* CNP_CALL cnp_mean_v2(
    const CnpArray *arr, int axis, bool axis_none, CNP_TYPE dtype) {
    int resolved_axis;
    if (!cnp_reduction_resolve_axis_strict_scalar(
            arr, axis, axis_none, "cnp_mean_v2", &resolved_axis)) return NULL;
    return reduction_mean(arr, resolved_axis, dtype);
}

/* Variance */
CNP_API CnpArray* CNP_CALL cnp_var(const CnpArray *arr, int axis, int ddof, CNP_TYPE dtype) {
    CnpArray *result = cnp_var_v2(
        arr, axis, axis == CNP_AXIS_NONE, ddof, dtype);
    if (!result) cnp_relabel_error("cnp_var");
    return result;
}

/* Standard deviation */
CNP_API CnpArray* CNP_CALL cnp_std(const CnpArray *arr, int axis, int ddof, CNP_TYPE dtype) {
    CnpArray *result = cnp_std_v2(
        arr, axis, axis == CNP_AXIS_NONE, ddof, dtype);
    if (!result) cnp_relabel_error("cnp_std");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_var_v2(
    const CnpArray *arr, int axis, bool axis_none,
    int ddof, CNP_TYPE dtype) {
    int resolved_axis;
    if (!cnp_reduction_resolve_axis_strict_scalar(
            arr, axis, axis_none, "cnp_var_v2", &resolved_axis)) return NULL;
    return reduction_variance(arr, resolved_axis, ddof, dtype, false);
}

CNP_API CnpArray* CNP_CALL cnp_std_v2(
    const CnpArray *arr, int axis, bool axis_none,
    int ddof, CNP_TYPE dtype) {
    int resolved_axis;
    if (!cnp_reduction_resolve_axis_strict_scalar(
            arr, axis, axis_none, "cnp_std_v2", &resolved_axis)) return NULL;
    return reduction_variance(arr, resolved_axis, ddof, dtype, true);
}

/* Cumulative sum */
CNP_API CnpArray* CNP_CALL cnp_cumsum(const CnpArray *arr, int axis, CNP_TYPE dtype) {
    CnpArray *result = cnp_cumsum_v2(
        arr, axis, axis == CNP_AXIS_NONE, dtype);
    if (!result) cnp_relabel_error("cnp_cumsum");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_cumsum_v2(
    const CnpArray *arr, int axis, bool axis_none, CNP_TYPE dtype) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_cumsum_v2", &resolved_axis)) return NULL;
    CNP_TYPE out_dtype = dtype == CNP_NOTYPE
        ? reduction_default_sum_dtype(arr->dtype->type_num) : dtype;
    if (reduction_type_is_integer(arr->dtype->type_num) &&
        reduction_type_is_integer(out_dtype))
        return reduction_cumulative_integer(
            arr, resolved_axis, out_dtype, false);
    return reduction_cumulative_real(
        arr, resolved_axis, out_dtype, false);
}

/* Cumulative product */
CNP_API CnpArray* CNP_CALL cnp_cumprod(const CnpArray *arr, int axis, CNP_TYPE dtype) {
    CnpArray *result = cnp_cumprod_v2(
        arr, axis, axis == CNP_AXIS_NONE, dtype);
    if (!result) cnp_relabel_error("cnp_cumprod");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_cumprod_v2(
    const CnpArray *arr, int axis, bool axis_none, CNP_TYPE dtype) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_cumprod_v2", &resolved_axis)) return NULL;
    CNP_TYPE out_dtype = dtype == CNP_NOTYPE
        ? reduction_default_sum_dtype(arr->dtype->type_num) : dtype;
    if (reduction_type_is_integer(arr->dtype->type_num) &&
        reduction_type_is_integer(out_dtype))
        return reduction_cumulative_integer(
            arr, resolved_axis, out_dtype, true);
    return reduction_cumulative_real(
        arr, resolved_axis, out_dtype, true);
}

/* Argmax / Argmin */
CNP_API CnpArray* CNP_CALL cnp_argmax(const CnpArray *arr, int axis) {
    CnpArray *result = cnp_argmax_v2(
        arr, axis, axis == CNP_AXIS_NONE);
    if (!result) cnp_relabel_error("cnp_argmax");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_argmin(const CnpArray *arr, int axis) {
    CnpArray *result = cnp_argmin_v2(
        arr, axis, axis == CNP_AXIS_NONE);
    if (!result) cnp_relabel_error("cnp_argmin");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_argmax_v2(
    const CnpArray *arr, int axis, bool axis_none) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_argmax_v2", &resolved_axis)) return NULL;
    return reduction_arg_extrema(
        arr, resolved_axis, true, "cnp_argmax_v2");
}

CNP_API CnpArray* CNP_CALL cnp_argmin_v2(
    const CnpArray *arr, int axis, bool axis_none) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_argmin_v2", &resolved_axis)) return NULL;
    return reduction_arg_extrema(
        arr, resolved_axis, false, "cnp_argmin_v2");
}

/* Peak to peak */
CNP_API CnpArray* CNP_CALL cnp_ptp(const CnpArray *arr, int axis) {
    CnpArray *result = cnp_ptp_v2(
        arr, axis, axis == CNP_AXIS_NONE);
    if (!result) cnp_relabel_error("cnp_ptp");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_ptp_v2(
    const CnpArray *arr, int axis, bool axis_none) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none,
            "cnp_ptp_v2", &resolved_axis)) return NULL;
    CnpArray *maximum = reduction_extrema(
        arr, resolved_axis, true, "cnp_ptp_v2");
    if (!maximum) return NULL;
    CnpArray *minimum = reduction_extrema(
        arr, resolved_axis, false, "cnp_ptp_v2");
    if (!minimum) {
        cnp_array_free(maximum);
        return NULL;
    }
    if (cnp_type_is_integer(arr->dtype->type_num)) {
        for (int64_t index = 0; index < maximum->size; ++index) {
            int64_t offset = index * maximum->dtype->elsize;
            uint64_t maximum_bits = reduction_read_integer_bits(
                maximum, offset);
            uint64_t minimum_bits = reduction_read_integer_bits(
                minimum, offset);
            reduction_store_integer_bits(
                maximum, index, maximum_bits - minimum_bits);
        }
        cnp_array_free(minimum);
        return maximum;
    }
    CnpArray *result = cnp_subtract(maximum, minimum);
    cnp_array_free(maximum);
    cnp_array_free(minimum);
    if (!result) cnp_relabel_error("cnp_ptp_v2");
    return result;
}

/* Trace */
CNP_API CnpArray* CNP_CALL cnp_trace(const CnpArray *arr, int offset, int axis1, int axis2) {
    const char *function_name = "cnp_trace";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must not be null");
        return NULL;
    }
    if (arr->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source array must have at least two dimensions");
        return NULL;
    }
    int first_axis = cnp_normalize_axis(axis1, arr->ndim);
    int second_axis = cnp_normalize_axis(axis2, arr->ndim);
    if (first_axis < 0 || second_axis < 0 || first_axis == second_axis) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "trace axes %d and %d must be distinct valid axes",
            axis1, axis2);
        return NULL;
    }
    CnpArray *diag = cnp_diagonal(
        arr, offset, first_axis, second_axis);
    if (!diag) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpArray *result;
    if (cnp_type_is_complex(diag->dtype->type_num)) {
        const CnpArray *operands[1] = {diag};
        result = cnp_einsum_generic(
            "...i->...", 1, operands, function_name);
    } else {
        result = cnp_sum_v2(diag, -1, false, CNP_NOTYPE);
    }
    cnp_array_free(diag);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* Scalar reductions */
static bool scalar_reduction_validate_array(
        const CnpArray *arr, const char *function_name) {
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "source array must not be null");
        return false;
    }
    if (!arr->dtype ||
            !(arr->dtype->type_num == CNP_BOOL ||
              cnp_type_is_integer(arr->dtype->type_num) ||
              cnp_type_is_float(arr->dtype->type_num) ||
              cnp_type_is_complex(arr->dtype->type_num))) {
        cnp_set_error(CNP_ERR_TYPE, function_name,
                      "source array must have a numeric dtype");
        return false;
    }
    if (cnp_type_is_complex(arr->dtype->type_num)) {
        cnp_set_error(CNP_ERR_TYPE, function_name,
                      "double-return scalar ABI cannot represent complex results");
        return false;
    }
    if (arr->ndim < 0 || arr->ndim > CNP_MAXDIMS ||
            arr->size < 0 || arr->offset < 0 ||
            (arr->ndim > 0 && (!arr->shape || !arr->strides))) {
        cnp_set_error(CNP_ERR_SHAPE, function_name,
                      "source array has invalid shape metadata");
        return false;
    }
    int64_t expected_size = 1;
    for (int dimension = 0; dimension < arr->ndim; ++dimension) {
        int64_t extent = arr->shape[dimension];
        if (extent < 0 ||
                (expected_size != 0 &&
                 extent > INT64_MAX / expected_size)) {
            cnp_set_error(CNP_ERR_SHAPE, function_name,
                          "source array has invalid shape metadata");
            return false;
        }
        expected_size *= extent;
    }
    if (expected_size != arr->size) {
        cnp_set_error(CNP_ERR_SHAPE, function_name,
                      "source array has invalid shape metadata");
        return false;
    }
    if (arr->size > 0 && !arr->data) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "source array requires a data buffer");
        return false;
    }
    return true;
}

CNP_API double CNP_CALL cnp_sum_scalar(const CnpArray *arr) {
    const char *function_name = "cnp_sum_scalar";
    if (!scalar_reduction_validate_array(arr, function_name)) return NAN;
    CnpArray *result = cnp_sum_v2(
        arr, 0, true, CNP_NOTYPE);
    if (!result) {
        cnp_relabel_error(function_name);
        return NAN;
    }
    double value = cnp_get_element_double(
        result->data, result->offset, result->dtype->type_num);
    cnp_array_free(result);
    return value;
}

CNP_API double CNP_CALL cnp_prod_scalar(const CnpArray *arr) {
    if (!scalar_reduction_validate_array(arr, "cnp_prod_scalar")) return NAN;
    double acc = 1.0;
    int64_t coords[CNP_MAXDIMS] = {0};
    for (int64_t i = 0; i < arr->size; i++) {
        int64_t off = arr->offset + cnp_multi_to_offset(arr->ndim, coords, arr->strides);
        acc *= cnp_get_element_double(arr->data, off, arr->dtype->type_num);
        for (int d = arr->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < arr->shape[d]) break;
            coords[d] = 0;
        }
    }
    return acc;
}

CNP_API double CNP_CALL cnp_mean_scalar(const CnpArray *arr) {
    const char *function_name = "cnp_mean_scalar";
    if (!scalar_reduction_validate_array(arr, function_name)) return NAN;
    CnpArray *result = cnp_mean_v2(
        arr, 0, true, CNP_NOTYPE);
    if (!result) {
        cnp_relabel_error(function_name);
        return NAN;
    }
    double value = cnp_get_element_double(
        result->data, result->offset, result->dtype->type_num);
    cnp_array_free(result);
    return value;
}

/* NaN-aware reductions */
CNP_API CnpArray* CNP_CALL cnp_nansum(const CnpArray *arr, int axis, CNP_TYPE dtype) {
    CnpArray *result = cnp_nansum_v2(
        arr, axis, axis == CNP_AXIS_NONE, dtype);
    if (!result) cnp_relabel_error("cnp_nansum");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_nanprod(const CnpArray *arr, int axis, CNP_TYPE dtype) {
    CnpArray *result = cnp_nanprod_v2(
        arr, axis, axis == CNP_AXIS_NONE, dtype);
    if (!result) cnp_relabel_error("cnp_nanprod");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_nanmax(const CnpArray *arr, int axis) {
    CnpArray *result = cnp_nanmax_v2(
        arr, axis, axis == CNP_AXIS_NONE);
    if (!result) cnp_relabel_error("cnp_nanmax");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_nanmin(const CnpArray *arr, int axis) {
    CnpArray *result = cnp_nanmin_v2(
        arr, axis, axis == CNP_AXIS_NONE);
    if (!result) cnp_relabel_error("cnp_nanmin");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_nanmean(const CnpArray *arr, int axis, CNP_TYPE dtype) {
    CnpArray *result = cnp_nanmean_v2(
        arr, axis, axis == CNP_AXIS_NONE, dtype);
    if (!result) cnp_relabel_error("cnp_nanmean");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_nanstd(const CnpArray *arr, int axis, int ddof, CNP_TYPE dtype) {
    CnpArray *result = cnp_nanstd_v2(
        arr, axis, axis == CNP_AXIS_NONE, ddof, dtype);
    if (!result) cnp_relabel_error("cnp_nanstd");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_nanvar(const CnpArray *arr, int axis, int ddof, CNP_TYPE dtype) {
    CnpArray *result = cnp_nanvar_v2(
        arr, axis, axis == CNP_AXIS_NONE, ddof, dtype);
    if (!result) cnp_relabel_error("cnp_nanvar");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_nansum_v2(
    const CnpArray *arr, int axis, bool axis_none, CNP_TYPE dtype) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_nansum_v2", &resolved_axis)) return NULL;
    CNP_TYPE out_dtype = dtype == CNP_NOTYPE
        ? reduction_default_sum_dtype(arr->dtype->type_num) : dtype;
    if (reduction_type_is_integer(arr->dtype->type_num) &&
        reduction_type_is_integer(out_dtype))
        return reduction_sum_integer(arr, resolved_axis, out_dtype);
    return reduction_nan_sumprod(arr, resolved_axis, out_dtype, false);
}

CNP_API CnpArray* CNP_CALL cnp_nanprod_v2(
    const CnpArray *arr, int axis, bool axis_none, CNP_TYPE dtype) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_nanprod_v2", &resolved_axis)) return NULL;
    CNP_TYPE out_dtype = dtype == CNP_NOTYPE
        ? reduction_default_sum_dtype(arr->dtype->type_num) : dtype;
    if (reduction_type_is_integer(arr->dtype->type_num) &&
        reduction_type_is_integer(out_dtype))
        return reduction_prod_integer(arr, resolved_axis, out_dtype);
    return reduction_nan_sumprod(arr, resolved_axis, out_dtype, true);
}

CNP_API CnpArray* CNP_CALL cnp_nanmean_v2(
    const CnpArray *arr, int axis, bool axis_none, CNP_TYPE dtype) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_nanmean_v2", &resolved_axis)) return NULL;
    return reduction_nan_mean_or_deviation(
        arr, resolved_axis, 0, dtype, false, false);
}

CNP_API CnpArray* CNP_CALL cnp_nanvar_v2(
    const CnpArray *arr, int axis, bool axis_none,
    int ddof, CNP_TYPE dtype) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_nanvar_v2", &resolved_axis)) return NULL;
    return reduction_nan_mean_or_deviation(
        arr, resolved_axis, ddof, dtype, true, false);
}

CNP_API CnpArray* CNP_CALL cnp_nanstd_v2(
    const CnpArray *arr, int axis, bool axis_none,
    int ddof, CNP_TYPE dtype) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_nanstd_v2", &resolved_axis)) return NULL;
    return reduction_nan_mean_or_deviation(
        arr, resolved_axis, ddof, dtype, true, true);
}

CNP_API CnpArray* CNP_CALL cnp_nanmax_v2(
    const CnpArray *arr, int axis, bool axis_none) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_nanmax_v2", &resolved_axis)) return NULL;
    if (reduction_type_is_integer(arr->dtype->type_num))
        return reduction_extrema(
            arr, resolved_axis, true, "cnp_nanmax_v2");
    return reduction_nan_extrema(
        arr, resolved_axis, true, "cnp_nanmax_v2");
}

CNP_API CnpArray* CNP_CALL cnp_nanmin_v2(
    const CnpArray *arr, int axis, bool axis_none) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_nanmin_v2", &resolved_axis)) return NULL;
    if (reduction_type_is_integer(arr->dtype->type_num))
        return reduction_extrema(
            arr, resolved_axis, false, "cnp_nanmin_v2");
    return reduction_nan_extrema(
        arr, resolved_axis, false, "cnp_nanmin_v2");
}

CNP_API CnpArray* CNP_CALL cnp_nancumsum_v2(
    const CnpArray *arr, int axis, bool axis_none, CNP_TYPE dtype) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_nancumsum_v2", &resolved_axis)) return NULL;
    CNP_TYPE out_dtype = dtype == CNP_NOTYPE
        ? reduction_default_sum_dtype(arr->dtype->type_num) : dtype;
    if (reduction_type_is_integer(arr->dtype->type_num) &&
        reduction_type_is_integer(out_dtype))
        return reduction_cumulative_integer(
            arr, resolved_axis, out_dtype, false);
    return reduction_nan_cumulative(
        arr, resolved_axis, out_dtype, false);
}

CNP_API CnpArray* CNP_CALL cnp_nancumprod_v2(
    const CnpArray *arr, int axis, bool axis_none, CNP_TYPE dtype) {
    int resolved_axis;
    if (!reduction_resolve_axis(
            arr, axis, axis_none, "cnp_nancumprod_v2", &resolved_axis)) return NULL;
    CNP_TYPE out_dtype = dtype == CNP_NOTYPE
        ? reduction_default_sum_dtype(arr->dtype->type_num) : dtype;
    if (reduction_type_is_integer(arr->dtype->type_num) &&
        reduction_type_is_integer(out_dtype))
        return reduction_cumulative_integer(
            arr, resolved_axis, out_dtype, true);
    return reduction_nan_cumulative(
        arr, resolved_axis, out_dtype, true);
}
