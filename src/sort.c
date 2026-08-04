/**
 * cnumpy sorting and searching operations
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>

/* =========================================================================
 * Internal comparison for qsort
 * ========================================================================= */
#define CNP_COMPARE_RAW_TYPED(type) do { \
    type left_value = *(const type*)left; \
    type right_value = *(const type*)right; \
    if (left_value < right_value) return -1; \
    if (left_value > right_value) return 1; \
    return 0; \
} while (0)

static int compare_numpy_long_doubles(
    long double left, long double right) {
    bool left_nan = isnan(left);
    bool right_nan = isnan(right);
    if (left_nan && right_nan) return 0;
    if (left_nan) return 1;
    if (right_nan) return -1;
    if (left < right) return -1;
    if (left > right) return 1;
    return 0;
}

static int compare_numpy_complex_components(
    long double left_real, long double left_imag,
    long double right_real, long double right_imag) {
    int left_class = isnan(left_real)
        ? (isnan(left_imag) ? 3 : 2)
        : (isnan(left_imag) ? 1 : 0);
    int right_class = isnan(right_real)
        ? (isnan(right_imag) ? 3 : 2)
        : (isnan(right_imag) ? 1 : 0);
    if (left_class < right_class) return -1;
    if (left_class > right_class) return 1;
    if (left_class == 0) {
        int real_order = compare_numpy_long_doubles(
            left_real, right_real);
        return real_order != 0 ? real_order
            : compare_numpy_long_doubles(left_imag, right_imag);
    }
    if (left_class == 1)
        return compare_numpy_long_doubles(left_real, right_real);
    if (left_class == 2)
        return compare_numpy_long_doubles(left_imag, right_imag);
    return 0;
}

static int compare_raw_elements(
    const void *left, const void *right, CNP_TYPE dtype) {
    switch (dtype) {
        case CNP_BOOL:
        case CNP_BYTE: CNP_COMPARE_RAW_TYPED(int8_t);
        case CNP_UBYTE: CNP_COMPARE_RAW_TYPED(uint8_t);
        case CNP_SHORT: CNP_COMPARE_RAW_TYPED(int16_t);
        case CNP_USHORT: CNP_COMPARE_RAW_TYPED(uint16_t);
        case CNP_INT: CNP_COMPARE_RAW_TYPED(int32_t);
        case CNP_UINT: CNP_COMPARE_RAW_TYPED(uint32_t);
        case CNP_LONG:
        case CNP_LONGLONG:
        case CNP_DATETIME:
        case CNP_TIMEDELTA: CNP_COMPARE_RAW_TYPED(int64_t);
        case CNP_ULONG:
        case CNP_ULONGLONG: CNP_COMPARE_RAW_TYPED(uint64_t);
        case CNP_FLOAT:
            return cnp_compare_numpy_doubles(
                (double)*(const float*)left,
                (double)*(const float*)right);
        case CNP_DOUBLE:
            return cnp_compare_numpy_doubles(
                *(const double*)left, *(const double*)right);
        case CNP_LONGDOUBLE:
            return compare_numpy_long_doubles(
                *(const long double*)left,
                *(const long double*)right);
        case CNP_HALF:
            return cnp_compare_numpy_doubles(
                cnp_half_to_float(*(const uint16_t*)left),
                cnp_half_to_float(*(const uint16_t*)right));
        case CNP_CFLOAT: {
            const cnp_cfloat *left_value = (const cnp_cfloat*)left;
            const cnp_cfloat *right_value = (const cnp_cfloat*)right;
            return compare_numpy_complex_components(
                (long double)left_value->real,
                (long double)left_value->imag,
                (long double)right_value->real,
                (long double)right_value->imag);
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *left_value = (const cnp_cdouble*)left;
            const cnp_cdouble *right_value = (const cnp_cdouble*)right;
            return compare_numpy_complex_components(
                (long double)left_value->real,
                (long double)left_value->imag,
                (long double)right_value->real,
                (long double)right_value->imag);
        }
        case CNP_CLONGDOUBLE: {
            const cnp_clongdouble *left_value =
                (const cnp_clongdouble*)left;
            const cnp_clongdouble *right_value =
                (const cnp_clongdouble*)right;
            return compare_numpy_complex_components(
                left_value->real, left_value->imag,
                right_value->real, right_value->imag);
        }
        default:
            return 0;
    }
}

static int compare_raw_values(
    const void *data, int64_t left_index, int64_t right_index,
    CNP_TYPE dtype, int element_size) {
    const char *left = (const char*)data + left_index * element_size;
    const char *right = (const char*)data + right_index * element_size;
    return compare_raw_elements(left, right, dtype);
}

typedef union {
    cnp_clongdouble complex_value;
    long double floating_value;
    uint64_t integer_value;
} CnpSortScalarBuffer;

static bool searchsorted_comparison_dtype_is_supported(CNP_TYPE dtype) {
    return dtype == CNP_BOOL || cnp_type_is_integer(dtype) ||
           cnp_type_is_float(dtype) || cnp_type_is_complex(dtype);
}

CNP_STATUS cnp_compare_numeric_elements(
    const void *left, CNP_TYPE left_type,
    const void *right, CNP_TYPE right_type,
    CNP_TYPE comparison_type, int *order,
    const char *function_name) {
    if (left_type == right_type) {
        *order = compare_raw_elements(left, right, left_type);
        return CNP_OK;
    }

    CnpSortScalarBuffer left_value = {0};
    CnpSortScalarBuffer right_value = {0};
    CNP_STATUS status = cnp_cast_scalar_value(
        left, left_type, &left_value, comparison_type, function_name);
    if (status != CNP_OK) return status;
    status = cnp_cast_scalar_value(
        right, right_type, &right_value, comparison_type, function_name);
    if (status != CNP_OK) return status;
    *order = compare_raw_elements(
        &left_value, &right_value, comparison_type);
    return CNP_OK;
}

#undef CNP_COMPARE_RAW_TYPED

static bool sort_dtype_is_supported(CNP_TYPE dtype) {
    return dtype == CNP_BOOL || cnp_type_is_integer(dtype) ||
           cnp_type_is_float(dtype) || cnp_type_is_complex(dtype) ||
           dtype == CNP_DATETIME || dtype == CNP_TIMEDELTA;
}

static bool sort_kind_is_valid(CNP_SORT_KIND kind) {
    return kind == CNP_SORT_QUICKSORT ||
           kind == CNP_SORT_MERGESORT ||
           kind == CNP_SORT_HEAPSORT ||
           kind == CNP_SORT_STABLE;
}

static CNP_STATUS stable_argsort_raw(
    const void *data, int64_t count, CNP_TYPE dtype, int element_size,
    int64_t *indices, const char *function_name) {
    if (!sort_dtype_is_supported(dtype)) {
        cnp_set_error(CNP_ERR_TYPE, function_name,
                      "Sorting dtype %d is not supported", (int)dtype);
        return CNP_ERR_TYPE;
    }
    if (count <= 0) return CNP_OK;
    size_t bytes = (size_t)count * sizeof(int64_t);
    int64_t *temporary = (int64_t*)cnp_malloc(bytes);
    if (!temporary && count > 0) {
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "Failed to allocate stable argsort indices");
        return CNP_ERR_MEMORY;
    }
    for (int64_t index = 0; index < count; ++index)
        indices[index] = index;

    for (int64_t width = 1; width < count;) {
        for (int64_t start = 0; start < count; start += 2 * width) {
            int64_t middle = start + width < count
                ? start + width : count;
            int64_t end = start + 2 * width < count
                ? start + 2 * width : count;
            int64_t left = start;
            int64_t right = middle;
            int64_t destination = start;
            while (left < middle && right < end) {
                int order = compare_raw_values(
                    data, indices[left], indices[right],
                    dtype, element_size);
                temporary[destination++] = order <= 0
                    ? indices[left++] : indices[right++];
            }
            while (left < middle)
                temporary[destination++] = indices[left++];
            while (right < end)
                temporary[destination++] = indices[right++];
        }
        memcpy(indices, temporary, bytes);
        if (width > count / 2) break;
        width *= 2;
    }
    cnp_free(temporary, bytes);
    return CNP_OK;
}

static bool raw_value_less(
    const void *data, int64_t left_index, int64_t right_index,
    CNP_TYPE dtype, int element_size) {
    return compare_raw_values(
        data, left_index, right_index, dtype, element_size) < 0;
}

static void swap_indices(int64_t *left, int64_t *right) {
    int64_t temporary = *left;
    *left = *right;
    *right = temporary;
}

/* NumPy 1.25 aheapsort_ translated to zero-safe one-based indexing. */
static void heapsort_indices(
    const void *data, int64_t *indices, int64_t count,
    CNP_TYPE dtype, int element_size) {
    int64_t heap_size = count;
    for (int64_t level = heap_size >> 1; level > 0; --level) {
        int64_t temporary = indices[level - 1];
        int64_t parent = level;
        int64_t child = level << 1;
        while (child <= heap_size) {
            if (child < heap_size && raw_value_less(
                    data, indices[child - 1], indices[child],
                    dtype, element_size)) {
                ++child;
            }
            if (raw_value_less(
                    data, temporary, indices[child - 1],
                    dtype, element_size)) {
                indices[parent - 1] = indices[child - 1];
                parent = child;
                child += child;
            } else {
                break;
            }
        }
        indices[parent - 1] = temporary;
    }

    while (heap_size > 1) {
        int64_t temporary = indices[heap_size - 1];
        indices[heap_size - 1] = indices[0];
        --heap_size;
        int64_t parent = 1;
        int64_t child = 2;
        while (child <= heap_size) {
            if (child < heap_size && raw_value_less(
                    data, indices[child - 1], indices[child],
                    dtype, element_size)) {
                ++child;
            }
            if (raw_value_less(
                    data, temporary, indices[child - 1],
                    dtype, element_size)) {
                indices[parent - 1] = indices[child - 1];
                parent = child;
                child += child;
            } else {
                break;
            }
        }
        indices[parent - 1] = temporary;
    }
}

static int most_significant_bit(int64_t value) {
    int result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

#define CNP_QUICKSORT_STACK 128
#define CNP_SMALL_QUICKSORT 15

/* NumPy 1.25 aquicksort_: introsort with the same partition and tie order. */
static void quicksort_indices(
    const void *data, int64_t *indices, int64_t count,
    CNP_TYPE dtype, int element_size) {
    if (count <= 1) return;

    int64_t *left = indices;
    int64_t *right = indices + count - 1;
    int64_t *stack[CNP_QUICKSORT_STACK];
    int64_t **stack_pointer = stack;
    int depth_stack[CNP_QUICKSORT_STACK];
    int *depth_pointer = depth_stack;
    int current_depth = most_significant_bit(count) * 2;

    for (;;) {
        if (current_depth < 0) {
            heapsort_indices(
                data, left, right - left + 1, dtype, element_size);
            goto stack_pop;
        }
        while ((right - left) > CNP_SMALL_QUICKSORT) {
            int64_t *middle = left + ((right - left) >> 1);
            if (raw_value_less(
                    data, *middle, *left, dtype, element_size))
                swap_indices(middle, left);
            if (raw_value_less(
                    data, *right, *middle, dtype, element_size))
                swap_indices(right, middle);
            if (raw_value_less(
                    data, *middle, *left, dtype, element_size))
                swap_indices(middle, left);

            int64_t pivot = *middle;
            int64_t *forward = left;
            int64_t *backward = right - 1;
            swap_indices(middle, backward);
            for (;;) {
                do {
                    ++forward;
                } while (raw_value_less(
                    data, *forward, pivot, dtype, element_size));
                do {
                    --backward;
                } while (raw_value_less(
                    data, pivot, *backward, dtype, element_size));
                if (forward >= backward) break;
                swap_indices(forward, backward);
            }
            swap_indices(forward, right - 1);

            if (forward - left < right - forward) {
                *stack_pointer++ = forward + 1;
                *stack_pointer++ = right;
                right = forward - 1;
            } else {
                *stack_pointer++ = left;
                *stack_pointer++ = forward - 1;
                left = forward + 1;
            }
            *depth_pointer++ = --current_depth;
        }

        for (int64_t *current = left + 1; current <= right; ++current) {
            int64_t value_index = *current;
            int64_t *destination = current;
            int64_t *previous = current - 1;
            while (destination > left && raw_value_less(
                    data, value_index, *previous, dtype, element_size)) {
                *destination-- = *previous--;
            }
            *destination = value_index;
        }

    stack_pop:
        if (stack_pointer == stack) break;
        right = *(--stack_pointer);
        left = *(--stack_pointer);
        current_depth = *(--depth_pointer);
    }
}

static bool double_value_less(double left, double right) {
    return cnp_compare_numpy_doubles(left, right) < 0;
}

static void heapsort_double_indices(
    const double *values, int64_t *indices, int64_t count) {
    int64_t heap_size = count;
    for (int64_t level = heap_size >> 1; level > 0; --level) {
        int64_t temporary = indices[level - 1];
        int64_t parent = level;
        int64_t child = level << 1;
        while (child <= heap_size) {
            if (child < heap_size && double_value_less(
                    values[indices[child - 1]], values[indices[child]])) {
                ++child;
            }
            if (double_value_less(
                    values[temporary], values[indices[child - 1]])) {
                indices[parent - 1] = indices[child - 1];
                parent = child;
                child += child;
            } else {
                break;
            }
        }
        indices[parent - 1] = temporary;
    }

    while (heap_size > 1) {
        int64_t temporary = indices[heap_size - 1];
        indices[heap_size - 1] = indices[0];
        --heap_size;
        int64_t parent = 1;
        int64_t child = 2;
        while (child <= heap_size) {
            if (child < heap_size && double_value_less(
                    values[indices[child - 1]], values[indices[child]])) {
                ++child;
            }
            if (double_value_less(
                    values[temporary], values[indices[child - 1]])) {
                indices[parent - 1] = indices[child - 1];
                parent = child;
                child += child;
            } else {
                break;
            }
        }
        indices[parent - 1] = temporary;
    }
}

static void quicksort_double_indices(
    const double *values, int64_t *indices, int64_t count) {
    if (count <= 1) return;

    int64_t *left = indices;
    int64_t *right = indices + count - 1;
    int64_t *stack[CNP_QUICKSORT_STACK];
    int64_t **stack_pointer = stack;
    int depth_stack[CNP_QUICKSORT_STACK];
    int *depth_pointer = depth_stack;
    int current_depth = most_significant_bit(count) * 2;

    for (;;) {
        if (current_depth < 0) {
            heapsort_double_indices(
                values, left, right - left + 1);
            goto stack_pop;
        }
        while ((right - left) > CNP_SMALL_QUICKSORT) {
            int64_t *middle = left + ((right - left) >> 1);
            if (double_value_less(values[*middle], values[*left]))
                swap_indices(middle, left);
            if (double_value_less(values[*right], values[*middle]))
                swap_indices(right, middle);
            if (double_value_less(values[*middle], values[*left]))
                swap_indices(middle, left);

            int64_t pivot = *middle;
            int64_t *forward = left;
            int64_t *backward = right - 1;
            swap_indices(middle, backward);
            for (;;) {
                do {
                    ++forward;
                } while (double_value_less(values[*forward], values[pivot]));
                do {
                    --backward;
                } while (double_value_less(values[pivot], values[*backward]));
                if (forward >= backward) break;
                swap_indices(forward, backward);
            }
            swap_indices(forward, right - 1);

            if (forward - left < right - forward) {
                *stack_pointer++ = forward + 1;
                *stack_pointer++ = right;
                right = forward - 1;
            } else {
                *stack_pointer++ = left;
                *stack_pointer++ = forward - 1;
                left = forward + 1;
            }
            *depth_pointer++ = --current_depth;
        }

        for (int64_t *current = left + 1; current <= right; ++current) {
            int64_t value_index = *current;
            int64_t *destination = current;
            int64_t *previous = current - 1;
            while (destination > left && double_value_less(
                    values[value_index], values[*previous])) {
                *destination-- = *previous--;
            }
            *destination = value_index;
        }

    stack_pop:
        if (stack_pointer == stack) break;
        right = *(--stack_pointer);
        left = *(--stack_pointer);
        current_depth = *(--depth_pointer);
    }
}

static void swap_double_values(double *left, double *right) {
    double temporary = *left;
    *left = *right;
    *right = temporary;
}

static void heapsort_double_values(double *values, int64_t count) {
    int64_t heap_size = count;
    for (int64_t level = heap_size >> 1; level > 0; --level) {
        double temporary = values[level - 1];
        int64_t parent = level;
        int64_t child = level << 1;
        while (child <= heap_size) {
            if (child < heap_size && double_value_less(
                    values[child - 1], values[child])) {
                ++child;
            }
            if (double_value_less(temporary, values[child - 1])) {
                values[parent - 1] = values[child - 1];
                parent = child;
                child += child;
            } else {
                break;
            }
        }
        values[parent - 1] = temporary;
    }

    while (heap_size > 1) {
        double temporary = values[heap_size - 1];
        values[heap_size - 1] = values[0];
        --heap_size;
        int64_t parent = 1;
        int64_t child = 2;
        while (child <= heap_size) {
            if (child < heap_size && double_value_less(
                    values[child - 1], values[child])) {
                ++child;
            }
            if (double_value_less(temporary, values[child - 1])) {
                values[parent - 1] = values[child - 1];
                parent = child;
                child += child;
            } else {
                break;
            }
        }
        values[parent - 1] = temporary;
    }
}

/* NumPy 1.25 quicksort translated to operate on the result values directly. */
static void quicksort_double_values(double *values, int64_t count) {
    if (count <= 1) return;

    double *left = values;
    double *right = values + count - 1;
    double *stack[CNP_QUICKSORT_STACK];
    double **stack_pointer = stack;
    int depth_stack[CNP_QUICKSORT_STACK];
    int *depth_pointer = depth_stack;
    int current_depth = most_significant_bit(count) * 2;

    for (;;) {
        if (current_depth < 0) {
            heapsort_double_values(left, right - left + 1);
            goto stack_pop;
        }
        while ((right - left) > CNP_SMALL_QUICKSORT) {
            double *middle = left + ((right - left) >> 1);
            if (double_value_less(*middle, *left))
                swap_double_values(middle, left);
            if (double_value_less(*right, *middle))
                swap_double_values(right, middle);
            if (double_value_less(*middle, *left))
                swap_double_values(middle, left);

            double pivot = *middle;
            double *forward = left;
            double *backward = right - 1;
            swap_double_values(middle, backward);
            for (;;) {
                do {
                    ++forward;
                } while (double_value_less(*forward, pivot));
                do {
                    --backward;
                } while (double_value_less(pivot, *backward));
                if (forward >= backward) break;
                swap_double_values(forward, backward);
            }
            swap_double_values(forward, right - 1);

            if (forward - left < right - forward) {
                *stack_pointer++ = forward + 1;
                *stack_pointer++ = right;
                right = forward - 1;
            } else {
                *stack_pointer++ = left;
                *stack_pointer++ = forward - 1;
                left = forward + 1;
            }
            *depth_pointer++ = --current_depth;
        }

        for (double *current = left + 1; current <= right; ++current) {
            double value = *current;
            double *destination = current;
            double *previous = current - 1;
            while (destination > left &&
                   double_value_less(value, *previous)) {
                *destination-- = *previous--;
            }
            *destination = value;
        }

    stack_pop:
        if (stack_pointer == stack) break;
        right = *(--stack_pointer);
        left = *(--stack_pointer);
        current_depth = *(--depth_pointer);
    }
}

static void sort_double_values(
    double *values, int64_t count, CNP_SORT_KIND kind) {
    if (kind == CNP_SORT_HEAPSORT) {
        heapsort_double_values(values, count);
    } else if (kind == CNP_SORT_QUICKSORT) {
        quicksort_double_values(values, count);
    }
}

static CNP_STATUS argsort_raw(
    const void *data, int64_t count, CNP_TYPE dtype, int element_size,
    CNP_SORT_KIND kind, int64_t *indices, const char *function_name) {
    if (!sort_dtype_is_supported(dtype)) {
        cnp_set_error(CNP_ERR_TYPE, function_name,
                      "Sorting dtype %d is not supported", (int)dtype);
        return CNP_ERR_TYPE;
    }
    if (!sort_kind_is_valid(kind)) {
        cnp_set_error(CNP_ERR_TYPE, function_name,
                      "Invalid sort kind: %d", (int)kind);
        return CNP_ERR_TYPE;
    }
    if (kind == CNP_SORT_STABLE || kind == CNP_SORT_MERGESORT) {
        return stable_argsort_raw(
            data, count, dtype, element_size, indices, function_name);
    }
    for (int64_t index = 0; index < count; ++index)
        indices[index] = index;
    if (dtype == CNP_DOUBLE) {
        const double *values = (const double*)data;
        if (kind == CNP_SORT_HEAPSORT) {
            heapsort_double_indices(values, indices, count);
        } else {
            quicksort_double_indices(values, indices, count);
        }
        return CNP_OK;
    }
    if (kind == CNP_SORT_HEAPSORT) {
        heapsort_indices(data, indices, count, dtype, element_size);
    } else {
        quicksort_indices(data, indices, count, dtype, element_size);
    }
    return CNP_OK;
}

/* =========================================================================
 * LSD Radix Sort for IEEE 754 doubles
 * Uses 11-bit radix (2048 buckets), 6 passes for 64 bits
 * ========================================================================= */
#define RADIX_BITS 11
#define RADIX_SIZE (1 << RADIX_BITS)  /* 2048 */
#define RADIX_MASK (RADIX_SIZE - 1)
#define CNP_RADIX_SORT_THRESHOLD 512

static CNP_STATUS radix_argsort_doubles(const double *data, int64_t n,
                                        int64_t *result_indices,
                                        const char *function_name);

static CNP_STATUS radix_sort_doubles(
    double *data, int64_t n, const char *function_name) {
    size_t index_bytes = (size_t)n * sizeof(int64_t);
    size_t value_bytes = (size_t)n * sizeof(double);
    int64_t *indices = (int64_t*)cnp_malloc(index_bytes);
    double *sorted = (double*)cnp_malloc(value_bytes);
    if ((!indices || !sorted) && n > 0) {
        if (indices) cnp_free(indices, index_bytes);
        if (sorted) cnp_free(sorted, value_bytes);
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "Failed to allocate radix sort buffers");
        return CNP_ERR_MEMORY;
    }
    CNP_STATUS status = radix_argsort_doubles(
        data, n, indices, function_name);
    if (status != CNP_OK) {
        cnp_free(indices, index_bytes);
        cnp_free(sorted, value_bytes);
        return status;
    }
    for (int64_t index = 0; index < n; ++index)
        sorted[index] = data[indices[index]];
    memcpy(data, sorted, value_bytes);
    cnp_free(indices, index_bytes);
    cnp_free(sorted, value_bytes);
    return CNP_OK;
}

static int64_t sort_axis_offset(
    const CnpArray *arr, int axis,
    int64_t outer_index, int64_t inner_index, int64_t axis_index) {
    int64_t coordinates[CNP_MAXDIMS] = {0};
    int64_t remaining = outer_index;
    for (int dimension = axis - 1; dimension >= 0; --dimension) {
        coordinates[dimension] = remaining % arr->shape[dimension];
        remaining /= arr->shape[dimension];
    }
    remaining = inner_index;
    for (int dimension = arr->ndim - 1; dimension > axis; --dimension) {
        coordinates[dimension] = remaining % arr->shape[dimension];
        remaining /= arr->shape[dimension];
    }
    coordinates[axis] = axis_index;
    return arr->offset + cnp_multi_to_offset(
        arr->ndim, coordinates, arr->strides);
}

static int64_t sort_flat_offset(
    const CnpArray *arr, int64_t flat_index) {
    int64_t offset = arr->offset;
    for (int dimension = arr->ndim - 1; dimension >= 0; --dimension) {
        int64_t coordinate = flat_index % arr->shape[dimension];
        flat_index /= arr->shape[dimension];
        offset += coordinate * arr->strides[dimension];
    }
    return offset;
}

static CnpArray *sort_axis_result(
    const CnpArray *arr, int axis, bool return_indices, CNP_SORT_KIND kind,
    const char *function_name) {
    CNP_TYPE result_type = return_indices
        ? CNP_LONGLONG : arr->dtype->type_num;
    CNP_ORDER result_order = !return_indices &&
        (arr->flags & CNP_ARRAY_F_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        arr->ndim, arr->shape, result_type, result_order);
    if (!result) return NULL;

    int64_t axis_size = arr->shape[axis];
    if (axis_size == 0) return result;
    int64_t outer = 1;
    int64_t inner = 1;
    for (int dimension = 0; dimension < axis; ++dimension)
        outer *= arr->shape[dimension];
    for (int dimension = axis + 1; dimension < arr->ndim; ++dimension)
        inner *= arr->shape[dimension];

    size_t value_bytes = (size_t)axis_size * arr->dtype->elsize;
    size_t index_bytes = (size_t)axis_size * sizeof(int64_t);
    bool direct_double_sort = !return_indices &&
        (kind == CNP_SORT_QUICKSORT || kind == CNP_SORT_HEAPSORT) &&
        arr->dtype->type_num == CNP_DOUBLE;
    if (direct_double_sort &&
        (arr->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        axis == arr->ndim - 1) {
        memcpy(
            result->data,
            (const char*)arr->data + arr->offset,
            (size_t)arr->size * sizeof(double));
        int64_t slice_count = arr->size / axis_size;
        double *result_values = (double*)result->data;
        for (int64_t slice = 0; slice < slice_count; ++slice) {
            sort_double_values(
                result_values + slice * axis_size, axis_size, kind);
        }
        return result;
    }
    if (direct_double_sort) {
        double *values = (double*)cnp_malloc(value_bytes);
        if (!values) {
            cnp_array_free(result);
            cnp_set_error(CNP_ERR_MEMORY, function_name,
                          "Failed to allocate axis sort values");
            return NULL;
        }
        for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
            for (int64_t inner_index = 0;
                 inner_index < inner; ++inner_index) {
                for (int64_t item = 0; item < axis_size; ++item) {
                    int64_t source_offset = sort_axis_offset(
                        arr, axis, outer_index, inner_index, item);
                    values[item] = *(const double*)(
                        (const char*)arr->data + source_offset);
                }
                sort_double_values(values, axis_size, kind);
                for (int64_t item = 0; item < axis_size; ++item) {
                    int64_t destination_offset = sort_axis_offset(
                        result, axis, outer_index, inner_index, item);
                    *(double*)((char*)result->data + destination_offset) =
                        values[item];
                }
            }
        }
        cnp_free(values, value_bytes);
        return result;
    }

    void *values = cnp_malloc(value_bytes);
    int64_t *indices = (int64_t*)cnp_malloc(index_bytes);
    if (!values || !indices) {
        if (values) cnp_free(values, value_bytes);
        if (indices) cnp_free(indices, index_bytes);
        cnp_array_free(result);
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "Failed to allocate axis sort buffers");
        return NULL;
    }

    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
        for (int64_t inner_index = 0; inner_index < inner; ++inner_index) {
            for (int64_t item = 0; item < axis_size; ++item) {
                int64_t source_offset = sort_axis_offset(
                    arr, axis, outer_index, inner_index, item);
                memcpy(
                    (char*)values + item * arr->dtype->elsize,
                    (const char*)arr->data + source_offset,
                    arr->dtype->elsize);
            }
            CNP_STATUS status = argsort_raw(
                values, axis_size, arr->dtype->type_num,
                arr->dtype->elsize, kind, indices, function_name);
            if (status != CNP_OK) {
                cnp_free(values, value_bytes);
                cnp_free(indices, index_bytes);
                cnp_array_free(result);
                return NULL;
            }
            for (int64_t item = 0; item < axis_size; ++item) {
                if (return_indices) {
                    int64_t destination_offset = sort_axis_offset(
                        result, axis, outer_index, inner_index, item);
                    *(int64_t*)((char*)result->data + destination_offset) =
                        indices[item];
                } else {
                    int64_t destination_offset = sort_axis_offset(
                        result, axis, outer_index, inner_index, item);
                    memcpy(
                        (char*)result->data + destination_offset,
                        (const char*)values +
                            indices[item] * arr->dtype->elsize,
                        arr->dtype->elsize);
                }
            }
        }
    }
    cnp_free(values, value_bytes);
    cnp_free(indices, index_bytes);
    return result;
}

static CNP_STATUS radix_argsort_doubles(const double *data, int64_t n,
                                        int64_t *result_indices,
                                        const char *function_name) {
    const size_t key_bytes = (size_t)n * sizeof(uint64_t);
    const size_t index_bytes = (size_t)n * sizeof(int64_t);
    uint64_t *keys_a = (uint64_t*)cnp_malloc(key_bytes);
    uint64_t *keys_b = (uint64_t*)cnp_malloc(key_bytes);
    int64_t *indices_tmp = (int64_t*)cnp_malloc(index_bytes);

    if (!keys_a || !keys_b || !indices_tmp) {
        if (keys_a) cnp_free(keys_a, key_bytes);
        if (keys_b) cnp_free(keys_b, key_bytes);
        if (indices_tmp) cnp_free(indices_tmp, index_bytes);
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "Failed to allocate radix argsort buffers");
        return CNP_ERR_MEMORY;
    }

    for (int64_t i = 0; i < n; i++) {
        keys_a[i] = cnp_double_to_sortable(data[i]);
        result_indices[i] = i;
    }

    uint64_t *key_in = keys_a;
    uint64_t *key_out = keys_b;
    int64_t *index_in = result_indices;
    int64_t *index_out = indices_tmp;
    int64_t count[RADIX_SIZE];

    for (int pass = 0; pass < 6; pass++) {
        const int shift = pass * RADIX_BITS;
        memset(count, 0, sizeof(count));

        for (int64_t i = 0; i < n; i++)
            count[(key_in[i] >> shift) & RADIX_MASK]++;

        int64_t total = 0;
        for (int bucket = 0; bucket < RADIX_SIZE; bucket++) {
            int64_t bucket_count = count[bucket];
            count[bucket] = total;
            total += bucket_count;
        }

        for (int64_t i = 0; i < n; i++) {
            int64_t destination = count[(key_in[i] >> shift) & RADIX_MASK]++;
            key_out[destination] = key_in[i];
            index_out[destination] = index_in[i];
        }

        {
            uint64_t *tmp = key_in;
            key_in = key_out;
            key_out = tmp;
        }
        {
            int64_t *tmp = index_in;
            index_in = index_out;
            index_out = tmp;
        }
    }

    if (key_in != keys_a || index_in != result_indices) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "Radix pass parity did not return data to result storage");
        cnp_free(keys_a, key_bytes);
        cnp_free(keys_b, key_bytes);
        cnp_free(indices_tmp, index_bytes);
        return CNP_ERR_GENERIC;
    }

    cnp_free(keys_a, key_bytes);
    cnp_free(keys_b, key_bytes);
    cnp_free(indices_tmp, index_bytes);
    return CNP_OK;
}

/* =========================================================================
 * Sort
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_sort(
    const CnpArray *arr, int axis, CNP_SORT_KIND kind) {
    CnpArray *result = cnp_sort_v2(
        arr, axis, axis == CNP_AXIS_NONE, kind);
    if (!result) cnp_relabel_error("cnp_sort");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_sort_v2(
    const CnpArray *arr, int axis, bool axis_none, CNP_SORT_KIND kind) {
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_sort_v2",
                      "arr must not be NULL");
        return NULL;
    }
    if (!sort_dtype_is_supported(arr->dtype->type_num)) {
        cnp_set_error(CNP_ERR_TYPE, "cnp_sort_v2",
                      "Sorting dtype %d is not supported",
                      (int)arr->dtype->type_num);
        return NULL;
    }
    if (!sort_kind_is_valid(kind)) {
        cnp_set_error(CNP_ERR_TYPE, "cnp_sort_v2",
                      "Invalid sort kind: %d", (int)kind);
        return NULL;
    }

    if (axis_none) {
        int64_t n = arr->size;
        CnpArray *result = cnp_array_new(1, &n, arr->dtype->type_num, CNP_ORDER_C);
        if (!result) return NULL;
        if (n == 0) return result;

        bool direct_double_sort =
            (kind == CNP_SORT_QUICKSORT || kind == CNP_SORT_HEAPSORT) &&
            arr->dtype->type_num == CNP_DOUBLE &&
            (arr->flags & CNP_ARRAY_C_CONTIGUOUS);
        if (direct_double_sort) {
            memcpy(result->data, (const char*)arr->data + arr->offset,
                   (size_t)n * sizeof(double));
            sort_double_values((double*)result->data, n, kind);
            return result;
        }

        bool stable_kind = kind == CNP_SORT_STABLE ||
                           kind == CNP_SORT_MERGESORT;
        if (stable_kind &&
            (arr->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            arr->dtype->type_num == CNP_DOUBLE &&
            n >= CNP_RADIX_SORT_THRESHOLD) {
            memcpy(result->data, (const char*)arr->data + arr->offset,
                   (size_t)n * sizeof(double));
            if (radix_sort_doubles(
                    (double*)result->data, n, "cnp_sort_v2") != CNP_OK) {
                cnp_array_free(result);
                return NULL;
            }
            return result;
        }

        CnpArray *flat = NULL;
        const char *source_data;
        if (arr->flags & CNP_ARRAY_C_CONTIGUOUS) {
            source_data = (const char*)arr->data + arr->offset;
        } else {
            flat = cnp_flatten(arr, CNP_ORDER_C);
            if (!flat) {
                cnp_relabel_error("cnp_sort_v2");
                cnp_array_free(result);
                return NULL;
            }
            source_data = (const char*)flat->data + flat->offset;
        }

        size_t index_bytes = (size_t)n * sizeof(int64_t);
        int64_t *indices = (int64_t*)cnp_malloc(index_bytes);
        if (!indices) {
            cnp_set_error(CNP_ERR_MEMORY, "cnp_sort_v2",
                          "Failed to allocate sort indices");
            if (flat) cnp_array_free(flat);
            cnp_array_free(result);
            return NULL;
        }
        CNP_STATUS status = argsort_raw(
            source_data, n, arr->dtype->type_num, arr->dtype->elsize,
            kind, indices, "cnp_sort_v2");
        if (status != CNP_OK) {
            cnp_free(indices, index_bytes);
            if (flat) cnp_array_free(flat);
            cnp_array_free(result);
            return NULL;
        }
        for (int64_t index = 0; index < n; ++index) {
            memcpy(
                (char*)result->data + index * result->dtype->elsize,
                source_data + indices[index] * arr->dtype->elsize,
                arr->dtype->elsize);
        }
        cnp_free(indices, index_bytes);
        if (flat) cnp_array_free(flat);
        return result;
    }

    /* Sort along axis */
    int requested_axis = axis;
    axis = cnp_normalize_axis(axis, arr->ndim);
    if (arr->ndim == 0 || axis < 0 || axis >= arr->ndim) {
        cnp_set_error(CNP_ERR_AXIS, "cnp_sort_v2",
                      "axis %d is out of bounds for array of dimension %d",
                      requested_axis, arr->ndim);
        return NULL;
    }
    return sort_axis_result(arr, axis, false, kind, "cnp_sort_v2");
}

static CNP_TYPE sort_complex_result_type(CNP_TYPE source_type) {
    switch (source_type) {
        case CNP_CFLOAT:
        case CNP_CDOUBLE:
        case CNP_CLONGDOUBLE:
            return source_type;
        case CNP_BYTE:
        case CNP_UBYTE:
        case CNP_SHORT:
        case CNP_USHORT:
            return CNP_CFLOAT;
        case CNP_LONGDOUBLE:
            return CNP_CLONGDOUBLE;
        default:
            return CNP_CDOUBLE;
    }
}

CNP_API CnpArray* CNP_CALL cnp_sort_complex(const CnpArray *arr) {
    const char *function_name = "cnp_sort_complex";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "arr must not be NULL");
        return NULL;
    }

    CnpArray *sorted = cnp_sort_v2(
        arr, -1, false, CNP_SORT_QUICKSORT);
    if (!sorted) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (cnp_type_is_complex(sorted->dtype->type_num)) return sorted;

    CNP_TYPE result_type = sort_complex_result_type(
        sorted->dtype->type_num);
    CNP_ORDER result_order = sorted->flags & CNP_ARRAY_F_CONTIGUOUS
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        sorted->ndim, sorted->shape, result_type, result_order);
    if (!result) {
        cnp_array_free(sorted);
        cnp_relabel_error(function_name);
        return NULL;
    }

    CNP_STATUS status = CNP_OK;
    for (int64_t index = 0; index < sorted->size; ++index) {
        int64_t source_offset = sort_flat_offset(sorted, index);
        int64_t destination_offset = sort_flat_offset(result, index);
        status = cnp_cast_scalar_value(
            (const char*)sorted->data + source_offset,
            sorted->dtype->type_num,
            (char*)result->data + destination_offset,
            result_type,
            function_name);
        if (status != CNP_OK) break;
    }
    cnp_array_free(sorted);
    if (status != CNP_OK) {
        cnp_array_free(result);
        return NULL;
    }
    return result;
}

/* =========================================================================
 * Argsort
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_argsort(
    const CnpArray *arr, int axis, CNP_SORT_KIND kind) {
    CnpArray *result = cnp_argsort_v2(
        arr, axis, axis == CNP_AXIS_NONE, kind);
    if (!result) cnp_relabel_error("cnp_argsort");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_argsort_v2(
    const CnpArray *arr, int axis, bool axis_none, CNP_SORT_KIND kind) {
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_argsort_v2",
                      "arr must not be NULL");
        return NULL;
    }
    if (!sort_dtype_is_supported(arr->dtype->type_num)) {
        cnp_set_error(CNP_ERR_TYPE, "cnp_argsort_v2",
                      "Sorting dtype %d is not supported",
                      (int)arr->dtype->type_num);
        return NULL;
    }
    if (!sort_kind_is_valid(kind)) {
        cnp_set_error(CNP_ERR_TYPE, "cnp_argsort_v2",
                      "Invalid sort kind: %d", (int)kind);
        return NULL;
    }

    if (axis_none) {
        int64_t n = arr->size;
        int64_t shape[1] = {n};
        CnpArray *result = cnp_array_new(1, shape, CNP_LONGLONG, CNP_ORDER_C);
        if (!result) return NULL;
        if (n == 0) return result;

        if ((kind == CNP_SORT_STABLE || kind == CNP_SORT_MERGESORT) &&
            (arr->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            arr->dtype->type_num == CNP_DOUBLE &&
            n >= CNP_RADIX_SORT_THRESHOLD) {
            const double *data =
                (const double*)((const char*)arr->data + arr->offset);
            CNP_STATUS status = radix_argsort_doubles(
                data, n, (int64_t*)result->data, "cnp_argsort_v2");
            if (status != CNP_OK) {
                cnp_array_free(result);
                return NULL;
            }
            return result;
        }

        CnpArray *flat = NULL;
        const char *source_data;
        if (arr->flags & CNP_ARRAY_C_CONTIGUOUS) {
            source_data = (const char*)arr->data + arr->offset;
        } else {
            flat = cnp_flatten(arr, CNP_ORDER_C);
            if (!flat) {
                cnp_relabel_error("cnp_argsort_v2");
                cnp_array_free(result);
                return NULL;
            }
            source_data = (const char*)flat->data + flat->offset;
        }
        CNP_STATUS status = argsort_raw(
            source_data, n, arr->dtype->type_num, arr->dtype->elsize,
            kind, (int64_t*)result->data, "cnp_argsort_v2");
        if (flat) cnp_array_free(flat);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        return result;
    }

    int requested_axis = axis;
    axis = cnp_normalize_axis(axis, arr->ndim);
    if (arr->ndim == 0 || axis < 0 || axis >= arr->ndim) {
        cnp_set_error(CNP_ERR_AXIS, "cnp_argsort_v2",
                      "axis %d is out of bounds for array of dimension %d",
                      requested_axis, arr->ndim);
        return NULL;
    }
    return sort_axis_result(arr, axis, true, kind, "cnp_argsort_v2");
}

/* =========================================================================
 * Partition (introselect-compatible observable contract)
 * ========================================================================= */
static int compare_int64_values(const void *left, const void *right) {
    int64_t left_value = *(const int64_t*)left;
    int64_t right_value = *(const int64_t*)right;
    if (left_value < right_value) return -1;
    if (left_value > right_value) return 1;
    return 0;
}

static int64_t partition_indices_once(
    const void *data, int64_t *indices, int64_t left, int64_t right,
    int64_t pivot_position, CNP_TYPE dtype, int element_size) {
    int64_t pivot_index = indices[pivot_position];
    swap_indices(&indices[pivot_position], &indices[right]);
    int64_t store = left;
    for (int64_t position = left; position < right; ++position) {
        if (compare_raw_values(
                data, indices[position], pivot_index,
                dtype, element_size) < 0) {
            swap_indices(&indices[store], &indices[position]);
            ++store;
        }
    }
    swap_indices(&indices[store], &indices[right]);
    return store;
}

static void quickselect_indices(
    const void *data, int64_t *indices, int64_t left, int64_t right,
    int64_t kth, CNP_TYPE dtype, int element_size) {
    while (left < right) {
        int64_t pivot = left + (right - left) / 2;
        pivot = partition_indices_once(
            data, indices, left, right, pivot, dtype, element_size);
        if (pivot == kth) return;
        if (pivot < kth) left = pivot + 1;
        else right = pivot - 1;
    }
}

static int64_t partition_double_values_once(
    double *values, int64_t left, int64_t right,
    int64_t pivot_position) {
    double pivot = values[pivot_position];
    double temporary = values[pivot_position];
    values[pivot_position] = values[right];
    values[right] = temporary;
    int64_t store = left;
    for (int64_t position = left; position < right; ++position) {
        if (cnp_compare_numpy_doubles(values[position], pivot) < 0) {
            temporary = values[store];
            values[store] = values[position];
            values[position] = temporary;
            ++store;
        }
    }
    temporary = values[store];
    values[store] = values[right];
    values[right] = temporary;
    return store;
}

static void quickselect_double_values(
    double *values, int64_t left, int64_t right, int64_t kth) {
    while (left < right) {
        int64_t pivot = left + (right - left) / 2;
        pivot = partition_double_values_once(
            values, left, right, pivot);
        if (pivot == kth) return;
        if (pivot < kth) left = pivot + 1;
        else right = pivot - 1;
    }
}

static void partition_double_values_raw(
    double *values, int64_t count,
    const int64_t *normalized_kth, int kth_count) {
    int64_t completed_kth = -1;
    for (int index = 0; index < kth_count; ++index) {
        int64_t current_kth = normalized_kth[index];
        if (current_kth == completed_kth) continue;
        quickselect_double_values(
            values, completed_kth + 1, count - 1, current_kth);
        completed_kth = current_kth;
    }
}

static int64_t partition_double_pairs_once(
    double *values, int64_t *indices,
    int64_t left, int64_t right, int64_t pivot_position) {
    double pivot = values[pivot_position];
    double temporary_value = values[pivot_position];
    values[pivot_position] = values[right];
    values[right] = temporary_value;
    swap_indices(&indices[pivot_position], &indices[right]);
    int64_t store = left;
    for (int64_t position = left; position < right; ++position) {
        if (cnp_compare_numpy_doubles(values[position], pivot) < 0) {
            temporary_value = values[store];
            values[store] = values[position];
            values[position] = temporary_value;
            swap_indices(&indices[store], &indices[position]);
            ++store;
        }
    }
    temporary_value = values[store];
    values[store] = values[right];
    values[right] = temporary_value;
    swap_indices(&indices[store], &indices[right]);
    return store;
}

static void quickselect_double_pairs(
    double *values, int64_t *indices,
    int64_t left, int64_t right, int64_t kth) {
    while (left < right) {
        int64_t pivot = left + (right - left) / 2;
        pivot = partition_double_pairs_once(
            values, indices, left, right, pivot);
        if (pivot == kth) return;
        if (pivot < kth) left = pivot + 1;
        else right = pivot - 1;
    }
}

static void partition_double_pairs_raw(
    double *values, int64_t *indices, int64_t count,
    const int64_t *normalized_kth, int kth_count) {
    for (int64_t index = 0; index < count; ++index)
        indices[index] = index;
    int64_t completed_kth = -1;
    for (int index = 0; index < kth_count; ++index) {
        int64_t current_kth = normalized_kth[index];
        if (current_kth == completed_kth) continue;
        quickselect_double_pairs(
            values, indices,
            completed_kth + 1, count - 1, current_kth);
        completed_kth = current_kth;
    }
}

static int64_t partition_double_indices_once(
    const double *data, int64_t *indices,
    int64_t left, int64_t right, int64_t pivot_position) {
    int64_t pivot_index = indices[pivot_position];
    swap_indices(&indices[pivot_position], &indices[right]);
    int64_t store = left;
    for (int64_t position = left; position < right; ++position) {
        if (cnp_compare_numpy_doubles(
                data[indices[position]], data[pivot_index]) < 0) {
            swap_indices(&indices[store], &indices[position]);
            ++store;
        }
    }
    swap_indices(&indices[store], &indices[right]);
    return store;
}

static void quickselect_double_indices(
    const double *data, int64_t *indices,
    int64_t left, int64_t right, int64_t kth) {
    while (left < right) {
        int64_t pivot = left + (right - left) / 2;
        pivot = partition_double_indices_once(
            data, indices, left, right, pivot);
        if (pivot == kth) return;
        if (pivot < kth) left = pivot + 1;
        else right = pivot - 1;
    }
}

static CNP_STATUS normalize_partition_kth(
    const int64_t *kth, int kth_count, int64_t axis_size,
    int64_t **normalized_out, const char *function_name) {
    *normalized_out = NULL;
    if (kth_count < 0) {
        cnp_set_error(CNP_ERR_INDEX, function_name,
                      "kth_count must not be negative");
        return CNP_ERR_INDEX;
    }
    if (kth_count > 0 && !kth) {
        cnp_set_error(CNP_ERR_INDEX, function_name,
                      "kth must not be NULL when kth_count is positive");
        return CNP_ERR_INDEX;
    }
    if (kth_count == 0 || axis_size == 0) return CNP_OK;

    size_t bytes = (size_t)kth_count * sizeof(int64_t);
    int64_t *normalized = (int64_t*)cnp_malloc(bytes);
    if (!normalized) {
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "Failed to allocate normalized kth values");
        return CNP_ERR_MEMORY;
    }
    for (int index = 0; index < kth_count; ++index) {
        int64_t value = kth[index];
        if (value < 0) value += axis_size;
        if (value < 0 || value >= axis_size) {
            cnp_free(normalized, bytes);
            cnp_set_error(
                CNP_ERR_INDEX, function_name,
                "kth(=%lld) out of bounds (%lld)",
                (long long)kth[index], (long long)axis_size);
            return CNP_ERR_INDEX;
        }
        normalized[index] = value;
    }
    qsort(normalized, (size_t)kth_count, sizeof(int64_t),
          compare_int64_values);
    *normalized_out = normalized;
    return CNP_OK;
}

static void partition_indices_raw(
    const void *data, int64_t count, CNP_TYPE dtype, int element_size,
    const int64_t *normalized_kth, int kth_count, int64_t *indices) {
    for (int64_t index = 0; index < count; ++index)
        indices[index] = index;
    int64_t completed_kth = -1;
    for (int index = 0; index < kth_count; ++index) {
        int64_t current_kth = normalized_kth[index];
        if (current_kth == completed_kth) continue;
        if (dtype == CNP_DOUBLE) {
            quickselect_double_indices(
                (const double*)data, indices,
                completed_kth + 1, count - 1, current_kth);
        } else {
            quickselect_indices(
                data, indices, completed_kth + 1, count - 1,
                current_kth, dtype, element_size);
        }
        completed_kth = current_kth;
    }
}

static CnpArray *partition_axis_result(
    const CnpArray *arr, int axis, bool return_indices,
    const int64_t *normalized_kth, int kth_count,
    const char *function_name) {
    CNP_TYPE result_type = return_indices
        ? CNP_LONGLONG : arr->dtype->type_num;
    CNP_ORDER result_order = !return_indices &&
        (arr->flags & CNP_ARRAY_F_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        arr->ndim, arr->shape, result_type, result_order);
    if (!result || arr->size == 0) return result;

    int64_t axis_size = arr->shape[axis];
    int64_t outer = 1;
    int64_t inner = 1;
    for (int dimension = 0; dimension < axis; ++dimension)
        outer *= arr->shape[dimension];
    for (int dimension = axis + 1; dimension < arr->ndim; ++dimension)
        inner *= arr->shape[dimension];

    size_t value_bytes = (size_t)axis_size * arr->dtype->elsize;
    size_t index_bytes = (size_t)axis_size * sizeof(int64_t);
    void *values = cnp_malloc(value_bytes);
    bool direct_double_values = !return_indices &&
        arr->dtype->type_num == CNP_DOUBLE;
    bool direct_double_indices = return_indices &&
        arr->dtype->type_num == CNP_DOUBLE;
    int64_t *indices = direct_double_values
        ? NULL : (int64_t*)cnp_malloc(index_bytes);
    if (!values || (!direct_double_values && !indices)) {
        if (values) cnp_free(values, value_bytes);
        if (indices) cnp_free(indices, index_bytes);
        cnp_array_free(result);
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "Failed to allocate axis partition buffers");
        return NULL;
    }

    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
        for (int64_t inner_index = 0; inner_index < inner; ++inner_index) {
            for (int64_t item = 0; item < axis_size; ++item) {
                int64_t source_offset = sort_axis_offset(
                    arr, axis, outer_index, inner_index, item);
                memcpy(
                    (char*)values + item * arr->dtype->elsize,
                    (const char*)arr->data + source_offset,
                    arr->dtype->elsize);
            }
            if (direct_double_values) {
                partition_double_values_raw(
                    (double*)values, axis_size,
                    normalized_kth, kth_count);
            } else if (direct_double_indices) {
                partition_double_pairs_raw(
                    (double*)values, indices, axis_size,
                    normalized_kth, kth_count);
            } else {
                partition_indices_raw(
                    values, axis_size, arr->dtype->type_num,
                    arr->dtype->elsize,
                    normalized_kth, kth_count, indices);
            }
            for (int64_t item = 0; item < axis_size; ++item) {
                int64_t destination_offset = sort_axis_offset(
                    result, axis, outer_index, inner_index, item);
                if (return_indices) {
                    *(int64_t*)((char*)result->data + destination_offset) =
                        indices[item];
                } else if (direct_double_values) {
                    memcpy(
                        (char*)result->data + destination_offset,
                        (const char*)values +
                            item * arr->dtype->elsize,
                        arr->dtype->elsize);
                } else {
                    memcpy(
                        (char*)result->data + destination_offset,
                        (const char*)values +
                            indices[item] * arr->dtype->elsize,
                        arr->dtype->elsize);
                }
            }
        }
    }
    cnp_free(values, value_bytes);
    if (indices) cnp_free(indices, index_bytes);
    return result;
}

static CnpArray *partition_flat_result(
    const CnpArray *arr, bool return_indices,
    const int64_t *normalized_kth, int kth_count,
    const char *function_name) {
    int64_t count = arr->size;
    CNP_TYPE result_type = return_indices
        ? CNP_LONGLONG : arr->dtype->type_num;
    CnpArray *result = cnp_array_new(
        1, &count, result_type, CNP_ORDER_C);
    if (!result || count == 0) return result;

    CnpArray *flat = NULL;
    const void *source_data;
    if (arr->flags & CNP_ARRAY_C_CONTIGUOUS) {
        source_data = (const char*)arr->data + arr->offset;
    } else {
        flat = cnp_flatten(arr, CNP_ORDER_C);
        if (!flat) {
            cnp_relabel_error(function_name);
            cnp_array_free(result);
            return NULL;
        }
        source_data = (const char*)flat->data + flat->offset;
    }

    if (!return_indices && arr->dtype->type_num == CNP_DOUBLE) {
        size_t value_bytes = (size_t)count * sizeof(double);
        memcpy(result->data, source_data, value_bytes);
        partition_double_values_raw(
            (double*)result->data, count, normalized_kth, kth_count);
        if (flat) cnp_array_free(flat);
        return result;
    }
    if (return_indices && arr->dtype->type_num == CNP_DOUBLE) {
        size_t value_bytes = (size_t)count * sizeof(double);
        double *values = (double*)cnp_malloc(value_bytes);
        if (!values) {
            if (flat) cnp_array_free(flat);
            cnp_array_free(result);
            cnp_set_error(CNP_ERR_MEMORY, function_name,
                          "Failed to allocate flat argpartition values");
            return NULL;
        }
        memcpy(values, source_data, value_bytes);
        partition_double_pairs_raw(
            values, (int64_t*)result->data, count,
            normalized_kth, kth_count);
        cnp_free(values, value_bytes);
        if (flat) cnp_array_free(flat);
        return result;
    }

    size_t index_bytes = (size_t)count * sizeof(int64_t);
    int64_t *indices = (int64_t*)cnp_malloc(index_bytes);
    if (!indices) {
        if (flat) cnp_array_free(flat);
        cnp_array_free(result);
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "Failed to allocate flat partition indices");
        return NULL;
    }
    partition_indices_raw(
        source_data, count, arr->dtype->type_num, arr->dtype->elsize,
        normalized_kth, kth_count, indices);
    if (return_indices) {
        memcpy(result->data, indices, index_bytes);
    } else {
        for (int64_t index = 0; index < count; ++index) {
            memcpy(
                (char*)result->data + index * arr->dtype->elsize,
                (const char*)source_data +
                    indices[index] * arr->dtype->elsize,
                arr->dtype->elsize);
        }
    }
    cnp_free(indices, index_bytes);
    if (flat) cnp_array_free(flat);
    return result;
}

static CnpArray *partition_v2_impl(
    const CnpArray *arr, const int64_t *kth, int kth_count,
    int axis, bool axis_none, bool return_indices,
    const char *function_name) {
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "arr must not be NULL");
        return NULL;
    }
    if (!sort_dtype_is_supported(arr->dtype->type_num)) {
        cnp_set_error(CNP_ERR_TYPE, function_name,
                      "Partition dtype %d is not supported",
                      (int)arr->dtype->type_num);
        return NULL;
    }
    if (kth_count < 0 || (kth_count > 0 && !kth)) {
        int64_t *unused = NULL;
        normalize_partition_kth(
            kth, kth_count, 0, &unused, function_name);
        return NULL;
    }

    bool flatten = axis_none;
    int resolved_axis = axis;
    int64_t axis_size;
    if (arr->ndim == 0) {
        if (!return_indices && !axis_none) {
            cnp_set_error(
                CNP_ERR_AXIS, function_name,
                "axis %d is out of bounds for array of dimension 0", axis);
            return NULL;
        }
        if (return_indices && !axis_none) {
            resolved_axis = cnp_normalize_axis(axis, 1);
            if (resolved_axis != 0) {
                cnp_set_error(
                    CNP_ERR_AXIS, function_name,
                    "axis %d is out of bounds for array of dimension 1", axis);
                return NULL;
            }
        }
        flatten = true;
        axis_size = 1;
    } else if (axis_none) {
        axis_size = arr->size;
    } else {
        resolved_axis = cnp_normalize_axis(axis, arr->ndim);
        if (resolved_axis < 0 || resolved_axis >= arr->ndim) {
            cnp_set_error(
                CNP_ERR_AXIS, function_name,
                "axis %d is out of bounds for array of dimension %d",
                axis, arr->ndim);
            return NULL;
        }
        axis_size = arr->shape[resolved_axis];
    }

    int64_t *normalized_kth = NULL;
    if (arr->size != 0 && normalize_partition_kth(
            kth, kth_count, axis_size,
            &normalized_kth, function_name) != CNP_OK) {
        return NULL;
    }
    CnpArray *result = flatten
        ? partition_flat_result(
            arr, return_indices, normalized_kth, kth_count, function_name)
        : partition_axis_result(
            arr, resolved_axis, return_indices,
            normalized_kth, kth_count, function_name);
    if (normalized_kth) {
        cnp_free(
            normalized_kth, (size_t)kth_count * sizeof(int64_t));
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_partition_v2(
    const CnpArray *arr, const int64_t *kth, int kth_count,
    int axis, bool axis_none) {
    return partition_v2_impl(
        arr, kth, kth_count, axis, axis_none,
        false, "cnp_partition_v2");
}

CNP_API CnpArray* CNP_CALL cnp_argpartition_v2(
    const CnpArray *arr, const int64_t *kth, int kth_count,
    int axis, bool axis_none) {
    return partition_v2_impl(
        arr, kth, kth_count, axis, axis_none,
        true, "cnp_argpartition_v2");
}

CNP_API CnpArray* CNP_CALL cnp_partition(
    const CnpArray *arr, int64_t kth, int axis) {
    CnpArray *result = cnp_partition_v2(
        arr, &kth, 1, axis, axis == CNP_AXIS_NONE);
    if (!result) cnp_relabel_error("cnp_partition");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_argpartition(
    const CnpArray *arr, int64_t kth, int axis) {
    CnpArray *result = cnp_argpartition_v2(
        arr, &kth, 1, axis, axis == CNP_AXIS_NONE);
    if (!result) cnp_relabel_error("cnp_argpartition");
    return result;
}

/* =========================================================================
 * Searchsorted
 * ========================================================================= */
static bool searchsorted_contiguous_float64(
    const CnpArray *arr, const CnpArray *values,
    bool left, CnpArray *result) {
    if (!(arr->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            !(values->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            arr->dtype->type_num != CNP_DOUBLE ||
            values->dtype->type_num != CNP_DOUBLE) {
        return false;
    }

    int64_t value_count = values->size;
    int64_t source_count = arr->size;
    int64_t *output = (int64_t*)result->data;
    if (value_count == 0) return true;
    if (source_count == 0) {
        for (int64_t index = 0; index < value_count; ++index)
            output[index] = 0;
        return true;
    }

    const double *source = (const double*)(
        (const char*)arr->data + arr->offset);
    const double *queries = (const double*)(
        (const char*)values->data + values->offset);
    for (int64_t index = 0; index < value_count; ++index) {
        double query = queries[index];
        int64_t low = 0;
        int64_t high = source_count;
        while (low < high) {
            int64_t middle = low + (high - low) / 2;
            int order = cnp_compare_numpy_doubles(source[middle], query);
            if (order < 0 || (!left && order == 0)) low = middle + 1;
            else high = middle;
        }
        output[index] = low;
    }
    return true;
}

CNP_API CnpArray* CNP_CALL cnp_searchsorted_v2(
    const CnpArray *arr, const CnpArray *values,
    const char *side, const CnpArray *sorter) {
    const char *function_name = "cnp_searchsorted_v2";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "source array is NULL");
        return NULL;
    }
    if (!values) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "values array is NULL");
        return NULL;
    }
    if (arr->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source array must be one-dimensional, got rank %d",
            arr->ndim);
        return NULL;
    }
    if (side && strcmp(side, "left") != 0 &&
            strcmp(side, "right") != 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "search side must be 'left' or 'right' (got '%s')", side);
        return NULL;
    }
    bool left = (!side || strcmp(side, "left") == 0);

    int64_t n = arr->size;
    CNP_TYPE comparison_type = cnp_promote_type_full(
        arr->dtype->type_num, values->dtype->type_num);
    if (!searchsorted_comparison_dtype_is_supported(comparison_type)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "searchsorted cannot compare source dtype %d with values dtype %d",
            (int)arr->dtype->type_num, (int)values->dtype->type_num);
        return NULL;
    }
    if (sorter) {
        if (sorter->ndim != 1) {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "sorter must be one-dimensional, got rank %d",
                sorter->ndim);
            return NULL;
        }
        if (!cnp_type_is_integer(sorter->dtype->type_num)) {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "sorter must contain integer indices, got dtype %d",
                (int)sorter->dtype->type_num);
            return NULL;
        }
        if (sorter->size != n) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "sorter size %lld must equal source size %lld",
                (long long)sorter->size, (long long)n);
            return NULL;
        }
    }

    CnpArray *result = cnp_array_new(
        values->ndim, values->shape, CNP_LONGLONG, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    if (!sorter && searchsorted_contiguous_float64(
            arr, values, left, result)) {
        return result;
    }

    for (int64_t i = 0; i < values->size; i++) {
        int64_t value_offset = sort_flat_offset(values, i);
        const void *value_data =
            (const char*)values->data + value_offset;
        int64_t lo = 0, hi = n;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            int64_t source_index = mid;
            if (sorter) {
                int64_t sorter_offset = sort_flat_offset(sorter, mid);
                source_index = cnp_get_element_int(
                    sorter->data, sorter_offset,
                    sorter->dtype->type_num);
                if (source_index < 0 || source_index >= n) {
                    cnp_array_free(result);
                    cnp_set_error(
                        CNP_ERR_INDEX, function_name,
                        "sorter index %lld is out of range for source size %lld",
                        (long long)source_index, (long long)n);
                    return NULL;
                }
            }
            int64_t source_offset = sort_flat_offset(arr, source_index);
            const void *source_data =
                (const char*)arr->data + source_offset;
            int order = 0;
            CNP_STATUS compare_status = cnp_compare_numeric_elements(
                source_data, arr->dtype->type_num,
                value_data, values->dtype->type_num,
                comparison_type, &order, function_name);
            if (compare_status != CNP_OK) {
                cnp_array_free(result);
                return NULL;
            }
            if (order < 0 || (!left && order == 0)) lo = mid + 1;
            else hi = mid;
        }
        *((int64_t*)result->data + i) = lo;
    }

    return result;
}

CNP_API CnpArray* CNP_CALL cnp_searchsorted(
    const CnpArray *arr, const CnpArray *values, const char *side) {
    CnpArray *result = cnp_searchsorted_v2(arr, values, side, NULL);
    if (!result) cnp_relabel_error("cnp_searchsorted");
    return result;
}
