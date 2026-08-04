/**
 * cnumpy final comprehensive module
 * Corresponds to numpy: ascontiguousarray, asfortranarray, resize, byte_bounds,
 *   mat/asmatrix, matlib functions, polyfromroots, polyroots,
 *   rollaxis, compress_ext, place_ext, put, putmask, take,
 *   nan_to_num_ext, real_if_close, isclose_ext, isin
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>

static int64_t final_ext_flat_offset(
    const CnpArray *array, int64_t flat_index) {
    int64_t offset = array->offset;
    for (int dimension = array->ndim - 1; dimension >= 0; --dimension) {
        int64_t coordinate = flat_index % array->shape[dimension];
        flat_index /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

/* =========================================================================
 * cnp_ascontiguousarray - Return contiguous array
 * numpy.ascontiguousarray(a)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_ascontiguousarray(const CnpArray *arr) {
    const char *function_name = "cnp_ascontiguousarray";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return NULL;
    }
    int64_t scalar_shape[1] = {1};
    int result_ndim = arr->ndim == 0 ? 1 : arr->ndim;
    const int64_t *result_shape = arr->ndim == 0 ? scalar_shape : arr->shape;
    CnpArray *result = cnp_array_new(
        result_ndim, result_shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CNP_STATUS status = cnp_copyto(result, arr, CNP_CAST_NO);
    if (status != CNP_OK) {
        cnp_array_free(result);
        cnp_relabel_error(function_name);
        return NULL;
    }
    return result;
}

/* =========================================================================
 * cnp_asfortranarray - Return Fortran-contiguous array
 * numpy.asfortranarray(a)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_asfortranarray(const CnpArray *arr) {
    const char *function_name = "cnp_asfortranarray";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return NULL;
    }
    int64_t scalar_shape[1] = {1};
    int result_ndim = arr->ndim == 0 ? 1 : arr->ndim;
    const int64_t *result_shape = arr->ndim == 0 ? scalar_shape : arr->shape;
    CnpArray *result = cnp_array_new(
        result_ndim, result_shape, arr->dtype->type_num, CNP_ORDER_F);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CNP_STATUS status = cnp_copyto(result, arr, CNP_CAST_NO);
    if (status != CNP_OK) {
        cnp_array_free(result);
        cnp_relabel_error(function_name);
        return NULL;
    }
    return result;
}

/* =========================================================================
 * cnp_resize - Resize array to new shape (repeats data if needed)
 * numpy.resize(a, new_shape)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_resize(const CnpArray *arr, int ndim, const int64_t *new_shape) {
    const char *function_name = "cnp_resize";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return NULL;
    }
    if (ndim < 0 || ndim > CNP_MAXDIMS || (ndim > 0 && !new_shape)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "new shape metadata is invalid");
        return NULL;
    }
    int64_t new_size = 1;
    for (int dimension = 0; dimension < ndim; ++dimension) {
        if (new_shape[dimension] < 0) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "new shape must not contain negative dimensions");
            return NULL;
        }
        if (new_size != 0 && new_shape[dimension] != 0 &&
                new_size > INT64_MAX / new_shape[dimension]) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "new shape product exceeds int64");
            return NULL;
        }
        new_size *= new_shape[dimension];
    }

    CnpArray *result = cnp_array_new(
        ndim, new_shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    if (arr->size == 0) {
        memset(
            result->data, 0,
            (size_t)new_size * (size_t)result->dtype->elsize);
        return result;
    }

    for (int64_t index = 0; index < new_size; ++index) {
        int64_t source_offset = final_ext_flat_offset(
            arr, index % arr->size);
        memcpy(
            (char*)result->data + index * result->dtype->elsize,
            (const char*)arr->data + source_offset,
            (size_t)result->dtype->elsize);
    }
    return result;
}

/* =========================================================================
 * cnp_byte_bounds - Return pointer to first and last byte of array data
 * numpy.byte_bounds(a)
 * ========================================================================= */
CNP_API CNP_STATUS CNP_CALL cnp_byte_bounds(const CnpArray *arr, void **low, void **high) {
    const char *function_name = "cnp_byte_bounds";
    if (low) *low = NULL;
    if (high) *high = NULL;
    if (!arr || !low || !high || low == high) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "array and distinct low/high outputs are required");
        return CNP_ERR_GENERIC;
    }
    if (!arr->dtype || !arr->data || arr->ndim < 0 ||
            arr->ndim > CNP_MAXDIMS ||
            (arr->ndim > 0 && (!arr->shape || !arr->strides))) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "array metadata and data buffer must be valid");
        return CNP_ERR_GENERIC;
    }

    int64_t minimum_offset = arr->offset;
    int64_t maximum_offset = arr->offset;
    for (int dimension = 0; dimension < arr->ndim; ++dimension) {
        if (arr->shape[dimension] < 0) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "array shape must not contain negative dimensions");
            return CNP_ERR_SHAPE;
        }
        if (arr->shape[dimension] == 0) {
            *low = (char*)arr->data + arr->offset;
            *high = *low;
            return CNP_OK;
        }
        int64_t extent =
            (arr->shape[dimension] - 1) * arr->strides[dimension];
        if (extent < 0) minimum_offset += extent;
        else maximum_offset += extent;
    }
    *low = (char*)arr->data + minimum_offset;
    *high = (char*)arr->data + maximum_offset + arr->dtype->elsize;
    return CNP_OK;
}

/* =========================================================================
 * cnp_putmask - Put values where mask is True
 * numpy.putmask(a, mask, values)
 * ========================================================================= */
static int64_t putmask_flat_offset(
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

CNP_API CNP_STATUS CNP_CALL cnp_putmask(CnpArray *arr, const CnpArray *mask, const CnpArray *values) {
    const char *function_name = "cnp_putmask";
    if (!arr || !mask || !values) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "array, mask, and values are required");
        return CNP_ERR_GENERIC;
    }
    if (!(arr->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "destination array is not writeable");
        return CNP_ERR_GENERIC;
    }
    if (!cnp_dtype_can_cast(
            values->dtype->type_num,
            arr->dtype->type_num,
            CNP_CAST_SAFE)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "values dtype cannot safely cast to the destination dtype");
        return CNP_ERR_TYPE;
    }
    if (mask->size != arr->size) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "mask and destination must have the same size");
        return CNP_ERR_SHAPE;
    }
    if (values->size == 0 || arr->size == 0) return CNP_OK;

    CnpArray *snapshot = cnp_array_copy(arr);
    if (!snapshot) {
        cnp_relabel_error(function_name);
        return cnp_get_error(NULL);
    }

    CNP_STATUS status = CNP_OK;
    for (int64_t index = 0; index < arr->size; ++index) {
        uint8_t selected = 0;
        int64_t mask_offset = putmask_flat_offset(mask, index);
        status = cnp_cast_scalar_value(
            (const char*)mask->data + mask_offset,
            mask->dtype->type_num,
            &selected, CNP_BOOL, function_name);
        if (status != CNP_OK) break;
        if (!selected) continue;

        int64_t value_offset = putmask_flat_offset(
            values, index % values->size);
        int64_t destination_offset = putmask_flat_offset(snapshot, index);
        status = cnp_cast_scalar_value(
            (const char*)values->data + value_offset,
            values->dtype->type_num,
            (char*)snapshot->data + destination_offset,
            snapshot->dtype->type_num,
            function_name);
        if (status != CNP_OK) break;
    }
    if (status == CNP_OK) {
        status = cnp_copyto(arr, snapshot, CNP_CAST_NO);
        if (status != CNP_OK) cnp_relabel_error(function_name);
    }
    cnp_array_free(snapshot);
    return status;
}

/* =========================================================================
 * cnp_real_if_close - Convert complex to real if imaginary part is small
 * numpy.real_if_close(a, tol=100)
 * ========================================================================= */
static CNP_STATUS real_if_close_validate_array(const CnpArray *arr) {
    const char *function_name = "cnp_real_if_close";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return CNP_ERR_GENERIC;
    }
    if (!arr->dtype || arr->dtype->type_num <= CNP_NOTYPE ||
            arr->dtype->type_num >= CNP_NTYPES) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input array must have a valid dtype");
        return CNP_ERR_TYPE;
    }
    if (arr->ndim < 0 || arr->ndim > CNP_MAXDIMS ||
            (arr->ndim > 0 && (!arr->shape || !arr->strides))) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input array has invalid shape metadata");
        return CNP_ERR_SHAPE;
    }
    if (arr->size > 0 && !arr->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array requires a data buffer");
        return CNP_ERR_GENERIC;
    }
    return CNP_OK;
}

static long double real_if_close_epsilon(CNP_TYPE type) {
    if (type == CNP_CFLOAT) return (long double)FLT_EPSILON;
    if (type == CNP_CDOUBLE) return (long double)DBL_EPSILON;
    return (long double)LDBL_EPSILON;
}

static long double real_if_close_imaginary_at(
    const CnpArray *arr, int64_t offset) {
    const char *element = (const char*)arr->data + offset;
    if (arr->dtype->type_num == CNP_CFLOAT)
        return (long double)((const float*)element)[1];
    if (arr->dtype->type_num == CNP_CDOUBLE)
        return (long double)((const double*)element)[1];
    return ((const cnp_clongdouble*)element)->imag;
}

static bool real_if_close_double_is_nan(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) ==
            UINT64_C(0x7ff0000000000000) &&
        (bits & UINT64_C(0x000fffffffffffff)) != 0;
}

static bool real_if_close_imaginary_is_nonfinite(
    const CnpArray *arr, int64_t offset) {
    const char *element = (const char*)arr->data + offset;
    if (arr->dtype->type_num == CNP_CFLOAT) {
        uint32_t bits;
        memcpy(&bits, element + sizeof(float), sizeof(bits));
        return (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000);
    }
    if (arr->dtype->type_num == CNP_CDOUBLE) {
        uint64_t bits;
        memcpy(&bits, element + sizeof(double), sizeof(bits));
        return (bits & UINT64_C(0x7ff0000000000000)) ==
            UINT64_C(0x7ff0000000000000);
    }
#if defined(_MSC_VER)
    double converted = (double)((const cnp_clongdouble*)element)->imag;
    uint64_t bits;
    memcpy(&bits, &converted, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) ==
        UINT64_C(0x7ff0000000000000);
#else
    return !isfinite(((const cnp_clongdouble*)element)->imag);
#endif
}

static int64_t real_if_close_flat_offset(
    const CnpArray *arr, int64_t flat_index) {
    int64_t offset = arr->offset;
    for (int axis = arr->ndim - 1; axis >= 0; --axis) {
        int64_t coordinate = flat_index % arr->shape[axis];
        flat_index /= arr->shape[axis];
        offset += coordinate * arr->strides[axis];
    }
    return offset;
}

static bool real_if_close_all_imaginary_below(
    const CnpArray *arr, long double tolerance) {
    bool contiguous = (arr->flags &
        (CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) != 0;
    for (int64_t index = 0; index < arr->size; ++index) {
        int64_t offset = contiguous
            ? arr->offset + index * arr->dtype->elsize
            : real_if_close_flat_offset(arr, index);
        if (real_if_close_imaginary_is_nonfinite(arr, offset)) return false;
        long double imaginary = real_if_close_imaginary_at(arr, offset);
        if (!(fabsl(imaginary) < tolerance)) return false;
    }
    return true;
}

CNP_API CnpArray* CNP_CALL cnp_real_if_close(const CnpArray *arr, double tol) {
    if (real_if_close_validate_array(arr) != CNP_OK) return NULL;
    if (!cnp_type_is_complex(arr->dtype->type_num)) {
        cnp_array_incref((CnpArray*)arr);
        return (CnpArray*)arr;
    }

    long double tolerance = (long double)tol;
    if (tol > 1.0)
        tolerance *= real_if_close_epsilon(arr->dtype->type_num);
    bool all_close = arr->size == 0 ||
        (!real_if_close_double_is_nan(tol) &&
         real_if_close_all_imaginary_below(arr, tolerance));
    if (all_close) {
        CnpArray *result = cnp_real(arr);
        if (!result) cnp_relabel_error("cnp_real_if_close");
        return result;
    }

    cnp_array_incref((CnpArray*)arr);
    return (CnpArray*)arr;
}

/* =========================================================================
 * cnp_polyfromroots - Generate polynomial from roots
 * numpy.polynomial.polynomial.polyfromroots(roots)
 * ========================================================================= */
typedef struct {
    double real;
    double imag;
} CnpPolyFromRootsValue;

static CnpPolyFromRootsValue cnp_polyfromroots_value_at(
    const CnpArray *roots, int64_t index) {
    int64_t coordinate = index;
    const void *pointer = cnp_array_at(roots, &coordinate);
    CnpPolyFromRootsValue value = {0.0, 0.0};
    switch (roots->dtype->type_num) {
        case CNP_HALF:
            value.real = cnp_half_to_float(*(const uint16_t*)pointer);
            break;
        case CNP_CFLOAT: {
            const cnp_cfloat *source = (const cnp_cfloat*)pointer;
            value.real = source->real;
            value.imag = source->imag;
            break;
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *source = (const cnp_cdouble*)pointer;
            value.real = source->real;
            value.imag = source->imag;
            break;
        }
        case CNP_CLONGDOUBLE: {
            const cnp_clongdouble *source =
                (const cnp_clongdouble*)pointer;
            value.real = (double)source->real;
            value.imag = (double)source->imag;
            break;
        }
        default:
            value.real = cnp_get_element_double(
                pointer, 0, roots->dtype->type_num);
            break;
    }
    return value;
}

static CnpPolyFromRootsValue cnp_polyfromroots_multiply(
    CnpPolyFromRootsValue left,
    CnpPolyFromRootsValue right,
    bool complex_arithmetic) {
    if (!complex_arithmetic) {
        return (CnpPolyFromRootsValue){
            left.real * right.real, 0.0};
    }
    return (CnpPolyFromRootsValue){
        left.real * right.real - left.imag * right.imag,
        left.real * right.imag + left.imag * right.real};
}

typedef struct {
    size_t offset;
    int64_t length;
} CnpPolyFromRootsPolynomial;

static int cnp_polyfromroots_component_compare(double left, double right) {
    bool left_nan = isnan(left);
    bool right_nan = isnan(right);
    if (left_nan) return right_nan ? 0 : 1;
    if (right_nan) return -1;
    if (left < right) return -1;
    if (left > right) return 1;
    return 0;
}

static int cnp_polyfromroots_root_compare(
    const void *left_pointer, const void *right_pointer) {
    const CnpPolyFromRootsValue *left =
        (const CnpPolyFromRootsValue*)left_pointer;
    const CnpPolyFromRootsValue *right =
        (const CnpPolyFromRootsValue*)right_pointer;
    int comparison = cnp_polyfromroots_component_compare(
        left->real, right->real);
    if (comparison != 0) return comparison;
    return cnp_polyfromroots_component_compare(
        left->imag, right->imag);
}

static void cnp_polyfromroots_multiply_polynomials(
    const CnpPolyFromRootsValue *left,
    int64_t left_length,
    const CnpPolyFromRootsValue *right,
    int64_t right_length,
    CnpPolyFromRootsValue *output,
    bool complex_arithmetic) {
    int64_t output_length = left_length + right_length - 1;
    for (int64_t index = 0; index < output_length; ++index) {
        output[index] = (CnpPolyFromRootsValue){0.0, 0.0};
    }
    for (int64_t left_index = 0;
         left_index < left_length; ++left_index) {
        for (int64_t right_index = 0;
             right_index < right_length; ++right_index) {
            CnpPolyFromRootsValue product = cnp_polyfromroots_multiply(
                left[left_index], right[right_index], complex_arithmetic);
            int64_t output_index = left_index + right_index;
            output[output_index].real += product.real;
            output[output_index].imag += product.imag;
        }
    }
}

CNP_API CnpArray* CNP_CALL cnp_polyfromroots(const CnpArray *roots) {
    const char *function_name = "cnp_polyfromroots";
    bool complex_result;
    int64_t n;
    size_t root_bytes;
    size_t polynomial_bytes;
    size_t arena_bytes;
    size_t scratch_bytes;
    CnpPolyFromRootsValue *root_values = NULL;
    CnpPolyFromRootsPolynomial *current_polynomials = NULL;
    CnpPolyFromRootsPolynomial *next_polynomials = NULL;
    CnpPolyFromRootsValue *current_coefficients = NULL;
    CnpPolyFromRootsValue *next_coefficients = NULL;
    CnpPolyFromRootsValue *scratch = NULL;
    CnpArray *result = NULL;
    int64_t shape[1];
    if (!roots) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "root array is required");
        return NULL;
    }
    if (roots->ndim == 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "root array must be one-dimensional");
        return NULL;
    }
    if (roots->shape[0] == 0) {
        int64_t identity_shape[1] = {1};
        result = cnp_array_ones(
            1, identity_shape, CNP_DOUBLE, CNP_ORDER_C);
        if (!result) cnp_relabel_error(function_name);
        return result;
    }
    if (roots->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "root array must be one-dimensional");
        return NULL;
    }
    if (roots->dtype->type_num == CNP_BOOL ||
            (!cnp_type_is_integer(roots->dtype->type_num) &&
             !cnp_type_is_float(roots->dtype->type_num) &&
             !cnp_type_is_complex(roots->dtype->type_num))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "Coefficient arrays have no common type");
        return NULL;
    }
    n = roots->size;
    complex_result = cnp_type_is_complex(roots->dtype->type_num);
    if ((uint64_t)n > SIZE_MAX / sizeof(*root_values) ||
            (uint64_t)n > SIZE_MAX / sizeof(*current_polynomials) ||
            (uint64_t)n > SIZE_MAX / (2 * sizeof(*current_coefficients)) ||
            (uint64_t)n + 1 > SIZE_MAX / sizeof(*scratch)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "polynomial workspace size overflows");
        return NULL;
    }
    root_bytes = (size_t)n * sizeof(*root_values);
    polynomial_bytes = (size_t)n * sizeof(*current_polynomials);
    arena_bytes = (size_t)(2 * n) * sizeof(*current_coefficients);
    scratch_bytes = (size_t)(n + 1) * sizeof(*scratch);
    root_values = (CnpPolyFromRootsValue*)cnp_malloc(root_bytes);
    current_polynomials = (CnpPolyFromRootsPolynomial*)cnp_malloc(
        polynomial_bytes);
    next_polynomials = (CnpPolyFromRootsPolynomial*)cnp_malloc(
        polynomial_bytes);
    current_coefficients = (CnpPolyFromRootsValue*)cnp_malloc(arena_bytes);
    next_coefficients = (CnpPolyFromRootsValue*)cnp_malloc(arena_bytes);
    scratch = (CnpPolyFromRootsValue*)cnp_malloc(scratch_bytes);
    if (!root_values || !current_polynomials || !next_polynomials ||
            !current_coefficients || !next_coefficients || !scratch) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate polynomial workspace");
        goto cleanup;
    }
    for (int64_t index = 0; index < n; ++index) {
        root_values[index] = cnp_polyfromroots_value_at(roots, index);
    }
    qsort(
        root_values, (size_t)n, sizeof(*root_values),
        cnp_polyfromroots_root_compare);
    for (int64_t index = 0; index < n; ++index) {
        size_t offset = (size_t)(2 * index);
        current_polynomials[index] =
            (CnpPolyFromRootsPolynomial){offset, 2};
        current_coefficients[offset] = (CnpPolyFromRootsValue){
            -root_values[index].real, -root_values[index].imag};
        current_coefficients[offset + 1] =
            (CnpPolyFromRootsValue){1.0, 0.0};
    }
    int64_t polynomial_count = n;
    while (polynomial_count > 1) {
        int64_t pair_count = polynomial_count / 2;
        bool has_remainder = polynomial_count % 2 != 0;
        size_t next_offset = 0;
        for (int64_t pair = 0; pair < pair_count; ++pair) {
            CnpPolyFromRootsPolynomial left = current_polynomials[pair];
            CnpPolyFromRootsPolynomial right =
                current_polynomials[pair + pair_count];
            int64_t pair_length = left.length + right.length - 1;
            next_polynomials[pair].offset = next_offset;
            if (has_remainder && pair == 0) {
                CnpPolyFromRootsPolynomial remainder =
                    current_polynomials[polynomial_count - 1];
                cnp_polyfromroots_multiply_polynomials(
                    current_coefficients + left.offset, left.length,
                    current_coefficients + right.offset, right.length,
                    scratch, complex_result);
                next_polynomials[pair].length =
                    pair_length + remainder.length - 1;
                cnp_polyfromroots_multiply_polynomials(
                    scratch, pair_length,
                    current_coefficients + remainder.offset,
                    remainder.length,
                    next_coefficients + next_offset,
                    complex_result);
            } else {
                next_polynomials[pair].length = pair_length;
                cnp_polyfromroots_multiply_polynomials(
                    current_coefficients + left.offset, left.length,
                    current_coefficients + right.offset, right.length,
                    next_coefficients + next_offset,
                    complex_result);
            }
            next_offset += (size_t)next_polynomials[pair].length;
        }
        CnpPolyFromRootsPolynomial *polynomial_swap = current_polynomials;
        CnpPolyFromRootsValue *coefficient_swap = current_coefficients;
        current_polynomials = next_polynomials;
        next_polynomials = polynomial_swap;
        current_coefficients = next_coefficients;
        next_coefficients = coefficient_swap;
        polynomial_count = pair_count;
    }

    shape[0] = n + 1;
    result = cnp_array_new(
        1, shape, complex_result ? CNP_CDOUBLE : CNP_DOUBLE, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        goto cleanup;
    }
    if (complex_result) {
        for (int64_t index = 0; index <= n; ++index) {
            CnpPolyFromRootsValue value = current_coefficients[
                current_polynomials[0].offset + (size_t)index];
            ((cnp_cdouble*)result->data)[index].real = value.real;
            ((cnp_cdouble*)result->data)[index].imag = value.imag;
        }
    } else {
        for (int64_t index = 0; index <= n; ++index) {
            ((double*)result->data)[index] = current_coefficients[
                current_polynomials[0].offset + (size_t)index].real;
        }
    }

cleanup:
    if (root_values) cnp_free(root_values, root_bytes);
    if (current_polynomials) {
        cnp_free(current_polynomials, polynomial_bytes);
    }
    if (next_polynomials) cnp_free(next_polynomials, polynomial_bytes);
    if (current_coefficients) {
        cnp_free(current_coefficients, arena_bytes);
    }
    if (next_coefficients) cnp_free(next_coefficients, arena_bytes);
    if (scratch) cnp_free(scratch, scratch_bytes);
    return result;
}

/* =========================================================================
 * cnp_mat - Create matrix from 2D array (alias)
 * numpy.mat(a) / numpy.asmatrix(a)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_mat(const CnpArray *arr) {
    const char *function_name = "cnp_mat";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return NULL;
    }
    if (arr->ndim < 0 || arr->ndim > 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input array must have at most two dimensions");
        return NULL;
    }

    int64_t shape[2] = {1, 1};
    if (arr->ndim == 1) {
        shape[1] = arr->shape[0];
    } else if (arr->ndim == 2) {
        shape[0] = arr->shape[0];
        shape[1] = arr->shape[1];
    }

    CnpArray *result = cnp_array_new(
        2, shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CNP_STATUS status = cnp_copyto(result, arr, CNP_CAST_NO);
    if (status != CNP_OK) {
        cnp_array_free(result);
        cnp_relabel_error(function_name);
        return NULL;
    }
    return result;
}

/* =========================================================================
 * cnp_bmat - Build matrix from blocks
 * numpy.bmat(obj)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_bmat(int nrows, int ncols, CnpArray **blocks) {
    CnpArray *result = cnp_block(nrows, ncols, blocks);
    if (!result) cnp_relabel_error("cnp_bmat");
    return result;
}

/* =========================================================================
 * cnp_matlib_rand - Random matrix
 * numpy.matlib.rand(rows, cols)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_matlib_rand(int64_t rows, int64_t cols) {
    int64_t shape[2] = {rows, cols};
    CnpArray *result = cnp_random_random(2, shape);
    if (!result) cnp_relabel_error("cnp_matlib_rand");
    return result;
}

/* =========================================================================
 * cnp_matlib_randn - Random normal matrix
 * numpy.matlib.randn(rows, cols)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_matlib_randn(int64_t rows, int64_t cols) {
    int64_t shape[2] = {rows, cols};
    CnpArray *result = cnp_random_standard_normal(2, shape);
    if (!result) cnp_relabel_error("cnp_matlib_randn");
    return result;
}

/* =========================================================================
 * cnp_matlib_eye - Identity matrix
 * numpy.matlib.eye(n, M=None, k=0)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_matlib_eye(int64_t n, int64_t m, int k) {
    const char *function_name = "cnp_matlib_eye";
    if (n < 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "matrix dimensions must not be negative");
        return NULL;
    }
    if (m <= 0) m = n;
    int64_t shape[2] = {n, m};
    CnpArray *result = cnp_array_zeros(2, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int64_t i = 0; i < n; i++) {
        int64_t j = i + k;
        if (j >= 0 && j < m) {
            ((double*)result->data)[i * m + j] = 1.0;
        }
    }
    return result;
}

/* =========================================================================
 * cnp_matlib_ones - Ones matrix
 * numpy.matlib.ones(shape)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_matlib_ones(int64_t rows, int64_t cols) {
    const char *function_name = "cnp_matlib_ones";
    if (rows < 0 || cols < 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "matrix dimensions must not be negative");
        return NULL;
    }
    int64_t shape[2] = {rows, cols};
    CnpArray *result = cnp_array_ones(
        2, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_matlib_zeros - Zeros matrix
 * numpy.matlib.zeros(shape)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_matlib_zeros(int64_t rows, int64_t cols) {
    const char *function_name = "cnp_matlib_zeros";
    if (rows < 0 || cols < 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "matrix dimensions must not be negative");
        return NULL;
    }
    int64_t shape[2] = {rows, cols};
    CnpArray *result = cnp_array_zeros(
        2, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_matlib_repmat - Repeat matrix
 * numpy.matlib.repmat(a, m, n)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_matlib_repmat(const CnpArray *arr, int64_t m, int64_t n) {
    const char *function_name = "cnp_matlib_repmat";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return NULL;
    }
    if (arr->ndim < 0 || arr->ndim > 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input array must have at most two dimensions");
        return NULL;
    }
    if (m < 0 || n < 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "repetition counts must not be negative");
        return NULL;
    }

    int64_t reps[2] = {m, n};
    CnpArray *result = cnp_tile(arr, 2, reps);
    if (!result) cnp_relabel_error(function_name);
    return result;
}
