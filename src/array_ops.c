/**
 * cnumpy additional array operations
 * Corresponds to numpy: atleast_1d/2d/3d, fliplr, flipud, lexsort, conj,
 *   allclose, isclose, array_equal, array_equiv, pinv, lstsq, slogdet,
 *   multi_dot, shares_memory
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>

/* =========================================================================
 * cnp_atleast_1d - Ensure array has at least 1 dimension
 * numpy.atleast_1d(*arys)
 * ========================================================================= */
static bool atleast_validate_array(
    const CnpArray *array, const char *function_name) {
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            array, function_name, &ignored_nbytes)) return false;
    if (array->ndim < 0 || array->ndim > CNP_MAXDIMS) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source array ndim must be between 0 and %d", CNP_MAXDIMS);
        return false;
    }
    if (!array->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array must have a data buffer");
        return false;
    }
    if (array->ndim > 0 && (!array->shape || !array->strides)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source array requires shape and strides");
        return false;
    }

    int64_t expected_size = 1;
    for (int axis = 0; axis < array->ndim; ++axis) {
        int64_t dimension = array->shape[axis];
        if (dimension < 0) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "source array shape contains a negative dimension at axis %d",
                axis);
            return false;
        }
        if (expected_size != 0 &&
                dimension > INT64_MAX / expected_size) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "source array shape product exceeds INT64_MAX");
            return false;
        }
        expected_size *= dimension;
    }
    if (expected_size != array->size) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source array size does not match its shape");
        return false;
    }
    return true;
}

static CnpArray *atleast_same_array_reference(CnpArray *array) {
    cnp_array_incref(array);
    return array;
}

CNP_API CnpArray* CNP_CALL cnp_atleast_1d(const CnpArray *arr) {
    const char *function_name = "cnp_atleast_1d";
    if (!atleast_validate_array(arr, function_name)) return NULL;
    if (arr->ndim >= 1)
        return atleast_same_array_reference((CnpArray*)arr);

    int64_t shape[1] = {1};
    int64_t strides[1] = {arr->dtype->elsize};
    CnpArray *result = cnp_array_view_from_metadata(
        (CnpArray*)arr, 1, shape, strides, arr->offset, 0);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_atleast_2d - Ensure array has at least 2 dimensions
 * numpy.atleast_2d(*arys)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_atleast_2d(const CnpArray *arr) {
    const char *function_name = "cnp_atleast_2d";
    if (!atleast_validate_array(arr, function_name)) return NULL;
    if (arr->ndim >= 2)
        return atleast_same_array_reference((CnpArray*)arr);

    int64_t shape[2];
    int64_t strides[2];
    if (arr->ndim == 1) {
        shape[0] = 1;
        shape[1] = arr->shape[0];
        strides[0] = 0;
        strides[1] = arr->strides[0];
    } else {
        shape[0] = 1;
        shape[1] = 1;
        strides[0] = arr->dtype->elsize;
        strides[1] = arr->dtype->elsize;
    }

    CnpArray *result = cnp_array_view_from_metadata(
        (CnpArray*)arr, 2, shape, strides, arr->offset, 0);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_atleast_3d - Ensure array has at least 3 dimensions
 * numpy.atleast_3d(*arys)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_atleast_3d(const CnpArray *arr) {
    const char *function_name = "cnp_atleast_3d";
    if (!atleast_validate_array(arr, function_name)) return NULL;
    if (arr->ndim >= 3)
        return atleast_same_array_reference((CnpArray*)arr);

    int64_t shape[3];
    int64_t strides[3];
    if (arr->ndim == 2) {
        shape[0] = arr->shape[0];
        shape[1] = arr->shape[1];
        shape[2] = 1;
        strides[0] = arr->strides[0];
        strides[1] = arr->strides[1];
        strides[2] = 0;
    } else if (arr->ndim == 1) {
        shape[0] = 1;
        shape[1] = arr->shape[0];
        shape[2] = 1;
        strides[0] = 0;
        strides[1] = arr->strides[0];
        strides[2] = 0;
    } else {
        shape[0] = 1;
        shape[1] = 1;
        shape[2] = 1;
        strides[0] = arr->dtype->elsize;
        strides[1] = arr->dtype->elsize;
        strides[2] = arr->dtype->elsize;
    }

    CnpArray *result = cnp_array_view_from_metadata(
        (CnpArray*)arr, 3, shape, strides, arr->offset, 0);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_fliplr - Flip array left-right (along axis 1)
 * numpy.fliplr(m)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_fliplr(const CnpArray *arr) {
    const char *function_name = "cnp_fliplr";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return NULL;
    }
    if (arr->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input array must have at least two dimensions");
        return NULL;
    }
    CnpArray *result = cnp_flip(arr, 1);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_flipud - Flip array up-down (along axis 0)
 * numpy.flipud(m)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_flipud(const CnpArray *arr) {
    const char *function_name = "cnp_flipud";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return NULL;
    }
    if (arr->ndim < 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input array must have at least one dimension");
        return NULL;
    }
    CnpArray *result = cnp_flip(arr, 0);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_allclose - Check if all elements are close
 * numpy.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
 * ========================================================================= */

static bool allclose_numeric_dtype(CNP_TYPE type) {
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

static CNP_STATUS allclose_validate_array(
    const CnpArray *array, const char *role, const char *function_name) {
    if (!array) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "%s array is required", role);
        return CNP_ERR_GENERIC;
    }
    if (!array->dtype || !allclose_numeric_dtype(array->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "%s array must have a numeric dtype", role);
        return CNP_ERR_TYPE;
    }
    if (array->ndim < 0 || array->ndim > CNP_MAXDIMS ||
            (array->ndim > 0 && (!array->shape || !array->strides))) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "%s array has invalid shape metadata", role);
        return CNP_ERR_SHAPE;
    }
    for (int axis = 0; axis < array->ndim; ++axis) {
        if (array->shape[axis] < 0) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "%s array has a negative dimension at axis %d", role, axis);
            return CNP_ERR_SHAPE;
        }
    }
    if (array->size > 0 && !array->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "%s array requires a data buffer", role);
        return CNP_ERR_GENERIC;
    }
    return CNP_OK;
}

static CNP_STATUS allclose_broadcast_shape(
    const CnpArray *left, const CnpArray *right,
    int *result_ndim, int64_t *result_shape, int64_t *result_size,
    const char *function_name) {
    *result_ndim = left->ndim > right->ndim ? left->ndim : right->ndim;
    *result_size = 1;
    for (int axis = 0; axis < *result_ndim; ++axis) {
        int left_axis = axis - (*result_ndim - left->ndim);
        int right_axis = axis - (*result_ndim - right->ndim);
        int64_t left_dimension =
            left_axis < 0 ? 1 : left->shape[left_axis];
        int64_t right_dimension =
            right_axis < 0 ? 1 : right->shape[right_axis];
        int64_t dimension;
        if (left_dimension == right_dimension) {
            dimension = left_dimension;
        } else if (left_dimension == 1) {
            dimension = right_dimension;
        } else if (right_dimension == 1) {
            dimension = left_dimension;
        } else {
            cnp_set_error(
                CNP_ERR_BROADCAST, function_name,
                "array shapes cannot broadcast at axis %d: %lld and %lld",
                axis, (long long)left_dimension,
                (long long)right_dimension);
            return CNP_ERR_BROADCAST;
        }
        result_shape[axis] = dimension;
        if (*result_size != 0 && dimension != 0) {
            if (*result_size > INT64_MAX / dimension) {
                cnp_set_error(
                    CNP_ERR_SHAPE, function_name,
                    "broadcast result size overflows int64");
                return CNP_ERR_SHAPE;
            }
            *result_size *= dimension;
        } else {
            *result_size = 0;
        }
    }
    return CNP_OK;
}

static int64_t allclose_broadcast_offset(
    const CnpArray *array, const int64_t *coordinates, int result_ndim) {
    int result_axis = result_ndim - array->ndim;
    int64_t offset = array->offset;
    for (int axis = 0; axis < array->ndim; ++axis, ++result_axis) {
        int64_t coordinate =
            array->shape[axis] == 1 ? 0 : coordinates[result_axis];
        offset += coordinate * array->strides[axis];
    }
    return offset;
}

static CNP_STATUS allclose_read_value(
    const CnpArray *array, int64_t offset,
    cnp_clongdouble *value, const char *function_name) {
    return cnp_cast_scalar_value(
        (const char*)array->data + offset,
        array->dtype->type_num,
        value,
        CNP_CLONGDOUBLE,
        function_name);
}

typedef enum {
    CNP_ALLCLOSE_FINITE,
    CNP_ALLCLOSE_INFINITE,
    CNP_ALLCLOSE_NAN
} CnpAllcloseFpClass;

static CnpAllcloseFpClass allclose_classify(long double value) {
#if defined(_MSC_VER)
    double converted = (double)value;
    uint64_t bits;
    memcpy(&bits, &converted, sizeof(bits));
    uint64_t exponent = bits & UINT64_C(0x7ff0000000000000);
    if (exponent != UINT64_C(0x7ff0000000000000))
        return CNP_ALLCLOSE_FINITE;
    return (bits & UINT64_C(0x000fffffffffffff)) != 0
        ? CNP_ALLCLOSE_NAN : CNP_ALLCLOSE_INFINITE;
#else
    int classification = fpclassify(value);
    if (classification == FP_NAN) return CNP_ALLCLOSE_NAN;
    if (classification == FP_INFINITE) return CNP_ALLCLOSE_INFINITE;
    return CNP_ALLCLOSE_FINITE;
#endif
}

static bool allclose_complex_dtype(CNP_TYPE type) {
    return type == CNP_CFLOAT ||
        type == CNP_CDOUBLE || type == CNP_CLONGDOUBLE;
}

static bool allclose_values_match(
    cnp_clongdouble left, cnp_clongdouble right,
    bool left_complex, bool right_complex,
    double rtol, double atol, bool equal_nan) {
    CnpAllcloseFpClass left_real = allclose_classify(left.real);
    CnpAllcloseFpClass left_imaginary = allclose_classify(left.imag);
    CnpAllcloseFpClass right_real = allclose_classify(right.real);
    CnpAllcloseFpClass right_imaginary = allclose_classify(right.imag);
    bool left_nan = left_real == CNP_ALLCLOSE_NAN ||
        left_imaginary == CNP_ALLCLOSE_NAN;
    bool right_nan = right_real == CNP_ALLCLOSE_NAN ||
        right_imaginary == CNP_ALLCLOSE_NAN;
    bool left_infinite = left_real == CNP_ALLCLOSE_INFINITE ||
        left_imaginary == CNP_ALLCLOSE_INFINITE;
    bool right_infinite = right_real == CNP_ALLCLOSE_INFINITE ||
        right_imaginary == CNP_ALLCLOSE_INFINITE;
    bool left_finite = !left_nan && !left_infinite;
    bool right_finite = !right_nan && !right_infinite;
    if (!left_finite || !right_finite) {
        bool left_effective_nan =
            left_nan || (left_complex && left_infinite);
        bool right_effective_nan =
            right_nan || (right_complex && right_infinite);
        if (equal_nan && left_effective_nan && right_effective_nan)
            return true;
        if (left_effective_nan || right_effective_nan) return false;
        return left.real == right.real && left.imag == right.imag;
    }
    long double difference = hypotl(
        left.real - right.real, left.imag - right.imag);
    long double reference = hypotl(right.real, right.imag);
    long double tolerance =
        (long double)atol + (long double)rtol * reference;
    return difference <= tolerance;
}

static bool allclose_contiguous_f64(
    const CnpArray *left, const CnpArray *right,
    double rtol, double atol, bool equal_nan, bool *result) {
    if (left->dtype->type_num != CNP_DOUBLE ||
            right->dtype->type_num != CNP_DOUBLE ||
            !(left->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            !(right->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            left->ndim != right->ndim || left->size != right->size)
        return false;
    for (int axis = 0; axis < left->ndim; ++axis) {
        if (left->shape[axis] != right->shape[axis]) return false;
    }

    const double *left_data = (const double*)(
        (const char*)left->data + left->offset);
    const double *right_data = (const double*)(
        (const char*)right->data + right->offset);
    for (int64_t index = 0; index < left->size; ++index) {
        double left_value = left_data[index];
        double right_value = right_data[index];
        CnpAllcloseFpClass left_class = allclose_classify(left_value);
        CnpAllcloseFpClass right_class = allclose_classify(right_value);
        bool matches;
        if (left_class == CNP_ALLCLOSE_FINITE &&
                right_class == CNP_ALLCLOSE_FINITE) {
            matches = fabs(left_value - right_value) <=
                atol + rtol * fabs(right_value);
        } else if (left_class == CNP_ALLCLOSE_NAN ||
                right_class == CNP_ALLCLOSE_NAN) {
            matches = equal_nan &&
                left_class == CNP_ALLCLOSE_NAN &&
                right_class == CNP_ALLCLOSE_NAN;
        } else {
            matches = left_value == right_value;
        }
        if (!matches) {
            *result = false;
            return true;
        }
    }
    *result = true;
    return true;
}

CNP_API CNP_STATUS CNP_CALL cnp_allclose_v2(
    const CnpArray *a, const CnpArray *b,
    double rtol, double atol, bool equal_nan, bool *result) {
    const char *function_name = "cnp_allclose_v2";
    if (!result) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "result output is required");
        return CNP_ERR_GENERIC;
    }
    *result = false;
    CNP_STATUS status = allclose_validate_array(
        a, "left", function_name);
    if (status != CNP_OK) return status;
    status = allclose_validate_array(b, "right", function_name);
    if (status != CNP_OK) return status;

    int result_ndim;
    int64_t result_shape[CNP_MAXDIMS];
    int64_t result_size;
    status = allclose_broadcast_shape(
        a, b, &result_ndim, result_shape, &result_size, function_name);
    if (status != CNP_OK) return status;
    if (allclose_contiguous_f64(
            a, b, rtol, atol, equal_nan, result))
        return CNP_OK;

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result_size; ++index) {
        int64_t left_offset = allclose_broadcast_offset(
            a, coordinates, result_ndim);
        int64_t right_offset = allclose_broadcast_offset(
            b, coordinates, result_ndim);
        cnp_clongdouble left_value;
        cnp_clongdouble right_value;
        status = allclose_read_value(
            a, left_offset, &left_value, function_name);
        if (status != CNP_OK) return status;
        status = allclose_read_value(
            b, right_offset, &right_value, function_name);
        if (status != CNP_OK) return status;
        if (!allclose_values_match(
                left_value, right_value,
                allclose_complex_dtype(a->dtype->type_num),
                allclose_complex_dtype(b->dtype->type_num),
                rtol, atol, equal_nan))
            return CNP_OK;

        for (int axis = result_ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < result_shape[axis]) break;
            coordinates[axis] = 0;
        }
    }
    *result = true;
    return CNP_OK;
}

CNP_API bool CNP_CALL cnp_allclose(
    const CnpArray *a, const CnpArray *b, double rtol, double atol) {
    bool result = false;
    CNP_STATUS status = cnp_allclose_v2(
        a, b, rtol, atol, false, &result);
    if (status != CNP_OK) {
        cnp_relabel_error("cnp_allclose");
        return false;
    }
    return result;
}

/* =========================================================================
 * cnp_isclose - Element-wise close check
 * numpy.isclose(a, b, rtol=1e-05, atol=1e-08)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_isclose(
    const CnpArray *a, const CnpArray *b, double rtol, double atol) {
    const char *function_name = "cnp_isclose";
    CNP_STATUS status = allclose_validate_array(a, "left", function_name);
    if (status != CNP_OK) return NULL;
    status = allclose_validate_array(b, "right", function_name);
    if (status != CNP_OK) return NULL;

    int result_ndim;
    int64_t result_shape[CNP_MAXDIMS];
    int64_t result_size;
    status = allclose_broadcast_shape(
        a, b, &result_ndim, result_shape, &result_size, function_name);
    if (status != CNP_OK) return NULL;
    CnpArray *result = cnp_array_new(
        result_ndim, result_shape, CNP_BOOL, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    uint8_t *output = (uint8_t*)result->data;
    for (int64_t index = 0; index < result_size; ++index) {
        cnp_clongdouble left_value;
        cnp_clongdouble right_value;
        status = allclose_read_value(
            a, allclose_broadcast_offset(a, coordinates, result_ndim),
            &left_value, function_name);
        if (status != CNP_OK) break;
        status = allclose_read_value(
            b, allclose_broadcast_offset(b, coordinates, result_ndim),
            &right_value, function_name);
        if (status != CNP_OK) break;
        output[index] = (uint8_t)allclose_values_match(
            left_value, right_value,
            allclose_complex_dtype(a->dtype->type_num),
            allclose_complex_dtype(b->dtype->type_num),
            rtol, atol, false);

        for (int axis = result_ndim - 1; axis >= 0; --axis) {
            ++coordinates[axis];
            if (coordinates[axis] < result_shape[axis]) break;
            coordinates[axis] = 0;
        }
    }
    if (status != CNP_OK) {
        cnp_array_free(result);
        cnp_relabel_error(function_name);
        return NULL;
    }
    return result;
}

/* =========================================================================
 * cnp_array_equal - Check if two arrays have same shape and elements
 * numpy.array_equal(a1, a2)
 * ========================================================================= */
static bool array_relation_all_true(
    CnpArray *comparison, const char *function_name) {
    if (!comparison) {
        cnp_relabel_error(function_name);
        return false;
    }
    bool result = true;
    for (int64_t index = 0; index < comparison->size; ++index) {
        if (!cnp_array_flat_get(comparison, index)) {
            result = false;
            break;
        }
    }
    cnp_array_free(comparison);
    return result;
}

static bool array_relation_shapes_broadcast(
    const CnpArray *left, const CnpArray *right) {
    int ndim = left->ndim > right->ndim ? left->ndim : right->ndim;
    for (int axis = 0; axis < ndim; ++axis) {
        int left_axis = axis - (ndim - left->ndim);
        int right_axis = axis - (ndim - right->ndim);
        int64_t left_dimension =
            left_axis < 0 ? 1 : left->shape[left_axis];
        int64_t right_dimension =
            right_axis < 0 ? 1 : right->shape[right_axis];
        if (left_dimension != right_dimension &&
                left_dimension != 1 && right_dimension != 1)
            return false;
    }
    return true;
}

CNP_API bool CNP_CALL cnp_array_equal(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_array_equal";
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both input arrays are required");
        return false;
    }
    if (a->ndim != b->ndim) return false;
    for (int dimension = 0; dimension < a->ndim; ++dimension) {
        if (a->shape[dimension] != b->shape[dimension]) return false;
    }
    return array_relation_all_true(cnp_equal(a, b), function_name);
}

/* =========================================================================
 * cnp_array_equiv - Check if arrays are broadcastable and equal
 * numpy.array_equiv(a1, a2)
 * ========================================================================= */
CNP_API bool CNP_CALL cnp_array_equiv(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_array_equiv";
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both input arrays are required");
        return false;
    }
    if (!array_relation_shapes_broadcast(a, b)) return false;
    return array_relation_all_true(cnp_equal(a, b), function_name);
}

/* =========================================================================
 * cnp_lexsort - Stable indirect sort using a sequence of keys
 * numpy.lexsort(keys, axis=-1)
 * ========================================================================= */
static bool lexsort_dtype_is_supported(CNP_TYPE dtype) {
    return dtype == CNP_BOOL || cnp_type_is_integer(dtype) ||
           cnp_type_is_float(dtype) || cnp_type_is_complex(dtype);
}

static bool lexsort_validate_key(
    const CnpArray *key, int index, const char *function_name) {
    int64_t ignored_nbytes;
    if (!key) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "key %d must not be NULL", index);
        return false;
    }
    if (!cnp_array_nbytes_checked(
            key, function_name, &ignored_nbytes)) return false;
    if (key->ndim < 0 || key->ndim > CNP_MAXDIMS) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "key %d ndim must be between 0 and %d",
            index, CNP_MAXDIMS);
        return false;
    }
    if (key->ndim > 0 && (!key->shape || !key->strides)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "key %d requires shape and strides", index);
        return false;
    }
    int64_t logical_size = 1;
    for (int dimension = 0; dimension < key->ndim; ++dimension) {
        int64_t extent = key->shape[dimension];
        if (extent < 0 || (extent > 0 && logical_size > INT64_MAX / extent)) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "key %d has an invalid shape", index);
            return false;
        }
        logical_size *= extent;
    }
    if (logical_size != key->size) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "key %d shape does not match its size", index);
        return false;
    }
    if (key->size > 0 && !key->data) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "key %d must have a data buffer", index);
        return false;
    }
    if (!lexsort_dtype_is_supported(key->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "key %d dtype %d is not a supported numeric dtype",
            index, (int)key->dtype->type_num);
        return false;
    }
    return true;
}

static int64_t lexsort_slice_base(
    const CnpArray *array, int axis, int64_t outer, int64_t inner) {
    int64_t offset = array->offset;
    for (int dimension = axis - 1; dimension >= 0; --dimension) {
        int64_t coordinate = outer % array->shape[dimension];
        outer /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    for (int dimension = array->ndim - 1;
         dimension > axis; --dimension) {
        int64_t coordinate = inner % array->shape[dimension];
        inner /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

static CNP_STATUS lexsort_compare_indices(
    int nkeys, const CnpArray **keys, const int64_t *base_offsets,
    int axis, int64_t left, int64_t right, int *order,
    const char *function_name) {
    for (int key_index = nkeys - 1; key_index >= 0; --key_index) {
        const CnpArray *key = keys[key_index];
        int64_t left_offset = base_offsets[key_index];
        int64_t right_offset = base_offsets[key_index];
        if (key->ndim > 0) {
            left_offset += left * key->strides[axis];
            right_offset += right * key->strides[axis];
        }
        CNP_STATUS status = cnp_compare_numeric_elements(
            (const char*)key->data + left_offset,
            key->dtype->type_num,
            (const char*)key->data + right_offset,
            key->dtype->type_num,
            key->dtype->type_num,
            order,
            function_name);
        if (status != CNP_OK || *order != 0) return status;
    }
    *order = 0;
    return CNP_OK;
}

static CNP_STATUS lexsort_merge_indices(
    int nkeys, const CnpArray **keys, const int64_t *base_offsets,
    int axis, int64_t count, int64_t *indices, int64_t *temporary,
    int64_t **sorted_indices, const char *function_name) {
    int64_t *source = indices;
    int64_t *destination = temporary;
    for (int64_t width = 1; width < count;) {
        for (int64_t start = 0; start < count; start += width * 2) {
            int64_t middle = start + width;
            int64_t end = start + width * 2;
            if (middle > count) middle = count;
            if (end > count) end = count;
            int64_t left = start;
            int64_t right = middle;
            int64_t output = start;
            while (left < middle && right < end) {
                int order = 0;
                CNP_STATUS status = lexsort_compare_indices(
                    nkeys, keys, base_offsets, axis,
                    source[left], source[right], &order, function_name);
                if (status != CNP_OK) return status;
                if (order <= 0) destination[output++] = source[left++];
                else destination[output++] = source[right++];
            }
            while (left < middle) destination[output++] = source[left++];
            while (right < end) destination[output++] = source[right++];
        }
        int64_t *swap = source;
        source = destination;
        destination = swap;
        if (width > count / 2) break;
        width *= 2;
    }
    *sorted_indices = source;
    return CNP_OK;
}

#define CNP_LEXSORT_RADIX_BITS 11
#define CNP_LEXSORT_RADIX_SIZE (1 << CNP_LEXSORT_RADIX_BITS)
#define CNP_LEXSORT_RADIX_MASK (CNP_LEXSORT_RADIX_SIZE - 1)
#define CNP_LEXSORT_RADIX_PASSES 6
#define CNP_LEXSORT_RADIX_THRESHOLD 512

static bool lexsort_contiguous_float64(
    int nkeys, const CnpArray **keys, int axis,
    int64_t outer, int64_t axis_length, CnpArray *result,
    CNP_STATUS *status, const char *function_name) {
    const CnpArray *first = keys[0];
    if (first->ndim == 0 || axis != first->ndim - 1 ||
            axis_length < CNP_LEXSORT_RADIX_THRESHOLD)
        return false;
    for (int key_index = 0; key_index < nkeys; ++key_index) {
        const CnpArray *key = keys[key_index];
        if (!(key->flags & CNP_ARRAY_C_CONTIGUOUS) ||
                key->dtype->type_num != CNP_DOUBLE ||
                key->strides[axis] != (int64_t)sizeof(double))
            return false;
    }

    size_t index_bytes = (size_t)axis_length * sizeof(int64_t);
    size_t key_bytes = (size_t)axis_length * sizeof(uint64_t);
    int64_t *indices_a = (int64_t*)cnp_malloc(index_bytes);
    int64_t *indices_b = (int64_t*)cnp_malloc(index_bytes);
    uint64_t *keys_a = (uint64_t*)cnp_malloc(key_bytes);
    uint64_t *keys_b = (uint64_t*)cnp_malloc(key_bytes);
    if (!indices_a || !indices_b || !keys_a || !keys_b) {
        if (indices_a) cnp_free(indices_a, index_bytes);
        if (indices_b) cnp_free(indices_b, index_bytes);
        if (keys_a) cnp_free(keys_a, key_bytes);
        if (keys_b) cnp_free(keys_b, key_bytes);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate lexsort radix storage");
        *status = CNP_ERR_MEMORY;
        return true;
    }

    int64_t counts[CNP_LEXSORT_RADIX_SIZE];
    *status = CNP_OK;
    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
        for (int64_t index = 0; index < axis_length; ++index)
            indices_a[index] = index;

        for (int key_index = 0; key_index < nkeys; ++key_index) {
            const double *data = (const double*)(
                (const char*)keys[key_index]->data +
                keys[key_index]->offset +
                outer_index * axis_length * sizeof(double));
            for (int64_t index = 0; index < axis_length; ++index) {
                keys_a[index] = cnp_double_to_sortable(
                    data[indices_a[index]]);
            }

            uint64_t *key_input = keys_a;
            uint64_t *key_output = keys_b;
            int64_t *index_input = indices_a;
            int64_t *index_output = indices_b;
            for (int pass = 0; pass < CNP_LEXSORT_RADIX_PASSES; ++pass) {
                int shift = pass * CNP_LEXSORT_RADIX_BITS;
                memset(counts, 0, sizeof(counts));
                for (int64_t index = 0; index < axis_length; ++index) {
                    ++counts[
                        (key_input[index] >> shift) &
                        CNP_LEXSORT_RADIX_MASK];
                }
                int64_t total = 0;
                for (int bucket = 0;
                     bucket < CNP_LEXSORT_RADIX_SIZE; ++bucket) {
                    int64_t count = counts[bucket];
                    counts[bucket] = total;
                    total += count;
                }
                for (int64_t index = 0; index < axis_length; ++index) {
                    int bucket = (int)(
                        (key_input[index] >> shift) &
                        CNP_LEXSORT_RADIX_MASK);
                    int64_t destination = counts[bucket]++;
                    key_output[destination] = key_input[index];
                    index_output[destination] = index_input[index];
                }
                uint64_t *key_swap = key_input;
                key_input = key_output;
                key_output = key_swap;
                int64_t *index_swap = index_input;
                index_input = index_output;
                index_output = index_swap;
            }
            if (key_input != keys_a || index_input != indices_a) {
                cnp_set_error(
                    CNP_ERR_GENERIC, function_name,
                    "lexsort radix pass parity is invalid");
                *status = CNP_ERR_GENERIC;
                break;
            }
        }
        if (*status != CNP_OK) break;
        memcpy(
            (char*)result->data + result->offset +
                outer_index * axis_length * sizeof(int64_t),
            indices_a, index_bytes);
    }

    cnp_free(indices_a, index_bytes);
    cnp_free(indices_b, index_bytes);
    cnp_free(keys_a, key_bytes);
    cnp_free(keys_b, key_bytes);
    return true;
}

static CnpArray *lexsort_impl(
    int nkeys, const CnpArray **keys, int axis,
    const char *function_name) {
    if (nkeys <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "lexsort requires at least one key");
        return NULL;
    }
    if (!keys) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "keys must not be NULL");
        return NULL;
    }
    for (int index = 0; index < nkeys; ++index) {
        if (!lexsort_validate_key(keys[index], index, function_name))
            return NULL;
    }

    const CnpArray *first = keys[0];
    for (int key_index = 1; key_index < nkeys; ++key_index) {
        const CnpArray *key = keys[key_index];
        if (key->ndim != first->ndim) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "all keys must have the same shape");
            return NULL;
        }
        for (int dimension = 0; dimension < first->ndim; ++dimension) {
            if (key->shape[dimension] != first->shape[dimension]) {
                cnp_set_error(
                    CNP_ERR_SHAPE, function_name,
                    "all keys must have the same shape");
                return NULL;
            }
        }
    }

    int resolved_axis;
    if (!cnp_reduction_resolve_axis(
            first, axis, false, function_name, &resolved_axis)) return NULL;
    CnpArray *result = cnp_array_new(
        first->ndim, first->shape, CNP_LONGLONG, CNP_ORDER_C);
    if (!result) return NULL;

    int actual_axis = resolved_axis;
    int64_t outer = 1;
    int64_t inner = 1;
    int64_t axis_length = 1;
    if (first->ndim > 0) {
        axis_length = first->shape[actual_axis];
        for (int dimension = 0; dimension < actual_axis; ++dimension)
            outer *= first->shape[dimension];
        for (int dimension = actual_axis + 1;
             dimension < first->ndim; ++dimension)
            inner *= first->shape[dimension];
    }
    if (axis_length == 0 || outer == 0 || inner == 0) return result;

    if (axis_length == 1) {
        int64_t *output = (int64_t*)result->data;
        for (int64_t index = 0; index < result->size; ++index)
            output[index] = 0;
        return result;
    }

    if ((uint64_t)axis_length > SIZE_MAX / sizeof(int64_t) ||
            (uint64_t)nkeys > SIZE_MAX / sizeof(int64_t)) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "lexsort temporary storage size is not representable");
        return NULL;
    }

    CNP_STATUS fast_status = CNP_OK;
    if (lexsort_contiguous_float64(
            nkeys, keys, actual_axis, outer, axis_length,
            result, &fast_status, function_name)) {
        if (fast_status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        return result;
    }

    size_t index_bytes = (size_t)axis_length * sizeof(int64_t);
    size_t base_bytes = (size_t)nkeys * sizeof(int64_t);
    int64_t *indices = (int64_t*)cnp_malloc(index_bytes);
    int64_t *temporary = (int64_t*)cnp_malloc(index_bytes);
    int64_t *base_offsets = (int64_t*)cnp_malloc(base_bytes);
    if (!indices || !temporary || !base_offsets) {
        if (indices) cnp_free(indices, index_bytes);
        if (temporary) cnp_free(temporary, index_bytes);
        if (base_offsets) cnp_free(base_offsets, base_bytes);
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate lexsort temporary storage");
        return NULL;
    }

    CNP_STATUS status = CNP_OK;
    for (int64_t outer_index = 0;
         outer_index < outer && status == CNP_OK; ++outer_index) {
        for (int64_t inner_index = 0;
             inner_index < inner && status == CNP_OK; ++inner_index) {
            for (int key_index = 0; key_index < nkeys; ++key_index) {
                base_offsets[key_index] = keys[key_index]->ndim == 0
                    ? keys[key_index]->offset
                    : lexsort_slice_base(
                        keys[key_index], actual_axis,
                        outer_index, inner_index);
            }
            for (int64_t index = 0; index < axis_length; ++index)
                indices[index] = index;

            int64_t *sorted_indices = NULL;
            status = lexsort_merge_indices(
                nkeys, keys, base_offsets, actual_axis,
                axis_length, indices, temporary,
                &sorted_indices, function_name);
            if (status != CNP_OK) break;

            int64_t result_base = first->ndim == 0
                ? result->offset
                : lexsort_slice_base(
                    result, actual_axis, outer_index, inner_index);
            for (int64_t position = 0;
                 position < axis_length; ++position) {
                int64_t result_offset = result_base;
                if (result->ndim > 0)
                    result_offset += position * result->strides[actual_axis];
                *(int64_t*)((char*)result->data + result_offset) =
                    sorted_indices[position];
            }
        }
    }

    cnp_free(indices, index_bytes);
    cnp_free(temporary, index_bytes);
    cnp_free(base_offsets, base_bytes);
    if (status != CNP_OK) {
        cnp_array_free(result);
        return NULL;
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_lexsort_v2(
    int nkeys, const CnpArray **keys, int axis) {
    return lexsort_impl(nkeys, keys, axis, "cnp_lexsort_v2");
}

CNP_API CnpArray* CNP_CALL cnp_lexsort(
    int nkeys, const CnpArray **keys) {
    return lexsort_impl(nkeys, keys, -1, "cnp_lexsort");
}

/* =========================================================================
 * cnp_pinv - Moore-Penrose pseudo-inverse
 * numpy.linalg.pinv(a, rcond=1e-15)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_pinv(const CnpArray *a, double rcond) {
    CnpArray *result = cnp_linalg_pinv(a, rcond);
    if (!result) cnp_relabel_error("cnp_pinv");
    return result;
}

/* =========================================================================
 * cnp_lstsq - Least squares solution
 * numpy.linalg.lstsq(a, b, rcond=None)
 * Returns x that minimizes ||Ax - b||^2
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_lstsq(const CnpArray *a, const CnpArray *b, double rcond) {
    const char *function_name = "cnp_lstsq";
    CnpArray *x = NULL;
    CnpArray *residuals = NULL;
    CnpArray *rank = NULL;
    CnpArray *singular_values = NULL;
    CNP_STATUS status = cnp_linalg_lstsq_v2(
        a, b, rcond, false,
        &x, &residuals, &rank, &singular_values);
    if (residuals) cnp_array_free(residuals);
    if (rank) cnp_array_free(rank);
    if (singular_values) cnp_array_free(singular_values);
    if (status != CNP_OK) {
        if (x) cnp_array_free(x);
        cnp_relabel_error(function_name);
        return NULL;
    }
    return x;
}

/* =========================================================================
 * cnp_slogdet - Sign and log of determinant
 * numpy.linalg.slogdet(a) -> (sign, logdet)
 * ========================================================================= */
CNP_API CNP_STATUS CNP_CALL cnp_slogdet(const CnpArray *a, double *sign, double *logdet) {
    const char *function_name = "cnp_slogdet";
    CnpArray *sign_array = NULL;
    CnpArray *logabsdet_array = NULL;
    double sign_value;
    double logabsdet_value;
    CNP_STATUS status;

    if (sign) *sign = 0.0;
    if (logdet) *logdet = 0.0;
    if (!sign || !logdet || sign == logdet) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "distinct sign and logabsdet outputs are required");
        return CNP_ERR_GENERIC;
    }
    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array must not be null");
        return CNP_ERR_GENERIC;
    }
    if (a->ndim != 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "legacy scalar outputs require a two-dimensional matrix");
        return CNP_ERR_SHAPE;
    }
    if (cnp_type_is_complex(a->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "legacy double sign output cannot represent complex phase");
        return CNP_ERR_TYPE;
    }

    status = cnp_linalg_slogdet_v2(
        a, &sign_array, &logabsdet_array);
    if (status != CNP_OK) {
        if (sign_array) cnp_array_free(sign_array);
        if (logabsdet_array) cnp_array_free(logabsdet_array);
        cnp_relabel_error(function_name);
        return status;
    }
    sign_value = cnp_get_element_double(
        sign_array->data, 0, sign_array->dtype->type_num);
    logabsdet_value = cnp_get_element_double(
        logabsdet_array->data, 0, logabsdet_array->dtype->type_num);
    cnp_array_free(sign_array);
    cnp_array_free(logabsdet_array);
    *sign = sign_value;
    *logdet = logabsdet_value;
    return CNP_OK;
}

/* =========================================================================
 * cnp_multi_dot - Chain matrix multiplication
 * numpy.linalg.multi_dot(arrays)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_multi_dot(int narrays, const CnpArray **arrays) {
    const char *function_name = "cnp_multi_dot";
    if (narrays < 2) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "multi_dot requires at least two arrays");
        return NULL;
    }
    if (!arrays) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "array pointer list must not be null");
        return NULL;
    }
    for (int index = 0; index < narrays; ++index) {
        if (!arrays[index]) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "array %d must not be null", index);
            return NULL;
        }
        bool endpoint = index == 0 || index == narrays - 1;
        if (arrays[index]->ndim != 2 &&
                !(endpoint && arrays[index]->ndim == 1)) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "array %d must be two-dimensional%s",
                index, endpoint ? " or a one-dimensional endpoint" : "");
            return NULL;
        }
        if (index > 0) {
            const CnpArray *previous = arrays[index - 1];
            int64_t previous_columns = previous->shape[previous->ndim - 1];
            int64_t current_rows = arrays[index]->shape[0];
            if (previous_columns != current_rows) {
                cnp_set_error(
                    CNP_ERR_SHAPE, function_name,
                    "arrays %d and %d have unaligned dimensions %lld and %lld",
                    index - 1, index,
                    (long long)previous_columns, (long long)current_rows);
                return NULL;
            }
        }
    }
    CnpArray *result = cnp_matmul(arrays[0], arrays[1]);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int i = 2; i < narrays && result; i++) {
        CnpArray *tmp = cnp_matmul(result, arrays[i]);
        cnp_array_free(result);
        result = tmp;
    }
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_shares_memory - Check if two arrays share memory
 * numpy.shares_memory(a, b)
 * ========================================================================= */
static bool memory_relation_validate(
    const CnpArray *array, const char *role, const char *function_name) {
    if (!array) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "%s array is required", role);
        return false;
    }
    if (!array->dtype || !array->data || array->ndim < 0 ||
            array->ndim > CNP_MAXDIMS ||
            (array->ndim > 0 && (!array->shape || !array->strides))) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "%s array metadata and data buffer must be valid", role);
        return false;
    }
    return true;
}

static int64_t memory_relation_flat_offset(
    const CnpArray *array, int64_t flat_index) {
    int64_t offset = array->offset;
    for (int dimension = array->ndim - 1; dimension >= 0; --dimension) {
        int64_t coordinate = flat_index % array->shape[dimension];
        flat_index /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

static int memory_relation_compare_address(const void *left, const void *right) {
    uintptr_t left_value = *(const uintptr_t*)left;
    uintptr_t right_value = *(const uintptr_t*)right;
    return left_value < right_value ? -1 : left_value > right_value ? 1 : 0;
}

static bool memory_relation_bounds_overlap(
    const CnpArray *left, const CnpArray *right,
    const char *function_name, bool *overlap) {
    void *left_low = NULL;
    void *left_high = NULL;
    void *right_low = NULL;
    void *right_high = NULL;
    CNP_STATUS status = cnp_byte_bounds(left, &left_low, &left_high);
    if (status != CNP_OK) {
        cnp_relabel_error(function_name);
        return false;
    }
    status = cnp_byte_bounds(right, &right_low, &right_high);
    if (status != CNP_OK) {
        cnp_relabel_error(function_name);
        return false;
    }
    *overlap = (uintptr_t)left_low < (uintptr_t)right_high &&
        (uintptr_t)right_low < (uintptr_t)left_high;
    return true;
}

CNP_API bool CNP_CALL cnp_shares_memory(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_shares_memory";
    if (!memory_relation_validate(a, "left", function_name) ||
            !memory_relation_validate(b, "right", function_name)) return false;
    if (a->size == 0 || b->size == 0) return false;

    bool bounds_overlap;
    if (!memory_relation_bounds_overlap(
            a, b, function_name, &bounds_overlap)) return false;
    if (!bounds_overlap) return false;

    const CnpArray *indexed = a->size <= b->size ? a : b;
    const CnpArray *probed = indexed == a ? b : a;
    if ((uint64_t)indexed->size > SIZE_MAX / sizeof(uintptr_t)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "address table size is not representable");
        return false;
    }
    size_t address_bytes = (size_t)indexed->size * sizeof(uintptr_t);
    uintptr_t *addresses = (uintptr_t*)cnp_malloc(address_bytes);
    if (!addresses) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate exact overlap address table");
        return false;
    }
    for (int64_t index = 0; index < indexed->size; ++index) {
        addresses[index] = (uintptr_t)((char*)indexed->data +
            memory_relation_flat_offset(indexed, index));
    }
    qsort(
        addresses, (size_t)indexed->size, sizeof(uintptr_t),
        memory_relation_compare_address);

    uintptr_t indexed_width = (uintptr_t)indexed->dtype->elsize;
    uintptr_t probed_width = (uintptr_t)probed->dtype->elsize;
    bool overlap = false;
    for (int64_t index = 0; index < probed->size && !overlap; ++index) {
        uintptr_t low = (uintptr_t)((char*)probed->data +
            memory_relation_flat_offset(probed, index));
        uintptr_t high = low + probed_width;
        size_t left = 0;
        size_t right = (size_t)indexed->size;
        while (left < right) {
            size_t middle = left + (right - left) / 2;
            if (addresses[middle] < low) left = middle + 1;
            else right = middle;
        }
        if (left < (size_t)indexed->size && addresses[left] < high)
            overlap = true;
        if (!overlap && left > 0 &&
                addresses[left - 1] + indexed_width > low)
            overlap = true;
    }
    cnp_free(addresses, address_bytes);
    return overlap;
}

/* =========================================================================
 * cnp_may_share_memory - Check if arrays might share memory (bounds check)
 * numpy.may_share_memory(a, b)
 * ========================================================================= */
CNP_API bool CNP_CALL cnp_may_share_memory(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_may_share_memory";
    if (!memory_relation_validate(a, "left", function_name) ||
            !memory_relation_validate(b, "right", function_name)) return false;
    if (a->size == 0 || b->size == 0) return false;
    bool overlap;
    if (!memory_relation_bounds_overlap(
            a, b, function_name, &overlap)) return false;
    return overlap;
}

/* =========================================================================
 * cnp_msort - Sort along first axis
 * numpy.msort(a)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_msort(const CnpArray *arr) {
    const char *function_name = "cnp_msort";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "arr must not be NULL");
        return NULL;
    }
    CnpArray *result = cnp_sort_v2(
        arr, 0, false, CNP_SORT_QUICKSORT);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_swapaxes_2d - Swap axes for 2D (transpose)
 * Already exists as cnp_swapaxes in shape.c - skip
 * ========================================================================= */

/* cnp_broadcast_shapes already in broadcast.c */
