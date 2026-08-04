/**
 * cnumpy array utilities extension
 * numpy: tobytes, tolist, item, view, getfield, setfield, require,
 *   asarray_chkfinite, newbyteorder, conjugate, format_float,
 *   getbufsize, setbufsize, nbytes, disp, datetime_as_string,
 *   multivariate_normal, random_bytes, einsum_path, multi_dot
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    size_t allocation_size;
    unsigned char data[];
} CnpOwnedBuffer;

static void* cnp_owned_buffer_allocate(
    size_t payload_size, const char *function_name) {
    size_t stored_size = payload_size > 0 ? payload_size : 1;
    if (stored_size > SIZE_MAX - offsetof(CnpOwnedBuffer, data)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "owned buffer allocation size overflows size_t");
        return NULL;
    }
    size_t allocation_size = offsetof(CnpOwnedBuffer, data) + stored_size;
    CnpOwnedBuffer *owner = (CnpOwnedBuffer*)cnp_malloc(allocation_size);
    if (!owner) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "cannot allocate owned result buffer");
        return NULL;
    }
    owner->allocation_size = allocation_size;
    return owner->data;
}

CNP_API void CNP_CALL cnp_buffer_free(void *buffer) {
    if (!buffer) return;
    CnpOwnedBuffer *owner = (CnpOwnedBuffer*)(
        (unsigned char*)buffer - offsetof(CnpOwnedBuffer, data));
    cnp_free(owner, owner->allocation_size);
}

/* =========================================================================
 * cnp_tobytes - Return array data as bytes
 * numpy.ndarray.tobytes(order='C')
 * ========================================================================= */
CNP_API void* CNP_CALL cnp_tobytes(const CnpArray *arr, int64_t *out_size) {
    const char *function_name = "cnp_tobytes";
    if (!arr || !out_size) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "array and output-size pointer must not be null");
        return NULL;
    }
    if (arr->size < 0 || arr->dtype->elsize <= 0 ||
        arr->size > INT64_MAX / arr->dtype->elsize) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "array byte size is invalid or overflows int64");
        return NULL;
    }
    int64_t nbytes = arr->size * arr->dtype->elsize;
    void *buf = cnp_owned_buffer_allocate((size_t)nbytes, function_name);
    if (!buf) return NULL;
    if (arr->flags & CNP_ARRAY_C_CONTIGUOUS) {
        if (nbytes > 0)
            memcpy(buf, (const char*)arr->data + arr->offset, (size_t)nbytes);
    } else {
        int64_t coordinates[CNP_MAXDIMS] = {0};
        for (int64_t flat = 0; flat < arr->size; ++flat) {
            int64_t remainder = flat;
            for (int dimension = arr->ndim - 1; dimension >= 0; --dimension) {
                coordinates[dimension] = remainder % arr->shape[dimension];
                remainder /= arr->shape[dimension];
            }
            int64_t source_offset = arr->offset + cnp_multi_to_offset(
                arr->ndim, coordinates, arr->strides);
            memcpy(
                (unsigned char*)buf + (size_t)flat * arr->dtype->elsize,
                (const unsigned char*)arr->data + source_offset,
                (size_t)arr->dtype->elsize);
        }
    }
    *out_size = nbytes;
    return buf;
}

/* =========================================================================
 * cnp_tolist - Convert array to nested Python-like list (as flat double array)
 * numpy.ndarray.tolist()
 * Returns flat array of doubles representing all elements
 * ========================================================================= */
CNP_API double* CNP_CALL cnp_tolist(const CnpArray *arr, int64_t *out_size) {
    const char *function_name = "cnp_tolist";
    if (!arr || !out_size) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "array and output-size pointer must not be null");
        return NULL;
    }
    if (arr->size < 0 || (uint64_t)arr->size > SIZE_MAX / sizeof(double)) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "flat list size is invalid or overflows size_t");
        return NULL;
    }
    double *list = (double*)cnp_owned_buffer_allocate(
        (size_t)arr->size * sizeof(double), function_name);
    if (!list) return NULL;
    for (int64_t i = 0; i < arr->size; i++) {
        list[i] = cnp_array_flat_get(arr, i);
    }
    *out_size = arr->size;
    return list;
}

/* =========================================================================
 * cnp_item - Get single element as double
 * numpy.ndarray.item(*args)
 * ========================================================================= */
CNP_API double CNP_CALL cnp_item(const CnpArray *arr, int64_t flat_index) {
    const char *function_name = "cnp_item";
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            arr, function_name, &ignored_nbytes)) return NAN;
    if (arr->ndim < 0 || arr->ndim > CNP_MAXDIMS ||
            (arr->ndim > 0 && (!arr->shape || !arr->strides)) ||
            (arr->size > 0 && !arr->data)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source array shape metadata and data buffer must be valid");
        return NAN;
    }
    if (cnp_type_is_complex(arr->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "legacy double result cannot represent a complex scalar");
        return NAN;
    }
    if (flat_index < 0) flat_index += arr->size;
    if (flat_index < 0 || flat_index >= arr->size) {
        cnp_set_error(
            CNP_ERR_INDEX, function_name,
            "flat index %lld is out of bounds for array of size %lld",
            (long long)flat_index, (long long)arr->size);
        return NAN;
    }
    return cnp_array_flat_get(arr, flat_index);
}

/* =========================================================================
 * cnp_view - Return view with different dtype (reinterpret cast)
 * numpy.ndarray.view(dtype)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_view(const CnpArray *arr, int dtype_num) {
    const char *function_name = "cnp_view";
    int64_t shape[CNP_MAXDIMS];
    int64_t strides[CNP_MAXDIMS];
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    CnpDtype *new_dtype = cnp_dtype_new((CNP_TYPE)dtype_num);
    if (!new_dtype) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    int old_itemsize = arr->dtype->elsize;
    int new_itemsize = new_dtype->elsize;
    if (old_itemsize <= 0 || new_itemsize <= 0) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source and destination dtypes must have positive item sizes");
        return NULL;
    }
    for (int dimension = 0; dimension < arr->ndim; dimension++) {
        shape[dimension] = arr->shape[dimension];
        strides[dimension] = arr->strides[dimension];
    }
    if (old_itemsize != new_itemsize) {
        if (arr->ndim == 0) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "a scalar view cannot change item size");
            return NULL;
        }
        int last = arr->ndim - 1;
        if (arr->strides[last] != old_itemsize) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "the last axis must be C-contiguous to change item size");
            return NULL;
        }
        if (shape[last] != 0 &&
                shape[last] > INT64_MAX / old_itemsize) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "last-axis byte length overflows int64");
            return NULL;
        }
        int64_t last_axis_bytes = shape[last] * old_itemsize;
        if (last_axis_bytes % new_itemsize != 0) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "last-axis byte length is not divisible by destination item size");
            return NULL;
        }
        shape[last] = last_axis_bytes / new_itemsize;
        strides[last] = new_itemsize;
    }
    CnpArray *result = cnp_array_view_from_metadata(
        (CnpArray*)arr, arr->ndim,
        arr->ndim ? shape : NULL,
        arr->ndim ? strides : NULL,
        arr->offset, 0);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    result->dtype = new_dtype;
    result->flags = (result->flags &
        ~(CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) |
        cnp_compute_layout_flags(
            result->ndim, result->shape,
            result->strides, new_itemsize);
    return result;
}

/* =========================================================================
 * cnp_getfield - Get field from structured array
 * numpy.ndarray.getfield(dtype, offset=0)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_getfield(const CnpArray *arr, int dtype_num, int64_t offset) {
    const char *function_name = "cnp_getfield";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    CnpDtype *new_dtype = cnp_dtype_new((CNP_TYPE)dtype_num);
    if (!new_dtype) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (offset < 0 || new_dtype->elsize <= 0 ||
            offset > arr->dtype->elsize - new_dtype->elsize) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "field byte range [%lld, %lld) exceeds item size %d",
            (long long)offset,
            (long long)(offset + new_dtype->elsize),
            arr->dtype->elsize);
        return NULL;
    }
    CnpArray *result = cnp_array_view_from_metadata(
        (CnpArray*)arr, arr->ndim, arr->shape, arr->strides,
        arr->offset + offset, 0);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    result->dtype = new_dtype;
    result->flags = (result->flags &
        ~(CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) |
        cnp_compute_layout_flags(
            result->ndim, result->shape,
            result->strides, new_dtype->elsize);
    return result;
}

/* =========================================================================
 * cnp_setfield - Set field in structured array
 * numpy.ndarray.setfield(val, dtype, offset=0)
 * ========================================================================= */
CNP_API CNP_STATUS CNP_CALL cnp_setfield(CnpArray *arr, const CnpArray *val,
                                          int dtype_num, int64_t offset) {
    const char *function_name = "cnp_setfield";
    CnpArray *cast_values = NULL;
    CnpArray *broadcast_values = NULL;
    if (!arr || !val) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "destination and values arrays are required");
        return CNP_ERR_GENERIC;
    }
    if (!(arr->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "destination array is not writeable");
        return CNP_ERR_GENERIC;
    }
    CnpDtype *field_dtype = cnp_dtype_new((CNP_TYPE)dtype_num);
    if (!field_dtype) {
        cnp_relabel_error(function_name);
        return CNP_ERR_TYPE;
    }
    if (offset < 0 || field_dtype->elsize <= 0 ||
            offset > arr->dtype->elsize - field_dtype->elsize) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "field byte range [%lld, %lld) exceeds item size %d",
            (long long)offset,
            (long long)(offset + field_dtype->elsize),
            arr->dtype->elsize);
        return CNP_ERR_VALUE;
    }
    cast_values = cnp_astype(
        val, (CNP_TYPE)dtype_num, CNP_CAST_UNSAFE);
    if (!cast_values) goto failure;
    broadcast_values = cnp_broadcast_to(
        cast_values, arr->ndim, arr->shape);
    if (!broadcast_values) goto failure;
    for (int64_t index = 0; index < arr->size; index++) {
        int64_t remaining = index;
        int64_t destination_offset = arr->offset + offset;
        int64_t source_offset = broadcast_values->offset;
        for (int dimension = arr->ndim - 1;
                dimension >= 0; dimension--) {
            int64_t coordinate = remaining % arr->shape[dimension];
            remaining /= arr->shape[dimension];
            destination_offset += coordinate * arr->strides[dimension];
            source_offset += coordinate * broadcast_values->strides[dimension];
        }
        memcpy(
            (char*)arr->data + destination_offset,
            (const char*)broadcast_values->data + source_offset,
            (size_t)field_dtype->elsize);
    }
    cnp_array_free(broadcast_values);
    cnp_array_free(cast_values);
    return CNP_OK;

failure:
    if (broadcast_values) cnp_array_free(broadcast_values);
    if (cast_values) cnp_array_free(cast_values);
    cnp_relabel_error(function_name);
    return cnp_get_error(NULL);
}

/* =========================================================================
 * cnp_require - Return array meeting requirements
 * numpy.require(a, dtype=None, requirements=None)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_require(const CnpArray *arr, int dtype_num, bool c_contiguous) {
    const char *function_name = "cnp_require";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return NULL;
    }
    CnpArray *result;
    if (dtype_num > 0 && dtype_num != arr->dtype->type_num) {
        result = cnp_astype(arr, (CNP_TYPE)dtype_num, CNP_CAST_UNSAFE);
    } else {
        result = cnp_array_copy(arr);
    }
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (c_contiguous && !(result->flags & CNP_ARRAY_C_CONTIGUOUS)) {
        CnpArray *contig = cnp_ascontiguousarray(result);
        cnp_array_free(result);
        result = contig;
    }
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_asarray_chkfinite - Convert to array, checking for NaN/Inf
 * numpy.asarray_chkfinite(a, dtype=None)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_asarray_chkfinite(const CnpArray *arr, int dtype_num) {
    const char *function_name = "cnp_asarray_chkfinite";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return NULL;
    }
    for (int64_t index = 0; index < arr->size; ++index) {
        int64_t remaining = index;
        int64_t offset = arr->offset;
        for (int dimension = arr->ndim - 1;
             dimension >= 0; --dimension) {
            int64_t coordinate = remaining % arr->shape[dimension];
            remaining /= arr->shape[dimension];
            offset += coordinate * arr->strides[dimension];
        }
        cnp_clongdouble value;
        CNP_STATUS status = cnp_cast_scalar_value(
            (const char*)arr->data + offset,
            arr->dtype->type_num,
            &value, CNP_CLONGDOUBLE, function_name);
        if (status != CNP_OK) return NULL;
        if (!isfinite(value.real) || !isfinite(value.imag)) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "array must not contain infinities or NaNs");
            return NULL;
        }
    }
    CnpArray *result;
    if (dtype_num > 0 && dtype_num != arr->dtype->type_num) {
        result = cnp_astype(arr, (CNP_TYPE)dtype_num, CNP_CAST_UNSAFE);
    } else {
        result = cnp_array_copy(arr);
    }
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_newbyteorder - Return array with different byte order
 * numpy.ndarray.newbyteorder(new_order='S')
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_newbyteorder(const CnpArray *arr) {
    const char *function_name = "cnp_newbyteorder";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    CnpArray *result = cnp_array_copy(arr);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    int64_t elsize = result->dtype->elsize;
    if (elsize <= 0) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtype must have a positive fixed item size");
        return NULL;
    }
    int64_t component_size = cnp_type_is_complex(
        result->dtype->type_num) ? elsize / 2 : elsize;
    char *data = (char*)result->data;
    for (int64_t i = 0; i < result->size; i++) {
        char *elem = data + i * elsize;
        for (int64_t component = 0;
                component < elsize; component += component_size) {
            for (int64_t j = 0; j < component_size / 2; j++) {
                char tmp = elem[component + j];
                elem[component + j] =
                    elem[component + component_size - 1 - j];
                elem[component + component_size - 1 - j] = tmp;
            }
        }
    }
    return result;
}

/* =========================================================================
 * cnp_format_float - Format floating point as string
 * numpy.format_float_scientific / format_float_positional
 * ========================================================================= */
CNP_API CNP_STATUS CNP_CALL cnp_format_float(double val, char *buf, int64_t buf_size,
                                              int precision, bool scientific) {
    const char *function_name = "cnp_format_float";
    if (!buf || buf_size <= 0) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "a positive output buffer is required");
        return CNP_ERR_GENERIC;
    }
    if (precision < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "precision must be nonnegative");
        return CNP_ERR_VALUE;
    }
#if defined(_MSC_VER)
    int required = scientific
        ? _scprintf("%.*e", precision, val)
        : _scprintf("%.*f", precision, val);
#else
    int required = scientific
        ? snprintf(NULL, 0, "%.*e", precision, val)
        : snprintf(NULL, 0, "%.*f", precision, val);
#endif
    if (required < 0) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "floating-point formatting failed");
        return CNP_ERR_GENERIC;
    }
    if ((int64_t)required + 1 > buf_size) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "output buffer requires %d bytes but only %lld are available",
            required + 1, (long long)buf_size);
        return CNP_ERR_VALUE;
    }
    int written = scientific
        ? snprintf(buf, (size_t)buf_size, "%.*e", precision, val)
        : snprintf(buf, (size_t)buf_size, "%.*f", precision, val);
    if (written != required) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "floating-point formatting wrote an unexpected length");
        return CNP_ERR_GENERIC;
    }
    return CNP_OK;
}

/* =========================================================================
 * Buffer size management
 * numpy.getbufsize() / numpy.setbufsize(size)
 * ========================================================================= */
static int64_t g_buffer_size = 8192;

CNP_API int64_t CNP_CALL cnp_getbufsize(void) {
    return g_buffer_size;
}

CNP_API CNP_STATUS CNP_CALL cnp_setbufsize(int64_t size) {
    const char *function_name = "cnp_setbufsize";
    if (size < 16) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "buffer size must be at least 16 bytes");
        return CNP_ERR_VALUE;
    }
    if (size > 10000000) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "buffer size must not exceed 10000000 bytes");
        return CNP_ERR_VALUE;
    }
    if (size % 16 != 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "buffer size must be a multiple of 16 bytes");
        return CNP_ERR_VALUE;
    }
    g_buffer_size = size;
    return CNP_OK;
}

/* =========================================================================
 * cnp_nbytes - Return total bytes consumed by array data
 * numpy.ndarray.nbytes
 * ========================================================================= */
CNP_API int64_t CNP_CALL cnp_nbytes(const CnpArray *arr) {
    int64_t nbytes;
    if (!cnp_array_nbytes_checked(
            arr, "cnp_nbytes", &nbytes)) return -1;
    return nbytes;
}

/* =========================================================================
 * cnp_disp - Display array information
 * numpy.disp(mesg, device=None, linefeed=True)
 * ========================================================================= */
CNP_API CNP_STATUS CNP_CALL cnp_disp(const char *mesg) {
    const char *function_name = "cnp_disp";
    if (!mesg) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "message is required");
        return CNP_ERR_VALUE;
    }
    if (printf("%s\n", mesg) < 0 || fflush(stdout) != 0) {
        cnp_set_error(
            CNP_ERR_IO, function_name,
            "cannot write message to stdout");
        return CNP_ERR_IO;
    }
    return CNP_OK;
}

/* =========================================================================
 * cnp_datetime_as_string - Convert datetime64 to string
 * numpy.datetime_as_string(arr, unit=None)
 * ========================================================================= */
CNP_API CNP_STATUS CNP_CALL cnp_datetime_as_string(const CnpArray *arr, char *buf,
                                                    int64_t buf_size) {
    const char *function_name = "cnp_datetime_as_string";
    if (!arr || !buf || buf_size <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "a scalar datetime array and writable buffer are required");
        return CNP_ERR_VALUE;
    }
    if (!arr->dtype || arr->dtype->type_num != CNP_DATETIME) {
        cnp_set_error(CNP_ERR_TYPE, function_name, "a datetime64 array is required");
        return CNP_ERR_TYPE;
    }
    if (arr->size != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "legacy ABI requires exactly one value; use cnp_datetime_as_string_v2");
        return CNP_ERR_SHAPE;
    }
    char *result = NULL;
    CNP_STATUS status = cnp_datetime_as_string_v2(
        arr, CNP_FR_D, &result, 1);
    if (status != CNP_OK) {
        cnp_relabel_error(function_name);
        return status;
    }
    size_t required = strlen(result) + 1;
    if ((uint64_t)buf_size < required) {
        cnp_char_free_string(result);
        cnp_set_error(CNP_ERR_VALUE, function_name, "output buffer is too small");
        return CNP_ERR_VALUE;
    }
    memcpy(buf, result, required);
    cnp_char_free_string(result);
    return CNP_OK;
}

/* =========================================================================
 * cnp_random_multivariate_normal - Multivariate normal distribution
 * numpy.random.multivariate_normal(mean, cov, size=None)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_random_multivariate_normal(const CnpArray *mean,
                                                           const CnpArray *cov,
                                                           int64_t size) {
    const char *function_name = "cnp_random_multivariate_normal";
    if (!mean || !cov) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "mean and covariance arrays are required");
        return NULL;
    }
    if (mean->ndim != 1 || cov->ndim != 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "mean must be one-dimensional and covariance must be two-dimensional");
        return NULL;
    }
    int64_t dim = mean->size;
    if (cov->shape[0] != dim || cov->shape[1] != dim) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "covariance shape must be (mean.size, mean.size)");
        return NULL;
    }
    if (size < 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "sample count must be non-negative");
        return NULL;
    }
    CNP_TYPE mean_type = mean->dtype->type_num;
    CNP_TYPE covariance_type = cov->dtype->type_num;
    if (!(mean_type == CNP_BOOL || cnp_type_is_integer(mean_type) ||
            cnp_type_is_float(mean_type)) ||
            !(covariance_type == CNP_BOOL ||
                cnp_type_is_integer(covariance_type) ||
                cnp_type_is_float(covariance_type))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "mean and covariance must have represented real numeric dtypes");
        return NULL;
    }
    if ((uint64_t)dim > SIZE_MAX / sizeof(double) ||
            (dim > 0 && (uint64_t)dim >
                SIZE_MAX / (uint64_t)dim / sizeof(double))) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "multivariate workspace is too large");
        return NULL;
    }

    size_t matrix_bytes = (size_t)dim * (size_t)dim * sizeof(double);
    size_t vector_bytes = (size_t)dim * sizeof(double);
    double *L = dim > 0 ? (double*)cnp_calloc(
        (size_t)dim * (size_t)dim, sizeof(double)) : NULL;
    double *z = dim > 0 ? (double*)cnp_malloc(vector_bytes) : NULL;
    if (dim > 0 && (!L || !z)) {
        if (L) cnp_free(L, matrix_bytes);
        if (z) cnp_free(z, vector_bytes);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "unable to allocate multivariate workspace");
        return NULL;
    }

    for (int64_t i = 0; i < dim; i++) {
        for (int64_t j = 0; j < i; ++j) {
            if (cnp_array_flat_get(cov, i * dim + j) !=
                    cnp_array_flat_get(cov, j * dim + i)) {
                cnp_free(z, vector_bytes);
                cnp_free(L, matrix_bytes);
                cnp_set_error(
                    CNP_ERR_VALUE, function_name,
                    "covariance must be symmetric");
                return NULL;
            }
        }
        for (int64_t j = 0; j <= i; j++) {
            double sum = 0;
            for (int64_t k = 0; k < j; k++) sum += L[i*dim+k] * L[j*dim+k];
            if (i == j) {
                double diag = cnp_array_flat_get(cov, i * dim + i) - sum;
                if (!isfinite(diag) || diag < 0.0) {
                    cnp_free(z, vector_bytes);
                    cnp_free(L, matrix_bytes);
                    cnp_set_error(
                        CNP_ERR_VALUE, function_name,
                        "covariance must be finite and positive semidefinite");
                    return NULL;
                }
                L[i*dim+j] = sqrt(diag);
            } else {
                double residual =
                    cnp_array_flat_get(cov, i * dim + j) - sum;
                if (L[j*dim+j] == 0.0) {
                    if (residual != 0.0) {
                        cnp_free(z, vector_bytes);
                        cnp_free(L, matrix_bytes);
                        cnp_set_error(
                            CNP_ERR_VALUE, function_name,
                            "covariance is not positive semidefinite");
                        return NULL;
                    }
                    L[i*dim+j] = 0.0;
                } else {
                    L[i*dim+j] = residual / L[j*dim+j];
                }
            }
        }
    }

    int64_t shape[2] = {size, dim};
    CnpArray *result = cnp_random_output_new(
        2, shape, CNP_DOUBLE, function_name);
    if (!result) {
        if (z) cnp_free(z, vector_bytes);
        if (L) cnp_free(L, matrix_bytes);
        return NULL;
    }

    for (int64_t s = 0; s < size; s++) {
        for (int64_t i = 0; i < dim; i++) z[i] = cnp_random_gauss();
        for (int64_t i = 0; i < dim; i++) {
            double val = cnp_array_flat_get(mean, i);
            for (int64_t j = 0; j <= i; j++) {
                val += L[i*dim+j] * z[j];
            }
            ((double*)result->data)[s * dim + i] = val;
        }
    }

    if (z) cnp_free(z, vector_bytes);
    if (L) cnp_free(L, matrix_bytes);
    return result;
}

/* =========================================================================
 * cnp_random_bytes - Generate random bytes
 * numpy.random.bytes(length)
 * ========================================================================= */
typedef struct {
    size_t allocation_size;
    unsigned char data[];
} CnpRandomBytesOwner;

CNP_API void* CNP_CALL cnp_random_bytes(int64_t length) {
    const char *function_name = "cnp_random_bytes";
    if (length < 0 ||
            (uint64_t)length > SIZE_MAX - sizeof(CnpRandomBytesOwner) - 1) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "byte count must be non-negative and representable");
        return NULL;
    }
    size_t payload_size = length > 0 ? (size_t)length : 1;
    size_t allocation_size = sizeof(CnpRandomBytesOwner) + payload_size;
    CnpRandomBytesOwner *owner =
        (CnpRandomBytesOwner*)cnp_malloc(allocation_size);
    if (!owner) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "unable to allocate random byte result");
        return NULL;
    }
    owner->allocation_size = allocation_size;
    for (int64_t i = 0; i < length; i++) {
        owner->data[i] = (unsigned char)cnp_random_uint64();
    }
    if (length == 0) owner->data[0] = 0;
    return owner->data;
}

CNP_API void CNP_CALL cnp_random_bytes_free(void *buffer) {
    if (!buffer) return;
    CnpRandomBytesOwner *owner = (CnpRandomBytesOwner*)(
        (unsigned char*)buffer - offsetof(CnpRandomBytesOwner, data));
    cnp_free(owner, owner->allocation_size);
}

/* =========================================================================
 * cnp_float_to_half - Convert float64 to float16 representation
 * numpy.float16 conversion
 * ========================================================================= */
CNP_API uint16_t CNP_CALL cnp_float_to_half(double val) {
    uint64_t bits = 0;
    memcpy(&bits, &val, sizeof(bits));
    uint16_t sign = (uint16_t)((bits >> 48) & 0x8000u);
    uint64_t exponent_bits = (bits >> 52) & 0x7ffu;
    uint64_t fraction = bits & UINT64_C(0x000fffffffffffff);

    if (exponent_bits == 0x7ffu) {
        if (fraction == 0)
            return (uint16_t)(sign | 0x7c00u);
        uint16_t payload = (uint16_t)(fraction >> 42);
        if (payload == 0) payload = 1;
        return (uint16_t)(sign | 0x7c00u | payload);
    }
    if (exponent_bits == 0)
        return sign;

    int exponent = (int)exponent_bits - 1023;
    if (exponent < -25)
        return sign;
    if (exponent > 15)
        return (uint16_t)(sign | 0x7c00u);

    uint64_t significand = UINT64_C(0x0010000000000000) | fraction;
    int shift = exponent >= -14 ? 42 : 28 - exponent;
    uint64_t truncated = significand >> shift;
    uint64_t remainder_mask = (UINT64_C(1) << shift) - 1;
    uint64_t remainder = significand & remainder_mask;
    uint64_t halfway = UINT64_C(1) << (shift - 1);
    if (remainder > halfway ||
        (remainder == halfway && (truncated & 1u)))
        ++truncated;

    if (exponent < -14)
        return (uint16_t)(sign | (uint16_t)truncated);

    int half_exponent = exponent + 15;
    if (truncated == 0x800u) {
        truncated = 0x400u;
        ++half_exponent;
    }
    if (half_exponent >= 31)
        return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(
        sign | ((uint16_t)half_exponent << 10) |
        (uint16_t)(truncated & 0x3ffu));
}

/* =========================================================================
 * cnp_half_to_float - Convert float16 to float64
 * numpy.float16 to float64 conversion
 * ========================================================================= */
CNP_API double CNP_CALL cnp_half_to_float(uint16_t h) {
    uint64_t sign = (uint64_t)(h & 0x8000u) << 48;
    uint16_t exponent = (uint16_t)((h >> 10) & 0x1fu);
    uint16_t fraction = (uint16_t)(h & 0x03ffu);
    uint64_t bits;

    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            int highest_bit = 0;
            uint16_t scan = fraction;
            while (scan > 1) {
                scan >>= 1;
                ++highest_bit;
            }
            uint64_t double_exponent =
                (uint64_t)(highest_bit - 24 + 1023);
            uint64_t double_fraction =
                (uint64_t)(fraction - (1u << highest_bit))
                << (52 - highest_bit);
            bits = sign | (double_exponent << 52) | double_fraction;
        }
    } else if (exponent == 0x1fu) {
        bits = sign | UINT64_C(0x7ff0000000000000) |
            ((uint64_t)fraction << 42);
    } else {
        uint64_t double_exponent = (uint64_t)(exponent + 1008u);
        bits = sign | (double_exponent << 52) |
            ((uint64_t)fraction << 42);
    }

    double result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

typedef enum {
    CNP_REGEX_LITERAL = 0,
    CNP_REGEX_DOT,
    CNP_REGEX_CLASS,
    CNP_REGEX_CAPTURE_START,
    CNP_REGEX_CAPTURE_END,
    CNP_REGEX_ANCHOR_START,
    CNP_REGEX_ANCHOR_END
} CnpRegexTokenKind;

typedef struct {
    CnpRegexTokenKind kind;
    unsigned char literal;
    unsigned char characters[32];
    bool negate;
    int minimum;
    int maximum;
    int capture;
} CnpRegexToken;

typedef struct {
    CnpRegexToken *tokens;
    int token_count;
    int capture_count;
    size_t allocation_count;
} CnpRegexPattern;

typedef struct {
    int64_t start;
    int64_t end;
} CnpRegexCapture;

struct CnpRegexResult {
    int64_t count;
    int nfields;
    char **field_names;
    CNP_TYPE *field_types;
    CnpArray **fields;
};

static void cnp_regex_class_add(CnpRegexToken *token, unsigned char value) {
    token->characters[value >> 3] |= (unsigned char)(1u << (value & 7u));
}

static void cnp_regex_class_add_range(
        CnpRegexToken *token, unsigned char first, unsigned char last) {
    for (unsigned int value = first; value <= last; ++value) {
        cnp_regex_class_add(token, (unsigned char)value);
    }
}

static void cnp_regex_class_add_escape(
        CnpRegexToken *token, unsigned char escape) {
    switch (escape) {
        case 'd':
            cnp_regex_class_add_range(token, '0', '9');
            break;
        case 's':
            cnp_regex_class_add(token, ' ');
            cnp_regex_class_add(token, '\t');
            cnp_regex_class_add(token, '\n');
            cnp_regex_class_add(token, '\r');
            cnp_regex_class_add(token, '\f');
            cnp_regex_class_add(token, '\v');
            break;
        case 'w':
            cnp_regex_class_add_range(token, '0', '9');
            cnp_regex_class_add_range(token, 'A', 'Z');
            cnp_regex_class_add_range(token, 'a', 'z');
            cnp_regex_class_add(token, '_');
            break;
        default:
            cnp_regex_class_add(token, escape);
            break;
    }
}

static bool cnp_regex_compile(
        const char *pattern, CnpRegexPattern *compiled,
        const char *function_name) {
    size_t length = strlen(pattern);
    size_t allocation_count = length + 1;
    CnpRegexToken *tokens = (CnpRegexToken*)cnp_calloc(
        allocation_count, sizeof(CnpRegexToken));
    int *capture_stack = (int*)cnp_malloc(
        allocation_count * sizeof(int));
    if (!tokens || !capture_stack) {
        if (tokens) {
            cnp_free(tokens, allocation_count * sizeof(CnpRegexToken));
        }
        if (capture_stack) {
            cnp_free(capture_stack, allocation_count * sizeof(int));
        }
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate regex pattern storage");
        return false;
    }

    int token_count = 0;
    int capture_count = 0;
    int stack_size = 0;
    for (size_t index = 0; index < length; ++index) {
        unsigned char current = (unsigned char)pattern[index];
        if (current == '*' || current == '+' || current == '?') {
            if (token_count == 0 ||
                    tokens[token_count - 1].kind > CNP_REGEX_CLASS ||
                    tokens[token_count - 1].minimum != 1 ||
                    tokens[token_count - 1].maximum != 1) {
                cnp_set_error(
                    CNP_ERR_VALUE, function_name,
                    "regex pattern has a misplaced quantifier at byte %llu",
                    (unsigned long long)index);
                goto compile_failure;
            }
            CnpRegexToken *previous = &tokens[token_count - 1];
            previous->minimum = current == '+' ? 1 : 0;
            previous->maximum = current == '?' ? 1 : -1;
            continue;
        }

        CnpRegexToken *token = &tokens[token_count];
        token->minimum = 1;
        token->maximum = 1;
        if (current == '(') {
            if (index + 1 < length && pattern[index + 1] == '?') {
                cnp_set_error(
                    CNP_ERR_VALUE, function_name,
                    "regex pattern uses an unsupported group extension");
                goto compile_failure;
            }
            token->kind = CNP_REGEX_CAPTURE_START;
            token->capture = capture_count;
            capture_stack[stack_size++] = capture_count;
            ++capture_count;
        } else if (current == ')') {
            if (stack_size <= 0) {
                cnp_set_error(
                    CNP_ERR_VALUE, function_name,
                    "regex pattern has an unmatched closing parenthesis");
                goto compile_failure;
            }
            token->kind = CNP_REGEX_CAPTURE_END;
            token->capture = capture_stack[--stack_size];
        } else if (current == '[') {
            token->kind = CNP_REGEX_CLASS;
            size_t cursor = index + 1;
            if (cursor < length && pattern[cursor] == '^') {
                token->negate = true;
                ++cursor;
            }
            bool have_character = false;
            while (cursor < length && pattern[cursor] != ']') {
                unsigned char first = (unsigned char)pattern[cursor++];
                if (first == '\\') {
                    if (cursor >= length) {
                        cnp_set_error(
                            CNP_ERR_VALUE, function_name,
                            "regex pattern ends inside a character class escape");
                        goto compile_failure;
                    }
                    first = (unsigned char)pattern[cursor++];
                    cnp_regex_class_add_escape(token, first);
                    have_character = true;
                    continue;
                }
                if (cursor + 1 < length && pattern[cursor] == '-' &&
                        pattern[cursor + 1] != ']') {
                    cursor++;
                    unsigned char last = (unsigned char)pattern[cursor++];
                    if (last == '\\') {
                        if (cursor >= length) {
                            cnp_set_error(
                                CNP_ERR_VALUE, function_name,
                                "regex pattern ends inside a range escape");
                            goto compile_failure;
                        }
                        last = (unsigned char)pattern[cursor++];
                    }
                    if (first > last) {
                        cnp_set_error(
                            CNP_ERR_VALUE, function_name,
                            "regex character-class range is reversed");
                        goto compile_failure;
                    }
                    cnp_regex_class_add_range(token, first, last);
                } else {
                    cnp_regex_class_add(token, first);
                }
                have_character = true;
            }
            if (cursor >= length || pattern[cursor] != ']' || !have_character) {
                cnp_set_error(
                    CNP_ERR_VALUE, function_name,
                    "regex pattern has an invalid character class");
                goto compile_failure;
            }
            index = cursor;
        } else if (current == '\\') {
            if (++index >= length) {
                cnp_set_error(
                    CNP_ERR_VALUE, function_name,
                    "regex pattern ends with an escape");
                goto compile_failure;
            }
            unsigned char escape = (unsigned char)pattern[index];
            if (escape == 'd' || escape == 's' || escape == 'w' ||
                    escape == 'D' || escape == 'S' || escape == 'W') {
                token->kind = CNP_REGEX_CLASS;
                token->negate = escape == 'D' || escape == 'S' || escape == 'W';
                cnp_regex_class_add_escape(
                    token, (unsigned char)tolower(escape));
            } else {
                token->kind = CNP_REGEX_LITERAL;
                switch (escape) {
                    case 't': token->literal = '\t'; break;
                    case 'n': token->literal = '\n'; break;
                    case 'r': token->literal = '\r'; break;
                    case 'f': token->literal = '\f'; break;
                    case 'v': token->literal = '\v'; break;
                    default: token->literal = escape; break;
                }
            }
        } else if (current == '.') {
            token->kind = CNP_REGEX_DOT;
        } else if (current == '^') {
            token->kind = CNP_REGEX_ANCHOR_START;
        } else if (current == '$') {
            token->kind = CNP_REGEX_ANCHOR_END;
        } else if (current == '|' || current == '{' || current == '}') {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "regex pattern uses unsupported operator '%c'", current);
            goto compile_failure;
        } else {
            token->kind = CNP_REGEX_LITERAL;
            token->literal = current;
        }
        ++token_count;
    }

    if (stack_size != 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "regex pattern has an unmatched opening parenthesis");
        goto compile_failure;
    }
    cnp_free(capture_stack, allocation_count * sizeof(int));
    compiled->tokens = tokens;
    compiled->token_count = token_count;
    compiled->capture_count = capture_count;
    compiled->allocation_count = allocation_count;
    return true;

compile_failure:
    cnp_free(capture_stack, allocation_count * sizeof(int));
    cnp_free(tokens, allocation_count * sizeof(CnpRegexToken));
    return false;
}

static void cnp_regex_pattern_destroy(CnpRegexPattern *pattern) {
    if (!pattern || !pattern->tokens) return;
    cnp_free(
        pattern->tokens,
        pattern->allocation_count * sizeof(CnpRegexToken));
    memset(pattern, 0, sizeof(*pattern));
}

static bool cnp_regex_token_matches(
        const CnpRegexToken *token, unsigned char value) {
    if (token->kind == CNP_REGEX_LITERAL) return value == token->literal;
    if (token->kind == CNP_REGEX_DOT) return value != '\n';
    bool member =
        (token->characters[value >> 3] & (1u << (value & 7u))) != 0;
    return token->negate ? !member : member;
}

static bool cnp_regex_match_from(
        const CnpRegexPattern *pattern, int token_index,
        const unsigned char *text, int64_t text_length, int64_t position,
        CnpRegexCapture *captures, int64_t *match_end) {
    if (token_index >= pattern->token_count) {
        *match_end = position;
        return true;
    }

    const CnpRegexToken *token = &pattern->tokens[token_index];
    if (token->kind == CNP_REGEX_CAPTURE_START ||
            token->kind == CNP_REGEX_CAPTURE_END) {
        int64_t *slot = token->kind == CNP_REGEX_CAPTURE_START
            ? &captures[token->capture].start
            : &captures[token->capture].end;
        int64_t previous = *slot;
        *slot = position;
        if (cnp_regex_match_from(
                pattern, token_index + 1, text, text_length, position,
                captures, match_end)) {
            return true;
        }
        *slot = previous;
        return false;
    }
    if (token->kind == CNP_REGEX_ANCHOR_START) {
        return position == 0 && cnp_regex_match_from(
            pattern, token_index + 1, text, text_length, position,
            captures, match_end);
    }
    if (token->kind == CNP_REGEX_ANCHOR_END) {
        return position == text_length && cnp_regex_match_from(
            pattern, token_index + 1, text, text_length, position,
            captures, match_end);
    }

    int64_t available = text_length - position;
    int64_t maximum = token->maximum < 0
        ? available : (int64_t)token->maximum;
    if (maximum > available) maximum = available;
    int64_t matched = 0;
    while (matched < maximum &&
            cnp_regex_token_matches(token, text[position + matched])) {
        ++matched;
    }
    if (matched < token->minimum) return false;
    for (int64_t consumed = matched;
            consumed >= token->minimum; --consumed) {
        if (cnp_regex_match_from(
                pattern, token_index + 1, text, text_length,
                position + consumed, captures, match_end)) {
            return true;
        }
    }
    return false;
}

static bool cnp_regex_find_next(
        const CnpRegexPattern *pattern,
        const unsigned char *text, int64_t text_length, int64_t search_start,
        CnpRegexCapture *captures, int64_t *match_start,
        int64_t *match_end) {
    bool anchored = pattern->token_count > 0 &&
        pattern->tokens[0].kind == CNP_REGEX_ANCHOR_START;
    int64_t first = anchored ? 0 : search_start;
    int64_t last = anchored ? 0 : text_length;
    if (anchored && search_start > 0) return false;

    for (int64_t start = first; start <= last; ++start) {
        for (int capture = 0; capture < pattern->capture_count; ++capture) {
            captures[capture].start = -1;
            captures[capture].end = -1;
        }
        if (cnp_regex_match_from(
                pattern, 0, text, text_length, start,
                captures, match_end)) {
            *match_start = start;
            return true;
        }
    }
    return false;
}

static bool cnp_regex_numeric_dtype(CNP_TYPE dtype) {
    if (dtype <= CNP_NOTYPE || dtype >= CNP_NTYPES) return false;
    char kind = cnp_dtype_kind(dtype);
    return kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f';
}

static bool cnp_regex_store_capture(
        CnpArray *field, int64_t row, char *text,
        CnpRegexCapture capture, const char *function_name) {
    char saved = text[capture.end];
    text[capture.end] = '\0';
    char *source = text + capture.start;
    char *end = NULL;
    char kind = cnp_dtype_kind(field->dtype->type_num);
    errno = 0;

    if (kind == 'i' || kind == 'b') {
        long long value = _strtoi64(source, &end, 10);
        int64_t minimum = INT64_MIN;
        int64_t maximum = INT64_MAX;
        switch (field->dtype->type_num) {
            case CNP_BYTE: minimum = INT8_MIN; maximum = INT8_MAX; break;
            case CNP_SHORT: minimum = INT16_MIN; maximum = INT16_MAX; break;
            case CNP_INT: minimum = INT32_MIN; maximum = INT32_MAX; break;
            default: break;
        }
        if (end == source || *end || errno == ERANGE ||
                value < minimum || value > maximum) {
            text[capture.end] = saved;
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "regex capture %lld cannot be converted to dtype %d",
                (long long)row, (int)field->dtype->type_num);
            return false;
        }
        cnp_set_element_int(
            field->data, row * field->dtype->elsize,
            field->dtype->type_num, (int64_t)value);
    } else if (kind == 'u') {
        const char *sign = source;
        while (isspace((unsigned char)*sign)) ++sign;
        unsigned long long value = _strtoui64(source, &end, 10);
        uint64_t maximum = UINT64_MAX;
        switch (field->dtype->type_num) {
            case CNP_UBYTE: maximum = UINT8_MAX; break;
            case CNP_USHORT: maximum = UINT16_MAX; break;
            case CNP_UINT: maximum = UINT32_MAX; break;
            default: break;
        }
        if (*sign == '-' || end == source || *end ||
                errno == ERANGE || value > maximum) {
            text[capture.end] = saved;
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "regex capture %lld cannot be converted to dtype %d",
                (long long)row, (int)field->dtype->type_num);
            return false;
        }
        switch (field->dtype->type_num) {
            case CNP_UBYTE: ((uint8_t*)field->data)[row] = (uint8_t)value; break;
            case CNP_USHORT: ((uint16_t*)field->data)[row] = (uint16_t)value; break;
            case CNP_UINT: ((uint32_t*)field->data)[row] = (uint32_t)value; break;
            default: ((uint64_t*)field->data)[row] = (uint64_t)value; break;
        }
    } else {
        double value = strtod(source, &end);
        if (end == source || *end) {
            text[capture.end] = saved;
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "regex capture %lld cannot be converted to dtype %d",
                (long long)row, (int)field->dtype->type_num);
            return false;
        }
        cnp_set_element_double(
            field->data, row * field->dtype->elsize,
            field->dtype->type_num, value);
    }
    text[capture.end] = saved;
    return true;
}

static void cnp_regex_result_destroy(CnpRegexResult *result) {
    if (!result) return;
    if (result->fields) {
        for (int field = 0; field < result->nfields; ++field) {
            if (result->fields[field]) {
                cnp_array_decref(result->fields[field]);
            }
        }
        cnp_free(result->fields, result->nfields * sizeof(CnpArray*));
    }
    if (result->field_names) {
        for (int field = 0; field < result->nfields; ++field) {
            if (result->field_names[field]) {
                size_t length = strlen(result->field_names[field]);
                cnp_free(result->field_names[field], length + 1);
            }
        }
        cnp_free(result->field_names, result->nfields * sizeof(char*));
    }
    if (result->field_types) {
        cnp_free(result->field_types, result->nfields * sizeof(CNP_TYPE));
    }
    cnp_free(result, sizeof(CnpRegexResult));
}

CNP_API CnpRegexResult* CNP_CALL cnp_fromregex_v2(
        const char *str, const char *pattern,
        const char *const *field_names, const CNP_TYPE *field_types,
        int nfields, int64_t max_matches) {
    const char *function_name = "cnp_fromregex_v2";
    if (!str) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "input text is required");
        return NULL;
    }
    if (!pattern || !*pattern) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "regex pattern is required");
        return NULL;
    }
    if (!field_names || !field_types || nfields <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "field names and dtypes are required");
        return NULL;
    }
    if (max_matches < -1) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "max_matches must be -1 or nonnegative");
        return NULL;
    }
    for (int field = 0; field < nfields; ++field) {
        if (!field_names[field] || !*field_names[field]) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "field name %d must not be empty", field);
            return NULL;
        }
        if (!cnp_regex_numeric_dtype(field_types[field])) {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "field dtype %d is not a supported numeric dtype",
                (int)field_types[field]);
            return NULL;
        }
        for (int earlier = 0; earlier < field; ++earlier) {
            if (strcmp(field_names[earlier], field_names[field]) == 0) {
                cnp_set_error(
                    CNP_ERR_VALUE, function_name,
                    "field names must be unique");
                return NULL;
            }
        }
    }

    CnpRegexPattern compiled = {0};
    if (!cnp_regex_compile(pattern, &compiled, function_name)) return NULL;
    if (compiled.capture_count != nfields) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "regex capture group count %d does not match field count %d",
            compiled.capture_count, nfields);
        cnp_regex_pattern_destroy(&compiled);
        return NULL;
    }

    size_t text_length_size = strlen(str);
    if (text_length_size > INT64_MAX) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "input text is too large");
        cnp_regex_pattern_destroy(&compiled);
        return NULL;
    }
    int64_t text_length = (int64_t)text_length_size;
    CnpRegexCapture *captures = (CnpRegexCapture*)cnp_malloc(
        nfields * sizeof(CnpRegexCapture));
    char *text = (char*)cnp_malloc(text_length_size + 1);
    if (!captures || !text) {
        if (captures) cnp_free(captures, nfields * sizeof(CnpRegexCapture));
        if (text) cnp_free(text, text_length_size + 1);
        cnp_regex_pattern_destroy(&compiled);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate regex matching storage");
        return NULL;
    }
    memcpy(text, str, text_length_size + 1);

    int64_t match_count = 0;
    int64_t search = 0;
    while ((max_matches < 0 || match_count < max_matches) &&
            search <= text_length) {
        int64_t match_start = 0;
        int64_t match_end = 0;
        if (!cnp_regex_find_next(
                &compiled, (const unsigned char*)text, text_length, search,
                captures, &match_start, &match_end)) {
            break;
        }
        ++match_count;
        search = match_end > match_start ? match_end : match_start + 1;
    }

    CnpRegexResult *result = (CnpRegexResult*)cnp_calloc(
        1, sizeof(CnpRegexResult));
    if (!result) {
        cnp_set_error(CNP_ERR_MEMORY, function_name, "failed to allocate result");
        goto result_failure;
    }
    result->count = match_count;
    result->nfields = nfields;
    result->field_names = (char**)cnp_calloc(nfields, sizeof(char*));
    result->field_types = (CNP_TYPE*)cnp_malloc(
        nfields * sizeof(CNP_TYPE));
    result->fields = (CnpArray**)cnp_calloc(nfields, sizeof(CnpArray*));
    if (!result->field_names || !result->field_types || !result->fields) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate regex result fields");
        goto result_failure;
    }

    int64_t shape[1] = {match_count};
    for (int field = 0; field < nfields; ++field) {
        size_t name_length = strlen(field_names[field]);
        result->field_names[field] = (char*)cnp_malloc(name_length + 1);
        if (!result->field_names[field]) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "failed to copy regex field name %d", field);
            goto result_failure;
        }
        memcpy(
            result->field_names[field], field_names[field], name_length + 1);
        result->field_types[field] = field_types[field];
        result->fields[field] = cnp_array_new(
            1, shape, field_types[field], CNP_ORDER_C);
        if (!result->fields[field]) {
            cnp_relabel_error(function_name);
            goto result_failure;
        }
    }

    search = 0;
    for (int64_t row = 0; row < match_count; ++row) {
        int64_t match_start = 0;
        int64_t match_end = 0;
        if (!cnp_regex_find_next(
                &compiled, (const unsigned char*)text, text_length, search,
                captures, &match_start, &match_end)) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "regex matching changed between validation and conversion");
            goto result_failure;
        }
        for (int field = 0; field < nfields; ++field) {
            if (captures[field].start < 0 ||
                    captures[field].end < captures[field].start ||
                    !cnp_regex_store_capture(
                        result->fields[field], row, text,
                        captures[field], function_name)) {
                if (cnp_get_error(NULL) == CNP_OK) {
                    cnp_set_error(
                        CNP_ERR_VALUE, function_name,
                        "regex capture group %d did not participate", field);
                }
                goto result_failure;
            }
        }
        search = match_end > match_start ? match_end : match_start + 1;
    }

    cnp_free(text, text_length_size + 1);
    cnp_free(captures, nfields * sizeof(CnpRegexCapture));
    cnp_regex_pattern_destroy(&compiled);
    return result;

result_failure:
    if (result) cnp_regex_result_destroy(result);
    cnp_free(text, text_length_size + 1);
    cnp_free(captures, nfields * sizeof(CnpRegexCapture));
    cnp_regex_pattern_destroy(&compiled);
    return NULL;
}

CNP_API CnpArray* CNP_CALL cnp_fromregex(
        const char *str, const char *pattern,
        int dtype_num, int64_t max_matches) {
    const char *function_name = "cnp_fromregex";
    const char *field_names[1] = {"f0"};
    CNP_TYPE field_types[1] = {(CNP_TYPE)dtype_num};
    CnpRegexResult *result = cnp_fromregex_v2(
        str, pattern, field_names, field_types, 1, max_matches);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpArray *field = cnp_regex_result_field(result, 0);
    cnp_regex_result_destroy(result);
    if (!field) cnp_relabel_error(function_name);
    return field;
}

CNP_API int64_t CNP_CALL cnp_regex_result_count(
        const CnpRegexResult *result) {
    if (!result) {
        cnp_set_error(
            CNP_ERR_VALUE, "cnp_regex_result_count",
            "regex result is required");
        return -1;
    }
    return result->count;
}

CNP_API int CNP_CALL cnp_regex_result_nfields(
        const CnpRegexResult *result) {
    if (!result) {
        cnp_set_error(
            CNP_ERR_VALUE, "cnp_regex_result_nfields",
            "regex result is required");
        return -1;
    }
    return result->nfields;
}

CNP_API const char* CNP_CALL cnp_regex_result_field_name(
        const CnpRegexResult *result, int field_index) {
    const char *function_name = "cnp_regex_result_field_name";
    if (!result) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "regex result is required");
        return NULL;
    }
    if (field_index < 0 || field_index >= result->nfields) {
        cnp_set_error(
            CNP_ERR_INDEX, function_name,
            "field index %d is out of bounds for %d fields",
            field_index, result->nfields);
        return NULL;
    }
    return result->field_names[field_index];
}

CNP_API CnpArray* CNP_CALL cnp_regex_result_field(
        const CnpRegexResult *result, int field_index) {
    const char *function_name = "cnp_regex_result_field";
    if (!result) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "regex result is required");
        return NULL;
    }
    if (field_index < 0 || field_index >= result->nfields) {
        cnp_set_error(
            CNP_ERR_INDEX, function_name,
            "field index %d is out of bounds for %d fields",
            field_index, result->nfields);
        return NULL;
    }
    cnp_array_incref(result->fields[field_index]);
    return result->fields[field_index];
}

CNP_API void CNP_CALL cnp_regex_result_free(CnpRegexResult *result) {
    cnp_regex_result_destroy(result);
}

typedef struct {
    const char *start;
    const char *cursor;
    const char *function_name;
    bool failed;
} CnpSafeEvalParser;

static void cnp_safe_eval_skip_space(CnpSafeEvalParser *parser) {
    while (isspace((unsigned char)*parser->cursor)) ++parser->cursor;
}

static void cnp_safe_eval_syntax_error(
        CnpSafeEvalParser *parser, const char *message) {
    if (parser->failed) return;
    parser->failed = true;
    cnp_set_error(
        CNP_ERR_VALUE, parser->function_name,
        "%s at byte offset %lld", message,
        (long long)(parser->cursor - parser->start));
}

static bool cnp_safe_eval_is_base_digit(unsigned char value, int base) {
    if (value >= '0' && value <= '9') return value - '0' < base;
    if (base == 16 && value >= 'a' && value <= 'f') return true;
    if (base == 16 && value >= 'A' && value <= 'F') return true;
    return false;
}

static bool cnp_safe_eval_copy_without_underscores(
        CnpSafeEvalParser *parser, const char *first, const char *last,
        char **out_text, size_t *out_size) {
    size_t source_size = (size_t)(last - first);
    char *text = (char*)cnp_malloc(source_size + 1);
    if (!text) {
        parser->failed = true;
        cnp_set_error(
            CNP_ERR_MEMORY, parser->function_name,
            "failed to allocate numeric literal storage");
        return false;
    }
    size_t written = 0;
    for (const char *cursor = first; cursor < last; ++cursor) {
        if (*cursor != '_') text[written++] = *cursor;
    }
    text[written] = '\0';
    *out_text = text;
    *out_size = source_size + 1;
    return true;
}

static bool cnp_safe_eval_number(
        CnpSafeEvalParser *parser, double *value) {
    const char *first = parser->cursor;
    const char *cursor = first;
    int base = 10;
    bool based_integer = false;

    if (cursor[0] == '0' &&
            (cursor[1] == 'x' || cursor[1] == 'X' ||
             cursor[1] == 'b' || cursor[1] == 'B' ||
             cursor[1] == 'o' || cursor[1] == 'O')) {
        based_integer = true;
        base = cursor[1] == 'x' || cursor[1] == 'X' ? 16
            : cursor[1] == 'b' || cursor[1] == 'B' ? 2 : 8;
        cursor += 2;
        if (*cursor == '_') ++cursor;
        bool have_digit = false;
        bool previous_underscore = false;
        while (cnp_safe_eval_is_base_digit((unsigned char)*cursor, base) ||
                *cursor == '_') {
            if (*cursor == '_') {
                if (!have_digit || previous_underscore ||
                        !cnp_safe_eval_is_base_digit(
                            (unsigned char)cursor[1], base)) {
                    parser->cursor = cursor;
                    cnp_safe_eval_syntax_error(
                        parser, "invalid numeric separator");
                    return false;
                }
                previous_underscore = true;
            } else {
                have_digit = true;
                previous_underscore = false;
            }
            ++cursor;
        }
        if (!have_digit) {
            parser->cursor = cursor;
            cnp_safe_eval_syntax_error(parser, "invalid based integer literal");
            return false;
        }
    } else {
        char *direct_end = NULL;
        errno = 0;
        double direct_value = strtod(first, &direct_end);
        if (direct_end > first && *direct_end != '_') {
            parser->cursor = direct_end;
            *value = direct_value;
            return true;
        }

        bool have_mantissa_digit = false;
        bool previous_digit = false;
        while (isdigit((unsigned char)*cursor) || *cursor == '_') {
            if (*cursor == '_') {
                if (!previous_digit || !isdigit((unsigned char)cursor[1])) {
                    parser->cursor = cursor;
                    cnp_safe_eval_syntax_error(
                        parser, "invalid numeric separator");
                    return false;
                }
                previous_digit = false;
            } else {
                have_mantissa_digit = true;
                previous_digit = true;
            }
            ++cursor;
        }
        if (*cursor == '.') {
            ++cursor;
            previous_digit = false;
            while (isdigit((unsigned char)*cursor) || *cursor == '_') {
                if (*cursor == '_') {
                    if (!previous_digit ||
                            !isdigit((unsigned char)cursor[1])) {
                        parser->cursor = cursor;
                        cnp_safe_eval_syntax_error(
                            parser, "invalid numeric separator");
                        return false;
                    }
                    previous_digit = false;
                } else {
                    have_mantissa_digit = true;
                    previous_digit = true;
                }
                ++cursor;
            }
        }
        if (!have_mantissa_digit) {
            cnp_safe_eval_syntax_error(parser, "invalid numeric literal");
            return false;
        }
        if (*cursor == 'e' || *cursor == 'E') {
            ++cursor;
            if (*cursor == '+' || *cursor == '-') ++cursor;
            bool have_exponent_digit = false;
            previous_digit = false;
            while (isdigit((unsigned char)*cursor) || *cursor == '_') {
                if (*cursor == '_') {
                    if (!previous_digit ||
                            !isdigit((unsigned char)cursor[1])) {
                        parser->cursor = cursor;
                        cnp_safe_eval_syntax_error(
                            parser, "invalid numeric separator");
                        return false;
                    }
                    previous_digit = false;
                } else {
                    have_exponent_digit = true;
                    previous_digit = true;
                }
                ++cursor;
            }
            if (!have_exponent_digit) {
                parser->cursor = cursor;
                cnp_safe_eval_syntax_error(parser, "invalid exponent literal");
                return false;
            }
        }
    }

    char *normalized = NULL;
    size_t normalized_size = 0;
    if (!cnp_safe_eval_copy_without_underscores(
            parser, first, cursor, &normalized, &normalized_size)) {
        return false;
    }
    char *conversion_end = NULL;
    errno = 0;
    if (based_integer) {
        unsigned long long integer = _strtoui64(
            normalized + 2, &conversion_end, base);
        if (!conversion_end || *conversion_end || errno == ERANGE) {
            cnp_free(normalized, normalized_size);
            parser->cursor = cursor;
            cnp_safe_eval_syntax_error(parser, "invalid based integer literal");
            return false;
        }
        *value = (double)integer;
    } else {
        double number = strtod(normalized, &conversion_end);
        if (!conversion_end || *conversion_end) {
            cnp_free(normalized, normalized_size);
            parser->cursor = cursor;
            cnp_safe_eval_syntax_error(parser, "invalid numeric literal");
            return false;
        }
        *value = number;
    }
    cnp_free(normalized, normalized_size);
    parser->cursor = cursor;
    return true;
}

static bool cnp_safe_eval_literal(
        CnpSafeEvalParser *parser, double *value, bool *signed_literal) {
    cnp_safe_eval_skip_space(parser);
    bool has_sign = *parser->cursor == '+' || *parser->cursor == '-';
    bool negate = has_sign && *parser->cursor == '-';
    if (has_sign) {
        ++parser->cursor;
        cnp_safe_eval_skip_space(parser);
    }

    bool inner_signed = false;
    if (*parser->cursor == '(') {
        ++parser->cursor;
        if (!cnp_safe_eval_literal(parser, value, &inner_signed)) return false;
        cnp_safe_eval_skip_space(parser);
        if (*parser->cursor != ')') {
            cnp_safe_eval_syntax_error(
                parser, "expected a closing parenthesis");
            return false;
        }
        ++parser->cursor;
    } else {
        unsigned char first = (unsigned char)*parser->cursor;
        if (!isdigit(first) && first != '.') {
            cnp_safe_eval_syntax_error(
                parser, "expected a numeric literal or opening parenthesis");
            return false;
        }
        if (!cnp_safe_eval_number(parser, value)) return false;
    }

    if (has_sign && inner_signed) {
        cnp_safe_eval_syntax_error(
            parser, "nested unary signs are not literal syntax");
        return false;
    }
    if (negate) *value = -*value;
    *signed_literal = has_sign || inner_signed;
    return true;
}

CNP_API CNP_STATUS CNP_CALL cnp_safe_eval_v2(
        const char *expr, double *out_value) {
    const char *function_name = "cnp_safe_eval_v2";
    if (!expr) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "expression string is required");
        return CNP_ERR_VALUE;
    }
    if (!out_value) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "output value pointer is required");
        return CNP_ERR_VALUE;
    }

    const char *direct = expr;
    while (isspace((unsigned char)*direct)) ++direct;
    const char *unsigned_start = direct;
    if (*unsigned_start == '+' || *unsigned_start == '-') ++unsigned_start;
    bool base_prefixed = unsigned_start[0] == '0' &&
        (unsigned_start[1] == 'x' || unsigned_start[1] == 'X' ||
         unsigned_start[1] == 'b' || unsigned_start[1] == 'B' ||
         unsigned_start[1] == 'o' || unsigned_start[1] == 'O');
    if (!base_prefixed &&
            (isdigit((unsigned char)*unsigned_start) ||
             *unsigned_start == '.')) {
        char *direct_end = NULL;
        errno = 0;
        double direct_value = strtod(direct, &direct_end);
        if (direct_end > direct && *direct_end != '_') {
            const char *tail = direct_end;
            while (isspace((unsigned char)*tail)) ++tail;
            if (!*tail) {
                *out_value = direct_value;
                return CNP_OK;
            }
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "unexpected trailing expression syntax at byte offset %lld",
                (long long)(tail - expr));
            return CNP_ERR_VALUE;
        }
    }

    CnpSafeEvalParser parser = {expr, expr, function_name, false};
    cnp_safe_eval_skip_space(&parser);
    if (!*parser.cursor) {
        cnp_safe_eval_syntax_error(&parser, "expression must not be empty");
        return CNP_ERR_VALUE;
    }

    double value = 0.0;
    bool signed_literal = false;
    if (!cnp_safe_eval_literal(&parser, &value, &signed_literal)) {
        return cnp_get_error(NULL);
    }
    cnp_safe_eval_skip_space(&parser);
    if (*parser.cursor) {
        cnp_safe_eval_syntax_error(
            &parser, "unexpected trailing expression syntax");
        return CNP_ERR_VALUE;
    }
    *out_value = value;
    return CNP_OK;
}

/* =========================================================================
 * cnp_safe_eval - Legacy scalar projection of the bounded numeric parser
 * ========================================================================= */
CNP_API double CNP_CALL cnp_safe_eval(const char *expr) {
    if (expr) {
        const char *direct = expr;
        while (isspace((unsigned char)*direct)) ++direct;
        const char *unsigned_start = direct;
        if (*unsigned_start == '+' || *unsigned_start == '-') ++unsigned_start;
        bool base_prefixed = unsigned_start[0] == '0' &&
            (unsigned_start[1] == 'x' || unsigned_start[1] == 'X' ||
             unsigned_start[1] == 'b' || unsigned_start[1] == 'B' ||
             unsigned_start[1] == 'o' || unsigned_start[1] == 'O');
        if (!base_prefixed &&
                (isdigit((unsigned char)*unsigned_start) ||
                 *unsigned_start == '.')) {
            char *direct_end = NULL;
            double direct_value = strtod(direct, &direct_end);
            if (direct_end > direct && *direct_end != '_') {
                while (isspace((unsigned char)*direct_end)) ++direct_end;
                if (!*direct_end) return direct_value;
            }
        }
    }
    double value = 0.0;
    CNP_STATUS status = cnp_safe_eval_v2(expr, &value);
    if (status != CNP_OK) {
        cnp_relabel_error("cnp_safe_eval");
        return NAN;
    }
    return value;
}
