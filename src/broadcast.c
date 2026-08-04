/**
 * cnumpy broadcasting and iterator implementation
 */
#include "../include/cnumpy/cnumpy_internal.h"

/* =========================================================================
 * Broadcasting
 * ========================================================================= */
CNP_API CNP_STATUS CNP_CALL cnp_broadcast_shapes(int narrays, const int64_t **shapes, const int *ndims,
                                                   int *out_ndim, int64_t **out_shape) {
    const char *function_name = "cnp_broadcast_shapes";
    if (!out_ndim || !out_shape) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "output ndim and shape pointers must not be null");
        return CNP_ERR_VALUE;
    }
    *out_ndim = 0;
    *out_shape = NULL;
    if (narrays <= 0 || narrays > CNP_MAXARGS || !shapes || !ndims) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "array count must be in [1, %d] and inputs must not be null",
            CNP_MAXARGS);
        return CNP_ERR_VALUE;
    }

    /* Find maximum ndim */
    int max_ndim = 0;
    for (int i = 0; i < narrays; i++) {
        if (ndims[i] < 0 || ndims[i] > CNP_MAXDIMS ||
                (ndims[i] > 0 && !shapes[i])) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "shape %d has invalid ndim or a null dimension pointer", i);
            return CNP_ERR_SHAPE;
        }
        for (int dimension = 0; dimension < ndims[i]; ++dimension) {
            if (shapes[i][dimension] < 0) {
                cnp_set_error(
                    CNP_ERR_SHAPE, function_name,
                    "shape %d dimension %d must be non-negative",
                    i, dimension);
                return CNP_ERR_SHAPE;
            }
        }
        if (ndims[i] > max_ndim) max_ndim = ndims[i];
    }

    int64_t *result_shape = NULL;
    if (max_ndim > 0) {
        result_shape = (int64_t*)cnp_malloc(max_ndim * sizeof(int64_t));
        if (!result_shape) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "cannot allocate broadcast shape");
            return CNP_ERR_MEMORY;
        }
    }

    /* Initialize with 1s */
    for (int i = 0; i < max_ndim; i++) result_shape[i] = 1;

    /* Broadcast from right to left */
    for (int i = 0; i < narrays; i++) {
        int offset = max_ndim - ndims[i];
        for (int d = 0; d < ndims[i]; d++) {
            int64_t dim = shapes[i][d];
            int64_t cur = result_shape[offset + d];
            if (dim == 1) continue;
            if (cur == 1) {
                result_shape[offset + d] = dim;
            } else if (cur != dim) {
                cnp_free(result_shape, max_ndim * sizeof(int64_t));
                cnp_set_error(CNP_ERR_BROADCAST, function_name,
                             "Cannot broadcast shapes: dimension %d has %lld and %lld",
                             offset + d, (long long)cur, (long long)dim);
                return CNP_ERR_BROADCAST;
            }
        }
    }

    *out_ndim = max_ndim;
    *out_shape = result_shape;
    return CNP_OK;
}

CNP_API void CNP_CALL cnp_broadcast_shape_free(int64_t *shape, int ndim) {
    if (!shape) return;
    if (ndim <= 0 || ndim > CNP_MAXDIMS) {
        cnp_set_error(
            CNP_ERR_VALUE, "cnp_broadcast_shape_free",
            "ndim must describe the allocated broadcast shape");
        return;
    }
    cnp_free(shape, (size_t)ndim * sizeof(int64_t));
}

CNP_API bool CNP_CALL cnp_can_broadcast(const CnpArray *a, const CnpArray *b) {
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_VALUE, "cnp_can_broadcast",
            "both arrays must not be null");
        return false;
    }
    int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;

    for (int i = 0; i < max_ndim; i++) {
        int64_t da = (i < max_ndim - a->ndim) ? 1 : a->shape[i - (max_ndim - a->ndim)];
        int64_t db = (i < max_ndim - b->ndim) ? 1 : b->shape[i - (max_ndim - b->ndim)];
        if (da != db && da != 1 && db != 1) return false;
    }
    return true;
}

/* Internal: create result array for binary ops with broadcasting */
CnpArray* cnp_binary_op_prepare(const CnpArray *a, const CnpArray *b, CNP_TYPE out_dtype) {
    if (!cnp_can_broadcast(a, b)) {
        cnp_set_error(CNP_ERR_BROADCAST, "binary_op", "Cannot broadcast arrays");
        return NULL;
    }

    int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    int64_t shape[CNP_MAXDIMS];

    for (int i = 0; i < max_ndim; i++) {
        int64_t da = (i < max_ndim - a->ndim) ? 1 : a->shape[i - (max_ndim - a->ndim)];
        int64_t db = (i < max_ndim - b->ndim) ? 1 : b->shape[i - (max_ndim - b->ndim)];
        shape[i] = da > db ? da : db;
    }

    return cnp_array_new(max_ndim, shape, out_dtype, CNP_ORDER_C);
}

/* Internal: create result array for unary ops */
CnpArray* cnp_unary_op_prepare(const CnpArray *a, CNP_TYPE out_dtype) {
    return cnp_array_new(a->ndim, a->shape, out_dtype, CNP_ORDER_C);
}

/* Helper: get element from array with broadcasting index */
static double get_broadcast_element(const CnpArray *arr, const int64_t *bcast_coords, int bcast_ndim) {
    int offset = bcast_ndim - arr->ndim;
    int64_t byte_offset = arr->offset;
    for (int d = 0; d < arr->ndim; d++) {
        int64_t coord = bcast_coords[offset + d];
        if (arr->shape[d] == 1) coord = 0;  /* Broadcast dimension */
        byte_offset += coord * arr->strides[d];
    }
    return cnp_get_element_double(arr->data, byte_offset, arr->dtype->type_num);
}

static const void* get_broadcast_element_pointer(
    const CnpArray *arr,
    const int64_t *broadcast_coordinates,
    int broadcast_ndim) {
    int coordinate_offset = broadcast_ndim - arr->ndim;
    int64_t byte_offset = arr->offset;
    for (int axis = 0; axis < arr->ndim; ++axis) {
        int64_t coordinate =
            broadcast_coordinates[coordinate_offset + axis];
        if (arr->shape[axis] == 1) coordinate = 0;
        byte_offset += coordinate * arr->strides[axis];
    }
    return (const char*)arr->data + byte_offset;
}

CNP_STATUS cnp_validate_logical_truth_dtype(
    CNP_TYPE dtype,
    bool allow_temporal,
    const char *function_name) {
    switch (dtype) {
        case CNP_BOOL:
        case CNP_BYTE:
        case CNP_UBYTE:
        case CNP_SHORT:
        case CNP_USHORT:
        case CNP_HALF:
        case CNP_INT:
        case CNP_UINT:
        case CNP_LONG:
        case CNP_ULONG:
        case CNP_LONGLONG:
        case CNP_ULONGLONG:
        case CNP_FLOAT:
        case CNP_DOUBLE:
        case CNP_CFLOAT:
        case CNP_CDOUBLE:
            return CNP_OK;
        case CNP_LONGDOUBLE:
            if (sizeof(long double) == sizeof(uint64_t)) return CNP_OK;
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "long double logical truth requires 64-bit storage");
            return CNP_ERR_TYPE;
        case CNP_CLONGDOUBLE:
            if (sizeof(long double) == sizeof(uint64_t)) return CNP_OK;
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "complex long double logical truth requires 64-bit components");
            return CNP_ERR_TYPE;
        case CNP_DATETIME:
        case CNP_TIMEDELTA:
            if (allow_temporal) return CNP_OK;
            break;
        default:
            break;
    }
    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "dtype %d does not support logical truth evaluation",
        (int)dtype);
    return CNP_ERR_TYPE;
}

CNP_STATUS cnp_scalar_truth(
    const void *source,
    CNP_TYPE dtype,
    bool *truth,
    const char *function_name) {
#define CNP_SCALAR_TRUTH(c_type) do { \
    c_type value; \
    memcpy(&value, source, sizeof(value)); \
    *truth = value != (c_type)0; \
    return CNP_OK; \
} while (0)

#define CNP_SCALAR_TRUTH_FLOAT_BITS(uint_type, sign_mask) do { \
    uint_type bits; \
    memcpy(&bits, source, sizeof(bits)); \
    *truth = (bits & ~(uint_type)(sign_mask)) != 0; \
    return CNP_OK; \
} while (0)

    switch (dtype) {
        case CNP_BOOL:
        case CNP_BYTE:
            CNP_SCALAR_TRUTH(int8_t);
        case CNP_UBYTE:
            CNP_SCALAR_TRUTH(uint8_t);
        case CNP_SHORT:
            CNP_SCALAR_TRUTH(int16_t);
        case CNP_USHORT:
            CNP_SCALAR_TRUTH(uint16_t);
        case CNP_HALF:
            CNP_SCALAR_TRUTH_FLOAT_BITS(uint16_t, UINT16_C(0x8000));
        case CNP_INT:
            CNP_SCALAR_TRUTH(int32_t);
        case CNP_UINT:
            CNP_SCALAR_TRUTH(uint32_t);
        case CNP_LONG:
        case CNP_LONGLONG:
            CNP_SCALAR_TRUTH(int64_t);
        case CNP_ULONG:
        case CNP_ULONGLONG:
            CNP_SCALAR_TRUTH(uint64_t);
        case CNP_FLOAT:
            CNP_SCALAR_TRUTH_FLOAT_BITS(uint32_t, UINT32_C(0x80000000));
        case CNP_DOUBLE:
            CNP_SCALAR_TRUTH_FLOAT_BITS(
                uint64_t, UINT64_C(0x8000000000000000));
        case CNP_LONGDOUBLE:
            if (sizeof(long double) != sizeof(uint64_t)) {
                cnp_set_error(
                    CNP_ERR_TYPE, function_name,
                    "long double logical truth requires 64-bit storage");
                return CNP_ERR_TYPE;
            }
            CNP_SCALAR_TRUTH_FLOAT_BITS(
                uint64_t, UINT64_C(0x8000000000000000));
        case CNP_CFLOAT: {
            uint32_t components[2];
            memcpy(&components, source, sizeof(components));
            *truth = (components[0] & UINT32_C(0x7fffffff)) != 0 ||
                (components[1] & UINT32_C(0x7fffffff)) != 0;
            return CNP_OK;
        }
        case CNP_CDOUBLE: {
            uint64_t components[2];
            memcpy(&components, source, sizeof(components));
            *truth =
                (components[0] & UINT64_C(0x7fffffffffffffff)) != 0 ||
                (components[1] & UINT64_C(0x7fffffffffffffff)) != 0;
            return CNP_OK;
        }
        case CNP_CLONGDOUBLE: {
            if (sizeof(long double) != sizeof(uint64_t)) {
                cnp_set_error(
                    CNP_ERR_TYPE, function_name,
                    "complex long double logical truth requires 64-bit components");
                return CNP_ERR_TYPE;
            }
            uint64_t components[2];
            memcpy(&components, source, sizeof(components));
            *truth =
                (components[0] & UINT64_C(0x7fffffffffffffff)) != 0 ||
                (components[1] & UINT64_C(0x7fffffffffffffff)) != 0;
            return CNP_OK;
        }
        default:
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "dtype %d does not support logical truth evaluation",
                (int)dtype);
            return CNP_ERR_TYPE;
    }

#undef CNP_SCALAR_TRUTH
#undef CNP_SCALAR_TRUTH_FLOAT_BITS
}

static CNP_STATUS cnp_binary_logical_truth(
    const void *source,
    CNP_TYPE dtype,
    bool *truth,
    const char *function_name) {
    if (dtype == CNP_DATETIME || dtype == CNP_TIMEDELTA) {
        int64_t value;
        memcpy(&value, source, sizeof(value));
        *truth = value != 0;
        return CNP_OK;
    }
    return cnp_scalar_truth(
        source, dtype, truth, function_name);
}

CNP_ORDER cnp_logical_result_order(
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

static CnpArray* cnp_logical_prepare_result(
    const CnpArray *left,
    const CnpArray *right,
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
        result_ndim, result_shape, CNP_BOOL,
        cnp_logical_result_order(left, right));
}

/* Internal: generic binary operation with broadcasting */
CnpArray* cnp_binary_op(const CnpArray *a, const CnpArray *b, cnp_binary_func func, CNP_TYPE out_dtype) {
    if (!a || !b) return NULL;

    CnpArray *result = cnp_binary_op_prepare(a, b, out_dtype);
    if (!result) return NULL;

    /* Fast path: both inputs contiguous float64, same shape, output float64 */
    if ((a->flags & CNP_ARRAY_C_CONTIGUOUS) && (b->flags & CNP_ARRAY_C_CONTIGUOUS) &&
        a->dtype->type_num == CNP_DOUBLE && b->dtype->type_num == CNP_DOUBLE &&
        out_dtype == CNP_DOUBLE && a->size == b->size && a->size == result->size &&
        a->ndim == b->ndim) {
        const double *pa = (const double*)((const char*)a->data + a->offset);
        const double *pb = (const double*)((const char*)b->data + b->offset);
        double *pr = (double*)result->data;
        int64_t n = a->size;
        int64_t i = 0;
        /* Process 4 elements at a time for ILP */
        for (; i + 3 < n; i += 4) {
            pr[i]   = func(pa[i],   pb[i]);
            pr[i+1] = func(pa[i+1], pb[i+1]);
            pr[i+2] = func(pa[i+2], pb[i+2]);
            pr[i+3] = func(pa[i+3], pb[i+3]);
        }
        for (; i < n; i++) {
            pr[i] = func(pa[i], pb[i]);
        }
        return result;
    }

    int ndim = result->ndim;
    int64_t coords[CNP_MAXDIMS] = {0};
    int elsize = result->dtype->elsize;

    for (int64_t i = 0; i < result->size; i++) {
        double va = get_broadcast_element(a, coords, ndim);
        double vb = get_broadcast_element(b, coords, ndim);
        double vr = func(va, vb);
        cnp_set_element_double(result->data, i * elsize, out_dtype, vr);

        /* Increment coordinates (C order) */
        for (int d = ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < result->shape[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}

/* Internal: generic unary operation */
CnpArray* cnp_unary_op(const CnpArray *a, cnp_unary_func func, CNP_TYPE out_dtype) {
    if (!a) return NULL;

    CnpArray *result = cnp_unary_op_prepare(a, out_dtype);
    if (!result) return NULL;

    /* Fast path: contiguous float64 input and output */
    if ((a->flags & CNP_ARRAY_C_CONTIGUOUS) && a->dtype->type_num == CNP_DOUBLE &&
        out_dtype == CNP_DOUBLE) {
        const double *pa = (const double*)((const char*)a->data + a->offset);
        double *pr = (double*)result->data;
        int64_t n = a->size;
        int64_t i = 0;
        for (; i + 3 < n; i += 4) {
            pr[i]   = func(pa[i]);
            pr[i+1] = func(pa[i+1]);
            pr[i+2] = func(pa[i+2]);
            pr[i+3] = func(pa[i+3]);
        }
        for (; i < n; i++) {
            pr[i] = func(pa[i]);
        }
        return result;
    }

    int elsize = result->dtype->elsize;
    int64_t coords[CNP_MAXDIMS] = {0};

    for (int64_t i = 0; i < result->size; i++) {
        int64_t src_offset = a->offset + cnp_multi_to_offset(a->ndim, coords, a->strides);
        double val = cnp_get_element_double(a->data, src_offset, a->dtype->type_num);
        cnp_set_element_double(result->data, i * elsize, out_dtype, func(val));

        /* Increment coordinates */
        for (int d = a->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < a->shape[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}

/* Internal: comparison operation */
CnpArray* cnp_compare_op(const CnpArray *a, const CnpArray *b, cnp_compare_func func) {
    if (!a || !b) return NULL;

    CnpArray *result = cnp_binary_op_prepare(a, b, CNP_BOOL);
    if (!result) return NULL;

    int ndim = result->ndim;
    int64_t coords[CNP_MAXDIMS] = {0};

    for (int64_t i = 0; i < result->size; i++) {
        double va = get_broadcast_element(a, coords, ndim);
        double vb = get_broadcast_element(b, coords, ndim);
        bool vr = func(va, vb);
        *((int8_t*)result->data + i) = vr ? 1 : 0;

        for (int d = ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < result->shape[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}

/* Internal: logical operation */
CnpArray* cnp_logical_op(
    const CnpArray *a,
    const CnpArray *b,
    cnp_logical_func func,
    cnp_logical_f64_binary_kernel simd_kernel,
    const char *function_name) {
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays are required");
        return NULL;
    }
    if (!a->dtype || !b->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "left and right arrays must have dtypes");
        return NULL;
    }
    if ((a->size > 0 && !a->data) || (b->size > 0 && !b->data)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "left and right arrays require data buffers");
        return NULL;
    }
    if (cnp_validate_logical_truth_dtype(
            a->dtype->type_num, true, function_name) != CNP_OK ||
            cnp_validate_logical_truth_dtype(
                b->dtype->type_num, true, function_name) != CNP_OK)
        return NULL;

    CnpArray *result = cnp_logical_prepare_result(a, b, function_name);
    if (!result) return NULL;

    if (result->size == 0) return result;

    bool exact_shape = a->ndim == b->ndim;
    for (int axis = 0; exact_shape && axis < a->ndim; ++axis)
        exact_shape = a->shape[axis] == b->shape[axis];
    if (simd_kernel && exact_shape &&
            (a->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            (b->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            (result->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            a->dtype->type_num == CNP_DOUBLE &&
            b->dtype->type_num == CNP_DOUBLE) {
        simd_kernel(
            (const double*)((const char*)a->data + a->offset),
            (const double*)((const char*)b->data + b->offset),
            (uint8_t*)((char*)result->data + result->offset),
            result->size);
        return result;
    }

    int ndim = result->ndim;
    int64_t coords[CNP_MAXDIMS] = {0};

    for (int64_t i = 0; i < result->size; i++) {
        bool va;
        bool vb;
        CNP_STATUS status = cnp_binary_logical_truth(
            get_broadcast_element_pointer(a, coords, ndim),
            a->dtype->type_num, &va, function_name);
        if (status == CNP_OK) {
            status = cnp_binary_logical_truth(
                get_broadcast_element_pointer(b, coords, ndim),
                b->dtype->type_num, &vb, function_name);
        }
        if (status != CNP_OK) {
            cnp_array_decref(result);
            return NULL;
        }
        bool vr = func(va, vb);
        int64_t result_offset = result->offset + cnp_multi_to_offset(
            result->ndim, coords, result->strides);
        *((int8_t*)result->data + result_offset) = vr ? 1 : 0;

        for (int d = ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < result->shape[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}

/* =========================================================================
 * Iterator implementation
 * ========================================================================= */
CNP_API CnpIter* CNP_CALL cnp_iter_new(CnpArray *arr) {
    const char *function_name = "cnp_iter_new";
    if (!arr) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "array must not be null");
        return NULL;
    }
    CnpIter *iter = (CnpIter*)cnp_calloc(1, sizeof(CnpIter));
    if (!iter) {
        cnp_set_error(CNP_ERR_MEMORY, function_name, "cannot allocate iterator");
        return NULL;
    }

    iter->array = arr;
    iter->size = arr->size;
    iter->index = 0;
    iter->done = (arr->size == 0);

    if (arr->ndim > 0) {
        iter->coordinates = (int64_t*)cnp_calloc(arr->ndim, sizeof(int64_t));
        if (!iter->coordinates) {
            cnp_free(iter, sizeof(CnpIter));
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "cannot allocate iterator coordinates");
            return NULL;
        }
    } else {
        iter->coordinates = NULL;
    }

    cnp_array_incref(arr);
    return iter;
}

CNP_API bool CNP_CALL cnp_iter_next(CnpIter *iter) {
    if (!iter) {
        cnp_set_error(CNP_ERR_VALUE, "cnp_iter_next", "iterator must not be null");
        return false;
    }
    if (iter->done) return false;

    iter->index++;
    if (iter->index >= iter->size) {
        iter->done = true;
        return false;
    }

    /* Increment coordinates */
    CnpArray *arr = iter->array;
    for (int d = arr->ndim - 1; d >= 0; d--) {
        iter->coordinates[d]++;
        if (iter->coordinates[d] < arr->shape[d]) break;
        iter->coordinates[d] = 0;
    }
    return true;
}

CNP_API void* CNP_CALL cnp_iter_data(CnpIter *iter) {
    if (!iter) {
        cnp_set_error(CNP_ERR_VALUE, "cnp_iter_data", "iterator must not be null");
        return NULL;
    }
    if (iter->done) return NULL;
    CnpArray *arr = iter->array;
    int64_t offset = arr->offset + cnp_multi_to_offset(arr->ndim, iter->coordinates, arr->strides);
    return (char*)arr->data + offset;
}

CNP_API void CNP_CALL cnp_iter_free(CnpIter *iter) {
    if (!iter) return;
    if (iter->coordinates) cnp_free(iter->coordinates, iter->array->ndim * sizeof(int64_t));
    if (iter->array) cnp_array_decref(iter->array);
    cnp_free(iter, sizeof(CnpIter));
}

CNP_API void CNP_CALL cnp_iter_reset(CnpIter *iter) {
    if (!iter) {
        cnp_set_error(CNP_ERR_VALUE, "cnp_iter_reset", "iterator must not be null");
        return;
    }
    iter->index = 0;
    iter->done = (iter->size == 0);
    if (iter->coordinates) {
        memset(iter->coordinates, 0, iter->array->ndim * sizeof(int64_t));
    }
}

/* =========================================================================
 * Multi-array iterator (for broadcasting)
 * ========================================================================= */
CNP_API CnpMultiIter* CNP_CALL cnp_multi_iter_new(int narrays, CnpArray **arrays) {
    const char *function_name = "cnp_multi_iter_new";
    if (narrays <= 0 || narrays > CNP_MAXARGS) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "array count must be in [1, %d]", CNP_MAXARGS);
        return NULL;
    }
    if (!arrays) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "array pointer list must not be null");
        return NULL;
    }

    const int64_t *shapes[CNP_MAXARGS];
    int ndims[CNP_MAXARGS];
    for (int index = 0; index < narrays; ++index) {
        if (!arrays[index]) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "array %d must not be null", index);
            return NULL;
        }
        shapes[index] = arrays[index]->shape;
        ndims[index] = arrays[index]->ndim;
    }

    int broadcast_ndim = 0;
    int64_t *broadcast_shape = NULL;
    CNP_STATUS broadcast_status = cnp_broadcast_shapes(
        narrays, shapes, ndims, &broadcast_ndim, &broadcast_shape);
    if (broadcast_status != CNP_OK) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    CnpMultiIter *miter = (CnpMultiIter*)cnp_calloc(1, sizeof(CnpMultiIter));
    if (!miter) {
        if (broadcast_shape)
            cnp_free(broadcast_shape, broadcast_ndim * sizeof(int64_t));
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "cannot allocate multi-iterator");
        return NULL;
    }

    miter->num_arrays = narrays;
    miter->arrays = (CnpArray**)cnp_malloc(narrays * sizeof(CnpArray*));
    miter->data_pointers = (void**)cnp_malloc(narrays * sizeof(void*));
    miter->ndim = broadcast_ndim;
    miter->shape = broadcast_shape;
    if (broadcast_ndim > 0)
        miter->coordinates = (int64_t*)cnp_calloc(
            broadcast_ndim, sizeof(int64_t));
    if (!miter->arrays || !miter->data_pointers ||
        (broadcast_ndim > 0 && !miter->coordinates)) {
        if (miter->arrays)
            cnp_free(miter->arrays, narrays * sizeof(CnpArray*));
        if (miter->data_pointers)
            cnp_free(miter->data_pointers, narrays * sizeof(void*));
        if (miter->coordinates)
            cnp_free(
                miter->coordinates, broadcast_ndim * sizeof(int64_t));
        if (broadcast_shape)
            cnp_free(broadcast_shape, broadcast_ndim * sizeof(int64_t));
        cnp_free(miter, sizeof(CnpMultiIter));
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "cannot allocate multi-iterator state");
        return NULL;
    }

    for (int index = 0; index < narrays; ++index) {
        miter->arrays[index] = arrays[index];
        cnp_array_incref(arrays[index]);
    }

    miter->size = cnp_compute_size(broadcast_ndim, broadcast_shape);
    miter->index = 0;
    miter->done = (miter->size == 0);

    return miter;
}

CNP_API bool CNP_CALL cnp_multi_iter_next(CnpMultiIter *miter) {
    if (!miter) {
        cnp_set_error(
            CNP_ERR_VALUE, "cnp_multi_iter_next",
            "multi-iterator must not be null");
        return false;
    }
    if (miter->done) return false;

    miter->index++;
    if (miter->index >= miter->size) {
        miter->done = true;
        return false;
    }

    for (int d = miter->ndim - 1; d >= 0; d--) {
        miter->coordinates[d]++;
        if (miter->coordinates[d] < miter->shape[d]) break;
        miter->coordinates[d] = 0;
    }
    return true;
}

CNP_API void** CNP_CALL cnp_multi_iter_data(CnpMultiIter *miter) {
    if (!miter) {
        cnp_set_error(
            CNP_ERR_VALUE, "cnp_multi_iter_data",
            "multi-iterator must not be null");
        return NULL;
    }
    if (miter->done) return NULL;
    for (int i = 0; i < miter->num_arrays; i++) {
        CnpArray *arr = miter->arrays[i];
        int offset = miter->ndim - arr->ndim;
        int64_t byte_offset = arr->offset;
        for (int d = 0; d < arr->ndim; d++) {
            int64_t coord = miter->coordinates[offset + d];
            if (arr->shape[d] == 1) coord = 0;
            byte_offset += coord * arr->strides[d];
        }
        miter->data_pointers[i] = (char*)arr->data + byte_offset;
    }
    return miter->data_pointers;
}

CNP_API void CNP_CALL cnp_multi_iter_free(CnpMultiIter *miter) {
    if (!miter) return;
    for (int i = 0; i < miter->num_arrays; i++) {
        if (miter->arrays[i]) cnp_array_decref(miter->arrays[i]);
    }
    if (miter->arrays) cnp_free(miter->arrays, miter->num_arrays * sizeof(CnpArray*));
    if (miter->iters) cnp_free(miter->iters, miter->num_arrays * sizeof(CnpIter));
    if (miter->data_pointers)
        cnp_free(
            miter->data_pointers, miter->num_arrays * sizeof(void*));
    if (miter->shape) cnp_free(miter->shape, miter->ndim * sizeof(int64_t));
    if (miter->coordinates) cnp_free(miter->coordinates, miter->ndim * sizeof(int64_t));
    cnp_free(miter, sizeof(CnpMultiIter));
}
