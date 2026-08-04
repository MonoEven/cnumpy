/**
 * cnumpy array creation, destruction, and properties
 */
#include "../include/cnumpy/cnumpy_internal.h"

#define CNP_VIRTUAL_ZERO_THRESHOLD (64u * 1024u)
#define CNP_COPY_TILE_SIZE 32

uint32_t cnp_compute_layout_flags(
    int ndim, const int64_t *shape, const int64_t *strides, int itemsize) {
    if (ndim == 0) {
        return CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS;
    }

    bool c_contiguous = true;
    bool f_contiguous = true;
    int64_t expected = itemsize;
    for (int dimension = ndim - 1; dimension >= 0; --dimension) {
        if (shape[dimension] == 0) {
            return CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS;
        }
        if (shape[dimension] > 1) {
            if (strides[dimension] != expected) c_contiguous = false;
            if (expected > INT64_MAX / shape[dimension]) {
                c_contiguous = false;
            } else {
                expected *= shape[dimension];
            }
        }
    }

    expected = itemsize;
    for (int dimension = 0; dimension < ndim; ++dimension) {
        if (shape[dimension] > 1) {
            if (strides[dimension] != expected) f_contiguous = false;
            if (expected > INT64_MAX / shape[dimension]) {
                f_contiguous = false;
            } else {
                expected *= shape[dimension];
            }
        }
    }

    uint32_t flags = 0;
    if (c_contiguous) flags |= CNP_ARRAY_C_CONTIGUOUS;
    if (f_contiguous) flags |= CNP_ARRAY_F_CONTIGUOUS;
    return flags;
}

/* =========================================================================
 * Array allocation and deallocation
 * ========================================================================= */
static CnpArray* alloc_array_struct(void) {
    CnpArray *arr = (CnpArray*)cnp_calloc(1, sizeof(CnpArray));
    if (!arr) {
        cnp_set_error(CNP_ERR_MEMORY, "alloc_array", "Failed to allocate array structure");
        return NULL;
    }
    arr->refcount = 1;
    return arr;
}

static CnpArray* alloc_array_metadata(int ndim, const int64_t *shape,
                                      CNP_TYPE dtype, CNP_ORDER order,
                                      const char *func) {
    if (ndim < 0 || ndim > CNP_MAXDIMS) {
        cnp_set_error(CNP_ERR_SHAPE, func, "Invalid ndim: %d", ndim);
        return NULL;
    }

    CnpArray *arr = alloc_array_struct();
    if (!arr) return NULL;

    arr->ndim = ndim;
    arr->dtype = cnp_dtype_new(dtype);
    if (!arr->dtype) {
        cnp_free(arr, sizeof(CnpArray));
        cnp_set_error(CNP_ERR_TYPE, func, "Invalid dtype: %d", dtype);
        return NULL;
    }

    if (ndim > 0) {
        /* Single allocation for shape + strides (reduces malloc overhead) */
        int64_t *shape_strides = (int64_t*)cnp_malloc(2 * ndim * sizeof(int64_t));
        if (!shape_strides) {
            cnp_free(arr, sizeof(CnpArray));
            cnp_set_error(CNP_ERR_MEMORY, func,
                          "Failed to allocate shape/strides");
            return NULL;
        }
        arr->shape = shape_strides;
        arr->strides = shape_strides + ndim;
        for (int i = 0; i < ndim; i++) arr->shape[i] = shape[i];
        cnp_compute_strides(ndim, shape, arr->dtype->elsize, order, arr->strides);
        for (int i = 0; i < ndim; ++i) {
            if (shape[i] == 0) {
                memset(arr->strides, 0,
                       (size_t)ndim * sizeof(int64_t));
                break;
            }
        }
    } else {
        arr->shape = NULL;
        arr->strides = NULL;
    }

    arr->size = cnp_compute_size(ndim, shape);
    arr->flags = CNP_ARRAY_OWNDATA | CNP_ARRAY_ALIGNED |
        CNP_ARRAY_WRITEABLE |
        cnp_compute_layout_flags(
            ndim, arr->shape, arr->strides, arr->dtype->elsize);
    arr->base = NULL;
    arr->offset = 0;

    return arr;
}

CnpArray *cnp_array_adopt_external_data(
    int ndim, const int64_t *shape, CNP_TYPE dtype, CNP_ORDER order,
    void *data, uint32_t extra_flags,
    void *owner, CnpArrayOwnerRelease owner_release,
    const char *function_name) {
    if (!data || !owner || !owner_release) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "external data, owner, and release callback are required");
        return NULL;
    }
    CnpArray *arr = alloc_array_metadata(
        ndim, shape, dtype, order, function_name);
    if (!arr) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    arr->data = data;
    arr->flags &= ~(
        CNP_ARRAY_OWNDATA | CNP_ARRAY_VIRTUAL_ALLOC |
        CNP_ARRAY_WRITEABLE | CNP_ARRAY_ALIGNED);
    arr->flags |= extra_flags;
    arr->owner = owner;
    arr->owner_release = owner_release;
    return arr;
}

CnpArray* cnp_array_view_from_metadata(
    CnpArray *base, int ndim, const int64_t *shape,
    const int64_t *strides, int64_t offset,
    uint32_t layout_flags) {
    if (!base) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_array_view_from_metadata",
                      "Base array is NULL");
        return NULL;
    }
    if (ndim < 0 || ndim > CNP_MAXDIMS) {
        cnp_set_error(CNP_ERR_SHAPE, "cnp_array_view_from_metadata",
                      "Invalid ndim: %d", ndim);
        return NULL;
    }
    if (ndim > 0 && (!shape || !strides)) {
        cnp_set_error(CNP_ERR_SHAPE, "cnp_array_view_from_metadata",
                      "Shape and strides are required for ndim %d", ndim);
        return NULL;
    }
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) {
            cnp_set_error(CNP_ERR_SHAPE, "cnp_array_view_from_metadata",
                          "Negative dimension at axis %d", i);
            return NULL;
        }
    }

    CnpArray *view = alloc_array_struct();
    if (!view) return NULL;

    int64_t *metadata = NULL;
    if (ndim > 0) {
        metadata = (int64_t*)cnp_malloc(
            2 * (size_t)ndim * sizeof(int64_t));
        if (!metadata) {
            cnp_free(view, sizeof(CnpArray));
            cnp_set_error(CNP_ERR_MEMORY, "cnp_array_view_from_metadata",
                          "Failed to allocate merged shape/strides");
            return NULL;
        }
        memcpy(metadata, shape, (size_t)ndim * sizeof(int64_t));
        memcpy(metadata + ndim, strides,
               (size_t)ndim * sizeof(int64_t));
    }

    view->ndim = ndim;
    view->shape = metadata;
    view->strides = metadata ? metadata + ndim : NULL;
    view->size = cnp_compute_size(ndim, shape);
    view->data = base->data;
    view->dtype = base->dtype;
    view->offset = offset;
    (void)layout_flags;
    view->flags = (base->flags &
        ~(CNP_ARRAY_OWNDATA | CNP_ARRAY_VIRTUAL_ALLOC |
          CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) |
        cnp_compute_layout_flags(
            ndim, view->shape, view->strides, view->dtype->elsize);
    view->base = base;
    cnp_array_incref(base);
    return view;
}

CNP_API CnpArray* CNP_CALL cnp_array_new(int ndim, const int64_t *shape,
                                          CNP_TYPE dtype, CNP_ORDER order) {
    CnpArray *arr = alloc_array_metadata(
        ndim, shape, dtype, order, "cnp_array_new");
    if (!arr) return NULL;

    size_t data_size = (size_t)arr->size * (size_t)arr->dtype->elsize;
    arr->data = cnp_malloc(data_size > 0 ? data_size : 1);
    if (!arr->data) {
        cnp_array_free(arr);
        cnp_set_error(CNP_ERR_MEMORY, "cnp_array_new",
                      "Failed to allocate data buffer");
        return NULL;
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_array_zeros(int ndim, const int64_t *shape, CNP_TYPE dtype, CNP_ORDER order) {
    CnpArray *arr = alloc_array_metadata(
        ndim, shape, dtype, order, "cnp_array_zeros");
    if (!arr) return NULL;

    size_t data_size = (size_t)arr->size * (size_t)arr->dtype->elsize;
    if (data_size >= CNP_VIRTUAL_ZERO_THRESHOLD) {
        arr->data = cnp_virtual_alloc(data_size);
        if (!arr->data) {
            cnp_array_free(arr);
            return NULL;
        }
        arr->flags |= CNP_ARRAY_VIRTUAL_ALLOC;
        return arr;
    }

    arr->data = cnp_malloc(data_size > 0 ? data_size : 1);
    if (!arr->data) {
        cnp_array_free(arr);
        cnp_set_error(CNP_ERR_MEMORY, "cnp_array_zeros",
                      "Failed to allocate data buffer");
        return NULL;
    }

    if (dtype == CNP_DOUBLE && arr->size > 0) {
        cnp_simd_zeros((double*)arr->data, arr->size);
    } else {
        memset(arr->data, 0, data_size);
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_array_ones(int ndim, const int64_t *shape, CNP_TYPE dtype, CNP_ORDER order) {
    return cnp_array_full(ndim, shape, 1.0, dtype, order);
}

CNP_API CnpArray* CNP_CALL cnp_array_empty(int ndim, const int64_t *shape, CNP_TYPE dtype, CNP_ORDER order) {
    return cnp_array_new(ndim, shape, dtype, order);
}

CNP_API CnpArray* CNP_CALL cnp_array_full(int ndim, const int64_t *shape, double fill_value, CNP_TYPE dtype, CNP_ORDER order) {
    CnpArray *arr = cnp_array_new(ndim, shape, dtype, order);
    if (!arr) return NULL;

    /* SIMD fast path for float64 */
    if (dtype == CNP_DOUBLE && arr->size > 0) {
        cnp_simd_fill((double*)arr->data, fill_value, arr->size);
        return arr;
    }

    int elsize = arr->dtype->elsize;
    for (int64_t i = 0; i < arr->size; i++) {
        cnp_set_element_double(arr->data, i * elsize, dtype, fill_value);
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_array_from_data(const void *data, int ndim, const int64_t *shape, CNP_TYPE dtype, CNP_ORDER order) {
    CnpArray *arr = cnp_array_new(ndim, shape, dtype, order);
    if (!arr) return NULL;
    memcpy(arr->data, data, (size_t)(arr->size * arr->dtype->elsize));
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_array_from_scalar(double value, CNP_TYPE dtype) {
    int64_t shape[1] = {1};
    CnpArray *arr = cnp_array_new(0, shape, dtype, CNP_ORDER_C);
    if (!arr) return NULL;
    cnp_set_element_double(arr->data, 0, dtype, value);
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_array_copy(const CnpArray *src) {
    if (!src) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_array_copy", "Source array is required");
        return NULL;
    }
    CnpArray *arr = cnp_array_new(src->ndim, src->shape, src->dtype->type_num, CNP_ORDER_C);
    if (!arr) {
        cnp_relabel_error("cnp_array_copy");
        return NULL;
    }

    /* Copy data respecting strides */
    int elsize = src->dtype->elsize;
    if (src->flags & CNP_ARRAY_C_CONTIGUOUS) {
        memcpy(arr->data, (char*)src->data + src->offset, (size_t)(src->size * elsize));
    } else if (src->ndim == 2 && src->dtype->type_num == CNP_DOUBLE) {
        const int64_t row_count = src->shape[0];
        const int64_t column_count = src->shape[1];
        const int64_t row_stride = src->strides[0];
        const int64_t column_stride = src->strides[1];
        const char *source_base = (const char*)src->data + src->offset;
        double *destination = (double*)arr->data;

        for (int64_t row_block = 0; row_block < row_count;
             row_block += CNP_COPY_TILE_SIZE) {
            int64_t row_end = row_block + CNP_COPY_TILE_SIZE;
            if (row_end > row_count) row_end = row_count;
            for (int64_t column_block = 0; column_block < column_count;
                 column_block += CNP_COPY_TILE_SIZE) {
                int64_t column_end = column_block + CNP_COPY_TILE_SIZE;
                if (column_end > column_count) column_end = column_count;
                for (int64_t row = row_block; row < row_end; ++row) {
                    const char *source_row = source_base + row * row_stride;
                    double *destination_row = destination + row * column_count;
                    for (int64_t column = column_block;
                         column < column_end; ++column) {
                        memcpy(destination_row + column,
                               source_row + column * column_stride,
                               sizeof(double));
                    }
                }
            }
        }
    } else {
        /* Non-contiguous: iterate */
        int64_t coords[CNP_MAXDIMS] = {0};
        for (int64_t i = 0; i < src->size; i++) {
            int64_t src_offset = src->offset + cnp_multi_to_offset(src->ndim, coords, src->strides);
            memcpy((char*)arr->data + i * elsize, (char*)src->data + src_offset, elsize);
            /* Increment coordinates */
            for (int d = src->ndim - 1; d >= 0; d--) {
                coords[d]++;
                if (coords[d] < src->shape[d]) break;
                coords[d] = 0;
            }
        }
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_array_view(CnpArray *src) {
    if (!src) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_array_view", "Source array is required");
        return NULL;
    }
    uint32_t layout_flags = src->flags &
        (CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS);
    CnpArray *view = cnp_array_view_from_metadata(
        src, src->ndim, src->shape, src->strides,
        src->offset, layout_flags);
    if (!view) cnp_relabel_error("cnp_array_view");
    return view;
}

CnpArray* cnp_array_reshape_view(CnpArray *src, int ndim,
                                 const int64_t *shape, CNP_ORDER order) {
    if (!src) return NULL;
    if (ndim < 0 || ndim > CNP_MAXDIMS ||
        (ndim > 0 && !shape)) {
        cnp_set_error(CNP_ERR_SHAPE, "cnp_array_reshape_view",
                      "Invalid reshape metadata");
        return NULL;
    }
    int64_t strides[CNP_MAXDIMS];
    if (ndim > 0) {
        cnp_compute_strides(
            ndim, shape, src->dtype->elsize, order, strides);
    }
    uint32_t layout_flags = order == CNP_ORDER_F
        ? CNP_ARRAY_F_CONTIGUOUS : CNP_ARRAY_C_CONTIGUOUS;
    return cnp_array_view_from_metadata(
        src, ndim, shape, strides, src->offset, layout_flags);
}

CNP_API void CNP_CALL cnp_array_incref(CnpArray *arr) {
    if (arr) arr->refcount++;
}

CNP_API void CNP_CALL cnp_array_decref(CnpArray *arr) {
    if (!arr) return;
    arr->refcount--;
    if (arr->refcount <= 0) {
        cnp_array_free(arr);
    }
}

CNP_API void CNP_CALL cnp_array_free(CnpArray *arr) {
    if (!arr) return;
    if (arr->owner_release && arr->owner) {
        CnpArrayOwnerRelease release = arr->owner_release;
        void *owner = arr->owner;
        arr->owner_release = NULL;
        arr->owner = NULL;
        release(owner);
        arr->data = NULL;
    } else if ((arr->flags & CNP_ARRAY_OWNDATA) && arr->data) {
        size_t data_size =
            (size_t)arr->size * (size_t)arr->dtype->elsize;
        if (arr->flags & CNP_ARRAY_VIRTUAL_ALLOC) {
            if (cnp_virtual_free(arr->data, data_size) != CNP_OK)
                return;
        } else {
            cnp_free(arr->data, data_size > 0 ? data_size : 1);
        }
    }
    if (arr->shape) {
        /* Merged allocation: strides == shape + ndim */
        if (arr->strides == arr->shape + arr->ndim) {
            cnp_free(arr->shape, 2 * arr->ndim * sizeof(int64_t));
        } else {
            cnp_free(arr->shape, arr->ndim * sizeof(int64_t));
            if (arr->strides) cnp_free(arr->strides, arr->ndim * sizeof(int64_t));
        }
    } else if (arr->strides) {
        cnp_free(arr->strides, arr->ndim * sizeof(int64_t));
    }
    if (arr->base) {
        cnp_array_decref(arr->base);
    }
    cnp_free(arr, sizeof(CnpArray));
}

/* =========================================================================
 * Array creation - range functions
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_arange(double start, double stop, double step, CNP_TYPE dtype) {
    if (step == 0.0) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_arange", "step cannot be zero");
        return NULL;
    }

    int64_t n = (int64_t)ceil((stop - start) / step);
    if (n < 0) n = 0;

    int64_t shape[1] = {n};
    CnpArray *arr = cnp_array_new(1, shape, dtype, CNP_ORDER_C);
    if (!arr) return NULL;

    if (dtype == CNP_DOUBLE) {
        cnp_simd_arange((double*)arr->data, start, step, n);
        return arr;
    }
    int elsize = arr->dtype->elsize;
    for (int64_t i = 0; i < n; i++) {
        cnp_set_element_double(arr->data, i * elsize, dtype, start + i * step);
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_linspace(double start, double stop, int64_t num, bool endpoint, CNP_TYPE dtype) {
    if (num < 0) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_linspace", "num must be non-negative");
        return NULL;
    }

    int64_t shape[1] = {num};
    CnpArray *arr = cnp_array_new(1, shape, dtype, CNP_ORDER_C);
    if (!arr) return NULL;

    if (num == 0) return arr;

    double div = endpoint ? (double)(num - 1) : (double)num;
    if (num == 1) {
        cnp_set_element_double(arr->data, 0, dtype, start);
        return arr;
    }

    double step = (stop - start) / div;
    if (dtype == CNP_DOUBLE) {
        cnp_simd_arange((double*)arr->data, start, step, num);
        if (endpoint) ((double*)arr->data)[num - 1] = stop;
        return arr;
    }
    int elsize = arr->dtype->elsize;
    for (int64_t i = 0; i < num; i++) {
        cnp_set_element_double(arr->data, i * elsize, dtype, start + i * step);
    }
    /* Ensure exact endpoint */
    if (endpoint && num > 1) {
        cnp_set_element_double(arr->data, (num - 1) * elsize, dtype, stop);
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_logspace(double start, double stop, int64_t num, bool endpoint, double base, CNP_TYPE dtype) {
    CnpArray *lin = cnp_linspace(start, stop, num, endpoint, CNP_DOUBLE);
    if (!lin) return NULL;

    int64_t shape[1] = {num};
    CnpArray *arr = cnp_array_new(1, shape, dtype, CNP_ORDER_C);
    if (!arr) { cnp_array_free(lin); return NULL; }

    int elsize = arr->dtype->elsize;
    for (int64_t i = 0; i < num; i++) {
        double exp_val = cnp_get_element_double(lin->data, i * sizeof(double), CNP_DOUBLE);
        cnp_set_element_double(arr->data, i * elsize, dtype, pow(base, exp_val));
    }
    cnp_array_free(lin);
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_geomspace(double start, double stop, int64_t num, bool endpoint, CNP_TYPE dtype) {
    if (start == 0.0 || stop == 0.0) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_geomspace", "start and stop must be non-zero");
        return NULL;
    }
    bool negative = start < 0.0 && stop < 0.0;
    double log_start = log10(negative ? -start : start);
    double log_stop = log10(negative ? -stop : stop);
    CnpArray *result = cnp_logspace(
        log_start, log_stop, num, endpoint, 10.0, dtype);
    if (!result) {
        cnp_relabel_error("cnp_geomspace");
        return NULL;
    }
    if (negative) {
        int elsize = result->dtype->elsize;
        for (int64_t index = 0; index < result->size; ++index) {
            double value = cnp_get_element_double(
                result->data, index * elsize, dtype);
            cnp_set_element_double(
                result->data, index * elsize, dtype, -value);
        }
    }
    return result;
}

/* =========================================================================
 * Array creation - special matrices
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_eye(int64_t n, int64_t m, int64_t k, CNP_TYPE dtype) {
    if (m <= 0) m = n;
    int64_t shape[2] = {n, m};
    CnpArray *arr = cnp_array_zeros(2, shape, dtype, CNP_ORDER_C);
    if (!arr) return NULL;

    int elsize = arr->dtype->elsize;
    int64_t stride0 = arr->strides[0];
    int64_t stride1 = arr->strides[1];

    for (int64_t i = 0; i < n; i++) {
        int64_t j = i + k;
        if (j >= 0 && j < m) {
            cnp_set_element_double(arr->data, i * stride0 + j * stride1, dtype, 1.0);
        }
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_identity(int64_t n, CNP_TYPE dtype) {
    return cnp_eye(n, n, 0, dtype);
}

CNP_API CnpArray* CNP_CALL cnp_diag(const CnpArray *v, int64_t k) {
    if (!v) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_diag", "Input array is required");
        return NULL;
    }

    if (v->ndim == 1) {
        /* Vector -> diagonal matrix */
        int64_t n = v->shape[0] + (k >= 0 ? k : -k);
        int64_t shape[2] = {n, n};
        CnpArray *arr = cnp_array_zeros(2, shape, v->dtype->type_num, CNP_ORDER_C);
        if (!arr) return NULL;

        for (int64_t i = 0; i < v->shape[0]; i++) {
            int64_t row = k >= 0 ? i : i - k;
            int64_t col = k >= 0 ? i + k : i;
            double val = cnp_get_element_double(v->data, i * v->strides[0], v->dtype->type_num);
            cnp_set_element_double(arr->data, row * arr->strides[0] + col * arr->strides[1], v->dtype->type_num, val);
        }
        return arr;
    } else if (v->ndim == 2) {
        /* Matrix -> extract diagonal */
        int64_t n = v->shape[0] < v->shape[1] ? v->shape[0] : v->shape[1];
        if (k > 0) n = (v->shape[1] - k) < n ? (v->shape[1] - k) : n;
        else if (k < 0) n = (v->shape[0] + k) < n ? (v->shape[0] + k) : n;
        if (n < 0) n = 0;

        int64_t shape[1] = {n};
        CnpArray *arr = cnp_array_new(1, shape, v->dtype->type_num, CNP_ORDER_C);
        if (!arr) return NULL;

        for (int64_t i = 0; i < n; i++) {
            int64_t row = k >= 0 ? i : i - k;
            int64_t col = k >= 0 ? i + k : i;
            double val = cnp_get_element_double(v->data, row * v->strides[0] + col * v->strides[1], v->dtype->type_num);
            cnp_set_element_double(arr->data, i * arr->dtype->elsize, v->dtype->type_num, val);
        }
        return arr;
    }

    cnp_set_error(CNP_ERR_SHAPE, "cnp_diag", "Input must be 1-D or 2-D");
    return NULL;
}

CNP_API CnpArray* CNP_CALL cnp_tri(int64_t n, int64_t m, int64_t k, CNP_TYPE dtype) {
    if (m <= 0) m = n;
    int64_t shape[2] = {n, m};
    CnpArray *arr = cnp_array_zeros(2, shape, dtype, CNP_ORDER_C);
    if (!arr) return NULL;

    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < m; j++) {
            if (j <= i + k) {
                cnp_set_element_double(arr->data, i * arr->strides[0] + j * arr->strides[1], dtype, 1.0);
            }
        }
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_zeros_like(const CnpArray *arr) {
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_zeros_like", "Input array is required");
        return NULL;
    }
    CnpArray *result = cnp_array_zeros(
        arr->ndim, arr->shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) cnp_relabel_error("cnp_zeros_like");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_ones_like(const CnpArray *arr) {
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_ones_like", "Input array is required");
        return NULL;
    }
    CnpArray *result = cnp_array_ones(
        arr->ndim, arr->shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) cnp_relabel_error("cnp_ones_like");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_empty_like(const CnpArray *arr) {
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_empty_like", "Input array is required");
        return NULL;
    }
    CnpArray *result = cnp_array_empty(
        arr->ndim, arr->shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) cnp_relabel_error("cnp_empty_like");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_full_like(const CnpArray *arr, double fill_value) {
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_full_like", "Input array is required");
        return NULL;
    }
    CnpArray *result = cnp_array_full(
        arr->ndim, arr->shape, fill_value,
        arr->dtype->type_num, CNP_ORDER_C);
    if (!result) cnp_relabel_error("cnp_full_like");
    return result;
}

/* =========================================================================
 * Array properties
 * ========================================================================= */
CNP_API int CNP_CALL cnp_array_ndim(const CnpArray *arr) { return arr ? arr->ndim : 0; }
CNP_API int64_t CNP_CALL cnp_array_size(const CnpArray *arr) { return arr ? arr->size : 0; }
CNP_API int CNP_CALL cnp_array_itemsize(const CnpArray *arr) { return arr ? arr->dtype->elsize : 0; }

bool cnp_array_nbytes_checked(
    const CnpArray *arr, const char *function_name, int64_t *nbytes) {
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "source array is required");
        return false;
    }
    if (!arr->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array must have a dtype");
        return false;
    }
    if (arr->dtype->type_num <= CNP_NOTYPE ||
            arr->dtype->type_num >= CNP_NTYPES) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array dtype %d is not a valid CNP_TYPE",
            (int)arr->dtype->type_num);
        return false;
    }
    if (arr->dtype->elsize < 0) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array dtype itemsize must not be negative");
        return false;
    }
    if (arr->size < 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source array size must not be negative");
        return false;
    }
    if (arr->dtype->elsize > 0 &&
            arr->size > INT64_MAX / arr->dtype->elsize) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source array nbytes exceeds INT64_MAX");
        return false;
    }
    *nbytes = arr->size * arr->dtype->elsize;
    return true;
}

CNP_API int64_t CNP_CALL cnp_array_nbytes(const CnpArray *arr) {
    int64_t nbytes;
    if (!cnp_array_nbytes_checked(
            arr, "cnp_array_nbytes", &nbytes)) return -1;
    return nbytes;
}

static bool array_is_contiguous(
    const CnpArray *arr, uint32_t flag, const char *function_name) {
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "source array is required");
        return false;
    }
    return (arr->flags & flag) != 0;
}

CNP_API bool CNP_CALL cnp_array_is_c_contiguous(const CnpArray *arr) {
    return array_is_contiguous(
        arr, CNP_ARRAY_C_CONTIGUOUS, "cnp_array_is_c_contiguous");
}

CNP_API bool CNP_CALL cnp_array_is_f_contiguous(const CnpArray *arr) {
    return array_is_contiguous(
        arr, CNP_ARRAY_F_CONTIGUOUS, "cnp_array_is_f_contiguous");
}
CNP_API CNP_TYPE CNP_CALL cnp_array_dtype_num(const CnpArray *arr) { return arr ? arr->dtype->type_num : CNP_NOTYPE; }
CNP_API const int64_t* CNP_CALL cnp_array_shape(const CnpArray *arr) { return arr ? arr->shape : NULL; }
CNP_API const int64_t* CNP_CALL cnp_array_strides(const CnpArray *arr) { return arr ? arr->strides : NULL; }

/* =========================================================================
 * Element access
 * ========================================================================= */
CNP_API void* CNP_CALL cnp_array_at(const CnpArray *arr, const int64_t *indices) {
    if (!arr || !indices) return NULL;
    int64_t offset = arr->offset + cnp_multi_to_offset(arr->ndim, indices, arr->strides);
    return (char*)arr->data + offset;
}

CNP_API double CNP_CALL cnp_array_get_double(const CnpArray *arr, const int64_t *indices) {
    if (!arr) return 0.0;
    int64_t offset = arr->offset + cnp_multi_to_offset(arr->ndim, indices, arr->strides);
    return cnp_get_element_double(arr->data, offset, arr->dtype->type_num);
}

CNP_API int64_t CNP_CALL cnp_array_get_int(const CnpArray *arr, const int64_t *indices) {
    if (!arr) return 0;
    int64_t offset = arr->offset + cnp_multi_to_offset(arr->ndim, indices, arr->strides);
    return cnp_get_element_int(arr->data, offset, arr->dtype->type_num);
}

CNP_API CNP_STATUS CNP_CALL cnp_array_set_double(CnpArray *arr, const int64_t *indices, double value) {
    if (!arr) return CNP_ERR_GENERIC;
    if (!(arr->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_array_set_double", "Array is not writeable");
        return CNP_ERR_GENERIC;
    }
    int64_t offset = arr->offset + cnp_multi_to_offset(arr->ndim, indices, arr->strides);
    cnp_set_element_double(arr->data, offset, arr->dtype->type_num, value);
    return CNP_OK;
}

CNP_API CNP_STATUS CNP_CALL cnp_array_set_int(CnpArray *arr, const int64_t *indices, int64_t value) {
    if (!arr) return CNP_ERR_GENERIC;
    if (!(arr->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_array_set_int", "Array is not writeable");
        return CNP_ERR_GENERIC;
    }
    int64_t offset = arr->offset + cnp_multi_to_offset(arr->ndim, indices, arr->strides);
    cnp_set_element_int(arr->data, offset, arr->dtype->type_num, value);
    return CNP_OK;
}

CNP_API double CNP_CALL cnp_array_flat_get(const CnpArray *arr, int64_t flat_index) {
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_array_flat_get", "Array is required");
        return NAN;
    }
    if (flat_index < 0 || flat_index >= arr->size) {
        cnp_set_error(
            CNP_ERR_INDEX, "cnp_array_flat_get",
            "Flat index %lld is out of bounds for size %lld",
            (long long)flat_index, (long long)arr->size);
        return NAN;
    }
    /* Convert flat index to multi-index for strided access */
    int64_t coords[CNP_MAXDIMS] = {0};
    int64_t remaining = flat_index;
    for (int d = arr->ndim - 1; d >= 0; d--) {
        coords[d] = remaining % arr->shape[d];
        remaining /= arr->shape[d];
    }
    int64_t offset = arr->offset + cnp_multi_to_offset(arr->ndim, coords, arr->strides);
    return cnp_get_element_double(arr->data, offset, arr->dtype->type_num);
}

CNP_API CNP_STATUS CNP_CALL cnp_array_flat_set(CnpArray *arr, int64_t flat_index, double value) {
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_array_flat_set", "Array is required");
        return CNP_ERR_GENERIC;
    }
    if (flat_index < 0 || flat_index >= arr->size) {
        cnp_set_error(
            CNP_ERR_INDEX, "cnp_array_flat_set",
            "Flat index %lld is out of bounds for size %lld",
            (long long)flat_index, (long long)arr->size);
        return CNP_ERR_INDEX;
    }
    if (!(arr->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_array_flat_set",
            "Array is not writeable");
        return CNP_ERR_GENERIC;
    }
    int64_t coords[CNP_MAXDIMS] = {0};
    int64_t remaining = flat_index;
    for (int d = arr->ndim - 1; d >= 0; d--) {
        coords[d] = remaining % arr->shape[d];
        remaining /= arr->shape[d];
    }
    int64_t offset = arr->offset + cnp_multi_to_offset(arr->ndim, coords, arr->strides);
    cnp_set_element_double(arr->data, offset, arr->dtype->type_num, value);
    return CNP_OK;
}

/* =========================================================================
 * Type conversion
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_astype(const CnpArray *arr, CNP_TYPE dtype, CNP_CASTING casting) {
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_astype", "Input array is required");
        return NULL;
    }
    if (arr->dtype->type_num == dtype) {
        CnpArray *copy = cnp_array_copy(arr);
        if (!copy) cnp_relabel_error("cnp_astype");
        return copy;
    }

    if (!cnp_dtype_can_cast(arr->dtype->type_num, dtype, casting)) {
        cnp_set_error(CNP_ERR_TYPE, "cnp_astype", "Cannot cast %s to %s",
                      arr->dtype->name, cnp_dtype_new(dtype)->name);
        return NULL;
    }

    CnpArray *result = cnp_array_new(arr->ndim, arr->shape, dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error("cnp_astype");
        return NULL;
    }

    CNP_STATUS status = cnp_copyto(result, arr, casting);
    if (status != CNP_OK) {
        cnp_array_free(result);
        cnp_relabel_error("cnp_astype");
        return NULL;
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_array_from_int_array(const int64_t *data, int64_t size) {
    int64_t shape[1] = {size};
    return cnp_array_from_data(data, 1, shape, CNP_LONGLONG, CNP_ORDER_C);
}

CNP_API CnpArray* CNP_CALL cnp_array_from_double_array(const double *data, int64_t size) {
    int64_t shape[1] = {size};
    return cnp_array_from_data(data, 1, shape, CNP_DOUBLE, CNP_ORDER_C);
}

CNP_API CnpArray* CNP_CALL cnp_array_from_float_array(const float *data, int64_t size) {
    int64_t shape[1] = {size};
    return cnp_array_from_data(data, 1, shape, CNP_FLOAT, CNP_ORDER_C);
}

/* Internal: ensure contiguous */
CnpArray* cnp_ensure_contiguous(const CnpArray *arr, CNP_ORDER order) {
    if (!arr) return NULL;
    if (order == CNP_ORDER_C && (arr->flags & CNP_ARRAY_C_CONTIGUOUS)) return (CnpArray*)arr;
    if (order == CNP_ORDER_F && (arr->flags & CNP_ARRAY_F_CONTIGUOUS)) return (CnpArray*)arr;
    return cnp_array_copy(arr);
}

/* Internal: scalar array */
CnpArray* cnp_scalar_array(double value, CNP_TYPE dtype) {
    return cnp_array_from_scalar(value, dtype);
}
