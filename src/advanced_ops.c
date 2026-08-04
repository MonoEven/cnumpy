/**
 * cnumpy advanced array operations (unique functions not in other modules)
 * Functions already elsewhere:
 *   outer/inner/matmul/einsum/tensordot/cross (linalg.c)
 *   linspace/logspace/geomspace/eye/identity/tri (array.c)
 *   cumsum/cumprod (reduce.c), nan_to_num (math_ops.c)
 *   trapz/ediff1d (extra.c), diff/gradient (stats.c)
 *   meshgrid/indices/fromfunction (io.c)
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>

static int64_t advanced_flat_offset(
    const CnpArray *array, int64_t flat_index) {
    int64_t offset = array->offset;
    for (int dimension = array->ndim - 1;
            dimension >= 0; --dimension) {
        int64_t coordinate = flat_index % array->shape[dimension];
        flat_index /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

/* =========================================================================
 * cnp_vdot - Dot product of two vectors (flattened, conjugate for complex)
 * numpy.vdot(a, b)
 * ========================================================================= */
CNP_API double CNP_CALL cnp_vdot(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_vdot";
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both operands must not be null");
        return NAN;
    }
    if (a->size != b->size) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "flattened operand sizes %lld and %lld do not match",
            (long long)a->size, (long long)b->size);
        return NAN;
    }
    if (cnp_type_is_complex(a->dtype->type_num) ||
            cnp_type_is_complex(b->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "double-return ABI cannot represent complex vdot results");
        return NAN;
    }
    CnpArray *left = cnp_flatten(a, CNP_ORDER_C);
    CnpArray *right = cnp_flatten(b, CNP_ORDER_C);
    CnpArray *product = NULL;
    double value = NAN;
    if (left && right) product = cnp_dot(left, right);
    bool succeeded = product != NULL;
    if (product) {
        value = cnp_get_element_double(
            product->data, product->offset, product->dtype->type_num);
    }
    if (product) cnp_array_free(product);
    if (right) cnp_array_free(right);
    if (left) cnp_array_free(left);
    if (!succeeded) cnp_relabel_error(function_name);
    return value;
}

/* =========================================================================
 * cnp_tensordot_default - tensordot with default axes
 * Uses cnp_tensordot from linalg.c which takes (a, b, axes_a, axes_b)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_tensordot_default(const CnpArray *a, const CnpArray *b, int axes) {
    const char *function_name = "cnp_tensordot_default";
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both operands must not be null");
        return NULL;
    }
    CnpArray *result = cnp_tensordot(a, b, axes, axes);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_dot_general - Generalized dot product (delegates to matmul)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_dot_general(const CnpArray *a, const CnpArray *b) {
    CnpArray *result = cnp_dot(a, b);
    if (!result) cnp_relabel_error("cnp_dot_general");
    return result;
}

/* =========================================================================
 * cnp_tril - Lower triangle of array
 * numpy.tril(m, k=0)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_tril(const CnpArray *arr, int k) {
    const char *function_name = "cnp_tril";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "array is NULL");
        return NULL;
    }
    if (arr->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "array must have at least two dimensions");
        return NULL;
    }
    CnpArray *result = cnp_array_zeros(
        arr->ndim, arr->shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t columns = arr->shape[arr->ndim - 1];
    int64_t rows = arr->shape[arr->ndim - 2];
    int itemsize = arr->dtype->elsize;
    for (int64_t index = 0; index < arr->size; ++index) {
        int64_t column = index % columns;
        int64_t row = (index / columns) % rows;
        if (column - row > (int64_t)k) continue;
        int64_t source_offset = advanced_flat_offset(arr, index);
        memcpy(
            (char*)result->data + index * itemsize,
            (const char*)arr->data + source_offset,
            (size_t)itemsize);
    }
    return result;
}

/* =========================================================================
 * cnp_triu - Upper triangle of array
 * numpy.triu(m, k=0)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_triu(const CnpArray *arr, int k) {
    const char *function_name = "cnp_triu";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "array is NULL");
        return NULL;
    }
    if (arr->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "array must have at least two dimensions");
        return NULL;
    }
    CnpArray *result = cnp_array_zeros(
        arr->ndim, arr->shape, arr->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t columns = arr->shape[arr->ndim - 1];
    int64_t rows = arr->shape[arr->ndim - 2];
    int itemsize = arr->dtype->elsize;
    for (int64_t index = 0; index < arr->size; ++index) {
        int64_t column = index % columns;
        int64_t row = (index / columns) % rows;
        if (column - row < (int64_t)k) continue;
        int64_t source_offset = advanced_flat_offset(arr, index);
        memcpy(
            (char*)result->data + index * itemsize,
            (const char*)arr->data + source_offset,
            (size_t)itemsize);
    }
    return result;
}

/* =========================================================================
 * cnp_nancumsum - Cumulative sum ignoring NaN
 * numpy.nancumsum(a, axis=None)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_nancumsum(const CnpArray *arr, int axis) {
    CnpArray *result = cnp_nancumsum_v2(
        arr, axis, axis == CNP_AXIS_NONE, CNP_NOTYPE);
    if (!result) cnp_relabel_error("cnp_nancumsum");
    return result;
}

/* =========================================================================
 * cnp_nancumprod - Cumulative product ignoring NaN
 * numpy.nancumprod(a, axis=None)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_nancumprod(const CnpArray *arr, int axis) {
    CnpArray *result = cnp_nancumprod_v2(
        arr, axis, axis == CNP_AXIS_NONE, CNP_NOTYPE);
    if (!result) cnp_relabel_error("cnp_nancumprod");
    return result;
}

/* =========================================================================
 * cnp_interp_nd - Linear interpolation (numpy.interp)
 * ========================================================================= */
static bool interp_real_type(CNP_TYPE type) {
    return type == CNP_BOOL ||
        cnp_type_is_integer(type) || cnp_type_is_float(type);
}

static bool interp_numeric_type(CNP_TYPE type) {
    return interp_real_type(type) || cnp_type_is_complex(type);
}

static bool interp_validate_array(
    const CnpArray *array, const char *role,
    const char *function_name) {
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            array, function_name, &ignored_nbytes)) return false;
    if (array->ndim < 0 || array->ndim > CNP_MAXDIMS ||
            (array->ndim > 0 && (!array->shape || !array->strides)) ||
            (array->size > 0 && !array->data)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "%s array shape metadata and data buffer must be valid", role);
        return false;
    }
    return true;
}

CnpArray *cnp_interp_core(
    const CnpArray *x, const CnpArray *xp, const CnpArray *fp,
    bool explicit_bounds, double left, double right,
    const char *function_name) {
    if (!interp_validate_array(x, "x", function_name) ||
        !interp_validate_array(xp, "xp", function_name) ||
        !interp_validate_array(fp, "fp", function_name)) return NULL;
    if (!interp_real_type(x->dtype->type_num) ||
            !interp_real_type(xp->dtype->type_num) ||
            !interp_numeric_type(fp->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "x and xp must be real numeric and fp must be numeric");
        return NULL;
    }
    if (xp->ndim != 1 || fp->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "xp and fp must be one-dimensional");
        return NULL;
    }
    if (xp->size == 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "xp must contain at least one sample point");
        return NULL;
    }
    if (fp->size != xp->size) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "fp and xp must have the same length");
        return NULL;
    }
    for (int64_t index = 0; index < xp->size; ++index) {
        if (isnan(cnp_array_flat_get(xp, index))) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "xp must not contain NaN values");
            return NULL;
        }
    }

    CNP_TYPE result_type = cnp_type_is_complex(fp->dtype->type_num)
        ? CNP_CDOUBLE : CNP_DOUBLE;
    CnpArray *result = cnp_array_new(
        x->ndim, x->shape, result_type, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    double xp_first = cnp_array_flat_get(xp, 0);
    double xp_last = cnp_array_flat_get(xp, xp->size - 1);
    for (int64_t index = 0; index < x->size; ++index) {
        double point = cnp_array_flat_get(x, index);
        cnp_clongdouble value = {0};
        CNP_STATUS status = CNP_OK;
        if (isnan(point)) {
            value.real = NAN;
        } else if (point < xp_first) {
            if (explicit_bounds) {
                value.real = left;
            } else {
                int64_t offset = advanced_flat_offset(fp, 0);
                status = cnp_cast_scalar_value(
                    (const char*)fp->data + offset,
                    fp->dtype->type_num,
                    &value, CNP_CLONGDOUBLE, function_name);
            }
        } else if (point > xp_last) {
            if (explicit_bounds) {
                value.real = right;
            } else {
                int64_t offset = advanced_flat_offset(fp, fp->size - 1);
                status = cnp_cast_scalar_value(
                    (const char*)fp->data + offset,
                    fp->dtype->type_num,
                    &value, CNP_CLONGDOUBLE, function_name);
            }
        } else {
            int64_t lower = 0;
            int64_t upper = xp->size - 1;
            while (upper - lower > 1) {
                int64_t middle = lower + (upper - lower) / 2;
                if (cnp_array_flat_get(xp, middle) <= point)
                    lower = middle;
                else
                    upper = middle;
            }
            if (xp->size == 1 || point == xp_last) lower = xp->size - 1;
            if (lower == xp->size - 1) {
                int64_t offset = advanced_flat_offset(fp, lower);
                status = cnp_cast_scalar_value(
                    (const char*)fp->data + offset,
                    fp->dtype->type_num,
                    &value, CNP_CLONGDOUBLE, function_name);
            } else {
                cnp_clongdouble lower_value = {0};
                cnp_clongdouble upper_value = {0};
                int64_t lower_offset = advanced_flat_offset(fp, lower);
                int64_t upper_offset = advanced_flat_offset(fp, upper);
                status = cnp_cast_scalar_value(
                    (const char*)fp->data + lower_offset,
                    fp->dtype->type_num,
                    &lower_value, CNP_CLONGDOUBLE, function_name);
                if (status == CNP_OK)
                    status = cnp_cast_scalar_value(
                        (const char*)fp->data + upper_offset,
                        fp->dtype->type_num,
                        &upper_value, CNP_CLONGDOUBLE, function_name);
                double lower_point = cnp_array_flat_get(xp, lower);
                double upper_point = cnp_array_flat_get(xp, upper);
                long double fraction =
                    (point - lower_point) / (upper_point - lower_point);
                value.real = lower_value.real +
                    fraction * (upper_value.real - lower_value.real);
                value.imag = lower_value.imag +
                    fraction * (upper_value.imag - lower_value.imag);
            }
        }
        if (status == CNP_OK)
            status = cnp_cast_scalar_value(
                &value, CNP_CLONGDOUBLE,
                (char*)result->data + index * result->dtype->elsize,
                result_type, function_name);
        if (status != CNP_OK) {
            cnp_array_decref(result);
            return NULL;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_interp_nd(
    const CnpArray *x, const CnpArray *xp, const CnpArray *fp,
    double left, double right) {
    return cnp_interp_core(
        x, xp, fp, true, left, right, "cnp_interp_nd");
}

/* =========================================================================
 * cnp_kron_product - Kronecker product (if not already in linalg_ext.c)
 * Already in linalg_ext.c as cnp_kron - skip
 * ========================================================================= */

/* =========================================================================
 * cnp_matmul_1d1d - Inner product of two 1D arrays (returns scalar array)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_dot_1d(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_dot_1d";
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both operands must not be null");
        return NULL;
    }
    if (a->ndim != 1 || b->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "both operands must be one-dimensional");
        return NULL;
    }
    CnpArray *result = cnp_dot(a, b);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_outer_product - Outer product (wrapper using existing cnp_outer)
 * Already in linalg.c - skip
 * ========================================================================= */

/* =========================================================================
 * cnp_trace_ext - Trace with dtype support
 * cnp_trace already in reduce.c, this adds offset support for 2D
 * ========================================================================= */
CNP_API double CNP_CALL cnp_trace_ext(const CnpArray *arr, int offset) {
    const char *function_name = "cnp_trace_ext";
    if (!arr || arr->ndim != 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "legacy scalar trace requires a two-dimensional array");
        return NAN;
    }
    if (cnp_type_is_complex(arr->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "double-return ABI cannot represent complex trace results");
        return NAN;
    }
    CnpArray *result = cnp_trace(arr, offset, 0, 1);
    if (!result) {
        cnp_relabel_error(function_name);
        return NAN;
    }
    double value = cnp_get_element_double(
        result->data, result->offset, result->dtype->type_num);
    cnp_array_free(result);
    return value;
}

/* =========================================================================
 * cnp_einsum_matmul - einsum "ij,jk->ik" pattern (matrix multiply)
 * Uses existing cnp_matmul
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_einsum_matmul(const CnpArray *a, const CnpArray *b) {
    const CnpArray *operands[2] = {a, b};
    CnpArray *result = cnp_einsum("ij,jk->ik", 2, operands);
    if (!result) cnp_relabel_error("cnp_einsum_matmul");
    return result;
}

/* =========================================================================
 * cnp_einsum_dot - einsum "i,i->" pattern (dot product)
 * ========================================================================= */
static bool advanced_scalar_einsum_input(
    const CnpArray *array, const char *role, const char *function_name) {
    if (!array) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "%s array is required", role);
        return false;
    }
    if (!array->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name, "%s array requires a dtype", role);
        return false;
    }
    CNP_TYPE type = array->dtype->type_num;
    if (!(type == CNP_BOOL || cnp_type_is_integer(type) ||
            cnp_type_is_float(type))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "%s array must have a represented real numeric dtype because this legacy ABI returns double",
            role);
        return false;
    }
    return true;
}

static double advanced_scalar_einsum(
    const char *subscripts, int operand_count,
    const CnpArray *const *operands, const char *function_name) {
    CnpArray *result = cnp_einsum(
        subscripts, operand_count, operands);
    if (!result) {
        cnp_relabel_error(function_name);
        return 0.0;
    }
    if (result->ndim != 0 || result->size != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "einsum scalar projection did not produce a scalar result");
        cnp_array_decref(result);
        return 0.0;
    }
    double value = cnp_array_flat_get(result, 0);
    cnp_array_decref(result);
    return value;
}

CNP_API double CNP_CALL cnp_einsum_dot(
    const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_einsum_dot";
    if (!advanced_scalar_einsum_input(a, "left", function_name) ||
        !advanced_scalar_einsum_input(b, "right", function_name)) return 0.0;
    const CnpArray *operands[2] = {a, b};
    return advanced_scalar_einsum("i,i->", 2, operands, function_name);
}

/* =========================================================================
 * cnp_einsum_outer - einsum "i,j->ij" pattern (outer product)
 * Uses existing cnp_outer from linalg.c
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_einsum_outer(const CnpArray *a, const CnpArray *b) {
    const CnpArray *operands[2] = {a, b};
    CnpArray *result = cnp_einsum("i,j->ij", 2, operands);
    if (!result) cnp_relabel_error("cnp_einsum_outer");
    return result;
}

/* =========================================================================
 * cnp_einsum_trace - einsum "ii->" pattern (trace/sum of diagonal)
 * ========================================================================= */
CNP_API double CNP_CALL cnp_einsum_trace(const CnpArray *arr) {
    const char *function_name = "cnp_einsum_trace";
    if (!advanced_scalar_einsum_input(
            arr, "input", function_name)) return 0.0;
    const CnpArray *operands[1] = {arr};
    return advanced_scalar_einsum("ii->", 1, operands, function_name);
}

/* =========================================================================
 * cnp_einsum_sum - einsum "ij->" pattern (sum all elements)
 * ========================================================================= */
CNP_API double CNP_CALL cnp_einsum_sum(const CnpArray *arr) {
    const char *function_name = "cnp_einsum_sum";
    if (!advanced_scalar_einsum_input(
            arr, "input", function_name)) return 0.0;
    const CnpArray *operands[1] = {arr};
    return advanced_scalar_einsum("ij->", 1, operands, function_name);
}

/* =========================================================================
 * cnp_einsum_diag - einsum "ii->i" pattern (extract diagonal)
 * Uses existing cnp_diagonal from stride_tricks.c
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_einsum_diag(const CnpArray *arr) {
    const CnpArray *operands[1] = {arr};
    CnpArray *result = cnp_einsum("ii->i", 1, operands);
    if (!result) cnp_relabel_error("cnp_einsum_diag");
    return result;
}

/* =========================================================================
 * cnp_einsum_transpose - einsum "ij->ji" pattern (transpose)
 * Uses existing cnp_transpose
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_einsum_transpose(const CnpArray *arr) {
    const CnpArray *operands[1] = {arr};
    CnpArray *result = cnp_einsum("ij->ji", 1, operands);
    if (!result) cnp_relabel_error("cnp_einsum_transpose");
    return result;
}

/* =========================================================================
 * cnp_einsum_matvec - einsum "ij,j->i" pattern (matrix-vector multiply)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_einsum_matvec(const CnpArray *a, const CnpArray *b) {
    const CnpArray *operands[2] = {a, b};
    CnpArray *result = cnp_einsum("ij,j->i", 2, operands);
    if (!result) cnp_relabel_error("cnp_einsum_matvec");
    return result;
}
