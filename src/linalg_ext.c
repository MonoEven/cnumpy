/**
 * cnumpy extended linear algebra
 * Additional functions: norm_ext, cond, tensorinv, tensorsolve, kron
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>

/* cnp_linalg_cholesky, cnp_linalg_eigh, cnp_linalg_eig, cnp_linalg_svd, cnp_linalg_qr
 * are already defined in linalg.c */

/* =========================================================================
 * cnp_linalg_norm_ext - Legacy scalar vector/matrix norm projection
 * ========================================================================= */
static double norm_ext_absolute(const CnpArray *array, int64_t offset) {
    const char *pointer = (const char*)array->data + offset;
    switch (array->dtype->type_num) {
        case CNP_CFLOAT: {
            const cnp_cfloat *value = (const cnp_cfloat*)pointer;
            return hypot((double)value->real, (double)value->imag);
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *value = (const cnp_cdouble*)pointer;
            return hypot(value->real, value->imag);
        }
        case CNP_CLONGDOUBLE: {
            const cnp_clongdouble *value = (const cnp_clongdouble*)pointer;
            return (double)hypotl(value->real, value->imag);
        }
        default:
            return fabs(cnp_get_element_double(
                array->data, offset, array->dtype->type_num));
    }
}

CNP_API double CNP_CALL cnp_linalg_norm_ext(
        const CnpArray *a, double ord, int axis) {
    const char *function_name = "cnp_linalg_norm_ext";
    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array must not be null");
        return NAN;
    }
    if (!(a->dtype->type_num == CNP_BOOL ||
          cnp_type_is_integer(a->dtype->type_num) ||
          cnp_type_is_float(a->dtype->type_num) ||
          cnp_type_is_complex(a->dtype->type_num))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input dtype is not supported by linear algebra");
        return NAN;
    }
    if (a->ndim != 1 && a->ndim != 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "legacy scalar projection requires a vector or matrix");
        return NAN;
    }
    if (cnp_normalize_axis(axis, a->ndim) < 0) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "axis %d is invalid for rank %d", axis, a->ndim);
        return NAN;
    }

    if (a->ndim == 1) {
        int64_t length = a->shape[0];
        if (isinf(ord) && length == 0) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "infinite-order norm has no identity for an empty vector");
            return NAN;
        }
        if (ord == 0.0) {
            double count = 0.0;
            for (int64_t index = 0; index < length; index++) {
                if (norm_ext_absolute(
                        a, a->offset + index * a->strides[0]) != 0.0) {
                    count += 1.0;
                }
            }
            return count;
        }
        if (isinf(ord)) {
            double selected = norm_ext_absolute(a, a->offset);
            for (int64_t index = 1; index < length; index++) {
                double value = norm_ext_absolute(
                    a, a->offset + index * a->strides[0]);
                if (isnan(value)) return NAN;
                if ((ord > 0.0 && value > selected) ||
                        (ord < 0.0 && value < selected)) {
                    selected = value;
                }
            }
            return selected;
        }
        if (ord == 2.0) {
            double magnitude = 0.0;
            for (int64_t index = 0; index < length; index++) {
                magnitude = hypot(
                    magnitude,
                    norm_ext_absolute(
                        a, a->offset + index * a->strides[0]));
            }
            return magnitude;
        }
        if (ord == 1.0) {
            double sum = 0.0;
            for (int64_t index = 0; index < length; index++) {
                sum += norm_ext_absolute(
                    a, a->offset + index * a->strides[0]);
            }
            return sum;
        }
        double powered_sum = 0.0;
        for (int64_t index = 0; index < length; index++) {
            powered_sum += pow(
                norm_ext_absolute(
                    a, a->offset + index * a->strides[0]), ord);
        }
        return pow(powered_sum, 1.0 / ord);
    }

    int64_t rows = a->shape[0];
    int64_t columns = a->shape[1];
    if (ord == 1.0 || ord == -1.0) {
        if (columns == 0) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "column-sum matrix norm has no identity for zero columns");
            return NAN;
        }
        double selected = 0.0;
        for (int64_t column = 0; column < columns; column++) {
            double sum = 0.0;
            for (int64_t row = 0; row < rows; row++) {
                sum += norm_ext_absolute(
                    a, a->offset + row * a->strides[0] +
                        column * a->strides[1]);
            }
            if (isnan(sum)) return NAN;
            if (column == 0 || (ord > 0.0 && sum > selected) ||
                    (ord < 0.0 && sum < selected)) {
                selected = sum;
            }
        }
        return selected;
    }
    if (isinf(ord)) {
        if (rows == 0) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "row-sum matrix norm has no identity for zero rows");
            return NAN;
        }
        double selected = 0.0;
        for (int64_t row = 0; row < rows; row++) {
            double sum = 0.0;
            for (int64_t column = 0; column < columns; column++) {
                sum += norm_ext_absolute(
                    a, a->offset + row * a->strides[0] +
                        column * a->strides[1]);
            }
            if (isnan(sum)) return NAN;
            if (row == 0 || (ord > 0.0 && sum > selected) ||
                    (ord < 0.0 && sum < selected)) {
                selected = sum;
            }
        }
        return selected;
    }
    if (ord == 2.0 || ord == -2.0) {
        CnpArray *singular_values = NULL;
        if (rows == 0 || columns == 0) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "spectral matrix norm has no singular values");
            return NAN;
        }
        CNP_STATUS status = cnp_linalg_svd_v2(
            a, false, false, false,
            NULL, &singular_values, NULL);
        if (status != CNP_OK || !singular_values) {
            if (singular_values) cnp_array_free(singular_values);
            cnp_relabel_error(function_name);
            return NAN;
        }
        int64_t index = ord > 0.0 ? 0 : singular_values->size - 1;
        double value = cnp_get_element_double(
            singular_values->data,
            singular_values->offset + index * singular_values->dtype->elsize,
            singular_values->dtype->type_num);
        cnp_array_free(singular_values);
        return value;
    }

    cnp_set_error(
        CNP_ERR_VALUE, function_name,
        "order %.17g is not a represented matrix norm", ord);
    return NAN;
}

/* =========================================================================
 * cnp_linalg_cond - Condition number
 * ========================================================================= */
CNP_API double CNP_CALL cnp_linalg_cond(const CnpArray *a) {
    const char *function_name = "cnp_linalg_cond";
    CnpArray *condition;
    double value;
    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array must not be null");
        return NAN;
    }
    if (a->ndim != 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "legacy scalar result requires a two-dimensional input");
        return NAN;
    }
    condition = cnp_linalg_cond_v2(a);
    if (!condition) {
        cnp_relabel_error(function_name);
        return NAN;
    }
    value = cnp_get_element_double(
        condition->data, condition->offset,
        condition->dtype->type_num);
    cnp_array_free(condition);
    return value;
}

/* =========================================================================
 * cnp_linalg_tensorinv - Inverse of N-dimensional array
 * ========================================================================= */
static bool tensor_product_dimensions(
        const int64_t *shape, int begin, int end,
        int64_t *product, const char *function_name) {
    int64_t value = 1;
    for (int dimension = begin; dimension < end; dimension++) {
        int64_t length = shape[dimension];
        if (length != 0 && value > INT64_MAX / length) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "tensor dimension product overflows int64");
            return false;
        }
        value *= length;
    }
    *product = value;
    return true;
}

CNP_API CnpArray* CNP_CALL cnp_linalg_tensorinv(
        const CnpArray *a, int ind) {
    const char *function_name = "cnp_linalg_tensorinv";
    CnpArray *flat = NULL;
    CnpArray *matrix = NULL;
    CnpArray *inverse = NULL;
    CnpArray *result = NULL;
    int64_t matrix_shape[2];
    int64_t result_shape[CNP_MAXDIMS];
    int effective_ind;
    int64_t leading_size;
    int64_t trailing_size;

    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array must not be null");
        return NULL;
    }
    if (a->ndim == 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "tensor inverse requires at least one dimension");
        return NULL;
    }
    if (ind <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "ind must be greater than zero");
        return NULL;
    }
    effective_ind = ind < a->ndim ? ind : a->ndim;
    if (!tensor_product_dimensions(
            a->shape, 0, effective_ind,
            &leading_size, function_name) ||
            !tensor_product_dimensions(
                a->shape, effective_ind, a->ndim,
                &trailing_size, function_name)) {
        return NULL;
    }
    if (leading_size != trailing_size) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "reshaped tensor must be square, got %lld by %lld",
            (long long)leading_size, (long long)trailing_size);
        return NULL;
    }
    matrix_shape[0] = leading_size;
    matrix_shape[1] = trailing_size;
    int destination = 0;
    for (int dimension = effective_ind;
            dimension < a->ndim; dimension++) {
        result_shape[destination++] = a->shape[dimension];
    }
    for (int dimension = 0; dimension < effective_ind; dimension++) {
        result_shape[destination++] = a->shape[dimension];
    }

    flat = cnp_flatten(a, CNP_ORDER_C);
    if (!flat) goto failure;
    matrix = cnp_reshape(flat, 2, matrix_shape, CNP_ORDER_C);
    if (!matrix) goto failure;
    cnp_array_decref(flat);
    flat = NULL;
    inverse = cnp_linalg_inv(matrix);
    if (!inverse) goto failure;
    cnp_array_free(matrix);
    matrix = NULL;
    result = cnp_reshape(
        inverse, a->ndim, result_shape, CNP_ORDER_C);
    if (!result) goto failure;
    cnp_array_decref(inverse);
    return result;

failure:
    if (inverse) cnp_array_free(inverse);
    if (matrix) cnp_array_free(matrix);
    if (flat) cnp_array_free(flat);
    cnp_relabel_error(function_name);
    return NULL;
}

/* =========================================================================
 * cnp_linalg_tensorsolve - Solve tensor equation a x = b
 * ========================================================================= */
static CnpArray *linalg_tensorsolve_core(
        const CnpArray *a, const CnpArray *b,
        const char *function_name) {
    CnpArray *flat_a = NULL;
    CnpArray *matrix = NULL;
    CnpArray *flat_b = NULL;
    CnpArray *solution = NULL;
    CnpArray *result = NULL;
    int64_t matrix_shape[2];
    int64_t result_shape[CNP_MAXDIMS];
    int solution_ndim;
    int64_t solution_size;

    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both operands must not be null");
        return NULL;
    }
    if (a->ndim < b->ndim) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "coefficient tensor rank must not be smaller than rhs rank");
        return NULL;
    }
    solution_ndim = a->ndim - b->ndim;
    if (!tensor_product_dimensions(
            a->shape, b->ndim, a->ndim,
            &solution_size, function_name)) {
        return NULL;
    }
    if (b->size != solution_size ||
            (solution_size != 0 && a->size / solution_size != solution_size) ||
            (solution_size == 0 && a->size != 0)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "tensor equation does not reshape to a %lld by %lld system",
            (long long)b->size, (long long)solution_size);
        return NULL;
    }
    for (int dimension = 0; dimension < solution_ndim; dimension++) {
        result_shape[dimension] = a->shape[b->ndim + dimension];
    }
    matrix_shape[0] = b->size;
    matrix_shape[1] = solution_size;

    flat_a = cnp_flatten(a, CNP_ORDER_C);
    if (!flat_a) goto failure;
    matrix = cnp_reshape(flat_a, 2, matrix_shape, CNP_ORDER_C);
    if (!matrix) goto failure;
    cnp_array_decref(flat_a);
    flat_a = NULL;
    flat_b = cnp_flatten(b, CNP_ORDER_C);
    if (!flat_b) goto failure;
    if (cnp_linalg_solve(matrix, flat_b, &solution) != CNP_OK || !solution) {
        if (solution) {
            cnp_array_free(solution);
            solution = NULL;
        }
        goto failure;
    }
    cnp_array_free(flat_b);
    flat_b = NULL;
    cnp_array_free(matrix);
    matrix = NULL;
    result = cnp_reshape(
        solution, solution_ndim,
        solution_ndim ? result_shape : NULL, CNP_ORDER_C);
    if (!result) goto failure;
    cnp_array_decref(solution);
    return result;

failure:
    if (solution) cnp_array_free(solution);
    if (flat_b) cnp_array_free(flat_b);
    if (matrix) cnp_array_free(matrix);
    if (flat_a) cnp_array_free(flat_a);
    cnp_relabel_error(function_name);
    return NULL;
}

CNP_API CnpArray* CNP_CALL cnp_linalg_tensorsolve(
        const CnpArray *a, const CnpArray *b, int *axes) {
    const char *function_name = "cnp_linalg_tensorsolve";
    if (!a || !b)
        return linalg_tensorsolve_core(a, b, function_name);
    if (axes) {
        cnp_set_error(
            CNP_ERR_NOT_IMPLEMENTED, function_name,
            "non-null axes cannot be represented without an axes length");
        return NULL;
    }
    return linalg_tensorsolve_core(a, b, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_linalg_tensorsolve_v2(
        const CnpArray *a, const CnpArray *b,
        int naxes, const int *axes) {
    const char *function_name = "cnp_linalg_tensorsolve_v2";
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both operands must not be null");
        return NULL;
    }
    if (naxes < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "axes length must not be negative");
        return NULL;
    }
    if (naxes > 0 && !axes) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "axes must not be null when axes length is positive");
        return NULL;
    }
    for (int index = 0; index < naxes; ++index) {
        if (axes[index] < 0 || axes[index] >= a->ndim) {
            cnp_set_error(
                CNP_ERR_AXIS, function_name,
                "axis %d is out of bounds for array of dimension %d",
                axes[index], a->ndim);
            return NULL;
        }
    }
    if (naxes == 0)
        return linalg_tensorsolve_core(a, b, function_name);

    int permutation[CNP_MAXDIMS];
    for (int dimension = 0; dimension < a->ndim; ++dimension)
        permutation[dimension] = dimension;
    for (int index = 0; index < naxes; ++index) {
        int axis = axes[index];
        int position = 0;
        while (permutation[position] != axis) ++position;
        for (int dimension = position;
             dimension + 1 < a->ndim; ++dimension)
            permutation[dimension] = permutation[dimension + 1];
        permutation[a->ndim - 1] = axis;
    }

    CnpArray *transposed = cnp_transpose(a, permutation);
    if (!transposed) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpArray *result = linalg_tensorsolve_core(
        transposed, b, function_name);
    cnp_array_free(transposed);
    return result;
}

/* =========================================================================
 * cnp_kron - Kronecker product
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_kron(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_kron";
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both operands must not be null");
        return NULL;
    }
    CNP_TYPE output_type = cnp_promote_type_full(
        a->dtype->type_num, b->dtype->type_num);
    if (output_type == CNP_NOTYPE ||
            !(output_type == CNP_BOOL ||
              cnp_type_is_integer(output_type) ||
              cnp_type_is_float(output_type) ||
              cnp_type_is_complex(output_type))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "kron requires represented numeric dtypes");
        return NULL;
    }

    int output_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    int64_t output_shape[CNP_MAXDIMS];
    int left_padding = output_ndim - a->ndim;
    int right_padding = output_ndim - b->ndim;
    for (int axis = 0; axis < output_ndim; ++axis) {
        int64_t left_length = axis < left_padding
            ? 1 : a->shape[axis - left_padding];
        int64_t right_length = axis < right_padding
            ? 1 : b->shape[axis - right_padding];
        if (left_length != 0 && right_length > INT64_MAX / left_length) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "output dimension %d overflows int64", axis);
            return NULL;
        }
        output_shape[axis] = left_length * right_length;
    }

    CnpArray *result = cnp_array_new(
        output_ndim, output_ndim ? output_shape : NULL,
        output_type, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t remaining = index;
        int64_t left_offset = a->offset;
        int64_t right_offset = b->offset;
        for (int axis = output_ndim - 1; axis >= 0; --axis) {
            int64_t coordinate = remaining % output_shape[axis];
            remaining /= output_shape[axis];
            int64_t right_length = axis < right_padding
                ? 1 : b->shape[axis - right_padding];
            if (axis >= left_padding) {
                left_offset += (coordinate / right_length) *
                    a->strides[axis - left_padding];
            }
            if (axis >= right_padding) {
                right_offset += (coordinate % right_length) *
                    b->strides[axis - right_padding];
            }
        }
        CNP_STATUS status = cnp_multiply_scalar_values(
            (const char*)a->data + left_offset,
            a->dtype->type_num,
            (const char*)b->data + right_offset,
            b->dtype->type_num,
            (char*)result->data + index * result->dtype->elsize,
            output_type, function_name);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
    }
    return result;
}
