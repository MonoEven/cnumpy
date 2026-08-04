/**
 * cnumpy linear algebra - dot, matmul, inv, det, solve, eig, svd, qr, etc.
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <float.h>

/* =========================================================================
 * Dot product
 * ========================================================================= */
static char product_label_character(int label) {
    return label < 26 ? (char)('a' + label) : (char)('A' + label - 26);
}

static CnpArray *dot_generic(
        const CnpArray *a, const CnpArray *b,
        const char *function_name) {
    char expression[4 * CNP_MAXDIMS + 8];
    char left_labels[CNP_MAXDIMS];
    char right_labels[CNP_MAXDIMS];
    char output_labels[CNP_MAXDIMS];
    int left_count = 0;
    int right_count = 0;
    int output_count = 0;
    int next_label = 0;

    if (a->ndim == 0 || b->ndim == 0) {
        const CnpArray *nonscalar = a->ndim == 0 ? b : a;
        for (int axis = 0; axis < nonscalar->ndim; ++axis) {
            char label = product_label_character(next_label++);
            output_labels[output_count++] = label;
            if (a->ndim == 0) right_labels[right_count++] = label;
            else left_labels[left_count++] = label;
        }
    } else {
        int output_ndim = a->ndim - 1 + (b->ndim == 1 ? 0 : b->ndim - 1);
        if (output_ndim > CNP_MAXDIMS) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "dot result rank %d exceeds CNP_MAXDIMS", output_ndim);
            return NULL;
        }
        int64_t left_contract = a->shape[a->ndim - 1];
        int64_t right_contract = b->shape[b->ndim == 1 ? 0 : b->ndim - 2];
        if (left_contract != right_contract) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "contracted dimensions %lld and %lld do not match",
                (long long)left_contract, (long long)right_contract);
            return NULL;
        }
        for (int axis = 0; axis < a->ndim - 1; ++axis) {
            char label = product_label_character(next_label++);
            left_labels[left_count++] = label;
            output_labels[output_count++] = label;
        }
        char contraction = product_label_character(next_label++);
        left_labels[left_count++] = contraction;
        if (b->ndim == 1) {
            right_labels[right_count++] = contraction;
        } else {
            for (int axis = 0; axis < b->ndim - 2; ++axis) {
                char label = product_label_character(next_label++);
                right_labels[right_count++] = label;
                output_labels[output_count++] = label;
            }
            right_labels[right_count++] = contraction;
            char last = product_label_character(next_label++);
            right_labels[right_count++] = last;
            output_labels[output_count++] = last;
        }
    }

    int cursor = 0;
    for (int index = 0; index < left_count; ++index)
        expression[cursor++] = left_labels[index];
    expression[cursor++] = ',';
    for (int index = 0; index < right_count; ++index)
        expression[cursor++] = right_labels[index];
    expression[cursor++] = '-';
    expression[cursor++] = '>';
    for (int index = 0; index < output_count; ++index)
        expression[cursor++] = output_labels[index];
    expression[cursor] = '\0';

    const CnpArray *operands[2] = {a, b};
    return cnp_einsum_generic(
        expression, 2, operands, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_dot(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_dot";
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both operands must not be null");
        return NULL;
    }
    if (a->ndim == 1 && b->ndim == 1 &&
            a->shape[0] == b->shape[0] &&
            (a->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            (b->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            a->dtype->type_num == CNP_DOUBLE &&
            b->dtype->type_num == CNP_DOUBLE) {
        const double *left = (const double*)((const char*)a->data + a->offset);
        const double *right = (const double*)((const char*)b->data + b->offset);
        return cnp_array_from_scalar(
            cnp_simd_dot(left, right, a->shape[0]), CNP_DOUBLE);
    }
    if (a->ndim == 2 && b->ndim == 2) {
        CnpArray *result = cnp_matmul(a, b);
        if (!result) cnp_relabel_error(function_name);
        return result;
    }
    return dot_generic(a, b, function_name);
}

/* =========================================================================
 * Matrix multiplication
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_matmul(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_matmul";
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both operands must not be null");
        return NULL;
    }
    if (a->ndim == 0 || b->ndim == 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "matmul operands must each have at least one dimension");
        return NULL;
    }
    int64_t left_contract = a->shape[a->ndim - 1];
    int64_t right_contract = b->shape[b->ndim == 1 ? 0 : b->ndim - 2];
    if (left_contract != right_contract) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "contracted dimensions %lld and %lld do not match",
            (long long)left_contract, (long long)right_contract);
        return NULL;
    }

    if (a->ndim == 2 && b->ndim == 2 &&
            (a->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            (b->flags & CNP_ARRAY_C_CONTIGUOUS) &&
            a->dtype->type_num == CNP_DOUBLE &&
            b->dtype->type_num == CNP_DOUBLE) {
        int64_t shape[2] = {a->shape[0], b->shape[1]};
        CnpArray *result = cnp_array_zeros(
            2, shape, CNP_DOUBLE, CNP_ORDER_C);
        if (!result) {
            cnp_relabel_error(function_name);
            return NULL;
        }
        CNP_STATUS status = cnp_gemm_f64(
            (const double*)((const char*)a->data + a->offset),
            (const double*)((const char*)b->data + b->offset),
            (double*)result->data,
            a->shape[0], b->shape[1], left_contract);
        if (status != CNP_OK) {
            cnp_array_free(result);
            cnp_relabel_error(function_name);
            return NULL;
        }
        return result;
    }

    const CnpArray *operands[2] = {a, b};
    const char *expression;
    if (a->ndim == 1 && b->ndim == 1)
        expression = "i,i->";
    else if (a->ndim == 1)
        expression = "k,...kj->...j";
    else if (b->ndim == 1)
        expression = "...ik,k->...i";
    else
        expression = "...ik,...kj->...ij";
    return cnp_einsum_generic(
        expression, 2, operands, function_name);
}

/* =========================================================================
 * Inner and Outer products
 * ========================================================================= */
static CnpArray *tensordot_counts(
        const CnpArray *a, const CnpArray *b, int axes,
        const char *function_name) {
    if (axes < 0 || axes > a->ndim || axes > b->ndim) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "contracted axis count %d is invalid for ranks %d and %d",
            axes, a->ndim, b->ndim);
        return NULL;
    }
    int output_ndim = a->ndim + b->ndim - 2 * axes;
    if (output_ndim > CNP_MAXDIMS) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "tensordot result rank %d exceeds CNP_MAXDIMS", output_ndim);
        return NULL;
    }
    for (int index = 0; index < axes; ++index) {
        int64_t left_length = a->shape[a->ndim - axes + index];
        int64_t right_length = b->shape[index];
        if (left_length != right_length) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "contracted dimension %d has lengths %lld and %lld",
                index, (long long)left_length, (long long)right_length);
            return NULL;
        }
    }

    char expression[4 * CNP_MAXDIMS + 8];
    char left_labels[CNP_MAXDIMS];
    char right_labels[CNP_MAXDIMS];
    char output_labels[CNP_MAXDIMS];
    char contraction_labels[CNP_MAXDIMS];
    int left_count = 0;
    int right_count = 0;
    int output_count = 0;
    int next_label = 0;
    for (int axis = 0; axis < a->ndim - axes; ++axis) {
        char label = product_label_character(next_label++);
        left_labels[left_count++] = label;
        output_labels[output_count++] = label;
    }
    for (int axis = 0; axis < axes; ++axis) {
        char label = product_label_character(next_label++);
        contraction_labels[axis] = label;
        left_labels[left_count++] = label;
    }
    for (int axis = 0; axis < axes; ++axis)
        right_labels[right_count++] = contraction_labels[axis];
    for (int axis = axes; axis < b->ndim; ++axis) {
        char label = product_label_character(next_label++);
        right_labels[right_count++] = label;
        output_labels[output_count++] = label;
    }

    int cursor = 0;
    for (int index = 0; index < left_count; ++index)
        expression[cursor++] = left_labels[index];
    expression[cursor++] = ',';
    for (int index = 0; index < right_count; ++index)
        expression[cursor++] = right_labels[index];
    expression[cursor++] = '-';
    expression[cursor++] = '>';
    for (int index = 0; index < output_count; ++index)
        expression[cursor++] = output_labels[index];
    expression[cursor] = '\0';
    const CnpArray *operands[2] = {a, b};
    return cnp_einsum_generic(
        expression, 2, operands, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_inner(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_inner";
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both operands must not be null");
        return NULL;
    }
    if (a->ndim == 0 || b->ndim == 0)
        return dot_generic(a, b, function_name);
    CnpArray *right_oriented = cnp_moveaxis(b, b->ndim - 1, 0);
    if (!right_oriented) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpArray *result = tensordot_counts(
        a, right_oriented, 1, function_name);
    cnp_array_free(right_oriented);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_outer(const CnpArray *a, const CnpArray *b) {
    const char *function_name = "cnp_outer";
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both operands must not be null");
        return NULL;
    }
    CnpArray *fa = cnp_flatten(a, CNP_ORDER_C);
    if (!fa) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpArray *fb = cnp_flatten(b, CNP_ORDER_C);
    if (!fb) {
        cnp_array_free(fa);
        cnp_relabel_error(function_name);
        return NULL;
    }
    const CnpArray *operands[2] = {fa, fb};
    CnpArray *result = cnp_einsum_generic(
        "i,j->ij", 2, operands, function_name);
    cnp_array_free(fa);
    cnp_array_free(fb);
    return result;
}

static CnpArray *cross_pad_last_axis(
        const CnpArray *source, const char *function_name) {
    int64_t shape[CNP_MAXDIMS];
    for (int axis = 0; axis < source->ndim; ++axis)
        shape[axis] = source->shape[axis];
    shape[source->ndim - 1] = 3;
    CnpArray *result = cnp_array_zeros(
        source->ndim, shape, source->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    int64_t coordinates[CNP_MAXDIMS] = {0};
    int itemsize = source->dtype->elsize;
    for (int64_t index = 0; index < source->size; ++index) {
        int64_t source_offset = source->offset +
            cnp_multi_to_offset(
                source->ndim, coordinates, source->strides);
        int64_t component = coordinates[source->ndim - 1];
        int64_t batch_index = index / 2;
        memcpy(
            (char*)result->data + (batch_index * 3 + component) * itemsize,
            (const char*)source->data + source_offset,
            (size_t)itemsize);
        for (int axis = source->ndim - 1; axis >= 0; --axis) {
            coordinates[axis]++;
            if (coordinates[axis] < source->shape[axis]) break;
            coordinates[axis] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_cross(const CnpArray *a, const CnpArray *b, int axis) {
    const char *function_name = "cnp_cross";
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both operands must not be null");
        return NULL;
    }
    int left_axis = cnp_normalize_axis(axis, a->ndim);
    int right_axis = cnp_normalize_axis(axis, b->ndim);
    if (left_axis < 0 || right_axis < 0) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "axis %d is invalid for ranks %d and %d",
            axis, a->ndim, b->ndim);
        return NULL;
    }
    int64_t left_length = a->shape[left_axis];
    int64_t right_length = b->shape[right_axis];
    if (!((left_length == 2 || left_length == 3) &&
          (right_length == 2 || right_length == 3))) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "vector dimensions must each have length 2 or 3");
        return NULL;
    }
    CNP_TYPE output_type = cnp_promote_type_full(
        a->dtype->type_num, b->dtype->type_num);
    if (output_type == CNP_NOTYPE || output_type == CNP_BOOL ||
            !(cnp_type_is_integer(output_type) ||
              cnp_type_is_float(output_type) ||
              cnp_type_is_complex(output_type))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "cross requires represented non-boolean numeric dtypes");
        return NULL;
    }

    CnpArray *left_moved = cnp_moveaxis(a, left_axis, a->ndim - 1);
    CnpArray *right_moved = cnp_moveaxis(b, right_axis, b->ndim - 1);
    CnpArray *left_padded = NULL;
    CnpArray *right_padded = NULL;
    CnpArray *epsilon = NULL;
    CnpArray *result = NULL;
    if (!left_moved || !right_moved) goto cleanup;

    if (left_length == 2) {
        left_padded = cross_pad_last_axis(left_moved, function_name);
        if (!left_padded) goto cleanup;
    }
    if (right_length == 2) {
        right_padded = cross_pad_last_axis(right_moved, function_name);
        if (!right_padded) goto cleanup;
    }
    const CnpArray *left_operand = left_padded ? left_padded : left_moved;
    const CnpArray *right_operand = right_padded ? right_padded : right_moved;
    if (left_length == 2 && right_length == 2) {
        int64_t epsilon_shape[2] = {3, 3};
        epsilon = cnp_array_zeros(
            2, epsilon_shape, output_type, CNP_ORDER_C);
        if (!epsilon) goto cleanup;
        cnp_set_element_double(
            epsilon->data, 1 * epsilon->dtype->elsize, output_type, 1.0);
        cnp_set_element_double(
            epsilon->data, 3 * epsilon->dtype->elsize, output_type, -1.0);
        const CnpArray *scalar_operands[3] = {
            left_operand, right_operand, epsilon};
        result = cnp_einsum_generic(
            "...i,...j,ij->...", 3, scalar_operands, function_name);
    } else {
        int64_t epsilon_shape[3] = {3, 3, 3};
        epsilon = cnp_array_zeros(
            3, epsilon_shape, output_type, CNP_ORDER_C);
        if (!epsilon) goto cleanup;
        int positive[3] = {5, 15, 19};
        int negative[3] = {7, 11, 21};
        for (int index = 0; index < 3; ++index) {
            cnp_set_element_double(
                epsilon->data,
                (int64_t)positive[index] * epsilon->dtype->elsize,
                output_type, 1.0);
            cnp_set_element_double(
                epsilon->data,
                (int64_t)negative[index] * epsilon->dtype->elsize,
                output_type, -1.0);
        }
        const CnpArray *operands[3] = {
            left_operand, right_operand, epsilon};
        result = cnp_einsum_generic(
            "...i,...j,kij->...k", 3, operands, function_name);
        if (result && axis != -1) {
            int destination = cnp_normalize_axis(axis, result->ndim);
            if (destination < 0) {
                cnp_array_free(result);
                result = NULL;
                cnp_set_error(
                    CNP_ERR_AXIS, function_name,
                    "output axis %d is invalid for result rank", axis);
                goto cleanup;
            }
            if (destination != result->ndim - 1) {
                CnpArray *moved = cnp_moveaxis(
                    result, result->ndim - 1, destination);
                cnp_array_free(result);
                result = moved;
                if (!result) cnp_relabel_error(function_name);
            }
        }
    }

cleanup:
    if (epsilon) cnp_array_free(epsilon);
    if (right_padded) cnp_array_free(right_padded);
    if (left_padded) cnp_array_free(left_padded);
    if (right_moved) cnp_array_free(right_moved);
    if (left_moved) cnp_array_free(left_moved);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_tensordot(const CnpArray *a, const CnpArray *b, int axes_a, int axes_b) {
    const char *function_name = "cnp_tensordot";
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both operands must not be null");
        return NULL;
    }
    if (axes_a != axes_b) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "legacy axis-count arguments must be equal");
        return NULL;
    }
    return tensordot_counts(a, b, axes_a, function_name);
}

/* =========================================================================
 * LU decomposition helper (internal)
 * ========================================================================= */
static CNP_STATUS lu_decompose(double *A, int n, int *pivot) {
    for (int i = 0; i < n; i++) pivot[i] = i;

    for (int k = 0; k < n; k++) {
        /* Find pivot */
        double max_val = fabs(A[k*n + k]);
        int max_idx = k;
        for (int i = k + 1; i < n; i++) {
            if (fabs(A[i*n + k]) > max_val) {
                max_val = fabs(A[i*n + k]);
                max_idx = i;
            }
        }
        if (max_val < 1e-15) return CNP_ERR_SINGULAR;

        /* Swap rows */
        if (max_idx != k) {
            int tmp = pivot[k]; pivot[k] = pivot[max_idx]; pivot[max_idx] = tmp;
            for (int j = 0; j < n; j++) {
                double t = A[k*n+j]; A[k*n+j] = A[max_idx*n+j]; A[max_idx*n+j] = t;
            }
        }

        /* Eliminate */
        for (int i = k + 1; i < n; i++) {
            A[i*n + k] /= A[k*n + k];
            for (int j = k + 1; j < n; j++) {
                A[i*n + j] -= A[i*n + k] * A[k*n + j];
            }
        }
    }
    return CNP_OK;
}

/* =========================================================================
 * Matrix inverse
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_linalg_inv(const CnpArray *a) {
    const char *function_name = "cnp_linalg_inv";
    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array must not be null");
        return NULL;
    }
    if (a->ndim < 2 || a->shape[a->ndim - 2] != a->shape[a->ndim - 1]) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input must contain square matrices in its final two axes");
        return NULL;
    }
    CNP_TYPE input_type = a->dtype->type_num;
    CNP_TYPE identity_type;
    if (input_type == CNP_FLOAT || input_type == CNP_DOUBLE ||
            input_type == CNP_CFLOAT || input_type == CNP_CDOUBLE) {
        identity_type = input_type;
    } else if (input_type == CNP_BOOL || cnp_type_is_integer(input_type)) {
        identity_type = CNP_DOUBLE;
    } else {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input dtype is not supported by the linear solve engine");
        return NULL;
    }

    CnpArray *identity = cnp_array_zeros(
        a->ndim, a->shape, identity_type, CNP_ORDER_C);
    if (!identity) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    int64_t matrix_size = a->shape[a->ndim - 1];
    int64_t batch_count = 1;
    for (int axis = 0; axis < a->ndim - 2; ++axis)
        batch_count *= a->shape[axis];
    for (int64_t batch = 0; batch < batch_count; ++batch) {
        for (int64_t diagonal = 0; diagonal < matrix_size; ++diagonal) {
            int64_t index =
                (batch * matrix_size + diagonal) * matrix_size + diagonal;
            cnp_set_element_double(
                identity->data,
                index * identity->dtype->elsize,
                identity_type, 1.0);
        }
    }

    CnpArray *result = NULL;
    CNP_STATUS status = cnp_linalg_solve(a, identity, &result);
    cnp_array_free(identity);
    if (status != CNP_OK || !result) {
        if (result) cnp_array_free(result);
        cnp_relabel_error(function_name);
        return NULL;
    }
    return result;
}

/* =========================================================================
 * Determinant
 * ========================================================================= */
#pragma float_control(precise, on, push)
typedef struct {
    double real;
    double imag;
} CnpDetComplex;

static CnpDetComplex det_complex(double real, double imag) {
    CnpDetComplex value = {real, imag};
    return value;
}

static CnpDetComplex det_add(
        CnpDetComplex left, CnpDetComplex right) {
    return det_complex(left.real + right.real, left.imag + right.imag);
}

static CnpDetComplex det_multiply(
        CnpDetComplex left, CnpDetComplex right) {
    return det_complex(
        left.real * right.real - left.imag * right.imag,
        left.real * right.imag + left.imag * right.real);
}

static double det_nan(void);

static CnpDetComplex det_divide(
        CnpDetComplex numerator, CnpDetComplex denominator) {
    double ratio;
    double scale;
    if (denominator.real == 0.0 && denominator.imag == 0.0) {
        double nan_value = det_nan();
        return det_complex(nan_value, nan_value);
    }
    if (denominator.imag == 0.0) {
        return det_complex(
            numerator.real / denominator.real,
            numerator.imag / denominator.real);
    }
    if (denominator.real == 0.0) {
        return det_complex(
            numerator.imag / denominator.imag,
            -numerator.real / denominator.imag);
    }
    if (fabs(denominator.real) >= fabs(denominator.imag)) {
        ratio = denominator.imag / denominator.real;
        scale = denominator.real + denominator.imag * ratio;
        return det_complex(
            (numerator.real + numerator.imag * ratio) / scale,
            (numerator.imag - numerator.real * ratio) / scale);
    }
    ratio = denominator.real / denominator.imag;
    scale = denominator.imag + denominator.real * ratio;
    return det_complex(
        (numerator.real * ratio + numerator.imag) / scale,
        (numerator.imag * ratio - numerator.real) / scale);
}

static double det_absolute(CnpDetComplex value) {
    return hypot(value.real, value.imag);
}

static double det_absolute_one(CnpDetComplex value) {
    return fabs(value.real) + fabs(value.imag);
}

static double det_negative_infinity(void) {
    uint64_t bits = UINT64_C(0xfff0000000000000);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static double det_nan(void) {
    uint64_t bits = UINT64_C(0x7ff8000000000000);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool det_is_nan(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) ==
            UINT64_C(0x7ff0000000000000) &&
        (bits & UINT64_C(0x000fffffffffffff)) != 0;
}

static bool det_type_is_supported(CNP_TYPE type) {
    return type == CNP_BOOL || cnp_type_is_integer(type) ||
        type == CNP_FLOAT || type == CNP_DOUBLE ||
        type == CNP_CFLOAT || type == CNP_CDOUBLE;
}

static CNP_TYPE det_result_type(CNP_TYPE input_type) {
    if (input_type == CNP_FLOAT) return CNP_FLOAT;
    if (input_type == CNP_CFLOAT) return CNP_CFLOAT;
    if (input_type == CNP_CDOUBLE) return CNP_CDOUBLE;
    return CNP_DOUBLE;
}

static CnpDetComplex det_read(
        const CnpArray *array, int64_t batch_offset,
        int64_t row, int64_t column) {
    int64_t offset = batch_offset +
        row * array->strides[array->ndim - 2] +
        column * array->strides[array->ndim - 1];
    const char *pointer = (const char*)array->data + offset;
    if (array->dtype->type_num == CNP_CFLOAT) {
        const cnp_cfloat *value = (const cnp_cfloat*)pointer;
        return det_complex(value->real, value->imag);
    }
    if (array->dtype->type_num == CNP_CDOUBLE) {
        const cnp_cdouble *value = (const cnp_cdouble*)pointer;
        return det_complex(value->real, value->imag);
    }
    return det_complex(
        cnp_get_element_double(
            array->data, offset, array->dtype->type_num),
        0.0);
}

static double det_read_real(
        const CnpArray *array, int64_t batch_offset,
        int64_t row, int64_t column) {
    int64_t offset = batch_offset +
        row * array->strides[array->ndim - 2] +
        column * array->strides[array->ndim - 1];
    return cnp_get_element_double(
        array->data, offset, array->dtype->type_num);
}

static int64_t det_batch_offset(
        const CnpArray *array, int64_t batch_index) {
    int64_t offset = array->offset;
    for (int dimension = array->ndim - 3; dimension >= 0; dimension--) {
        int64_t coordinate = batch_index % array->shape[dimension];
        batch_index /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

static bool det_factor_matrix(
        CnpDetComplex *matrix, int64_t n, int *row_sign) {
    int sign = 1;
    for (int64_t pivot_column = 0;
            pivot_column < n; pivot_column++) {
        int64_t pivot_row = pivot_column;
        double pivot_magnitude = det_absolute_one(
            matrix[pivot_column * n + pivot_column]);
        for (int64_t row = pivot_column + 1; row < n; row++) {
            double magnitude = det_absolute_one(
                matrix[row * n + pivot_column]);
            if (magnitude > pivot_magnitude) {
                pivot_magnitude = magnitude;
                pivot_row = row;
            }
        }
        if (pivot_row != pivot_column) {
            sign = -sign;
            for (int64_t column = 0; column < n; column++) {
                CnpDetComplex temporary =
                    matrix[pivot_column * n + column];
                matrix[pivot_column * n + column] =
                    matrix[pivot_row * n + column];
                matrix[pivot_row * n + column] = temporary;
            }
        }
        CnpDetComplex pivot =
            matrix[pivot_column * n + pivot_column];
        if (pivot.real == 0.0 && pivot.imag == 0.0) {
            *row_sign = sign;
            return false;
        }
        if (!det_is_nan(pivot_magnitude)) {
            if (pivot_magnitude >= DBL_MIN) {
                CnpDetComplex reciprocal = det_divide(
                    det_complex(1.0, 0.0), pivot);
                if (reciprocal.real == 0.0 &&
                        reciprocal.imag == 0.0) {
                    for (int64_t row = pivot_column + 1;
                            row < n; row++) {
                        matrix[row * n + pivot_column] =
                            det_complex(0.0, 0.0);
                    }
                } else {
                    for (int64_t row = pivot_column + 1;
                            row < n; row++) {
                        matrix[row * n + pivot_column] = det_multiply(
                            reciprocal,
                            matrix[row * n + pivot_column]);
                    }
                }
            }
        }
        for (int64_t column = pivot_column + 1;
                column < n; column++) {
            CnpDetComplex upper =
                matrix[pivot_column * n + column];
            CnpDetComplex temporary = det_multiply(
                det_complex(-1.0, 0.0), upper);
            for (int64_t row = pivot_column + 1;
                    row < n; row++) {
                matrix[row * n + column] = det_add(
                    matrix[row * n + column],
                    det_multiply(
                        matrix[row * n + pivot_column],
                        temporary));
            }
        }
    }
    *row_sign = sign;
    return true;
}

static bool det_factor_real_matrix(
        double *matrix, int64_t n, int *row_sign) {
    int sign = 1;
    for (int64_t pivot_column = 0;
            pivot_column < n; pivot_column++) {
        int64_t pivot_row = pivot_column;
        double pivot_magnitude = fabs(
            matrix[pivot_column * n + pivot_column]);
        for (int64_t row = pivot_column + 1; row < n; row++) {
            double magnitude = fabs(matrix[row * n + pivot_column]);
            if (magnitude > pivot_magnitude) {
                pivot_magnitude = magnitude;
                pivot_row = row;
            }
        }
        if (pivot_row != pivot_column) {
            sign = -sign;
            for (int64_t column = 0; column < n; column++) {
                double temporary = matrix[pivot_column * n + column];
                matrix[pivot_column * n + column] =
                    matrix[pivot_row * n + column];
                matrix[pivot_row * n + column] = temporary;
            }
        }
        double pivot = matrix[pivot_column * n + pivot_column];
        if (pivot == 0.0) {
            *row_sign = sign;
            return false;
        }
        if (!det_is_nan(pivot_magnitude)) {
            if (pivot_magnitude >= DBL_MIN) {
                double reciprocal = 1.0 / pivot;
                if (reciprocal == 0.0) {
                    for (int64_t row = pivot_column + 1;
                            row < n; row++) {
                        matrix[row * n + pivot_column] = 0.0;
                    }
                } else {
                    for (int64_t row = pivot_column + 1;
                            row < n; row++) {
                        matrix[row * n + pivot_column] *= reciprocal;
                    }
                }
            }
        }
        for (int64_t column = pivot_column + 1;
                column < n; column++) {
            double upper = matrix[pivot_column * n + column];
            double temporary = -upper;
            for (int64_t row = pivot_column + 1;
                    row < n; row++) {
                matrix[row * n + column] +=
                    matrix[row * n + pivot_column] * temporary;
            }
        }
    }
    *row_sign = sign;
    return true;
}

static void det_slog_from_factor(
        const CnpDetComplex *matrix, int64_t n,
        int row_sign,
        CnpDetComplex *sign, double *logabsdet) {
    *sign = det_complex((double)row_sign, 0.0);
    *logabsdet = 0.0;
    for (int64_t diagonal = 0; diagonal < n; diagonal++) {
        CnpDetComplex value = matrix[diagonal * n + diagonal];
        double magnitude = det_absolute(value);
        CnpDetComplex phase = det_complex(
            value.real / magnitude, value.imag / magnitude);
        *sign = det_multiply(*sign, phase);
        *logabsdet += log(magnitude);
    }
}

static void det_slog_from_real_factor(
        const double *matrix, int64_t n, int row_sign,
        CnpDetComplex *sign, double *logabsdet) {
    sign->real = (double)row_sign;
    sign->imag = 0.0;
    *logabsdet = 0.0;
    for (int64_t diagonal = 0; diagonal < n; diagonal++) {
        double value = matrix[diagonal * n + diagonal];
        double magnitude = value;
        if (magnitude < 0.0) {
            sign->real = -sign->real;
            magnitude = -magnitude;
        }
        *logabsdet += log(magnitude);
    }
}

static void det_write(
        CnpArray *result, int64_t index, CnpDetComplex value) {
    char *pointer = (char*)result->data +
        index * result->dtype->elsize;
    switch (result->dtype->type_num) {
        case CNP_FLOAT:
            *(float*)pointer = (float)value.real;
            break;
        case CNP_CFLOAT: {
            cnp_cfloat *destination = (cnp_cfloat*)pointer;
            destination->real = (float)value.real;
            destination->imag = (float)value.imag;
            break;
        }
        case CNP_CDOUBLE: {
            cnp_cdouble *destination = (cnp_cdouble*)pointer;
            destination->real = value.real;
            destination->imag = value.imag;
            break;
        }
        default:
            *(double*)pointer = value.real;
            break;
    }
}

CNP_API CnpArray* CNP_CALL cnp_linalg_det(const CnpArray *a) {
    if (!a || a->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, "cnp_linalg_det",
            "Input must have at least two dimensions");
        return NULL;
    }
    if (a->shape[a->ndim - 2] != a->shape[a->ndim - 1]) {
        cnp_set_error(
            CNP_ERR_SHAPE, "cnp_linalg_det",
            "Input must be square on its last two dimensions");
        return NULL;
    }
    if (!det_type_is_supported(a->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, "cnp_linalg_det",
            "Input dtype is not supported by linear algebra");
        return NULL;
    }

    int64_t n = a->shape[a->ndim - 1];
    int64_t batch_count = 1;
    int64_t matrix_size;
    bool complex_input = cnp_type_is_complex(a->dtype->type_num);
    size_t workspace_element_size = complex_input
        ? sizeof(CnpDetComplex) : sizeof(double);
    size_t workspace_bytes;
    void *workspace = NULL;
    CnpArray *result;

    for (int dimension = 0; dimension < a->ndim - 2; dimension++) {
        int64_t length = a->shape[dimension];
        if (length != 0 && batch_count > INT64_MAX / length) {
            cnp_set_error(
                CNP_ERR_SHAPE, "cnp_linalg_det",
                "Batch shape is too large");
            return NULL;
        }
        batch_count *= length;
    }

    if (n != 0 && n > INT64_MAX / n) {
        cnp_set_error(
            CNP_ERR_SHAPE, "cnp_linalg_det", "Matrix is too large");
        return NULL;
    }
    matrix_size = n * n;
    if ((uint64_t)matrix_size > SIZE_MAX / workspace_element_size) {
        cnp_set_error(
            CNP_ERR_MEMORY, "cnp_linalg_det",
            "Determinant workspace is too large");
        return NULL;
    }
    workspace_bytes = (size_t)matrix_size * workspace_element_size;
    result = cnp_array_new(
        a->ndim - 2, a->shape,
        det_result_type(a->dtype->type_num), CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error("cnp_linalg_det");
        return NULL;
    }
    if (batch_count != 0 && matrix_size != 0) {
        workspace = cnp_malloc(workspace_bytes);
        if (!workspace) {
            cnp_set_error(
                CNP_ERR_MEMORY, "cnp_linalg_det",
                "Unable to allocate determinant workspace");
            cnp_array_free(result);
            return NULL;
        }
    }

    for (int64_t batch = 0; batch < batch_count; batch++) {
        CnpDetComplex determinant = det_complex(1.0, 0.0);
        if (n != 0) {
            CnpDetComplex sign;
            double logabsdet;
            int row_sign;
            int64_t batch_offset = det_batch_offset(a, batch);
            bool factor_succeeded;
            if (complex_input) {
                CnpDetComplex *matrix = (CnpDetComplex*)workspace;
                for (int64_t row = 0; row < n; row++) {
                    for (int64_t column = 0; column < n; column++) {
                        matrix[row * n + column] = det_read(
                            a, batch_offset, row, column);
                    }
                }
                factor_succeeded = det_factor_matrix(
                    matrix, n, &row_sign);
                if (factor_succeeded) {
                    det_slog_from_factor(
                        matrix, n, row_sign, &sign, &logabsdet);
                }
            } else {
                double *matrix = (double*)workspace;
                for (int64_t row = 0; row < n; row++) {
                    for (int64_t column = 0; column < n; column++) {
                        matrix[row * n + column] = det_read_real(
                            a, batch_offset, row, column);
                    }
                }
                factor_succeeded = det_factor_real_matrix(
                    matrix, n, &row_sign);
                if (factor_succeeded) {
                    det_slog_from_real_factor(
                        matrix, n, row_sign, &sign, &logabsdet);
                }
            }
            if (!factor_succeeded) {
                sign = det_complex(0.0, 0.0);
                logabsdet = det_negative_infinity();
            }
            double magnitude = exp(logabsdet);
            if (complex_input) {
                determinant = det_multiply(
                    sign, det_complex(magnitude, 0.0));
            } else {
                determinant = det_complex(sign.real * magnitude, 0.0);
            }
        }
        det_write(result, batch, determinant);
    }
    if (workspace) cnp_free(workspace, workspace_bytes);
    return result;
}

static CNP_TYPE det_log_result_type(CNP_TYPE input_type) {
    return input_type == CNP_FLOAT || input_type == CNP_CFLOAT
        ? CNP_FLOAT : CNP_DOUBLE;
}

CNP_API CNP_STATUS CNP_CALL cnp_linalg_slogdet_v2(
        const CnpArray *a, CnpArray **sign_result,
        CnpArray **logabsdet_result) {
    const char *function_name = "cnp_linalg_slogdet_v2";
    int64_t n;
    int64_t batch_count = 1;
    int64_t matrix_size;
    bool complex_input;
    size_t workspace_element_size;
    size_t workspace_bytes;
    void *workspace = NULL;
    CnpArray *sign_array = NULL;
    CnpArray *logabsdet_array = NULL;

    if (sign_result) *sign_result = NULL;
    if (logabsdet_result) *logabsdet_result = NULL;
    if (!sign_result || !logabsdet_result ||
            sign_result == logabsdet_result) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "Distinct sign and logabsdet output pointers are required");
        return CNP_ERR_GENERIC;
    }
    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "Input array must not be null");
        return CNP_ERR_GENERIC;
    }
    if (a->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "Input must have at least two dimensions");
        return CNP_ERR_SHAPE;
    }
    if (a->shape[a->ndim - 2] != a->shape[a->ndim - 1]) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "Input must be square on its last two dimensions");
        return CNP_ERR_SHAPE;
    }
    if (!det_type_is_supported(a->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "Input dtype is not supported by linear algebra");
        return CNP_ERR_TYPE;
    }

    n = a->shape[a->ndim - 1];
    complex_input = cnp_type_is_complex(a->dtype->type_num);
    workspace_element_size = complex_input
        ? sizeof(CnpDetComplex) : sizeof(double);
    for (int dimension = 0; dimension < a->ndim - 2; dimension++) {
        int64_t length = a->shape[dimension];
        if (length != 0 && batch_count > INT64_MAX / length) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "Batch shape is too large");
            return CNP_ERR_SHAPE;
        }
        batch_count *= length;
    }
    if (n != 0 && n > INT64_MAX / n) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name, "Matrix is too large");
        return CNP_ERR_SHAPE;
    }
    matrix_size = n * n;
    if ((uint64_t)matrix_size > SIZE_MAX / workspace_element_size) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "Slogdet workspace is too large");
        return CNP_ERR_MEMORY;
    }
    workspace_bytes = (size_t)matrix_size * workspace_element_size;

    sign_array = cnp_array_new(
        a->ndim - 2, a->shape,
        det_result_type(a->dtype->type_num), CNP_ORDER_C);
    if (!sign_array) {
        cnp_relabel_error(function_name);
        return CNP_ERR_MEMORY;
    }
    logabsdet_array = cnp_array_new(
        a->ndim - 2, a->shape,
        det_log_result_type(a->dtype->type_num), CNP_ORDER_C);
    if (!logabsdet_array) {
        cnp_array_free(sign_array);
        cnp_relabel_error(function_name);
        return CNP_ERR_MEMORY;
    }
    if (batch_count != 0 && matrix_size != 0) {
        workspace = cnp_malloc(workspace_bytes);
        if (!workspace) {
            cnp_array_free(sign_array);
            cnp_array_free(logabsdet_array);
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "Unable to allocate slogdet workspace");
            return CNP_ERR_MEMORY;
        }
    }

    for (int64_t batch = 0; batch < batch_count; batch++) {
        CnpDetComplex sign = det_complex(1.0, 0.0);
        double logabsdet = 0.0;
        if (n != 0) {
            int row_sign;
            int64_t batch_offset = det_batch_offset(a, batch);
            bool factor_succeeded;
            if (complex_input) {
                CnpDetComplex *matrix = (CnpDetComplex*)workspace;
                for (int64_t row = 0; row < n; row++) {
                    for (int64_t column = 0; column < n; column++) {
                        matrix[row * n + column] = det_read(
                            a, batch_offset, row, column);
                    }
                }
                factor_succeeded = det_factor_matrix(
                    matrix, n, &row_sign);
                if (factor_succeeded) {
                    det_slog_from_factor(
                        matrix, n, row_sign, &sign, &logabsdet);
                }
            } else {
                double *matrix = (double*)workspace;
                for (int64_t row = 0; row < n; row++) {
                    for (int64_t column = 0; column < n; column++) {
                        matrix[row * n + column] = det_read_real(
                            a, batch_offset, row, column);
                    }
                }
                factor_succeeded = det_factor_real_matrix(
                    matrix, n, &row_sign);
                if (factor_succeeded) {
                    det_slog_from_real_factor(
                        matrix, n, row_sign, &sign, &logabsdet);
                }
            }
            if (!factor_succeeded) {
                sign = det_complex(0.0, 0.0);
                logabsdet = det_negative_infinity();
            }
        }
        det_write(sign_array, batch, sign);
        if (logabsdet_array->dtype->type_num == CNP_FLOAT) {
            ((float*)logabsdet_array->data)[batch] = (float)logabsdet;
        } else {
            ((double*)logabsdet_array->data)[batch] = logabsdet;
        }
    }
    if (workspace) cnp_free(workspace, workspace_bytes);
    *sign_result = sign_array;
    *logabsdet_result = logabsdet_array;
    return CNP_OK;
}

CNP_API CnpArray* CNP_CALL cnp_linalg_slogdet(const CnpArray *a) {
    const char *function_name = "cnp_linalg_slogdet";
    CnpArray *sign_array = NULL;
    CnpArray *logabsdet_array = NULL;
    CnpArray *result;
    int64_t shape[1] = {2};
    CNP_STATUS status;

    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array must not be null");
        return NULL;
    }
    if (a->ndim != 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "legacy packed result requires a two-dimensional matrix");
        return NULL;
    }
    if (cnp_type_is_complex(a->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "legacy packed result cannot preserve complex sign phase");
        return NULL;
    }
    status = cnp_linalg_slogdet_v2(
        a, &sign_array, &logabsdet_array);
    if (status != CNP_OK) {
        if (sign_array) cnp_array_free(sign_array);
        if (logabsdet_array) cnp_array_free(logabsdet_array);
        cnp_relabel_error(function_name);
        return NULL;
    }
    result = cnp_array_new(1, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!result) {
        cnp_array_free(sign_array);
        cnp_array_free(logabsdet_array);
        cnp_relabel_error(function_name);
        return NULL;
    }
    ((double*)result->data)[0] = cnp_get_element_double(
        sign_array->data, 0, sign_array->dtype->type_num);
    ((double*)result->data)[1] = cnp_get_element_double(
        logabsdet_array->data, 0, logabsdet_array->dtype->type_num);
    cnp_array_free(sign_array);
    cnp_array_free(logabsdet_array);
    return result;
}
#pragma float_control(pop)

/* =========================================================================
 * Solve linear system Ax = b
 * ========================================================================= */
typedef struct {
    double real;
    double imag;
} CnpSolveValue;

static CnpSolveValue solve_value(double real, double imag) {
    CnpSolveValue value = {real, imag};
    return value;
}

static CnpSolveValue solve_subtract(
        CnpSolveValue left, CnpSolveValue right) {
    return solve_value(left.real - right.real, left.imag - right.imag);
}

static CnpSolveValue solve_add(
        CnpSolveValue left, CnpSolveValue right) {
    return solve_value(left.real + right.real, left.imag + right.imag);
}

static CnpSolveValue solve_conjugate(CnpSolveValue value) {
    return solve_value(value.real, -value.imag);
}

static CnpSolveValue solve_multiply(
        CnpSolveValue left, CnpSolveValue right) {
    return solve_value(
        left.real * right.real - left.imag * right.imag,
        left.real * right.imag + left.imag * right.real);
}

static CnpSolveValue solve_divide(
        CnpSolveValue numerator, CnpSolveValue denominator) {
    double ratio;
    double scale;
    if (fabs(denominator.real) >= fabs(denominator.imag)) {
        ratio = denominator.imag / denominator.real;
        scale = denominator.real + denominator.imag * ratio;
        return solve_value(
            (numerator.real + numerator.imag * ratio) / scale,
            (numerator.imag - numerator.real * ratio) / scale);
    }
    ratio = denominator.real / denominator.imag;
    scale = denominator.imag + denominator.real * ratio;
    return solve_value(
        (numerator.real * ratio + numerator.imag) / scale,
        (numerator.imag * ratio - numerator.real) / scale);
}

static double solve_absolute(CnpSolveValue value) {
    return hypot(value.real, value.imag);
}

static bool solve_type_is_supported(CNP_TYPE type) {
    return type == CNP_BOOL || cnp_type_is_integer(type) ||
        type == CNP_FLOAT || type == CNP_DOUBLE ||
        type == CNP_CFLOAT || type == CNP_CDOUBLE;
}

static CNP_TYPE solve_input_calculation_type(CNP_TYPE type) {
    if (type == CNP_FLOAT || type == CNP_CFLOAT ||
            type == CNP_DOUBLE || type == CNP_CDOUBLE) {
        return type;
    }
    return CNP_DOUBLE;
}

static CNP_TYPE solve_result_type(CNP_TYPE left, CNP_TYPE right) {
    return cnp_promote_type(
        solve_input_calculation_type(left),
        solve_input_calculation_type(right));
}

static CnpSolveValue solve_read_offset(
        const CnpArray *array, int64_t offset) {
    const char *pointer = (const char*)array->data + offset;
    if (array->dtype->type_num == CNP_CFLOAT) {
        const cnp_cfloat *value = (const cnp_cfloat*)pointer;
        return solve_value(value->real, value->imag);
    }
    if (array->dtype->type_num == CNP_CDOUBLE) {
        const cnp_cdouble *value = (const cnp_cdouble*)pointer;
        return solve_value(value->real, value->imag);
    }
    return solve_value(cnp_get_element_double(
        array->data, offset, array->dtype->type_num), 0.0);
}

static void solve_write_value(
        CnpArray *array, int64_t index, CnpSolveValue value) {
    switch (array->dtype->type_num) {
        case CNP_FLOAT:
            ((float*)array->data)[index] = (float)value.real;
            break;
        case CNP_CFLOAT:
            ((cnp_cfloat*)array->data)[index].real = (float)value.real;
            ((cnp_cfloat*)array->data)[index].imag = (float)value.imag;
            break;
        case CNP_CDOUBLE:
            ((cnp_cdouble*)array->data)[index].real = value.real;
            ((cnp_cdouble*)array->data)[index].imag = value.imag;
            break;
        default:
            ((double*)array->data)[index] = value.real;
            break;
    }
}

static int64_t solve_broadcast_batch_offset(
        const CnpArray *array, int core_dimensions,
        int output_batch_ndim, const int64_t *output_batch_shape,
        int64_t batch_index) {
    int array_batch_ndim = array->ndim - core_dimensions;
    int dimension_shift = output_batch_ndim - array_batch_ndim;
    int64_t coordinates[CNP_MAXDIMS];
    int64_t offset = array->offset;
    for (int dimension = output_batch_ndim - 1; dimension >= 0; dimension--) {
        coordinates[dimension] = output_batch_shape[dimension] == 0
            ? 0 : batch_index % output_batch_shape[dimension];
        if (output_batch_shape[dimension] != 0) {
            batch_index /= output_batch_shape[dimension];
        }
    }
    for (int dimension = 0; dimension < array_batch_ndim; dimension++) {
        int output_dimension = dimension + dimension_shift;
        if (array->shape[dimension] != 1) {
            offset += coordinates[output_dimension] * array->strides[dimension];
        }
    }
    return offset;
}

static CNP_STATUS solve_factor_and_apply(
        CnpSolveValue *matrix, CnpSolveValue *rhs,
        int64_t n, int64_t rhs_columns) {
    for (int64_t pivot = 0; pivot < n; pivot++) {
        int64_t pivot_row = pivot;
        double pivot_absolute = solve_absolute(matrix[pivot * n + pivot]);
        for (int64_t row = pivot + 1; row < n; row++) {
            double candidate = solve_absolute(matrix[row * n + pivot]);
            if (candidate > pivot_absolute || isnan(candidate)) {
                pivot_absolute = candidate;
                pivot_row = row;
            }
        }
        if (pivot_absolute == 0.0) return CNP_ERR_SINGULAR;
        if (pivot_row != pivot) {
            for (int64_t column = 0; column < n; column++) {
                CnpSolveValue temporary = matrix[pivot * n + column];
                matrix[pivot * n + column] =
                    matrix[pivot_row * n + column];
                matrix[pivot_row * n + column] = temporary;
            }
            for (int64_t column = 0; column < rhs_columns; column++) {
                CnpSolveValue temporary = rhs[pivot * rhs_columns + column];
                rhs[pivot * rhs_columns + column] =
                    rhs[pivot_row * rhs_columns + column];
                rhs[pivot_row * rhs_columns + column] = temporary;
            }
        }
        for (int64_t row = pivot + 1; row < n; row++) {
            CnpSolveValue factor = solve_divide(
                matrix[row * n + pivot], matrix[pivot * n + pivot]);
            matrix[row * n + pivot] = solve_value(0.0, 0.0);
            for (int64_t column = pivot + 1; column < n; column++) {
                matrix[row * n + column] = solve_subtract(
                    matrix[row * n + column],
                    solve_multiply(factor, matrix[pivot * n + column]));
            }
            for (int64_t column = 0; column < rhs_columns; column++) {
                rhs[row * rhs_columns + column] = solve_subtract(
                    rhs[row * rhs_columns + column],
                    solve_multiply(
                        factor, rhs[pivot * rhs_columns + column]));
            }
        }
    }
    for (int64_t row = n; row-- > 0;) {
        for (int64_t column = 0; column < rhs_columns; column++) {
            CnpSolveValue value = rhs[row * rhs_columns + column];
            for (int64_t inner = row + 1; inner < n; inner++) {
                value = solve_subtract(
                    value,
                    solve_multiply(
                        matrix[row * n + inner],
                        rhs[inner * rhs_columns + column]));
            }
            rhs[row * rhs_columns + column] = solve_divide(
                value, matrix[row * n + row]);
        }
    }
    return CNP_OK;
}

CNP_API CNP_STATUS CNP_CALL cnp_linalg_solve(const CnpArray *a, const CnpArray *b, CnpArray **result) {
    const char *function_name = "cnp_linalg_solve";
    int64_t output_shape[CNP_MAXDIMS];
    int64_t batch_shape[CNP_MAXDIMS];
    int a_batch_ndim;
    int b_batch_ndim;
    int batch_ndim;
    int output_ndim;
    bool vector_rhs;
    int64_t n;
    int64_t rhs_columns;
    int64_t batch_count = 1;
    int64_t matrix_count;
    int64_t rhs_count;
    CNP_TYPE output_type;
    CnpSolveValue *matrix = NULL;
    CnpSolveValue *rhs = NULL;
    CnpArray *output = NULL;
    CNP_STATUS status = CNP_OK;

    if (!result) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "result output must not be null");
        return CNP_ERR_GENERIC;
    }
    *result = NULL;
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input arrays must not be null");
        return CNP_ERR_GENERIC;
    }
    if (a->ndim < 2 ||
            a->shape[a->ndim - 2] != a->shape[a->ndim - 1]) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "A must be square on its last two dimensions");
        return CNP_ERR_SHAPE;
    }
    if (!solve_type_is_supported(a->dtype->type_num) ||
            !solve_type_is_supported(b->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input dtypes are not supported by linear algebra");
        return CNP_ERR_TYPE;
    }
    n = a->shape[a->ndim - 1];
    vector_rhs = b->ndim == a->ndim - 1;
    if (vector_rhs) {
        if (b->ndim < 1 || b->shape[b->ndim - 1] != n) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "vector right-hand side core dimension does not match A");
            return CNP_ERR_SHAPE;
        }
        rhs_columns = 1;
    } else {
        if (b->ndim < 2 || b->shape[b->ndim - 2] != n) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "right-hand side core dimension does not match A");
            return CNP_ERR_SHAPE;
        }
        rhs_columns = b->shape[b->ndim - 1];
    }

    a_batch_ndim = a->ndim - 2;
    b_batch_ndim = b->ndim - (vector_rhs ? 1 : 2);
    batch_ndim = a_batch_ndim > b_batch_ndim
        ? a_batch_ndim : b_batch_ndim;
    if (batch_ndim + (vector_rhs ? 1 : 2) > CNP_MAXDIMS) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "broadcast result has too many dimensions");
        return CNP_ERR_SHAPE;
    }
    for (int dimension = 0; dimension < batch_ndim; dimension++) {
        int a_dimension = dimension - (batch_ndim - a_batch_ndim);
        int b_dimension = dimension - (batch_ndim - b_batch_ndim);
        int64_t a_length = a_dimension < 0 ? 1 : a->shape[a_dimension];
        int64_t b_length = b_dimension < 0 ? 1 : b->shape[b_dimension];
        int64_t length;
        if (a_length != b_length && a_length != 1 && b_length != 1) {
            cnp_set_error(
                CNP_ERR_BROADCAST, function_name,
                "A and right-hand side batch dimensions are not broadcastable");
            return CNP_ERR_BROADCAST;
        }
        if (a_length == b_length) length = a_length;
        else if (a_length == 1) length = b_length;
        else length = a_length;
        batch_shape[dimension] = length;
        output_shape[dimension] = length;
        if (length != 0 && batch_count > INT64_MAX / length) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "broadcast batch shape is too large");
            return CNP_ERR_SHAPE;
        }
        batch_count *= length;
    }
    output_shape[batch_ndim] = n;
    if (!vector_rhs) output_shape[batch_ndim + 1] = rhs_columns;
    output_ndim = batch_ndim + (vector_rhs ? 1 : 2);
    output_type = solve_result_type(
        a->dtype->type_num, b->dtype->type_num);
    if (output_type == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input dtypes do not have a linear algebra result type");
        return CNP_ERR_TYPE;
    }
    if (batch_count == 0) {
        output = cnp_array_new(
            output_ndim, output_shape, output_type, CNP_ORDER_C);
        if (!output) {
            cnp_relabel_error(function_name);
            return CNP_ERR_MEMORY;
        }
        *result = output;
        return CNP_OK;
    }
    if (n != 0 && n > INT64_MAX / n) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "matrix workspace is too large");
        return CNP_ERR_MEMORY;
    }
    matrix_count = n * n;
    if (n != 0 && rhs_columns > INT64_MAX / n) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "right-hand side workspace is too large");
        return CNP_ERR_MEMORY;
    }
    rhs_count = n * rhs_columns;
    if ((uint64_t)matrix_count > SIZE_MAX / sizeof(CnpSolveValue) ||
            (uint64_t)rhs_count > SIZE_MAX / sizeof(CnpSolveValue)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "linear solve workspace is too large");
        return CNP_ERR_MEMORY;
    }
    if (matrix_count > 0) {
        matrix = (CnpSolveValue*)cnp_malloc(
            (size_t)matrix_count * sizeof(CnpSolveValue));
    }
    if (rhs_count > 0) {
        rhs = (CnpSolveValue*)cnp_malloc(
            (size_t)rhs_count * sizeof(CnpSolveValue));
    }
    if ((matrix_count > 0 && !matrix) || (rhs_count > 0 && !rhs)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "unable to allocate linear solve workspace");
        status = CNP_ERR_MEMORY;
        goto cleanup;
    }
    output = cnp_array_new(
        output_ndim, output_shape, output_type, CNP_ORDER_C);
    if (!output) {
        cnp_relabel_error(function_name);
        status = CNP_ERR_MEMORY;
        goto cleanup;
    }
    for (int64_t batch = 0; batch < batch_count; batch++) {
        int64_t a_offset = solve_broadcast_batch_offset(
            a, 2, batch_ndim, batch_shape, batch);
        int64_t b_offset = solve_broadcast_batch_offset(
            b, vector_rhs ? 1 : 2,
            batch_ndim, batch_shape, batch);
        for (int64_t row = 0; row < n; row++) {
            for (int64_t column = 0; column < n; column++) {
                matrix[row * n + column] = solve_read_offset(
                    a, a_offset + row * a->strides[a->ndim - 2] +
                        column * a->strides[a->ndim - 1]);
            }
            for (int64_t column = 0; column < rhs_columns; column++) {
                int64_t offset = vector_rhs
                    ? b_offset + row * b->strides[b->ndim - 1]
                    : b_offset + row * b->strides[b->ndim - 2] +
                        column * b->strides[b->ndim - 1];
                rhs[row * rhs_columns + column] =
                    solve_read_offset(b, offset);
            }
        }
        status = solve_factor_and_apply(matrix, rhs, n, rhs_columns);
        if (status != CNP_OK) {
            cnp_set_error(
                CNP_ERR_SINGULAR, function_name,
                "matrix is singular");
            goto cleanup;
        }
        for (int64_t row = 0; row < n; row++) {
            for (int64_t column = 0; column < rhs_columns; column++) {
                int64_t output_index = vector_rhs
                    ? batch * n + row
                    : batch * n * rhs_columns + row * rhs_columns + column;
                solve_write_value(
                    output, output_index,
                    rhs[row * rhs_columns + column]);
            }
        }
    }
    *result = output;
    output = NULL;

cleanup:
    if (matrix) {
        cnp_free(
            matrix, (size_t)matrix_count * sizeof(CnpSolveValue));
    }
    if (rhs) {
        cnp_free(rhs, (size_t)rhs_count * sizeof(CnpSolveValue));
    }
    if (output) cnp_array_free(output);
    if (status != CNP_OK) *result = NULL;
    return status;
}

/* =========================================================================
 * Least squares and default 2-norm condition number
 * ========================================================================= */
static CnpSolveValue lstsq_read_matrix_value(
        const CnpArray *array, int64_t row, int64_t column) {
    return solve_read_offset(
        array,
        array->offset + row * array->strides[array->ndim - 2] +
            column * array->strides[array->ndim - 1]);
}

static CnpSolveValue lstsq_read_rhs_value(
        const CnpArray *array, int64_t row, int64_t column) {
    int64_t offset = array->offset + row * array->strides[0];
    if (array->ndim == 2) offset += column * array->strides[1];
    return solve_read_offset(array, offset);
}

CNP_API CNP_STATUS CNP_CALL cnp_linalg_lstsq_v2(
        const CnpArray *a, const CnpArray *b,
        double rcond, bool rcond_none,
        CnpArray **x, CnpArray **residuals,
        CnpArray **rank, CnpArray **singular_values) {
    const char *function_name = "cnp_linalg_lstsq_v2";
    CnpArray *calculation_a = NULL;
    CnpArray *u = NULL;
    CnpArray *s = NULL;
    CnpArray *vh = NULL;
    CnpArray *x_result = NULL;
    CnpArray *residual_result = NULL;
    CnpArray *rank_result = NULL;
    int64_t x_shape[2];
    int64_t residual_shape[1];
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t singular_count;
    int64_t numerical_rank = 0;
    double maximum_singular = 0.0;
    double cutoff;
    double epsilon;
    CNP_TYPE calculation_type;
    CNP_TYPE residual_type;
    CNP_STATUS status = CNP_OK;

    if (x) *x = NULL;
    if (residuals) *residuals = NULL;
    if (rank) *rank = NULL;
    if (singular_values) *singular_values = NULL;
    if (!x || !residuals || !rank || !singular_values ||
            x == residuals || x == rank || x == singular_values ||
            residuals == rank || residuals == singular_values ||
            rank == singular_values) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "four distinct result output pointers are required");
        return CNP_ERR_GENERIC;
    }
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input arrays must not be null");
        return CNP_ERR_GENERIC;
    }
    if (a->ndim != 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "coefficient array must be two-dimensional");
        return CNP_ERR_SHAPE;
    }
    if (b->ndim != 1 && b->ndim != 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "right-hand side must be one- or two-dimensional");
        return CNP_ERR_SHAPE;
    }
    if (b->shape[0] != a->shape[0]) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "right-hand side row count must match the coefficient array");
        return CNP_ERR_SHAPE;
    }
    if (!solve_type_is_supported(a->dtype->type_num) ||
            !solve_type_is_supported(b->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input dtypes are not supported by linear algebra");
        return CNP_ERR_TYPE;
    }
    m = a->shape[0];
    n = a->shape[1];
    k = b->ndim == 1 ? 1 : b->shape[1];
    singular_count = m < n ? m : n;
    calculation_type = solve_result_type(
        a->dtype->type_num, b->dtype->type_num);
    residual_type = calculation_type == CNP_FLOAT ||
            calculation_type == CNP_CFLOAT
        ? CNP_FLOAT : CNP_DOUBLE;
    calculation_a = cnp_astype(a, calculation_type, CNP_CAST_UNSAFE);
    if (!calculation_a) {
        cnp_relabel_error(function_name);
        return cnp_get_error(NULL);
    }
    status = cnp_linalg_svd_v2(
        calculation_a, false, true, false, &u, &s, &vh);
    if (status != CNP_OK) {
        cnp_relabel_error(function_name);
        goto cleanup;
    }
    if (singular_count > 0) {
        maximum_singular = cnp_get_element_double(
            s->data, s->offset, s->dtype->type_num);
    }
    epsilon = residual_type == CNP_FLOAT ? FLT_EPSILON : DBL_EPSILON;
    if (rcond_none) {
        cutoff = epsilon * (double)(m > n ? m : n) * maximum_singular;
    } else if (!(rcond >= 0.0 && rcond < 1.0)) {
        /* NumPy 1.25 inherits xGELSD's machine-precision treatment for
         * negative, NaN, infinite, and >= 1 RCOND values. */
        cutoff = epsilon * maximum_singular;
    } else {
        cutoff = rcond * maximum_singular;
    }
    for (int64_t index = 0; index < singular_count; index++) {
        double singular = cnp_get_element_double(
            s->data, s->offset + index * s->strides[0],
            s->dtype->type_num);
        if (singular > cutoff) numerical_rank++;
    }

    x_shape[0] = n;
    x_shape[1] = k;
    x_result = cnp_array_new(
        b->ndim, x_shape, calculation_type, CNP_ORDER_C);
    residual_shape[0] = m > n && numerical_rank == n ? k : 0;
    residual_result = cnp_array_new(
        1, residual_shape, residual_type, CNP_ORDER_C);
    rank_result = cnp_array_new(0, NULL, CNP_INT, CNP_ORDER_C);
    if (!x_result || !residual_result || !rank_result) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "unable to allocate least-squares results");
        status = CNP_ERR_MEMORY;
        goto cleanup;
    }
    cnp_set_element_int(
        rank_result->data, rank_result->offset,
        rank_result->dtype->type_num, numerical_rank);

    for (int64_t output_row = 0; output_row < n; output_row++) {
        for (int64_t rhs_column = 0; rhs_column < k; rhs_column++) {
            CnpSolveValue solution = solve_value(0.0, 0.0);
            for (int64_t singular_index = 0;
                    singular_index < numerical_rank; singular_index++) {
                CnpSolveValue projection = solve_value(0.0, 0.0);
                double singular = cnp_get_element_double(
                    s->data,
                    s->offset + singular_index * s->strides[0],
                    s->dtype->type_num);
                for (int64_t input_row = 0; input_row < m; input_row++) {
                    CnpSolveValue u_value = lstsq_read_matrix_value(
                        u, input_row, singular_index);
                    projection = solve_add(
                        projection,
                        solve_multiply(
                            solve_conjugate(u_value),
                            lstsq_read_rhs_value(
                                b, input_row, rhs_column)));
                }
                projection = solve_divide(
                    projection, solve_value(singular, 0.0));
                solution = solve_add(
                    solution,
                    solve_multiply(
                        solve_conjugate(lstsq_read_matrix_value(
                            vh, singular_index, output_row)),
                        projection));
            }
            solve_write_value(
                x_result, output_row * k + rhs_column, solution);
        }
    }

    if (residual_shape[0] != 0) {
        for (int64_t rhs_column = 0; rhs_column < k; rhs_column++) {
            double residual = 0.0;
            for (int64_t row = 0; row < m; row++) {
                CnpSolveValue predicted = solve_value(0.0, 0.0);
                CnpSolveValue difference;
                for (int64_t column = 0; column < n; column++) {
                    int64_t x_index = column * k + rhs_column;
                    int64_t x_offset = x_result->offset +
                        x_index * x_result->dtype->elsize;
                    predicted = solve_add(
                        predicted,
                        solve_multiply(
                            lstsq_read_matrix_value(a, row, column),
                            solve_read_offset(x_result, x_offset)));
                }
                difference = solve_subtract(
                    lstsq_read_rhs_value(b, row, rhs_column), predicted);
                residual += difference.real * difference.real +
                    difference.imag * difference.imag;
            }
            if (residual_type == CNP_FLOAT) {
                ((float*)residual_result->data)[rhs_column] = (float)residual;
            } else {
                ((double*)residual_result->data)[rhs_column] = residual;
            }
        }
    }

    *x = x_result;
    *residuals = residual_result;
    *rank = rank_result;
    *singular_values = s;
    x_result = NULL;
    residual_result = NULL;
    rank_result = NULL;
    s = NULL;

cleanup:
    if (calculation_a) cnp_array_free(calculation_a);
    if (u) cnp_array_free(u);
    if (s) cnp_array_free(s);
    if (vh) cnp_array_free(vh);
    if (x_result) cnp_array_free(x_result);
    if (residual_result) cnp_array_free(residual_result);
    if (rank_result) cnp_array_free(rank_result);
    if (status != CNP_OK) {
        *x = NULL;
        *residuals = NULL;
        *rank = NULL;
        *singular_values = NULL;
    }
    return status;
}

CNP_API CnpArray* CNP_CALL cnp_linalg_cond_v2(const CnpArray *a) {
    const char *function_name = "cnp_linalg_cond_v2";
    CnpArray *calculation_a = NULL;
    CnpArray *singular_values = NULL;
    CnpArray *result = NULL;
    int64_t result_shape[CNP_MAXDIMS];
    int64_t batch_count = 1;
    int64_t singular_count;
    CNP_TYPE calculation_type;
    CNP_STATUS status;

    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array must not be null");
        return NULL;
    }
    if (a->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input must have at least two dimensions");
        return NULL;
    }
    if (a->shape[a->ndim - 2] == 0 || a->shape[a->ndim - 1] == 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "condition number is not defined on empty matrices");
        return NULL;
    }
    if (!solve_type_is_supported(a->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input dtype is not supported by linear algebra");
        return NULL;
    }
    calculation_type = solve_input_calculation_type(a->dtype->type_num);
    calculation_a = cnp_astype(a, calculation_type, CNP_CAST_UNSAFE);
    if (!calculation_a) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    status = cnp_linalg_svd_v2(
        calculation_a, false, false, false,
        NULL, &singular_values, NULL);
    cnp_array_free(calculation_a);
    if (status != CNP_OK) {
        if (singular_values) cnp_array_free(singular_values);
        cnp_relabel_error(function_name);
        return NULL;
    }
    singular_count = singular_values->shape[singular_values->ndim - 1];
    for (int dimension = 0; dimension < a->ndim - 2; dimension++) {
        int64_t length = a->shape[dimension];
        result_shape[dimension] = length;
        if (length != 0 && batch_count > INT64_MAX / length) {
            cnp_array_free(singular_values);
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "condition result batch shape is too large");
            return NULL;
        }
        batch_count *= length;
    }
    result = cnp_array_new(
        a->ndim - 2, result_shape,
        singular_values->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_array_free(singular_values);
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int64_t batch = 0; batch < batch_count; batch++) {
        int64_t base_index = batch * singular_count;
        double maximum = cnp_get_element_double(
            singular_values->data,
            singular_values->offset +
                base_index * singular_values->dtype->elsize,
            singular_values->dtype->type_num);
        double minimum = cnp_get_element_double(
            singular_values->data,
            singular_values->offset +
                (base_index + singular_count - 1) *
                    singular_values->dtype->elsize,
            singular_values->dtype->type_num);
        double condition = minimum == 0.0 ? INFINITY : maximum / minimum;
        if (result->dtype->type_num == CNP_FLOAT) {
            ((float*)result->data)[batch] = (float)condition;
        } else {
            ((double*)result->data)[batch] = condition;
        }
    }
    cnp_array_free(singular_values);
    return result;
}

/* =========================================================================
 * QR decomposition (reduced Householder factorization)
 * ========================================================================= */
static CNP_STATUS qr_factor_batch(
        CnpSolveValue *matrix, CnpSolveValue *orthogonal,
        CnpSolveValue *reflector, int64_t rows, int64_t columns) {
    int64_t reduced = rows < columns ? rows : columns;
    memset(
        orthogonal, 0,
        (size_t)rows * (size_t)rows * sizeof(CnpSolveValue));
    for (int64_t diagonal = 0; diagonal < rows; diagonal++) {
        orthogonal[diagonal * rows + diagonal] = solve_value(1.0, 0.0);
    }

    for (int64_t step = 0; step < reduced; step++) {
        int64_t length = rows - step;
        double scale = 0.0;
        double sum = 1.0;
        for (int64_t index = 0; index < length; index++) {
            CnpSolveValue value = matrix[(step + index) * columns + step];
            double absolute = solve_absolute(value);
            reflector[index] = value;
            if (absolute == 0.0) continue;
            if (scale < absolute) {
                double ratio = scale / absolute;
                sum = 1.0 + sum * ratio * ratio;
                scale = absolute;
            } else {
                double ratio = absolute / scale;
                sum += ratio * ratio;
            }
        }
        double norm = scale == 0.0 ? 0.0 : scale * sqrt(sum);
        if (norm == 0.0) continue;

        double first_absolute = solve_absolute(reflector[0]);
        CnpSolveValue phase = first_absolute == 0.0
            ? solve_value(1.0, 0.0)
            : solve_value(
                reflector[0].real / first_absolute,
                reflector[0].imag / first_absolute);
        CnpSolveValue alpha = solve_value(
            -phase.real * norm, -phase.imag * norm);
        reflector[0] = solve_subtract(reflector[0], alpha);

        double denominator = 0.0;
        for (int64_t index = 0; index < length; index++) {
            double absolute = solve_absolute(reflector[index]);
            denominator += absolute * absolute;
        }
        if (denominator == 0.0) continue;
        double beta = 2.0 / denominator;

        for (int64_t column = step; column < columns; column++) {
            CnpSolveValue product = solve_value(0.0, 0.0);
            for (int64_t index = 0; index < length; index++) {
                product = solve_add(
                    product,
                    solve_multiply(
                        solve_conjugate(reflector[index]),
                        matrix[(step + index) * columns + column]));
            }
            product.real *= beta;
            product.imag *= beta;
            for (int64_t index = 0; index < length; index++) {
                int64_t offset = (step + index) * columns + column;
                matrix[offset] = solve_subtract(
                    matrix[offset],
                    solve_multiply(reflector[index], product));
            }
        }
        for (int64_t row = 0; row < rows; row++) {
            CnpSolveValue product = solve_value(0.0, 0.0);
            for (int64_t index = 0; index < length; index++) {
                product = solve_add(
                    product,
                    solve_multiply(
                        orthogonal[row * rows + step + index],
                        reflector[index]));
            }
            product.real *= beta;
            product.imag *= beta;
            for (int64_t index = 0; index < length; index++) {
                int64_t offset = row * rows + step + index;
                orthogonal[offset] = solve_subtract(
                    orthogonal[offset],
                    solve_multiply(
                        product, solve_conjugate(reflector[index])));
            }
        }
        for (int64_t row = step + 1; row < rows; row++) {
            matrix[row * columns + step] = solve_value(0.0, 0.0);
        }
    }
    return CNP_OK;
}

CNP_API CNP_STATUS CNP_CALL cnp_linalg_qr(
        const CnpArray *a, CnpArray **q, CnpArray **r) {
    const char *function_name = "cnp_linalg_qr";
    CnpArray *q_result = NULL;
    CnpArray *r_result = NULL;
    CnpSolveValue *matrix = NULL;
    CnpSolveValue *orthogonal = NULL;
    CnpSolveValue *reflector = NULL;
    int64_t q_shape[CNP_MAXDIMS];
    int64_t r_shape[CNP_MAXDIMS];
    int64_t rows;
    int64_t columns;
    int64_t reduced;
    int64_t batch_count = 1;
    size_t matrix_count;
    size_t orthogonal_count;
    size_t reflector_count;
    CNP_TYPE output_type;
    CNP_STATUS status = CNP_OK;

    if (!q || !r) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "both output slots must not be null");
        return CNP_ERR_GENERIC;
    }
    *q = NULL;
    *r = NULL;
    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array must not be null");
        return CNP_ERR_GENERIC;
    }
    if (a->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input must have at least two dimensions");
        return CNP_ERR_SHAPE;
    }
    if (!solve_type_is_supported(a->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input dtype is not supported by linear algebra");
        return CNP_ERR_TYPE;
    }

    rows = a->shape[a->ndim - 2];
    columns = a->shape[a->ndim - 1];
    reduced = rows < columns ? rows : columns;
    output_type = solve_input_calculation_type(a->dtype->type_num);
    for (int dimension = 0; dimension < a->ndim - 2; dimension++) {
        q_shape[dimension] = a->shape[dimension];
        r_shape[dimension] = a->shape[dimension];
        if (a->shape[dimension] != 0 &&
                batch_count > INT64_MAX / a->shape[dimension]) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "QR batch shape is too large");
            return CNP_ERR_SHAPE;
        }
        batch_count *= a->shape[dimension];
    }
    q_shape[a->ndim - 2] = rows;
    q_shape[a->ndim - 1] = reduced;
    r_shape[a->ndim - 2] = reduced;
    r_shape[a->ndim - 1] = columns;
    q_result = cnp_array_new(
        a->ndim, q_shape, output_type, CNP_ORDER_C);
    if (!q_result) {
        cnp_relabel_error(function_name);
        return cnp_get_error(NULL);
    }
    r_result = cnp_array_new(
        a->ndim, r_shape, output_type, CNP_ORDER_C);
    if (!r_result) {
        cnp_array_free(q_result);
        cnp_relabel_error(function_name);
        return cnp_get_error(NULL);
    }

    if (rows != 0 && (uint64_t)columns > SIZE_MAX / (uint64_t)rows) {
        status = CNP_ERR_MEMORY;
        cnp_set_error(status, function_name, "QR workspace is too large");
        goto cleanup;
    }
    matrix_count = (size_t)rows * (size_t)columns;
    if (rows != 0 && (uint64_t)rows > SIZE_MAX / (uint64_t)rows) {
        status = CNP_ERR_MEMORY;
        cnp_set_error(status, function_name, "QR workspace is too large");
        goto cleanup;
    }
    orthogonal_count = (size_t)rows * (size_t)rows;
    reflector_count = (size_t)rows;
    if (matrix_count > SIZE_MAX / sizeof(CnpSolveValue) ||
            orthogonal_count > SIZE_MAX / sizeof(CnpSolveValue) ||
            reflector_count > SIZE_MAX / sizeof(CnpSolveValue)) {
        status = CNP_ERR_MEMORY;
        cnp_set_error(status, function_name, "QR workspace is too large");
        goto cleanup;
    }
    if (matrix_count != 0) {
        matrix = (CnpSolveValue*)cnp_malloc(
            matrix_count * sizeof(CnpSolveValue));
    }
    if (orthogonal_count != 0) {
        orthogonal = (CnpSolveValue*)cnp_malloc(
            orthogonal_count * sizeof(CnpSolveValue));
    }
    if (reflector_count != 0) {
        reflector = (CnpSolveValue*)cnp_malloc(
            reflector_count * sizeof(CnpSolveValue));
    }
    if ((matrix_count != 0 && !matrix) ||
            (orthogonal_count != 0 && !orthogonal) ||
            (reflector_count != 0 && !reflector)) {
        status = CNP_ERR_MEMORY;
        cnp_set_error(status, function_name, "failed to allocate QR workspace");
        goto cleanup;
    }

    for (int64_t batch = 0; batch < batch_count; batch++) {
        int64_t source_base = solve_broadcast_batch_offset(
            a, 2, a->ndim - 2, a->shape, batch);
        for (int64_t row = 0; row < rows; row++) {
            for (int64_t column = 0; column < columns; column++) {
                matrix[row * columns + column] = solve_read_offset(
                    a,
                    source_base + row * a->strides[a->ndim - 2] +
                        column * a->strides[a->ndim - 1]);
            }
        }
        if (rows != 0) {
            status = qr_factor_batch(
                matrix, orthogonal, reflector, rows, columns);
            if (status != CNP_OK) goto cleanup;
        }
        int64_t q_base = batch * rows * reduced;
        int64_t r_base = batch * reduced * columns;
        for (int64_t row = 0; row < rows; row++) {
            for (int64_t column = 0; column < reduced; column++) {
                solve_write_value(
                    q_result, q_base + row * reduced + column,
                    orthogonal[row * rows + column]);
            }
        }
        for (int64_t row = 0; row < reduced; row++) {
            for (int64_t column = 0; column < columns; column++) {
                solve_write_value(
                    r_result, r_base + row * columns + column,
                    matrix[row * columns + column]);
            }
        }
    }

    *q = q_result;
    *r = r_result;
    q_result = NULL;
    r_result = NULL;

cleanup:
    if (reflector) {
        cnp_free(reflector, reflector_count * sizeof(CnpSolveValue));
    }
    if (orthogonal) {
        cnp_free(orthogonal, orthogonal_count * sizeof(CnpSolveValue));
    }
    if (matrix) cnp_free(matrix, matrix_count * sizeof(CnpSolveValue));
    if (q_result) cnp_array_free(q_result);
    if (r_result) cnp_array_free(r_result);
    if (status != CNP_OK) {
        *q = NULL;
        *r = NULL;
    }
    return status;
}

/* =========================================================================
 * Cholesky decomposition
 * ========================================================================= */
static float cholesky_read_float(
        const CnpArray *array, int64_t batch_offset,
        int64_t row, int64_t column) {
    int64_t offset = batch_offset +
        row * array->strides[array->ndim - 2] +
        column * array->strides[array->ndim - 1];
    return *(const float*)((const char*)array->data + offset);
}

static cnp_cfloat cholesky_read_cfloat(
        const CnpArray *array, int64_t batch_offset,
        int64_t row, int64_t column) {
    int64_t offset = batch_offset +
        row * array->strides[array->ndim - 2] +
        column * array->strides[array->ndim - 1];
    return *(const cnp_cfloat*)((const char*)array->data + offset);
}

static cnp_cdouble cholesky_read_cdouble(
        const CnpArray *array, int64_t batch_offset,
        int64_t row, int64_t column) {
    int64_t offset = batch_offset +
        row * array->strides[array->ndim - 2] +
        column * array->strides[array->ndim - 1];
    return *(const cnp_cdouble*)((const char*)array->data + offset);
}

static bool cholesky_float_is_nonpositive(float value) {
    uint32_t bits;
    uint32_t magnitude;
    memcpy(&bits, &value, sizeof(bits));
    magnitude = bits & UINT32_C(0x7fffffff);
    if (magnitude > UINT32_C(0x7f800000)) return false;
    return (bits & UINT32_C(0x80000000)) != 0 || magnitude == 0;
}

static bool cholesky_double_is_nonpositive(double value) {
    uint64_t bits;
    uint64_t magnitude;
    memcpy(&bits, &value, sizeof(bits));
    magnitude = bits & UINT64_C(0x7fffffffffffffff);
    if (magnitude > UINT64_C(0x7ff0000000000000)) return false;
    return (bits & UINT64_C(0x8000000000000000)) != 0 || magnitude == 0;
}

static bool cholesky_factor_float(
        const CnpArray *source, int64_t source_offset,
        float *destination, int64_t n) {
    for (int64_t row = 0; row < n; row++) {
        for (int64_t column = 0; column <= row; column++) {
            float sum = 0.0f;
            for (int64_t inner = 0; inner < column; inner++) {
                sum += destination[row * n + inner] *
                    destination[column * n + inner];
            }
            float value = cholesky_read_float(
                source, source_offset, row, column) - sum;
            if (row == column) {
                if (cholesky_float_is_nonpositive(value)) return false;
                destination[row * n + column] = sqrtf(value);
            } else {
                destination[row * n + column] = value /
                    destination[column * n + column];
            }
        }
    }
    return true;
}

static bool cholesky_factor_double(
        const CnpArray *source, int64_t source_offset,
        double *destination, int64_t n) {
    for (int64_t row = 0; row < n; row++) {
        for (int64_t column = 0; column <= row; column++) {
            double sum = 0.0;
            for (int64_t inner = 0; inner < column; inner++) {
                sum += destination[row * n + inner] *
                    destination[column * n + inner];
            }
            double value = det_read_real(
                source, source_offset, row, column) - sum;
            if (row == column) {
                if (cholesky_double_is_nonpositive(value)) return false;
                destination[row * n + column] = sqrt(value);
            } else {
                destination[row * n + column] = value /
                    destination[column * n + column];
            }
        }
    }
    return true;
}

static bool cholesky_factor_contiguous_double(
        const double *source, double *destination, int n) {
    for (int row = 0; row < n; row++) {
        const double *source_row = source + row * n;
        double *destination_row = destination + row * n;
        for (int column = 0; column <= row; column++) {
            const double *destination_column = destination + column * n;
            double sum = 0.0;
            for (int inner = 0; inner < column; inner++) {
                sum += destination_row[inner] *
                    destination_column[inner];
            }
            double value = source_row[column] - sum;
            if (row == column) {
                if (cholesky_double_is_nonpositive(value)) return false;
                destination_row[column] = sqrt(value);
            } else {
                destination_row[column] = value /
                    destination_column[column];
            }
        }
    }
    return true;
}

static bool cholesky_factor_cfloat(
        const CnpArray *source, int64_t source_offset,
        cnp_cfloat *destination, int64_t n) {
    for (int64_t row = 0; row < n; row++) {
        for (int64_t column = 0; column <= row; column++) {
            float sum_real = 0.0f;
            float sum_imag = 0.0f;
            for (int64_t inner = 0; inner < column; inner++) {
                cnp_cfloat left = destination[row * n + inner];
                cnp_cfloat right = destination[column * n + inner];
                sum_real += left.real * right.real +
                    left.imag * right.imag;
                sum_imag += left.imag * right.real -
                    left.real * right.imag;
            }
            cnp_cfloat input = cholesky_read_cfloat(
                source, source_offset, row, column);
            if (row == column) {
                float value = input.real - sum_real;
                if (cholesky_float_is_nonpositive(value)) return false;
                destination[row * n + column].real = sqrtf(value);
                destination[row * n + column].imag = 0.0f;
            } else {
                float diagonal = destination[column * n + column].real;
                destination[row * n + column].real =
                    (input.real - sum_real) / diagonal;
                destination[row * n + column].imag =
                    (input.imag - sum_imag) / diagonal;
            }
        }
    }
    return true;
}

static bool cholesky_factor_cdouble(
        const CnpArray *source, int64_t source_offset,
        cnp_cdouble *destination, int64_t n) {
    for (int64_t row = 0; row < n; row++) {
        for (int64_t column = 0; column <= row; column++) {
            double sum_real = 0.0;
            double sum_imag = 0.0;
            for (int64_t inner = 0; inner < column; inner++) {
                cnp_cdouble left = destination[row * n + inner];
                cnp_cdouble right = destination[column * n + inner];
                sum_real += left.real * right.real +
                    left.imag * right.imag;
                sum_imag += left.imag * right.real -
                    left.real * right.imag;
            }
            cnp_cdouble input = cholesky_read_cdouble(
                source, source_offset, row, column);
            if (row == column) {
                double value = input.real - sum_real;
                if (cholesky_double_is_nonpositive(value)) return false;
                destination[row * n + column].real = sqrt(value);
                destination[row * n + column].imag = 0.0;
            } else {
                double diagonal = destination[column * n + column].real;
                destination[row * n + column].real =
                    (input.real - sum_real) / diagonal;
                destination[row * n + column].imag =
                    (input.imag - sum_imag) / diagonal;
            }
        }
    }
    return true;
}

CNP_API CNP_STATUS CNP_CALL cnp_linalg_cholesky(
        const CnpArray *a, CnpArray **result) {
    const char *function_name = "cnp_linalg_cholesky";
    int64_t n;
    int64_t matrix_size;
    int64_t batch_count = 1;
    CNP_TYPE result_type;
    CnpArray *output;

    if (result) *result = NULL;
    if (!result) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "Result output pointer must not be null");
        return CNP_ERR_GENERIC;
    }
    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "Input array must not be null");
        return CNP_ERR_GENERIC;
    }
    if (a->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "Input must have at least two dimensions");
        return CNP_ERR_SHAPE;
    }
    if (a->shape[a->ndim - 2] != a->shape[a->ndim - 1]) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "Input must be square on its last two dimensions");
        return CNP_ERR_SHAPE;
    }
    if (!det_type_is_supported(a->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "Input dtype is unsupported in linear algebra");
        return CNP_ERR_TYPE;
    }

    n = a->shape[a->ndim - 1];
    if (n != 0 && n > INT64_MAX / n) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name, "Matrix is too large");
        return CNP_ERR_SHAPE;
    }
    matrix_size = n * n;
    for (int dimension = 0; dimension < a->ndim - 2; dimension++) {
        int64_t length = a->shape[dimension];
        if (length != 0 && batch_count > INT64_MAX / length) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "Batch shape is too large");
            return CNP_ERR_SHAPE;
        }
        batch_count *= length;
    }

    result_type = det_result_type(a->dtype->type_num);
    output = cnp_array_zeros(
        a->ndim, a->shape, result_type, CNP_ORDER_C);
    if (!output) {
        cnp_relabel_error(function_name);
        return CNP_ERR_MEMORY;
    }

    for (int64_t batch = 0; batch < batch_count; batch++) {
        int64_t source_offset = det_batch_offset(a, batch);
        int64_t destination_offset = batch * matrix_size;
        bool positive_definite;
        switch (result_type) {
            case CNP_FLOAT:
                positive_definite = cholesky_factor_float(
                    a, source_offset,
                    (float*)output->data + destination_offset, n);
                break;
            case CNP_CFLOAT:
                positive_definite = cholesky_factor_cfloat(
                    a, source_offset,
                    (cnp_cfloat*)output->data + destination_offset, n);
                break;
            case CNP_CDOUBLE:
                positive_definite = cholesky_factor_cdouble(
                    a, source_offset,
                    (cnp_cdouble*)output->data + destination_offset, n);
                break;
            default:
                if (a->dtype->type_num == CNP_DOUBLE &&
                        (a->flags & CNP_ARRAY_C_CONTIGUOUS) &&
                        n <= INT_MAX) {
                    positive_definite =
                        cholesky_factor_contiguous_double(
                            (const double*)((const char*)a->data +
                                source_offset),
                            (double*)output->data + destination_offset,
                            (int)n);
                } else {
                    positive_definite = cholesky_factor_double(
                        a, source_offset,
                        (double*)output->data + destination_offset, n);
                }
                break;
        }
        if (!positive_definite) {
            cnp_array_free(output);
            cnp_set_error(
                CNP_ERR_SINGULAR, function_name,
                "Matrix is not positive definite");
            return CNP_ERR_SINGULAR;
        }
    }

    *result = output;
    return CNP_OK;
}

/* =========================================================================
 * Norm
 * ========================================================================= */
static bool norm_type_is_supported(CNP_TYPE type) {
    return type == CNP_BOOL || cnp_type_is_integer(type) ||
        cnp_type_is_float(type) || cnp_type_is_complex(type);
}

static CNP_TYPE norm_result_type(CNP_TYPE type) {
    if (type == CNP_HALF) return CNP_HALF;
    if (type == CNP_FLOAT || type == CNP_CFLOAT) return CNP_FLOAT;
    return CNP_DOUBLE;
}

static double norm_read_absolute(const CnpArray *array, int64_t offset) {
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

static CNP_STATUS norm_reduce_line(
        const CnpArray *array, int64_t base_offset, int64_t stride,
        int64_t length, double order, double *result,
        const char *function_name) {
    if ((isinf(order)) && length == 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "infinite-order norm has no identity for an empty axis");
        return CNP_ERR_VALUE;
    }
    if (order == 0.0) {
        double count = 0.0;
        for (int64_t index = 0; index < length; index++) {
            if (norm_read_absolute(array, base_offset + index * stride) != 0.0)
                count += 1.0;
        }
        *result = count;
        return CNP_OK;
    }
    if (isinf(order)) {
        double selected = norm_read_absolute(array, base_offset);
        for (int64_t index = 1; index < length; index++) {
            double value = norm_read_absolute(
                array, base_offset + index * stride);
            if (isnan(value)) {
                selected = NAN;
                break;
            }
            if ((order > 0.0 && value > selected) ||
                    (order < 0.0 && value < selected)) {
                selected = value;
            }
        }
        *result = selected;
        return CNP_OK;
    }
    if (order == 2.0) {
        double magnitude = 0.0;
        for (int64_t index = 0; index < length; index++) {
            magnitude = hypot(
                magnitude,
                norm_read_absolute(array, base_offset + index * stride));
        }
        *result = magnitude;
        return CNP_OK;
    }
    if (order == 1.0) {
        double sum = 0.0;
        for (int64_t index = 0; index < length; index++) {
            sum += norm_read_absolute(
                array, base_offset + index * stride);
        }
        *result = sum;
        return CNP_OK;
    }
    double powered_sum = 0.0;
    for (int64_t index = 0; index < length; index++) {
        powered_sum += pow(
            norm_read_absolute(array, base_offset + index * stride), order);
    }
    *result = pow(powered_sum, 1.0 / order);
    return CNP_OK;
}

CNP_API CnpArray* CNP_CALL cnp_linalg_norm(
        const CnpArray *a, const char *ord, int axis) {
    const char *function_name = "cnp_linalg_norm";
    CNP_TYPE result_type;
    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array must not be null");
        return NULL;
    }
    if (!norm_type_is_supported(a->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input dtype is not supported by linear algebra");
        return NULL;
    }
    result_type = norm_result_type(a->dtype->type_num);

    if (!ord) {
        if ((a->flags & CNP_ARRAY_C_CONTIGUOUS) &&
                a->dtype->type_num == CNP_DOUBLE) {
            const double *data = (const double*)((const char*)
                a->data + a->offset);
            double squared_magnitude = cnp_simd_dot(data, data, a->size);
            return cnp_array_from_scalar(
                sqrt(squared_magnitude), CNP_DOUBLE);
        }
        double magnitude = 0.0;
        int64_t coords[CNP_MAXDIMS] = {0};
        for (int64_t index = 0; index < a->size; index++) {
            int64_t offset = a->offset + cnp_multi_to_offset(
                a->ndim, coords, a->strides);
            magnitude = hypot(magnitude, norm_read_absolute(a, offset));
            for (int dimension = a->ndim - 1; dimension >= 0; dimension--) {
                coords[dimension]++;
                if (coords[dimension] < a->shape[dimension]) break;
                coords[dimension] = 0;
            }
        }
        return cnp_array_from_scalar(magnitude, result_type);
    }

    char *end = NULL;
    double order = strtod(ord, &end);
    if (end == ord || !end || *end != '\0') {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "order '%s' is not a represented vector norm", ord);
        return NULL;
    }
    if (a->ndim == 0) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "an explicit norm axis is invalid for a scalar input");
        return NULL;
    }
    int normalized_axis = cnp_normalize_axis(axis, a->ndim);
    if (normalized_axis < 0) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "axis %d is invalid for rank %d", axis, a->ndim);
        return NULL;
    }

    int output_ndim = a->ndim - 1;
    int64_t output_shape[CNP_MAXDIMS];
    int destination_dimension = 0;
    for (int dimension = 0; dimension < a->ndim; dimension++) {
        if (dimension != normalized_axis) {
            output_shape[destination_dimension++] = a->shape[dimension];
        }
    }
    CnpArray *result = cnp_array_new(
        output_ndim, output_ndim ? output_shape : NULL,
        result_type, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    for (int64_t output_index = 0;
            output_index < result->size; output_index++) {
        int64_t remaining = output_index;
        int64_t base_offset = a->offset;
        for (int output_dimension = output_ndim - 1;
                output_dimension >= 0; output_dimension--) {
            int64_t coordinate = remaining % output_shape[output_dimension];
            remaining /= output_shape[output_dimension];
            int source_dimension = output_dimension >= normalized_axis
                ? output_dimension + 1 : output_dimension;
            base_offset += coordinate * a->strides[source_dimension];
        }
        double value;
        CNP_STATUS status = norm_reduce_line(
            a, base_offset, a->strides[normalized_axis],
            a->shape[normalized_axis], order, &value, function_name);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
        cnp_set_element_double(
            result->data, output_index * result->dtype->elsize,
            result_type, value);
    }
    return result;
}

/* =========================================================================
 * Matrix rank
 * ========================================================================= */
static bool matrix_rank_float_is_nonzero(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7fffffff)) != 0;
}

static bool matrix_rank_double_is_nonzero(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7fffffffffffffff)) != 0;
}

static bool matrix_rank_longdouble_is_nonzero(long double value) {
    if (sizeof(long double) == sizeof(double)) {
        double double_value;
        memcpy(&double_value, &value, sizeof(double_value));
        return matrix_rank_double_is_nonzero(double_value);
    }
    return value != 0.0L || isnan(value);
}

static double matrix_rank_read_real(
    const CnpArray *array, int64_t offset) {
    if (array->dtype->type_num == CNP_HALF) {
        const uint16_t *value = (const uint16_t*)((const char*)
            array->data + offset);
        return cnp_half_to_float(*value);
    }
    return cnp_get_element_double(
        array->data, offset, array->dtype->type_num);
}

static bool matrix_rank_element_is_nonzero(
    const CnpArray *array, int64_t offset) {
    const char *pointer = (const char*)array->data + offset;
    switch (array->dtype->type_num) {
        case CNP_BOOL:
        case CNP_BYTE:
            return *(const int8_t*)pointer != 0;
        case CNP_UBYTE:
            return *(const uint8_t*)pointer != 0;
        case CNP_SHORT:
            return *(const int16_t*)pointer != 0;
        case CNP_USHORT:
            return *(const uint16_t*)pointer != 0;
        case CNP_INT:
            return *(const int32_t*)pointer != 0;
        case CNP_UINT:
            return *(const uint32_t*)pointer != 0;
        case CNP_LONG:
        case CNP_LONGLONG:
            return *(const int64_t*)pointer != 0;
        case CNP_ULONG:
        case CNP_ULONGLONG:
            return *(const uint64_t*)pointer != 0;
        case CNP_HALF:
            return (*(const uint16_t*)pointer & UINT16_C(0x7fff)) != 0;
        case CNP_FLOAT:
            return matrix_rank_float_is_nonzero(*(const float*)pointer);
        case CNP_DOUBLE:
            return matrix_rank_double_is_nonzero(*(const double*)pointer);
        case CNP_LONGDOUBLE:
            return matrix_rank_longdouble_is_nonzero(
                *(const long double*)pointer);
        case CNP_CFLOAT: {
            const cnp_cfloat *value = (const cnp_cfloat*)pointer;
            return matrix_rank_float_is_nonzero(value->real) ||
                matrix_rank_float_is_nonzero(value->imag);
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *value = (const cnp_cdouble*)pointer;
            return matrix_rank_double_is_nonzero(value->real) ||
                matrix_rank_double_is_nonzero(value->imag);
        }
        case CNP_CLONGDOUBLE: {
            const cnp_clongdouble *value = (const cnp_clongdouble*)pointer;
            return matrix_rank_longdouble_is_nonzero(value->real) ||
                matrix_rank_longdouble_is_nonzero(value->imag);
        }
        default:
            return false;
    }
}

CNP_API CnpArray* CNP_CALL cnp_linalg_matrix_rank(const CnpArray *a, double tol) {
    const char *function_name = "cnp_linalg_matrix_rank";
    CnpArray *tolerance = NULL;
    CnpArray *result;
    bool use_default_tolerance = !isnan(tol) && tol <= 0.0;

    if (!use_default_tolerance) {
        tolerance = cnp_array_from_scalar(tol, CNP_DOUBLE);
        if (!tolerance) {
            cnp_relabel_error(function_name);
            return NULL;
        }
    }
    result = cnp_linalg_matrix_rank_v2(a, tolerance, false);
    if (tolerance) cnp_array_free(tolerance);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_linalg_matrix_rank_v2(
    const CnpArray *a, const CnpArray *tol, bool hermitian) {
    const char *function_name = "cnp_linalg_matrix_rank_v2";
    CnpArray *singular_values = NULL;
    CnpArray *result = NULL;
    int64_t result_shape[CNP_MAXDIMS] = {0};
    int64_t result_coordinates[CNP_MAXDIMS] = {0};
    int batch_ndim;
    int result_ndim;
    int64_t matrix_rank;
    int64_t max_dimension;
    CNP_STATUS status;

    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "input array is required");
        return NULL;
    }
    if (a->ndim < 2) {
        int64_t coordinates[CNP_MAXDIMS] = {0};
        int64_t rank = 0;
        if (a->dtype->kind != 'b' && a->dtype->kind != 'i' &&
                a->dtype->kind != 'u' && a->dtype->kind != 'f' &&
                a->dtype->kind != 'c') {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "input must have a numeric dtype");
            return NULL;
        }
        for (int64_t index = 0; index < a->size; index++) {
            int64_t offset = a->offset + cnp_multi_to_offset(
                a->ndim, coordinates, a->strides);
            if (matrix_rank_element_is_nonzero(a, offset)) {
                rank = 1;
                break;
            }
            for (int dimension = a->ndim - 1;
                    dimension >= 0; dimension--) {
                coordinates[dimension]++;
                if (coordinates[dimension] < a->shape[dimension]) break;
                coordinates[dimension] = 0;
            }
        }
        result = cnp_array_from_scalar((double)rank, CNP_LONGLONG);
        if (!result) cnp_relabel_error(function_name);
        return result;
    }
    if (tol && tol->dtype->kind != 'b' && tol->dtype->kind != 'i' &&
            tol->dtype->kind != 'u' && tol->dtype->kind != 'f') {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "matrix-rank tolerance must have a real numeric dtype");
        return NULL;
    }

    status = cnp_linalg_svd_v2(
        a, false, false, hermitian,
        NULL, &singular_values, NULL);
    if (status != CNP_OK) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    matrix_rank = singular_values->shape[singular_values->ndim - 1];
    if (!tol && matrix_rank == 0) {
        cnp_array_free(singular_values);
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "zero-size array to reduction operation maximum which has no identity");
        return NULL;
    }

    batch_ndim = a->ndim - 2;
    result_ndim = tol && tol->ndim > batch_ndim ? tol->ndim : batch_ndim;
    for (int dimension = 0; dimension < result_ndim; dimension++) {
        int batch_axis = dimension - (result_ndim - batch_ndim);
        int tolerance_axis = tol
            ? dimension - (result_ndim - tol->ndim) : -1;
        int64_t batch_length = batch_axis >= 0
            ? a->shape[batch_axis] : 1;
        int64_t tolerance_length = tolerance_axis >= 0
            ? tol->shape[tolerance_axis] : 1;
        if (batch_length != tolerance_length &&
                batch_length != 1 && tolerance_length != 1) {
            cnp_array_free(singular_values);
            cnp_set_error(
                CNP_ERR_BROADCAST, function_name,
                "matrix batch shape and tolerance shape cannot be broadcast");
            return NULL;
        }
        result_shape[dimension] = batch_length == 1
            ? tolerance_length : batch_length;
    }
    result = cnp_array_new(
        result_ndim, result_shape, CNP_LONGLONG, CNP_ORDER_C);
    if (!result) {
        cnp_array_free(singular_values);
        cnp_relabel_error(function_name);
        return NULL;
    }

    max_dimension = a->shape[a->ndim - 2] > a->shape[a->ndim - 1]
        ? a->shape[a->ndim - 2] : a->shape[a->ndim - 1];
    for (int64_t output_index = 0;
            output_index < result->size; output_index++) {
        int64_t singular_offset = singular_values->offset;
        int64_t tolerance_offset = tol ? tol->offset : 0;
        int64_t remaining = output_index;
        int64_t rank = 0;
        double explicit_tolerance = 0.0;

        for (int dimension = result_ndim - 1; dimension >= 0; dimension--) {
            result_coordinates[dimension] =
                remaining % result_shape[dimension];
            remaining /= result_shape[dimension];
        }
        for (int dimension = 0; dimension < batch_ndim; dimension++) {
            int result_axis = dimension + result_ndim - batch_ndim;
            int64_t coordinate = a->shape[dimension] == 1
                ? 0 : result_coordinates[result_axis];
            singular_offset +=
                coordinate * singular_values->strides[dimension];
        }
        if (tol) {
            for (int dimension = 0; dimension < tol->ndim; dimension++) {
                int result_axis = dimension + result_ndim - tol->ndim;
                int64_t coordinate = tol->shape[dimension] == 1
                    ? 0 : result_coordinates[result_axis];
                tolerance_offset += coordinate * tol->strides[dimension];
            }
            explicit_tolerance = matrix_rank_read_real(
                tol, tolerance_offset);
        }

        if (!tol && singular_values->dtype->type_num == CNP_FLOAT) {
            float maximum = 0.0f;
            float threshold;
            for (int64_t index = 0; index < matrix_rank; index++) {
                float value = *(const float*)((const char*)
                    singular_values->data + singular_offset +
                    index * singular_values->strides[batch_ndim]);
                if (value > maximum) maximum = value;
            }
            threshold = maximum * (float)max_dimension * FLT_EPSILON;
            for (int64_t index = 0; index < matrix_rank; index++) {
                float value = *(const float*)((const char*)
                    singular_values->data + singular_offset +
                    index * singular_values->strides[batch_ndim]);
                if (value > threshold) rank++;
            }
        } else if (!tol) {
            double maximum = 0.0;
            double threshold;
            for (int64_t index = 0; index < matrix_rank; index++) {
                double value = *(const double*)((const char*)
                    singular_values->data + singular_offset +
                    index * singular_values->strides[batch_ndim]);
                if (value > maximum) maximum = value;
            }
            threshold = maximum * (double)max_dimension * DBL_EPSILON;
            for (int64_t index = 0; index < matrix_rank; index++) {
                double value = *(const double*)((const char*)
                    singular_values->data + singular_offset +
                    index * singular_values->strides[batch_ndim]);
                if (value > threshold) rank++;
            }
        } else {
            for (int64_t index = 0; index < matrix_rank; index++) {
                double value = cnp_get_element_double(
                    singular_values->data,
                    singular_offset +
                        index * singular_values->strides[batch_ndim],
                    singular_values->dtype->type_num);
                if (value > explicit_tolerance) rank++;
            }
        }
        ((int64_t*)result->data)[output_index] = rank;
    }

    cnp_array_free(singular_values);
    return result;
}

/* =========================================================================
 * Pseudo-inverse
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_linalg_pinv(const CnpArray *a, double rcond) {
    const char *function_name = "cnp_linalg_pinv";
    CnpArray *u = NULL;
    CnpArray *singular_values = NULL;
    CnpArray *vh = NULL;
    CnpArray *inverse_singular_values = NULL;
    CnpArray *u_conjugate = NULL;
    CnpArray *vh_conjugate = NULL;
    CnpArray *result = NULL;
    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array must not be null");
        return NULL;
    }
    if (a->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input must have at least two dimensions");
        return NULL;
    }
    CNP_STATUS status = cnp_linalg_svd_v2(
        a, false, true, false, &u, &singular_values, &vh);
    if (status != CNP_OK || !u || !singular_values || !vh) {
        if (vh) cnp_array_free(vh);
        if (singular_values) cnp_array_free(singular_values);
        if (u) cnp_array_free(u);
        cnp_relabel_error(function_name);
        return NULL;
    }

    inverse_singular_values = cnp_array_zeros(
        singular_values->ndim, singular_values->shape,
        singular_values->dtype->type_num, CNP_ORDER_C);
    if (!inverse_singular_values) goto cleanup;
    int64_t singular_count =
        singular_values->shape[singular_values->ndim - 1];
    int64_t batch_count = 1;
    for (int axis = 0; axis < singular_values->ndim - 1; ++axis)
        batch_count *= singular_values->shape[axis];
    for (int64_t batch = 0; batch < batch_count; ++batch) {
        double maximum = singular_count == 0 ? 0.0 :
            cnp_get_element_double(
                singular_values->data,
                singular_values->offset +
                    batch * singular_count * singular_values->dtype->elsize,
                singular_values->dtype->type_num);
        double cutoff = rcond * maximum;
        for (int64_t index = 0; index < singular_count; ++index) {
            int64_t flat_index = batch * singular_count + index;
            double value = cnp_get_element_double(
                singular_values->data,
                singular_values->offset +
                    flat_index * singular_values->dtype->elsize,
                singular_values->dtype->type_num);
            double inverse = value > cutoff ? 1.0 / value : 0.0;
            cnp_set_element_double(
                inverse_singular_values->data,
                flat_index * inverse_singular_values->dtype->elsize,
                inverse_singular_values->dtype->type_num, inverse);
        }
    }

    vh_conjugate = cnp_conjugate(vh);
    if (!vh_conjugate) goto cleanup;
    u_conjugate = cnp_conjugate(u);
    if (!u_conjugate) goto cleanup;
    const CnpArray *operands[3] = {
        vh_conjugate, inverse_singular_values, u_conjugate};
    result = cnp_einsum_generic(
        "...ki,...k,...jk->...ij", 3, operands, function_name);

cleanup:
    if (u_conjugate) cnp_array_free(u_conjugate);
    if (vh_conjugate) cnp_array_free(vh_conjugate);
    if (inverse_singular_values) cnp_array_free(inverse_singular_values);
    cnp_array_free(vh);
    cnp_array_free(singular_values);
    cnp_array_free(u);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * Matrix power
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_linalg_matrix_power(const CnpArray *a, int64_t n) {
    return cnp_matrix_power_impl(a, n, "cnp_linalg_matrix_power");
}
