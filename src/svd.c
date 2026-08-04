#include "../include/cnumpy/cnumpy.h"
#include "../include/cnumpy/cnumpy_internal.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    double real;
    double imag;
} CnpSvdComplex;

static CnpSvdComplex svd_complex(double real, double imag) {
    CnpSvdComplex value;
    value.real = real;
    value.imag = imag;
    return value;
}

static CnpSvdComplex svd_add(
    CnpSvdComplex left, CnpSvdComplex right) {
    return svd_complex(left.real + right.real, left.imag + right.imag);
}

static CnpSvdComplex svd_subtract(
    CnpSvdComplex left, CnpSvdComplex right) {
    return svd_complex(left.real - right.real, left.imag - right.imag);
}

static CnpSvdComplex svd_scale(CnpSvdComplex value, double factor) {
    return svd_complex(value.real * factor, value.imag * factor);
}

static CnpSvdComplex svd_multiply(
    CnpSvdComplex left, CnpSvdComplex right) {
    return svd_complex(
        left.real * right.real - left.imag * right.imag,
        left.real * right.imag + left.imag * right.real);
}

static CnpSvdComplex svd_conjugate(CnpSvdComplex value) {
    return svd_complex(value.real, -value.imag);
}

static double svd_absolute(CnpSvdComplex value) {
    return hypot(value.real, value.imag);
}

static bool svd_checked_product(
    int64_t left, int64_t right, int64_t *result) {
    if (left < 0 || right < 0 || !result) return false;
    if (left != 0 && right > INT64_MAX / left) return false;
    *result = left * right;
    return true;
}

static CnpSvdComplex *svd_allocate(int64_t count) {
    if (count <= 0) return NULL;
    if ((uint64_t)count > SIZE_MAX / sizeof(CnpSvdComplex)) return NULL;
    return (CnpSvdComplex*)cnp_calloc(
        (size_t)count, sizeof(CnpSvdComplex));
}

static void svd_free(CnpSvdComplex *pointer, int64_t count) {
    if (pointer) {
        cnp_free(pointer, (size_t)count * sizeof(CnpSvdComplex));
    }
}

static bool svd_type_is_supported(CNP_TYPE type) {
    return type == CNP_BOOL || cnp_type_is_integer(type) ||
        type == CNP_FLOAT || type == CNP_DOUBLE ||
        type == CNP_CFLOAT || type == CNP_CDOUBLE;
}

static CnpSvdComplex svd_read_value(
    const CnpArray *array, int64_t batch_offset,
    int64_t row, int64_t column) {
    int64_t offset = batch_offset +
        row * array->strides[array->ndim - 2] +
        column * array->strides[array->ndim - 1];
    const char *pointer = (const char*)array->data + offset;
    CNP_TYPE type = array->dtype->type_num;
    if (type == CNP_CFLOAT) {
        const cnp_cfloat *value = (const cnp_cfloat*)pointer;
        return svd_complex(value->real, value->imag);
    }
    if (type == CNP_CDOUBLE) {
        const cnp_cdouble *value = (const cnp_cdouble*)pointer;
        return svd_complex(value->real, value->imag);
    }
    return svd_complex(
        cnp_get_element_double(array->data, offset, type), 0.0);
}

static int64_t svd_batch_offset(
    const CnpArray *array, int64_t batch_index) {
    int64_t offset = array->offset;
    for (int dimension = array->ndim - 3; dimension >= 0; dimension--) {
        int64_t coordinate = batch_index % array->shape[dimension];
        batch_index /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

static void svd_set_identity(CnpSvdComplex *matrix, int64_t size) {
    if (size <= 0) return;
    memset(
        matrix, 0,
        (size_t)size * (size_t)size * sizeof(CnpSvdComplex));
    for (int64_t index = 0; index < size; index++) {
        matrix[index * size + index].real = 1.0;
    }
}

static bool svd_build_householder(
    const CnpSvdComplex *source,
    int64_t length,
    CnpSvdComplex *vector) {
    double norm = 0.0;
    double first_absolute;
    CnpSvdComplex phase;
    CnpSvdComplex alpha;
    for (int64_t index = 0; index < length; index++) {
        norm = hypot(norm, svd_absolute(source[index]));
        vector[index] = source[index];
    }
    if (norm == 0.0) return false;
    first_absolute = svd_absolute(source[0]);
    phase = first_absolute == 0.0
        ? svd_complex(1.0, 0.0)
        : svd_scale(source[0], 1.0 / first_absolute);
    alpha = svd_scale(phase, -norm);
    vector[0] = svd_subtract(vector[0], alpha);
    norm = 0.0;
    for (int64_t index = 0; index < length; index++) {
        norm = hypot(norm, svd_absolute(vector[index]));
    }
    if (norm == 0.0) return false;
    for (int64_t index = 0; index < length; index++) {
        vector[index] = svd_scale(vector[index], 1.0 / norm);
    }
    return true;
}

static void svd_apply_householder_left(
    CnpSvdComplex *matrix,
    int64_t rows,
    int64_t columns,
    int64_t row_begin,
    int64_t column_begin,
    const CnpSvdComplex *vector,
    int64_t length) {
    (void)rows;
    for (int64_t column = column_begin; column < columns; column++) {
        CnpSvdComplex product = svd_complex(0.0, 0.0);
        for (int64_t index = 0; index < length; index++) {
            product = svd_add(
                product,
                svd_multiply(
                    svd_conjugate(vector[index]),
                    matrix[(row_begin + index) * columns + column]));
        }
        product = svd_scale(product, 2.0);
        for (int64_t index = 0; index < length; index++) {
            int64_t offset = (row_begin + index) * columns + column;
            matrix[offset] = svd_subtract(
                matrix[offset], svd_multiply(vector[index], product));
        }
    }
}

static void svd_apply_householder_right(
    CnpSvdComplex *matrix,
    int64_t rows,
    int64_t columns,
    int64_t column_begin,
    const CnpSvdComplex *vector,
    int64_t length) {
    for (int64_t row = 0; row < rows; row++) {
        CnpSvdComplex product = svd_complex(0.0, 0.0);
        for (int64_t index = 0; index < length; index++) {
            product = svd_add(
                product,
                svd_multiply(
                    matrix[row * columns + column_begin + index],
                    vector[index]));
        }
        product = svd_scale(product, 2.0);
        for (int64_t index = 0; index < length; index++) {
            int64_t offset = row * columns + column_begin + index;
            matrix[offset] = svd_subtract(
                matrix[offset],
                svd_multiply(product, svd_conjugate(vector[index])));
        }
    }
}

static void svd_scale_column(
    CnpSvdComplex *matrix,
    int64_t rows,
    int64_t columns,
    int64_t column,
    CnpSvdComplex factor) {
    for (int64_t row = 0; row < rows; row++) {
        matrix[row * columns + column] = svd_multiply(
            matrix[row * columns + column], factor);
    }
}

static void svd_rotate_columns(
    CnpSvdComplex *matrix,
    int64_t rows,
    int64_t columns,
    int64_t first,
    int64_t second,
    double cosine,
    double sine) {
    for (int64_t row = 0; row < rows; row++) {
        CnpSvdComplex first_value = matrix[row * columns + first];
        CnpSvdComplex second_value = matrix[row * columns + second];
        matrix[row * columns + first] = svd_add(
            svd_scale(first_value, cosine),
            svd_scale(second_value, sine));
        matrix[row * columns + second] = svd_add(
            svd_scale(first_value, -sine),
            svd_scale(second_value, cosine));
    }
}

static void svd_swap_columns(
    CnpSvdComplex *matrix,
    int64_t rows,
    int64_t columns,
    int64_t first,
    int64_t second) {
    for (int64_t row = 0; row < rows; row++) {
        CnpSvdComplex temporary = matrix[row * columns + first];
        matrix[row * columns + first] = matrix[row * columns + second];
        matrix[row * columns + second] = temporary;
    }
}

static CNP_STATUS svd_bidiagonalize(
    CnpSvdComplex *matrix,
    int64_t rows,
    int64_t columns,
    CnpSvdComplex *left,
    CnpSvdComplex *right,
    double *diagonal,
    double *superdiagonal,
    CnpSvdComplex *vector,
    CnpSvdComplex *right_phases) {
    svd_set_identity(left, rows);
    svd_set_identity(right, columns);
    for (int64_t step = 0; step < columns; step++) {
        int64_t left_length = rows - step;
        for (int64_t index = 0; index < left_length; index++) {
            vector[index] = matrix[(step + index) * columns + step];
        }
        if (svd_build_householder(vector, left_length, vector)) {
            svd_apply_householder_left(
                matrix, rows, columns,
                step, step, vector, left_length);
            svd_apply_householder_right(
                left, rows, rows, step, vector, left_length);
        }
        for (int64_t row = step + 1; row < rows; row++) {
            matrix[row * columns + step] = svd_complex(0.0, 0.0);
        }

        if (step + 1 < columns) {
            int64_t right_length = columns - step - 1;
            for (int64_t index = 0; index < right_length; index++) {
                vector[index] = svd_conjugate(
                    matrix[step * columns + step + 1 + index]);
            }
            if (svd_build_householder(vector, right_length, vector)) {
                svd_apply_householder_right(
                    matrix, rows, columns,
                    step + 1, vector, right_length);
                svd_apply_householder_right(
                    right, columns, columns,
                    step + 1, vector, right_length);
            }
            for (int64_t column = step + 2;
                 column < columns; column++) {
                matrix[step * columns + column] =
                    svd_complex(0.0, 0.0);
            }
        }
    }

    if (columns == 0) return CNP_OK;
    right_phases[0] = svd_complex(1.0, 0.0);
    for (int64_t index = 0; index < columns; index++) {
        CnpSvdComplex d = matrix[index * columns + index];
        double d_absolute = svd_absolute(d);
        CnpSvdComplex left_phase = d_absolute == 0.0
            ? svd_complex(1.0, 0.0)
            : svd_scale(
                svd_multiply(d, right_phases[index]),
                1.0 / d_absolute);
        diagonal[index] = d_absolute;
        svd_scale_column(left, rows, rows, index, left_phase);
        svd_scale_column(
            right, columns, columns, index, right_phases[index]);
        if (index + 1 < columns) {
            CnpSvdComplex transformed_super = svd_multiply(
                svd_conjugate(left_phase),
                matrix[index * columns + index + 1]);
            double absolute = svd_absolute(transformed_super);
            superdiagonal[index] = absolute;
            right_phases[index + 1] = absolute == 0.0
                ? svd_complex(1.0, 0.0)
                : svd_scale(
                    svd_conjugate(transformed_super),
                    1.0 / absolute);
        }
    }
    superdiagonal[columns - 1] = 0.0;
    return CNP_OK;
}

static CNP_STATUS svd_diagonalize_bidiagonal(
    double *singular_values,
    double *superdiagonal,
    int64_t size,
    CnpSvdComplex *left,
    int64_t left_rows,
    CnpSvdComplex *right) {
    int64_t active = size;
    uint64_t qr_steps = 0;
    uint64_t max_qr_steps = size > 0
        ? (uint64_t)size * (uint64_t)size * UINT64_C(4096)
        : 0;
    const double tiny = DBL_MIN;
    const double epsilon = DBL_EPSILON;

    while (active > 0) {
        int64_t split;
        int64_t negligible;
        int operation;
        for (split = active - 2; split >= -1; split--) {
            if (split == -1) break;
            if (fabs(superdiagonal[split]) <= tiny + epsilon *
                    (fabs(singular_values[split]) +
                     fabs(singular_values[split + 1]))) {
                superdiagonal[split] = 0.0;
                break;
            }
        }
        if (split == active - 2) {
            operation = 4;
        } else {
            for (negligible = active - 1;
                 negligible >= split; negligible--) {
                double neighbor;
                if (negligible == split) break;
                neighbor = fabs(superdiagonal[negligible]);
                if (negligible != split + 1) {
                    neighbor += fabs(superdiagonal[negligible - 1]);
                }
                if (fabs(singular_values[negligible]) <=
                        tiny + epsilon * neighbor) {
                    singular_values[negligible] = 0.0;
                    break;
                }
            }
            if (negligible == split) {
                operation = 3;
            } else if (negligible == active - 1) {
                operation = 1;
            } else {
                operation = 2;
                split = negligible;
            }
        }
        split++;

        if (operation == 1) {
            double carry = superdiagonal[active - 2];
            superdiagonal[active - 2] = 0.0;
            for (int64_t column = active - 2;
                 column >= split; column--) {
                double magnitude = hypot(
                    singular_values[column], carry);
                double cosine = magnitude == 0.0
                    ? 1.0 : singular_values[column] / magnitude;
                double sine = magnitude == 0.0 ? 0.0 : carry / magnitude;
                singular_values[column] = magnitude;
                if (column != split) {
                    carry = -sine * superdiagonal[column - 1];
                    superdiagonal[column - 1] *= cosine;
                }
                svd_rotate_columns(
                    right, size, size,
                    column, active - 1, cosine, sine);
            }
        } else if (operation == 2) {
            double carry = superdiagonal[split - 1];
            superdiagonal[split - 1] = 0.0;
            for (int64_t column = split;
                 column < active; column++) {
                double magnitude = hypot(
                    singular_values[column], carry);
                double cosine = magnitude == 0.0
                    ? 1.0 : singular_values[column] / magnitude;
                double sine = magnitude == 0.0 ? 0.0 : carry / magnitude;
                singular_values[column] = magnitude;
                carry = -sine * superdiagonal[column];
                superdiagonal[column] *= cosine;
                svd_rotate_columns(
                    left, left_rows, left_rows,
                    column, split - 1, cosine, sine);
            }
        } else if (operation == 3) {
            double scale = fmax(
                fmax(
                    fmax(
                        fmax(fabs(singular_values[active - 1]),
                             fabs(singular_values[active - 2])),
                        fabs(superdiagonal[active - 2])),
                    fabs(singular_values[split])),
                fabs(superdiagonal[split]));
            double trailing;
            double previous;
            double trailing_super;
            double leading;
            double leading_super;
            double middle;
            double coupling;
            double shift = 0.0;
            double first;
            double second;
            if (scale == 0.0) {
                cnp_set_error(
                    CNP_ERR_CONVERGENCE, "cnp_linalg_svd",
                    "SVD QR step encountered a zero scale");
                return CNP_ERR_CONVERGENCE;
            }
            if (qr_steps++ >= max_qr_steps) {
                cnp_set_error(
                    CNP_ERR_CONVERGENCE, "cnp_linalg_svd",
                    "SVD bidiagonal QR iteration did not converge");
                return CNP_ERR_CONVERGENCE;
            }
            trailing = singular_values[active - 1] / scale;
            previous = singular_values[active - 2] / scale;
            trailing_super = superdiagonal[active - 2] / scale;
            leading = singular_values[split] / scale;
            leading_super = superdiagonal[split] / scale;
            middle = ((previous + trailing) * (previous - trailing) +
                      trailing_super * trailing_super) * 0.5;
            coupling = trailing * trailing_super;
            coupling *= coupling;
            if (middle != 0.0 || coupling != 0.0) {
                shift = hypot(middle, sqrt(coupling));
                if (middle < 0.0) shift = -shift;
                shift = coupling / (middle + shift);
            }
            first = (leading + trailing) * (leading - trailing) + shift;
            second = leading * leading_super;
            for (int64_t column = split;
                 column < active - 1; column++) {
                double magnitude = hypot(first, second);
                double cosine = magnitude == 0.0 ? 1.0 : first / magnitude;
                double sine = magnitude == 0.0 ? 0.0 : second / magnitude;
                if (column != split) {
                    superdiagonal[column - 1] = magnitude;
                }
                first = cosine * singular_values[column] +
                    sine * superdiagonal[column];
                superdiagonal[column] =
                    cosine * superdiagonal[column] -
                    sine * singular_values[column];
                second = sine * singular_values[column + 1];
                singular_values[column + 1] *= cosine;
                svd_rotate_columns(
                    right, size, size,
                    column, column + 1, cosine, sine);

                magnitude = hypot(first, second);
                cosine = magnitude == 0.0 ? 1.0 : first / magnitude;
                sine = magnitude == 0.0 ? 0.0 : second / magnitude;
                singular_values[column] = magnitude;
                first = cosine * superdiagonal[column] +
                    sine * singular_values[column + 1];
                singular_values[column + 1] =
                    -sine * superdiagonal[column] +
                    cosine * singular_values[column + 1];
                second = sine * superdiagonal[column + 1];
                superdiagonal[column + 1] *= cosine;
                svd_rotate_columns(
                    left, left_rows, left_rows,
                    column, column + 1, cosine, sine);
            }
            superdiagonal[active - 2] = first;
        } else {
            if (singular_values[split] <= 0.0) {
                singular_values[split] =
                    singular_values[split] < 0.0
                    ? -singular_values[split] : 0.0;
                svd_scale_column(
                    right, size, size, split,
                    svd_complex(-1.0, 0.0));
            }
            while (split < active - 1 &&
                    singular_values[split] <
                    singular_values[split + 1]) {
                double temporary = singular_values[split];
                singular_values[split] = singular_values[split + 1];
                singular_values[split + 1] = temporary;
                svd_swap_columns(
                    right, size, size, split, split + 1);
                svd_swap_columns(
                    left, left_rows, left_rows, split, split + 1);
                split++;
            }
            active--;
        }
    }
    return CNP_OK;
}

static CNP_STATUS svd_solve_tall(
    CnpSvdComplex *matrix,
    int64_t rows,
    int64_t columns,
    CnpSvdComplex **left_result,
    double **singular_result,
    CnpSvdComplex **right_result) {
    CnpSvdComplex *left = NULL;
    CnpSvdComplex *right = NULL;
    CnpSvdComplex *vector = NULL;
    CnpSvdComplex *right_phases = NULL;
    double *singular_values = NULL;
    double *superdiagonal = NULL;
    CNP_STATUS status = CNP_OK;
    int64_t vector_length = rows > columns ? rows : columns;
    int64_t left_count;
    int64_t right_count;

    *left_result = NULL;
    *singular_result = NULL;
    *right_result = NULL;
    if (rows < columns) {
        cnp_set_error(
            CNP_ERR_SHAPE, "cnp_linalg_svd",
            "internal SVD matrix must be tall or square");
        return CNP_ERR_SHAPE;
    }
    if (!svd_checked_product(rows, rows, &left_count) ||
            !svd_checked_product(columns, columns, &right_count)) {
        cnp_set_error(
            CNP_ERR_MEMORY, "cnp_linalg_svd",
            "SVD workspace dimensions are too large");
        return CNP_ERR_MEMORY;
    }
    if (rows > 0) left = svd_allocate(left_count);
    if (columns > 0) {
        right = svd_allocate(right_count);
        singular_values = (double*)cnp_calloc(
            (size_t)columns, sizeof(double));
        superdiagonal = (double*)cnp_calloc(
            (size_t)columns, sizeof(double));
        right_phases = svd_allocate(columns);
    }
    if (vector_length > 0) vector = svd_allocate(vector_length);
    if ((rows > 0 && !left) ||
            (columns > 0 &&
             (!right || !singular_values ||
              !superdiagonal || !right_phases)) ||
            (vector_length > 0 && !vector)) {
        cnp_set_error(
            CNP_ERR_MEMORY, "cnp_linalg_svd",
            "unable to allocate SVD workspace");
        status = CNP_ERR_MEMORY;
        goto cleanup;
    }

    status = svd_bidiagonalize(
        matrix, rows, columns, left, right,
        singular_values, superdiagonal, vector, right_phases);
    if (status != CNP_OK) goto cleanup;
    status = svd_diagonalize_bidiagonal(
        singular_values, superdiagonal, columns,
        left, rows, right);
    if (status != CNP_OK) goto cleanup;
    for (int64_t index = 0; index < columns; index++) {
        int64_t largest = index;
        for (int64_t candidate = index + 1;
             candidate < columns; candidate++) {
            if (singular_values[candidate] > singular_values[largest]) {
                largest = candidate;
            }
        }
        if (largest != index) {
            double temporary = singular_values[index];
            singular_values[index] = singular_values[largest];
            singular_values[largest] = temporary;
            svd_swap_columns(left, rows, rows, index, largest);
            svd_swap_columns(right, columns, columns, index, largest);
        }
    }

    *left_result = left;
    *singular_result = singular_values;
    *right_result = right;
    left = NULL;
    singular_values = NULL;
    right = NULL;

cleanup:
    svd_free(left, left_count);
    svd_free(right, right_count);
    if (singular_values) {
        cnp_free(singular_values, (size_t)columns * sizeof(double));
    }
    if (superdiagonal) {
        cnp_free(superdiagonal, (size_t)columns * sizeof(double));
    }
    svd_free(vector, vector_length);
    svd_free(right_phases, columns);
    return status;
}

static CNP_TYPE svd_vector_type(CNP_TYPE input_type) {
    if (input_type == CNP_CFLOAT) return CNP_CFLOAT;
    if (input_type == CNP_CDOUBLE) return CNP_CDOUBLE;
    if (input_type == CNP_FLOAT) return CNP_FLOAT;
    return CNP_DOUBLE;
}

static CNP_TYPE svd_singular_type(CNP_TYPE input_type) {
    return input_type == CNP_FLOAT || input_type == CNP_CFLOAT
        ? CNP_FLOAT : CNP_DOUBLE;
}

static void svd_write_vector_value(
    CnpArray *array, int64_t index, CnpSvdComplex value) {
    switch (array->dtype->type_num) {
        case CNP_FLOAT:
            ((float*)array->data)[index] = (float)value.real;
            break;
        case CNP_DOUBLE:
            ((double*)array->data)[index] = value.real;
            break;
        case CNP_CFLOAT:
            ((cnp_cfloat*)array->data)[index].real = (float)value.real;
            ((cnp_cfloat*)array->data)[index].imag = (float)value.imag;
            break;
        default:
            ((cnp_cdouble*)array->data)[index].real = value.real;
            ((cnp_cdouble*)array->data)[index].imag = value.imag;
            break;
    }
}

static void svd_write_singular_value(
    CnpArray *array, int64_t index, double value) {
    if (array->dtype->type_num == CNP_FLOAT) {
        ((float*)array->data)[index] = (float)value;
    } else {
        ((double*)array->data)[index] = value;
    }
}

static CNP_STATUS svd_solve_and_store_batch(
    const CnpArray *source,
    int64_t batch_offset,
    int64_t batch,
    int64_t rows,
    int64_t columns,
    bool full_matrices,
    bool compute_uv,
    bool hermitian,
    CnpArray *u,
    CnpArray *s,
    CnpArray *vh) {
    int64_t rank = rows < columns ? rows : columns;
    int64_t u_columns = full_matrices ? rows : rank;
    int64_t vh_rows = full_matrices ? columns : rank;
    int64_t tall_rows = rows >= columns ? rows : columns;
    int64_t tall_columns = rank;
    CnpSvdComplex *matrix = NULL;
    CnpSvdComplex *left = NULL;
    CnpSvdComplex *right = NULL;
    double *singular_values = NULL;
    CNP_STATUS status = CNP_OK;
    int64_t matrix_count;
    int64_t left_count;
    int64_t right_count;

    if (!svd_checked_product(tall_rows, tall_columns, &matrix_count) ||
            !svd_checked_product(tall_rows, tall_rows, &left_count) ||
            !svd_checked_product(
                tall_columns, tall_columns, &right_count)) {
        cnp_set_error(
            CNP_ERR_MEMORY, "cnp_linalg_svd",
            "SVD workspace dimensions are too large");
        return CNP_ERR_MEMORY;
    }
    if (matrix_count > 0) {
        matrix = svd_allocate(matrix_count);
        if (!matrix) {
            cnp_set_error(
                CNP_ERR_MEMORY, "cnp_linalg_svd",
                "unable to allocate SVD matrix workspace");
            return CNP_ERR_MEMORY;
        }
    }
    if (rows >= columns) {
        for (int64_t row = 0; row < rows; row++) {
            for (int64_t column = 0; column < columns; column++) {
                CnpSvdComplex value;
                if (hermitian && row < column) {
                    value = svd_conjugate(svd_read_value(
                        source, batch_offset, column, row));
                } else {
                    value = svd_read_value(
                        source, batch_offset, row, column);
                    if (hermitian && row == column) value.imag = 0.0;
                }
                if (!isfinite(value.real) || !isfinite(value.imag)) {
                    cnp_set_error(
                        CNP_ERR_CONVERGENCE, "cnp_linalg_svd",
                        "input must not contain NaN or infinity");
                    status = CNP_ERR_CONVERGENCE;
                    goto cleanup;
                }
                matrix[row * columns + column] = value;
            }
        }
    } else {
        for (int64_t row = 0; row < columns; row++) {
            for (int64_t column = 0; column < rows; column++) {
                CnpSvdComplex value = svd_read_value(
                    source, batch_offset, column, row);
                if (!isfinite(value.real) || !isfinite(value.imag)) {
                    cnp_set_error(
                        CNP_ERR_CONVERGENCE, "cnp_linalg_svd",
                        "input must not contain NaN or infinity");
                    status = CNP_ERR_CONVERGENCE;
                    goto cleanup;
                }
                matrix[row * rows + column] = svd_conjugate(value);
            }
        }
    }

    status = svd_solve_tall(
        matrix, tall_rows, tall_columns,
        &left, &singular_values, &right);
    if (status != CNP_OK) goto cleanup;
    for (int64_t index = 0; index < rank; index++) {
        svd_write_singular_value(
            s, batch * rank + index, singular_values[index]);
    }
    if (!compute_uv) {
        goto cleanup;
    }
    if (rows >= columns) {
        for (int64_t row = 0; row < rows; row++) {
            for (int64_t column = 0; column < u_columns; column++) {
                svd_write_vector_value(
                    u,
                    batch * rows * u_columns + row * u_columns + column,
                    left[row * rows + column]);
            }
        }
        for (int64_t row = 0; row < vh_rows; row++) {
            for (int64_t column = 0; column < columns; column++) {
                svd_write_vector_value(
                    vh,
                    batch * vh_rows * columns + row * columns + column,
                    svd_conjugate(right[column * columns + row]));
            }
        }
    } else {
        for (int64_t row = 0; row < rows; row++) {
            for (int64_t column = 0; column < u_columns; column++) {
                svd_write_vector_value(
                    u,
                    batch * rows * u_columns + row * u_columns + column,
                    right[row * rows + column]);
            }
        }
        for (int64_t row = 0; row < vh_rows; row++) {
            for (int64_t column = 0; column < columns; column++) {
                svd_write_vector_value(
                    vh,
                    batch * vh_rows * columns + row * columns + column,
                    svd_conjugate(left[column * columns + row]));
            }
        }
    }

cleanup:
    svd_free(matrix, matrix_count);
    svd_free(left, left_count);
    svd_free(right, right_count);
    if (singular_values) {
        cnp_free(
            singular_values, (size_t)tall_columns * sizeof(double));
    }
    return status;
}

static CNP_STATUS svd_execute(
    const CnpArray *a,
    bool full_matrices,
    bool compute_uv,
    bool hermitian,
    CnpArray **u,
    CnpArray **s,
    CnpArray **vh,
    const char *function_name) {
    int64_t u_shape[CNP_MAXDIMS];
    int64_t s_shape[CNP_MAXDIMS];
    int64_t vh_shape[CNP_MAXDIMS];
    int64_t batch_count = 1;
    int64_t rows;
    int64_t columns;
    int64_t rank;
    int64_t u_columns;
    int64_t vh_rows;
    CnpArray *u_result = NULL;
    CnpArray *s_result = NULL;
    CnpArray *vh_result = NULL;
    CNP_STATUS status = CNP_OK;

    if (u) *u = NULL;
    if (s) *s = NULL;
    if (vh) *vh = NULL;
    if (!s || (compute_uv && (!u || !vh)) ||
            (u && u == s) || (u && vh && u == vh) ||
            (vh && vh == s)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            compute_uv
                ? "distinct U, singular value, and Vh output pointers are required"
                : "a distinct singular value output pointer is required");
        return CNP_ERR_GENERIC;
    }
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
    if (!svd_type_is_supported(a->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input dtype is not supported by linear algebra");
        return CNP_ERR_TYPE;
    }

    rows = a->shape[a->ndim - 2];
    columns = a->shape[a->ndim - 1];
    rank = rows < columns ? rows : columns;
    u_columns = full_matrices ? rows : rank;
    vh_rows = full_matrices ? columns : rank;
    if (hermitian && rows != columns) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "hermitian SVD requires a square input matrix");
        return CNP_ERR_SHAPE;
    }
    for (int dimension = 0; dimension < a->ndim - 2; dimension++) {
        int64_t length = a->shape[dimension];
        if (length != 0 && batch_count > INT64_MAX / length) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "batch shape is too large");
            return CNP_ERR_SHAPE;
        }
        batch_count *= length;
        u_shape[dimension] = length;
        s_shape[dimension] = length;
        vh_shape[dimension] = length;
    }
    u_shape[a->ndim - 2] = rows;
    u_shape[a->ndim - 1] = u_columns;
    s_shape[a->ndim - 2] = rank;
    vh_shape[a->ndim - 2] = vh_rows;
    vh_shape[a->ndim - 1] = columns;

    if (compute_uv) {
        u_result = cnp_array_new(
            a->ndim, u_shape,
            svd_vector_type(a->dtype->type_num), CNP_ORDER_C);
        if (!u_result) {
            cnp_relabel_error(function_name);
            return CNP_ERR_MEMORY;
        }
    }
    s_result = cnp_array_new(
        a->ndim - 1, s_shape,
        svd_singular_type(a->dtype->type_num), CNP_ORDER_C);
    if (!s_result) {
        if (u_result) cnp_array_free(u_result);
        cnp_relabel_error(function_name);
        return CNP_ERR_MEMORY;
    }
    if (compute_uv) {
        vh_result = cnp_array_new(
            a->ndim, vh_shape,
            svd_vector_type(a->dtype->type_num), CNP_ORDER_C);
        if (!vh_result) {
            cnp_array_free(u_result);
            cnp_array_free(s_result);
            cnp_relabel_error(function_name);
            return CNP_ERR_MEMORY;
        }
    }

    for (int64_t batch = 0; batch < batch_count; batch++) {
        status = svd_solve_and_store_batch(
            a, svd_batch_offset(a, batch), batch,
            rows, columns, full_matrices, compute_uv, hermitian,
            u_result, s_result, vh_result);
        if (status != CNP_OK) {
            if (u_result) cnp_array_free(u_result);
            cnp_array_free(s_result);
            if (vh_result) cnp_array_free(vh_result);
            cnp_relabel_error(function_name);
            return status;
        }
    }
    if (u) *u = u_result;
    *s = s_result;
    if (vh) *vh = vh_result;
    return CNP_OK;
}

CNP_API CNP_STATUS CNP_CALL cnp_linalg_svd_v2(
    const CnpArray *a,
    bool full_matrices,
    bool compute_uv,
    bool hermitian,
    CnpArray **u,
    CnpArray **s,
    CnpArray **vh) {
    return svd_execute(
        a, full_matrices, compute_uv, hermitian,
        u, s, vh, "cnp_linalg_svd_v2");
}

CNP_API CNP_STATUS CNP_CALL cnp_linalg_svd(
    const CnpArray *a,
    CnpArray **u,
    CnpArray **s,
    CnpArray **vh) {
    return svd_execute(
        a, true, true, false,
        u, s, vh, "cnp_linalg_svd");
}
