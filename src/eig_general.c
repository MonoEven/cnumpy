#include "../include/cnumpy/cnumpy.h"
#include "../include/cnumpy/cnumpy_internal.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    double real;
    double imag;
} CnpEigComplex;

static CnpEigComplex eig_complex(double real, double imag) {
    CnpEigComplex value;
    value.real = real;
    value.imag = imag;
    return value;
}

static CnpEigComplex eig_add(CnpEigComplex left, CnpEigComplex right) {
    return eig_complex(left.real + right.real, left.imag + right.imag);
}

static CnpEigComplex eig_subtract(
    CnpEigComplex left, CnpEigComplex right) {
    return eig_complex(left.real - right.real, left.imag - right.imag);
}

static CnpEigComplex eig_scale(CnpEigComplex value, double factor) {
    return eig_complex(value.real * factor, value.imag * factor);
}

static CnpEigComplex eig_multiply(
    CnpEigComplex left, CnpEigComplex right) {
    return eig_complex(
        left.real * right.real - left.imag * right.imag,
        left.real * right.imag + left.imag * right.real);
}

static CnpEigComplex eig_conjugate(CnpEigComplex value) {
    return eig_complex(value.real, -value.imag);
}

static double eig_absolute(CnpEigComplex value) {
    return hypot(value.real, value.imag);
}

static CnpEigComplex eig_divide(
    CnpEigComplex numerator, CnpEigComplex denominator) {
    double real_abs = fabs(denominator.real);
    double imag_abs = fabs(denominator.imag);
    if (real_abs >= imag_abs) {
        double ratio = denominator.imag / denominator.real;
        double scale = denominator.real + denominator.imag * ratio;
        return eig_complex(
            (numerator.real + numerator.imag * ratio) / scale,
            (numerator.imag - numerator.real * ratio) / scale);
    }
    {
        double ratio = denominator.real / denominator.imag;
        double scale = denominator.real * ratio + denominator.imag;
        return eig_complex(
            (numerator.real * ratio + numerator.imag) / scale,
            (numerator.imag * ratio - numerator.real) / scale);
    }
}

static CnpEigComplex eig_sqrt(CnpEigComplex value) {
    double magnitude = eig_absolute(value);
    double real = sqrt(fmax(0.0, 0.5 * (magnitude + value.real)));
    double imag = sqrt(fmax(0.0, 0.5 * (magnitude - value.real)));
    if (value.imag < 0.0) imag = -imag;
    return eig_complex(real, imag);
}

static CnpEigComplex *eig_allocate_square(int64_t n) {
    size_t count;
    if (n <= 0) return NULL;
    if ((uint64_t)n > SIZE_MAX / (uint64_t)n) return NULL;
    count = (size_t)n * (size_t)n;
    if (count > SIZE_MAX / sizeof(CnpEigComplex)) return NULL;
    return (CnpEigComplex*)cnp_calloc(count, sizeof(CnpEigComplex));
}

static void eig_free_square(CnpEigComplex *matrix, int64_t n) {
    if (matrix) {
        cnp_free(
            matrix,
            (size_t)n * (size_t)n * sizeof(CnpEigComplex));
    }
}

static CnpEigComplex eig_read_value(
    const CnpArray *array, int64_t batch_offset,
    int64_t row, int64_t column) {
    int64_t offset = batch_offset + row * array->strides[array->ndim - 2] +
        column * array->strides[array->ndim - 1];
    const char *pointer = (const char*)array->data + offset;
    CNP_TYPE type = array->dtype->type_num;
    if (type == CNP_CFLOAT) {
        const cnp_cfloat *value = (const cnp_cfloat*)pointer;
        return eig_complex(value->real, value->imag);
    }
    if (type == CNP_CDOUBLE) {
        const cnp_cdouble *value = (const cnp_cdouble*)pointer;
        return eig_complex(value->real, value->imag);
    }
    return eig_complex(
        cnp_get_element_double(array->data, offset, type), 0.0);
}

static bool eig_type_is_supported(CNP_TYPE type) {
    return type == CNP_BOOL || cnp_type_is_integer(type) ||
        type == CNP_FLOAT || type == CNP_DOUBLE ||
        type == CNP_CFLOAT || type == CNP_CDOUBLE;
}

static bool eig_build_householder(
    const CnpEigComplex *source, int64_t length, CnpEigComplex *vector) {
    double norm = 0.0;
    double first_absolute;
    CnpEigComplex phase;
    CnpEigComplex alpha;
    for (int64_t index = 0; index < length; index++) {
        norm = hypot(norm, eig_absolute(source[index]));
        vector[index] = source[index];
    }
    if (norm == 0.0) return false;
    first_absolute = eig_absolute(source[0]);
    phase = first_absolute == 0.0
        ? eig_complex(1.0, 0.0)
        : eig_scale(source[0], 1.0 / first_absolute);
    alpha = eig_scale(phase, -norm);
    vector[0] = eig_subtract(vector[0], alpha);
    norm = 0.0;
    for (int64_t index = 0; index < length; index++) {
        norm = hypot(norm, eig_absolute(vector[index]));
    }
    if (norm == 0.0) return false;
    for (int64_t index = 0; index < length; index++) {
        vector[index] = eig_scale(vector[index], 1.0 / norm);
    }
    return true;
}

static void eig_apply_householder_left(
    CnpEigComplex *matrix, int64_t rows, int64_t columns,
    int64_t row_begin, int64_t column_begin,
    const CnpEigComplex *vector, int64_t length) {
    (void)rows;
    for (int64_t column = column_begin; column < columns; column++) {
        CnpEigComplex product = eig_complex(0.0, 0.0);
        for (int64_t index = 0; index < length; index++) {
            product = eig_add(
                product,
                eig_multiply(
                    eig_conjugate(vector[index]),
                    matrix[(row_begin + index) * columns + column]));
        }
        product = eig_scale(product, 2.0);
        for (int64_t index = 0; index < length; index++) {
            int64_t offset = (row_begin + index) * columns + column;
            matrix[offset] = eig_subtract(
                matrix[offset], eig_multiply(vector[index], product));
        }
    }
}

static void eig_apply_householder_right(
    CnpEigComplex *matrix, int64_t rows, int64_t columns,
    int64_t column_begin,
    const CnpEigComplex *vector, int64_t length) {
    for (int64_t row = 0; row < rows; row++) {
        CnpEigComplex product = eig_complex(0.0, 0.0);
        for (int64_t index = 0; index < length; index++) {
            product = eig_add(
                product,
                eig_multiply(
                    matrix[row * columns + column_begin + index],
                    vector[index]));
        }
        product = eig_scale(product, 2.0);
        for (int64_t index = 0; index < length; index++) {
            int64_t offset = row * columns + column_begin + index;
            matrix[offset] = eig_subtract(
                matrix[offset],
                eig_multiply(product, eig_conjugate(vector[index])));
        }
    }
}

static void eig_set_identity(CnpEigComplex *matrix, int64_t n) {
    memset(matrix, 0, (size_t)n * (size_t)n * sizeof(CnpEigComplex));
    for (int64_t index = 0; index < n; index++) {
        matrix[index * n + index].real = 1.0;
    }
}

static bool eig_reduce_to_hessenberg(
    CnpEigComplex *matrix, CnpEigComplex *transform,
    int64_t n, CnpEigComplex *vector) {
    if (transform) eig_set_identity(transform, n);
    for (int64_t column = 0; column + 2 < n; column++) {
        int64_t start = column + 1;
        int64_t length = n - start;
        for (int64_t index = 0; index < length; index++) {
            vector[index] = matrix[(start + index) * n + column];
        }
        if (!eig_build_householder(vector, length, vector)) continue;
        eig_apply_householder_left(
            matrix, n, n, start, column, vector, length);
        eig_apply_householder_right(
            matrix, n, n, start, vector, length);
        if (transform) {
            eig_apply_householder_right(
                transform, n, n, start, vector, length);
        }
        for (int64_t row = column + 2; row < n; row++) {
            matrix[row * n + column] = eig_complex(0.0, 0.0);
        }
    }
    return true;
}

static bool eig_qr_factor(
    const CnpEigComplex *source, int64_t n,
    CnpEigComplex *q, CnpEigComplex *r,
    CnpEigComplex *vector) {
    memcpy(r, source, (size_t)n * (size_t)n * sizeof(CnpEigComplex));
    eig_set_identity(q, n);
    for (int64_t column = 0; column < n; column++) {
        int64_t length = n - column;
        for (int64_t index = 0; index < length; index++) {
            vector[index] = r[(column + index) * n + column];
        }
        if (!eig_build_householder(vector, length, vector)) continue;
        eig_apply_householder_left(
            r, n, n, column, column, vector, length);
        eig_apply_householder_right(
            q, n, n, column, vector, length);
        for (int64_t row = column + 1; row < n; row++) {
            r[row * n + column] = eig_complex(0.0, 0.0);
        }
    }
    return true;
}

static CnpEigComplex eig_trailing_shift(
    const CnpEigComplex *matrix, int64_t stride, int64_t active) {
    CnpEigComplex a = matrix[(active - 2) * stride + active - 2];
    CnpEigComplex b = matrix[(active - 2) * stride + active - 1];
    CnpEigComplex c = matrix[(active - 1) * stride + active - 2];
    CnpEigComplex d = matrix[(active - 1) * stride + active - 1];
    CnpEigComplex half_trace = eig_scale(eig_add(a, d), 0.5);
    CnpEigComplex half_difference = eig_scale(eig_subtract(a, d), 0.5);
    CnpEigComplex discriminant = eig_sqrt(eig_add(
        eig_multiply(half_difference, half_difference),
        eig_multiply(b, c)));
    CnpEigComplex first = eig_add(half_trace, discriminant);
    CnpEigComplex second = eig_subtract(half_trace, discriminant);
    return eig_absolute(eig_subtract(first, d)) <=
            eig_absolute(eig_subtract(second, d))
        ? first : second;
}

static void eig_multiply_active(
    const CnpEigComplex *left, const CnpEigComplex *right,
    CnpEigComplex *result, int64_t n) {
    for (int64_t row = 0; row < n; row++) {
        for (int64_t column = 0; column < n; column++) {
            CnpEigComplex value = eig_complex(0.0, 0.0);
            for (int64_t inner = 0; inner < n; inner++) {
                value = eig_add(
                    value,
                    eig_multiply(
                        left[row * n + inner],
                        right[inner * n + column]));
            }
            result[row * n + column] = value;
        }
    }
}

static CNP_STATUS eig_schur_decomposition(
    const CnpEigComplex *source, int64_t n,
    CnpEigComplex *schur, CnpEigComplex *transform) {
    CnpEigComplex *q = NULL;
    CnpEigComplex *r = NULL;
    CnpEigComplex *shifted = NULL;
    CnpEigComplex *work = NULL;
    CnpEigComplex *vector = NULL;
    CnpEigComplex *row_work = NULL;
    int64_t active = n;
    uint64_t iterations = 0;
    uint64_t max_iterations;
    CNP_STATUS status = CNP_OK;

    memcpy(
        schur, source,
        (size_t)n * (size_t)n * sizeof(CnpEigComplex));
    if (n == 0) return CNP_OK;
    vector = (CnpEigComplex*)cnp_calloc(
        (size_t)n, sizeof(CnpEigComplex));
    if (transform) {
        row_work = (CnpEigComplex*)cnp_calloc(
            (size_t)n, sizeof(CnpEigComplex));
    }
    q = eig_allocate_square(n);
    r = eig_allocate_square(n);
    shifted = eig_allocate_square(n);
    work = eig_allocate_square(n);
    if (!vector || (transform && !row_work) ||
            !q || !r || !shifted || !work) {
        cnp_set_error(
            CNP_ERR_MEMORY, "cnp_linalg_eig",
            "unable to allocate general eig workspace");
        status = CNP_ERR_MEMORY;
        goto cleanup;
    }

    eig_reduce_to_hessenberg(schur, transform, n, vector);
    max_iterations = (uint64_t)n * (uint64_t)n * UINT64_C(4096);
    while (active > 1) {
        CnpEigComplex diagonal_above =
            schur[(active - 2) * n + active - 2];
        CnpEigComplex diagonal = schur[(active - 1) * n + active - 1];
        CnpEigComplex subdiagonal = schur[(active - 1) * n + active - 2];
        double diagonal_scale =
            eig_absolute(diagonal_above) + eig_absolute(diagonal);
        double threshold = fmax(
            DBL_MIN, 128.0 * DBL_EPSILON * diagonal_scale);
        if (eig_absolute(subdiagonal) <= threshold) {
            schur[(active - 1) * n + active - 2] =
                eig_complex(0.0, 0.0);
            active--;
            continue;
        }
        if (iterations++ >= max_iterations) {
            cnp_set_error(
                CNP_ERR_CONVERGENCE, "cnp_linalg_eig",
                "general eig QR iteration did not converge");
            status = CNP_ERR_CONVERGENCE;
            goto cleanup;
        }

        {
            CnpEigComplex shift = eig_trailing_shift(schur, n, active);
            memset(
                shifted, 0,
                (size_t)active * (size_t)active * sizeof(CnpEigComplex));
            for (int64_t row = 0; row < active; row++) {
                for (int64_t column = 0; column < active; column++) {
                    shifted[row * active + column] =
                        schur[row * n + column];
                }
                shifted[row * active + row] = eig_subtract(
                    shifted[row * active + row], shift);
            }
            eig_qr_factor(shifted, active, q, r, vector);
            eig_multiply_active(r, q, work, active);
            for (int64_t row = 0; row < active; row++) {
                for (int64_t column = 0; column < active; column++) {
                    CnpEigComplex value = work[row * active + column];
                    if (row == column) value = eig_add(value, shift);
                    schur[row * n + column] = value;
                }
            }
            for (int64_t column = active; column < n; column++) {
                for (int64_t row = 0; row < active; row++) {
                    CnpEigComplex value = eig_complex(0.0, 0.0);
                    for (int64_t inner = 0; inner < active; inner++) {
                        value = eig_add(
                            value,
                            eig_multiply(
                                eig_conjugate(q[inner * active + row]),
                                schur[inner * n + column]));
                    }
                    work[row * n + column] = value;
                }
                for (int64_t row = 0; row < active; row++) {
                    schur[row * n + column] = work[row * n + column];
                }
            }
            if (transform) {
                for (int64_t row = 0; row < n; row++) {
                    for (int64_t column = 0; column < active; column++) {
                        CnpEigComplex value = eig_complex(0.0, 0.0);
                        for (int64_t inner = 0; inner < active; inner++) {
                            value = eig_add(
                                value,
                                eig_multiply(
                                    transform[row * n + inner],
                                    q[inner * active + column]));
                        }
                        row_work[column] = value;
                    }
                    for (int64_t column = 0; column < active; column++) {
                        transform[row * n + column] = row_work[column];
                    }
                }
            }
            for (int64_t row = 2; row < active; row++) {
                for (int64_t column = 0; column + 1 < row; column++) {
                    schur[row * n + column] = eig_complex(0.0, 0.0);
                }
            }
        }
    }

cleanup:
    if (vector) cnp_free(vector, (size_t)n * sizeof(CnpEigComplex));
    if (row_work) cnp_free(row_work, (size_t)n * sizeof(CnpEigComplex));
    eig_free_square(q, n);
    eig_free_square(r, n);
    eig_free_square(shifted, n);
    eig_free_square(work, n);
    return status;
}

static CNP_STATUS eig_schur_eigenvectors(
    const CnpEigComplex *original,
    const CnpEigComplex *schur,
    const CnpEigComplex *transform,
    int64_t n,
    CnpEigComplex *eigenvalues,
    CnpEigComplex *eigenvectors) {
    CnpEigComplex *triangular_vector = NULL;
    CnpEigComplex *vector = NULL;
    double matrix_norm = 0.0;
    double source_norm = 0.0;
    CNP_STATUS status = CNP_OK;
    if (n == 0) return CNP_OK;
    triangular_vector = (CnpEigComplex*)cnp_calloc(
        (size_t)n, sizeof(CnpEigComplex));
    vector = (CnpEigComplex*)cnp_calloc(
        (size_t)n, sizeof(CnpEigComplex));
    if (!triangular_vector || !vector) {
        cnp_set_error(
            CNP_ERR_MEMORY, "cnp_linalg_eig",
            "unable to allocate eigenvector workspace");
        status = CNP_ERR_MEMORY;
        goto cleanup;
    }
    for (int64_t row = 0; row < n; row++) {
        double schur_row_norm = 0.0;
        double source_row_norm = 0.0;
        for (int64_t column = 0; column < n; column++) {
            schur_row_norm += eig_absolute(schur[row * n + column]);
            source_row_norm += eig_absolute(original[row * n + column]);
        }
        matrix_norm = fmax(matrix_norm, schur_row_norm);
        source_norm = fmax(source_norm, source_row_norm);
        eigenvalues[row] = schur[row * n + row];
    }

    for (int64_t eigen_index = 0; eigen_index < n; eigen_index++) {
        CnpEigComplex eigenvalue = eigenvalues[eigen_index];
        double norm = 0.0;
        double residual = 0.0;
        memset(
            triangular_vector, 0,
            (size_t)n * sizeof(CnpEigComplex));
        triangular_vector[eigen_index] = eig_complex(1.0, 0.0);
        for (int64_t row = eigen_index; row-- > 0;) {
            CnpEigComplex sum = eig_complex(0.0, 0.0);
            CnpEigComplex denominator = eig_subtract(
                schur[row * n + row], eigenvalue);
            double separation = 128.0 * DBL_EPSILON *
                (matrix_norm + eig_absolute(eigenvalue));
            for (int64_t column = row + 1;
                 column <= eigen_index; column++) {
                sum = eig_add(
                    sum,
                    eig_multiply(
                        schur[row * n + column],
                        triangular_vector[column]));
            }
            if (matrix_norm == 0.0 &&
                    eig_absolute(eigenvalue) == 0.0 &&
                    eig_absolute(denominator) == 0.0 &&
                    eig_absolute(sum) == 0.0) {
                /* Every coordinate is free in the exact-zero eigenspace.
                 * Keep earlier coordinates zero to construct the canonical
                 * basis instead of evaluating the indeterminate 0 / 0. */
                triangular_vector[row] = eig_complex(0.0, 0.0);
                continue;
            }
            if (eig_absolute(denominator) < separation) {
                double absolute = eig_absolute(denominator);
                denominator = absolute == 0.0
                    ? eig_complex(separation, 0.0)
                    : eig_scale(denominator, separation / absolute);
            }
            triangular_vector[row] = eig_scale(
                eig_divide(sum, denominator), -1.0);
        }

        for (int64_t row = 0; row < n; row++) {
            CnpEigComplex value = eig_complex(0.0, 0.0);
            for (int64_t inner = 0; inner < n; inner++) {
                value = eig_add(
                    value,
                    eig_multiply(
                        transform[row * n + inner],
                        triangular_vector[inner]));
            }
            vector[row] = value;
            norm = hypot(norm, eig_absolute(value));
        }
        if (norm == 0.0 || !isfinite(norm)) {
            cnp_set_error(
                CNP_ERR_CONVERGENCE, "cnp_linalg_eig",
                "general eig produced an invalid eigenvector");
            status = CNP_ERR_CONVERGENCE;
            goto cleanup;
        }
        for (int64_t row = 0; row < n; row++) {
            vector[row] = eig_scale(vector[row], 1.0 / norm);
            eigenvectors[row * n + eigen_index] = vector[row];
        }
        for (int64_t row = 0; row < n; row++) {
            CnpEigComplex value = eig_scale(
                eig_multiply(eigenvalue, vector[row]), -1.0);
            for (int64_t column = 0; column < n; column++) {
                value = eig_add(
                    value,
                    eig_multiply(
                        original[row * n + column], vector[column]));
            }
            residual = hypot(residual, eig_absolute(value));
        }
        if (residual > 8192.0 * DBL_EPSILON * (double)n *
                (source_norm + eig_absolute(eigenvalue))) {
            cnp_set_error(
                CNP_ERR_CONVERGENCE, "cnp_linalg_eig",
                "general eig eigenvector residual exceeded tolerance");
            status = CNP_ERR_CONVERGENCE;
            goto cleanup;
        }
    }

cleanup:
    if (triangular_vector) {
        cnp_free(
            triangular_vector, (size_t)n * sizeof(CnpEigComplex));
    }
    if (vector) cnp_free(vector, (size_t)n * sizeof(CnpEigComplex));
    return status;
}

static CNP_STATUS eig_write_results(
    const CnpEigComplex *values,
    const CnpEigComplex *vectors,
    int64_t value_count,
    int64_t vector_count,
    int value_ndim,
    const int64_t *value_shape,
    int vector_ndim,
    const int64_t *vector_shape,
    CNP_TYPE input_type,
    bool force_complex,
    bool compute_vectors,
    CnpArray **eigenvalues,
    CnpArray **eigenvectors) {
    bool complex_result = force_complex || input_type == CNP_CFLOAT ||
        input_type == CNP_CDOUBLE;
    bool single_precision = input_type == CNP_FLOAT ||
        input_type == CNP_CFLOAT;
    double value_scale = 0.0;
    CNP_TYPE result_type;
    for (int64_t index = 0; index < value_count; index++) {
        value_scale = fmax(value_scale, eig_absolute(values[index]));
    }
    if (!complex_result) {
        double tolerance = fmax(
            DBL_MIN, 4096.0 * DBL_EPSILON * value_scale);
        for (int64_t index = 0; index < value_count; index++) {
            if (fabs(values[index].imag) > tolerance) {
                complex_result = true;
                break;
            }
        }
    }
    if (complex_result) {
        result_type = single_precision ? CNP_CFLOAT : CNP_CDOUBLE;
    } else {
        result_type = single_precision ? CNP_FLOAT : CNP_DOUBLE;
    }
    CnpArray *value_result = cnp_array_new(
        value_ndim, value_shape, result_type, CNP_ORDER_C);
    CnpArray *vector_result = NULL;
    if (!value_result) {
        cnp_relabel_error("cnp_linalg_eig");
        return CNP_ERR_MEMORY;
    }
    if (compute_vectors) {
        vector_result = cnp_array_new(
            vector_ndim, vector_shape, result_type, CNP_ORDER_C);
        if (!vector_result) {
            cnp_array_free(value_result);
            cnp_relabel_error("cnp_linalg_eig");
            return CNP_ERR_MEMORY;
        }
    }
    if (result_type == CNP_CDOUBLE) {
        for (int64_t index = 0; index < value_count; index++) {
            ((cnp_cdouble*)value_result->data)[index].real = values[index].real;
            ((cnp_cdouble*)value_result->data)[index].imag = values[index].imag;
        }
        for (int64_t index = 0; index < vector_count; index++) {
            ((cnp_cdouble*)vector_result->data)[index].real = vectors[index].real;
            ((cnp_cdouble*)vector_result->data)[index].imag = vectors[index].imag;
        }
    } else if (result_type == CNP_CFLOAT) {
        for (int64_t index = 0; index < value_count; index++) {
            ((cnp_cfloat*)value_result->data)[index].real =
                (float)values[index].real;
            ((cnp_cfloat*)value_result->data)[index].imag =
                (float)values[index].imag;
        }
        for (int64_t index = 0; index < vector_count; index++) {
            ((cnp_cfloat*)vector_result->data)[index].real =
                (float)vectors[index].real;
            ((cnp_cfloat*)vector_result->data)[index].imag =
                (float)vectors[index].imag;
        }
    } else if (result_type == CNP_FLOAT) {
        for (int64_t index = 0; index < value_count; index++) {
            ((float*)value_result->data)[index] = (float)values[index].real;
        }
        for (int64_t index = 0; index < vector_count; index++) {
            ((float*)vector_result->data)[index] = (float)vectors[index].real;
        }
    } else {
        for (int64_t index = 0; index < value_count; index++) {
            ((double*)value_result->data)[index] = values[index].real;
        }
        for (int64_t index = 0; index < vector_count; index++) {
            ((double*)vector_result->data)[index] = vectors[index].real;
        }
    }
    *eigenvalues = value_result;
    if (compute_vectors) *eigenvectors = vector_result;
    return CNP_OK;
}

static int64_t eig_batch_offset(
    const CnpArray *array, int64_t batch_index) {
    int64_t offset = array->offset;
    for (int dimension = array->ndim - 3; dimension >= 0; dimension--) {
        int64_t coordinate = batch_index % array->shape[dimension];
        batch_index /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

static CNP_STATUS eig_solve_matrix(
    const CnpArray *array,
    int64_t batch_offset,
    int64_t n,
    CnpEigComplex *values,
    CnpEigComplex *vectors,
    bool compute_vectors) {
    CnpEigComplex *source = NULL;
    CnpEigComplex *schur = NULL;
    CnpEigComplex *transform = NULL;
    CNP_STATUS status = CNP_OK;
    if (n == 0) return CNP_OK;
    source = eig_allocate_square(n);
    schur = eig_allocate_square(n);
    if (compute_vectors) transform = eig_allocate_square(n);
    if (!source || !schur || (compute_vectors && !transform)) {
        cnp_set_error(
            CNP_ERR_MEMORY, "cnp_linalg_eig",
            "unable to allocate general eig workspace");
        status = CNP_ERR_MEMORY;
        goto cleanup;
    }
    for (int64_t row = 0; row < n; row++) {
        for (int64_t column = 0; column < n; column++) {
            CnpEigComplex value = eig_read_value(
                array, batch_offset, row, column);
            if (!isfinite(value.real) || !isfinite(value.imag)) {
                cnp_set_error(
                    CNP_ERR_CONVERGENCE, "cnp_linalg_eig",
                    "input must not contain NaN or infinity");
                status = CNP_ERR_CONVERGENCE;
                goto cleanup;
            }
            source[row * n + column] = value;
        }
    }

    status = eig_schur_decomposition(source, n, schur, transform);
    if (status != CNP_OK) goto cleanup;
    if (compute_vectors) {
        status = eig_schur_eigenvectors(
            source, schur, transform, n, values, vectors);
    } else {
        for (int64_t index = 0; index < n; ++index) {
            values[index] = schur[index * n + index];
        }
    }

cleanup:
    eig_free_square(source, n);
    eig_free_square(schur, n);
    eig_free_square(transform, n);
    return status;
}

static CNP_STATUS eig_linalg_eig_impl(
    const CnpArray *a,
    CnpArray **eigenvalues,
    CnpArray **eigenvectors,
    bool force_complex,
    bool compute_vectors) {
    CnpEigComplex *values = NULL;
    CnpEigComplex *vectors = NULL;
    int64_t value_shape[CNP_MAXDIMS];
    int64_t vector_shape[CNP_MAXDIMS];
    int64_t batch_count = 1;
    int64_t n;
    int64_t value_count;
    int64_t vector_count;
    CNP_STATUS status = CNP_OK;

    if (eigenvalues) *eigenvalues = NULL;
    if (eigenvectors) *eigenvectors = NULL;
    if (!eigenvalues ||
            (compute_vectors &&
             (!eigenvectors || eigenvalues == eigenvectors))) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_linalg_eig",
            compute_vectors
                ? "distinct eigenvalue and eigenvector output pointers are required"
                : "eigenvalue output pointer is required");
        return CNP_ERR_GENERIC;
    }
    if (!a) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_linalg_eig",
            "input array must not be null");
        return CNP_ERR_GENERIC;
    }
    if (a->ndim < 2 ||
            a->shape[a->ndim - 2] != a->shape[a->ndim - 1]) {
        cnp_set_error(
            CNP_ERR_SHAPE, "cnp_linalg_eig",
            "input must be square on its last two dimensions");
        return CNP_ERR_SHAPE;
    }
    if (!eig_type_is_supported(a->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, "cnp_linalg_eig",
            "input dtype is not supported by linear algebra");
        return CNP_ERR_TYPE;
    }

    n = a->shape[a->ndim - 1];
    for (int dimension = 0; dimension < a->ndim - 2; dimension++) {
        int64_t length = a->shape[dimension];
        if (length != 0 && batch_count > INT64_MAX / length) {
            cnp_set_error(
                CNP_ERR_SHAPE, "cnp_linalg_eig",
                "batch shape is too large");
            return CNP_ERR_SHAPE;
        }
        batch_count *= length;
        value_shape[dimension] = length;
        vector_shape[dimension] = length;
    }
    value_shape[a->ndim - 2] = n;
    vector_shape[a->ndim - 2] = n;
    vector_shape[a->ndim - 1] = n;
    if (n != 0 && batch_count > INT64_MAX / n) {
        cnp_set_error(
            CNP_ERR_SHAPE, "cnp_linalg_eig",
            "eigenvalue result is too large");
        return CNP_ERR_SHAPE;
    }
    value_count = batch_count * n;
    if (compute_vectors && n != 0 && value_count > INT64_MAX / n) {
        cnp_set_error(
            CNP_ERR_SHAPE, "cnp_linalg_eig",
            "eigenvector result is too large");
        return CNP_ERR_SHAPE;
    }
    vector_count = compute_vectors ? value_count * n : 0;
    if ((uint64_t)value_count > SIZE_MAX / sizeof(CnpEigComplex) ||
            (uint64_t)vector_count > SIZE_MAX / sizeof(CnpEigComplex)) {
        cnp_set_error(
            CNP_ERR_MEMORY, "cnp_linalg_eig",
            "general eig result workspace is too large");
        return CNP_ERR_MEMORY;
    }
    if (value_count > 0) {
        values = (CnpEigComplex*)cnp_calloc(
            (size_t)value_count, sizeof(CnpEigComplex));
    }
    if (vector_count > 0) {
        vectors = (CnpEigComplex*)cnp_calloc(
            (size_t)vector_count, sizeof(CnpEigComplex));
    }
    if ((value_count > 0 && !values) ||
            (vector_count > 0 && !vectors)) {
        cnp_set_error(
            CNP_ERR_MEMORY, "cnp_linalg_eig",
            "unable to allocate batched general eig results");
        status = CNP_ERR_MEMORY;
        goto cleanup;
    }

    for (int64_t batch = 0; batch < batch_count; batch++) {
        status = eig_solve_matrix(
            a, eig_batch_offset(a, batch), n,
            values ? values + batch * n : NULL,
            vectors ? vectors + batch * n * n : NULL,
            compute_vectors);
        if (status != CNP_OK) goto cleanup;
    }
    status = eig_write_results(
        values, vectors, value_count, vector_count,
        a->ndim - 1, value_shape,
        a->ndim, vector_shape,
        a->dtype->type_num, force_complex, compute_vectors,
        eigenvalues, eigenvectors);

cleanup:
    if (values) {
        cnp_free(
            values, (size_t)value_count * sizeof(CnpEigComplex));
    }
    if (vectors) {
        cnp_free(
            vectors, (size_t)vector_count * sizeof(CnpEigComplex));
    }
    return status;
}

CNP_STATUS cnp_linalg_eigvals_force_complex(
    const CnpArray *a,
    CnpArray **eigenvalues) {
    return eig_linalg_eig_impl(
        a, eigenvalues, NULL, true, false);
}

CNP_API CNP_STATUS CNP_CALL cnp_linalg_eig(
    const CnpArray *a,
    CnpArray **eigenvalues,
    CnpArray **eigenvectors) {
    return eig_linalg_eig_impl(
        a, eigenvalues, eigenvectors, false, true);
}
