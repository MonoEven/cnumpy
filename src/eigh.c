#include "../include/cnumpy/cnumpy_internal.h"

typedef struct {
    double real;
    double imag;
} CnpEighComplex;

static CnpEighComplex eigh_complex(double real, double imag) {
    CnpEighComplex value = {real, imag};
    return value;
}

static CnpEighComplex eigh_add(
        CnpEighComplex left, CnpEighComplex right) {
    return eigh_complex(left.real + right.real, left.imag + right.imag);
}

static CnpEighComplex eigh_subtract(
        CnpEighComplex left, CnpEighComplex right) {
    return eigh_complex(left.real - right.real, left.imag - right.imag);
}

static CnpEighComplex eigh_multiply(
        CnpEighComplex left, CnpEighComplex right) {
    return eigh_complex(
        left.real * right.real - left.imag * right.imag,
        left.real * right.imag + left.imag * right.real);
}

static CnpEighComplex eigh_scale(CnpEighComplex value, double scale) {
    return eigh_complex(value.real * scale, value.imag * scale);
}

static CnpEighComplex eigh_conjugate(CnpEighComplex value) {
    return eigh_complex(value.real, -value.imag);
}

static double eigh_absolute(CnpEighComplex value) {
    return hypot(value.real, value.imag);
}

static bool eigh_type_is_supported(CNP_TYPE type) {
    return type == CNP_BOOL || cnp_type_is_integer(type) ||
        type == CNP_FLOAT || type == CNP_DOUBLE ||
        type == CNP_CFLOAT || type == CNP_CDOUBLE;
}

static CNP_TYPE eigh_value_type(CNP_TYPE input_type) {
    return input_type == CNP_FLOAT || input_type == CNP_CFLOAT
        ? CNP_FLOAT : CNP_DOUBLE;
}

static CNP_TYPE eigh_vector_type(CNP_TYPE input_type) {
    if (input_type == CNP_FLOAT) return CNP_FLOAT;
    if (input_type == CNP_CFLOAT) return CNP_CFLOAT;
    if (input_type == CNP_CDOUBLE) return CNP_CDOUBLE;
    return CNP_DOUBLE;
}

static CnpEighComplex eigh_read_raw(
        const CnpArray *array, int64_t batch_offset,
        int64_t row, int64_t column) {
    int64_t offset = batch_offset +
        row * array->strides[array->ndim - 2] +
        column * array->strides[array->ndim - 1];
    const char *pointer = (const char*)array->data + offset;
    switch (array->dtype->type_num) {
        case CNP_CFLOAT: {
            const cnp_cfloat *value = (const cnp_cfloat*)pointer;
            return eigh_complex(value->real, value->imag);
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *value = (const cnp_cdouble*)pointer;
            return eigh_complex(value->real, value->imag);
        }
        default:
            return eigh_complex(
                cnp_get_element_double(
                    array->data, offset, array->dtype->type_num),
                0.0);
    }
}

static int64_t eigh_batch_offset(
        const CnpArray *array, int64_t batch_index) {
    int64_t offset = array->offset;
    for (int dimension = array->ndim - 3; dimension >= 0; dimension--) {
        int64_t coordinate = batch_index % array->shape[dimension];
        batch_index /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

static void eigh_set_identity(CnpEighComplex *matrix, int64_t n) {
    memset(matrix, 0, (size_t)(n * n) * sizeof(CnpEighComplex));
    for (int64_t index = 0; index < n; index++) {
        matrix[index * n + index].real = 1.0;
    }
}

static CNP_STATUS eigh_load_hermitian(
        const CnpArray *source, int64_t batch_offset,
        int64_t n, bool upper, CnpEighComplex *matrix,
        bool *nonfinite_off_diagonal,
        const char *function_name) {
    *nonfinite_off_diagonal = false;
    for (int64_t row = 0; row < n; row++) {
        CnpEighComplex diagonal = eigh_read_raw(
            source, batch_offset, row, row);
        if (!isfinite(diagonal.real)) {
            cnp_set_error(
                CNP_ERR_CONVERGENCE, function_name,
                "selected Hermitian diagonal contains NaN or infinity");
            return CNP_ERR_CONVERGENCE;
        }
        matrix[row * n + row] = eigh_complex(diagonal.real, 0.0);
        for (int64_t column = row + 1; column < n; column++) {
            CnpEighComplex selected = upper
                ? eigh_read_raw(source, batch_offset, row, column)
                : eigh_conjugate(eigh_read_raw(
                    source, batch_offset, column, row));
            if (!isfinite(selected.real) || !isfinite(selected.imag)) {
                *nonfinite_off_diagonal = true;
            }
            matrix[row * n + column] = selected;
            matrix[column * n + row] = eigh_conjugate(selected);
        }
    }
    return CNP_OK;
}

static void eigh_rotate_pair(
        CnpEighComplex *matrix, CnpEighComplex *vectors,
        int64_t n, int64_t p, int64_t q) {
    CnpEighComplex off_diagonal = matrix[p * n + q];
    double magnitude = eigh_absolute(off_diagonal);
    double app = matrix[p * n + p].real;
    double aqq = matrix[q * n + q].real;
    double tau = (0.5 * aqq - 0.5 * app) / magnitude;
    double tangent = tau >= 0.0
        ? 1.0 / (tau + hypot(1.0, tau))
        : -1.0 / (-tau + hypot(1.0, tau));
    double cosine = 1.0 / sqrt(1.0 + tangent * tangent);
    double sine = tangent * cosine;
    CnpEighComplex phase = eigh_scale(off_diagonal, 1.0 / magnitude);
    CnpEighComplex conjugate_phase = eigh_conjugate(phase);

    for (int64_t index = 0; index < n; index++) {
        CnpEighComplex old_ip;
        CnpEighComplex old_iq;
        CnpEighComplex new_ip;
        CnpEighComplex new_iq;
        if (index == p || index == q) continue;
        old_ip = matrix[index * n + p];
        old_iq = matrix[index * n + q];
        new_ip = eigh_subtract(
            eigh_scale(old_ip, cosine),
            eigh_scale(eigh_multiply(conjugate_phase, old_iq), sine));
        new_iq = eigh_add(
            eigh_scale(eigh_multiply(phase, old_ip), sine),
            eigh_scale(old_iq, cosine));
        matrix[index * n + p] = new_ip;
        matrix[p * n + index] = eigh_conjugate(new_ip);
        matrix[index * n + q] = new_iq;
        matrix[q * n + index] = eigh_conjugate(new_iq);
    }

    matrix[p * n + p] = eigh_complex(
        app - tangent * magnitude, 0.0);
    matrix[q * n + q] = eigh_complex(
        aqq + tangent * magnitude, 0.0);
    matrix[p * n + q] = eigh_complex(0.0, 0.0);
    matrix[q * n + p] = eigh_complex(0.0, 0.0);

    if (!vectors) return;
    for (int64_t row = 0; row < n; row++) {
        CnpEighComplex old_rp = vectors[row * n + p];
        CnpEighComplex old_rq = vectors[row * n + q];
        vectors[row * n + p] = eigh_subtract(
            eigh_scale(old_rp, cosine),
            eigh_scale(eigh_multiply(conjugate_phase, old_rq), sine));
        vectors[row * n + q] = eigh_add(
            eigh_scale(eigh_multiply(phase, old_rp), sine),
            eigh_scale(old_rq, cosine));
    }
}

static CNP_STATUS eigh_diagonalize(
        CnpEighComplex *matrix, CnpEighComplex *vectors,
        int64_t n, const char *function_name) {
    double scale = 0.0;
    double tolerance;

    if (n < 2) return CNP_OK;
    for (int64_t row = 0; row < n; row++) {
        scale = fmax(scale, fabs(matrix[row * n + row].real));
        for (int64_t column = row + 1; column < n; column++) {
            scale = fmax(scale, eigh_absolute(matrix[row * n + column]));
        }
    }
    tolerance = 32.0 * DBL_EPSILON * scale;

    for (;;) {
        double largest = 0.0;
        for (int64_t p = 0; p < n; p++) {
            for (int64_t q = p + 1; q < n; q++) {
                double magnitude = eigh_absolute(matrix[p * n + q]);
                if (!isfinite(magnitude)) {
                    cnp_set_error(
                        CNP_ERR_CONVERGENCE, function_name,
                        "Hermitian Jacobi iteration produced a non-finite value");
                    return CNP_ERR_CONVERGENCE;
                }
                if (magnitude > tolerance) {
                    eigh_rotate_pair(matrix, vectors, n, p, q);
                }
            }
        }
        for (int64_t p = 0; p < n; p++) {
            for (int64_t q = p + 1; q < n; q++) {
                double magnitude = eigh_absolute(matrix[p * n + q]);
                if (!isfinite(magnitude)) {
                    cnp_set_error(
                        CNP_ERR_CONVERGENCE, function_name,
                        "Hermitian Jacobi iteration produced a non-finite value");
                    return CNP_ERR_CONVERGENCE;
                }
                largest = fmax(largest, magnitude);
            }
        }
        if (largest <= tolerance) return CNP_OK;
    }
}

static void eigh_sort(
        CnpEighComplex *matrix, CnpEighComplex *vectors, int64_t n) {
    for (int64_t destination = 0; destination < n; destination++) {
        int64_t selected = destination;
        for (int64_t candidate = destination + 1;
                candidate < n; candidate++) {
            if (matrix[candidate * n + candidate].real <
                    matrix[selected * n + selected].real) {
                selected = candidate;
            }
        }
        if (selected == destination) continue;
        {
            CnpEighComplex temporary =
                matrix[destination * n + destination];
            matrix[destination * n + destination] =
                matrix[selected * n + selected];
            matrix[selected * n + selected] = temporary;
        }
        if (vectors) {
            for (int64_t row = 0; row < n; row++) {
                CnpEighComplex temporary = vectors[row * n + destination];
                vectors[row * n + destination] = vectors[row * n + selected];
                vectors[row * n + selected] = temporary;
            }
        }
    }
}

static void eigh_write_value(
        CnpArray *array, int64_t index, double value) {
    if (array->dtype->type_num == CNP_FLOAT) {
        ((float*)array->data)[index] = (float)value;
    } else {
        ((double*)array->data)[index] = value;
    }
}

static void eigh_write_vector(
        CnpArray *array, int64_t index, CnpEighComplex value) {
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

static CNP_STATUS eigh_execute(
        const CnpArray *source, bool upper, bool compute_vectors,
        CnpArray **eigenvalues, CnpArray **eigenvectors,
        const char *function_name) {
    int64_t value_shape[CNP_MAXDIMS];
    int64_t vector_shape[CNP_MAXDIMS];
    int64_t batch_count = 1;
    int64_t n;
    int64_t matrix_count;
    int64_t value_count;
    CnpEighComplex *matrix = NULL;
    CnpEighComplex *vectors = NULL;
    CnpArray *value_result = NULL;
    CnpArray *vector_result = NULL;
    CNP_STATUS status = CNP_OK;

    if (eigenvalues) *eigenvalues = NULL;
    if (eigenvectors) *eigenvectors = NULL;
    if (!eigenvalues ||
            (compute_vectors &&
             (!eigenvectors || eigenvalues == eigenvectors))) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            compute_vectors
                ? "distinct eigenvalue and eigenvector output pointers are required"
                : "eigenvalue output pointer is required");
        return CNP_ERR_GENERIC;
    }
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input array must not be null");
        return CNP_ERR_GENERIC;
    }
    if (source->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input must have at least two dimensions");
        return CNP_ERR_SHAPE;
    }
    if (source->shape[source->ndim - 2] !=
            source->shape[source->ndim - 1]) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "input must be square on its last two dimensions");
        return CNP_ERR_SHAPE;
    }
    if (!eigh_type_is_supported(source->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "input dtype is not supported by linear algebra");
        return CNP_ERR_TYPE;
    }

    n = source->shape[source->ndim - 1];
    for (int dimension = 0; dimension < source->ndim - 2; dimension++) {
        int64_t length = source->shape[dimension];
        if (length != 0 && batch_count > INT64_MAX / length) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "batch shape is too large");
            return CNP_ERR_SHAPE;
        }
        batch_count *= length;
        value_shape[dimension] = length;
        vector_shape[dimension] = length;
    }
    value_shape[source->ndim - 2] = n;
    vector_shape[source->ndim - 2] = n;
    vector_shape[source->ndim - 1] = n;
    if (n != 0 && n > INT64_MAX / n) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "matrix dimension is too large");
        return CNP_ERR_SHAPE;
    }
    matrix_count = n * n;
    if (n != 0 && batch_count > INT64_MAX / n) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "eigenvalue result is too large");
        return CNP_ERR_SHAPE;
    }
    value_count = batch_count * n;
    if (n != 0 && value_count > INT64_MAX / n) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "eigenvector result is too large");
        return CNP_ERR_SHAPE;
    }
    if ((uint64_t)matrix_count >
            SIZE_MAX / sizeof(CnpEighComplex)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "Hermitian eig workspace is too large");
        return CNP_ERR_MEMORY;
    }

    value_result = cnp_array_new(
        source->ndim - 1, value_shape,
        eigh_value_type(source->dtype->type_num), CNP_ORDER_C);
    if (!value_result) {
        cnp_relabel_error(function_name);
        return CNP_ERR_MEMORY;
    }
    if (compute_vectors) {
        vector_result = cnp_array_new(
            source->ndim, vector_shape,
            eigh_vector_type(source->dtype->type_num), CNP_ORDER_C);
        if (!vector_result) {
            cnp_array_free(value_result);
            cnp_relabel_error(function_name);
            return CNP_ERR_MEMORY;
        }
    }

    if (batch_count != 0 && n != 0) {
        size_t workspace_bytes =
            (size_t)matrix_count * sizeof(CnpEighComplex);
        matrix = (CnpEighComplex*)cnp_malloc(workspace_bytes);
        if (compute_vectors) {
            vectors = (CnpEighComplex*)cnp_malloc(workspace_bytes);
        }
        if (!matrix || (compute_vectors && !vectors)) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "unable to allocate Hermitian eig workspace");
            status = CNP_ERR_MEMORY;
            goto cleanup;
        }
    }

    for (int64_t batch = 0; batch < batch_count; batch++) {
        bool nonfinite_off_diagonal = false;
        status = eigh_load_hermitian(
            source, eigh_batch_offset(source, batch),
            n, upper, matrix, &nonfinite_off_diagonal, function_name);
        if (status != CNP_OK) goto cleanup;
        if (nonfinite_off_diagonal) {
            CnpEighComplex nan_vector = eigh_complex(NAN, NAN);
            for (int64_t index = 0; index < n; index++) {
                eigh_write_value(value_result, batch * n + index, NAN);
            }
            if (compute_vectors) {
                for (int64_t index = 0; index < matrix_count; index++) {
                    eigh_write_vector(
                        vector_result,
                        batch * matrix_count + index,
                        nan_vector);
                }
            }
            continue;
        }
        if (compute_vectors) eigh_set_identity(vectors, n);
        status = eigh_diagonalize(matrix, vectors, n, function_name);
        if (status != CNP_OK) goto cleanup;
        eigh_sort(matrix, vectors, n);
        for (int64_t index = 0; index < n; index++) {
            eigh_write_value(
                value_result, batch * n + index,
                matrix[index * n + index].real);
        }
        if (compute_vectors) {
            for (int64_t row = 0; row < n; row++) {
                for (int64_t column = 0; column < n; column++) {
                    eigh_write_vector(
                        vector_result,
                        batch * matrix_count + row * n + column,
                        vectors[row * n + column]);
                }
            }
        }
    }

    *eigenvalues = value_result;
    if (compute_vectors) *eigenvectors = vector_result;

cleanup:
    if (matrix) {
        cnp_free(
            matrix, (size_t)matrix_count * sizeof(CnpEighComplex));
    }
    if (vectors) {
        cnp_free(
            vectors, (size_t)matrix_count * sizeof(CnpEighComplex));
    }
    if (status != CNP_OK) {
        cnp_array_free(value_result);
        if (vector_result) cnp_array_free(vector_result);
    }
    return status;
}

CNP_API CNP_STATUS CNP_CALL cnp_linalg_eigh_v2(
        const CnpArray *source, bool upper,
        CnpArray **eigenvalues, CnpArray **eigenvectors) {
    return eigh_execute(
        source, upper, true, eigenvalues, eigenvectors,
        "cnp_linalg_eigh_v2");
}

CNP_API CNP_STATUS CNP_CALL cnp_linalg_eigh(
        const CnpArray *source,
        CnpArray **eigenvalues, CnpArray **eigenvectors) {
    return eigh_execute(
        source, false, true, eigenvalues, eigenvectors,
        "cnp_linalg_eigh");
}

CNP_API CnpArray* CNP_CALL cnp_eigvalsh_v2(
        const CnpArray *source, bool upper) {
    CnpArray *eigenvalues = NULL;
    CNP_STATUS status = eigh_execute(
        source, upper, false, &eigenvalues, NULL,
        "cnp_eigvalsh_v2");
    return status == CNP_OK ? eigenvalues : NULL;
}
