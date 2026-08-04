/**
 * cnumpy remaining functions for full numpy coverage
 * Corresponds to numpy: eigvals, eigvalsh, take_along_axis, put_along_axis,
 *   divmod, roots, nanargmax, nanargmin, sort_complex, and more random distributions
 */
#include "../include/cnumpy/cnumpy_internal.h"

/* =========================================================================
 * cnp_eigvals - Eigenvalues only (no eigenvectors)
 * numpy.linalg.eigvals(a)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_eigvals(const CnpArray *a) {
    CnpArray *eigenvalues = NULL, *eigenvectors = NULL;
    CNP_STATUS status = cnp_linalg_eig(a, &eigenvalues, &eigenvectors);
    if (status != CNP_OK) {
        cnp_relabel_error("cnp_eigvals");
        return NULL;
    }
    if (eigenvectors) cnp_array_free(eigenvectors);
    return eigenvalues;
}

/* =========================================================================
 * cnp_eigvalsh - Eigenvalues of Hermitian/symmetric matrix
 * numpy.linalg.eigvalsh(a)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_eigvalsh(const CnpArray *a) {
    CnpArray *result = cnp_eigvalsh_v2(a, false);
    if (!result) cnp_relabel_error("cnp_eigvalsh");
    return result;
}

/* =========================================================================
 * cnp_take_along_axis - Take values along axis using indices
 * numpy.take_along_axis(arr, indices, axis)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_take_along_axis(const CnpArray *arr, const CnpArray *indices, int axis) {
    return cnp_take_along_axis_v2(arr, indices, axis, false);
}

/* =========================================================================
 * cnp_nanargmax - Index of maximum ignoring NaN
 * numpy.nanargmax(a, axis=None)
 * ========================================================================= */
static int64_t cnp_legacy_nanarg_extrema(
        const CnpArray *arr, int axis, bool maximum,
        const char *function_name) {
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return -1;
    }
    CnpArray *result = maximum
        ? cnp_nanargmax_v2(arr, axis, axis == CNP_AXIS_NONE)
        : cnp_nanargmin_v2(arr, axis, axis == CNP_AXIS_NONE);
    if (!result) {
        cnp_relabel_error(function_name);
        return -1;
    }
    if (result->ndim != 0) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "legacy scalar return cannot represent an array result");
        return -1;
    }
    int64_t index = cnp_array_get_int(result, NULL);
    cnp_array_free(result);
    return index;
}

CNP_API int64_t CNP_CALL cnp_nanargmax(const CnpArray *arr, int axis) {
    return cnp_legacy_nanarg_extrema(
        arr, axis, true, "cnp_nanargmax");
}

/* =========================================================================
 * cnp_nanargmin - Index of minimum ignoring NaN
 * numpy.nanargmin(a, axis=None)
 * ========================================================================= */
CNP_API int64_t CNP_CALL cnp_nanargmin(const CnpArray *arr, int axis) {
    return cnp_legacy_nanarg_extrema(
        arr, axis, false, "cnp_nanargmin");
}

static CnpArray *cnp_nanarg_extrema_v2(
    const CnpArray *arr, int axis, bool axis_none,
    bool maximum, const char *function_name) {
    int resolved_axis;
    if (!cnp_reduction_resolve_axis(
            arr, axis, axis_none,
            function_name, &resolved_axis)) return NULL;
    if (!cnp_type_is_float(arr->dtype->type_num))
        return maximum
            ? cnp_argmax_v2(arr, axis, axis_none)
            : cnp_argmin_v2(arr, axis, axis_none);

    CnpReductionTraversal traversal;
    cnp_reduction_traversal_init(arr, resolved_axis, &traversal);
    CnpArray *result = cnp_array_new(
        traversal.result_ndim, traversal.result_shape,
        CNP_LONGLONG, CNP_ORDER_C);
    if (!result) return NULL;
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            bool found = false;
            double selected_value = 0.0;
            int64_t selected_item = 0;
            for (int64_t item = 0;
                 item < traversal.axis_length; ++item) {
                int64_t source_offset = cnp_reduction_source_offset(
                    &traversal, outer, inner, item);
                double value = cnp_get_element_double(
                    arr->data, source_offset,
                    arr->dtype->type_num);
                if (isnan(value)) continue;
                if (!found || (maximum
                        ? value > selected_value
                        : value < selected_value)) {
                    selected_value = value;
                    selected_item = item;
                    found = true;
                }
            }
            if (!found) {
                cnp_array_free(result);
                cnp_set_error(CNP_ERR_GENERIC, function_name,
                              "All-NaN slice encountered");
                return NULL;
            }
            ((int64_t*)result->data)[
                outer * traversal.inner + inner] = selected_item;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_nanargmax_v2(
    const CnpArray *arr, int axis, bool axis_none) {
    return cnp_nanarg_extrema_v2(
        arr, axis, axis_none, true, "cnp_nanargmax_v2");
}

CNP_API CnpArray* CNP_CALL cnp_nanargmin_v2(
    const CnpArray *arr, int axis, bool axis_none) {
    return cnp_nanarg_extrema_v2(
        arr, axis, axis_none, false, "cnp_nanargmin_v2");
}

/* =========================================================================
 * cnp_random_chisquare - Chi-square distribution
 * numpy.random.chisquare(df, size)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_random_chisquare(double df, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_chisquare";
    if (!isfinite(df) || df <= 0.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "degrees of freedom must be finite and positive");
        return NULL;
    }
    CnpArray *result = cnp_random_gamma(df / 2.0, 2.0, ndim, shape);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * cnp_random_geometric - Geometric distribution
 * numpy.random.geometric(p, size)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_random_geometric(double p, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_geometric";
    if (!isfinite(p) || p <= 0.0 || p > 1.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "probability must be finite and in (0, 1]");
        return NULL;
    }
    CnpArray *result = cnp_random_output_new(
        ndim, shape, CNP_INT, function_name);
    if (!result) return NULL;
    int32_t *output = (int32_t*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        if (p == 1.0) {
            output[index] = 1;
            continue;
        }
        double uniform;
        do {
            uniform = cnp_random_double();
        } while (uniform == 0.0);
        double sample = ceil(log(uniform) / log1p(-p));
        if (sample > INT32_MAX) {
            cnp_array_decref(result);
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "sample exceeds NumPy's Windows int32 result range");
            return NULL;
        }
        output[index] = (int32_t)sample;
    }
    return result;
}

/* =========================================================================
 * cnp_random_zipf - Zipf distribution
 * numpy.random.zipf(a, size)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_random_zipf(double a, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_zipf";
    if (!isfinite(a) || a <= 1.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "shape must be finite and greater than one");
        return NULL;
    }
    CnpArray *result = cnp_random_output_new(
        ndim, shape, CNP_INT, function_name);
    if (!result) return NULL;
    int32_t *output = (int32_t*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        double u, v, x, t;
        double am1 = a - 1.0;
        double b = pow(2.0, am1);
        do {
            do {
                u = cnp_random_double();
            } while (u == 0.0);
            v = cnp_random_double();
            x = floor(pow(u, -1.0 / am1));
            t = pow(1.0 + 1.0 / x, am1);
        } while (x < 1.0 ||
            v * x * (t - 1.0) / (b - 1.0) > t / b);
        if (x > INT32_MAX) {
            cnp_array_decref(result);
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "sample exceeds NumPy's Windows int32 result range");
            return NULL;
        }
        output[index] = (int32_t)x;
    }
    return result;
}

/* =========================================================================
 * cnp_random_wald - Wald (inverse Gaussian) distribution
 * numpy.random.wald(mean, scale, size)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_random_wald(double mean, double scale, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_wald";
    if (!isfinite(mean) || !isfinite(scale) ||
            mean <= 0.0 || scale <= 0.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "mean and scale must be finite and positive");
        return NULL;
    }
    CnpArray *result = cnp_random_output_new(
        ndim, shape, CNP_DOUBLE, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        double z = cnp_random_gauss();
        double y = z * z;
        double x = mean + (mean * mean * y) / (2.0 * scale)
                 - (mean / (2.0 * scale)) * sqrt(4.0 * mean * scale * y + mean * mean * y * y);
        double u = cnp_random_double();
        if (u <= mean / (mean + x)) {
            output[index] = x;
        } else {
            output[index] = mean * mean / x;
        }
    }
    return result;
}

/* =========================================================================
 * cnp_random_hypergeometric - Hypergeometric distribution
 * numpy.random.hypergeometric(ngood, nbad, nsample, size)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_random_hypergeometric(int64_t ngood, int64_t nbad, int64_t nsample,
                                                      int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_hypergeometric";
    if (ngood < 0 || nbad < 0 || nsample < 0 ||
            ngood > INT32_MAX || nbad > INT32_MAX ||
            nsample > INT32_MAX || ngood > INT64_MAX - nbad ||
            nsample > ngood + nbad) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "population and sample counts are outside valid Windows int32 bounds");
        return NULL;
    }
    CnpArray *result = cnp_random_output_new(
        ndim, shape, CNP_INT, function_name);
    if (!result) return NULL;
    int32_t *output = (int32_t*)result->data;
    int64_t ntotal = ngood + nbad;
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t good_remaining = ngood;
        int64_t total_remaining = ntotal;
        int64_t drawn_good = 0;
        for (int64_t s = 0; s < nsample && total_remaining > 0; s++) {
            double u = cnp_random_double();
            if (u < (double)good_remaining / (double)total_remaining) {
                drawn_good++;
                good_remaining--;
            }
            total_remaining--;
        }
        output[index] = (int32_t)drawn_good;
    }
    return result;
}
