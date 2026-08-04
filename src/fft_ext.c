/**
 * cnumpy extended FFT functions - NumPy 1.25 axes projections.
 */
#include "../include/cnumpy/cnumpy_internal.h"

static bool fft_ext_array_valid(
    const CnpArray *array, const char *function_name) {
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            array, function_name, &ignored_nbytes)) return false;
    if (array->ndim <= 0 || array->ndim > CNP_MAXDIMS ||
            !array->shape || !array->strides ||
            (array->size > 0 && !array->data)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source must be a valid array with at least one dimension");
        return false;
    }
    CNP_TYPE type = array->dtype->type_num;
    if (!(type == CNP_BOOL ||
          cnp_type_is_integer(type) ||
          cnp_type_is_float(type) ||
          cnp_type_is_complex(type))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array must have a represented numeric dtype");
        return false;
    }
    return true;
}

static bool fft_ext_prepare_axes(
    const CnpArray *array, int naxes, const int *axes,
    int *normalized, int *count,
    const char *function_name) {
    if (!fft_ext_array_valid(array, function_name)) return false;
    if (naxes <= 0) {
        *count = array->ndim;
        for (int index = 0; index < *count; ++index)
            normalized[index] = index;
        return true;
    }
    if (!axes) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "axes must not be NULL when an axis count is supplied");
        return false;
    }
    if (naxes > CNP_MAXDIMS) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "axis count %d exceeds CNP_MAXDIMS", naxes);
        return false;
    }
    *count = naxes;
    for (int index = 0; index < naxes; ++index) {
        int axis = axes[index];
        if (axis < 0) axis += array->ndim;
        if (axis < 0 || axis >= array->ndim) {
            cnp_set_error(
                CNP_ERR_AXIS, function_name,
                "axis %d is out of bounds for array of dimension %d",
                axes[index], array->ndim);
            return false;
        }
        normalized[index] = axis;
    }
    return true;
}

static CnpArray *fft_ext_complex_transform(
    const CnpArray *source, int naxes, const int *axes,
    bool inverse, const char *function_name) {
    int normalized[CNP_MAXDIMS];
    int count;
    if (!fft_ext_prepare_axes(
            source, naxes, axes,
            normalized, &count, function_name)) return NULL;

    const CnpArray *current = source;
    CnpArray *owned = NULL;
    for (int index = count - 1; index >= 0; --index) {
        int axis = normalized[index];
        int64_t length = current->shape[axis];
        if (length <= 0) {
            if (owned) cnp_array_decref(owned);
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "number of FFT data points must be positive");
            return NULL;
        }
        CnpArray *next = cnp_fft_axis_transform(
            current, axis, length, inverse, function_name);
        if (!next) {
            if (owned) cnp_array_decref(owned);
            return NULL;
        }
        if (owned) cnp_array_decref(owned);
        owned = next;
        current = owned;
    }
    return owned;
}

CNP_API CnpArray* CNP_CALL cnp_fftn(
    const CnpArray *a, int naxes, const int *axes) {
    return fft_ext_complex_transform(
        a, naxes, axes, false, "cnp_fftn");
}

CNP_API CnpArray* CNP_CALL cnp_ifftn(
    const CnpArray *a, int naxes, const int *axes) {
    return fft_ext_complex_transform(
        a, naxes, axes, true, "cnp_ifftn");
}

CNP_API CnpArray* CNP_CALL cnp_rfftn(
    const CnpArray *a, int naxes, const int *axes) {
    const char *function_name = "cnp_rfftn";
    int normalized[CNP_MAXDIMS];
    int count;
    if (!fft_ext_prepare_axes(
            a, naxes, axes,
            normalized, &count, function_name)) return NULL;

    int real_axis = normalized[count - 1];
    int64_t real_length = a->shape[real_axis];
    if (real_length <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "number of FFT data points must be positive");
        return NULL;
    }
    CnpArray *owned = cnp_fft_axis_real_forward(
        a, real_axis, real_length, function_name);
    if (!owned) return NULL;

    for (int index = count - 2; index >= 0; --index) {
        int axis = normalized[index];
        int64_t length = owned->shape[axis];
        if (length <= 0) {
            cnp_array_decref(owned);
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "number of FFT data points must be positive");
            return NULL;
        }
        CnpArray *next = cnp_fft_axis_transform(
            owned, axis, length, false, function_name);
        cnp_array_decref(owned);
        if (!next) return NULL;
        owned = next;
    }
    return owned;
}

static bool fft_ext_prepare_irfftn_axes(
    const CnpArray *array,
    int naxes, const int *axes,
    int ndims, const int64_t *shape,
    int *normalized, int *count,
    const char *function_name) {
    if (!fft_ext_array_valid(array, function_name)) return false;
    bool has_shape = ndims > 0 || shape != NULL;
    if (has_shape && (ndims <= 0 || !shape)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "shape pointer and positive shape count must be supplied together");
        return false;
    }
    if (naxes > 0) {
        if (!fft_ext_prepare_axes(
                array, naxes, axes,
                normalized, count, function_name)) return false;
        if (has_shape && ndims != *count) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "shape and axes must have the same length");
            return false;
        }
    } else if (has_shape) {
        if (ndims > array->ndim || ndims > CNP_MAXDIMS) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "shape has more dimensions than the source array");
            return false;
        }
        *count = ndims;
        for (int index = 0; index < *count; ++index)
            normalized[index] = array->ndim - *count + index;
    } else {
        *count = array->ndim;
        for (int index = 0; index < *count; ++index)
            normalized[index] = index;
    }
    if (has_shape) {
        for (int index = 0; index < ndims; ++index) {
            if (shape[index] <= 0) {
                cnp_set_error(
                    CNP_ERR_VALUE, function_name,
                    "number of FFT data points must be positive");
                return false;
            }
        }
    }
    return true;
}

CNP_API CnpArray* CNP_CALL cnp_irfftn(
    const CnpArray *a, int naxes, const int *axes,
    int ndims, const int64_t *s) {
    const char *function_name = "cnp_irfftn";
    int normalized[CNP_MAXDIMS];
    int count;
    if (!fft_ext_prepare_irfftn_axes(
            a, naxes, axes, ndims, s,
            normalized, &count, function_name)) return NULL;

    const CnpArray *current = a;
    CnpArray *owned = NULL;
    for (int index = count - 2; index >= 0; --index) {
        int axis = normalized[index];
        int64_t length = s ? s[index] : current->shape[axis];
        CnpArray *next = cnp_fft_axis_transform(
            current, axis, length, true, function_name);
        if (!next) {
            if (owned) cnp_array_decref(owned);
            return NULL;
        }
        if (owned) cnp_array_decref(owned);
        owned = next;
        current = owned;
    }

    int real_axis = normalized[count - 1];
    int64_t real_length = s
        ? s[count - 1]
        : (current->shape[real_axis] - 1) * 2;
    if (real_length <= 0) {
        if (owned) cnp_array_decref(owned);
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "number of FFT data points must be positive");
        return NULL;
    }
    CnpArray *result = cnp_fft_axis_real_inverse(
        current, real_axis, real_length, function_name);
    if (owned) cnp_array_decref(owned);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_hfft(const CnpArray *a, int64_t n) {
    const char *function_name = "cnp_hfft";
    if (!fft_ext_array_valid(a, function_name)) return NULL;
    int axis = a->ndim - 1;
    int64_t length = n < 0 ? (a->shape[axis] - 1) * 2 : n;
    if (length <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "number of FFT data points must be positive");
        return NULL;
    }
    CnpArray *conjugated = cnp_conjugate(a);
    if (!conjugated) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpArray *result = cnp_fft_axis_real_inverse(
        conjugated, axis, length, function_name);
    cnp_array_decref(conjugated);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t index = 0; index < result->size; ++index)
        output[index] *= (double)length;
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_ihfft(const CnpArray *a) {
    const char *function_name = "cnp_ihfft";
    if (!fft_ext_array_valid(a, function_name)) return NULL;
    int axis = a->ndim - 1;
    int64_t length = a->shape[axis];
    if (length <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "number of FFT data points must be positive");
        return NULL;
    }
    CnpArray *result = cnp_fft_axis_real_forward(
        a, axis, length, function_name);
    if (!result) return NULL;
    cnp_cdouble *output = (cnp_cdouble*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        output[index].real /= (double)length;
        output[index].imag = -output[index].imag / (double)length;
    }
    return result;
}
