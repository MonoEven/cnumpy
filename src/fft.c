/**
 * cnumpy FFT module - NumPy 1.25 compatible default-axis transforms.
 */
#include "../include/cnumpy/cnumpy_internal.h"

#include <limits.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    double real;
    double imag;
} CnpFftComplex;

static CnpFftComplex fft_add(CnpFftComplex left, CnpFftComplex right) {
    return (CnpFftComplex){
        left.real + right.real,
        left.imag + right.imag,
    };
}

static CnpFftComplex fft_subtract(
    CnpFftComplex left, CnpFftComplex right) {
    return (CnpFftComplex){
        left.real - right.real,
        left.imag - right.imag,
    };
}

static CnpFftComplex fft_multiply(
    CnpFftComplex left, CnpFftComplex right) {
    return (CnpFftComplex){
        left.real * right.real - left.imag * right.imag,
        left.real * right.imag + left.imag * right.real,
    };
}

static CnpFftComplex fft_scale(CnpFftComplex value, double scale) {
    return (CnpFftComplex){value.real * scale, value.imag * scale};
}

static bool fft_is_power_of_two(int64_t length) {
    return length > 0 && (length & (length - 1)) == 0;
}

static void fft_bit_reverse(CnpFftComplex *values, int64_t length) {
    int64_t reversed = 0;
    for (int64_t index = 0; index < length - 1; ++index) {
        if (index < reversed) {
            CnpFftComplex temporary = values[index];
            values[index] = values[reversed];
            values[reversed] = temporary;
        }
        int64_t bit = length >> 1;
        while (bit >= 1 && reversed >= bit) {
            reversed -= bit;
            bit >>= 1;
        }
        reversed += bit;
    }
}

static void fft_radix_two(
    CnpFftComplex *values, int64_t length, int sign) {
    fft_bit_reverse(values, length);
    for (int64_t width = 2;; width <<= 1) {
        double angle = sign * 2.0 * M_PI / (double)width;
        CnpFftComplex root = {cos(angle), sin(angle)};
        for (int64_t block = 0; block < length; block += width) {
            CnpFftComplex factor = {1.0, 0.0};
            for (int64_t offset = 0; offset < width / 2; ++offset) {
                CnpFftComplex even = values[block + offset];
                CnpFftComplex odd = fft_multiply(
                    values[block + offset + width / 2], factor);
                values[block + offset] = fft_add(even, odd);
                values[block + offset + width / 2] =
                    fft_subtract(even, odd);
                factor = fft_multiply(factor, root);
            }
        }
        if (width == length) break;
    }
}

static bool fft_bluestein(
    CnpFftComplex *values, int64_t length, int sign,
    const char *function_name) {
    if (length > INT64_MAX / 2 + 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "FFT workspace length exceeds INT64_MAX");
        return false;
    }
    int64_t required = 2 * length - 1;
    int64_t workspace = 1;
    while (workspace < required) {
        if (workspace > INT64_MAX / 2) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "FFT workspace length exceeds INT64_MAX");
            return false;
        }
        workspace <<= 1;
    }

    CnpFftComplex *left = (CnpFftComplex*)cnp_calloc(
        (size_t)workspace, sizeof(CnpFftComplex));
    CnpFftComplex *right = (CnpFftComplex*)cnp_calloc(
        (size_t)workspace, sizeof(CnpFftComplex));
    if (!left || !right) {
        if (right)
            cnp_free(right, (size_t)workspace * sizeof(CnpFftComplex));
        if (left)
            cnp_free(left, (size_t)workspace * sizeof(CnpFftComplex));
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate FFT workspace");
        return false;
    }

    for (int64_t index = 0; index < length; ++index) {
        long double square = (long double)index * (long double)index;
        double angle = (double)(sign * M_PI * square / length);
        CnpFftComplex chirp = {cos(angle), sin(angle)};
        left[index] = fft_multiply(values[index], chirp);
        right[index] = (CnpFftComplex){chirp.real, -chirp.imag};
        if (index > 0) right[workspace - index] = right[index];
    }

    fft_radix_two(left, workspace, -1);
    fft_radix_two(right, workspace, -1);
    for (int64_t index = 0; index < workspace; ++index)
        left[index] = fft_multiply(left[index], right[index]);
    fft_radix_two(left, workspace, 1);
    double workspace_scale = 1.0 / (double)workspace;
    for (int64_t index = 0; index < workspace; ++index)
        left[index] = fft_scale(left[index], workspace_scale);

    for (int64_t index = 0; index < length; ++index) {
        long double square = (long double)index * (long double)index;
        double angle = (double)(sign * M_PI * square / length);
        CnpFftComplex chirp = {cos(angle), sin(angle)};
        values[index] = fft_multiply(left[index], chirp);
    }

    cnp_free(right, (size_t)workspace * sizeof(CnpFftComplex));
    cnp_free(left, (size_t)workspace * sizeof(CnpFftComplex));
    return true;
}

static bool fft_execute(
    CnpFftComplex *values, int64_t length, bool inverse,
    const char *function_name) {
    if (length <= 1) return true;
    int sign = inverse ? 1 : -1;
    bool succeeded = true;
    if (fft_is_power_of_two(length))
        fft_radix_two(values, length, sign);
    else
        succeeded = fft_bluestein(values, length, sign, function_name);
    if (!succeeded) return false;
    if (inverse) {
        double scale = 1.0 / (double)length;
        for (int64_t index = 0; index < length; ++index)
            values[index] = fft_scale(values[index], scale);
    }
    return true;
}

static bool fft_array_valid(
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

static int64_t fft_flat_offset(
    const CnpArray *array, int64_t flat_index) {
    int64_t offset = array->offset;
    for (int dimension = array->ndim - 1; dimension >= 0; --dimension) {
        int64_t coordinate = flat_index % array->shape[dimension];
        flat_index /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

static bool fft_read_value(
    const CnpArray *array, int64_t flat_index,
    bool real_projection, CnpFftComplex *value,
    const char *function_name) {
    int64_t offset = fft_flat_offset(array, flat_index);
    cnp_clongdouble converted = {0};
    CNP_STATUS status = cnp_cast_scalar_value(
        (const char*)array->data + offset,
        array->dtype->type_num,
        &converted, CNP_CLONGDOUBLE, function_name);
    if (status != CNP_OK) return false;
    value->real = (double)converted.real;
    value->imag = real_projection ? 0.0 : (double)converted.imag;
    return true;
}

static bool fft_axis_parameters(
    const CnpArray *source, int axis, int64_t length,
    int64_t *outer, int64_t *inner,
    const char *function_name) {
    if (!fft_array_valid(source, function_name)) return false;
    if (axis < 0) axis += source->ndim;
    if (axis < 0 || axis >= source->ndim) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "axis %d is out of bounds for array of dimension %d",
            axis, source->ndim);
        return false;
    }
    if (length <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "number of FFT data points must be positive");
        return false;
    }
    *outer = 1;
    for (int dimension = 0; dimension < axis; ++dimension)
        *outer *= source->shape[dimension];
    *inner = 1;
    for (int dimension = axis + 1; dimension < source->ndim; ++dimension)
        *inner *= source->shape[dimension];
    return true;
}

CnpArray *cnp_fft_axis_transform(
    const CnpArray *source, int axis, int64_t length,
    bool inverse, const char *function_name) {
    int normalized_axis = axis < 0 ? axis + source->ndim : axis;
    int64_t outer;
    int64_t inner;
    if (!fft_axis_parameters(
            source, normalized_axis, length,
            &outer, &inner, function_name)) return NULL;

    int64_t result_shape[CNP_MAXDIMS];
    memcpy(result_shape, source->shape,
        (size_t)source->ndim * sizeof(int64_t));
    result_shape[normalized_axis] = length;
    CnpArray *result = cnp_array_new(
        source->ndim, result_shape, CNP_CDOUBLE, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpFftComplex *line = (CnpFftComplex*)cnp_calloc(
        (size_t)length, sizeof(CnpFftComplex));
    if (!line) {
        cnp_array_decref(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate FFT line workspace");
        return NULL;
    }

    int64_t source_length = source->shape[normalized_axis];
    int64_t copy_length = source_length < length ? source_length : length;
    cnp_cdouble *output = (cnp_cdouble*)result->data;
    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
        for (int64_t inner_index = 0; inner_index < inner; ++inner_index) {
            memset(line, 0, (size_t)length * sizeof(CnpFftComplex));
            for (int64_t item = 0; item < copy_length; ++item) {
                int64_t source_index =
                    (outer_index * source_length + item) * inner + inner_index;
                if (!fft_read_value(
                        source, source_index, false,
                        &line[item], function_name)) {
                    cnp_free(line, (size_t)length * sizeof(CnpFftComplex));
                    cnp_array_decref(result);
                    return NULL;
                }
            }
            if (!fft_execute(line, length, inverse, function_name)) {
                cnp_free(line, (size_t)length * sizeof(CnpFftComplex));
                cnp_array_decref(result);
                return NULL;
            }
            for (int64_t item = 0; item < length; ++item) {
                int64_t result_index =
                    (outer_index * length + item) * inner + inner_index;
                output[result_index].real = line[item].real;
                output[result_index].imag = line[item].imag;
            }
        }
    }
    cnp_free(line, (size_t)length * sizeof(CnpFftComplex));
    return result;
}

CnpArray *cnp_fft_axis_real_forward(
    const CnpArray *source, int axis, int64_t length,
    const char *function_name) {
    int normalized_axis = axis < 0 ? axis + source->ndim : axis;
    int64_t outer;
    int64_t inner;
    if (!fft_axis_parameters(
            source, normalized_axis, length,
            &outer, &inner, function_name)) return NULL;

    int64_t half_length = length / 2 + 1;
    int64_t result_shape[CNP_MAXDIMS];
    memcpy(result_shape, source->shape,
        (size_t)source->ndim * sizeof(int64_t));
    result_shape[normalized_axis] = half_length;
    CnpArray *result = cnp_array_new(
        source->ndim, result_shape, CNP_CDOUBLE, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpFftComplex *line = (CnpFftComplex*)cnp_calloc(
        (size_t)length, sizeof(CnpFftComplex));
    if (!line) {
        cnp_array_decref(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate FFT line workspace");
        return NULL;
    }

    int64_t source_length = source->shape[normalized_axis];
    int64_t copy_length = source_length < length ? source_length : length;
    cnp_cdouble *output = (cnp_cdouble*)result->data;
    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
        for (int64_t inner_index = 0; inner_index < inner; ++inner_index) {
            memset(line, 0, (size_t)length * sizeof(CnpFftComplex));
            for (int64_t item = 0; item < copy_length; ++item) {
                int64_t source_index =
                    (outer_index * source_length + item) * inner + inner_index;
                if (!fft_read_value(
                        source, source_index, true,
                        &line[item], function_name)) {
                    cnp_free(line, (size_t)length * sizeof(CnpFftComplex));
                    cnp_array_decref(result);
                    return NULL;
                }
            }
            if (!fft_execute(line, length, false, function_name)) {
                cnp_free(line, (size_t)length * sizeof(CnpFftComplex));
                cnp_array_decref(result);
                return NULL;
            }
            for (int64_t item = 0; item < half_length; ++item) {
                int64_t result_index =
                    (outer_index * half_length + item) * inner + inner_index;
                output[result_index].real = line[item].real;
                output[result_index].imag = line[item].imag;
            }
        }
    }
    cnp_free(line, (size_t)length * sizeof(CnpFftComplex));
    return result;
}

CnpArray *cnp_fft_axis_real_inverse(
    const CnpArray *source, int axis, int64_t length,
    const char *function_name) {
    int normalized_axis = axis < 0 ? axis + source->ndim : axis;
    int64_t outer;
    int64_t inner;
    if (!fft_axis_parameters(
            source, normalized_axis, length,
            &outer, &inner, function_name)) return NULL;

    int64_t result_shape[CNP_MAXDIMS];
    memcpy(result_shape, source->shape,
        (size_t)source->ndim * sizeof(int64_t));
    result_shape[normalized_axis] = length;
    CnpArray *result = cnp_array_new(
        source->ndim, result_shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CnpFftComplex *line = (CnpFftComplex*)cnp_calloc(
        (size_t)length, sizeof(CnpFftComplex));
    if (!line) {
        cnp_array_decref(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate FFT line workspace");
        return NULL;
    }

    int64_t source_length = source->shape[normalized_axis];
    int64_t required_half = length / 2 + 1;
    int64_t copy_length = source_length < required_half
        ? source_length : required_half;
    double *output = (double*)result->data;
    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
        for (int64_t inner_index = 0; inner_index < inner; ++inner_index) {
            memset(line, 0, (size_t)length * sizeof(CnpFftComplex));
            for (int64_t item = 0; item < copy_length; ++item) {
                int64_t source_index =
                    (outer_index * source_length + item) * inner + inner_index;
                if (!fft_read_value(
                        source, source_index, false,
                        &line[item], function_name)) {
                    cnp_free(line, (size_t)length * sizeof(CnpFftComplex));
                    cnp_array_decref(result);
                    return NULL;
                }
            }
            int64_t last_mirrored = (length - 1) / 2;
            for (int64_t item = 1; item <= last_mirrored; ++item) {
                line[length - item].real = line[item].real;
                line[length - item].imag = -line[item].imag;
            }
            if (!fft_execute(line, length, true, function_name)) {
                cnp_free(line, (size_t)length * sizeof(CnpFftComplex));
                cnp_array_decref(result);
                return NULL;
            }
            for (int64_t item = 0; item < length; ++item) {
                int64_t result_index =
                    (outer_index * length + item) * inner + inner_index;
                output[result_index] = line[item].real;
            }
        }
    }
    cnp_free(line, (size_t)length * sizeof(CnpFftComplex));
    return result;
}

static int64_t fft_resolve_length(
    const CnpArray *source, int64_t requested,
    const char *function_name) {
    if (!fft_array_valid(source, function_name)) return -1;
    int64_t length = requested < 0
        ? source->shape[source->ndim - 1] : requested;
    if (length <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "number of FFT data points must be positive");
        return -1;
    }
    return length;
}

CNP_API CnpArray* CNP_CALL cnp_fft(const CnpArray *a, int64_t n) {
    const char *function_name = "cnp_fft";
    int64_t length = fft_resolve_length(a, n, function_name);
    if (length < 0) return NULL;
    return cnp_fft_axis_transform(
        a, a->ndim - 1, length, false, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_ifft(const CnpArray *a, int64_t n) {
    const char *function_name = "cnp_ifft";
    int64_t length = fft_resolve_length(a, n, function_name);
    if (length < 0) return NULL;
    return cnp_fft_axis_transform(
        a, a->ndim - 1, length, true, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_rfft(const CnpArray *a, int64_t n) {
    const char *function_name = "cnp_rfft";
    int64_t length = fft_resolve_length(a, n, function_name);
    if (length < 0) return NULL;
    return cnp_fft_axis_real_forward(
        a, a->ndim - 1, length, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_irfft(const CnpArray *a, int64_t n) {
    const char *function_name = "cnp_irfft";
    if (!fft_array_valid(a, function_name)) return NULL;
    int64_t length = n < 0
        ? (a->shape[a->ndim - 1] - 1) * 2 : n;
    if (length <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "number of FFT data points must be positive");
        return NULL;
    }
    return cnp_fft_axis_real_inverse(
        a, a->ndim - 1, length, function_name);
}

static CnpArray *fft_two_axes(
    const CnpArray *source, bool inverse,
    const char *function_name) {
    if (!fft_array_valid(source, function_name)) return NULL;
    if (source->ndim < 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source must have at least two dimensions");
        return NULL;
    }
    int last_axis = source->ndim - 1;
    int64_t last_length = source->shape[last_axis];
    int64_t preceding_length = source->shape[last_axis - 1];
    if (last_length <= 0 || preceding_length <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "number of FFT data points must be positive");
        return NULL;
    }
    CnpArray *last = cnp_fft_axis_transform(
        source, last_axis, last_length, inverse, function_name);
    if (!last) return NULL;
    CnpArray *result = cnp_fft_axis_transform(
        last, last_axis - 1, preceding_length, inverse, function_name);
    cnp_array_decref(last);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_fft2(const CnpArray *a) {
    return fft_two_axes(a, false, "cnp_fft2");
}

CNP_API CnpArray* CNP_CALL cnp_ifft2(const CnpArray *a) {
    return fft_two_axes(a, true, "cnp_ifft2");
}

static CnpArray *fft_frequency(
    int64_t n, double d, bool real_only,
    const char *function_name) {
    if (n <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "frequency sample count must be positive");
        return NULL;
    }
    if (d == 0.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "sample spacing must not be zero");
        return NULL;
    }
    int64_t length = real_only ? n / 2 + 1 : n;
    int64_t shape[1] = {length};
    CnpArray *result = cnp_array_new(
        1, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    double scale = 1.0 / ((double)n * d);
    double *output = (double*)result->data;
    if (real_only) {
        for (int64_t index = 0; index < length; ++index)
            output[index] = (double)index * scale;
    } else {
        int64_t positive = (n - 1) / 2 + 1;
        for (int64_t index = 0; index < positive; ++index)
            output[index] = (double)index * scale;
        for (int64_t index = positive; index < n; ++index)
            output[index] = (double)(index - n) * scale;
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_fftfreq(int64_t n, double d) {
    return fft_frequency(n, d, false, "cnp_fftfreq");
}

CNP_API CnpArray* CNP_CALL cnp_rfftfreq(int64_t n, double d) {
    return fft_frequency(n, d, true, "cnp_rfftfreq");
}

static CnpArray *fft_shift_all_axes(
    const CnpArray *source, bool inverse,
    const char *function_name) {
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            source, function_name, &ignored_nbytes)) return NULL;
    if (source->ndim < 0 || source->ndim > CNP_MAXDIMS ||
            (source->ndim > 0 && (!source->shape || !source->strides)) ||
            (source->size > 0 && !source->data)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source array shape metadata and data buffer must be valid");
        return NULL;
    }
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape,
        source->dtype->type_num, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    int itemsize = source->dtype->elsize;
    int64_t destination_coordinates[CNP_MAXDIMS] = {0};
    int64_t source_coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < source->size; ++index) {
        for (int dimension = 0; dimension < source->ndim; ++dimension) {
            int64_t extent = source->shape[dimension];
            int64_t shift = extent / 2;
            int64_t source_shift = inverse ? shift : extent - shift;
            source_coordinates[dimension] = extent == 0
                ? 0
                : (destination_coordinates[dimension] + source_shift) % extent;
        }
        int64_t source_offset = source->offset + cnp_multi_to_offset(
            source->ndim, source_coordinates, source->strides);
        memcpy(
            (char*)result->data + index * itemsize,
            (const char*)source->data + source_offset,
            (size_t)itemsize);
        for (int dimension = source->ndim - 1;
                dimension >= 0; --dimension) {
            ++destination_coordinates[dimension];
            if (destination_coordinates[dimension] <
                    source->shape[dimension]) break;
            destination_coordinates[dimension] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_fftshift(const CnpArray *a) {
    return fft_shift_all_axes(a, false, "cnp_fftshift");
}

CNP_API CnpArray* CNP_CALL cnp_ifftshift(const CnpArray *a) {
    return fft_shift_all_axes(a, true, "cnp_ifftshift");
}
