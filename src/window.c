/**
 * cnumpy window functions and additional array operations
 * Corresponds to numpy window functions and array manipulation
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* =========================================================================
 * Window functions
 * ========================================================================= */

static CnpArray* window_new(int64_t length, const char *function_name) {
    int64_t shape[1] = {length > 0 ? length : 0};
    CnpArray *result = cnp_array_new(
        1, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

static double window_i0(double value) {
    double scaled_square = value * value / 4.0;
    double sum = 1.0;
    double term = 1.0;
    for (int k = 1; ; ++k) {
        term *= scaled_square / ((double)k * (double)k);
        sum += term;
        if (fabs(term) <= DBL_EPSILON * fabs(sum)) return sum;
    }
}

/* Bartlett window (triangular) */
CNP_API CnpArray* CNP_CALL cnp_bartlett(int64_t M) {
    CnpArray *result = window_new(M, "cnp_bartlett");
    if (!result || M <= 0) return result;

    double *data = (double*)result->data;
    if (M == 1) {
        data[0] = 1.0;
    } else {
        double n = (double)(M - 1);
        for (int64_t i = 0; i < M; i++) {
            data[i] = 1.0 - fabs(2.0 * i / n - 1.0);
        }
    }
    return result;
}

/* Blackman window */
CNP_API CnpArray* CNP_CALL cnp_blackman(int64_t M) {
    CnpArray *result = window_new(M, "cnp_blackman");
    if (!result || M <= 0) return result;

    double *data = (double*)result->data;
    if (M == 1) {
        data[0] = 1.0;
    } else {
        double n = (double)(M - 1);
        for (int64_t i = 0; i < M; i++) {
            double x = 2.0 * M_PI * i / n;
            data[i] = 0.42 - 0.5 * cos(x) + 0.08 * cos(2.0 * x);
        }
    }
    return result;
}

/* Hamming window */
CNP_API CnpArray* CNP_CALL cnp_hamming(int64_t M) {
    CnpArray *result = window_new(M, "cnp_hamming");
    if (!result || M <= 0) return result;

    double *data = (double*)result->data;
    if (M == 1) {
        data[0] = 1.0;
    } else {
        double n = (double)(M - 1);
        for (int64_t i = 0; i < M; i++) {
            data[i] = 0.54 - 0.46 * cos(2.0 * M_PI * i / n);
        }
    }
    return result;
}

/* Hanning window */
CNP_API CnpArray* CNP_CALL cnp_hanning(int64_t M) {
    CnpArray *result = window_new(M, "cnp_hanning");
    if (!result || M <= 0) return result;

    double *data = (double*)result->data;
    if (M == 1) {
        data[0] = 1.0;
    } else {
        double n = (double)(M - 1);
        for (int64_t i = 0; i < M; i++) {
            data[i] = 0.5 - 0.5 * cos(2.0 * M_PI * i / n);
        }
    }
    return result;
}

/* Kaiser window */
CNP_API CnpArray* CNP_CALL cnp_kaiser(int64_t M, double beta) {
    CnpArray *result = window_new(M, "cnp_kaiser");
    if (!result || M <= 0) return result;

    double *data = (double*)result->data;
    if (M == 1) {
        data[0] = 1.0;
    } else if (!isfinite(beta)) {
        for (int64_t i = 0; i < M; ++i) data[i] = NAN;
    } else {
        double alpha = (double)(M - 1) / 2.0;
        double i0_beta = window_i0(beta);

        for (int64_t i = 0; i < M; i++) {
            double r = (i - alpha) / alpha;
            double arg = beta * sqrt(1.0 - r * r);
            data[i] = window_i0(arg) / i0_beta;
        }
    }
    return result;
}

/* =========================================================================
 * Financial functions (numpy financial module)
 * ========================================================================= */

static bool financial_validate_when(int when, const char *function_name) {
    if (when == 0 || when == 1) return true;
    cnp_set_error(CNP_ERR_VALUE, function_name,
                  "when must be 0 (end) or 1 (begin), got %d", when);
    return false;
}

static bool financial_validate_finite(double value, const char *role,
                                      const char *function_name) {
    if (isfinite(value)) return true;
    cnp_set_error(CNP_ERR_VALUE, function_name,
                  "%s must be finite", role);
    return false;
}

static double financial_error(const char *function_name,
                              const char *message) {
    cnp_set_error(CNP_ERR_VALUE, function_name, "%s", message);
    return NAN;
}

/* fv - Future value */
CNP_API double CNP_CALL cnp_fv(double rate, int64_t nper, double pmt, double pv, int when) {
    const char *function_name = "cnp_fv";
    if (!financial_validate_when(when, function_name) ||
        !financial_validate_finite(rate, "rate", function_name) ||
        !financial_validate_finite(pmt, "pmt", function_name) ||
        !financial_validate_finite(pv, "pv", function_name)) return NAN;
    double result;
    if (rate == 0.0) {
        result = -(pv + pmt * nper);
    } else {
        double factor = pow(1.0 + rate, (double)nper);
        if (when == 1) { /* beginning of period */
            result = -(pv * factor + pmt * (1.0 + rate) * (factor - 1.0) / rate);
        } else { /* end of period */
            result = -(pv * factor + pmt * (factor - 1.0) / rate);
        }
    }
    if (!isfinite(result))
        return financial_error(function_name, "financial equation is outside the finite domain");
    return result;
}

/* pv - Present value */
CNP_API double CNP_CALL cnp_pv(double rate, int64_t nper, double pmt, double fv_val, int when) {
    const char *function_name = "cnp_pv";
    if (!financial_validate_when(when, function_name) ||
        !financial_validate_finite(rate, "rate", function_name) ||
        !financial_validate_finite(pmt, "pmt", function_name) ||
        !financial_validate_finite(fv_val, "fv", function_name)) return NAN;
    double result;
    if (rate == 0.0) {
        result = -(fv_val + pmt * nper);
    } else {
        double factor = pow(1.0 + rate, (double)nper);
        if (when == 1) {
            result = -(fv_val + pmt * (1.0 + rate) * (factor - 1.0) / rate) / factor;
        } else {
            result = -(fv_val + pmt * (factor - 1.0) / rate) / factor;
        }
    }
    if (!isfinite(result))
        return financial_error(function_name, "financial equation is outside the finite domain");
    return result;
}

/* pmt - Payment */
CNP_API double CNP_CALL cnp_pmt(double rate, int64_t nper, double pv_val, double fv_val, int when) {
    const char *function_name = "cnp_pmt";
    if (!financial_validate_when(when, function_name) ||
        !financial_validate_finite(rate, "rate", function_name) ||
        !financial_validate_finite(pv_val, "pv", function_name) ||
        !financial_validate_finite(fv_val, "fv", function_name)) return NAN;
    if (nper == 0)
        return financial_error(function_name, "nper must be nonzero");
    double result;
    if (rate == 0.0) {
        result = -(pv_val + fv_val) / nper;
    } else {
        double factor = pow(1.0 + rate, (double)nper);
        double temp = pv_val * factor + fv_val;
        if (when == 1) {
            result = -temp * rate / ((1.0 + rate) * (factor - 1.0));
        } else {
            result = -temp * rate / (factor - 1.0);
        }
    }
    if (!isfinite(result))
        return financial_error(function_name, "financial equation is outside the finite domain");
    return result;
}

/* nper - Number of periods */
CNP_API double CNP_CALL cnp_nper(double rate, double pmt, double pv_val, double fv_val, int when) {
    const char *function_name = "cnp_nper";
    if (!financial_validate_when(when, function_name) ||
        !financial_validate_finite(rate, "rate", function_name) ||
        !financial_validate_finite(pmt, "pmt", function_name) ||
        !financial_validate_finite(pv_val, "pv", function_name) ||
        !financial_validate_finite(fv_val, "fv", function_name)) return NAN;
    if (rate == 0.0) {
        if (pmt == 0.0)
            return financial_error(function_name, "pmt must be nonzero when rate is zero");
        return -(pv_val + fv_val) / pmt;
    }
    if (rate <= -1.0)
        return financial_error(function_name, "rate must be greater than -1");
    double z = pmt * (1.0 + rate * when) / rate;
    double ratio = (-fv_val + z) / (pv_val + z);
    if (!(ratio > 0.0) || !isfinite(ratio))
        return financial_error(function_name, "financial equation has no real period count");
    double result = log(ratio) / log1p(rate);
    if (!isfinite(result))
        return financial_error(function_name, "financial equation has no finite period count");
    return result;
}

/* rate - Interest rate (Newton's method) */
CNP_API double CNP_CALL cnp_rate(int64_t nper, double pmt, double pv_val, double fv_val, int when) {
    const char *function_name = "cnp_rate";
    if (!financial_validate_when(when, function_name) ||
        !financial_validate_finite(pmt, "pmt", function_name) ||
        !financial_validate_finite(pv_val, "pv", function_name) ||
        !financial_validate_finite(fv_val, "fv", function_name)) return NAN;
    if (nper <= 0)
        return financial_error(function_name, "nper must be positive");
    double zero_residual = fv_val + pv_val + pmt * (double)nper;
    double zero_scale = fabs(fv_val) + fabs(pv_val) +
        fabs(pmt) * (double)nper + 1.0;
    if (fabs(zero_residual) <= DBL_EPSILON * zero_scale) return 0.0;

    /* Newton-Raphson iteration */
    double rate = 0.1; /* initial guess */
    for (int iter = 0; iter < 100; iter++) {
        double factor = pow(1.0 + rate, (double)nper);
        double f, df;
        if (when == 1) {
            f = fv_val + pv_val * factor + pmt * (1.0 + rate) * (factor - 1.0) / rate;
            df = pv_val * nper * pow(1.0 + rate, (double)(nper-1)) +
                 pmt * (nper * pow(1.0 + rate, (double)(nper-1)) * (1.0 + rate) + (factor - 1.0)) / rate -
                 pmt * (1.0 + rate) * (factor - 1.0) / (rate * rate);
        } else {
            f = fv_val + pv_val * factor + pmt * (factor - 1.0) / rate;
            df = pv_val * nper * pow(1.0 + rate, (double)(nper-1)) +
                 pmt * (nper * pow(1.0 + rate, (double)(nper-1)) * rate - (factor - 1.0)) / (rate * rate);
        }
        if (!isfinite(f) || !isfinite(df) || fabs(df) < DBL_EPSILON)
            return financial_error(function_name, "rate iteration did not converge");
        double new_rate = rate - f / df;
        if (!isfinite(new_rate) || new_rate <= -1.0)
            return financial_error(function_name, "rate iteration left the real domain");
        if (fabs(new_rate - rate) < 1e-12) {
            return new_rate;
        }
        rate = new_rate;
    }
    return financial_error(function_name, "rate iteration did not converge");
}

/* npv - Net present value */
CNP_API double CNP_CALL cnp_npv(double rate, const double *values, int64_t n) {
    const char *function_name = "cnp_npv";
    if (!financial_validate_finite(rate, "rate", function_name)) return NAN;
    if (n < 0)
        return financial_error(function_name, "value count must be non-negative");
    if (n > 0 && !values)
        return financial_error(function_name, "values are required when count is positive");
    if (n > 1 && rate <= -1.0)
        return financial_error(function_name, "rate must be greater than -1");
    double result = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (!isfinite(values[i]))
            return financial_error(function_name, "cash flows must be finite");
        result += values[i] / pow(1.0 + rate, (double)i);
    }
    if (!isfinite(result))
        return financial_error(function_name, "net present value is not finite");
    return result;
}

/* irr - Internal rate of return (Newton's method) */
CNP_API double CNP_CALL cnp_irr(const double *values, int64_t n) {
    const char *function_name = "cnp_irr";
    if (!values)
        return financial_error(function_name, "values are required");
    if (n < 2)
        return financial_error(function_name, "at least two cash flows are required");
    bool has_positive = false;
    bool has_negative = false;
    for (int64_t i = 0; i < n; ++i) {
        if (!isfinite(values[i]))
            return financial_error(function_name, "cash flows must be finite");
        has_positive = has_positive || values[i] > 0.0;
        has_negative = has_negative || values[i] < 0.0;
    }
    if (!has_positive || !has_negative)
        return financial_error(function_name, "cash flows must contain both signs");

    double rate = 0.1;
    for (int iter = 0; iter < 100; iter++) {
        double npv = 0.0, dnpv = 0.0;
        for (int64_t i = 0; i < n; i++) {
            double factor = pow(1.0 + rate, (double)i);
            npv += values[i] / factor;
            if (i > 0) {
                dnpv -= i * values[i] / pow(1.0 + rate, (double)(i + 1));
            }
        }
        if (!isfinite(npv) || !isfinite(dnpv) || fabs(dnpv) < DBL_EPSILON)
            return financial_error(function_name, "IRR iteration did not converge");
        double new_rate = rate - npv / dnpv;
        if (!isfinite(new_rate) || new_rate <= -1.0)
            return financial_error(function_name, "IRR iteration left the real domain");
        if (fabs(new_rate - rate) < 1e-12) {
            return new_rate;
        }
        rate = new_rate;
    }
    return financial_error(function_name, "IRR iteration did not converge");
}

/* =========================================================================
 * Additional array functions
 * ========================================================================= */

static int64_t mutation_flat_offset(
    const CnpArray *array, int64_t flat_index) {
    int64_t coordinates[CNP_MAXDIMS] = {0};
    int64_t remaining = flat_index;
    for (int dimension = array->ndim - 1; dimension >= 0; --dimension) {
        coordinates[dimension] = remaining % array->shape[dimension];
        remaining /= array->shape[dimension];
    }
    return array->offset + cnp_multi_to_offset(
        array->ndim, coordinates, array->strides);
}

static void mutation_release_arrays(CnpArray **arrays, int count) {
    if (!arrays) return;
    for (int index = 0; index < count; ++index) {
        if (arrays[index]) {
            cnp_array_free(arrays[index]);
            arrays[index] = NULL;
        }
    }
}

/* place - Change elements based on condition */
CNP_API CNP_STATUS CNP_CALL cnp_place(CnpArray *arr, const CnpArray *mask, const CnpArray *values) {
    const char *function_name = "cnp_place";
    if (!arr || !mask || !values) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "array, mask, and values are required");
        return CNP_ERR_GENERIC;
    }
    if (!(arr->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "destination array is not writeable");
        return CNP_ERR_GENERIC;
    }
    if (!cnp_dtype_can_cast(
            values->dtype->type_num,
            arr->dtype->type_num,
            CNP_CAST_SAFE)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "values dtype cannot safely cast to the destination dtype");
        return CNP_ERR_TYPE;
    }
    if (mask->size != arr->size) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "mask and destination must have the same size");
        return CNP_ERR_SHAPE;
    }

    bool has_selected = false;
    for (int64_t index = 0; index < mask->size; ++index) {
        uint8_t selected = 0;
        int64_t mask_offset = mutation_flat_offset(mask, index);
        CNP_STATUS status = cnp_cast_scalar_value(
            (const char*)mask->data + mask_offset,
            mask->dtype->type_num,
            &selected, CNP_BOOL, function_name);
        if (status != CNP_OK) return status;
        if (selected) has_selected = true;
    }
    if (has_selected && values->size == 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "values cannot be empty when the mask selects elements");
        return CNP_ERR_VALUE;
    }
    if (!has_selected) return CNP_OK;

    CnpArray *snapshot = cnp_array_copy(arr);
    if (!snapshot) {
        cnp_relabel_error(function_name);
        return cnp_get_error(NULL);
    }

    int64_t value_index = 0;
    CNP_STATUS status = CNP_OK;
    for (int64_t index = 0; index < mask->size; ++index) {
        uint8_t selected = 0;
        int64_t mask_offset = mutation_flat_offset(mask, index);
        status = cnp_cast_scalar_value(
            (const char*)mask->data + mask_offset,
            mask->dtype->type_num,
            &selected, CNP_BOOL, function_name);
        if (status != CNP_OK) break;
        if (!selected) continue;

        int64_t source_offset = mutation_flat_offset(
            values, value_index % values->size);
        int64_t destination_offset = mutation_flat_offset(snapshot, index);
        status = cnp_cast_scalar_value(
            (const char*)values->data + source_offset,
            values->dtype->type_num,
            (char*)snapshot->data + destination_offset,
            snapshot->dtype->type_num,
            function_name);
        if (status != CNP_OK) break;
        ++value_index;
    }
    if (status == CNP_OK) {
        status = cnp_copyto(arr, snapshot, CNP_CAST_NO);
        if (status != CNP_OK) cnp_relabel_error(function_name);
    }
    cnp_array_free(snapshot);
    return status;
}

/* put - Put values into array at indices */
CNP_API CNP_STATUS CNP_CALL cnp_put(CnpArray *arr, const CnpArray *indices, const CnpArray *values, const char *mode) {
    const char *function_name = "cnp_put";
    if (!arr || !indices || !values) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "array, indices, and values are required");
        return CNP_ERR_GENERIC;
    }
    if (!(arr->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "destination array is not writeable");
        return CNP_ERR_GENERIC;
    }
    if (!cnp_dtype_can_cast(
            values->dtype->type_num,
            arr->dtype->type_num,
            CNP_CAST_SAFE)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "values dtype cannot safely cast to the destination dtype");
        return CNP_ERR_TYPE;
    }
    if (!mode) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "mode is required and must be 'raise', 'wrap', or 'clip'");
        return CNP_ERR_VALUE;
    }

    bool raise_mode = strcmp(mode, "raise") == 0;
    bool wrap_mode = strcmp(mode, "wrap") == 0;
    bool clip_mode = strcmp(mode, "clip") == 0;
    if (!raise_mode && !wrap_mode && !clip_mode) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "mode must be 'raise', 'wrap', or 'clip'");
        return CNP_ERR_VALUE;
    }

    CNP_TYPE index_type = indices->dtype->type_num;
    bool supported_indices = index_type == CNP_BOOL ||
        (cnp_type_is_integer(index_type) &&
         index_type != CNP_ULONG && index_type != CNP_ULONGLONG);
    if (!supported_indices) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "indices dtype must safely cast to int64");
        return CNP_ERR_TYPE;
    }
    if (indices->size == 0 || values->size == 0) return CNP_OK;

    CnpArray *snapshot = cnp_array_copy(arr);
    if (!snapshot) {
        cnp_relabel_error(function_name);
        return cnp_get_error(NULL);
    }

    CNP_STATUS status = CNP_OK;
    for (int64_t index = 0; index < indices->size; ++index) {
        int64_t source_index = 0;
        int64_t index_offset = mutation_flat_offset(indices, index);
        status = cnp_cast_scalar_value(
            (const char*)indices->data + index_offset,
            index_type,
            &source_index, CNP_LONGLONG, function_name);
        if (status != CNP_OK) break;

        int64_t destination_index = source_index;
        if (arr->size == 0) {
            cnp_set_error(
                CNP_ERR_INDEX, function_name,
                "cannot put into an empty destination array");
            status = CNP_ERR_INDEX;
            break;
        }
        if (raise_mode) {
            if (destination_index < -arr->size ||
                    destination_index >= arr->size) {
                cnp_set_error(
                    CNP_ERR_INDEX, function_name,
                    "index %lld is out of bounds for flattened size %lld",
                    (long long)source_index, (long long)arr->size);
                status = CNP_ERR_INDEX;
                break;
            }
            if (destination_index < 0) destination_index += arr->size;
        } else if (wrap_mode) {
            destination_index %= arr->size;
            if (destination_index < 0) destination_index += arr->size;
        } else {
            if (destination_index < 0) destination_index = 0;
            if (destination_index >= arr->size)
                destination_index = arr->size - 1;
        }

        int64_t value_offset = mutation_flat_offset(
            values, index % values->size);
        int64_t destination_offset = mutation_flat_offset(
            snapshot, destination_index);
        status = cnp_cast_scalar_value(
            (const char*)values->data + value_offset,
            values->dtype->type_num,
            (char*)snapshot->data + destination_offset,
            snapshot->dtype->type_num,
            function_name);
        if (status != CNP_OK) break;
    }
    if (status == CNP_OK) {
        status = cnp_copyto(arr, snapshot, CNP_CAST_NO);
        if (status != CNP_OK) cnp_relabel_error(function_name);
    }
    cnp_array_free(snapshot);
    return status;
}

/* take - Take elements from array at indices */
CNP_API CnpArray* CNP_CALL cnp_take(const CnpArray *arr, const CnpArray *indices, int axis) {
    return cnp_take_v2(arr, indices, axis, axis < 0);
}

/* compress - Return selected elements along axis */
CNP_API CnpArray* CNP_CALL cnp_compress(const CnpArray *condition, const CnpArray *arr, int axis) {
    return cnp_compress_v2(condition, arr, axis, axis < 0);
}

/* choose - Construct array from index array and set of choices */
CNP_API CnpArray* CNP_CALL cnp_choose(const CnpArray *indices, int nchoices, const CnpArray **choices) {
    const char *function_name = "cnp_choose";
    if (!indices || nchoices <= 0 || !choices) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "indices and at least one choice are required");
        return NULL;
    }

    CNP_TYPE index_type = indices->dtype->type_num;
    bool supported_indices = index_type == CNP_BOOL ||
        (cnp_type_is_integer(index_type) &&
         index_type != CNP_ULONG && index_type != CNP_ULONGLONG);
    if (!supported_indices) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "indices dtype must safely cast to int64");
        return NULL;
    }
    for (int choice_index = 0; choice_index < nchoices; ++choice_index) {
        if (!choices[choice_index]) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "choice %d is required", choice_index);
            return NULL;
        }
    }

    CNP_TYPE result_type = cnp_result_type(nchoices, choices);
    if (result_type == CNP_NOTYPE) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int array_count = nchoices + 1;
    size_t pointer_bytes = (size_t)array_count * sizeof(CnpArray*);
    CnpArray **inputs = (CnpArray**)cnp_malloc(pointer_bytes);
    CnpArray **broadcasted = (CnpArray**)cnp_calloc(
        (size_t)array_count, sizeof(CnpArray*));
    if (!inputs || !broadcasted) {
        cnp_free(inputs, pointer_bytes);
        cnp_free(broadcasted, pointer_bytes);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate broadcast operand storage");
        return NULL;
    }
    inputs[0] = (CnpArray*)indices;
    for (int choice_index = 0; choice_index < nchoices; ++choice_index)
        inputs[choice_index + 1] = (CnpArray*)choices[choice_index];

    CNP_STATUS status = cnp_broadcast_arrays_v2(
        array_count, inputs, broadcasted, array_count);
    if (status != CNP_OK) {
        mutation_release_arrays(broadcasted, array_count);
        cnp_free(inputs, pointer_bytes);
        cnp_free(broadcasted, pointer_bytes);
        cnp_relabel_error(function_name);
        return NULL;
    }

    CnpArray *result = cnp_array_new(
        broadcasted[0]->ndim, broadcasted[0]->shape,
        result_type, CNP_ORDER_C);
    if (!result) {
        mutation_release_arrays(broadcasted, array_count);
        cnp_free(inputs, pointer_bytes);
        cnp_free(broadcasted, pointer_bytes);
        cnp_relabel_error(function_name);
        return NULL;
    }

    for (int64_t index = 0; index < result->size; ++index) {
        int64_t selected_index = 0;
        int64_t index_offset = mutation_flat_offset(broadcasted[0], index);
        status = cnp_cast_scalar_value(
            (const char*)broadcasted[0]->data + index_offset,
            index_type,
            &selected_index, CNP_LONGLONG, function_name);
        if (status != CNP_OK) break;
        if (selected_index < 0 || selected_index >= nchoices) {
            cnp_set_error(
                CNP_ERR_INDEX, function_name,
                "choice index %lld is outside [0, %d)",
                (long long)selected_index, nchoices);
            status = CNP_ERR_INDEX;
            break;
        }

        const CnpArray *selected = broadcasted[selected_index + 1];
        int64_t selected_offset = mutation_flat_offset(selected, index);
        int64_t result_offset = mutation_flat_offset(result, index);
        status = cnp_cast_scalar_value(
            (const char*)selected->data + selected_offset,
            selected->dtype->type_num,
            (char*)result->data + result_offset,
            result->dtype->type_num,
            function_name);
        if (status != CNP_OK) break;
    }

    mutation_release_arrays(broadcasted, array_count);
    cnp_free(inputs, pointer_bytes);
    cnp_free(broadcasted, pointer_bytes);
    if (status != CNP_OK) {
        cnp_array_free(result);
        cnp_relabel_error(function_name);
        return NULL;
    }
    return result;
}
