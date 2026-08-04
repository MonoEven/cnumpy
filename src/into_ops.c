/**
 * Allocation-free destination operations for contiguous float64 arrays.
 */
#include <cnumpy/cnumpy_internal.h>

static CNP_STATUS validate_f64_contiguous(
    const CnpArray *arr, bool require_writeable,
    const char *func, const char *role) {
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, func, "%s array is null", role);
        return CNP_ERR_GENERIC;
    }
    if (!arr->dtype || arr->dtype->type_num != CNP_DOUBLE) {
        cnp_set_error(CNP_ERR_TYPE, func,
                      "%s array must have float64 dtype", role);
        return CNP_ERR_TYPE;
    }
    if (!(arr->flags & CNP_ARRAY_C_CONTIGUOUS)) {
        cnp_set_error(CNP_ERR_GENERIC, func,
                      "%s array must be C-contiguous", role);
        return CNP_ERR_GENERIC;
    }
    if (require_writeable && !(arr->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(CNP_ERR_GENERIC, func,
                      "%s array must be writeable", role);
        return CNP_ERR_GENERIC;
    }
    if (arr->size < 0 || arr->offset < 0) {
        cnp_set_error(CNP_ERR_SHAPE, func,
                      "%s array has invalid size or offset", role);
        return CNP_ERR_SHAPE;
    }
    if (arr->size > 0 && !arr->data) {
        cnp_set_error(CNP_ERR_GENERIC, func,
                      "%s array has no data buffer", role);
        return CNP_ERR_GENERIC;
    }
    return CNP_OK;
}

static CNP_STATUS validate_same_shape(
    const CnpArray *left, const CnpArray *right,
    const char *func) {
    if (left->ndim != right->ndim || left->size != right->size) {
        cnp_set_error(CNP_ERR_SHAPE, func,
                      "Array ranks or sizes do not match");
        return CNP_ERR_SHAPE;
    }
    for (int axis = 0; axis < left->ndim; ++axis) {
        if (left->shape[axis] != right->shape[axis]) {
            cnp_set_error(CNP_ERR_SHAPE, func,
                          "Array shapes differ at axis %d", axis);
            return CNP_ERR_SHAPE;
        }
    }
    return CNP_OK;
}

static CNP_STATUS classify_f64_output_overlap(
    const CnpArray *source, const CnpArray *out,
    const char *func, int *direction) {
    uintptr_t source_start = (uintptr_t)source->data + (uintptr_t)source->offset;
    uintptr_t out_start = (uintptr_t)out->data + (uintptr_t)out->offset;
    uintptr_t source_bytes;
    uintptr_t out_bytes;

    *direction = 0;
    if (source_start == out_start || source->size == 0 || out->size == 0)
        return CNP_OK;
    if ((uint64_t)source->size > UINTPTR_MAX / sizeof(double) ||
        (uint64_t)out->size > UINTPTR_MAX / sizeof(double)) {
        cnp_set_error(CNP_ERR_SHAPE, func, "Array byte range is too large");
        return CNP_ERR_SHAPE;
    }
    source_bytes = (uintptr_t)source->size * sizeof(double);
    out_bytes = (uintptr_t)out->size * sizeof(double);
    if (source_start > UINTPTR_MAX - source_bytes ||
        out_start > UINTPTR_MAX - out_bytes) {
        cnp_set_error(CNP_ERR_SHAPE, func, "Array byte range overflows address space");
        return CNP_ERR_SHAPE;
    }
    if (source_start < out_start + out_bytes &&
        out_start < source_start + source_bytes) {
        *direction = out_start < source_start ? 1 : -1;
    }
    return CNP_OK;
}

static CNP_STATUS validate_output_overlap(
    const CnpArray *source, const CnpArray *out,
    const char *func) {
    int direction;
    CNP_STATUS status = classify_f64_output_overlap(
        source, out, func, &direction);
    if (status != CNP_OK) return status;
    if (direction != 0) {
        cnp_set_error(CNP_ERR_GENERIC, func,
                      "Partially overlapping input and output are unsupported");
        return CNP_ERR_GENERIC;
    }
    return CNP_OK;
}

static bool can_use_f64_add_fast_path(
    const CnpArray *left,
    const CnpArray *right,
    const CnpArray *out) {
    if (!left || !right || !out ||
            !left->dtype || !right->dtype || !out->dtype)
        return false;
    if (left->dtype->type_num != CNP_DOUBLE ||
            right->dtype->type_num != CNP_DOUBLE ||
            out->dtype->type_num != CNP_DOUBLE)
        return false;
    if (!(left->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            !(right->flags & CNP_ARRAY_C_CONTIGUOUS) ||
            !(out->flags & CNP_ARRAY_C_CONTIGUOUS))
        return false;
    if (left->ndim != right->ndim || left->ndim != out->ndim ||
            left->size != right->size || left->size != out->size)
        return false;
    for (int axis = 0; axis < left->ndim; ++axis) {
        if (left->shape[axis] != right->shape[axis] ||
                left->shape[axis] != out->shape[axis])
            return false;
    }
    return true;
}

CNP_API CNP_STATUS CNP_CALL cnp_add_into(
    const CnpArray *left, const CnpArray *right, CnpArray *out) {
    const char *func = "cnp_add_into";
    if (!can_use_f64_add_fast_path(left, right, out))
        return cnp_add_into_promoted(left, right, out);

    CNP_STATUS status = validate_f64_contiguous(left, false, func, "left");
    if (status != CNP_OK) return status;
    status = validate_f64_contiguous(right, false, func, "right");
    if (status != CNP_OK) return status;
    status = validate_f64_contiguous(out, true, func, "output");
    if (status != CNP_OK) return status;
    int left_direction;
    int right_direction;
    status = classify_f64_output_overlap(
        left, out, func, &left_direction);
    if (status != CNP_OK) return status;
    status = classify_f64_output_overlap(
        right, out, func, &right_direction);
    if (status != CNP_OK) return status;
    if (left_direction != 0 && right_direction != 0 &&
            left_direction != right_direction)
        return cnp_add_into_promoted(left, right, out);

    const double *left_data =
        (const double *)((const unsigned char *)left->data + left->offset);
    const double *right_data =
        (const double *)((const unsigned char *)right->data + right->offset);
    double *out_data = (double *)((unsigned char *)out->data + out->offset);
    int direction = left_direction != 0
        ? left_direction : right_direction;
    if (direction < 0) {
        for (int64_t index = left->size; index-- > 0;)
            out_data[index] = left_data[index] + right_data[index];
    } else if (direction > 0) {
        for (int64_t index = 0; index < left->size; ++index)
            out_data[index] = left_data[index] + right_data[index];
    } else {
        cnp_simd_add(left_data, right_data, out_data, left->size);
    }
    return CNP_OK;
}

CNP_API CNP_STATUS CNP_CALL cnp_sqrt_into(
    const CnpArray *source, CnpArray *out) {
    const char *func = "cnp_sqrt_into";
    CNP_STATUS status = validate_f64_contiguous(source, false, func, "source");
    if (status != CNP_OK) return status;
    status = validate_f64_contiguous(out, true, func, "output");
    if (status != CNP_OK) return status;
    status = validate_same_shape(source, out, func);
    if (status != CNP_OK) return status;
    status = validate_output_overlap(source, out, func);
    if (status != CNP_OK) return status;

    const double *source_data =
        (const double *)((const unsigned char *)source->data + source->offset);
    double *out_data = (double *)((unsigned char *)out->data + out->offset);
    cnp_simd_sqrt(source_data, out_data, source->size);
    return CNP_OK;
}

CNP_API CNP_STATUS CNP_CALL cnp_cumsum_into(
    const CnpArray *source, int axis, CnpArray *out) {
    const char *func = "cnp_cumsum_into";
    CNP_STATUS status = validate_f64_contiguous(source, false, func, "source");
    if (status != CNP_OK) return status;
    status = validate_f64_contiguous(out, true, func, "output");
    if (status != CNP_OK) return status;
    if (axis == CNP_AXIS_NONE) {
        if (out->ndim != 1 || out->size != source->size) {
            cnp_set_error(CNP_ERR_SHAPE, func,
                          "Flattened output must be one-dimensional with %lld elements",
                          (long long)source->size);
            return CNP_ERR_SHAPE;
        }
    } else {
        if (source->ndim != 1 || out->ndim != 1) {
            cnp_set_error(CNP_ERR_SHAPE, func,
                          "Explicit-axis output requires one-dimensional arrays");
            return CNP_ERR_SHAPE;
        }
        if (axis != 0) {
            cnp_set_error(CNP_ERR_AXIS, func, "Axis must be -1 or 0");
            return CNP_ERR_AXIS;
        }
        status = validate_same_shape(source, out, func);
        if (status != CNP_OK) return status;
    }
    status = validate_output_overlap(source, out, func);
    if (status != CNP_OK) return status;

    const double *source_data =
        (const double *)((const unsigned char *)source->data + source->offset);
    double *out_data = (double *)((unsigned char *)out->data + out->offset);
    double accumulator = 0.0;
    for (int64_t index = 0; index < source->size; ++index) {
        accumulator += source_data[index];
        out_data[index] = accumulator;
    }
    return CNP_OK;
}

CNP_API CNP_STATUS CNP_CALL cnp_sum_into_scalar(
    const CnpArray *source, double *out_value) {
    const char *func = "cnp_sum_into_scalar";
    CNP_STATUS status = validate_f64_contiguous(source, false, func, "source");
    if (status != CNP_OK) return status;
    if (!out_value) {
        cnp_set_error(CNP_ERR_GENERIC, func, "Output scalar pointer is null");
        return CNP_ERR_GENERIC;
    }

    const double *source_data =
        (const double *)((const unsigned char *)source->data + source->offset);
    *out_value = cnp_reduction_sum_contiguous_double(
        source_data, source->size);
    return CNP_OK;
}
