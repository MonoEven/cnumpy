/**
 * cnumpy remaining math and array functions
 * Corresponds to numpy: heaviside, signbit, unwrap, dsplit, array_split,
 *   block, row_stack, histogram2d, nanquantile, nanpercentile (array version)
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>

/* =========================================================================
 * cnp_unwrap - Unwrap phase angles
 * numpy.unwrap(p, discont=pi, axis=-1)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_unwrap(const CnpArray *p, double discont) {
    const char *function_name = "cnp_unwrap";
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            p, function_name, &ignored_nbytes)) return NULL;
    if (p->ndim <= 0 || p->ndim > CNP_MAXDIMS ||
            !p->shape || !p->strides ||
            (p->size > 0 && !p->data)) {
        cnp_set_error(
            CNP_ERR_AXIS, function_name,
            "source must have a valid last axis");
        return NULL;
    }
    CNP_TYPE source_type = p->dtype->type_num;
    if (!(source_type == CNP_BOOL ||
          cnp_type_is_integer(source_type) ||
          cnp_type_is_float(source_type))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array must have a real numeric dtype");
        return NULL;
    }
    CNP_TYPE result_type = cnp_type_is_float(source_type)
        ? source_type : CNP_DOUBLE;
    CnpArray *result = cnp_array_new(
        p->ndim, p->shape, result_type, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t axis_size = p->shape[p->ndim - 1];
    if (axis_size == 0) return result;
    int64_t lines = p->size / axis_size;
    const double pi = 3.14159265358979323846;
    const double period = 2.0 * pi;
    double threshold = discont < pi ? pi : discont;
    for (int64_t line = 0; line < lines; ++line) {
        int64_t first_index = line * axis_size;
        double previous = cnp_array_flat_get(p, first_index);
        cnp_set_element_double(
            result->data,
            first_index * result->dtype->elsize,
            result_type, previous);
        double cumulative_correction = 0.0;
        for (int64_t position = 1; position < axis_size; ++position) {
            int64_t index = first_index + position;
            double current = cnp_array_flat_get(p, index);
            double difference = current - previous;
            double wrapped = fmod(difference + pi, period);
            if (wrapped < 0.0) wrapped += period;
            wrapped -= pi;
            if (wrapped == -pi && difference > 0.0) wrapped = pi;
            double correction = wrapped - difference;
            if (fabs(difference) < threshold) correction = 0.0;
            cumulative_correction += correction;
            cnp_set_element_double(
                result->data,
                index * result->dtype->elsize,
                result_type, current + cumulative_correction);
            previous = current;
        }
    }
    return result;
}

/* =========================================================================
 * cnp_row_stack - Stack arrays as rows (alias for vstack)
 * numpy.row_stack(tup)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_row_stack(int narrays, CnpArray **arrays) {
    return cnp_vstack(narrays, arrays);
}

/* =========================================================================
 * cnp_block - Assemble array from nested lists of blocks
 * numpy.block(list) - simplified 2D version
 * ========================================================================= */
static CnpArray* block_impl(
    int nrows, int ncols, CnpArray **blocks, const char *function_name) {
    if (nrows <= 0 || ncols <= 0 || !blocks) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "positive row/column counts and blocks are required");
        return NULL;
    }

    CNP_TYPE result_type = CNP_NOTYPE;
    for (int row = 0; row < nrows; ++row) {
        for (int column = 0; column < ncols; ++column) {
            CnpArray *block = blocks[row * ncols + column];
            if (!block || block->ndim != 2 || !block->dtype) {
                cnp_set_error(
                    CNP_ERR_SHAPE, function_name,
                    "block (%d, %d) must be a two-dimensional array",
                    row, column);
                return NULL;
            }
            if (column > 0 && block->shape[0] != blocks[row * ncols]->shape[0]) {
                cnp_set_error(
                    CNP_ERR_SHAPE, function_name,
                    "blocks in row %d must have equal heights", row);
                return NULL;
            }
            if (row > 0 && block->shape[1] != blocks[column]->shape[1]) {
                cnp_set_error(
                    CNP_ERR_SHAPE, function_name,
                    "blocks in column %d must have equal widths", column);
                return NULL;
            }
            if (result_type == CNP_NOTYPE) {
                result_type = block->dtype->type_num;
            } else {
                result_type = cnp_promote_types_public(
                    result_type, block->dtype->type_num);
                if (result_type == CNP_NOTYPE) {
                    cnp_relabel_error(function_name);
                    return NULL;
                }
            }
        }
    }

    /* Compute total shape */
    int64_t total_rows = 0, total_cols = 0;
    for (int r = 0; r < nrows; r++) {
        int64_t row_height = blocks[r * ncols]->shape[0];
        if (row_height > INT64_MAX - total_rows) {
            cnp_set_error(CNP_ERR_SHAPE, function_name, "row size overflows int64");
            return NULL;
        }
        total_rows += row_height;
    }
    for (int c = 0; c < ncols; c++) {
        int64_t col_width = blocks[c]->shape[1];
        if (col_width > INT64_MAX - total_cols) {
            cnp_set_error(CNP_ERR_SHAPE, function_name, "column size overflows int64");
            return NULL;
        }
        total_cols += col_width;
    }

    int64_t shape[2] = {total_rows, total_cols};
    CnpArray *result = cnp_array_new(2, shape, result_type, CNP_ORDER_C);
    if (!result) return NULL;

    int64_t row_offset = 0;
    for (int r = 0; r < nrows; r++) {
        int64_t col_offset = 0;
        int64_t row_height = blocks[r * ncols]->shape[0];
        for (int c = 0; c < ncols; c++) {
            CnpArray *blk = blocks[r * ncols + c];
            int64_t blk_h = blk->shape[0];
            int64_t blk_w = blk->shape[1];
            for (int64_t i = 0; i < blk_h; i++) {
                for (int64_t j = 0; j < blk_w; j++) {
                    int64_t source_coordinates[2] = {i, j};
                    int64_t source_offset = blk->offset + cnp_multi_to_offset(
                        2, source_coordinates, blk->strides);
                    int64_t destination_index =
                        (row_offset + i) * total_cols + col_offset + j;
                    void *destination = (char*)result->data +
                        destination_index * result->dtype->elsize;
                    CNP_STATUS status = cnp_cast_scalar_value(
                        (const char*)blk->data + source_offset,
                        blk->dtype->type_num,
                        destination, result_type, function_name);
                    if (status != CNP_OK) {
                        cnp_array_decref(result);
                        return NULL;
                    }
                }
            }
            col_offset += blk_w;
        }
        row_offset += row_height;
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_block(int nrows, int ncols, CnpArray **blocks) {
    return block_impl(nrows, ncols, blocks, "cnp_block");
}

/* =========================================================================
 * cnp_histogram2d - 2D histogram
 * numpy.histogram2d(x, y, bins=10)
 * ========================================================================= */
static bool histogram2d_input_valid(
    const CnpArray *array, const char *role,
    const char *function_name) {
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            array, function_name, &ignored_nbytes)) return false;
    if (array->ndim != 1 || !array->shape || !array->strides ||
            (array->size > 0 && !array->data)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "%s must be a valid one-dimensional array", role);
        return false;
    }
    CNP_TYPE type = array->dtype->type_num;
    if (!(type == CNP_BOOL ||
          cnp_type_is_integer(type) ||
          cnp_type_is_float(type))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "%s must have a real numeric dtype", role);
        return false;
    }
    return true;
}

static bool histogram2d_autodetect_range(
    const CnpArray *array, const char *role,
    double *minimum, double *maximum,
    const char *function_name) {
    if (array->size == 0) {
        *minimum = 0.0;
        *maximum = 1.0;
        return true;
    }
    *minimum = cnp_array_flat_get(array, 0);
    *maximum = *minimum;
    for (int64_t index = 0; index < array->size; ++index) {
        double value = cnp_array_flat_get(array, index);
        if (!isfinite(value)) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "%s autodetected range must be finite", role);
            return false;
        }
        if (value < *minimum) *minimum = value;
        if (value > *maximum) *maximum = value;
    }
    if (*minimum == *maximum) {
        *minimum -= 0.5;
        *maximum += 0.5;
    }
    return true;
}

CNP_API CnpArray* CNP_CALL cnp_histogram2d(const CnpArray *x, const CnpArray *y, int64_t bins) {
    const char *function_name = "cnp_histogram2d";
    if (!histogram2d_input_valid(x, "x", function_name) ||
            !histogram2d_input_valid(y, "y", function_name)) return NULL;
    if (x->size != y->size) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "x and y must have the same length");
        return NULL;
    }
    if (bins <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "bin count must be positive");
        return NULL;
    }

    double xmin;
    double xmax;
    double ymin;
    double ymax;
    if (!histogram2d_autodetect_range(
            x, "x", &xmin, &xmax, function_name) ||
        !histogram2d_autodetect_range(
            y, "y", &ymin, &ymax, function_name)) return NULL;

    int64_t shape[2] = {bins, bins};
    CnpArray *result = cnp_array_zeros(2, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    double xspan = xmax - xmin;
    double yspan = ymax - ymin;
    for (int64_t index = 0; index < x->size; ++index) {
        double xvalue = cnp_array_flat_get(x, index);
        double yvalue = cnp_array_flat_get(y, index);
        int64_t xbin = xvalue == xmax
            ? bins - 1
            : (int64_t)(((xvalue - xmin) / xspan) * (double)bins);
        int64_t ybin = yvalue == ymax
            ? bins - 1
            : (int64_t)(((yvalue - ymin) / yspan) * (double)bins);
        if (xbin < 0 || xbin >= bins || ybin < 0 || ybin >= bins) continue;
        ((double*)result->data)[xbin * bins + ybin] += 1.0;
    }
    return result;
}

/* =========================================================================
 * cnp_nanquantile - Quantile ignoring NaN
 * numpy.nanquantile(a, q, axis=None)
 * ========================================================================= */
CNP_API double CNP_CALL cnp_nanquantile(const CnpArray *arr, double q, int axis) {
    const char *function_name = "cnp_nanquantile";
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NAN;
    }
    CnpArray *result = cnp_nanquantile_v2(
        arr, q, axis, axis == CNP_AXIS_NONE);
    if (!result) {
        cnp_relabel_error(function_name);
        return NAN;
    }
    if (result->ndim != 0) {
        cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "legacy scalar return cannot represent an array result");
        return NAN;
    }
    double value = cnp_array_get_double(result, NULL);
    cnp_array_free(result);
    return value;
}

/* =========================================================================
 * cnp_bitwise_count - Count number of set bits
 * numpy.bitwise_count(x)
 * ========================================================================= */
static int64_t bitwise_count_flat_offset(
    const CnpArray *array, int64_t flat_index) {
    int64_t offset = array->offset;
    for (int dimension = array->ndim - 1; dimension >= 0; --dimension) {
        int64_t extent = array->shape[dimension];
        int64_t coordinate = extent > 0 ? flat_index % extent : 0;
        if (extent > 0) flat_index /= extent;
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

static uint64_t bitwise_count_magnitude(
    const void *source, CNP_TYPE type) {
    int64_t signed_value;
    switch (type) {
        case CNP_BOOL:
        case CNP_BYTE:
            signed_value = *(const int8_t*)source;
            break;
        case CNP_UBYTE:
            return *(const uint8_t*)source;
        case CNP_SHORT:
            signed_value = *(const int16_t*)source;
            break;
        case CNP_USHORT:
            return *(const uint16_t*)source;
        case CNP_INT:
            signed_value = *(const int32_t*)source;
            break;
        case CNP_UINT:
            return *(const uint32_t*)source;
        case CNP_LONG:
        case CNP_LONGLONG:
            signed_value = *(const int64_t*)source;
            break;
        case CNP_ULONG:
        case CNP_ULONGLONG:
            return *(const uint64_t*)source;
        default:
            return 0;
    }
    return signed_value < 0
        ? UINT64_C(0) - (uint64_t)signed_value
        : (uint64_t)signed_value;
}

CNP_API CnpArray* CNP_CALL cnp_bitwise_count(const CnpArray *arr) {
    const char *function_name = "cnp_bitwise_count";
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            arr, function_name, &ignored_nbytes)) return NULL;
    if (arr->ndim < 0 || arr->ndim > CNP_MAXDIMS ||
            (arr->ndim > 0 && (!arr->shape || !arr->strides)) ||
            (arr->size > 0 && !arr->data)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "source array shape metadata and data buffer must be valid");
        return NULL;
    }
    if (arr->dtype->type_num != CNP_BOOL &&
            !cnp_type_is_integer(arr->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array must have an integer or boolean dtype");
        return NULL;
    }
    CnpArray *result = cnp_array_new(
        arr->ndim, arr->shape, CNP_UBYTE, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    uint8_t *out = (uint8_t*)result->data;
    for (int64_t index = 0; index < arr->size; ++index) {
        int64_t offset = bitwise_count_flat_offset(arr, index);
        uint64_t value = bitwise_count_magnitude(
            (const char*)arr->data + offset, arr->dtype->type_num);
        uint8_t count = 0;
        while (value != 0) {
            value &= value - UINT64_C(1);
            ++count;
        }
        out[index] = count;
    }
    return result;
}
