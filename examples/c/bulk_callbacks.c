#include <stdint.h>
#include <stdio.h>

#include "cnumpy/cnumpy_ahk.h"

typedef struct {
    const double *values;
    int64_t count;
    int64_t cursor;
} IteratorState;

static CNP_STATUS CNP_CALL summarize_lines(
    const double *lines,
    int64_t line_count,
    int64_t line_length,
    double *results,
    int64_t result_capacity,
    int64_t *produced_count,
    void *userdata) {
    int64_t line_index;
    (void)userdata;

    if (result_capacity != line_count * 2) {
        *produced_count = 0;
        return CNP_ERR_VALUE;
    }
    for (line_index = 0; line_index < line_count; ++line_index) {
        const double *line = lines + line_index * line_length;
        double sum = 0.0;
        int64_t item;
        for (item = 0; item < line_length; ++item) {
            sum += line[item];
        }
        results[line_index * 2] = sum;
        results[line_index * 2 + 1] = line[0] - line[line_length - 1];
    }
    *produced_count = result_capacity;
    return CNP_OK;
}

static CNP_STATUS CNP_CALL next_values(
    double *results,
    int64_t result_capacity,
    int64_t *produced_count,
    void *userdata) {
    IteratorState *state = (IteratorState *)userdata;
    int64_t remaining = state->count - state->cursor;
    int64_t count = remaining < result_capacity ? remaining : result_capacity;
    int64_t index;

    for (index = 0; index < count; ++index) {
        results[index] = state->values[state->cursor + index];
    }
    state->cursor += count;
    *produced_count = count;
    return CNP_OK;
}

static CNP_STATUS CNP_CALL reject_values(
    const double *values,
    int64_t value_count,
    double *results,
    int64_t result_capacity,
    int64_t *produced_count,
    void *userdata) {
    (void)values;
    (void)value_count;
    (void)results;
    (void)result_capacity;
    (void)userdata;
    *produced_count = 0;
    return CNP_ERR_VALUE;
}

static void print_array(const char *label, const CnpArray *array) {
    int ndim = cnp_array_ndim(array);
    const int64_t *shape = cnp_array_shape(array);
    int64_t size = cnp_array_size(array);
    int dimension;
    int64_t index;

    printf("%s shape=[", label);
    for (dimension = 0; dimension < ndim; ++dimension) {
        printf("%s%lld", dimension == 0 ? "" : ", ",
            (long long)shape[dimension]);
    }
    printf("] values=[");
    for (index = 0; index < size; ++index) {
        printf("%s%g", index == 0 ? "" : ", ",
            cnp_array_flat_get(array, index));
    }
    printf("]\n");
}

static void print_native_error(const char *label) {
    CnpErrorState error = {0};
    CNP_STATUS status = cnp_get_error(&error);
    fprintf(stderr, "%s status=%d function=%s message=%s\n",
        label, (int)status, error.func, error.message);
}

int main(void) {
    const int64_t source_shape[] = {2, 3};
    const double source_values[] = {1, 2, 3, 4, 5, 6};
    const int64_t callback_shape[] = {2};
    const double iterator_values[] = {1.5, -2.25, 7.0};
    IteratorState iterator = {
        iterator_values,
        (int64_t)(sizeof(iterator_values) / sizeof(iterator_values[0])),
        0
    };
    CnpArray *source = NULL;
    CnpArray *applied = NULL;
    CnpArray *iterated = NULL;
    CnpArray *unexpected = NULL;
    size_t baseline;
    size_t before_failure;
    size_t retained;
    int exit_code = 0;

    if (cnp_init() != CNP_OK) {
        print_native_error("cnp_init");
        return 1;
    }
    baseline = cnp_get_allocated_memory();

    source = cnp_array_from_data(
        source_values, 2, source_shape, CNP_DOUBLE, CNP_ORDER_C);
    if (source == NULL) {
        print_native_error("cnp_array_from_data");
        exit_code = 1;
        goto cleanup;
    }

    applied = (CnpArray *)cnp_ahk_apply_along_axis_v2(
        summarize_lines, NULL, 1, source, 1, callback_shape);
    if (applied == NULL) {
        print_native_error("cnp_ahk_apply_along_axis_v2");
        exit_code = 1;
        goto cleanup;
    }

    iterated = (CnpArray *)cnp_ahk_fromiter_v2(
        next_values, &iterator, -1, CNP_DOUBLE);
    if (iterated == NULL) {
        print_native_error("cnp_ahk_fromiter_v2");
        exit_code = 1;
        goto cleanup;
    }

    print_array("apply_along_axis_v2", applied);
    print_array("fromiter_v2_unknown_count", iterated);

    before_failure = cnp_get_allocated_memory();
    cnp_clear_error();
    unexpected = (CnpArray *)cnp_ahk_vectorize_v2(
        reject_values, NULL, source);
    if (unexpected != NULL) {
        fprintf(stderr, "callback failure unexpectedly returned an array\n");
        exit_code = 1;
        goto cleanup;
    }
    print_native_error("expected_callback_error");
    if (cnp_get_error(NULL) != CNP_ERR_VALUE) {
        fprintf(stderr, "callback failure did not preserve CNP_ERR_VALUE\n");
        exit_code = 1;
        goto cleanup;
    }
    if (cnp_get_allocated_memory() != before_failure) {
        fprintf(stderr, "callback failure retained native memory\n");
        exit_code = 1;
        goto cleanup;
    }

cleanup:
    if (unexpected != NULL) {
        cnp_array_free(unexpected);
    }
    if (iterated != NULL) {
        cnp_array_free(iterated);
    }
    if (applied != NULL) {
        cnp_array_free(applied);
    }
    if (source != NULL) {
        cnp_array_free(source);
    }
    retained = cnp_get_allocated_memory();
    if (retained != baseline) {
        fprintf(stderr, "retained native bytes: baseline=%llu final=%llu\n",
            (unsigned long long)baseline, (unsigned long long)retained);
        exit_code = 1;
    } else {
        printf("retained_bytes=0\n");
    }
    cnp_cleanup();
    return exit_code;
}
