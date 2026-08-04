#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "cnumpy/cnumpy_ahk.h"

enum { PIPELINE_ITERATIONS = 10000 };

static void print_native_error(const char *operation, int returned_status) {
    CnpErrorState error = {0};
    CNP_STATUS stored_status = cnp_get_error(&error);

    fprintf(stderr,
        "native_error operation=%s returned_status=%d status=%d "
        "function=%s message=%s\n",
        operation,
        returned_status,
        (int)stored_status,
        error.func,
        error.message);
}

static int require_status(int status, const char *operation) {
    if (status == CNP_OK) {
        return 1;
    }
    print_native_error(operation, status);
    return 0;
}

static int require_array(const CnpArray *array, const char *operation) {
    if (array != NULL) {
        return 1;
    }
    print_native_error(operation, CNP_ERR_GENERIC);
    return 0;
}

static void print_array(const char *label, const CnpArray *array) {
    int64_t size = cnp_array_size(array);
    int64_t index;

    printf("%s=[", label);
    for (index = 0; index < size; ++index) {
        printf("%s%g", index == 0 ? "" : ", ",
            cnp_array_flat_get(array, index));
    }
    printf("]\n");
}

static int array_matches(
    const CnpArray *array,
    const double *expected,
    int64_t count,
    double tolerance) {
    int64_t index;

    if (cnp_array_size(array) != count) {
        return 0;
    }
    for (index = 0; index < count; ++index) {
        if (fabs(cnp_array_flat_get(array, index) - expected[index])
                > tolerance) {
            return 0;
        }
    }
    return 1;
}

int main(void) {
    const int64_t shape[] = {4};
    const double power_samples[] = {3.0, 8.0, 15.0, 24.0};
    const double calibration_offsets[] = {1.0, 1.0, 1.0, 1.0};
    const double expected_amplitudes[] = {2.0, 3.0, 4.0, 5.0};
    const double expected_cumulative[] = {2.0, 5.0, 9.0, 14.0};
    CnpArray *samples = NULL;
    CnpArray *offsets = NULL;
    CnpArray *calibrated_power = NULL;
    CnpArray *amplitudes = NULL;
    CnpArray *cumulative = NULL;
    double amplitude_sum = 0.0;
    size_t baseline = 0;
    size_t hot_loop_baseline = 0;
    size_t hot_loop_final = 0;
    size_t retained = 0;
    int iteration;
    int exit_code = 0;

    if (!require_status(cnp_init(), "cnp_init")) {
        return 1;
    }
    baseline = cnp_get_allocated_memory();

    samples = cnp_array_from_data(
        power_samples, 1, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!require_array(samples, "cnp_array_from_data(samples)")) {
        exit_code = 1;
        goto cleanup;
    }
    offsets = cnp_array_from_data(
        calibration_offsets, 1, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!require_array(offsets, "cnp_array_from_data(offsets)")) {
        exit_code = 1;
        goto cleanup;
    }
    calibrated_power = cnp_array_empty(1, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!require_array(calibrated_power, "cnp_array_empty(calibrated_power)")) {
        exit_code = 1;
        goto cleanup;
    }
    amplitudes = cnp_array_empty(1, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!require_array(amplitudes, "cnp_array_empty(amplitudes)")) {
        exit_code = 1;
        goto cleanup;
    }
    cumulative = cnp_array_empty(1, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!require_array(cumulative, "cnp_array_empty(cumulative)")) {
        exit_code = 1;
        goto cleanup;
    }

    hot_loop_baseline = cnp_get_allocated_memory();
    for (iteration = 0; iteration < PIPELINE_ITERATIONS; ++iteration) {
        if (!require_status(
                cnp_ahk_add_into(samples, offsets, calibrated_power),
                "cnp_ahk_add_into")
            || !require_status(
                cnp_ahk_sqrt_into(calibrated_power, amplitudes),
                "cnp_ahk_sqrt_into")
            || !require_status(
                cnp_ahk_cumsum_into(amplitudes, -1, cumulative),
                "cnp_ahk_cumsum_into")
            || !require_status(
                cnp_ahk_sum_into_scalar(amplitudes, &amplitude_sum),
                "cnp_ahk_sum_into_scalar")) {
            fprintf(stderr, "failed_iteration=%d\n", iteration);
            exit_code = 1;
            goto cleanup;
        }
    }
    hot_loop_final = cnp_get_allocated_memory();
    if (hot_loop_final != hot_loop_baseline) {
        fprintf(stderr,
            "hot loop retained native bytes: baseline=%llu final=%llu\n",
            (unsigned long long)hot_loop_baseline,
            (unsigned long long)hot_loop_final);
        exit_code = 1;
        goto cleanup;
    }

    if (!array_matches(amplitudes, expected_amplitudes, 4, 1e-12)
        || !array_matches(cumulative, expected_cumulative, 4, 1e-12)
        || fabs(amplitude_sum - 14.0) > 1e-12) {
        fprintf(stderr, "pipeline produced unexpected numerical results\n");
        exit_code = 1;
        goto cleanup;
    }

    printf("iterations=%d\n", PIPELINE_ITERATIONS);
    print_array("amplitudes", amplitudes);
    print_array("cumulative", cumulative);
    printf("amplitude_sum=%g\n", amplitude_sum);
    printf("hot_loop_retained_bytes=0\n");

cleanup:
    if (cumulative != NULL) {
        cnp_array_free(cumulative);
    }
    if (amplitudes != NULL) {
        cnp_array_free(amplitudes);
    }
    if (calibrated_power != NULL) {
        cnp_array_free(calibrated_power);
    }
    if (offsets != NULL) {
        cnp_array_free(offsets);
    }
    if (samples != NULL) {
        cnp_array_free(samples);
    }

    retained = cnp_get_allocated_memory();
    if (retained != baseline) {
        fprintf(stderr,
            "retained native bytes: baseline=%llu final=%llu\n",
            (unsigned long long)baseline,
            (unsigned long long)retained);
        exit_code = 1;
    } else {
        printf("retained_bytes=0\n");
    }
    cnp_cleanup();
    return exit_code;
}
