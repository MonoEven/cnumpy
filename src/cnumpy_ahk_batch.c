/**
 * Direct AHK status wrappers and fixed-layout sequential batch dispatcher.
 */
#include <cnumpy/cnumpy_ahk.h>
#include <cnumpy/cnumpy_internal.h>

typedef char CnpAhkBatchCommandSizeCheck[
    sizeof(CnpAhkBatchCommand) == 40 ? 1 : -1];

CNP_API int CNP_CALL cnp_ahk_add_into(
    void *left, void *right, void *out) {
    cnp_clear_error();
    return (int)cnp_add_into(
        (const CnpArray *)left,
        (const CnpArray *)right,
        (CnpArray *)out);
}

CNP_API int CNP_CALL cnp_ahk_sqrt_into(void *source, void *out) {
    cnp_clear_error();
    return (int)cnp_sqrt_into(
        (const CnpArray *)source,
        (CnpArray *)out);
}

CNP_API int CNP_CALL cnp_ahk_cumsum_into(
    void *source, int axis, void *out) {
    cnp_clear_error();
    return (int)cnp_cumsum_into(
        (const CnpArray *)source,
        axis,
        (CnpArray *)out);
}

CNP_API int CNP_CALL cnp_ahk_sum_into_scalar(
    void *source, double *out_value) {
    cnp_clear_error();
    return (int)cnp_sum_into_scalar(
        (const CnpArray *)source,
        out_value);
}

CNP_API int CNP_CALL cnp_ahk_execute_batch(
    const CnpAhkBatchCommand *commands,
    int64_t command_count,
    int64_t *failed_index) {
    cnp_clear_error();
    if (!failed_index) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_ahk_execute_batch",
                      "failed_index pointer is null");
        return (int)CNP_ERR_GENERIC;
    }
    *failed_index = -1;
    if (!commands) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_ahk_execute_batch",
                      "Command array is null");
        return (int)CNP_ERR_GENERIC;
    }
    if (command_count <= 0) {
        cnp_set_error(CNP_ERR_GENERIC, "cnp_ahk_execute_batch",
                      "Command count must be positive");
        return (int)CNP_ERR_GENERIC;
    }

    for (int64_t i = 0; i < command_count; ++i) {
        const CnpAhkBatchCommand *command = &commands[i];
        CNP_STATUS status = CNP_OK;

        if (command->reserved != 0) {
            cnp_set_error(CNP_ERR_GENERIC, "cnp_ahk_execute_batch",
                          "Command %lld has nonzero reserved metadata",
                          (long long)i);
            *failed_index = i;
            return (int)CNP_ERR_GENERIC;
        }

        switch (command->opcode) {
            case CNP_AHK_BATCH_ADD_INTO:
                if (!command->input0 || !command->input1 || !command->output ||
                    command->axis != 0) {
                    cnp_set_error(CNP_ERR_GENERIC, "cnp_ahk_execute_batch",
                                  "Malformed add command at index %lld",
                                  (long long)i);
                    status = CNP_ERR_GENERIC;
                } else {
                    status = cnp_add_into(
                        (const CnpArray *)command->input0,
                        (const CnpArray *)command->input1,
                        (CnpArray *)command->output);
                }
                break;

            case CNP_AHK_BATCH_SQRT_INTO:
                if (!command->input0 || command->input1 || !command->output ||
                    command->axis != 0) {
                    cnp_set_error(CNP_ERR_GENERIC, "cnp_ahk_execute_batch",
                                  "Malformed sqrt command at index %lld",
                                  (long long)i);
                    status = CNP_ERR_GENERIC;
                } else {
                    status = cnp_sqrt_into(
                        (const CnpArray *)command->input0,
                        (CnpArray *)command->output);
                }
                break;

            case CNP_AHK_BATCH_CUMSUM_INTO:
                if (!command->input0 || command->input1 || !command->output ||
                    (command->axis != CNP_AXIS_NONE && command->axis != 0)) {
                    cnp_set_error(CNP_ERR_GENERIC, "cnp_ahk_execute_batch",
                                  "Malformed cumsum command at index %lld",
                                  (long long)i);
                    status = CNP_ERR_GENERIC;
                } else {
                    status = cnp_cumsum_into(
                        (const CnpArray *)command->input0,
                        (int)command->axis,
                        (CnpArray *)command->output);
                }
                break;

            case CNP_AHK_BATCH_SUM_SCALAR:
                if (!command->input0 || command->input1 || !command->output ||
                    command->axis != CNP_AXIS_NONE) {
                    cnp_set_error(CNP_ERR_GENERIC, "cnp_ahk_execute_batch",
                                  "Malformed sum-scalar command at index %lld",
                                  (long long)i);
                    status = CNP_ERR_GENERIC;
                } else {
                    status = cnp_sum_into_scalar(
                        (const CnpArray *)command->input0,
                        (double *)command->output);
                }
                break;

            default:
                cnp_set_error(CNP_ERR_GENERIC, "cnp_ahk_execute_batch",
                              "Unknown batch opcode %u at index %lld",
                              command->opcode, (long long)i);
                status = CNP_ERR_GENERIC;
                break;
        }

        if (status != CNP_OK) {
            *failed_index = i;
            return (int)status;
        }
    }
    return (int)CNP_OK;
}
