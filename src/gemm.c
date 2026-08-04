/**
 * Contiguous float64 GEMM scheduling and private Win32 thread-pool lifecycle.
 */
#include <cnumpy/cnumpy_internal.h>
#include <windows.h>

#define CNP_GEMM_PARALLEL_MIN_WORK INT64_C(16777216)
#define CNP_GEMM_PARALLEL_ROW_BLOCK INT64_C(32)

typedef struct {
    const double *a;
    const double *b;
    double *c;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t row_begin;
    int64_t row_end;
    volatile LONG completion_status;
} CnpGemmWorkItem;

static SRWLOCK g_gemm_pool_lock = SRWLOCK_INIT;
static PTP_POOL g_gemm_pool = NULL;
static PTP_CLEANUP_GROUP g_gemm_cleanup_group = NULL;
static TP_CALLBACK_ENVIRON g_gemm_callback_environment;
static bool g_gemm_environment_initialized = false;
static DWORD g_gemm_auto_thread_count = 0;
static int g_gemm_configured_thread_count = 0;

static CNP_STATUS cnp_gemm_set_win32_error(
    const char *function_name, const char *operation, DWORD error_code) {
    cnp_set_error(CNP_ERR_MEMORY, function_name,
                  "%s failed with Win32 error %lu",
                  operation, (unsigned long)error_code);
    return CNP_ERR_MEMORY;
}

static VOID CALLBACK cnp_gemm_work_callback(
    PTP_CALLBACK_INSTANCE instance, PVOID context, PTP_WORK work) {
    CnpGemmWorkItem *item = (CnpGemmWorkItem*)context;
    (void)instance;
    (void)work;
    cnp_simd_gemm_tile(
        item->a, item->b, item->c,
        item->m, item->n, item->k,
        item->row_begin, item->row_end);
    item->completion_status = (LONG)CNP_OK;
}

static void cnp_gemm_destroy_pool_locked(void) {
    if (g_gemm_cleanup_group) {
        CloseThreadpoolCleanupGroupMembers(
            g_gemm_cleanup_group, TRUE, NULL);
        CloseThreadpoolCleanupGroup(g_gemm_cleanup_group);
        g_gemm_cleanup_group = NULL;
    }
    if (g_gemm_environment_initialized) {
        DestroyThreadpoolEnvironment(&g_gemm_callback_environment);
        g_gemm_environment_initialized = false;
    }
    if (g_gemm_pool) {
        CloseThreadpool(g_gemm_pool);
        g_gemm_pool = NULL;
    }
    g_gemm_auto_thread_count = 0;
    g_gemm_configured_thread_count = 0;
}

CNP_STATUS cnp_gemm_thread_pool_init(void) {
    const char *function_name = "cnp_gemm_thread_pool_init";
    CNP_STATUS status = CNP_OK;
    AcquireSRWLockExclusive(&g_gemm_pool_lock);
    if (g_gemm_pool) {
        ReleaseSRWLockExclusive(&g_gemm_pool_lock);
        return CNP_OK;
    }

    DWORD active_processors = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (active_processors == 0) {
        status = cnp_gemm_set_win32_error(
            function_name, "GetActiveProcessorCount", GetLastError());
        goto complete;
    }

    g_gemm_pool = CreateThreadpool(NULL);
    if (!g_gemm_pool) {
        status = cnp_gemm_set_win32_error(
            function_name, "CreateThreadpool", GetLastError());
        goto complete;
    }
    g_gemm_cleanup_group = CreateThreadpoolCleanupGroup();
    if (!g_gemm_cleanup_group) {
        status = cnp_gemm_set_win32_error(
            function_name, "CreateThreadpoolCleanupGroup", GetLastError());
        goto complete;
    }

    InitializeThreadpoolEnvironment(&g_gemm_callback_environment);
    g_gemm_environment_initialized = true;
    SetThreadpoolCallbackPool(&g_gemm_callback_environment, g_gemm_pool);
    SetThreadpoolCallbackCleanupGroup(
        &g_gemm_callback_environment, g_gemm_cleanup_group, NULL);
    SetThreadpoolThreadMaximum(g_gemm_pool, active_processors);
    if (!SetThreadpoolThreadMinimum(g_gemm_pool, 1)) {
        status = cnp_gemm_set_win32_error(
            function_name, "SetThreadpoolThreadMinimum", GetLastError());
        goto complete;
    }

    g_gemm_auto_thread_count = active_processors;
    g_gemm_configured_thread_count = 0;

complete:
    if (status != CNP_OK) cnp_gemm_destroy_pool_locked();
    ReleaseSRWLockExclusive(&g_gemm_pool_lock);
    return status;
}

void cnp_gemm_thread_pool_cleanup(void) {
    AcquireSRWLockExclusive(&g_gemm_pool_lock);
    cnp_gemm_destroy_pool_locked();
    ReleaseSRWLockExclusive(&g_gemm_pool_lock);
}

CNP_API CNP_STATUS CNP_CALL cnp_set_num_threads(int count) {
    const char *function_name = "cnp_set_num_threads";
    if (count < 0) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "thread count must be non-negative");
        return CNP_ERR_GENERIC;
    }

    AcquireSRWLockExclusive(&g_gemm_pool_lock);
    if (!g_gemm_pool) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "GEMM thread pool is not initialized");
        ReleaseSRWLockExclusive(&g_gemm_pool_lock);
        return CNP_ERR_GENERIC;
    }
    DWORD effective_count = count == 0
        ? g_gemm_auto_thread_count : (DWORD)count;
    SetThreadpoolThreadMaximum(g_gemm_pool, effective_count);
    g_gemm_configured_thread_count = count;
    ReleaseSRWLockExclusive(&g_gemm_pool_lock);
    return CNP_OK;
}

CNP_API int CNP_CALL cnp_get_num_threads(void) {
    int count;
    AcquireSRWLockShared(&g_gemm_pool_lock);
    if (!g_gemm_pool) {
        ReleaseSRWLockShared(&g_gemm_pool_lock);
        cnp_set_error(CNP_ERR_GENERIC, "cnp_get_num_threads",
                      "GEMM thread pool is not initialized");
        return -1;
    }
    count = g_gemm_configured_thread_count;
    ReleaseSRWLockShared(&g_gemm_pool_lock);
    return count;
}

static bool cnp_gemm_reaches_parallel_threshold(
    int64_t m, int64_t n, int64_t k) {
    if (n > CNP_GEMM_PARALLEL_MIN_WORK / k) return true;
    int64_t work_per_row = n * k;
    int64_t required_rows =
        (CNP_GEMM_PARALLEL_MIN_WORK - 1) / work_per_row + 1;
    return m >= required_rows;
}

CNP_STATUS cnp_gemm_f64(
    const double *a, const double *b, double *c,
    int64_t m, int64_t n, int64_t k) {
    const char *function_name = "cnp_gemm_f64";
    CNP_STATUS status = CNP_OK;
    CnpGemmWorkItem *items = NULL;
    PTP_WORK *work_handles = NULL;
    size_t allocation_count = 0;

    if (m < 0 || n < 0 || k < 0) {
        cnp_set_error(CNP_ERR_SHAPE, function_name,
                      "GEMM dimensions must be non-negative");
        return CNP_ERR_SHAPE;
    }
    if (m == 0 || n == 0 || k == 0) return CNP_OK;
    if (!a || !b || !c) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "GEMM data pointers must not be null");
        return CNP_ERR_GENERIC;
    }

    AcquireSRWLockShared(&g_gemm_pool_lock);
    int configured_count = g_gemm_configured_thread_count;
    ReleaseSRWLockShared(&g_gemm_pool_lock);
    if (configured_count == 1
            || !cnp_gemm_reaches_parallel_threshold(m, n, k)) {
        cnp_simd_gemm_tile(a, b, c, m, n, k, 0, m);
        return CNP_OK;
    }

    AcquireSRWLockExclusive(&g_gemm_pool_lock);
    DWORD effective_count = g_gemm_configured_thread_count == 0
        ? g_gemm_auto_thread_count
        : (DWORD)g_gemm_configured_thread_count;
    int64_t row_block_count =
        (m - 1) / CNP_GEMM_PARALLEL_ROW_BLOCK + 1;
    int64_t task_count64 = row_block_count < (int64_t)effective_count
        ? row_block_count : (int64_t)effective_count;
    if (task_count64 <= 1) {
        ReleaseSRWLockExclusive(&g_gemm_pool_lock);
        cnp_simd_gemm_tile(a, b, c, m, n, k, 0, m);
        return CNP_OK;
    }
    if (!g_gemm_pool || !g_gemm_cleanup_group) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "parallel GEMM requested without an initialized thread pool");
        status = CNP_ERR_GENERIC;
        goto complete;
    }
    if ((uint64_t)task_count64 > SIZE_MAX / sizeof(*items)
            || (uint64_t)task_count64 > SIZE_MAX / sizeof(*work_handles)) {
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "parallel GEMM work-item allocation size overflow");
        status = CNP_ERR_MEMORY;
        goto complete;
    }

    allocation_count = (size_t)task_count64;
    items = (CnpGemmWorkItem*)cnp_malloc(
        allocation_count * sizeof(*items));
    work_handles = (PTP_WORK*)cnp_malloc(
        allocation_count * sizeof(*work_handles));
    if (!items || !work_handles) {
        cnp_set_error(CNP_ERR_MEMORY, function_name,
                      "parallel GEMM work-item allocation failed");
        status = CNP_ERR_MEMORY;
        goto complete;
    }

    int64_t blocks_per_task = row_block_count / task_count64;
    int64_t extra_blocks = row_block_count % task_count64;
    int64_t next_block = 0;
    for (size_t index = 0; index < allocation_count; ++index) {
        int64_t item_blocks = blocks_per_task
            + ((int64_t)index < extra_blocks ? 1 : 0);
        items[index].a = a;
        items[index].b = b;
        items[index].c = c;
        items[index].m = m;
        items[index].n = n;
        items[index].k = k;
        items[index].row_begin =
            next_block * CNP_GEMM_PARALLEL_ROW_BLOCK;
        next_block += item_blocks;
        items[index].row_end = next_block == row_block_count
            ? m : next_block * CNP_GEMM_PARALLEL_ROW_BLOCK;
        items[index].completion_status = (LONG)CNP_ERR_GENERIC;
        work_handles[index] = CreateThreadpoolWork(
            cnp_gemm_work_callback, &items[index],
            &g_gemm_callback_environment);
        if (!work_handles[index]) {
            status = cnp_gemm_set_win32_error(
                function_name, "CreateThreadpoolWork", GetLastError());
            CloseThreadpoolCleanupGroupMembers(
                g_gemm_cleanup_group, TRUE, NULL);
            goto complete;
        }
    }

    for (size_t index = 0; index < allocation_count; ++index)
        SubmitThreadpoolWork(work_handles[index]);
    CloseThreadpoolCleanupGroupMembers(
        g_gemm_cleanup_group, FALSE, NULL);
    for (size_t index = 0; index < allocation_count; ++index) {
        if (items[index].completion_status != (LONG)CNP_OK) {
            cnp_set_error(CNP_ERR_GENERIC, function_name,
                          "parallel GEMM worker did not complete");
            status = CNP_ERR_GENERIC;
            break;
        }
    }

complete:
    if (work_handles)
        cnp_free(work_handles, allocation_count * sizeof(*work_handles));
    if (items)
        cnp_free(items, allocation_count * sizeof(*items));
    ReleaseSRWLockExclusive(&g_gemm_pool_lock);
    return status;
}
