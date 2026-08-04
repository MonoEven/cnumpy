/**
 * cnumpy core implementation - memory, error handling, dtype, array creation
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <windows.h>

/* Global state */
size_t g_cnp_allocated_memory = 0;
static CnpErrorState g_error_state = { CNP_OK, "", "" };
static bool g_initialized = false;

/* Static dtype table */
static CnpDtype g_dtype_table[CNP_NTYPES];
static bool g_dtype_initialized = false;

/* NumPy 1.25 casting relations for the CNP_TYPE descriptors represented by
 * this ABI. Bit n describes whether a source dtype can cast to CNP_TYPE n+1.
 * CNP_LONG/CNP_LONGLONG and CNP_ULONG/CNP_ULONGLONG are equivalent aliases;
 * MSVC also represents long double and long complex with double precision. */
static const uint32_t g_equiv_cast_masks[CNP_NTYPES] = {
    UINT32_C(0x000000),
    UINT32_C(0x000001), UINT32_C(0x000002), UINT32_C(0x000004),
    UINT32_C(0x000008), UINT32_C(0x000010), UINT32_C(0x000020),
    UINT32_C(0x000040), UINT32_C(0x000280), UINT32_C(0x000500),
    UINT32_C(0x000280), UINT32_C(0x000500), UINT32_C(0x000800),
    UINT32_C(0x003000), UINT32_C(0x003000), UINT32_C(0x004000),
    UINT32_C(0x018000), UINT32_C(0x018000), UINT32_C(0x020000),
    UINT32_C(0x040000), UINT32_C(0x080000), UINT32_C(0x100000),
    UINT32_C(0x200000), UINT32_C(0x400000), UINT32_C(0x800000)
};

static const uint32_t g_safe_cast_masks[CNP_NTYPES] = {
    UINT32_C(0x000000),
    UINT32_C(0xDFFFFF), UINT32_C(0xDFFAAA), UINT32_C(0xDFFFFC),
    UINT32_C(0x5FFAA8), UINT32_C(0x5FFFF0), UINT32_C(0x5FB2A0),
    UINT32_C(0x5FB7C0), UINT32_C(0x5FB280), UINT32_C(0x1FB500),
    UINT32_C(0x5FB280), UINT32_C(0x1FB500), UINT32_C(0x1FF800),
    UINT32_C(0x1FB000), UINT32_C(0x1FB000), UINT32_C(0x1FC000),
    UINT32_C(0x1F8000), UINT32_C(0x1F8000), UINT32_C(0x020000),
    UINT32_C(0x1E0000), UINT32_C(0x1A0000), UINT32_C(0x120000),
    UINT32_C(0x320000), UINT32_C(0x520000), UINT32_C(0x9FF800)
};

static const uint32_t g_same_kind_cast_masks[CNP_NTYPES] = {
    UINT32_C(0x000000),
    UINT32_C(0xDFFFFF), UINT32_C(0xDFFAAA), UINT32_C(0xDFFFFE),
    UINT32_C(0xDFFAAA), UINT32_C(0xDFFFFE), UINT32_C(0xDFFAAA),
    UINT32_C(0xDFFFFE), UINT32_C(0xDFFAAA), UINT32_C(0xDFFFFE),
    UINT32_C(0xDFFAAA), UINT32_C(0xDFFFFE), UINT32_C(0x9FF800),
    UINT32_C(0x9FF800), UINT32_C(0x9FF800), UINT32_C(0x1FC000),
    UINT32_C(0x1FC000), UINT32_C(0x1FC000), UINT32_C(0x020000),
    UINT32_C(0x1E0000), UINT32_C(0x1A0000), UINT32_C(0x120000),
    UINT32_C(0x320000), UINT32_C(0x520000), UINT32_C(0x9FF800)
};

/* =========================================================================
 * Memory management
 * ========================================================================= */
void* cnp_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr) g_cnp_allocated_memory += size;
    return ptr;
}

void* cnp_calloc(size_t count, size_t size) {
    void *ptr = calloc(count, size);
    if (ptr) g_cnp_allocated_memory += count * size;
    return ptr;
}

void* cnp_realloc(void *ptr, size_t old_size, size_t new_size) {
    void *new_ptr = realloc(ptr, new_size);
    if (new_ptr) {
        g_cnp_allocated_memory -= old_size;
        g_cnp_allocated_memory += new_size;
    }
    return new_ptr;
}

void cnp_free(void *ptr, size_t size) {
    if (ptr) {
        free(ptr);
        g_cnp_allocated_memory -= size;
    }
}

void* cnp_virtual_alloc(size_t size) {
    void *ptr = VirtualAlloc(
        NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!ptr) {
        cnp_set_error(CNP_ERR_MEMORY, "cnp_virtual_alloc",
                      "VirtualAlloc failed with Win32 error %lu",
                      GetLastError());
        return NULL;
    }
    g_cnp_allocated_memory += size;
    return ptr;
}

CNP_STATUS cnp_virtual_free(void *ptr, size_t size) {
    if (!VirtualFree(ptr, 0, MEM_RELEASE)) {
        cnp_set_error(CNP_ERR_MEMORY, "cnp_virtual_free",
                      "VirtualFree failed with Win32 error %lu",
                      GetLastError());
        return CNP_ERR_MEMORY;
    }
    g_cnp_allocated_memory -= size;
    return CNP_OK;
}

/* =========================================================================
 * Error handling
 * ========================================================================= */
void cnp_set_error(CNP_STATUS status, const char *func, const char *fmt, ...) {
    va_list args;
    g_error_state.status = status;
    strncpy(g_error_state.func, func, sizeof(g_error_state.func) - 1);
    g_error_state.func[sizeof(g_error_state.func) - 1] = '\0';
    va_start(args, fmt);
    vsnprintf(g_error_state.message, sizeof(g_error_state.message), fmt, args);
    va_end(args);
}

void cnp_relabel_error(const char *func) {
    if (g_error_state.status == CNP_OK) return;
    strncpy(g_error_state.func, func, sizeof(g_error_state.func) - 1);
    g_error_state.func[sizeof(g_error_state.func) - 1] = '\0';
}

CNP_API CNP_STATUS CNP_CALL cnp_get_error(CnpErrorState *state) {
    if (state) *state = g_error_state;
    return g_error_state.status;
}

CNP_API void CNP_CALL cnp_clear_error(void) {
    g_error_state.status = CNP_OK;
    g_error_state.message[0] = '\0';
    g_error_state.func[0] = '\0';
}

CNP_API const char* CNP_CALL cnp_get_error_message(void) {
    return g_error_state.message;
}

/* =========================================================================
 * Library initialization
 * ========================================================================= */
static void init_dtype_table(void) {
    /* Always reinitialize - ensures correctness regardless of DLL load mechanism.
     * This is needed because AHK's DllCall auto-load may not properly preserve
     * static variable state. The cost is negligible (~24 struct assignments). */
    memset(g_dtype_table, 0, sizeof(g_dtype_table));

    /* Bool */
    g_dtype_table[CNP_BOOL] = (CnpDtype){CNP_BOOL, 1, 1, 'b', '=', "bool", 1};
    /* Signed integers */
    g_dtype_table[CNP_BYTE] = (CnpDtype){CNP_BYTE, 1, 1, 'i', '=', "int8", 1};
    g_dtype_table[CNP_SHORT] = (CnpDtype){CNP_SHORT, 2, 2, 'i', '=', "int16", 1};
    g_dtype_table[CNP_INT] = (CnpDtype){CNP_INT, 4, 4, 'i', '=', "int32", 1};
    g_dtype_table[CNP_LONG] = (CnpDtype){CNP_LONG, 8, 8, 'i', '=', "int64", 1};
    g_dtype_table[CNP_LONGLONG] = (CnpDtype){CNP_LONGLONG, 8, 8, 'i', '=', "int64", 1};
    /* Unsigned integers */
    g_dtype_table[CNP_UBYTE] = (CnpDtype){CNP_UBYTE, 1, 1, 'u', '=', "uint8", 1};
    g_dtype_table[CNP_USHORT] = (CnpDtype){CNP_USHORT, 2, 2, 'u', '=', "uint16", 1};
    g_dtype_table[CNP_UINT] = (CnpDtype){CNP_UINT, 4, 4, 'u', '=', "uint32", 1};
    g_dtype_table[CNP_ULONG] = (CnpDtype){CNP_ULONG, 8, 8, 'u', '=', "uint64", 1};
    g_dtype_table[CNP_ULONGLONG] = (CnpDtype){CNP_ULONGLONG, 8, 8, 'u', '=', "uint64", 1};
    /* Floating point */
    g_dtype_table[CNP_HALF] = (CnpDtype){CNP_HALF, 2, 2, 'f', '=', "float16", 1};
    g_dtype_table[CNP_FLOAT] = (CnpDtype){CNP_FLOAT, 4, 4, 'f', '=', "float32", 1};
    g_dtype_table[CNP_DOUBLE] = (CnpDtype){CNP_DOUBLE, 8, 8, 'f', '=', "float64", 1};
    g_dtype_table[CNP_LONGDOUBLE] = (CnpDtype){
        CNP_LONGDOUBLE, sizeof(long double), sizeof(long double),
        'f', '=', "longdouble", 1};
    /* Complex */
    g_dtype_table[CNP_CFLOAT] = (CnpDtype){CNP_CFLOAT, 8, 4, 'c', '=', "complex64", 1};
    g_dtype_table[CNP_CDOUBLE] = (CnpDtype){CNP_CDOUBLE, 16, 8, 'c', '=', "complex128", 1};
    g_dtype_table[CNP_CLONGDOUBLE] = (CnpDtype){
        CNP_CLONGDOUBLE, sizeof(cnp_clongdouble), sizeof(long double),
        'c', '=', "clongdouble", 1};
    /* String/Object */
    g_dtype_table[CNP_STRING] = (CnpDtype){CNP_STRING, 0, 1, 'S', '=', "bytes", 1};
    g_dtype_table[CNP_UNICODE] = (CnpDtype){CNP_UNICODE, 0, 4, 'U', '=', "str", 1};
    g_dtype_table[CNP_OBJECT] = (CnpDtype){
        CNP_OBJECT, sizeof(void*), sizeof(void*), 'O', '=', "object", 1};
    g_dtype_table[CNP_VOID] = (CnpDtype){CNP_VOID, 0, 1, 'V', '=', "void", 1};
    /* Date/time scalars use signed 64-bit unit counts. */
    g_dtype_table[CNP_DATETIME] = (CnpDtype){
        CNP_DATETIME, sizeof(int64_t), sizeof(int64_t),
        'M', '=', "datetime64", 1};
    g_dtype_table[CNP_TIMEDELTA] = (CnpDtype){
        CNP_TIMEDELTA, sizeof(int64_t), sizeof(int64_t),
        'm', '=', "timedelta64", 1};

    g_dtype_initialized = true;
}

CNP_API CNP_STATUS CNP_CALL cnp_init(void) {
    CNP_STATUS dispatch_status = cnp_simd_init_dispatch();
    if (dispatch_status != CNP_OK) return dispatch_status;
    CNP_STATUS thread_pool_status = cnp_gemm_thread_pool_init();
    if (thread_pool_status != CNP_OK) return thread_pool_status;
    /* Always reinitialize to ensure correctness across DLL load mechanisms */
    init_dtype_table();
    /* Initialize random state with default seed */
    cnp_random_seed(12345);
    g_initialized = true;
    return CNP_OK;
}

CNP_API void CNP_CALL cnp_cleanup(void) {
    cnp_reset_string_functions();
    cnp_structured_cleanup();
    cnp_gemm_thread_pool_cleanup();
    g_initialized = false;
}

CNP_API const char* CNP_CALL cnp_version(void) {
    return CNUMPY_VERSION_STRING;
}

CNP_API size_t CNP_CALL cnp_get_allocated_memory(void) {
    return g_cnp_allocated_memory;
}

/* =========================================================================
 * Dtype implementation
 * ========================================================================= */
CNP_API CnpDtype* CNP_CALL cnp_dtype_new(CNP_TYPE type_num) {
    if (type_num <= 0 || type_num >= CNP_NTYPES) {
        cnp_set_error(
            CNP_ERR_TYPE, "cnp_dtype_new",
            "dtype %d is not a valid CNP_TYPE", (int)type_num);
        return NULL;
    }
    init_dtype_table();
    return &g_dtype_table[type_num];
}

CNP_API CnpDtype* CNP_CALL cnp_dtype_from_char(char c) {
    init_dtype_table();
    switch (c) {
        case 'b': return &g_dtype_table[CNP_BYTE];
        case 'B': return &g_dtype_table[CNP_UBYTE];
        case 'h': return &g_dtype_table[CNP_SHORT];
        case 'H': return &g_dtype_table[CNP_USHORT];
        case 'i': return &g_dtype_table[CNP_INT];
        case 'I': return &g_dtype_table[CNP_UINT];
        case 'l': return &g_dtype_table[CNP_LONG];
        case 'L': return &g_dtype_table[CNP_ULONG];
        case 'q': return &g_dtype_table[CNP_LONGLONG];
        case 'Q': return &g_dtype_table[CNP_ULONGLONG];
        case 'f': return &g_dtype_table[CNP_FLOAT];
        case 'd': return &g_dtype_table[CNP_DOUBLE];
        case 'g': return &g_dtype_table[CNP_LONGDOUBLE];
        case 'e': return &g_dtype_table[CNP_HALF];
        case 'F': return &g_dtype_table[CNP_CFLOAT];
        case 'D': return &g_dtype_table[CNP_CDOUBLE];
        case 'G': return &g_dtype_table[CNP_CLONGDOUBLE];
        case '?': return &g_dtype_table[CNP_BOOL];
        case 'S': return &g_dtype_table[CNP_STRING];
        case 'U': return &g_dtype_table[CNP_UNICODE];
        case 'O': return &g_dtype_table[CNP_OBJECT];
        case 'V': return &g_dtype_table[CNP_VOID];
        case 'M': return &g_dtype_table[CNP_DATETIME];
        case 'm': return &g_dtype_table[CNP_TIMEDELTA];
        default:
            cnp_set_error(
                CNP_ERR_TYPE, "cnp_dtype_from_char",
                "dtype character '%c' is not recognized", c);
            return NULL;
    }
}

CNP_API CnpDtype* CNP_CALL cnp_dtype_from_string(const char *name) {
    init_dtype_table();
    if (!name) {
        cnp_set_error(
            CNP_ERR_TYPE, "cnp_dtype_from_string",
            "dtype name is required");
        return NULL;
    }
    if (strcmp(name, "bool") == 0 || strcmp(name, "bool8") == 0) return &g_dtype_table[CNP_BOOL];
    if (strcmp(name, "int8") == 0 || strcmp(name, "byte") == 0) return &g_dtype_table[CNP_BYTE];
    if (strcmp(name, "uint8") == 0 || strcmp(name, "ubyte") == 0) return &g_dtype_table[CNP_UBYTE];
    if (strcmp(name, "int16") == 0 || strcmp(name, "short") == 0) return &g_dtype_table[CNP_SHORT];
    if (strcmp(name, "uint16") == 0 || strcmp(name, "ushort") == 0) return &g_dtype_table[CNP_USHORT];
    if (strcmp(name, "int32") == 0 || strcmp(name, "int") == 0 || strcmp(name, "intc") == 0) return &g_dtype_table[CNP_INT];
    if (strcmp(name, "uint32") == 0 || strcmp(name, "uint") == 0 || strcmp(name, "uintc") == 0) return &g_dtype_table[CNP_UINT];
    if (strcmp(name, "int64") == 0 || strcmp(name, "long") == 0 || strcmp(name, "longlong") == 0) return &g_dtype_table[CNP_LONGLONG];
    if (strcmp(name, "uint64") == 0 || strcmp(name, "ulong") == 0 || strcmp(name, "ulonglong") == 0) return &g_dtype_table[CNP_ULONGLONG];
    if (strcmp(name, "float16") == 0 || strcmp(name, "half") == 0) return &g_dtype_table[CNP_HALF];
    if (strcmp(name, "float32") == 0 || strcmp(name, "float") == 0 || strcmp(name, "single") == 0) return &g_dtype_table[CNP_FLOAT];
    if (strcmp(name, "float64") == 0 || strcmp(name, "double") == 0) return &g_dtype_table[CNP_DOUBLE];
    if (strcmp(name, "longdouble") == 0 || strcmp(name, "float128") == 0) return &g_dtype_table[CNP_LONGDOUBLE];
    if (strcmp(name, "complex64") == 0 || strcmp(name, "csingle") == 0) return &g_dtype_table[CNP_CFLOAT];
    if (strcmp(name, "complex128") == 0 || strcmp(name, "complex") == 0 || strcmp(name, "cdouble") == 0) return &g_dtype_table[CNP_CDOUBLE];
    if (strcmp(name, "clongdouble") == 0 || strcmp(name, "complex256") == 0) return &g_dtype_table[CNP_CLONGDOUBLE];
    if (strcmp(name, "bytes") == 0 || strcmp(name, "string") == 0) return &g_dtype_table[CNP_STRING];
    if (strcmp(name, "str") == 0 || strcmp(name, "unicode") == 0) return &g_dtype_table[CNP_UNICODE];
    if (strcmp(name, "object") == 0) return &g_dtype_table[CNP_OBJECT];
    if (strcmp(name, "void") == 0) return &g_dtype_table[CNP_VOID];
    if (strcmp(name, "datetime64") == 0) return &g_dtype_table[CNP_DATETIME];
    if (strcmp(name, "timedelta64") == 0) return &g_dtype_table[CNP_TIMEDELTA];
    cnp_set_error(
        CNP_ERR_TYPE, "cnp_dtype_from_string",
        "dtype name '%s' is not recognized", name);
    return NULL;
}

CNP_API void CNP_CALL cnp_dtype_incref(CnpDtype *dtype) {
    if (dtype) dtype->refcount++;
}

CNP_API void CNP_CALL cnp_dtype_decref(CnpDtype *dtype) {
    /* Static descriptors remain alive at their baseline reference. */
    if (dtype && dtype->refcount > 1) dtype->refcount--;
}

CNP_API int CNP_CALL cnp_dtype_itemsize(CNP_TYPE type_num) {
    init_dtype_table();
    if (type_num <= 0 || type_num >= CNP_NTYPES) {
        cnp_set_error(
            CNP_ERR_TYPE, "cnp_dtype_itemsize",
            "dtype %d is not a valid CNP_TYPE", (int)type_num);
        return 0;
    }
    return g_dtype_table[type_num].elsize;
}

CNP_API char CNP_CALL cnp_dtype_kind(CNP_TYPE type_num) {
    init_dtype_table();
    if (type_num <= 0 || type_num >= CNP_NTYPES) {
        cnp_set_error(
            CNP_ERR_TYPE, "cnp_dtype_kind",
            "dtype %d is not a valid CNP_TYPE", (int)type_num);
        return '\0';
    }
    return g_dtype_table[type_num].kind;
}

CNP_API bool CNP_CALL cnp_dtype_can_cast(CNP_TYPE from, CNP_TYPE to, CNP_CASTING casting) {
    const char *function_name = "cnp_dtype_can_cast";
    if (from <= CNP_NOTYPE || from >= CNP_NTYPES ||
            to <= CNP_NOTYPE || to >= CNP_NTYPES) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "types %d and %d must be valid CNP_TYPE values",
            (int)from, (int)to);
        return false;
    }

    uint32_t destination_bit = UINT32_C(1) << (to - 1);
    switch (casting) {
        case CNP_CAST_NO:
        case CNP_CAST_EQUIV:
            return (g_equiv_cast_masks[from] & destination_bit) != 0;
        case CNP_CAST_SAFE:
            return (g_safe_cast_masks[from] & destination_bit) != 0;
        case CNP_CAST_SAME_KIND:
            return (g_same_kind_cast_masks[from] & destination_bit) != 0;
        case CNP_CAST_UNSAFE:
            return true;
        default:
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "casting %d must be a valid CNP_CASTING value",
                (int)casting);
            return false;
    }
}

CNP_API CNP_TYPE CNP_CALL cnp_dtype_result_type(CNP_TYPE a, CNP_TYPE b) {
    const char *function_name = "cnp_dtype_result_type";
    if (a <= CNP_NOTYPE || a >= CNP_NTYPES ||
            b <= CNP_NOTYPE || b >= CNP_NTYPES) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "types %d and %d must be valid CNP_TYPE values",
            (int)a, (int)b);
        return CNP_NOTYPE;
    }
    CNP_TYPE result = cnp_promote_type_full(a, b);
    if (result == CNP_NOTYPE) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "types %d and %d do not have a common dtype",
            (int)a, (int)b);
    }
    return result;
}

/* =========================================================================
 * Internal helpers
 * ========================================================================= */
int64_t cnp_compute_size(int ndim, const int64_t *shape) {
    int64_t size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

void cnp_compute_strides(int ndim, const int64_t *shape, int elsize, CNP_ORDER order, int64_t *strides) {
    if (ndim == 0) return;
    if (order == CNP_ORDER_F) {
        strides[0] = elsize;
        for (int i = 1; i < ndim; i++) {
            strides[i] = strides[i-1] * shape[i-1];
        }
    } else {
        /* C order (default) */
        strides[ndim-1] = elsize;
        for (int i = ndim - 2; i >= 0; i--) {
            strides[i] = strides[i+1] * shape[i+1];
        }
    }
}

int64_t cnp_multi_to_offset(int ndim, const int64_t *indices, const int64_t *strides) {
    int64_t offset = 0;
    for (int i = 0; i < ndim; i++) {
        offset += indices[i] * strides[i];
    }
    return offset;
}

int cnp_normalize_axis(int axis, int ndim) {
    if (axis < 0) axis += ndim;
    return axis;
}

bool cnp_type_is_float(CNP_TYPE type) {
    return type == CNP_HALF || type == CNP_FLOAT || type == CNP_DOUBLE || type == CNP_LONGDOUBLE;
}

bool cnp_type_is_integer(CNP_TYPE type) {
    return (type >= CNP_BYTE && type <= CNP_ULONGLONG);
}

bool cnp_type_is_complex(CNP_TYPE type) {
    return type == CNP_CFLOAT || type == CNP_CDOUBLE || type == CNP_CLONGDOUBLE;
}

bool cnp_type_is_unsigned(CNP_TYPE type) {
    return type == CNP_UBYTE || type == CNP_USHORT || type == CNP_UINT ||
           type == CNP_ULONG || type == CNP_ULONGLONG;
}

static bool cnp_type_is_numeric(CNP_TYPE type) {
    return type == CNP_BOOL || cnp_type_is_integer(type) ||
           cnp_type_is_float(type) || cnp_type_is_complex(type);
}

static int cnp_integer_storage_bits(CNP_TYPE type) {
    return cnp_dtype_itemsize(type) * 8;
}

static int cnp_integer_precision_bits(CNP_TYPE type) {
    if (type == CNP_BOOL) return 1;
    int bits = cnp_integer_storage_bits(type);
    return cnp_type_is_unsigned(type) ? bits : bits - 1;
}

static CNP_TYPE cnp_signed_integer_for_bits(int bits) {
    if (bits <= 8) return CNP_BYTE;
    if (bits <= 16) return CNP_SHORT;
    if (bits <= 32) return CNP_INT;
    return CNP_LONGLONG;
}

static CNP_TYPE cnp_unsigned_integer_for_bits(int bits) {
    if (bits <= 8) return CNP_UBYTE;
    if (bits <= 16) return CNP_USHORT;
    if (bits <= 32) return CNP_UINT;
    return CNP_ULONGLONG;
}

static CNP_TYPE cnp_promote_integer_types(CNP_TYPE a, CNP_TYPE b) {
    int a_bits = cnp_integer_storage_bits(a);
    int b_bits = cnp_integer_storage_bits(b);
    bool a_unsigned = cnp_type_is_unsigned(a);
    bool b_unsigned = cnp_type_is_unsigned(b);
    if (a_unsigned == b_unsigned) {
        int bits = a_bits > b_bits ? a_bits : b_bits;
        return a_unsigned
            ? cnp_unsigned_integer_for_bits(bits)
            : cnp_signed_integer_for_bits(bits);
    }

    int signed_bits = a_unsigned ? b_bits : a_bits;
    int unsigned_bits = a_unsigned ? a_bits : b_bits;
    if (signed_bits > unsigned_bits)
        return cnp_signed_integer_for_bits(signed_bits);
    if (unsigned_bits < 16) return CNP_SHORT;
    if (unsigned_bits < 32) return CNP_INT;
    if (unsigned_bits < 64) return CNP_LONGLONG;
    return CNP_DOUBLE;
}

static CNP_TYPE cnp_complex_real_type(CNP_TYPE type) {
    switch (type) {
        case CNP_CFLOAT: return CNP_FLOAT;
        case CNP_CDOUBLE: return CNP_DOUBLE;
        case CNP_CLONGDOUBLE: return CNP_LONGDOUBLE;
        default: return type;
    }
}

static CNP_TYPE cnp_promote_real_numeric_types(CNP_TYPE a, CNP_TYPE b) {
    if (a == CNP_BOOL) return b;
    if (b == CNP_BOOL) return a;
    bool a_float = cnp_type_is_float(a);
    bool b_float = cnp_type_is_float(b);
    if (!a_float && !b_float)
        return cnp_promote_integer_types(a, b);

    if (a == CNP_LONGDOUBLE || b == CNP_LONGDOUBLE)
        return CNP_LONGDOUBLE;
    CNP_TYPE float_type =
        a == CNP_DOUBLE || b == CNP_DOUBLE ? CNP_DOUBLE :
        a == CNP_FLOAT || b == CNP_FLOAT ? CNP_FLOAT : CNP_HALF;
    int integer_precision = 0;
    if (!a_float) integer_precision = cnp_integer_precision_bits(a);
    if (!b_float) {
        int precision = cnp_integer_precision_bits(b);
        if (precision > integer_precision) integer_precision = precision;
    }
    if (float_type == CNP_DOUBLE) return CNP_DOUBLE;
    if (float_type == CNP_FLOAT)
        return integer_precision <= 24 ? CNP_FLOAT : CNP_DOUBLE;
    if (integer_precision <= 11) return CNP_HALF;
    if (integer_precision <= 24) return CNP_FLOAT;
    return CNP_DOUBLE;
}

CNP_TYPE cnp_promote_type(CNP_TYPE a, CNP_TYPE b) {
    if (a <= CNP_NOTYPE || a >= CNP_NTYPES ||
            b <= CNP_NOTYPE || b >= CNP_NTYPES)
        return CNP_NOTYPE;
    if (a == b) return a;
    if (!cnp_type_is_numeric(a) || !cnp_type_is_numeric(b))
        return CNP_NOTYPE;
    if (a == CNP_BOOL) return b;
    if (b == CNP_BOOL) return a;
    if (cnp_type_is_complex(a) || cnp_type_is_complex(b)) {
        CNP_TYPE real_type = cnp_promote_real_numeric_types(
            cnp_complex_real_type(a), cnp_complex_real_type(b));
        if (real_type == CNP_LONGDOUBLE) return CNP_CLONGDOUBLE;
        if (real_type == CNP_DOUBLE) return CNP_CDOUBLE;
        return CNP_CFLOAT;
    }
    return cnp_promote_real_numeric_types(a, b);
}

CNP_TYPE cnp_promote_type_full(CNP_TYPE a, CNP_TYPE b) {
    if (a <= CNP_NOTYPE || a >= CNP_NTYPES ||
            b <= CNP_NOTYPE || b >= CNP_NTYPES)
        return CNP_NOTYPE;
    if (a == b) return a;
    if (a == CNP_OBJECT || b == CNP_OBJECT)
        return CNP_OBJECT;
    if (a == CNP_VOID || b == CNP_VOID)
        return CNP_NOTYPE;
    if (a == CNP_UNICODE || b == CNP_UNICODE) {
        CNP_TYPE other = a == CNP_UNICODE ? b : a;
        if (other == CNP_DATETIME || other == CNP_TIMEDELTA)
            return CNP_NOTYPE;
        return CNP_UNICODE;
    }
    if (a == CNP_STRING || b == CNP_STRING) {
        CNP_TYPE other = a == CNP_STRING ? b : a;
        if (other == CNP_DATETIME || other == CNP_TIMEDELTA)
            return CNP_NOTYPE;
        return CNP_STRING;
    }
    if (a == CNP_DATETIME || b == CNP_DATETIME) {
        CNP_TYPE other = a == CNP_DATETIME ? b : a;
        if (other == CNP_DATETIME || other == CNP_TIMEDELTA)
            return CNP_DATETIME;
        return CNP_NOTYPE;
    }
    if (a == CNP_TIMEDELTA || b == CNP_TIMEDELTA) {
        CNP_TYPE other = a == CNP_TIMEDELTA ? b : a;
        if (other == CNP_TIMEDELTA)
            return CNP_TIMEDELTA;
        if (other == CNP_BOOL)
            return CNP_TIMEDELTA;
        if (cnp_type_is_integer(other) &&
                other != CNP_ULONG && other != CNP_ULONGLONG)
            return CNP_TIMEDELTA;
        return CNP_NOTYPE;
    }
    if (!cnp_type_is_numeric(a) || !cnp_type_is_numeric(b))
        return CNP_NOTYPE;
    return cnp_promote_type(a, b);
}

double cnp_get_element_double(const void *data, int64_t offset, CNP_TYPE dtype) {
    const char *ptr = (const char*)data + offset;
    switch (dtype) {
        case CNP_BOOL:   return (double)(*(const int8_t*)ptr);
        case CNP_BYTE:   return (double)(*(const int8_t*)ptr);
        case CNP_UBYTE:  return (double)(*(const uint8_t*)ptr);
        case CNP_SHORT:  return (double)(*(const int16_t*)ptr);
        case CNP_USHORT: return (double)(*(const uint16_t*)ptr);
        case CNP_INT:    return (double)(*(const int32_t*)ptr);
        case CNP_UINT:   return (double)(*(const uint32_t*)ptr);
        case CNP_LONG:
        case CNP_LONGLONG: return (double)(*(const int64_t*)ptr);
        case CNP_ULONG:
        case CNP_ULONGLONG: return (double)(*(const uint64_t*)ptr);
        case CNP_HALF:   return cnp_half_to_float(*(const uint16_t*)ptr);
        case CNP_FLOAT:  return (double)(*(const float*)ptr);
        case CNP_DOUBLE: return *(const double*)ptr;
        case CNP_LONGDOUBLE: return (double)(*(const long double*)ptr);
        default: return 0.0;
    }
}

void cnp_set_element_double(void *data, int64_t offset, CNP_TYPE dtype, double value) {
    char *ptr = (char*)data + offset;
    switch (dtype) {
        case CNP_BOOL:   *(int8_t*)ptr = (value != 0.0) ? 1 : 0; break;
        case CNP_BYTE:   *(int8_t*)ptr = (int8_t)value; break;
        case CNP_UBYTE:  *(uint8_t*)ptr = (uint8_t)value; break;
        case CNP_SHORT:  *(int16_t*)ptr = (int16_t)value; break;
        case CNP_USHORT: *(uint16_t*)ptr = (uint16_t)value; break;
        case CNP_INT:    *(int32_t*)ptr = (int32_t)value; break;
        case CNP_UINT:   *(uint32_t*)ptr = (uint32_t)value; break;
        case CNP_LONG:
        case CNP_LONGLONG: *(int64_t*)ptr = (int64_t)value; break;
        case CNP_ULONG:
        case CNP_ULONGLONG: *(uint64_t*)ptr = (uint64_t)value; break;
        case CNP_HALF:   *(uint16_t*)ptr = cnp_float_to_half(value); break;
        case CNP_FLOAT:  *(float*)ptr = (float)value; break;
        case CNP_DOUBLE: *(double*)ptr = value; break;
        case CNP_LONGDOUBLE: *(long double*)ptr = (long double)value; break;
        case CNP_CFLOAT:
            ((cnp_cfloat*)ptr)->real = (float)value;
            ((cnp_cfloat*)ptr)->imag = 0.0f;
            break;
        case CNP_CDOUBLE:
            ((cnp_cdouble*)ptr)->real = value;
            ((cnp_cdouble*)ptr)->imag = 0.0;
            break;
        case CNP_CLONGDOUBLE:
            ((cnp_clongdouble*)ptr)->real = (long double)value;
            ((cnp_clongdouble*)ptr)->imag = 0.0L;
            break;
        default: break;
    }
}

int64_t cnp_get_element_int(const void *data, int64_t offset, CNP_TYPE dtype) {
    const char *ptr = (const char*)data + offset;
    switch (dtype) {
        case CNP_BOOL:   return (int64_t)(*(const int8_t*)ptr);
        case CNP_BYTE:   return (int64_t)(*(const int8_t*)ptr);
        case CNP_UBYTE:  return (int64_t)(*(const uint8_t*)ptr);
        case CNP_SHORT:  return (int64_t)(*(const int16_t*)ptr);
        case CNP_USHORT: return (int64_t)(*(const uint16_t*)ptr);
        case CNP_INT:    return (int64_t)(*(const int32_t*)ptr);
        case CNP_UINT:   return (int64_t)(*(const uint32_t*)ptr);
        case CNP_LONG:
        case CNP_LONGLONG: return *(const int64_t*)ptr;
        case CNP_ULONG:
        case CNP_ULONGLONG: return (int64_t)(*(const uint64_t*)ptr);
        case CNP_FLOAT:  return (int64_t)(*(const float*)ptr);
        case CNP_DOUBLE: return (int64_t)(*(const double*)ptr);
        default: return 0;
    }
}

void cnp_set_element_int(void *data, int64_t offset, CNP_TYPE dtype, int64_t value) {
    char *ptr = (char*)data + offset;
    switch (dtype) {
        case CNP_BOOL: {
            int8_t converted = value != 0 ? 1 : 0;
            memcpy(ptr, &converted, sizeof(converted));
            break;
        }
        case CNP_BYTE: {
            int8_t converted = (int8_t)value;
            memcpy(ptr, &converted, sizeof(converted));
            break;
        }
        case CNP_UBYTE: {
            uint8_t converted = (uint8_t)value;
            memcpy(ptr, &converted, sizeof(converted));
            break;
        }
        case CNP_SHORT: {
            int16_t converted = (int16_t)value;
            memcpy(ptr, &converted, sizeof(converted));
            break;
        }
        case CNP_USHORT: {
            uint16_t converted = (uint16_t)value;
            memcpy(ptr, &converted, sizeof(converted));
            break;
        }
        case CNP_INT: {
            int32_t converted = (int32_t)value;
            memcpy(ptr, &converted, sizeof(converted));
            break;
        }
        case CNP_UINT: {
            uint32_t converted = (uint32_t)value;
            memcpy(ptr, &converted, sizeof(converted));
            break;
        }
        case CNP_LONG:
        case CNP_LONGLONG:
            memcpy(ptr, &value, sizeof(value));
            break;
        case CNP_ULONG:
        case CNP_ULONGLONG: {
            uint64_t converted = (uint64_t)value;
            memcpy(ptr, &converted, sizeof(converted));
            break;
        }
        default:
            cnp_set_element_double(data, offset, dtype, (double)value);
            break;
    }
}

bool cnp_value_is_true(double val) {
    return val != 0.0 && !isnan(val);
}

void cnp_swap_bytes(void *data, int size) {
    char *p = (char*)data;
    for (int i = 0; i < size / 2; i++) {
        char tmp = p[i];
        p[i] = p[size - 1 - i];
        p[size - 1 - i] = tmp;
    }
}
