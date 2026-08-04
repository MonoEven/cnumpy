/**
 * cnumpy memory-mapped file support
 * Corresponds to numpy.memmap
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#endif

typedef struct {
    void *map_base;
    size_t map_size;
    bool persistent_writes;
#ifdef _WIN32
    HANDLE file_handle;
    HANDLE map_handle;
#else
    int fd;
#endif
} CnpMemmap;

static int file_seek64(FILE *stream, int64_t offset, int origin) {
#ifdef _WIN32
    return _fseeki64(stream, offset, origin);
#else
    return fseeko(stream, (off_t)offset, origin);
#endif
}

static int64_t file_tell64(FILE *stream) {
#ifdef _WIN32
    return _ftelli64(stream);
#else
    return (int64_t)ftello(stream);
#endif
}

static void memmap_owner_release(void *opaque_owner) {
    CnpMemmap *owner = (CnpMemmap*)opaque_owner;
    bool failed = false;
#ifdef _WIN32
    DWORD error_code = ERROR_SUCCESS;
    if (owner->map_base && !UnmapViewOfFile(owner->map_base)) {
        failed = true;
        error_code = GetLastError();
    }
    if (owner->map_handle && !CloseHandle(owner->map_handle)) {
        if (!failed) error_code = GetLastError();
        failed = true;
    }
    if (owner->file_handle != INVALID_HANDLE_VALUE &&
            !CloseHandle(owner->file_handle)) {
        if (!failed) error_code = GetLastError();
        failed = true;
    }
#else
    int error_code = 0;
    if (owner->map_base && munmap(owner->map_base, owner->map_size) != 0) {
        failed = true;
        error_code = errno;
    }
    if (owner->fd >= 0 && close(owner->fd) != 0) {
        if (!failed) error_code = errno;
        failed = true;
    }
#endif
    cnp_free(owner, sizeof(CnpMemmap));
    if (failed) {
        cnp_set_error(
            CNP_ERR_IO, "cnp_memmap_release",
            "Cannot release memory mapping (system error %lu)",
            (unsigned long)error_code);
    }
}

static bool memmap_compute_sizes(
    int ndim, const int64_t *shape, CNP_TYPE dtype, int64_t offset,
    size_t *data_bytes, int64_t *required_file_size,
    const char *function_name) {
    if (ndim < 0 || ndim > CNP_MAXDIMS || (ndim > 0 && !shape)) {
        cnp_set_error(CNP_ERR_SHAPE, function_name, "Invalid memmap shape");
        return false;
    }
    if (offset < 0) {
        cnp_set_error(CNP_ERR_IO, function_name, "offset must be non-negative");
        return false;
    }
    int itemsize = cnp_dtype_itemsize(dtype);
    if (itemsize <= 0) {
        cnp_set_error(CNP_ERR_TYPE, function_name, "Invalid memmap dtype");
        return false;
    }

    uint64_t element_count = 1;
    for (int dimension = 0; dimension < ndim; ++dimension) {
        if (shape[dimension] < 0) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "Negative memmap dimension at axis %d", dimension);
            return false;
        }
        if (shape[dimension] != 0 &&
                element_count > UINT64_MAX / (uint64_t)shape[dimension]) {
            cnp_set_error(CNP_ERR_SHAPE, function_name, "Memmap size overflows");
            return false;
        }
        element_count *= (uint64_t)shape[dimension];
    }
    if (element_count == 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "Cannot memory-map an empty array");
        return false;
    }
    if (element_count > SIZE_MAX / (size_t)itemsize) {
        cnp_set_error(CNP_ERR_SHAPE, function_name, "Memmap size overflows");
        return false;
    }
    size_t bytes = (size_t)element_count * (size_t)itemsize;
    if ((uint64_t)bytes > INT64_MAX ||
            (uint64_t)offset > INT64_MAX - (uint64_t)bytes) {
        cnp_set_error(CNP_ERR_SHAPE, function_name, "Memmap file size overflows");
        return false;
    }
    *data_bytes = bytes;
    *required_file_size = offset + (int64_t)bytes;
    return true;
}

#ifdef _WIN32
static wchar_t *memmap_wide_path(
    const char *filename, const char *function_name, size_t *allocation_size) {
    int character_count = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, filename, -1, NULL, 0);
    if (character_count <= 0) {
        cnp_set_error(CNP_ERR_IO, function_name, "filename is not valid UTF-8");
        return NULL;
    }
    *allocation_size = (size_t)character_count * sizeof(wchar_t);
    wchar_t *wide = (wchar_t*)cnp_malloc(*allocation_size);
    if (!wide) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "Cannot allocate UTF-8 filename conversion");
        return NULL;
    }
    if (MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, filename, -1,
            wide, character_count) != character_count) {
        cnp_free(wide, *allocation_size);
        cnp_set_error(CNP_ERR_IO, function_name, "filename is not valid UTF-8");
        return NULL;
    }
    return wide;
}
#endif

/* mode: 0 = r, 1 = r+, 2 = c, 3 = w+ */
CNP_API CnpArray* CNP_CALL cnp_memmap_create(
    const char *filename, int ndim, const int64_t *shape,
    CNP_TYPE dtype, int mode, int64_t offset) {
    const char *function_name = "cnp_memmap_create";
    if (!filename) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "filename is required");
        return NULL;
    }
    if (mode < 0 || mode > 3) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "mode must be 0 (r), 1 (r+), 2 (c), or 3 (w+)");
        return NULL;
    }
    size_t data_bytes;
    int64_t required_file_size;
    if (!memmap_compute_sizes(
            ndim, shape, dtype, offset,
            &data_bytes, &required_file_size, function_name))
        return NULL;

    CnpMemmap *owner = (CnpMemmap*)cnp_calloc(1, sizeof(CnpMemmap));
    if (!owner) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "Cannot allocate memory-map owner");
        return NULL;
    }
    owner->persistent_writes = mode == 1 || mode == 3;

#ifdef _WIN32
    owner->file_handle = INVALID_HANDLE_VALUE;
    DWORD access, create_mode, protect, map_access;

    if (mode == 3) {
        /* Create new file */
        access = GENERIC_READ | GENERIC_WRITE;
        create_mode = CREATE_ALWAYS;
        protect = PAGE_READWRITE;
        map_access = FILE_MAP_WRITE;
    } else if (mode == 1) {
        /* Read-write existing */
        access = GENERIC_READ | GENERIC_WRITE;
        create_mode = OPEN_EXISTING;
        protect = PAGE_READWRITE;
        map_access = FILE_MAP_WRITE;
    } else if (mode == 2) {
        access = GENERIC_READ;
        create_mode = OPEN_EXISTING;
        protect = PAGE_WRITECOPY;
        map_access = FILE_MAP_COPY;
    } else {
        /* Read-only */
        access = GENERIC_READ;
        create_mode = OPEN_EXISTING;
        protect = PAGE_READONLY;
        map_access = FILE_MAP_READ;
    }

    size_t wide_size;
    wchar_t *wide_filename = memmap_wide_path(
        filename, function_name, &wide_size);
    if (!wide_filename) {
        cnp_free(owner, sizeof(CnpMemmap));
        return NULL;
    }
    owner->file_handle = CreateFileW(
        wide_filename, access,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        NULL, create_mode, FILE_ATTRIBUTE_NORMAL, NULL);
    cnp_free(wide_filename, wide_size);
    if (owner->file_handle == INVALID_HANDLE_VALUE) {
        DWORD error_code = GetLastError();
        cnp_free(owner, sizeof(CnpMemmap));
        cnp_set_error(
            CNP_ERR_IO, function_name,
            "Cannot open file (Win32 error %lu)", error_code);
        return NULL;
    }

    if (mode == 3) {
        LARGE_INTEGER li;
        li.QuadPart = required_file_size;
        if (!SetFilePointerEx(owner->file_handle, li, NULL, FILE_BEGIN) ||
                !SetEndOfFile(owner->file_handle)) {
            DWORD error_code = GetLastError();
            memmap_owner_release(owner);
            cnp_set_error(
                CNP_ERR_IO, function_name,
                "Cannot resize memmap file (Win32 error %lu)", error_code);
            return NULL;
        }
    } else {
        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(owner->file_handle, &file_size) ||
                file_size.QuadPart < required_file_size) {
            memmap_owner_release(owner);
            cnp_set_error(
                CNP_ERR_IO, function_name,
                "Memmap range exceeds existing file size");
            return NULL;
        }
    }

    owner->map_handle = CreateFileMappingW(
        owner->file_handle, NULL, protect, 0, 0, NULL);
    if (!owner->map_handle) {
        DWORD error_code = GetLastError();
        memmap_owner_release(owner);
        cnp_set_error(
            CNP_ERR_IO, function_name,
            "Cannot create file mapping (Win32 error %lu)", error_code);
        return NULL;
    }

    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    uint64_t granularity = system_info.dwAllocationGranularity;
    uint64_t aligned_offset = (uint64_t)offset / granularity * granularity;
    size_t data_delta = (size_t)((uint64_t)offset - aligned_offset);
    if (data_bytes > SIZE_MAX - data_delta) {
        memmap_owner_release(owner);
        cnp_set_error(CNP_ERR_SHAPE, function_name, "Mapping view size overflows");
        return NULL;
    }
    owner->map_size = data_delta + data_bytes;
    ULARGE_INTEGER view_offset;
    view_offset.QuadPart = aligned_offset;
    owner->map_base = MapViewOfFile(
        owner->map_handle, map_access,
        view_offset.HighPart, view_offset.LowPart, owner->map_size);
    if (!owner->map_base) {
        DWORD error_code = GetLastError();
        memmap_owner_release(owner);
        cnp_set_error(
            CNP_ERR_IO, function_name,
            "Cannot map view of file (Win32 error %lu)", error_code);
        return NULL;
    }
    void *array_data = (uint8_t*)owner->map_base + data_delta;

#else
    int flags, prot;

    if (mode == 3) {
        flags = O_RDWR | O_CREAT | O_TRUNC;
        prot = PROT_READ | PROT_WRITE;
    } else if (mode == 1) {
        flags = O_RDWR;
        prot = PROT_READ | PROT_WRITE;
    } else if (mode == 2) {
        flags = O_RDONLY;
        prot = PROT_READ | PROT_WRITE;
    } else {
        flags = O_RDONLY;
        prot = PROT_READ;
    }

    owner->fd = open(filename, flags, 0644);
    if (owner->fd < 0) {
        cnp_free(owner, sizeof(CnpMemmap));
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot open file: %s", filename);
        return NULL;
    }
    if (mode == 3) {
        if (ftruncate(owner->fd, required_file_size) < 0) {
            memmap_owner_release(owner);
            cnp_set_error(CNP_ERR_IO, function_name, "Cannot set file size");
            return NULL;
        }
    } else {
        struct stat file_status;
        if (fstat(owner->fd, &file_status) != 0 ||
                file_status.st_size < required_file_size) {
            memmap_owner_release(owner);
            cnp_set_error(
                CNP_ERR_IO, function_name,
                "Memmap range exceeds existing file size");
            return NULL;
        }
    }
    long page_size = sysconf(_SC_PAGE_SIZE);
    int64_t aligned_offset = offset / page_size * page_size;
    size_t data_delta = (size_t)(offset - aligned_offset);
    if (data_bytes > SIZE_MAX - data_delta) {
        memmap_owner_release(owner);
        cnp_set_error(CNP_ERR_SHAPE, function_name, "Mapping view size overflows");
        return NULL;
    }
    owner->map_size = data_delta + data_bytes;
    int mmap_flags = (mode == 2) ? MAP_PRIVATE : MAP_SHARED;
    owner->map_base = mmap(
        NULL, owner->map_size, prot, mmap_flags,
        owner->fd, aligned_offset);
    if (owner->map_base == MAP_FAILED) {
        owner->map_base = NULL;
        memmap_owner_release(owner);
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot mmap file");
        return NULL;
    }
    void *array_data = (uint8_t*)owner->map_base + data_delta;
#endif

    CnpDtype *descriptor = cnp_dtype_new(dtype);
    uint32_t flags = CNP_ARRAY_MEMMAP;
    if (mode != 0) flags |= CNP_ARRAY_WRITEABLE;
    if (descriptor && descriptor->alignment > 0 &&
            (uintptr_t)array_data % (uintptr_t)descriptor->alignment == 0)
        flags |= CNP_ARRAY_ALIGNED;
    CnpArray *array = cnp_array_adopt_external_data(
        ndim, shape, dtype, CNP_ORDER_C,
        array_data, flags, owner, memmap_owner_release, function_name);
    if (!array) {
        memmap_owner_release(owner);
        return NULL;
    }
    return array;
}

/* =========================================================================
 * cnp_memmap_open - Open existing file as memory-mapped array
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_memmap_open(const char *filename, int ndim, const int64_t *shape,
                                            CNP_TYPE dtype, int64_t offset) {
    CnpArray *array = cnp_memmap_create(
        filename, ndim, shape, dtype, 0, offset);
    if (!array) cnp_relabel_error("cnp_memmap_open");
    return array;
}

static CnpMemmap *memmap_owner_from_array(const CnpArray *array) {
    const CnpArray *root = array;
    while (root && root->base) root = root->base;
    if (!root || (root->flags & CNP_ARRAY_MEMMAP) == 0 ||
            root->owner_release != memmap_owner_release)
        return NULL;
    return (CnpMemmap*)root->owner;
}

CNP_API CNP_STATUS CNP_CALL cnp_memmap_flush(const CnpArray *arr) {
    const char *function_name = "cnp_memmap_flush";
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "array is required");
        return CNP_ERR_GENERIC;
    }
    CnpMemmap *owner = memmap_owner_from_array(arr);
    if (!owner) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "array is not backed by a memory mapping");
        return CNP_ERR_TYPE;
    }
    if (!owner->persistent_writes) return CNP_OK;

#ifdef _WIN32
    if (!FlushViewOfFile(owner->map_base, owner->map_size) ||
            !FlushFileBuffers(owner->file_handle)) {
        cnp_set_error(
            CNP_ERR_IO, function_name,
            "Cannot flush memory mapping (Win32 error %lu)", GetLastError());
        return CNP_ERR_IO;
    }
#else
    if (msync(owner->map_base, owner->map_size, MS_SYNC) != 0) {
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot flush memory mapping");
        return CNP_ERR_IO;
    }
#endif
    return CNP_OK;
}

CNP_API void CNP_CALL cnp_memmap_close(CnpArray *arr) {
    cnp_array_decref(arr);
}

/* =========================================================================
 * cnp_fromfile - Read array from binary file
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_fromfile(const char *filename, CNP_TYPE dtype, int64_t count, int64_t offset) {
    const char *function_name = "cnp_fromfile";
    if (!filename) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "filename is required");
        return NULL;
    }
    if (offset < 0) {
        cnp_set_error(
            CNP_ERR_IO, function_name, "offset must be non-negative");
        return NULL;
    }
    int elsize = cnp_dtype_itemsize(dtype);
    if (elsize <= 0) {
        cnp_set_error(CNP_ERR_TYPE, function_name, "Invalid dtype");
        return NULL;
    }

    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        cnp_set_error(
            CNP_ERR_IO, function_name, "Cannot open file: %s", filename);
        return NULL;
    }

    /* Get file size */
    if (file_seek64(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot seek to file end");
        return NULL;
    }
    int64_t file_size = file_tell64(fp);
    if (file_size < 0) {
        fclose(fp);
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot determine file size");
        return NULL;
    }
    if (offset > file_size) {
        fclose(fp);
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "offset exceeds the available file data");
        return NULL;
    }
    if (file_seek64(fp, offset, SEEK_SET) != 0) {
        fclose(fp);
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot seek to offset");
        return NULL;
    }

    int64_t available = (file_size - offset) / elsize;
    if (count < 0 || count > available) count = available;

    int64_t shape[1] = {count};
    CnpArray *result = cnp_array_new(1, shape, dtype, CNP_ORDER_C);
    if (!result) {
        fclose(fp);
        cnp_relabel_error(function_name);
        return NULL;
    }

    size_t read_count = fread(
        result->data, (size_t)elsize, (size_t)count, fp);
    fclose(fp);

    if ((int64_t)read_count != count) {
        cnp_array_free(result);
        cnp_set_error(CNP_ERR_IO, function_name, "Read error");
        return NULL;
    }

    return result;
}

/* =========================================================================
 * cnp_tofile - Write array to binary file
 * ========================================================================= */
CNP_API CNP_STATUS CNP_CALL cnp_tofile(const CnpArray *arr, const char *filename) {
    const char *function_name = "cnp_tofile";
    if (!arr || !filename) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "array and filename are required");
        return CNP_ERR_GENERIC;
    }

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        cnp_set_error(
            CNP_ERR_IO, function_name, "Cannot open file: %s", filename);
        return CNP_ERR_IO;
    }

    /* Make contiguous copy if needed */
    const CnpArray *write_source = arr;
    CnpArray *owned_copy = NULL;
    if (!(arr->flags & CNP_ARRAY_C_CONTIGUOUS)) {
        owned_copy = cnp_array_copy(arr);
        if (!owned_copy) {
            fclose(fp);
            cnp_relabel_error(function_name);
            return cnp_get_error(NULL);
        }
        write_source = owned_copy;
    }

    int64_t expected_count = write_source->size;
    size_t written = fwrite(
        write_source->data, (size_t)write_source->dtype->elsize,
        (size_t)expected_count, fp);
    int close_status = fclose(fp);

    if (owned_copy) cnp_array_free(owned_copy);

    if ((int64_t)written != expected_count || close_status != 0) {
        cnp_set_error(CNP_ERR_IO, function_name, "Write error");
        return CNP_ERR_IO;
    }

    return CNP_OK;
}

/* =========================================================================
 * cnp_frombuffer - Create array from buffer (memory)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_frombuffer(const void *buffer, int64_t size, CNP_TYPE dtype) {
    const char *function_name = "cnp_frombuffer";
    if (size < 0 || (!buffer && size > 0)) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "buffer must not be null and size must be non-negative");
        return NULL;
    }
    int elsize = cnp_dtype_itemsize(dtype);
    if (elsize <= 0) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (size % elsize != 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "buffer size %lld is not a multiple of dtype itemsize %d",
            (long long)size, elsize);
        return NULL;
    }
    int64_t count = size / elsize;

    int64_t shape[1] = {count};
    CnpArray *result = cnp_array_new(1, shape, dtype, CNP_ORDER_C);
    if (!result) return NULL;

    if (size > 0) memcpy(result->data, buffer, (size_t)size);
    return result;
}

static bool cnp_fromstring_dtype_is_supported(CNP_TYPE dtype) {
    if (dtype <= CNP_NOTYPE || dtype >= CNP_NTYPES) return false;
    char kind = cnp_dtype_kind(dtype);
    return kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f';
}

static bool cnp_fromstring_separator_is_whitespace(const char *separator) {
    const unsigned char *cursor = (const unsigned char*)separator;
    if (!*cursor) return false;
    while (*cursor) {
        if (!isspace(*cursor)) return false;
        ++cursor;
    }
    return true;
}

static bool cnp_fromstring_store_integer(
        CnpArray *result, int64_t index, const char *token,
        char **token_end, CNP_TYPE dtype, const char *function_name) {
    const bool is_unsigned = cnp_dtype_kind(dtype) == 'u';
    const bool is_bool = dtype == CNP_BOOL;
    errno = 0;

    if (is_unsigned) {
        const char *sign = token;
        while (isspace((unsigned char)*sign)) ++sign;
        if (*sign == '-') {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "numeric token is out of range for unsigned dtype %d",
                (int)dtype);
            return false;
        }
        unsigned long long value = _strtoui64(token, token_end, 10);
        uint64_t maximum = UINT64_MAX;
        switch (dtype) {
            case CNP_UBYTE: maximum = UINT8_MAX; break;
            case CNP_USHORT: maximum = UINT16_MAX; break;
            case CNP_UINT: maximum = UINT32_MAX; break;
            default: break;
        }
        if (*token_end == token) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name, "invalid numeric token");
            return false;
        }
        if (errno == ERANGE || value > maximum) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "numeric token is out of range for dtype %d", (int)dtype);
            return false;
        }
        switch (dtype) {
            case CNP_UBYTE: *(uint8_t*)((char*)result->data + index) = (uint8_t)value; break;
            case CNP_USHORT: *(uint16_t*)((char*)result->data + index * 2) = (uint16_t)value; break;
            case CNP_UINT: *(uint32_t*)((char*)result->data + index * 4) = (uint32_t)value; break;
            default: *(uint64_t*)((char*)result->data + index * 8) = (uint64_t)value; break;
        }
        return true;
    }

    long long value = _strtoi64(token, token_end, 10);
    int64_t minimum = INT64_MIN;
    int64_t maximum = INT64_MAX;
    switch (dtype) {
        case CNP_BYTE: minimum = INT8_MIN; maximum = INT8_MAX; break;
        case CNP_SHORT: minimum = INT16_MIN; maximum = INT16_MAX; break;
        case CNP_INT: minimum = INT32_MIN; maximum = INT32_MAX; break;
        case CNP_BOOL: minimum = INT64_MIN; maximum = INT64_MAX; break;
        default: break;
    }
    if (*token_end == token) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name, "invalid numeric token");
        return false;
    }
    if (errno == ERANGE || value < minimum || value > maximum) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "numeric token is out of range for dtype %d", (int)dtype);
        return false;
    }
    if (is_bool) {
        ((uint8_t*)result->data)[index] = value != 0;
    } else {
        cnp_set_element_int(
            result->data, index * result->dtype->elsize, dtype, (int64_t)value);
    }
    return true;
}

static bool cnp_fromstring_parse_token(
        CnpArray *result, int64_t index, const char *token,
        char **token_end, CNP_TYPE dtype, const char *function_name) {
    char kind = cnp_dtype_kind(dtype);
    if (kind == 'b' || kind == 'i' || kind == 'u') {
        return cnp_fromstring_store_integer(
            result, index, token, token_end, dtype, function_name);
    }

    errno = 0;
    double value = strtod(token, token_end);
    if (*token_end == token) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "invalid numeric token");
        return false;
    }
    if (errno == ERANGE && !isinf(value) && value != 0.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "numeric token is out of range for dtype %d", (int)dtype);
        return false;
    }
    cnp_set_element_double(
        result->data, index * result->dtype->elsize, dtype, value);
    return true;
}

static bool cnp_fromstring_scan(
        char *text, int64_t length, CNP_TYPE dtype, int64_t requested_count,
        const char *separator, CnpArray *result, int64_t *parsed_count,
        const char *function_name) {
    char *cursor = text;
    char *end = text + length;
    int64_t count = 0;
    const size_t separator_length = strlen(separator);
    const bool whitespace_separator =
        cnp_fromstring_separator_is_whitespace(separator);

    while (cursor < end && isspace((unsigned char)*cursor)) ++cursor;
    while (cursor < end && (requested_count < 0 || count < requested_count)) {
        char *token_end = NULL;
        CnpArray scratch;
        memset(&scratch, 0, sizeof(scratch));
        if (!result) {
            scratch.data = (void*)&scratch.offset;
            scratch.dtype = cnp_dtype_new(dtype);
            if (!scratch.dtype) {
                cnp_relabel_error(function_name);
                return false;
            }
            result = &scratch;
        }

        bool parsed = cnp_fromstring_parse_token(
            result, result == &scratch ? 0 : count,
            cursor, &token_end, dtype, function_name);
        if (result == &scratch) {
            cnp_dtype_decref(scratch.dtype);
            result = NULL;
        }
        if (!parsed) {
            return false;
        }

        cursor = token_end;
        ++count;
        if (requested_count >= 0 && count >= requested_count) break;

        bool skipped_whitespace = false;
        while (cursor < end && isspace((unsigned char)*cursor)) {
            skipped_whitespace = true;
            ++cursor;
        }
        if (cursor >= end) break;

        if (whitespace_separator) {
            if (!skipped_whitespace) {
                cnp_set_error(
                    CNP_ERR_VALUE, function_name,
                    "invalid numeric token at byte offset %lld",
                    (long long)(cursor - text));
                return false;
            }
        } else {
            if ((size_t)(end - cursor) < separator_length ||
                    memcmp(cursor, separator, separator_length) != 0) {
                cnp_set_error(
                    CNP_ERR_VALUE, function_name,
                    "invalid numeric token at byte offset %lld",
                    (long long)(cursor - text));
                return false;
            }
            cursor += separator_length;
            while (cursor < end && isspace((unsigned char)*cursor)) ++cursor;
        }
    }

    while (cursor < end && isspace((unsigned char)*cursor)) ++cursor;
    if ((requested_count < 0 || count < requested_count) && cursor < end) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "invalid numeric token at byte offset %lld",
            (long long)(cursor - text));
        return false;
    }
    *parsed_count = count;
    return true;
}

/* =========================================================================
 * cnp_fromstring - Create an array from binary bytes (legacy NumPy mode)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_fromstring(
        const char *str, int64_t len, CNP_TYPE dtype) {
    const char *function_name = "cnp_fromstring";
    if (!str) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "input string is required");
        return NULL;
    }
    if (len < 0) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "length must be nonnegative");
        return NULL;
    }
    int itemsize = cnp_dtype_itemsize(dtype);
    if (itemsize <= 0) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (len % itemsize != 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "binary string length must be a multiple of the dtype itemsize");
        return NULL;
    }

    int64_t shape[1] = {len / itemsize};
    CnpArray *result = cnp_array_new(1, shape, dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (len > 0) memcpy(result->data, str, (size_t)len);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_fromstring_v2(
        const char *str, int64_t len, CNP_TYPE dtype,
        int64_t count, const char *sep) {
    const char *function_name = "cnp_fromstring_v2";
    if (!str) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "input string is required");
        return NULL;
    }
    if (len < 0) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "length must be nonnegative");
        return NULL;
    }
    if (!cnp_fromstring_dtype_is_supported(dtype)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtype %d must be a valid numeric dtype", (int)dtype);
        return NULL;
    }
    if (count < -1) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "count must be -1 or nonnegative");
        return NULL;
    }
    if (!sep || !*sep) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "text separator must not be empty");
        return NULL;
    }
    if ((uint64_t)len > SIZE_MAX - 1) {
        cnp_set_error(CNP_ERR_MEMORY, function_name, "input string is too large");
        return NULL;
    }

    char *text = (char*)cnp_malloc((size_t)len + 1);
    if (!text) {
        cnp_set_error(CNP_ERR_MEMORY, function_name, "failed to copy input string");
        return NULL;
    }
    if (len > 0) memcpy(text, str, (size_t)len);
    text[len] = '\0';

    int64_t parsed_count = 0;
    if (!cnp_fromstring_scan(
            text, len, dtype, count, sep, NULL,
            &parsed_count, function_name)) {
        cnp_free(text, (size_t)len + 1);
        return NULL;
    }

    int64_t shape[1] = {parsed_count};
    CnpArray *result = cnp_array_new(1, shape, dtype, CNP_ORDER_C);
    if (!result) {
        cnp_free(text, (size_t)len + 1);
        cnp_relabel_error(function_name);
        return NULL;
    }
    int64_t stored_count = 0;
    if (!cnp_fromstring_scan(
            text, len, dtype, count, sep, result,
            &stored_count, function_name)) {
        cnp_array_free(result);
        cnp_free(text, (size_t)len + 1);
        return NULL;
    }
    cnp_free(text, (size_t)len + 1);
    return result;
}
