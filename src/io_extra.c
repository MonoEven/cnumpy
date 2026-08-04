/**
 * cnumpy additional I/O functions
 * Corresponds to numpy.savez, numpy.genfromtxt, numpy.recfromtxt, etc.
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>
#include <emmintrin.h>

/* =========================================================================
 * NPZ format: a ZIP container whose members are standard NPY files.
 * ========================================================================= */

typedef struct {
    uint16_t name_length;
    uint32_t crc;
    uint32_t size;
    uint32_t local_offset;
} ZipSavedMember;

static uint32_t zip_crc32(const uint8_t *data, size_t size);

static void zip_store_u16(uint8_t *destination, uint16_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8);
}

static void zip_store_u32(uint8_t *destination, uint32_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8);
    destination[2] = (uint8_t)(value >> 16);
    destination[3] = (uint8_t)(value >> 24);
}

static bool zip_write_bytes(
    FILE *file, const void *data, size_t size,
    uint64_t *position, const char *function_name) {
    if (size > 0 && fwrite(data, 1, size, file) != size) {
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot write ZIP archive");
        return false;
    }
    *position += size;
    return true;
}

static bool zip_write_member_name(
    FILE *file, const char *name, size_t name_length,
    uint64_t *position, const char *function_name) {
    static const char suffix[] = ".npy";
    return zip_write_bytes(
            file, name, name_length, position, function_name) &&
        zip_write_bytes(
            file, suffix, sizeof(suffix) - 1, position, function_name);
}

CNP_API CNP_STATUS CNP_CALL cnp_savez(
    const char *filename, int narrays,
    const char **names, const CnpArray **arrays) {
    const char *function_name = "cnp_savez";
    if (!filename || narrays <= 0 || !names || !arrays) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "filename, positive array count, names, and arrays are required");
        return CNP_ERR_GENERIC;
    }
    if (narrays > UINT16_MAX) {
        cnp_set_error(
            CNP_ERR_NOT_IMPLEMENTED, function_name,
            "ZIP64 is required for more than 65535 members");
        return CNP_ERR_NOT_IMPLEMENTED;
    }
    for (int index = 0; index < narrays; ++index) {
        if (!names[index] || !arrays[index]) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "Every NPZ member requires a name and array");
            return CNP_ERR_GENERIC;
        }
        size_t name_length = strlen(names[index]);
        if (name_length > UINT16_MAX - 4u) {
            cnp_set_error(
                CNP_ERR_NOT_IMPLEMENTED, function_name,
                "ZIP member name is too long");
            return CNP_ERR_NOT_IMPLEMENTED;
        }
    }

    ZipSavedMember *members = (ZipSavedMember*)cnp_calloc(
        (size_t)narrays, sizeof(ZipSavedMember));
    if (!members) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "Cannot allocate ZIP member metadata");
        return CNP_ERR_MEMORY;
    }
    FILE *file = fopen(filename, "wb");
    if (!file) {
        cnp_free(members, (size_t)narrays * sizeof(ZipSavedMember));
        cnp_set_error(
            CNP_ERR_IO, function_name, "Cannot open file: %s", filename);
        return CNP_ERR_IO;
    }

    uint64_t position = 0;
    for (int index = 0; index < narrays; ++index) {
        uint8_t *npy;
        size_t npy_size;
        if (!cnp_npy_save_buffer(
                arrays[index], &npy, &npy_size, function_name))
            goto savez_failure;
        if (npy_size > UINT32_MAX || position > UINT32_MAX) {
            cnp_free(npy, npy_size);
            cnp_set_error(
                CNP_ERR_NOT_IMPLEMENTED, function_name,
                "ZIP64 is required for this NPZ archive");
            goto savez_failure;
        }

        size_t base_name_length = strlen(names[index]);
        ZipSavedMember *member = &members[index];
        member->name_length = (uint16_t)(base_name_length + 4);
        member->crc = zip_crc32(npy, npy_size);
        member->size = (uint32_t)npy_size;
        member->local_offset = (uint32_t)position;

        uint8_t local_header[30] = {0};
        zip_store_u32(local_header, UINT32_C(0x04034b50));
        zip_store_u16(local_header + 4, 20);
        zip_store_u16(local_header + 6, UINT16_C(0x0800));
        zip_store_u16(local_header + 8, 0);
        zip_store_u16(local_header + 10, 0);
        zip_store_u16(local_header + 12, UINT16_C(0x0021));
        zip_store_u32(local_header + 14, member->crc);
        zip_store_u32(local_header + 18, member->size);
        zip_store_u32(local_header + 22, member->size);
        zip_store_u16(local_header + 26, member->name_length);
        zip_store_u16(local_header + 28, 0);
        bool wrote_member = zip_write_bytes(
                file, local_header, sizeof(local_header),
                &position, function_name) &&
            zip_write_member_name(
                file, names[index], base_name_length,
                &position, function_name) &&
            zip_write_bytes(
                file, npy, npy_size, &position, function_name);
        cnp_free(npy, npy_size);
        if (!wrote_member) goto savez_failure;
    }

    if (position > UINT32_MAX) {
        cnp_set_error(
            CNP_ERR_NOT_IMPLEMENTED, function_name,
            "ZIP64 is required for this NPZ archive");
        goto savez_failure;
    }
    uint32_t central_offset = (uint32_t)position;
    for (int index = 0; index < narrays; ++index) {
        const ZipSavedMember *member = &members[index];
        size_t base_name_length = strlen(names[index]);
        uint8_t central_header[46] = {0};
        zip_store_u32(central_header, UINT32_C(0x02014b50));
        zip_store_u16(central_header + 4, 20);
        zip_store_u16(central_header + 6, 20);
        zip_store_u16(central_header + 8, UINT16_C(0x0800));
        zip_store_u16(central_header + 10, 0);
        zip_store_u16(central_header + 12, 0);
        zip_store_u16(central_header + 14, UINT16_C(0x0021));
        zip_store_u32(central_header + 16, member->crc);
        zip_store_u32(central_header + 20, member->size);
        zip_store_u32(central_header + 24, member->size);
        zip_store_u16(central_header + 28, member->name_length);
        zip_store_u32(central_header + 42, member->local_offset);
        if (!zip_write_bytes(
                file, central_header, sizeof(central_header),
                &position, function_name) ||
                !zip_write_member_name(
                    file, names[index], base_name_length,
                    &position, function_name))
            goto savez_failure;
    }

    uint64_t central_size_64 = position - central_offset;
    if (central_size_64 > UINT32_MAX) {
        cnp_set_error(
            CNP_ERR_NOT_IMPLEMENTED, function_name,
            "ZIP64 is required for this NPZ archive");
        goto savez_failure;
    }
    uint8_t eocd[22] = {0};
    zip_store_u32(eocd, UINT32_C(0x06054b50));
    zip_store_u16(eocd + 8, (uint16_t)narrays);
    zip_store_u16(eocd + 10, (uint16_t)narrays);
    zip_store_u32(eocd + 12, (uint32_t)central_size_64);
    zip_store_u32(eocd + 16, central_offset);
    if (!zip_write_bytes(
            file, eocd, sizeof(eocd), &position, function_name))
        goto savez_failure;
    if (fclose(file) != 0) {
        cnp_free(members, (size_t)narrays * sizeof(ZipSavedMember));
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot close ZIP archive");
        return CNP_ERR_IO;
    }
    cnp_free(members, (size_t)narrays * sizeof(ZipSavedMember));
    return CNP_OK;

savez_failure: {
        CNP_STATUS status = cnp_get_error(NULL);
        fclose(file);
        cnp_free(members, (size_t)narrays * sizeof(ZipSavedMember));
        return status == CNP_OK ? CNP_ERR_IO : status;
    }
}

static uint16_t zip_read_u16(const uint8_t *value) {
    return (uint16_t)value[0] | ((uint16_t)value[1] << 8);
}

static uint32_t zip_read_u32(const uint8_t *value) {
    return (uint32_t)value[0] |
        ((uint32_t)value[1] << 8) |
        ((uint32_t)value[2] << 16) |
        ((uint32_t)value[3] << 24);
}

static uint32_t zip_crc32(const uint8_t *data, size_t size) {
    static const uint32_t table[16] = {
        UINT32_C(0x00000000), UINT32_C(0x1db71064),
        UINT32_C(0x3b6e20c8), UINT32_C(0x26d930ac),
        UINT32_C(0x76dc4190), UINT32_C(0x6b6b51f4),
        UINT32_C(0x4db26158), UINT32_C(0x5005713c),
        UINT32_C(0xedb88320), UINT32_C(0xf00f9344),
        UINT32_C(0xd6d6a3e8), UINT32_C(0xcb61b38c),
        UINT32_C(0x9b64c2b0), UINT32_C(0x86d3d2d4),
        UINT32_C(0xa00ae278), UINT32_C(0xbdbdf21c)
    };
    uint32_t crc = UINT32_MAX;
    for (size_t index = 0; index < size; ++index) {
        crc = table[(crc ^ data[index]) & 0x0fu] ^ (crc >> 4);
        crc = table[(crc ^ (data[index] >> 4)) & 0x0fu] ^ (crc >> 4);
    }
    return crc ^ UINT32_MAX;
}

static void loadz_release_results(CnpArray **arrays, int capacity) {
    if (!arrays) return;
    for (int index = 0; index < capacity; ++index) {
        if (arrays[index]) {
            cnp_array_free(arrays[index]);
            arrays[index] = NULL;
        }
    }
}

/* Load NPY members from a standard ZIP archive. */
CNP_API int CNP_CALL cnp_loadz(
    const char *filename, char names[][64],
    CnpArray **arrays, int max_arrays) {
    const char *function_name = "cnp_loadz";
    if (!filename || !names || !arrays || max_arrays <= 0) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "filename, names, arrays, and positive capacity are required");
        return 0;
    }
    memset(names, 0, (size_t)max_arrays * 64);
    for (int index = 0; index < max_arrays; ++index) arrays[index] = NULL;

    FILE *file = fopen(filename, "rb");
    if (!file) {
        cnp_set_error(
            CNP_ERR_IO, function_name, "Cannot open file: %s", filename);
        return 0;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot seek ZIP archive");
        return 0;
    }
    long file_length = ftell(file);
    if (file_length < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot size ZIP archive");
        return 0;
    }
    size_t archive_size = (size_t)file_length;
    uint8_t *archive = archive_size > 0
        ? (uint8_t*)cnp_malloc(archive_size) : NULL;
    if ((archive_size > 0 && !archive) ||
            fread(archive, 1, archive_size, file) != archive_size) {
        if (archive) cnp_free(archive, archive_size);
        fclose(file);
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot read ZIP archive");
        return 0;
    }
    fclose(file);

    if (archive_size < 22) {
        cnp_free(archive, archive_size);
        cnp_set_error(CNP_ERR_IO, function_name, "Incomplete ZIP archive");
        return 0;
    }
    size_t search_start = archive_size > 65557
        ? archive_size - 65557 : 0;
    size_t eocd_offset = SIZE_MAX;
    for (size_t cursor = archive_size - 22;; --cursor) {
        if (zip_read_u32(archive + cursor) == UINT32_C(0x06054b50)) {
            eocd_offset = cursor;
            break;
        }
        if (cursor == search_start) break;
    }
    if (eocd_offset == SIZE_MAX || eocd_offset + 22 > archive_size) {
        cnp_free(archive, archive_size);
        cnp_set_error(
            CNP_ERR_IO, function_name, "ZIP end-of-central-directory not found");
        return 0;
    }

    const uint8_t *eocd = archive + eocd_offset;
    uint16_t disk_number = zip_read_u16(eocd + 4);
    uint16_t central_disk = zip_read_u16(eocd + 6);
    uint16_t disk_entries = zip_read_u16(eocd + 8);
    uint16_t entry_count = zip_read_u16(eocd + 10);
    uint32_t central_size = zip_read_u32(eocd + 12);
    uint32_t central_offset = zip_read_u32(eocd + 16);
    uint16_t comment_length = zip_read_u16(eocd + 20);
    if (disk_number != 0 || central_disk != 0 ||
            disk_entries != entry_count) {
        cnp_free(archive, archive_size);
        cnp_set_error(
            CNP_ERR_NOT_IMPLEMENTED, function_name,
            "Multi-disk ZIP archives are unsupported");
        return 0;
    }
    if (entry_count == UINT16_MAX || central_size == UINT32_MAX ||
            central_offset == UINT32_MAX) {
        cnp_free(archive, archive_size);
        cnp_set_error(
            CNP_ERR_NOT_IMPLEMENTED, function_name,
            "ZIP64 archives are unsupported");
        return 0;
    }
    if ((size_t)comment_length > archive_size - (eocd_offset + 22) ||
            (size_t)central_offset > archive_size ||
            (size_t)central_size > archive_size - central_offset) {
        cnp_free(archive, archive_size);
        cnp_set_error(CNP_ERR_IO, function_name, "Invalid ZIP directory bounds");
        return 0;
    }

    int count = 0;
    size_t cursor = central_offset;
    for (uint16_t entry = 0; entry < entry_count; ++entry) {
        if (cursor > archive_size || archive_size - cursor < 46 ||
                zip_read_u32(archive + cursor) != UINT32_C(0x02014b50)) {
            cnp_set_error(CNP_ERR_IO, function_name, "Invalid ZIP central entry");
            goto loadz_failure;
        }
        const uint8_t *central = archive + cursor;
        uint16_t flags = zip_read_u16(central + 8);
        uint16_t method = zip_read_u16(central + 10);
        uint32_t expected_crc = zip_read_u32(central + 16);
        uint32_t compressed_size = zip_read_u32(central + 20);
        uint32_t uncompressed_size = zip_read_u32(central + 24);
        uint16_t name_length = zip_read_u16(central + 28);
        uint16_t extra_length = zip_read_u16(central + 30);
        uint16_t entry_comment_length = zip_read_u16(central + 32);
        uint32_t local_offset = zip_read_u32(central + 42);
        size_t entry_size = 46u + name_length +
            extra_length + entry_comment_length;
        if (entry_size > archive_size - cursor) {
            cnp_set_error(CNP_ERR_IO, function_name, "Incomplete ZIP central entry");
            goto loadz_failure;
        }
        const uint8_t *entry_name = central + 46;
        cursor += entry_size;

        if (name_length < 4 ||
                memcmp(entry_name + name_length - 4, ".npy", 4) != 0) {
            continue;
        }
        if (count == max_arrays) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "NPZ member count exceeds result capacity");
            goto loadz_failure;
        }
        if (name_length - 4 >= 64) {
            cnp_set_error(CNP_ERR_IO, function_name, "NPZ member name is too long");
            goto loadz_failure;
        }
        if ((flags & 1u) != 0) {
            cnp_set_error(
                CNP_ERR_NOT_IMPLEMENTED, function_name,
                "Encrypted ZIP members are unsupported");
            goto loadz_failure;
        }
        if (method != 0 && method != 8) {
            cnp_set_error(
                CNP_ERR_NOT_IMPLEMENTED, function_name,
                "ZIP compression method %u is unsupported", method);
            goto loadz_failure;
        }
        if ((method == 0 && compressed_size != uncompressed_size) ||
                local_offset > archive_size ||
                archive_size - local_offset < 30 ||
                zip_read_u32(archive + local_offset) !=
                    UINT32_C(0x04034b50)) {
            cnp_set_error(CNP_ERR_IO, function_name, "Invalid ZIP local entry");
            goto loadz_failure;
        }
        const uint8_t *local = archive + local_offset;
        uint16_t local_flags = zip_read_u16(local + 6);
        uint16_t local_method = zip_read_u16(local + 8);
        uint16_t local_name_length = zip_read_u16(local + 26);
        uint16_t local_extra_length = zip_read_u16(local + 28);
        size_t data_offset = (size_t)local_offset + 30u +
            local_name_length + local_extra_length;
        if (data_offset > archive_size ||
                compressed_size > archive_size - data_offset) {
            cnp_set_error(CNP_ERR_IO, function_name, "Incomplete ZIP member data");
            goto loadz_failure;
        }
        if (local_flags != flags || local_method != method) {
            cnp_set_error(
                CNP_ERR_IO, function_name,
                "ZIP local header differs from central directory");
            goto loadz_failure;
        }
        if (local_name_length != name_length ||
                memcmp(local + 30, entry_name, name_length) != 0) {
            cnp_set_error(
                CNP_ERR_IO, function_name,
                "ZIP local member name differs from central directory");
            goto loadz_failure;
        }

        uint8_t *inflated = NULL;
        const uint8_t *npy_buffer = archive + data_offset;
        if (method == 8) {
            if (uncompressed_size > 0) {
                inflated = (uint8_t*)cnp_malloc(uncompressed_size);
                if (!inflated) {
                    cnp_set_error(
                        CNP_ERR_MEMORY, function_name,
                        "Cannot allocate DEFLATE output buffer");
                    goto loadz_failure;
                }
            }
            size_t written = 0;
            if (!cnp_inflate_raw(
                    archive + data_offset, compressed_size,
                    inflated, uncompressed_size, &written, function_name)) {
                if (inflated) cnp_free(inflated, uncompressed_size);
                goto loadz_failure;
            }
            npy_buffer = inflated;
        }

        if (zip_crc32(npy_buffer, uncompressed_size) != expected_crc) {
            if (inflated) cnp_free(inflated, uncompressed_size);
            cnp_set_error(
                CNP_ERR_IO, function_name,
                "ZIP member CRC does not match central directory");
            goto loadz_failure;
        }

        CnpArray *array = cnp_npy_load_buffer(
            npy_buffer, uncompressed_size, function_name);
        if (inflated) cnp_free(inflated, uncompressed_size);
        if (!array) goto loadz_failure;
        memcpy(names[count], entry_name, name_length - 4);
        names[count][name_length - 4] = '\0';
        arrays[count++] = array;
    }

    cnp_free(archive, archive_size);
    return count;

loadz_failure:
    loadz_release_results(arrays, max_arrays);
    memset(names, 0, (size_t)max_arrays * 64);
    cnp_free(archive, archive_size);
    return 0;
}

typedef struct {
    char *data;
    size_t capacity;
    size_t length;
} CnpTextLine;

static void text_line_release(CnpTextLine *line) {
    if (line->data) cnp_free(line->data, line->capacity);
    memset(line, 0, sizeof(*line));
}

static bool text_line_reserve(
        CnpTextLine *line, size_t required,
        const char *function_name) {
    if (required <= line->capacity) return true;
    size_t capacity = line->capacity ? line->capacity : 256;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2) {
            capacity = required;
            break;
        }
        capacity *= 2;
    }
    char *resized = line->data
        ? (char*)cnp_realloc(line->data, line->capacity, capacity)
        : (char*)cnp_malloc(capacity);
    if (!resized) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "cannot grow text line buffer to %llu bytes",
            (unsigned long long)capacity);
        return false;
    }
    line->data = resized;
    line->capacity = capacity;
    return true;
}

static bool text_line_read(
        FILE *file, CnpTextLine *line, bool *at_end,
        const char *function_name) {
    line->length = 0;
    *at_end = false;
    for (;;) {
        int value = fgetc(file);
        if (value == EOF) {
            if (ferror(file)) {
                cnp_set_error(
                    CNP_ERR_IO, function_name,
                    "cannot read delimited text file");
                return false;
            }
            if (line->length == 0) {
                *at_end = true;
                return true;
            }
            break;
        }
        if (value == '\n') break;
        if (line->length == SIZE_MAX - 1 ||
                !text_line_reserve(
                    line, line->length + 2, function_name)) {
            if (line->length == SIZE_MAX - 1) {
                cnp_set_error(
                    CNP_ERR_MEMORY, function_name,
                    "text line length overflows size_t");
            }
            return false;
        }
        line->data[line->length++] = (char)value;
    }
    if (!text_line_reserve(line, line->length + 1, function_name))
        return false;
    line->data[line->length] = '\0';
    if (line->length > 0 && line->data[line->length - 1] == '\r')
        line->data[--line->length] = '\0';
    return true;
}

static char *text_trim(char *token) {
    while (*token && isspace((unsigned char)*token)) ++token;
    char *end = token + strlen(token);
    while (end > token && isspace((unsigned char)end[-1])) --end;
    *end = '\0';
    return token;
}

static char *text_data_line(char *line) {
    char *comment = strchr(line, '#');
    if (comment) *comment = '\0';
    return text_trim(line);
}

static int64_t text_count_fields(
        const char *line, bool whitespace, char delimiter) {
    if (!line[0]) return 0;
    if (!whitespace) {
        int64_t fields = 1;
        for (const char *cursor = line; *cursor; ++cursor) {
            if (*cursor == delimiter) ++fields;
        }
        return fields;
    }
    int64_t fields = 0;
    const char *cursor = line;
    while (*cursor) {
        while (*cursor && isspace((unsigned char)*cursor)) ++cursor;
        if (!*cursor) break;
        ++fields;
        while (*cursor && !isspace((unsigned char)*cursor)) ++cursor;
    }
    return fields;
}

static bool text_dtype_is_supported(
        CNP_TYPE dtype, const char *function_name) {
    if (dtype == CNP_BOOL || cnp_type_is_integer(dtype) ||
            cnp_type_is_float(dtype) || cnp_type_is_complex(dtype)) {
        return true;
    }
    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "dtype %d is not supported by numeric text I/O", (int)dtype);
    return false;
}

static bool text_parse_signed(
        const char *token, int64_t *value) {
    errno = 0;
    char *end = NULL;
    long long parsed = strtoll(token, &end, 10);
    if (errno == ERANGE || end == token || *end != '\0') return false;
    *value = (int64_t)parsed;
    return true;
}

static bool text_parse_unsigned(
        const char *token, uint64_t *value) {
    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(token, &end, 10);
    if (errno == ERANGE || end == token || *end != '\0') return false;
    *value = (uint64_t)parsed;
    return true;
}

static bool text_parse_real(
        const char *token, long double *value) {
    errno = 0;
    char *end = NULL;
    long double parsed = strtold(token, &end);
    if (errno == ERANGE || end == token || *end != '\0') return false;
    *value = parsed;
    return true;
}

static bool text_parse_complex(
        const char *token, long double *real, long double *imag) {
    size_t length = strlen(token);
    const char *start = token;
    if (length >= 2 && token[0] == '(' && token[length - 1] == ')') {
        ++start;
        length -= 2;
    }
    char *copy = (char*)cnp_malloc(length + 1);
    if (!copy) return false;
    memcpy(copy, start, length);
    copy[length] = '\0';
    bool imaginary = length > 0 &&
        (copy[length - 1] == 'j' || copy[length - 1] == 'J');
    if (imaginary) copy[--length] = '\0';
    char *separator = NULL;
    for (char *cursor = copy + 1; *cursor; ++cursor) {
        if ((*cursor == '+' || *cursor == '-') &&
                cursor[-1] != 'e' && cursor[-1] != 'E') {
            separator = cursor;
        }
    }
    bool ok = false;
    if (!imaginary) {
        ok = text_parse_real(copy, real);
        *imag = 0.0L;
    } else if (!separator) {
        *real = 0.0L;
        if (copy[0] == '\0') {
            *imag = 1.0L;
            ok = true;
        } else {
            ok = text_parse_real(copy, imag);
        }
    } else {
        char sign = *separator;
        *separator = '\0';
        ok = text_parse_real(copy, real);
        *separator = sign;
        long double magnitude;
        if (separator[1] == '\0') {
            magnitude = 1.0L;
        } else {
            ok = ok && text_parse_real(separator + 1, &magnitude);
        }
        *imag = sign == '-' ? -magnitude : magnitude;
    }
    cnp_free(copy, length + (imaginary ? 2 : 1));
    return ok;
}

static bool text_store_missing(
        void *destination, CNP_TYPE dtype) {
    switch (dtype) {
        case CNP_BOOL: *(uint8_t*)destination = 0; return true;
        case CNP_BYTE: *(int8_t*)destination = -1; return true;
        case CNP_UBYTE: *(uint8_t*)destination = UINT8_MAX; return true;
        case CNP_SHORT: *(int16_t*)destination = -1; return true;
        case CNP_USHORT: *(uint16_t*)destination = UINT16_MAX; return true;
        case CNP_INT: *(int32_t*)destination = -1; return true;
        case CNP_UINT: *(uint32_t*)destination = UINT32_MAX; return true;
        case CNP_LONG:
        case CNP_LONGLONG: *(int64_t*)destination = -1; return true;
        case CNP_ULONG:
        case CNP_ULONGLONG: *(uint64_t*)destination = UINT64_MAX; return true;
        case CNP_HALF:
            *(uint16_t*)destination = cnp_float_to_half(NAN); return true;
        case CNP_FLOAT: *(float*)destination = NAN; return true;
        case CNP_DOUBLE: *(double*)destination = NAN; return true;
        case CNP_LONGDOUBLE: *(long double*)destination = NAN; return true;
        case CNP_CFLOAT:
            *(cnp_cfloat*)destination = (cnp_cfloat){NAN, 0.0f}; return true;
        case CNP_CDOUBLE:
            *(cnp_cdouble*)destination = (cnp_cdouble){NAN, 0.0}; return true;
        case CNP_CLONGDOUBLE:
            *(cnp_clongdouble*)destination =
                (cnp_clongdouble){NAN, 0.0L}; return true;
        default: return false;
    }
}

static bool text_store_token(
        void *destination, CNP_TYPE dtype, const char *token,
        bool allow_missing, int64_t row, int64_t column,
        const char *function_name) {
    if (!token[0]) {
        if (allow_missing) return text_store_missing(destination, dtype);
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "empty value at row %lld column %lld",
            (long long)(row + 1), (long long)(column + 1));
        return false;
    }
    bool parsed = false;
    switch (dtype) {
        case CNP_BOOL:
            if (strcmp(token, "1") == 0 || strcmp(token, "True") == 0 ||
                    strcmp(token, "true") == 0) {
                *(uint8_t*)destination = 1;
                parsed = true;
            } else if (strcmp(token, "0") == 0 ||
                    strcmp(token, "False") == 0 ||
                    strcmp(token, "false") == 0) {
                *(uint8_t*)destination = 0;
                parsed = true;
            }
            break;
        case CNP_BYTE:
        case CNP_SHORT:
        case CNP_INT:
        case CNP_LONG:
        case CNP_LONGLONG: {
            int64_t value;
            parsed = text_parse_signed(token, &value);
            if (parsed) {
                if (dtype == CNP_BYTE && (value < INT8_MIN || value > INT8_MAX))
                    parsed = false;
                else if (dtype == CNP_SHORT &&
                        (value < INT16_MIN || value > INT16_MAX))
                    parsed = false;
                else if (dtype == CNP_INT &&
                        (value < INT32_MIN || value > INT32_MAX))
                    parsed = false;
                else if (dtype == CNP_BYTE) *(int8_t*)destination = (int8_t)value;
                else if (dtype == CNP_SHORT) *(int16_t*)destination = (int16_t)value;
                else if (dtype == CNP_INT) *(int32_t*)destination = (int32_t)value;
                else *(int64_t*)destination = value;
            }
            break;
        }
        case CNP_UBYTE:
        case CNP_USHORT:
        case CNP_UINT:
        case CNP_ULONG:
        case CNP_ULONGLONG: {
            uint64_t value;
            parsed = text_parse_unsigned(token, &value);
            if (parsed) {
                if (dtype == CNP_UBYTE && value > UINT8_MAX) parsed = false;
                else if (dtype == CNP_USHORT && value > UINT16_MAX) parsed = false;
                else if (dtype == CNP_UINT && value > UINT32_MAX) parsed = false;
                else if (dtype == CNP_UBYTE) *(uint8_t*)destination = (uint8_t)value;
                else if (dtype == CNP_USHORT) *(uint16_t*)destination = (uint16_t)value;
                else if (dtype == CNP_UINT) *(uint32_t*)destination = (uint32_t)value;
                else *(uint64_t*)destination = value;
            }
            break;
        }
        case CNP_HALF:
        case CNP_FLOAT:
        case CNP_DOUBLE:
        case CNP_LONGDOUBLE: {
            long double value;
            parsed = text_parse_real(token, &value);
            if (parsed) {
                if (dtype == CNP_HALF)
                    *(uint16_t*)destination = cnp_float_to_half((double)value);
                else if (dtype == CNP_FLOAT)
                    *(float*)destination = (float)value;
                else if (dtype == CNP_DOUBLE)
                    *(double*)destination = (double)value;
                else *(long double*)destination = value;
            }
            break;
        }
        case CNP_CFLOAT:
        case CNP_CDOUBLE:
        case CNP_CLONGDOUBLE: {
            long double real, imag;
            parsed = text_parse_complex(token, &real, &imag);
            if (parsed && dtype == CNP_CFLOAT)
                *(cnp_cfloat*)destination =
                    (cnp_cfloat){(float)real, (float)imag};
            else if (parsed && dtype == CNP_CDOUBLE)
                *(cnp_cdouble*)destination =
                    (cnp_cdouble){(double)real, (double)imag};
            else if (parsed)
                *(cnp_clongdouble*)destination =
                    (cnp_clongdouble){real, imag};
            break;
        }
        default: break;
    }
    if (parsed) return true;
    if (allow_missing) return text_store_missing(destination, dtype);
    cnp_set_error(
        CNP_ERR_VALUE, function_name,
        "cannot parse '%s' at row %lld column %lld as dtype %d",
        token, (long long)(row + 1),
        (long long)(column + 1), (int)dtype);
    return false;
}

static bool text_parse_row(
        char *line, bool whitespace, char delimiter,
        CnpArray *result, int64_t row, int64_t columns,
        bool allow_missing, const char *function_name) {
    char *cursor = line;
    for (int64_t column = 0; column < columns; ++column) {
        char *token;
        if (whitespace) {
            while (*cursor && isspace((unsigned char)*cursor)) ++cursor;
            token = cursor;
            while (*cursor && !isspace((unsigned char)*cursor)) ++cursor;
            if (*cursor) *cursor++ = '\0';
        } else {
            token = cursor;
            char *separator = strchr(cursor, delimiter);
            if (separator) {
                *separator = '\0';
                cursor = separator + 1;
            } else {
                cursor += strlen(cursor);
            }
        }
        token = text_trim(token);
        void *destination = (char*)result->data +
            (row * columns + column) * result->dtype->elsize;
        if (!text_store_token(
                destination, result->dtype->type_num,
                token, allow_missing, row, column,
                function_name)) return false;
    }
    return true;
}

CnpArray *cnp_text_load_file(
        const char *filename, const char *delimiter,
        int skip_header, int max_rows, bool missing_values,
        CNP_TYPE dtype, const char *function_name) {
    FILE *file = NULL;
    CnpArray *result = NULL;
    CnpTextLine line = {0};
    bool whitespace = delimiter == NULL;
    char delimiter_character = whitespace ? '\0' : delimiter[0];
    if (!filename || !filename[0]) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "filename is required");
        return NULL;
    }
    if (!whitespace && (!delimiter[0] || delimiter[1])) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "NumPy 1.25 text delimiter must be one character");
        return NULL;
    }
    if (skip_header < 0 || max_rows == 0 || max_rows < -1) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "skip_header must be nonnegative and max_rows must be -1 or positive");
        return NULL;
    }
    if (!text_dtype_is_supported(dtype, function_name)) return NULL;
    file = fopen(filename, "r");
    if (!file) {
        cnp_set_error(
            CNP_ERR_IO, function_name,
            "Cannot open file: %s", filename);
        return NULL;
    }

    int64_t rows = 0;
    int64_t columns = -1;
    int64_t physical_line = 0;
    for (;;) {
        bool at_end;
        if (!text_line_read(file, &line, &at_end, function_name)) goto failure;
        if (at_end) break;
        ++physical_line;
        if (physical_line <= skip_header) continue;
        char *data = text_data_line(line.data);
        int64_t line_columns = text_count_fields(
            data, whitespace, delimiter_character);
        if (line_columns == 0) continue;
        if (columns < 0) columns = line_columns;
        else if (line_columns != columns) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "column count changed from %lld to %lld at data row %lld",
                (long long)columns, (long long)line_columns,
                (long long)(rows + 1));
            goto failure;
        }
        ++rows;
        if (max_rows > 0 && rows >= max_rows) break;
    }

    if (rows == 0) {
        int64_t empty_shape[1] = {0};
        result = cnp_array_new(1, empty_shape, dtype, CNP_ORDER_C);
        if (!result) cnp_relabel_error(function_name);
        goto close_file;
    }
    int ndim = rows == 1 || columns == 1 ? 1 : 2;
    int64_t shape[2] = {
        ndim == 1 ? (rows == 1 ? columns : rows) : rows,
        columns
    };
    result = cnp_array_new(ndim, shape, dtype, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        goto failure;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        cnp_set_error(
            CNP_ERR_IO, function_name,
            "cannot rewind delimited text file");
        goto failure;
    }
    physical_line = 0;
    int64_t row = 0;
    while (row < rows) {
        bool at_end;
        if (!text_line_read(file, &line, &at_end, function_name)) goto failure;
        if (at_end) {
            cnp_set_error(
                CNP_ERR_IO, function_name,
                "delimited text file ended during its second pass");
            goto failure;
        }
        ++physical_line;
        if (physical_line <= skip_header) continue;
        char *data = text_data_line(line.data);
        if (text_count_fields(
                data, whitespace, delimiter_character) == 0) continue;
        if (!text_parse_row(
                data, whitespace, delimiter_character,
                result, row, columns, missing_values,
                function_name)) goto failure;
        ++row;
    }

close_file:
    text_line_release(&line);
    if (fclose(file) != 0) {
        if (result) cnp_array_free(result);
        cnp_set_error(
            CNP_ERR_IO, function_name,
            "cannot close delimited text file");
        return NULL;
    }
    return result;

failure:
    text_line_release(&line);
    if (result) cnp_array_free(result);
    if (file) fclose(file);
    return NULL;
}

/* =========================================================================
 * genfromtxt - Load data from text file with missing values
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_genfromtxt(
        const char *filename, const char *delimiter,
        int skip_header, int max_rows, CNP_TYPE dtype) {
    return cnp_text_load_file(
        filename, delimiter, skip_header, max_rows,
        true, dtype, "cnp_genfromtxt");
}

/* =========================================================================
 * recfromtxt - Load structured data from text file
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_recfromtxt(const char *filename, const char *delimiter,
                                           int skip_header, CNP_TYPE dtype) {
    CnpArray *result = cnp_genfromtxt(
        filename, delimiter, skip_header, -1, dtype);
    if (!result) cnp_relabel_error("cnp_recfromtxt");
    return result;
}

/* =========================================================================
 * save multiple arrays with automatic naming (arr_0, arr_1, ...)
 * ========================================================================= */
CNP_API CNP_STATUS CNP_CALL cnp_savez_auto(const char *filename, int narrays, const CnpArray **arrays) {
    if (!filename || narrays <= 0 || !arrays) return CNP_ERR_GENERIC;

    /* Generate names */
    char (*names)[64] = (char(*)[64])cnp_calloc(narrays, 64);
    if (!names) return CNP_ERR_MEMORY;

    const char **name_ptrs = (const char**)cnp_malloc(narrays * sizeof(char*));
    if (!name_ptrs) {
        cnp_free(names, narrays * 64);
        return CNP_ERR_MEMORY;
    }

    for (int i = 0; i < narrays; i++) {
        sprintf(names[i], "arr_%d", i);
        name_ptrs[i] = names[i];
    }

    CNP_STATUS status = cnp_savez(filename, narrays, name_ptrs, arrays);

    cnp_free(names, narrays * 64);
    cnp_free(name_ptrs, narrays * sizeof(char*));
    return status;
}

/* =========================================================================
 * packbits - Pack binary array into bits
 * ========================================================================= */
static bool bitorder_is_valid(
    CNP_BITORDER bitorder, const char *function_name) {
    if (bitorder == CNP_BITORDER_BIG ||
            bitorder == CNP_BITORDER_LITTLE)
        return true;
    cnp_set_error(CNP_ERR_GENERIC, function_name,
                  "bitorder must be CNP_BITORDER_BIG or CNP_BITORDER_LITTLE");
    return false;
}

static bool packbits_accepts_dtype(
    const CnpArray *arr, const char *function_name) {
    char kind = arr->dtype->kind;
    if (kind == 'b' || kind == 'i' || kind == 'u') return true;
    cnp_set_error(CNP_ERR_TYPE, function_name,
                  "input must have an integer or boolean dtype");
    return false;
}

static bool bit_result_size_is_valid(
    const CnpReductionTraversal *traversal,
    int64_t output_axis_length,
    const char *function_name) {
    int64_t result_size = traversal->outer;
    if (output_axis_length != 0 &&
            result_size > INT64_MAX / output_axis_length) {
        cnp_set_error(CNP_ERR_SHAPE, function_name,
                      "result size exceeds int64 range");
        return false;
    }
    result_size *= output_axis_length;
    if (traversal->inner != 0 &&
            result_size > INT64_MAX / traversal->inner) {
        cnp_set_error(CNP_ERR_SHAPE, function_name,
                      "result size exceeds int64 range");
        return false;
    }
    return true;
}

static uint8_t packbits_reverse_byte(uint8_t value) {
    value = (uint8_t)((value >> 4) | (value << 4));
    value = (uint8_t)(
        ((value & 0xccu) >> 2) | ((value & 0x33u) << 2));
    return (uint8_t)(
        ((value & 0xaau) >> 1) | ((value & 0x55u) << 1));
}

static void packbits_contiguous_byte_slice(
    const uint8_t *source, int64_t axis_length,
    CNP_BITORDER bitorder, uint8_t *output,
    int64_t output_stride) {
    if (output_stride == 1) {
        int64_t sse_blocks = axis_length / 16;
        __m128i zero = _mm_setzero_si128();
        for (int64_t block = 0; block < sse_blocks; ++block) {
            __m128i values = _mm_loadu_si128(
                (const __m128i*)(source + block * 16));
            unsigned nonzero_mask = (unsigned)(
                ~_mm_movemask_epi8(_mm_cmpeq_epi8(values, zero))) &
                0xffffu;
            uint8_t first = (uint8_t)nonzero_mask;
            uint8_t second = (uint8_t)(nonzero_mask >> 8);
            if (bitorder == CNP_BITORDER_BIG) {
                first = packbits_reverse_byte(first);
                second = packbits_reverse_byte(second);
            }
            output[block * 2] = first;
            output[block * 2 + 1] = second;
        }
        source += sse_blocks * 16;
        output += sse_blocks * 2;
        axis_length -= sse_blocks * 16;
    }
    int64_t full_bytes = axis_length / 8;
    int remaining_bits = (int)(axis_length % 8);
    for (int64_t byte_index = 0;
         byte_index < full_bytes; ++byte_index) {
        uint8_t packed;
        if (bitorder == CNP_BITORDER_BIG) {
            packed =
                (uint8_t)((source[0] != 0) << 7) |
                (uint8_t)((source[1] != 0) << 6) |
                (uint8_t)((source[2] != 0) << 5) |
                (uint8_t)((source[3] != 0) << 4) |
                (uint8_t)((source[4] != 0) << 3) |
                (uint8_t)((source[5] != 0) << 2) |
                (uint8_t)((source[6] != 0) << 1) |
                (uint8_t)(source[7] != 0);
        } else {
            packed =
                (uint8_t)(source[0] != 0) |
                (uint8_t)((source[1] != 0) << 1) |
                (uint8_t)((source[2] != 0) << 2) |
                (uint8_t)((source[3] != 0) << 3) |
                (uint8_t)((source[4] != 0) << 4) |
                (uint8_t)((source[5] != 0) << 5) |
                (uint8_t)((source[6] != 0) << 6) |
                (uint8_t)((source[7] != 0) << 7);
        }
        output[byte_index * output_stride] = packed;
        source += 8;
    }
    if (remaining_bits == 0) return;

    uint8_t packed = 0;
    for (int bit = 0; bit < remaining_bits; ++bit) {
        if (source[bit] != 0) {
            int shift = bitorder == CNP_BITORDER_BIG ? 7 - bit : bit;
            packed |= (uint8_t)(1u << shift);
        }
    }
    output[full_bytes * output_stride] = packed;
}

CNP_API CnpArray* CNP_CALL cnp_packbits_v2(
    const CnpArray *arr, int axis, bool axis_none,
    CNP_BITORDER bitorder) {
    const char *function_name = "cnp_packbits_v2";
    int resolved_axis;
    if (!cnp_reduction_resolve_axis(
            arr, axis, axis_none, function_name, &resolved_axis))
        return NULL;
    if (!packbits_accepts_dtype(arr, function_name) ||
            !bitorder_is_valid(bitorder, function_name))
        return NULL;

    CnpReductionTraversal traversal;
    cnp_reduction_traversal_init(arr, resolved_axis, &traversal);
    int64_t output_axis_length =
        traversal.axis_length / 8 +
        (traversal.axis_length % 8 != 0);
    int output_ndim;
    int64_t output_shape[CNP_MAXDIMS];
    if (axis_none || arr->ndim == 0) {
        output_ndim = 1;
        output_shape[0] = output_axis_length;
    } else {
        output_ndim = arr->ndim;
        memcpy(output_shape, arr->shape,
               sizeof(int64_t) * arr->ndim);
        output_shape[resolved_axis] = output_axis_length;
    }
    if (!bit_result_size_is_valid(
            &traversal, output_axis_length, function_name))
        return NULL;
    CnpArray *result = cnp_array_new(
        output_ndim, output_shape, CNP_UBYTE, CNP_ORDER_C);
    if (!result) return NULL;
    if (result->size == 0) return result;

    uint8_t *output = (uint8_t*)result->data;
    bool contiguous_byte_slices =
        (arr->dtype->type_num == CNP_BOOL ||
         arr->dtype->type_num == CNP_UBYTE) &&
        traversal.axis_stride == 1 &&
        (!traversal.axis_none ||
         (arr->flags & CNP_ARRAY_C_CONTIGUOUS) != 0);
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            if (contiguous_byte_slices) {
                int64_t slice_offset = cnp_reduction_source_offset(
                    &traversal, outer, inner, 0);
                const uint8_t *source =
                    (const uint8_t*)arr->data + slice_offset;
                uint8_t *destination = output +
                    outer * output_axis_length * traversal.inner + inner;
                packbits_contiguous_byte_slice(
                    source, traversal.axis_length, bitorder,
                    destination, traversal.inner);
                continue;
            }
            bool unit_stride =
                traversal.axis_stride == arr->dtype->elsize &&
                (!traversal.axis_none ||
                 (arr->flags & CNP_ARRAY_C_CONTIGUOUS) != 0);
            int64_t slice_offset = 0;
            if (unit_stride && traversal.axis_length > 0)
                slice_offset = cnp_reduction_source_offset(
                    &traversal, outer, inner, 0);
            for (int64_t byte_index = 0;
                 byte_index < output_axis_length; ++byte_index) {
                uint8_t packed = 0;
                for (int bit = 0; bit < 8; ++bit) {
                    int64_t item = byte_index * 8 + bit;
                    if (item >= traversal.axis_length) break;
                    int64_t source_offset = unit_stride
                        ? slice_offset + item * traversal.axis_stride
                        : cnp_reduction_source_offset(
                            &traversal, outer, inner, item);
                    double value = cnp_get_element_double(
                        arr->data, source_offset, arr->dtype->type_num);
                    if (value != 0.0) {
                        int shift = bitorder == CNP_BITORDER_BIG
                            ? 7 - bit : bit;
                        packed |= (uint8_t)(1u << shift);
                    }
                }
                int64_t output_index =
                    (outer * output_axis_length + byte_index) *
                    traversal.inner + inner;
                output[output_index] = packed;
            }
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_packbits(
    const CnpArray *arr, int axis) {
    CnpArray *result = cnp_packbits_v2(
        arr, axis, axis == CNP_AXIS_NONE, CNP_BITORDER_BIG);
    if (!result) cnp_relabel_error("cnp_packbits");
    return result;
}

/* =========================================================================
 * unpackbits - Unpack bits into binary array
 * ========================================================================= */
#define CNP_UNPACK_NIBBLE_BIG(value) \
    ((uint32_t)(((value) >> 3) & 1u) | \
     ((uint32_t)(((value) >> 2) & 1u) << 8) | \
     ((uint32_t)(((value) >> 1) & 1u) << 16) | \
     ((uint32_t)((value) & 1u) << 24))
#define CNP_UNPACK_NIBBLE_LITTLE(value) \
    ((uint32_t)((value) & 1u) | \
     ((uint32_t)(((value) >> 1) & 1u) << 8) | \
     ((uint32_t)(((value) >> 2) & 1u) << 16) | \
     ((uint32_t)(((value) >> 3) & 1u) << 24))

static const uint32_t unpackbits_nibble_big[16] = {
    CNP_UNPACK_NIBBLE_BIG(0), CNP_UNPACK_NIBBLE_BIG(1),
    CNP_UNPACK_NIBBLE_BIG(2), CNP_UNPACK_NIBBLE_BIG(3),
    CNP_UNPACK_NIBBLE_BIG(4), CNP_UNPACK_NIBBLE_BIG(5),
    CNP_UNPACK_NIBBLE_BIG(6), CNP_UNPACK_NIBBLE_BIG(7),
    CNP_UNPACK_NIBBLE_BIG(8), CNP_UNPACK_NIBBLE_BIG(9),
    CNP_UNPACK_NIBBLE_BIG(10), CNP_UNPACK_NIBBLE_BIG(11),
    CNP_UNPACK_NIBBLE_BIG(12), CNP_UNPACK_NIBBLE_BIG(13),
    CNP_UNPACK_NIBBLE_BIG(14), CNP_UNPACK_NIBBLE_BIG(15),
};

static const uint32_t unpackbits_nibble_little[16] = {
    CNP_UNPACK_NIBBLE_LITTLE(0), CNP_UNPACK_NIBBLE_LITTLE(1),
    CNP_UNPACK_NIBBLE_LITTLE(2), CNP_UNPACK_NIBBLE_LITTLE(3),
    CNP_UNPACK_NIBBLE_LITTLE(4), CNP_UNPACK_NIBBLE_LITTLE(5),
    CNP_UNPACK_NIBBLE_LITTLE(6), CNP_UNPACK_NIBBLE_LITTLE(7),
    CNP_UNPACK_NIBBLE_LITTLE(8), CNP_UNPACK_NIBBLE_LITTLE(9),
    CNP_UNPACK_NIBBLE_LITTLE(10), CNP_UNPACK_NIBBLE_LITTLE(11),
    CNP_UNPACK_NIBBLE_LITTLE(12), CNP_UNPACK_NIBBLE_LITTLE(13),
    CNP_UNPACK_NIBBLE_LITTLE(14), CNP_UNPACK_NIBBLE_LITTLE(15),
};

#undef CNP_UNPACK_NIBBLE_BIG
#undef CNP_UNPACK_NIBBLE_LITTLE

static void unpackbits_contiguous_byte_slice(
    const uint8_t *source, int64_t copied_bits,
    CNP_BITORDER bitorder, uint8_t *output,
    int64_t output_stride) {
    int64_t full_bytes = copied_bits / 8;
    int remaining_bits = (int)(copied_bits % 8);
    if (output_stride == 1) {
        for (int64_t byte_index = 0;
             byte_index < full_bytes; ++byte_index) {
            uint8_t byte = source[byte_index];
            uint32_t first;
            uint32_t second;
            if (bitorder == CNP_BITORDER_BIG) {
                first = unpackbits_nibble_big[byte >> 4];
                second = unpackbits_nibble_big[byte & 15u];
            } else {
                first = unpackbits_nibble_little[byte & 15u];
                second = unpackbits_nibble_little[byte >> 4];
            }
            memcpy(output + byte_index * 8, &first, sizeof(first));
            memcpy(output + byte_index * 8 + 4, &second, sizeof(second));
        }
        if (remaining_bits != 0) {
            uint8_t byte = source[full_bytes];
            uint8_t *bits = output + full_bytes * 8;
            for (int bit = 0; bit < remaining_bits; ++bit) {
                int shift = bitorder == CNP_BITORDER_BIG ? 7 - bit : bit;
                bits[bit] = (uint8_t)((byte >> shift) & 1u);
            }
        }
        return;
    }
    for (int64_t byte_index = 0;
         byte_index < full_bytes; ++byte_index) {
        uint8_t byte = source[byte_index];
        uint8_t *bits = output + byte_index * 8 * output_stride;
        if (bitorder == CNP_BITORDER_BIG) {
            bits[0 * output_stride] = (uint8_t)((byte >> 7) & 1u);
            bits[1 * output_stride] = (uint8_t)((byte >> 6) & 1u);
            bits[2 * output_stride] = (uint8_t)((byte >> 5) & 1u);
            bits[3 * output_stride] = (uint8_t)((byte >> 4) & 1u);
            bits[4 * output_stride] = (uint8_t)((byte >> 3) & 1u);
            bits[5 * output_stride] = (uint8_t)((byte >> 2) & 1u);
            bits[6 * output_stride] = (uint8_t)((byte >> 1) & 1u);
            bits[7 * output_stride] = (uint8_t)(byte & 1u);
        } else {
            bits[0 * output_stride] = (uint8_t)(byte & 1u);
            bits[1 * output_stride] = (uint8_t)((byte >> 1) & 1u);
            bits[2 * output_stride] = (uint8_t)((byte >> 2) & 1u);
            bits[3 * output_stride] = (uint8_t)((byte >> 3) & 1u);
            bits[4 * output_stride] = (uint8_t)((byte >> 4) & 1u);
            bits[5 * output_stride] = (uint8_t)((byte >> 5) & 1u);
            bits[6 * output_stride] = (uint8_t)((byte >> 6) & 1u);
            bits[7 * output_stride] = (uint8_t)((byte >> 7) & 1u);
        }
    }
    if (remaining_bits == 0) return;

    uint8_t byte = source[full_bytes];
    uint8_t *bits = output + full_bytes * 8 * output_stride;
    for (int bit = 0; bit < remaining_bits; ++bit) {
        int shift = bitorder == CNP_BITORDER_BIG ? 7 - bit : bit;
        bits[bit * output_stride] = (uint8_t)((byte >> shift) & 1u);
    }
}

static bool unpackbits_contiguous_interleaved_slices(
    const CnpArray *arr, const CnpReductionTraversal *traversal,
    int64_t copied_bits, int64_t output_axis_length,
    CNP_BITORDER bitorder, uint8_t *output) {
    if ((uint64_t)traversal->inner >
            SIZE_MAX / sizeof(int64_t)) {
        cnp_set_error(CNP_ERR_MEMORY, "cnp_unpackbits_v2",
                      "slice offset workspace exceeds addressable memory");
        return false;
    }
    size_t offsets_size =
        (size_t)traversal->inner * sizeof(int64_t);
    size_t gathered_size = (size_t)traversal->inner;
    int64_t *slice_offsets = (int64_t*)cnp_malloc(offsets_size);
    if (!slice_offsets) {
        cnp_set_error(CNP_ERR_MEMORY, "cnp_unpackbits_v2",
                      "failed to allocate slice offset workspace");
        return false;
    }
    uint8_t *gathered_bytes = (uint8_t*)cnp_malloc(gathered_size);
    if (!gathered_bytes) {
        cnp_free(slice_offsets, offsets_size);
        cnp_set_error(CNP_ERR_MEMORY, "cnp_unpackbits_v2",
                      "failed to allocate gathered byte workspace");
        return false;
    }

    const uint8_t *source = (const uint8_t*)arr->data;
    int64_t bytes_to_read = (copied_bits + 7) / 8;
    for (int64_t outer = 0; outer < traversal->outer; ++outer) {
        for (int64_t inner = 0;
             inner < traversal->inner; ++inner)
            slice_offsets[inner] = cnp_reduction_source_offset(
                traversal, outer, inner, 0);
        for (int64_t byte_index = 0;
             byte_index < bytes_to_read; ++byte_index) {
            for (int64_t inner = 0;
                 inner < traversal->inner; ++inner)
                gathered_bytes[inner] =
                    source[slice_offsets[inner] + byte_index];
            int64_t first_bit = byte_index * 8;
            int bits_in_byte = (int)(copied_bits - first_bit);
            if (bits_in_byte > 8) bits_in_byte = 8;
            for (int bit = 0; bit < bits_in_byte; ++bit) {
                int shift = bitorder == CNP_BITORDER_BIG ? 7 - bit : bit;
                uint8_t *destination = output +
                    (outer * output_axis_length + first_bit + bit) *
                    traversal->inner;
                for (int64_t inner = 0;
                     inner < traversal->inner; ++inner)
                    destination[inner] = (uint8_t)(
                        (gathered_bytes[inner] >> shift) & 1u);
            }
        }
    }

    cnp_free(gathered_bytes, gathered_size);
    cnp_free(slice_offsets, offsets_size);
    return true;
}

CNP_API CnpArray* CNP_CALL cnp_unpackbits_v2(
    const CnpArray *arr, int axis, bool axis_none,
    int64_t count, bool count_none, CNP_BITORDER bitorder) {
    const char *function_name = "cnp_unpackbits_v2";
    int resolved_axis;
    if (!cnp_reduction_resolve_axis(
            arr, axis, axis_none, function_name, &resolved_axis))
        return NULL;
    if (arr->dtype->type_num != CNP_UBYTE) {
        cnp_set_error(CNP_ERR_TYPE, function_name,
                      "input must have an unsigned byte dtype");
        return NULL;
    }
    if (!bitorder_is_valid(bitorder, function_name)) return NULL;

    CnpReductionTraversal traversal;
    cnp_reduction_traversal_init(arr, resolved_axis, &traversal);
    if (traversal.axis_length > INT64_MAX / 8) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "unpacked axis length exceeds int64 range");
        return NULL;
    }
    int64_t available_bits = traversal.axis_length * 8;
    int64_t output_axis_length = count_none
        ? available_bits
        : count < 0 ? available_bits + count : count;
    if (output_axis_length < 0) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "-count larger than number of elements");
        return NULL;
    }

    int output_ndim;
    int64_t output_shape[CNP_MAXDIMS];
    if (axis_none || arr->ndim == 0) {
        output_ndim = 1;
        output_shape[0] = output_axis_length;
    } else {
        output_ndim = arr->ndim;
        memcpy(output_shape, arr->shape,
               sizeof(int64_t) * arr->ndim);
        output_shape[resolved_axis] = output_axis_length;
    }
    if (!bit_result_size_is_valid(
            &traversal, output_axis_length, function_name))
        return NULL;
    CnpArray *result = cnp_array_new(
        output_ndim, output_shape, CNP_UBYTE, CNP_ORDER_C);
    if (!result) return NULL;
    if (result->size == 0) return result;

    uint8_t *output = (uint8_t*)result->data;
    if (result->size > 0 && output_axis_length > available_bits)
        memset(output, 0, (size_t)result->size);
    int64_t copied_bits = output_axis_length < available_bits
        ? output_axis_length : available_bits;
    bool contiguous_byte_slices =
        traversal.axis_stride == 1 &&
        (!traversal.axis_none ||
         (arr->flags & CNP_ARRAY_C_CONTIGUOUS) != 0);
    if (contiguous_byte_slices && traversal.inner > 1) {
        if (!unpackbits_contiguous_interleaved_slices(
                arr, &traversal, copied_bits,
                output_axis_length, bitorder, output)) {
            cnp_array_free(result);
            return NULL;
        }
        return result;
    }
    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            bool unit_stride = contiguous_byte_slices;
            int64_t slice_offset = 0;
            if (unit_stride && traversal.axis_length > 0)
                slice_offset = cnp_reduction_source_offset(
                    &traversal, outer, inner, 0);
            if (unit_stride) {
                const uint8_t *source = (const uint8_t*)arr->data +
                    slice_offset;
                uint8_t *destination = output +
                    outer * output_axis_length * traversal.inner + inner;
                unpackbits_contiguous_byte_slice(
                    source, copied_bits, bitorder,
                    destination, traversal.inner);
                continue;
            }
            int64_t bytes_to_read = (copied_bits + 7) / 8;
            for (int64_t byte_index = 0;
                 byte_index < bytes_to_read; ++byte_index) {
                int64_t source_offset = unit_stride
                    ? slice_offset + byte_index
                    : cnp_reduction_source_offset(
                        &traversal, outer, inner, byte_index);
                uint8_t byte = *(const uint8_t*)((const char*)arr->data +
                    source_offset);
                int64_t first_bit = byte_index * 8;
                int bits_in_byte = (int)(copied_bits - first_bit);
                if (bits_in_byte > 8) bits_in_byte = 8;
                for (int bit_in_byte = 0;
                     bit_in_byte < bits_in_byte; ++bit_in_byte) {
                    int shift = bitorder == CNP_BITORDER_BIG
                        ? 7 - bit_in_byte : bit_in_byte;
                    uint8_t unpacked =
                        (uint8_t)((byte >> shift) & 1u);
                    int64_t bit_index = first_bit + bit_in_byte;
                    int64_t output_index =
                        (outer * output_axis_length + bit_index) *
                        traversal.inner + inner;
                    output[output_index] = unpacked;
                }
            }
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_unpackbits(
    const CnpArray *arr, int axis, int64_t count) {
    CnpArray *result = cnp_unpackbits_v2(
        arr, axis, axis == CNP_AXIS_NONE,
        count, count <= 0, CNP_BITORDER_BIG);
    if (!result) cnp_relabel_error("cnp_unpackbits");
    return result;
}

/* =========================================================================
 * base_repr - Return string representation in given base
 * ========================================================================= */
CNP_API char* CNP_CALL cnp_base_repr(int64_t number, int base, int padding) {
    const char *function_name = "cnp_base_repr";
    if (base < 2 || base > 36) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "base must be between 2 and 36");
        return NULL;
    }

    char digits[65];
    int digit_count = 0;
    bool negative = number < 0;
    uint64_t n = negative
        ? (uint64_t)(-(number + 1)) + UINT64_C(1)
        : (uint64_t)number;

    if (n == 0) {
        digits[digit_count++] = '0';
    } else {
        while (n > 0) {
            int digit = (int)(n % base);
            digits[digit_count++] =
                (char)((digit < 10) ? ('0' + digit) : ('A' + digit - 10));
            n /= base;
        }
    }

    int zero_count = padding > 0 ? padding : 0;
    size_t length = (negative ? 1u : 0u) +
                    (size_t)zero_count + (size_t)digit_count;
    char *result = (char*)cnp_malloc(length + 1);
    if (!result) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "cannot allocate integer representation");
        return NULL;
    }
    size_t position = 0;
    if (negative) result[position++] = '-';
    for (int index = 0; index < zero_count; ++index)
        result[position++] = '0';
    for (int index = digit_count - 1; index >= 0; --index)
        result[position++] = digits[index];
    result[position] = '\0';
    return result;
}

/* =========================================================================
 * binary_repr - Return binary string representation
 * ========================================================================= */
CNP_API char* CNP_CALL cnp_binary_repr(int64_t number, int width) {
    const char *function_name = "cnp_binary_repr";
    if (number < 0 && width < 0) {
        char *result = cnp_base_repr(number, 2, 0);
        if (!result) cnp_relabel_error(function_name);
        return result;
    }

    if (number >= 0) {
        uint64_t bits = (uint64_t)number;
        uint64_t remaining = bits;
        int digit_count = 1;
        while (remaining >>= 1) ++digit_count;
        int output_width = width > digit_count ? width : digit_count;
        size_t length = (size_t)output_width;
        char *result = (char*)cnp_malloc(length + 1);
        if (!result) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "cannot allocate binary representation");
            return NULL;
        }
        for (int index = 0; index < output_width; ++index) {
            int bit_index = output_width - 1 - index;
            result[index] = bit_index >= 64
                ? '0'
                : (char)('0' + ((bits >> bit_index) & UINT64_C(1)));
        }
        result[length] = '\0';
        return result;
    }

    uint64_t magnitude = (uint64_t)(-(number + 1)) + UINT64_C(1);
    uint64_t reduced = magnitude - UINT64_C(1);
    int minimum_width = 1;
    while (reduced > 0) {
        ++minimum_width;
        reduced >>= 1;
    }
    int output_width = width > minimum_width ? width : minimum_width;
    size_t length = (size_t)output_width;
    char *result = (char*)cnp_malloc(length + 1);
    if (!result) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "cannot allocate binary representation");
        return NULL;
    }
    uint64_t bits = (uint64_t)number;
    for (int index = 0; index < output_width; ++index) {
        int bit_index = output_width - 1 - index;
        result[index] = bit_index >= 64
            ? '1'
            : (char)('0' + ((bits >> bit_index) & UINT64_C(1)));
    }
    result[length] = '\0';
    return result;
}
