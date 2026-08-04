/**
 * cnumpy I/O operations - save/load, text I/O, string conversion
 */
#include "../include/cnumpy/cnumpy_internal.h"

typedef struct {
    CnpStringFunction function;
    void *userdata;
} CnpStringFunctionState;

static CnpStringFunctionState g_repr_string_function = {NULL, NULL};
static CnpStringFunctionState g_str_string_function = {NULL, NULL};

void cnp_reset_string_functions(void) {
    g_repr_string_function.function = NULL;
    g_repr_string_function.userdata = NULL;
    g_str_string_function.function = NULL;
    g_str_string_function.userdata = NULL;
}

/* =========================================================================
 * NPY format constants
 * ========================================================================= */
static const char NPY_MAGIC[] = "\x93NUMPY";
#define NPY_MAGIC_LEN 6

static bool npy_append_text(
    char *buffer, size_t capacity, size_t *used,
    const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    int written = vsnprintf(
        buffer + *used, capacity - *used, format, arguments);
    va_end(arguments);
    if (written < 0 || (size_t)written >= capacity - *used) return false;
    *used += (size_t)written;
    return true;
}

bool cnp_npy_save_buffer(
    const CnpArray *arr, uint8_t **buffer, size_t *size,
    const char *function_name) {
    if (!arr || !buffer || !size) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "array, output buffer, and output size are required");
        return false;
    }
    *buffer = NULL;
    *size = 0;

    bool fortran_order =
        (arr->flags & CNP_ARRAY_F_CONTIGUOUS) != 0 &&
        (arr->flags & CNP_ARRAY_C_CONTIGUOUS) == 0;
    char dtype_str[16];
    uint16_t endian_probe = 1;
    bool host_is_little_endian = *(const uint8_t*)&endian_probe == 1;
    char byte_order = arr->dtype->elsize == 1
        ? '|' : host_is_little_endian ? '<' : '>';
    snprintf(dtype_str, sizeof(dtype_str), "%c%c%d",
             byte_order, arr->dtype->kind, arr->dtype->elsize);

    char shape_str[CNP_MAXDIMS * 24 + 3];
    size_t shape_length = 0;
    if (!npy_append_text(
            shape_str, sizeof(shape_str), &shape_length, "(")) {
        cnp_set_error(CNP_ERR_SHAPE, function_name, "NPY shape is too long");
        return false;
    }
    for (int i = 0; i < arr->ndim; i++) {
        if (!npy_append_text(
                shape_str, sizeof(shape_str), &shape_length,
                "%lld%s", (long long)arr->shape[i],
                i < arr->ndim - 1 || arr->ndim == 1 ? "," : "")) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name, "NPY shape is too long");
            return false;
        }
    }
    if (!npy_append_text(
            shape_str, sizeof(shape_str), &shape_length, ")")) {
        cnp_set_error(CNP_ERR_SHAPE, function_name, "NPY shape is too long");
        return false;
    }

    char header[CNP_MAXDIMS * 24 + 128];
    int header_length = snprintf(header, sizeof(header),
        "{'descr': '%s', 'fortran_order': %s, 'shape': %s, }",
        dtype_str, fortran_order ? "True" : "False", shape_str);
    if (header_length < 0 || (size_t)header_length >= sizeof(header)) {
        cnp_set_error(CNP_ERR_IO, function_name, "NPY header is too long");
        return false;
    }

    size_t unpadded_header_length = (size_t)header_length + 1;
    size_t padding =
        (64 - ((NPY_MAGIC_LEN + 2 + 2 + unpadded_header_length) % 64)) % 64;
    size_t padded_header_length = unpadded_header_length + padding;
    if (padded_header_length > UINT16_MAX) {
        cnp_set_error(
            CNP_ERR_NOT_IMPLEMENTED, function_name,
            "NPY v2 output is required for this header");
        return false;
    }

    if (arr->size < 0 || arr->dtype->elsize <= 0 ||
            (uint64_t)arr->size > SIZE_MAX / (size_t)arr->dtype->elsize) {
        cnp_set_error(CNP_ERR_SHAPE, function_name, "NPY data size overflows");
        return false;
    }
    size_t data_size =
        (size_t)arr->size * (size_t)arr->dtype->elsize;
    size_t prefix_size = NPY_MAGIC_LEN + 2 + 2;
    if (padded_header_length > SIZE_MAX - prefix_size ||
            data_size > SIZE_MAX - prefix_size - padded_header_length) {
        cnp_set_error(CNP_ERR_SHAPE, function_name, "NPY file size overflows");
        return false;
    }
    size_t output_size = prefix_size + padded_header_length + data_size;
    uint8_t *output = (uint8_t*)cnp_malloc(output_size);
    if (!output) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name, "Unable to allocate NPY output");
        return false;
    }

    memcpy(output, NPY_MAGIC, NPY_MAGIC_LEN);
    output[6] = 1;
    output[7] = 0;
    output[8] = (uint8_t)padded_header_length;
    output[9] = (uint8_t)(padded_header_length >> 8);
    memcpy(output + prefix_size, header, (size_t)header_length);
    memset(
        output + prefix_size + header_length, ' ', padding);
    output[prefix_size + padded_header_length - 1] = '\n';

    const CnpArray *source = arr;
    CnpArray *owned_copy = NULL;
    if ((arr->flags & CNP_ARRAY_C_CONTIGUOUS) == 0 && !fortran_order) {
        owned_copy = cnp_array_copy(arr);
        if (!owned_copy) {
            cnp_free(output, output_size);
            cnp_relabel_error(function_name);
            return false;
        }
        source = owned_copy;
    }
    if (data_size > 0) {
        memcpy(
            output + prefix_size + padded_header_length,
            (const uint8_t*)source->data + source->offset,
            data_size);
    }
    if (owned_copy) cnp_array_free(owned_copy);

    *buffer = output;
    *size = output_size;
    return true;
}

/* =========================================================================
 * Binary save (.npy format)
 * ========================================================================= */
CNP_API CNP_STATUS CNP_CALL cnp_save(const char *filename, const CnpArray *arr) {
    const char *function_name = "cnp_save";
    if (!filename || !arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "filename and array are required");
        return CNP_ERR_GENERIC;
    }

    uint8_t *output;
    size_t output_size;
    if (!cnp_npy_save_buffer(
            arr, &output, &output_size, function_name))
        return cnp_get_error(NULL);

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        cnp_free(output, output_size);
        cnp_set_error(
            CNP_ERR_IO, function_name, "Cannot open file: %s", filename);
        return CNP_ERR_IO;
    }
    bool write_ok = fwrite(output, 1, output_size, fp) == output_size;
    bool close_ok = fclose(fp) == 0;
    cnp_free(output, output_size);
    if (!write_ok || !close_ok) {
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot write NPY file");
        return CNP_ERR_IO;
    }
    return CNP_OK;
}

static uint16_t npy_read_u16(const uint8_t *value) {
    return (uint16_t)value[0] | ((uint16_t)value[1] << 8);
}

static uint32_t npy_read_u32(const uint8_t *value) {
    return (uint32_t)value[0] |
        ((uint32_t)value[1] << 8) |
        ((uint32_t)value[2] << 16) |
        ((uint32_t)value[3] << 24);
}

static bool npy_parse_dtype(
    const char *header, CNP_TYPE *dtype, char *byte_order) {
    const char *descriptor = strstr(header, "'descr': '");
    bool supported = false;
    if (!descriptor) return false;
    descriptor += 10;
    *byte_order = descriptor[0];
    char kind = descriptor[1];
    int itemsize = atoi(descriptor + 2);
    switch (kind) {
        case 'i':
            if (itemsize == 1) *dtype = CNP_BYTE;
            else if (itemsize == 2) *dtype = CNP_SHORT;
            else if (itemsize == 4) *dtype = CNP_INT;
            else if (itemsize == 8) *dtype = CNP_LONGLONG;
            else break;
            supported = true;
            break;
        case 'u':
            if (itemsize == 1) *dtype = CNP_UBYTE;
            else if (itemsize == 2) *dtype = CNP_USHORT;
            else if (itemsize == 4) *dtype = CNP_UINT;
            else if (itemsize == 8) *dtype = CNP_ULONGLONG;
            else break;
            supported = true;
            break;
        case 'f':
            if (itemsize == 4) *dtype = CNP_FLOAT;
            else if (itemsize == 8) *dtype = CNP_DOUBLE;
            else break;
            supported = true;
            break;
        case 'c':
            if (itemsize == 8) *dtype = CNP_CFLOAT;
            else if (itemsize == 16) *dtype = CNP_CDOUBLE;
            else break;
            supported = true;
            break;
        case 'b':
            if (itemsize == 1) {
                *dtype = CNP_BOOL;
                supported = true;
            }
            break;
        default:
            break;
    }
    return supported;
}

static void npy_byteswap(CnpArray *array, CNP_TYPE dtype) {
    uint8_t *bytes = (uint8_t*)array->data;
    int itemsize = array->dtype->elsize;
    int component_size = dtype == CNP_CFLOAT || dtype == CNP_CDOUBLE
        ? itemsize / 2 : itemsize;
    for (int64_t item = 0; item < array->size; ++item) {
        uint8_t *value = bytes + item * itemsize;
        for (int component = 0;
             component < itemsize; component += component_size) {
            for (int left = 0, right = component_size - 1;
                 left < right; ++left, --right) {
                uint8_t temporary = value[component + left];
                value[component + left] = value[component + right];
                value[component + right] = temporary;
            }
        }
    }
}

CnpArray *cnp_npy_load_buffer(
    const uint8_t *buffer, size_t size, const char *function_name) {
    if (!buffer || size < NPY_MAGIC_LEN + 2) {
        cnp_set_error(
            CNP_ERR_IO, function_name, "Incomplete NPY preamble");
        return NULL;
    }
    if (memcmp(buffer, NPY_MAGIC, NPY_MAGIC_LEN) != 0) {
        cnp_set_error(CNP_ERR_IO, function_name, "Invalid NPY magic");
        return NULL;
    }

    uint8_t major = buffer[6];
    uint8_t minor = buffer[7];
    size_t prefix_size;
    uint32_t header_length;
    if (major == 1) {
        prefix_size = 10;
        if (size < prefix_size) {
            cnp_set_error(
                CNP_ERR_IO, function_name, "Incomplete NPY v1 header length");
            return NULL;
        }
        header_length = npy_read_u16(buffer + 8);
    } else if (major == 2) {
        prefix_size = 12;
        if (size < prefix_size) {
            cnp_set_error(
                CNP_ERR_IO, function_name, "Incomplete NPY v2 header length");
            return NULL;
        }
        header_length = npy_read_u32(buffer + 8);
    } else {
        cnp_set_error(
            CNP_ERR_NOT_IMPLEMENTED, function_name,
            "Unsupported NPY version %u.%u", major, minor);
        return NULL;
    }
    if ((size_t)header_length > size - prefix_size) {
        cnp_set_error(CNP_ERR_IO, function_name, "Incomplete NPY header");
        return NULL;
    }

    size_t header_size = (size_t)header_length + 1;
    char *header = (char*)cnp_malloc(header_size);
    if (!header) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name, "Unable to allocate NPY header");
        return NULL;
    }
    memcpy(header, buffer + prefix_size, header_length);
    header[header_length] = '\0';

    CNP_TYPE dtype = CNP_NOTYPE;
    char byte_order = '|';
    if (!npy_parse_dtype(header, &dtype, &byte_order)) {
        cnp_free(header, header_size);
        cnp_set_error(
            CNP_ERR_TYPE, function_name, "Unsupported dtype descriptor");
        return NULL;
    }

    CNP_ORDER order;
    if (strstr(header, "'fortran_order': True")) {
        order = CNP_ORDER_F;
    } else if (strstr(header, "'fortran_order': False")) {
        order = CNP_ORDER_C;
    } else {
        cnp_free(header, header_size);
        cnp_set_error(
            CNP_ERR_IO, function_name, "Invalid NPY fortran_order");
        return NULL;
    }

    int ndim = 0;
    int64_t shape[CNP_MAXDIMS] = {0};
    char *shape_position = strstr(header, "'shape': (");
    if (!shape_position) {
        cnp_free(header, header_size);
        cnp_set_error(CNP_ERR_IO, function_name, "Missing NPY shape");
        return NULL;
    }
    shape_position += 10;
    while (*shape_position && *shape_position != ')') {
        if (*shape_position == '-') {
            cnp_free(header, header_size);
            cnp_set_error(CNP_ERR_SHAPE, function_name, "Negative NPY shape");
            return NULL;
        }
        if (*shape_position >= '0' && *shape_position <= '9') {
            if (ndim == CNP_MAXDIMS) {
                cnp_free(header, header_size);
                cnp_set_error(
                    CNP_ERR_SHAPE, function_name, "NPY rank exceeds CNP_MAXDIMS");
                return NULL;
            }
            shape[ndim++] = strtoll(
                shape_position, &shape_position, 10);
        } else {
            ++shape_position;
        }
    }
    if (*shape_position != ')') {
        cnp_free(header, header_size);
        cnp_set_error(CNP_ERR_IO, function_name, "Incomplete NPY shape");
        return NULL;
    }
    cnp_free(header, header_size);

    CnpArray *array = cnp_array_new(ndim, shape, dtype, order);
    if (!array) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    size_t data_offset = prefix_size + header_length;
    size_t data_size = (size_t)array->size * (size_t)array->dtype->elsize;
    if (data_size > size - data_offset) {
        cnp_array_free(array);
        cnp_set_error(CNP_ERR_IO, function_name, "Incomplete NPY data");
        return NULL;
    }
    if (data_size > 0) memcpy(array->data, buffer + data_offset, data_size);

    uint16_t endian_probe = 1;
    bool host_is_little_endian = *(const uint8_t*)&endian_probe == 1;
    bool needs_byteswap = array->dtype->elsize > 1 &&
        ((byte_order == '>' && host_is_little_endian) ||
         (byte_order == '<' && !host_is_little_endian));
    if (needs_byteswap) npy_byteswap(array, dtype);
    return array;
}

/* =========================================================================
 * Binary load (.npy format)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_load(const char *filename) {
    const char *function_name = "cnp_load";
    if (!filename) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "filename is required");
        return NULL;
    }

    FILE *file = fopen(filename, "rb");
    if (!file) {
        cnp_set_error(
            CNP_ERR_IO, function_name, "Cannot open file: %s", filename);
        return NULL;
    }
    if (_fseeki64(file, 0, SEEK_END) != 0) {
        fclose(file);
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot seek NPY file");
        return NULL;
    }
    __int64 file_length = _ftelli64(file);
    if (file_length < 0 || (uint64_t)file_length > SIZE_MAX ||
            _fseeki64(file, 0, SEEK_SET) != 0) {
        fclose(file);
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot size NPY file");
        return NULL;
    }

    size_t file_size = (size_t)file_length;
    uint8_t *file_data = file_size > 0
        ? (uint8_t*)cnp_malloc(file_size) : NULL;
    if (file_size > 0 && !file_data) {
        fclose(file);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name, "Cannot allocate NPY input buffer");
        return NULL;
    }
    bool read_ok =
        fread(file_data, 1, file_size, file) == file_size;
    bool close_ok = fclose(file) == 0;
    if (!read_ok || !close_ok) {
        if (file_data) cnp_free(file_data, file_size);
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot read NPY file");
        return NULL;
    }

    CnpArray *array = cnp_npy_load_buffer(
        file_data, file_size, function_name);
    if (file_data) cnp_free(file_data, file_size);
    return array;
}

/* =========================================================================
 * Text save (savetxt)
 * ========================================================================= */
CNP_API CNP_STATUS CNP_CALL cnp_savetxt(const char *filename, const CnpArray *arr,
                                         const char *delimiter, const char *fmt) {
    const char *function_name = "cnp_savetxt";
    if (!filename || !filename[0] || !arr || !arr->dtype) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "filename and array are required");
        return CNP_ERR_VALUE;
    }
    if (!delimiter) delimiter = " ";
    if (!fmt) fmt = "%.18e";
    if (!delimiter[0]) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "delimiter must not be empty");
        return CNP_ERR_VALUE;
    }
    if (arr->ndim != 1 && arr->ndim != 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "expected a 1-D or 2-D array, got rank %d", arr->ndim);
        return CNP_ERR_SHAPE;
    }
    if (cnp_type_is_complex(arr->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "legacy single floating format cannot represent complex values");
        return CNP_ERR_TYPE;
    }
    if (!cnp_text_float_format_is_valid(fmt, function_name))
        return cnp_get_error(NULL);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        cnp_set_error(CNP_ERR_IO, function_name, "Cannot open file: %s", filename);
        return CNP_ERR_IO;
    }

    bool write_ok = true;
    if (arr->ndim == 1) {
        for (int64_t i = 0; i < arr->shape[0]; i++) {
            if (fprintf(fp, fmt, cnp_array_flat_get(arr, i)) < 0 ||
                    fputc('\n', fp) == EOF) {
                write_ok = false;
                break;
            }
        }
    } else {
        for (int64_t i = 0; i < arr->shape[0]; i++) {
            for (int64_t j = 0; j < arr->shape[1]; j++) {
                if ((j > 0 && fputs(delimiter, fp) == EOF) ||
                        fprintf(
                            fp, fmt,
                            cnp_array_flat_get(
                                arr, i * arr->shape[1] + j)) < 0) {
                    write_ok = false;
                    break;
                }
            }
            if (!write_ok || fputc('\n', fp) == EOF) {
                write_ok = false;
                break;
            }
        }
    }
    if (fclose(fp) != 0) write_ok = false;
    if (!write_ok) {
        cnp_set_error(
            CNP_ERR_IO, function_name,
            "cannot write delimited text file: %s", filename);
        return CNP_ERR_IO;
    }
    return CNP_OK;
}

/* =========================================================================
 * Text load (loadtxt)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_loadtxt(const char *filename, const char *delimiter, CNP_TYPE dtype) {
    return cnp_text_load_file(
        filename, delimiter, 0, -1,
        false, dtype, "cnp_loadtxt");
}

/* =========================================================================
 * String conversion
 * ========================================================================= */
typedef struct {
    char *data;
    size_t capacity;
    size_t used;
    const char *function_name;
} CnpTextBuilder;

static bool text_builder_reserve(
        CnpTextBuilder *builder, size_t additional) {
    if (additional > SIZE_MAX - builder->used - 1) {
        cnp_set_error(
            CNP_ERR_MEMORY, builder->function_name,
            "text result size overflows size_t");
        return false;
    }
    size_t required = builder->used + additional + 1;
    if (required <= builder->capacity) return true;
    size_t capacity = builder->capacity ? builder->capacity : 128;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2) {
            capacity = required;
            break;
        }
        capacity *= 2;
    }
    char *resized = builder->data
        ? (char*)cnp_realloc(builder->data, builder->capacity, capacity)
        : (char*)cnp_malloc(capacity);
    if (!resized) {
        cnp_set_error(
            CNP_ERR_MEMORY, builder->function_name,
            "cannot grow text result to %llu bytes",
            (unsigned long long)capacity);
        return false;
    }
    builder->data = resized;
    builder->capacity = capacity;
    if (builder->used == 0) builder->data[0] = '\0';
    return true;
}

static bool text_builder_append_bytes(
        CnpTextBuilder *builder, const char *text, size_t length) {
    if (!text_builder_reserve(builder, length)) return false;
    memcpy(builder->data + builder->used, text, length);
    builder->used += length;
    builder->data[builder->used] = '\0';
    return true;
}

static bool text_builder_append(
        CnpTextBuilder *builder, const char *text) {
    return text_builder_append_bytes(builder, text, strlen(text));
}

static bool text_builder_append_format(
        CnpTextBuilder *builder, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    va_list measured;
    va_copy(measured, arguments);
    int length = vsnprintf(NULL, 0, format, measured);
    va_end(measured);
    if (length < 0) {
        va_end(arguments);
        cnp_set_error(
            CNP_ERR_VALUE, builder->function_name,
            "numeric format could not be evaluated");
        return false;
    }
    if (!text_builder_reserve(builder, (size_t)length)) {
        va_end(arguments);
        return false;
    }
    int written = vsnprintf(
        builder->data + builder->used,
        builder->capacity - builder->used,
        format, arguments);
    va_end(arguments);
    if (written != length) {
        cnp_set_error(
            CNP_ERR_VALUE, builder->function_name,
            "numeric format produced an inconsistent length");
        return false;
    }
    builder->used += (size_t)written;
    return true;
}

static char *text_builder_finish(CnpTextBuilder *builder) {
    if (!builder->data && !text_builder_reserve(builder, 0)) return NULL;
    size_t exact_size = builder->used + 1;
    if (builder->capacity == exact_size) return builder->data;
    char *exact = (char*)cnp_realloc(
        builder->data, builder->capacity, exact_size);
    if (!exact) {
        cnp_free(builder->data, builder->capacity);
        builder->data = NULL;
        cnp_set_error(
            CNP_ERR_MEMORY, builder->function_name,
            "cannot finalize text result");
        return NULL;
    }
    builder->data = exact;
    builder->capacity = exact_size;
    return exact;
}

static void text_builder_discard(CnpTextBuilder *builder) {
    if (builder->data) {
        cnp_free(builder->data, builder->capacity);
        builder->data = NULL;
    }
}

bool cnp_text_float_format_is_valid(
        const char *format, const char *function_name) {
    if (!format || !format[0]) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "a non-empty floating-point format is required");
        return false;
    }
    int conversions = 0;
    for (const char *cursor = format; *cursor; ++cursor) {
        if (*cursor != '%') continue;
        ++cursor;
        if (*cursor == '%') continue;
        if (!*cursor) break;
        while (*cursor && strchr("-+ #0", *cursor)) ++cursor;
        while (*cursor >= '0' && *cursor <= '9') ++cursor;
        if (*cursor == '.') {
            ++cursor;
            while (*cursor >= '0' && *cursor <= '9') ++cursor;
        }
        if (*cursor == 'l' || *cursor == 'L') ++cursor;
        if (!*cursor || !strchr("aAeEfFgG", *cursor)) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "format must contain exactly one floating-point conversion");
            return false;
        }
        ++conversions;
    }
    if (conversions != 1) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "format must contain exactly one floating-point conversion");
        return false;
    }
    return true;
}

static int64_t text_flat_offset(
        const CnpArray *array, int64_t flat_index) {
    int64_t remaining = flat_index;
    int64_t offset = array->offset;
    for (int dimension = array->ndim - 1; dimension >= 0; --dimension) {
        int64_t coordinate = remaining % array->shape[dimension];
        remaining /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

static bool text_append_default_scalar(
        CnpTextBuilder *builder, const CnpArray *array,
        int64_t flat_index, int precision, bool suppress_small) {
    int64_t offset = text_flat_offset(array, flat_index);
    const char *source = (const char*)array->data + offset;
    switch (array->dtype->type_num) {
        case CNP_BOOL:
            return text_builder_append(
                builder, *(const uint8_t*)source ? "True" : "False");
        case CNP_BYTE:
            return text_builder_append_format(
                builder, "%d", (int)*(const int8_t*)source);
        case CNP_UBYTE:
            return text_builder_append_format(
                builder, "%u", (unsigned int)*(const uint8_t*)source);
        case CNP_SHORT:
            return text_builder_append_format(
                builder, "%d", (int)*(const int16_t*)source);
        case CNP_USHORT:
            return text_builder_append_format(
                builder, "%u", (unsigned int)*(const uint16_t*)source);
        case CNP_INT:
            return text_builder_append_format(
                builder, "%d", *(const int32_t*)source);
        case CNP_UINT:
            return text_builder_append_format(
                builder, "%u", *(const uint32_t*)source);
        case CNP_LONG:
        case CNP_LONGLONG:
            return text_builder_append_format(
                builder, "%lld", (long long)*(const int64_t*)source);
        case CNP_ULONG:
        case CNP_ULONGLONG:
            return text_builder_append_format(
                builder, "%llu", (unsigned long long)*(const uint64_t*)source);
        case CNP_HALF: {
            double value = cnp_half_to_float(*(const uint16_t*)source);
            if (suppress_small && isfinite(value) &&
                    fabs(value) < pow(10.0, -precision)) value = 0.0;
            return text_builder_append_format(builder, "%.*g", precision, value);
        }
        case CNP_FLOAT: {
            double value = *(const float*)source;
            if (suppress_small && isfinite(value) &&
                    fabs(value) < pow(10.0, -precision)) value = 0.0;
            return text_builder_append_format(builder, "%.*g", precision, value);
        }
        case CNP_DOUBLE: {
            double value = *(const double*)source;
            if (suppress_small && isfinite(value) &&
                    fabs(value) < pow(10.0, -precision)) value = 0.0;
            return text_builder_append_format(builder, "%.*g", precision, value);
        }
        case CNP_LONGDOUBLE: {
            long double value = *(const long double*)source;
            if (suppress_small && isfinite((double)value) &&
                    fabsl(value) < powl(10.0L, -precision)) value = 0.0L;
            return text_builder_append_format(builder, "%.*Lg", precision, value);
        }
        case CNP_CFLOAT: {
            const cnp_cfloat *value = (const cnp_cfloat*)source;
            return text_builder_append_format(
                builder, "(%.*g%+.*gj)",
                precision, (double)value->real,
                precision, (double)value->imag);
        }
        case CNP_CDOUBLE: {
            const cnp_cdouble *value = (const cnp_cdouble*)source;
            return text_builder_append_format(
                builder, "(%.*g%+.*gj)",
                precision, value->real, precision, value->imag);
        }
        case CNP_CLONGDOUBLE: {
            const cnp_clongdouble *value = (const cnp_clongdouble*)source;
            return text_builder_append_format(
                builder, "(%.*Lg%+.*Lgj)",
                precision, value->real, precision, value->imag);
        }
        default:
            cnp_set_error(
                CNP_ERR_TYPE, builder->function_name,
                "dtype %s has no numeric text representation",
                array->dtype->name);
            return false;
    }
}

static bool text_append_scalar(
        CnpTextBuilder *builder, const CnpArray *array,
        int64_t flat_index, const char *format,
        int precision, bool suppress_small) {
    if (!format) {
        return text_append_default_scalar(
            builder, array, flat_index, precision, suppress_small);
    }
    if (cnp_type_is_complex(array->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, builder->function_name,
            "the legacy single floating format cannot represent complex values");
        return false;
    }
    return text_builder_append_format(
        builder, format, cnp_array_flat_get(array, flat_index));
}

static int64_t text_dimension_span(
        const CnpArray *array, int dimension) {
    int64_t span = 1;
    for (int index = dimension + 1; index < array->ndim; ++index)
        span *= array->shape[index];
    return span;
}

static bool text_append_dimension(
        CnpTextBuilder *builder, const CnpArray *array,
        int dimension, int64_t flat_start,
        const char *format, int precision,
        int64_t threshold, int edgeitems, bool suppress_small) {
    if (!text_builder_append(builder, "[")) return false;
    int64_t length = array->shape[dimension];
    bool summarized = array->size > threshold &&
        length > (int64_t)edgeitems * 2;
    int64_t span = text_dimension_span(array, dimension);
    int64_t leading = summarized ? edgeitems : length;
    int64_t trailing_start = summarized ? length - edgeitems : length;
    for (int64_t index = 0; index < length; ++index) {
        if (summarized && index == leading) {
            if (index > 0 && !text_builder_append(builder, ", ")) return false;
            if (!text_builder_append(builder, "...")) return false;
            index = trailing_start - 1;
            continue;
        }
        if (index > 0) {
            if (dimension == array->ndim - 1) {
                if (!text_builder_append(builder, ", ")) return false;
            } else {
                if (!text_builder_append(builder, ",\n")) return false;
                for (int spaces = 0; spaces <= dimension; ++spaces) {
                    if (!text_builder_append(builder, " ")) return false;
                }
            }
        }
        int64_t child_start = flat_start + index * span;
        if (dimension == array->ndim - 1) {
            if (!text_append_scalar(
                    builder, array, child_start, format,
                    precision, suppress_small)) return false;
        } else if (!text_append_dimension(
                builder, array, dimension + 1, child_start,
                format, precision, threshold, edgeitems,
                suppress_small)) {
            return false;
        }
    }
    return text_builder_append(builder, "]");
}

char *cnp_text_array_string(
        const CnpArray *arr, const char *format,
        int precision, int64_t threshold, int edgeitems, bool suppress_small,
        const char *function_name) {
    CnpTextBuilder builder = {0};
    builder.function_name = function_name;
    if (!arr || !arr->dtype) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "array is required");
        return NULL;
    }
    if (format && !cnp_text_float_format_is_valid(format, function_name))
        return NULL;
    if (precision < 0 || threshold < 0 || edgeitems < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "precision, threshold, and edgeitems must be nonnegative");
        return NULL;
    }
    bool success;
    if (arr->ndim == 0) {
        success = text_append_scalar(
            &builder, arr, 0, format, precision, suppress_small);
    } else if (arr->size == 0) {
        success = text_builder_append(&builder, "[]");
    } else {
        success = text_append_dimension(
            &builder, arr, 0, 0, format,
            precision, threshold, edgeitems, suppress_small);
    }
    if (!success) {
        text_builder_discard(&builder);
        return NULL;
    }
    return text_builder_finish(&builder);
}

static char *cnp_array_default_to_string(
        const CnpArray *arr, const char *format,
        const char *function_name) {
    return cnp_text_array_string(
        arr, format, 8, INT64_MAX, 3, false, function_name);
}

static char *cnp_copy_custom_string(
    const CnpArray *arr, CnpStringFunctionState state,
    const char *function_name) {
    const char *borrowed = state.function(arr, state.userdata);
    if (!borrowed) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "custom string callback returned null");
        return NULL;
    }
    size_t length = strlen(borrowed);
    char *owned = (char*)cnp_malloc(length + 1);
    if (!owned) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "cannot allocate custom string result");
        return NULL;
    }
    memcpy(owned, borrowed, length + 1);
    return owned;
}

CNP_API CNP_STATUS CNP_CALL cnp_set_string_function_v2(
    CnpStringFunction func, void *userdata, bool repr) {
    CnpStringFunctionState *state =
        repr ? &g_repr_string_function : &g_str_string_function;
    state->function = func;
    state->userdata = func ? userdata : NULL;
    return CNP_OK;
}

CNP_API CNP_STATUS CNP_CALL cnp_set_string_function(void *func) {
    CnpStringFunction callback = NULL;
    memcpy(&callback, &func, sizeof(callback));
    return cnp_set_string_function_v2(callback, NULL, true);
}

CNP_API char* CNP_CALL cnp_array_string_v2(
    const CnpArray *arr, bool repr) {
    const char *function_name = "cnp_array_string_v2";
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "array is required");
        return NULL;
    }
    CnpStringFunctionState state =
        repr ? g_repr_string_function : g_str_string_function;
    if (state.function)
        return cnp_copy_custom_string(arr, state, function_name);
    return cnp_array_default_to_string(arr, NULL, function_name);
}

CNP_API char* CNP_CALL cnp_array_to_string(
    const CnpArray *arr, const char *fmt) {
    if (g_repr_string_function.function)
        return cnp_array_string_v2(arr, true);
    return cnp_array_default_to_string(
        arr, fmt, "cnp_array_to_string");
}

CNP_API void CNP_CALL cnp_array_print(const CnpArray *arr) {
    const char *function_name = "cnp_array_print";
    if (!arr) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "array is required");
        return;
    }
    char *str = cnp_array_string_v2(arr, false);
    if (str) {
        if (printf("%s\n", str) < 0 || fflush(stdout) != 0) {
            cnp_set_error(
                CNP_ERR_IO, function_name,
                "cannot write array text to stdout");
        }
        cnp_free(str, strlen(str) + 1);
    }
}

CNP_API CNP_STATUS CNP_CALL cnp_array_to_csv(const CnpArray *arr, char *buffer, size_t bufsize, const char *delimiter) {
    const char *function_name = "cnp_array_to_csv";
    CnpTextBuilder builder = {0};
    builder.function_name = function_name;
    if (buffer && bufsize > 0) buffer[0] = '\0';
    if (!arr || !arr->dtype || !buffer || bufsize == 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "an array and non-empty writable buffer are required");
        return CNP_ERR_VALUE;
    }
    if (!delimiter) delimiter = ",";
    if (!delimiter[0]) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "delimiter must not be empty");
        return CNP_ERR_VALUE;
    }
    if (arr->ndim > 2) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "CSV projection requires a scalar, 1-D, or 2-D array");
        return CNP_ERR_SHAPE;
    }
    if (cnp_type_is_complex(arr->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "legacy CSV projection cannot represent complex values");
        return CNP_ERR_TYPE;
    }
    int64_t rows = arr->ndim == 2 ? arr->shape[0] : 1;
    int64_t columns = arr->ndim == 2
        ? arr->shape[1] : arr->ndim == 1 ? arr->shape[0] : 1;
    bool success = true;
    for (int64_t row = 0; success && row < rows; ++row) {
        for (int64_t column = 0; success && column < columns; ++column) {
            if (column > 0)
                success = text_builder_append(&builder, delimiter);
            int64_t flat = row * columns + column;
            if (success) {
                success = text_builder_append_format(
                    &builder, "%g", cnp_array_flat_get(arr, flat));
            }
        }
        if (success && row + 1 < rows)
            success = text_builder_append(&builder, "\n");
    }
    if (!success) {
        text_builder_discard(&builder);
        return cnp_get_error(NULL);
    }
    char *text = text_builder_finish(&builder);
    if (!text) return cnp_get_error(NULL);
    size_t required = strlen(text) + 1;
    if (required > bufsize) {
        cnp_free(text, required);
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "output buffer needs %llu bytes but has %llu",
            (unsigned long long)required,
            (unsigned long long)bufsize);
        return CNP_ERR_VALUE;
    }
    memcpy(buffer, text, required);
    cnp_free(text, required);
    return CNP_OK;
}

/* =========================================================================
 * Miscellaneous functions
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_copy(const CnpArray *arr) {
    return cnp_array_copy(arr);
}

static void cnp_meshgrid_release_results(CnpArray **results, int count) {
    if (!results) return;
    for (int index = 0; index < count; ++index) {
        if (results[index]) {
            cnp_array_decref(results[index]);
            results[index] = NULL;
        }
    }
}

static CNP_STATUS cnp_meshgrid_fail(
    CnpArray **results, int count, CNP_STATUS status) {
    cnp_meshgrid_release_results(results, count);
    return status;
}

CNP_API CNP_STATUS CNP_CALL cnp_meshgrid_v2(
    int narrays, CnpArray *const *arrays,
    bool sparse, bool indexing_ij, bool copy,
    CnpArray **results, int result_capacity) {
    const char *function_name = "cnp_meshgrid_v2";
    if (narrays < 0 || result_capacity < 0) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "array count and result capacity must be non-negative");
        return CNP_ERR_GENERIC;
    }
    if (results) {
        for (int index = 0; index < result_capacity; ++index)
            results[index] = NULL;
    }
    if (narrays > CNP_MAXDIMS) {
        cnp_set_error(CNP_ERR_SHAPE, function_name,
                      "meshgrid rank %d exceeds CNP_MAXDIMS %d",
                      narrays, CNP_MAXDIMS);
        return CNP_ERR_SHAPE;
    }
    if (result_capacity < narrays) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "result capacity %d is smaller than array count %d",
                      result_capacity, narrays);
        return CNP_ERR_GENERIC;
    }
    if ((narrays > 0 && !arrays) || (result_capacity > 0 && !results)) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "input and result arrays must not be null");
        return CNP_ERR_GENERIC;
    }
    if (narrays == 0) return CNP_OK;

    int64_t dense_shape[CNP_MAXDIMS];
    for (int array_index = 0; array_index < narrays; ++array_index) {
        if (!arrays[array_index]) {
            cnp_set_error(CNP_ERR_GENERIC, function_name,
                          "input array %d is null", array_index);
            return CNP_ERR_GENERIC;
        }
        dense_shape[array_index] = arrays[array_index]->size;
    }
    if (!indexing_ij && narrays > 1) {
        int64_t first = dense_shape[0];
        dense_shape[0] = dense_shape[1];
        dense_shape[1] = first;
    }

    for (int array_index = 0; array_index < narrays; ++array_index) {
        CnpArray *flat = NULL;
        if (arrays[array_index]->ndim == 1) {
            flat = arrays[array_index];
            cnp_array_incref(flat);
        } else {
            int64_t flat_shape = arrays[array_index]->size;
            flat = cnp_reshape(
                arrays[array_index], 1, &flat_shape, CNP_ORDER_C);
            if (!flat)
                return cnp_meshgrid_fail(
                    results, narrays, cnp_get_error(NULL));
        }

        int active_axis = array_index;
        if (!indexing_ij && narrays > 1) {
            if (array_index == 0) active_axis = 1;
            else if (array_index == 1) active_axis = 0;
        }

        int64_t sparse_shape[CNP_MAXDIMS];
        int64_t sparse_strides[CNP_MAXDIMS];
        for (int axis = 0; axis < narrays; ++axis)
            sparse_shape[axis] = axis == active_axis ? flat->size : 1;

        int64_t stride = flat->size == 0
            ? flat->dtype->elsize : flat->strides[0];
        for (int axis = narrays - 1; axis >= 0; --axis) {
            sparse_strides[axis] = stride;
            if (sparse_shape[axis] > 1)
                stride *= sparse_shape[axis];
        }

        CnpArray *candidate = cnp_array_view_from_metadata(
            flat, narrays, sparse_shape, sparse_strides, flat->offset, 0);
        cnp_array_decref(flat);
        if (!candidate)
            return cnp_meshgrid_fail(
                results, narrays, cnp_get_error(NULL));

        if (!sparse) {
            CnpArray *dense = cnp_broadcast_to(
                candidate, narrays, dense_shape);
            cnp_array_decref(candidate);
            candidate = dense;
            if (!candidate)
                return cnp_meshgrid_fail(
                    results, narrays, cnp_get_error(NULL));
        }

        if (copy) {
            CnpArray *owned = cnp_array_copy(candidate);
            cnp_array_decref(candidate);
            candidate = owned;
            if (!candidate)
                return cnp_meshgrid_fail(
                    results, narrays, cnp_get_error(NULL));
        }
        results[array_index] = candidate;
    }
    return CNP_OK;
}

CNP_API CnpArray* CNP_CALL cnp_meshgrid(
    int narrays, CnpArray **arrays, bool sparse, bool indexing_ij) {
    if (narrays <= 0 || narrays > CNP_MAXDIMS) return NULL;
    CnpArray *results[CNP_MAXDIMS] = {0};
    CNP_STATUS status = cnp_meshgrid_v2(
        narrays, arrays, sparse, indexing_ij, true,
        results, narrays);
    if (status != CNP_OK) return NULL;
    CnpArray *first = results[0];
    for (int index = 1; index < narrays; ++index)
        cnp_array_decref(results[index]);
    return first;
}

CNP_API CnpArray* CNP_CALL cnp_indices(int ndim, const int64_t *dimensions) {
    const char *function_name = "cnp_indices";
    if (ndim <= 0 || ndim >= CNP_MAXDIMS || !dimensions) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "dimension count must be in [1, %d] and dimensions must not be null",
            CNP_MAXDIMS - 1);
        return NULL;
    }
    int64_t total = 1;
    int64_t shape[CNP_MAXDIMS];
    shape[0] = ndim;
    for (int dimension = 0; dimension < ndim; ++dimension) {
        int64_t length = dimensions[dimension];
        if (length < 0) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "dimension %d must be non-negative", dimension);
            return NULL;
        }
        shape[dimension + 1] = length;
        if (length == 0) {
            total = 0;
        } else if (total > 0) {
            if (total > INT64_MAX / length) {
                cnp_set_error(
                    CNP_ERR_SHAPE, function_name,
                    "indices element count overflows int64");
                return NULL;
            }
            total *= length;
        }
    }
    CnpArray *result = cnp_array_new(
        ndim + 1, shape, CNP_LONGLONG, CNP_ORDER_C);
    if (!result) return NULL;

    int64_t coords[CNP_MAXDIMS] = {0};
    for (int64_t i = 0; i < total; i++) {
        for (int d = 0; d < ndim; d++) {
            *((int64_t*)result->data + d * total + i) = coords[d];
        }
        for (int d = ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < dimensions[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_fromfunction(double (*func)(const int64_t*, int, void*),
                                              int ndim, const int64_t *shape, void *userdata) {
    const char *function_name = "cnp_fromfunction";
    if (!func) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "callback is required");
        return NULL;
    }
    if (ndim < 0 || ndim > CNP_MAXDIMS) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "ndim must be in [0, %d], got %d", CNP_MAXDIMS, ndim);
        return NULL;
    }
    if (ndim > 0 && !shape) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "shape is required");
        return NULL;
    }
    CnpArray *arr = cnp_array_new(ndim, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!arr) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int64_t coords[CNP_MAXDIMS] = {0};
    double *data = (double*)arr->data;
    for (int64_t i = 0; i < arr->size; i++) {
        data[i] = func(coords, ndim, userdata);
        for (int d = ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < shape[d]) break;
            coords[d] = 0;
        }
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_fromiter(double (*iter_func)(void*), void *state, int64_t count, CNP_TYPE dtype) {
    const char *function_name = "cnp_fromiter";
    if (!iter_func) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "iterator callback is required");
        return NULL;
    }
    if (count < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "count must be non-negative because the callback ABI has no exhaustion signal");
        return NULL;
    }
    if (!(dtype == CNP_BOOL || cnp_type_is_integer(dtype) ||
          cnp_type_is_float(dtype) || cnp_type_is_complex(dtype))) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "dtype %d cannot be constructed from double callback values",
            (int)dtype);
        return NULL;
    }
    int64_t shape[1] = {count};
    CnpArray *arr = cnp_array_new(1, shape, dtype, CNP_ORDER_C);
    if (!arr) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int elsize = arr->dtype->elsize;
    for (int64_t i = 0; i < count; i++) {
        double val = iter_func(state);
        cnp_set_element_double(arr->data, i * elsize, dtype, val);
    }
    return arr;
}
