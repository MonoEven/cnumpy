/**
 * cnumpy structured arrays and record arrays.
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>

typedef struct {
    char *name;
    CNP_TYPE type;
    int64_t offset;
    int64_t size;
} CnpFieldDesc;

typedef struct {
    CnpFieldDesc *fields;
    int nfields;
    int64_t itemsize;
    char *joined_names;
} CnpStructuredDtype;

typedef struct {
    uint64_t magic;
    int dtype_id;
    size_t allocation_bytes;
    void *data;
    CnpDtype dtype;
} CnpRecordOwner;

#define CNP_RECORD_OWNER_MAGIC UINT64_C(0x434e505245434f52)

/* Registry metadata has process/DLL lifetime, like the static scalar dtype
 * table. It is intentionally outside retained-array byte accounting and is
 * released by cnp_cleanup. */
static CnpStructuredDtype *g_struct_dtypes = NULL;
static size_t g_struct_dtype_count = 0;
static size_t g_struct_dtype_capacity = 0;

static void structured_free_descriptor(CnpStructuredDtype *descriptor) {
    if (!descriptor) return;
    if (descriptor->fields) {
        for (int field = 0; field < descriptor->nfields; field++) {
            free(descriptor->fields[field].name);
        }
        free(descriptor->fields);
    }
    free(descriptor->joined_names);
    memset(descriptor, 0, sizeof(*descriptor));
}

void cnp_structured_cleanup(void) {
    for (size_t index = 0; index < g_struct_dtype_count; index++) {
        structured_free_descriptor(&g_struct_dtypes[index]);
    }
    free(g_struct_dtypes);
    g_struct_dtypes = NULL;
    g_struct_dtype_count = 0;
    g_struct_dtype_capacity = 0;
}

static CnpStructuredDtype *structured_lookup(
        int dtype_id, const char *function_name) {
    if (dtype_id < 0 || (size_t)dtype_id >= g_struct_dtype_count) {
        cnp_set_error(
            CNP_ERR_INDEX, function_name,
            "structured dtype id %d is invalid", dtype_id);
        return NULL;
    }
    return &g_struct_dtypes[dtype_id];
}

static int structured_find_index(
        const CnpStructuredDtype *descriptor, const char *name) {
    if (!descriptor || !name) return -1;
    for (int index = 0; index < descriptor->nfields; index++) {
        if (strcmp(descriptor->fields[index].name, name) == 0) return index;
    }
    return -1;
}

static char *structured_copy_name(const char *name) {
    size_t length = strlen(name);
    char *copy = (char*)malloc(length + 1);
    if (copy) memcpy(copy, name, length + 1);
    return copy;
}

CNP_API int CNP_CALL cnp_struct_dtype_create(
        const char **names, const CNP_TYPE *types, int nfields) {
    const char *function_name = "cnp_struct_dtype_create";
    CnpStructuredDtype descriptor = {0};
    size_t joined_length = 1;
    if (nfields <= 0 || !names || !types) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "names, types, and a positive field count are required");
        return -1;
    }
    if ((size_t)nfields > SIZE_MAX / sizeof(CnpFieldDesc)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "field descriptor table is too large");
        return -1;
    }
    descriptor.fields = (CnpFieldDesc*)calloc(
        (size_t)nfields, sizeof(CnpFieldDesc));
    if (!descriptor.fields) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate field descriptor table");
        return -1;
    }
    descriptor.nfields = nfields;
    for (int index = 0; index < nfields; index++) {
        if (!names[index] || names[index][0] == '\0') {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "field %d must have a non-empty name", index);
            goto failure;
        }
        for (int previous = 0; previous < index; previous++) {
            if (strcmp(names[index], names[previous]) == 0) {
                cnp_set_error(
                    CNP_ERR_VALUE, function_name,
                    "field name '%s' is duplicated", names[index]);
                goto failure;
            }
        }
        CnpDtype *dtype = cnp_dtype_new(types[index]);
        if (!dtype || dtype->elsize <= 0 ||
                types[index] == CNP_VOID || types[index] == CNP_OBJECT) {
            if (dtype) {
                cnp_set_error(
                    CNP_ERR_TYPE, function_name,
                    "field '%s' has an unsupported dtype %d",
                    names[index], (int)types[index]);
            } else {
                cnp_relabel_error(function_name);
            }
            goto failure;
        }
        if (descriptor.itemsize > INT_MAX - dtype->elsize) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "packed record item size exceeds the represented range");
            goto failure;
        }
        size_t name_length = strlen(names[index]);
        if (name_length > SIZE_MAX - joined_length - 1) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "joined field names are too large");
            goto failure;
        }
        descriptor.fields[index].name = structured_copy_name(names[index]);
        if (!descriptor.fields[index].name) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "failed to copy field name '%s'", names[index]);
            goto failure;
        }
        descriptor.fields[index].type = types[index];
        descriptor.fields[index].offset = descriptor.itemsize;
        descriptor.fields[index].size = dtype->elsize;
        descriptor.itemsize += dtype->elsize;
        joined_length += name_length + (index == 0 ? 0 : 1);
    }
    descriptor.joined_names = (char*)malloc(joined_length);
    if (!descriptor.joined_names) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate joined field names");
        goto failure;
    }
    char *cursor = descriptor.joined_names;
    for (int index = 0; index < nfields; index++) {
        if (index != 0) *cursor++ = ',';
        size_t length = strlen(descriptor.fields[index].name);
        memcpy(cursor, descriptor.fields[index].name, length);
        cursor += length;
    }
    *cursor = '\0';

    if (g_struct_dtype_count == g_struct_dtype_capacity) {
        size_t next_capacity = g_struct_dtype_capacity == 0
            ? 8 : g_struct_dtype_capacity * 2;
        if (next_capacity < g_struct_dtype_capacity ||
                next_capacity > SIZE_MAX / sizeof(CnpStructuredDtype)) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "structured dtype registry is too large");
            goto failure;
        }
        CnpStructuredDtype *resized = (CnpStructuredDtype*)realloc(
            g_struct_dtypes,
            next_capacity * sizeof(CnpStructuredDtype));
        if (!resized) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "failed to grow structured dtype registry");
            goto failure;
        }
        g_struct_dtypes = resized;
        g_struct_dtype_capacity = next_capacity;
    }
    int dtype_id = (int)g_struct_dtype_count;
    g_struct_dtypes[g_struct_dtype_count++] = descriptor;
    return dtype_id;

failure:
    structured_free_descriptor(&descriptor);
    return -1;
}

CNP_API int64_t CNP_CALL cnp_struct_dtype_itemsize(int dtype_id) {
    CnpStructuredDtype *descriptor = structured_lookup(
        dtype_id, "cnp_struct_dtype_itemsize");
    return descriptor ? descriptor->itemsize : -1;
}

CNP_API int CNP_CALL cnp_struct_dtype_nfields(int dtype_id) {
    CnpStructuredDtype *descriptor = structured_lookup(
        dtype_id, "cnp_struct_dtype_nfields");
    return descriptor ? descriptor->nfields : -1;
}

CNP_API const char* CNP_CALL cnp_struct_dtype_field_name(
        int dtype_id, int field_idx) {
    const char *function_name = "cnp_struct_dtype_field_name";
    CnpStructuredDtype *descriptor = structured_lookup(
        dtype_id, function_name);
    if (!descriptor) return NULL;
    if (field_idx < 0 || field_idx >= descriptor->nfields) {
        cnp_set_error(
            CNP_ERR_INDEX, function_name,
            "field index %d is invalid", field_idx);
        return NULL;
    }
    return descriptor->fields[field_idx].name;
}

CNP_API int64_t CNP_CALL cnp_struct_dtype_field_offset(
        int dtype_id, int field_idx) {
    const char *function_name = "cnp_struct_dtype_field_offset";
    CnpStructuredDtype *descriptor = structured_lookup(
        dtype_id, function_name);
    if (!descriptor) return -1;
    if (field_idx < 0 || field_idx >= descriptor->nfields) {
        cnp_set_error(
            CNP_ERR_INDEX, function_name,
            "field index %d is invalid", field_idx);
        return -1;
    }
    return descriptor->fields[field_idx].offset;
}

CNP_API int CNP_CALL cnp_struct_dtype_find_field(
        int dtype_id, const char *name) {
    const char *function_name = "cnp_struct_dtype_find_field";
    CnpStructuredDtype *descriptor = structured_lookup(
        dtype_id, function_name);
    if (!descriptor) return -1;
    if (!name) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "field name is required");
        return -1;
    }
    int index = structured_find_index(descriptor, name);
    if (index < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "field '%s' does not exist", name);
    }
    return index;
}

static void record_owner_release(void *owner_pointer) {
    CnpRecordOwner *owner = (CnpRecordOwner*)owner_pointer;
    if (!owner) return;
    if (owner->data) {
        cnp_free(owner->data, owner->allocation_bytes);
    }
    owner->magic = 0;
    cnp_free(owner, sizeof(CnpRecordOwner));
}

static CnpRecordOwner *record_owner_for(
        const CnpArray *array, int dtype_id,
        const char *function_name) {
    if (!array) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "record array is required");
        return NULL;
    }
    const CnpArray *root = array;
    while (root->base) root = root->base;
    if (root->owner_release != record_owner_release || !root->owner) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "array is not a cnumpy record array");
        return NULL;
    }
    CnpRecordOwner *owner = (CnpRecordOwner*)root->owner;
    if (owner->magic != CNP_RECORD_OWNER_MAGIC ||
            owner->dtype_id != dtype_id) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "record array does not use structured dtype id %d", dtype_id);
        return NULL;
    }
    return owner;
}

static int64_t structured_flat_offset(
        const CnpArray *array, int64_t flat_index) {
    int64_t remaining = flat_index;
    int64_t offset = array->offset;
    for (int dimension = array->ndim - 1; dimension >= 0; dimension--) {
        int64_t coordinate = remaining % array->shape[dimension];
        remaining /= array->shape[dimension];
        offset += coordinate * array->strides[dimension];
    }
    return offset;
}

CNP_API CnpArray* CNP_CALL cnp_recarray_new(
        int ndim, const int64_t *shape, int struct_dtype_id) {
    const char *function_name = "cnp_recarray_new";
    CnpStructuredDtype *descriptor = structured_lookup(
        struct_dtype_id, function_name);
    CnpArray *array = NULL;
    CnpRecordOwner *owner = NULL;
    int64_t size = 1;
    if (!descriptor) return NULL;
    if (ndim < 0 || ndim > CNP_MAXDIMS || (ndim > 0 && !shape)) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "shape metadata is invalid for rank %d", ndim);
        return NULL;
    }
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (shape[dimension] < 0 ||
                (shape[dimension] != 0 &&
                 size > INT64_MAX / shape[dimension])) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "record array shape is invalid or too large");
            return NULL;
        }
        size *= shape[dimension];
    }
    if ((uint64_t)size > SIZE_MAX / (uint64_t)descriptor->itemsize) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "record array data buffer is too large");
        return NULL;
    }
    size_t data_bytes = (size_t)size * (size_t)descriptor->itemsize;
    size_t allocation_bytes = data_bytes == 0 ? 1 : data_bytes;
    array = (CnpArray*)cnp_calloc(1, sizeof(CnpArray));
    owner = (CnpRecordOwner*)cnp_calloc(1, sizeof(CnpRecordOwner));
    if (!array || !owner) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate record array metadata");
        goto failure;
    }
    if (ndim > 0) {
        array->shape = (int64_t*)cnp_malloc(
            (size_t)ndim * sizeof(int64_t));
        array->strides = (int64_t*)cnp_malloc(
            (size_t)ndim * sizeof(int64_t));
        if (!array->shape || !array->strides) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "failed to allocate record array shape metadata");
            goto failure;
        }
        int64_t stride = descriptor->itemsize;
        for (int dimension = ndim - 1; dimension >= 0; dimension--) {
            array->shape[dimension] = shape[dimension];
            array->strides[dimension] = stride;
            stride *= shape[dimension];
        }
    }
    owner->data = cnp_calloc(allocation_bytes, 1);
    if (!owner->data) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate record array data");
        goto failure;
    }
    CnpDtype *void_dtype = cnp_dtype_new(CNP_VOID);
    if (!void_dtype) {
        cnp_relabel_error(function_name);
        goto failure;
    }
    owner->magic = CNP_RECORD_OWNER_MAGIC;
    owner->dtype_id = struct_dtype_id;
    owner->allocation_bytes = allocation_bytes;
    owner->dtype = *void_dtype;
    owner->dtype.elsize = (int)descriptor->itemsize;
    owner->dtype.alignment = 1;
    owner->dtype.byteorder = '|';
    strncpy(owner->dtype.name, "structured", sizeof(owner->dtype.name) - 1);
    owner->dtype.name[sizeof(owner->dtype.name) - 1] = '\0';
    owner->dtype.refcount = 1;

    array->ndim = ndim;
    array->size = size;
    array->data = owner->data;
    array->dtype = &owner->dtype;
    array->flags = CNP_ARRAY_ALIGNED | CNP_ARRAY_WRITEABLE |
        cnp_compute_layout_flags(
            ndim, array->shape, array->strides,
            (int)descriptor->itemsize);
    array->refcount = 1;
    array->owner = owner;
    array->owner_release = record_owner_release;
    return array;

failure:
    if (owner) {
        if (owner->data) cnp_free(owner->data, allocation_bytes);
        cnp_free(owner, sizeof(CnpRecordOwner));
    }
    if (array) {
        if (array->shape) {
            cnp_free(array->shape, (size_t)ndim * sizeof(int64_t));
        }
        if (array->strides) {
            cnp_free(array->strides, (size_t)ndim * sizeof(int64_t));
        }
        cnp_free(array, sizeof(CnpArray));
    }
    return NULL;
}

CNP_API CnpArray* CNP_CALL cnp_recarray_get_field(
        const CnpArray *array, const char *field_name,
        int struct_dtype_id) {
    const char *function_name = "cnp_recarray_get_field";
    CnpStructuredDtype *descriptor = structured_lookup(
        struct_dtype_id, function_name);
    if (!descriptor ||
            !record_owner_for(array, struct_dtype_id, function_name)) {
        return NULL;
    }
    if (!field_name) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "field name is required");
        return NULL;
    }
    int field_index = structured_find_index(descriptor, field_name);
    if (field_index < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "field '%s' does not exist", field_name);
        return NULL;
    }
    CnpFieldDesc *field = &descriptor->fields[field_index];
    CnpArray *result = cnp_array_view_from_metadata(
        (CnpArray*)array, array->ndim,
        array->shape, array->strides,
        array->offset + field->offset, 0);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    result->dtype = cnp_dtype_new(field->type);
    result->flags = (result->flags &
        ~(CNP_ARRAY_C_CONTIGUOUS | CNP_ARRAY_F_CONTIGUOUS)) |
        cnp_compute_layout_flags(
            result->ndim, result->shape,
            result->strides, result->dtype->elsize);
    return result;
}

CNP_API CNP_STATUS CNP_CALL cnp_recarray_set_field(
        CnpArray *array, const char *field_name,
        int struct_dtype_id, const CnpArray *values) {
    const char *function_name = "cnp_recarray_set_field";
    CnpArray *cast_values = NULL;
    CnpArray *broadcast_values = NULL;
    CnpStructuredDtype *descriptor = structured_lookup(
        struct_dtype_id, function_name);
    if (!descriptor ||
            !record_owner_for(array, struct_dtype_id, function_name)) {
        return cnp_get_error(NULL);
    }
    if (!field_name || !values) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "field name and values array are required");
        return CNP_ERR_GENERIC;
    }
    if (!(array->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "record array is not writeable");
        return CNP_ERR_GENERIC;
    }
    int field_index = structured_find_index(descriptor, field_name);
    if (field_index < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "field '%s' does not exist", field_name);
        return CNP_ERR_VALUE;
    }
    CnpFieldDesc *field = &descriptor->fields[field_index];
    cast_values = cnp_astype(values, field->type, CNP_CAST_UNSAFE);
    if (!cast_values) goto failure;
    broadcast_values = cnp_broadcast_to(
        cast_values, array->ndim, array->shape);
    if (!broadcast_values) goto failure;
    for (int64_t index = 0; index < array->size; index++) {
        int64_t remaining = index;
        int64_t destination_offset = array->offset + field->offset;
        int64_t source_offset = broadcast_values->offset;
        for (int dimension = array->ndim - 1;
                dimension >= 0; dimension--) {
            int64_t coordinate = remaining % array->shape[dimension];
            remaining /= array->shape[dimension];
            destination_offset += coordinate * array->strides[dimension];
            source_offset += coordinate * broadcast_values->strides[dimension];
        }
        memcpy(
            (char*)array->data + destination_offset,
            (const char*)broadcast_values->data + source_offset,
            (size_t)field->size);
    }
    cnp_array_free(broadcast_values);
    cnp_array_free(cast_values);
    return CNP_OK;

failure:
    if (broadcast_values) cnp_array_free(broadcast_values);
    if (cast_values) cnp_array_free(cast_values);
    cnp_relabel_error(function_name);
    return cnp_get_error(NULL);
}

CNP_API CnpArray* CNP_CALL cnp_recarray_get_record(
        const CnpArray *array, int64_t index,
        int struct_dtype_id) {
    const char *function_name = "cnp_recarray_get_record";
    CnpStructuredDtype *descriptor = structured_lookup(
        struct_dtype_id, function_name);
    if (!descriptor ||
            !record_owner_for(array, struct_dtype_id, function_name)) {
        return NULL;
    }
    if (index < 0) index += array->size;
    if (index < 0 || index >= array->size) {
        cnp_set_error(
            CNP_ERR_INDEX, function_name,
            "record index %lld is out of bounds",
            (long long)index);
        return NULL;
    }
    for (int field = 0; field < descriptor->nfields; field++) {
        CNP_TYPE type = descriptor->fields[field].type;
        if (cnp_type_is_complex(type) ||
                !(type == CNP_BOOL || cnp_type_is_integer(type) ||
                  cnp_type_is_float(type))) {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "legacy float64 record projection cannot represent field '%s'",
                descriptor->fields[field].name);
            return NULL;
        }
    }
    int64_t shape[1] = {descriptor->nfields};
    CnpArray *result = cnp_array_new(
        1, shape, CNP_DOUBLE, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    int64_t record_offset = structured_flat_offset(array, index);
    for (int field = 0; field < descriptor->nfields; field++) {
        CNP_STATUS status = cnp_cast_scalar_value(
            (const char*)array->data + record_offset +
                descriptor->fields[field].offset,
            descriptor->fields[field].type,
            &((double*)result->data)[field],
            CNP_DOUBLE, function_name);
        if (status != CNP_OK) {
            cnp_array_free(result);
            return NULL;
        }
    }
    return result;
}

CNP_API CNP_STATUS CNP_CALL cnp_recarray_set_record(
        CnpArray *array, int64_t index, int struct_dtype_id,
        const CnpArray *values) {
    const char *function_name = "cnp_recarray_set_record";
    CnpStructuredDtype *descriptor = structured_lookup(
        struct_dtype_id, function_name);
    if (!descriptor ||
            !record_owner_for(array, struct_dtype_id, function_name)) {
        return cnp_get_error(NULL);
    }
    if (!values) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "record values array is required");
        return CNP_ERR_GENERIC;
    }
    if (!(array->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "record array is not writeable");
        return CNP_ERR_GENERIC;
    }
    if (index < 0) index += array->size;
    if (index < 0 || index >= array->size) {
        cnp_set_error(
            CNP_ERR_INDEX, function_name,
            "record index %lld is out of bounds", (long long)index);
        return CNP_ERR_INDEX;
    }
    if (values->size != descriptor->nfields) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "record requires exactly %d field values",
            descriptor->nfields);
        return CNP_ERR_SHAPE;
    }
    size_t itemsize = (size_t)descriptor->itemsize;
    char *temporary = (char*)cnp_malloc(itemsize);
    if (!temporary) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate atomic record workspace");
        return CNP_ERR_MEMORY;
    }
    int64_t record_offset = structured_flat_offset(array, index);
    memcpy(temporary, (const char*)array->data + record_offset, itemsize);
    for (int field = 0; field < descriptor->nfields; field++) {
        int64_t source_offset = structured_flat_offset(values, field);
        CNP_STATUS status = cnp_cast_scalar_value(
            (const char*)values->data + source_offset,
            values->dtype->type_num,
            temporary + descriptor->fields[field].offset,
            descriptor->fields[field].type,
            function_name);
        if (status != CNP_OK) {
            cnp_free(temporary, itemsize);
            return status;
        }
    }
    memcpy((char*)array->data + record_offset, temporary, itemsize);
    cnp_free(temporary, itemsize);
    return CNP_OK;
}

CNP_API char* CNP_CALL cnp_recarray_names(int struct_dtype_id) {
    CnpStructuredDtype *descriptor = structured_lookup(
        struct_dtype_id, "cnp_recarray_names");
    return descriptor ? descriptor->joined_names : NULL;
}
