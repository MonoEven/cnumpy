/**
 * cnumpy string operations - char module functions
 * Corresponds to numpy.char (string operations on arrays)
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>
#include <ctype.h>

/* =========================================================================
 * Internal: String array structure
 * For simplicity, we store strings as array of char pointers
 * ========================================================================= */
typedef struct {
    char **strings;
    int64_t count;
    size_t pointer_count;
    size_t max_len;
} CnpStringArray;

struct CnpStringListResult {
    int64_t outer_count;
    int64_t total_count;
    int64_t *offsets;
    char **tokens;
};

/* =========================================================================
 * Internal: Create string array from CnpArray of strings
 * ========================================================================= */
static void free_string_array(void *owner);

static bool validate_string_inputs(
    const char **values, int64_t count, const char *function_name) {
    if (count < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name, "count must be non-negative");
        return false;
    }
    if (count > 0 && !values) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "input strings are required when count is positive");
        return false;
    }
    for (int64_t index = 0; index < count; ++index) {
        if (!values[index]) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "input string at index %lld is null", (long long)index);
            return false;
        }
    }
    return true;
}

static bool broadcast_string_counts(
    int64_t left_count, int64_t right_count,
    const char *function_name, int64_t *result_count) {
    if (left_count == right_count) {
        *result_count = left_count;
        return true;
    }
    if (left_count == 1) {
        *result_count = right_count;
        return true;
    }
    if (right_count == 1) {
        *result_count = left_count;
        return true;
    }
    cnp_set_error(
        CNP_ERR_BROADCAST, function_name,
        "input shapes (%lld,) and (%lld,) cannot be broadcast",
        (long long)left_count, (long long)right_count);
    return false;
}

static bool checked_size_add(size_t left, size_t right, size_t *result) {
    if (left > SIZE_MAX - right) return false;
    *result = left + right;
    return true;
}

static bool checked_size_multiply(size_t left, size_t right, size_t *result) {
    if (left != 0 && right > SIZE_MAX / left) return false;
    *result = left * right;
    return true;
}

static bool char_is_space(unsigned char value) {
    return value == ' ' || value == '\t' || value == '\n' ||
           value == '\r' || value == '\v' || value == '\f';
}

static unsigned char char_to_upper(unsigned char value) {
    return value >= 'a' && value <= 'z'
        ? (unsigned char)(value - ('a' - 'A')) : value;
}

static unsigned char char_to_lower(unsigned char value) {
    return value >= 'A' && value <= 'Z'
        ? (unsigned char)(value + ('a' - 'A')) : value;
}

static CnpStringArray* create_string_array(
    int64_t count, size_t max_len, const char *function_name) {
    if (count < 0 || max_len == SIZE_MAX) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "string result dimensions are too large");
        return NULL;
    }
    CnpStringArray *sa = (CnpStringArray*)cnp_malloc(sizeof(CnpStringArray));
    if (!sa) {
        cnp_set_error(CNP_ERR_MEMORY, function_name, "cannot allocate result owner");
        return NULL;
    }
    sa->count = count;
    sa->max_len = max_len;
    sa->pointer_count = count > 0 ? (size_t)count : 1;
    if (sa->pointer_count > SIZE_MAX / sizeof(char*)) {
        cnp_free(sa, sizeof(CnpStringArray));
        cnp_set_error(CNP_ERR_MEMORY, function_name, "string table is too large");
        return NULL;
    }
    sa->strings = (char**)cnp_calloc(sa->pointer_count, sizeof(char*));
    if (!sa->strings) {
        cnp_free(sa, sizeof(CnpStringArray));
        cnp_set_error(CNP_ERR_MEMORY, function_name, "cannot allocate string table");
        return NULL;
    }
    for (int64_t i = 0; i < count; i++) {
        sa->strings[i] = (char*)cnp_calloc(max_len + 1, 1);
        if (!sa->strings[i]) {
            for (int64_t j = 0; j < i; j++) cnp_free(sa->strings[j], max_len + 1);
            cnp_free(sa->strings, sa->pointer_count * sizeof(char*));
            cnp_free(sa, sizeof(CnpStringArray));
            cnp_set_error(CNP_ERR_MEMORY, function_name, "cannot allocate result strings");
            return NULL;
        }
    }
    return sa;
}

static void free_string_array(void *owner) {
    CnpStringArray *sa = (CnpStringArray*)owner;
    if (!sa) return;
    for (int64_t i = 0; i < sa->count; i++) {
        if (sa->strings[i]) cnp_free(sa->strings[i], sa->max_len + 1);
    }
    cnp_free(sa->strings, sa->pointer_count * sizeof(char*));
    cnp_free(sa, sizeof(CnpStringArray));
}

static CnpArray *finish_string_array(
    CnpStringArray *strings, const char *function_name) {
    int64_t shape[1] = {strings->count};
    CnpArray *result = cnp_array_adopt_external_data(
        1, shape, CNP_OBJECT, CNP_ORDER_C, strings->strings,
        CNP_ARRAY_OWNDATA | CNP_ARRAY_WRITEABLE | CNP_ARRAY_ALIGNED,
        strings, free_string_array, function_name);
    if (!result) free_string_array(strings);
    return result;
}

static void string_list_destroy(CnpStringListResult *result) {
    if (!result) return;
    if (result->tokens) {
        for (int64_t token = 0; token < result->total_count; ++token) {
            if (result->tokens[token]) {
                cnp_free(
                    result->tokens[token],
                    strlen(result->tokens[token]) + 1);
            }
        }
        cnp_free(
            result->tokens,
            (size_t)result->total_count * sizeof(char*));
    }
    if (result->offsets) {
        cnp_free(
            result->offsets,
            ((size_t)result->outer_count + 1) * sizeof(int64_t));
    }
    cnp_free(result, sizeof(CnpStringListResult));
}

static bool explicit_split_count(
    const char *value, const char *separator, int64_t maxsplit,
    int64_t *count) {
    size_t separator_length = strlen(separator);
    int64_t splits = 0;
    const char *cursor = value;
    while (maxsplit < 0 || splits < maxsplit) {
        const char *match = strstr(cursor, separator);
        if (!match) break;
        if (splits == INT64_MAX) return false;
        ++splits;
        cursor = match + separator_length;
    }
    *count = splits + 1;
    return true;
}

static int64_t whitespace_split_count(
    const char *value, int64_t maxsplit) {
    const unsigned char *cursor = (const unsigned char*)value;
    while (*cursor && char_is_space(*cursor)) ++cursor;
    if (!*cursor) return 0;

    int64_t count = 0;
    int64_t splits = 0;
    while (*cursor) {
        ++count;
        if (maxsplit >= 0 && splits >= maxsplit) break;
        while (*cursor && !char_is_space(*cursor)) ++cursor;
        while (*cursor && char_is_space(*cursor)) ++cursor;
        if (*cursor) ++splits;
    }
    return count;
}

static bool string_list_store_token(
    CnpStringListResult *result, int64_t index,
    const char *start, size_t length) {
    if (length == SIZE_MAX) return false;
    char *copy = (char*)cnp_malloc(length + 1);
    if (!copy) return false;
    memcpy(copy, start, length);
    copy[length] = '\0';
    result->tokens[index] = copy;
    return true;
}

/* =========================================================================
 * char_add - Add (concatenate) two string arrays element-wise
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_add(const char **a, int64_t na, const char **b, int64_t nb) {
    const char *function_name = "cnp_char_add";
    if (!validate_string_inputs(a, na, function_name) ||
        !validate_string_inputs(b, nb, function_name)) return NULL;
    int64_t n;
    if (!broadcast_string_counts(na, nb, function_name, &n)) return NULL;

    size_t max_len = 0;
    for (int64_t i = 0; i < n; i++) {
        const char *left = a[na == 1 ? 0 : i];
        const char *right = b[nb == 1 ? 0 : i];
        size_t len;
        if (!checked_size_add(strlen(left), strlen(right), &len)) {
            cnp_set_error(CNP_ERR_MEMORY, function_name, "result string is too large");
            return NULL;
        }
        if (len > max_len) max_len = len;
    }

    CnpStringArray *sa = create_string_array(n, max_len, function_name);
    if (!sa) return NULL;

    for (int64_t i = 0; i < n; i++) {
        const char *left = a[na == 1 ? 0 : i];
        const char *right = b[nb == 1 ? 0 : i];
        size_t left_len = strlen(left);
        size_t right_len = strlen(right);
        memcpy(sa->strings[i], left, left_len);
        memcpy(sa->strings[i] + left_len, right, right_len + 1);
    }
    return finish_string_array(sa, function_name);
}

/* =========================================================================
 * char_multiply - Repeat strings element-wise
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_multiply(const char **a, int64_t na, const int64_t *repeats, int64_t nr) {
    const char *function_name = "cnp_char_multiply";
    if (!validate_string_inputs(a, na, function_name)) return NULL;
    if (nr < 0 || (nr > 0 && !repeats)) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "repeats and a non-negative count are required");
        return NULL;
    }
    int64_t n;
    if (!broadcast_string_counts(na, nr, function_name, &n)) return NULL;

    size_t max_len = 0;
    for (int64_t i = 0; i < n; i++) {
        const char *value = a[na == 1 ? 0 : i];
        int64_t repeat = repeats[nr == 1 ? 0 : i];
        size_t len = 0;
        if (repeat > 0 && !checked_size_multiply(
                strlen(value), (size_t)repeat, &len)) {
            cnp_set_error(CNP_ERR_MEMORY, function_name, "result string is too large");
            return NULL;
        }
        if (len > max_len) max_len = len;
    }

    CnpStringArray *sa = create_string_array(n, max_len, function_name);
    if (!sa) return NULL;

    for (int64_t i = 0; i < n; i++) {
        const char *value = a[na == 1 ? 0 : i];
        int64_t repeat = repeats[nr == 1 ? 0 : i];
        size_t length = strlen(value);
        size_t position = 0;
        for (int64_t j = 0; j < repeat; j++) {
            memcpy(sa->strings[i] + position, value, length);
            position += length;
        }
        sa->strings[i][position] = '\0';
    }
    return finish_string_array(sa, function_name);
}

/* =========================================================================
 * char_upper - Convert strings to uppercase
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_upper(const char **a, int64_t n) {
    const char *function_name = "cnp_char_upper";
    if (!validate_string_inputs(a, n, function_name)) return NULL;

    size_t max_len = 0;
    for (int64_t i = 0; i < n; i++) {
        size_t len = strlen(a[i]);
        if (len > max_len) max_len = len;
    }

    CnpStringArray *sa = create_string_array(n, max_len, function_name);
    if (!sa) return NULL;

    for (int64_t i = 0; i < n; i++) {
        size_t len = strlen(a[i]);
        for (size_t j = 0; j < len; j++) {
            sa->strings[i][j] = (char)char_to_upper((unsigned char)a[i][j]);
        }
        sa->strings[i][len] = '\0';
    }
    return finish_string_array(sa, function_name);
}

/* =========================================================================
 * char_lower - Convert strings to lowercase
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_lower(const char **a, int64_t n) {
    const char *function_name = "cnp_char_lower";
    if (!validate_string_inputs(a, n, function_name)) return NULL;

    size_t max_len = 0;
    for (int64_t i = 0; i < n; i++) {
        size_t len = strlen(a[i]);
        if (len > max_len) max_len = len;
    }

    CnpStringArray *sa = create_string_array(n, max_len, function_name);
    if (!sa) return NULL;

    for (int64_t i = 0; i < n; i++) {
        size_t len = strlen(a[i]);
        for (size_t j = 0; j < len; j++) {
            sa->strings[i][j] = (char)char_to_lower((unsigned char)a[i][j]);
        }
        sa->strings[i][len] = '\0';
    }
    return finish_string_array(sa, function_name);
}

/* =========================================================================
 * char_strip - Strip leading/trailing characters
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_strip(const char **a, int64_t n, const char *chars) {
    const char *function_name = "cnp_char_strip";
    if (!validate_string_inputs(a, n, function_name)) return NULL;

    size_t max_len = 0;
    for (int64_t i = 0; i < n; i++) {
        size_t len = strlen(a[i]);
        if (len > max_len) max_len = len;
    }

    CnpStringArray *sa = create_string_array(n, max_len, function_name);
    if (!sa) return NULL;

    for (int64_t i = 0; i < n; i++) {
        const char *s = a[i];
        size_t len = strlen(s);
        size_t start = 0, end = len;

        if (chars) {
            while (start < len && strchr(chars, s[start])) start++;
            while (end > start && strchr(chars, s[end-1])) end--;
        } else {
            while (start < len && char_is_space((unsigned char)s[start])) start++;
            while (end > start && char_is_space((unsigned char)s[end-1])) end--;
        }

        size_t new_len = end - start;
        memcpy(sa->strings[i], s + start, new_len);
        sa->strings[i][new_len] = '\0';
    }
    return finish_string_array(sa, function_name);
}

/* =========================================================================
 * char_lstrip - Strip leading characters
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_lstrip(const char **a, int64_t n, const char *chars) {
    const char *function_name = "cnp_char_lstrip";
    if (!validate_string_inputs(a, n, function_name)) return NULL;

    size_t max_len = 0;
    for (int64_t i = 0; i < n; i++) {
        size_t len = strlen(a[i]);
        if (len > max_len) max_len = len;
    }

    CnpStringArray *sa = create_string_array(n, max_len, function_name);
    if (!sa) return NULL;

    for (int64_t i = 0; i < n; i++) {
        const char *s = a[i];
        size_t len = strlen(s);
        size_t start = 0;

        if (chars) {
            while (start < len && strchr(chars, s[start])) start++;
        } else {
            while (start < len && char_is_space((unsigned char)s[start])) start++;
        }

        memcpy(sa->strings[i], s + start, len - start + 1);
    }
    return finish_string_array(sa, function_name);
}

/* =========================================================================
 * char_rstrip - Strip trailing characters
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_rstrip(const char **a, int64_t n, const char *chars) {
    const char *function_name = "cnp_char_rstrip";
    if (!validate_string_inputs(a, n, function_name)) return NULL;

    size_t max_len = 0;
    for (int64_t i = 0; i < n; i++) {
        size_t len = strlen(a[i]);
        if (len > max_len) max_len = len;
    }

    CnpStringArray *sa = create_string_array(n, max_len, function_name);
    if (!sa) return NULL;

    for (int64_t i = 0; i < n; i++) {
        const char *s = a[i];
        size_t end = strlen(s);

        if (chars) {
            while (end > 0 && strchr(chars, s[end-1])) end--;
        } else {
            while (end > 0 && char_is_space((unsigned char)s[end-1])) end--;
        }

        memcpy(sa->strings[i], s, end);
        sa->strings[i][end] = '\0';
    }
    return finish_string_array(sa, function_name);
}

/* =========================================================================
 * char_center - Center strings in field of given width
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_center(const char **a, int64_t n, int64_t width, char fillchar) {
    const char *function_name = "cnp_char_center";
    if (!validate_string_inputs(a, n, function_name)) return NULL;
    if (width < 0) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "width must be non-negative");
        return NULL;
    }
    if (fillchar == '\0') {
        cnp_set_error(CNP_ERR_VALUE, function_name, "fill character cannot be NUL");
        return NULL;
    }
    size_t requested_width = (size_t)width;
    size_t output_width = width > 0 ? requested_width : 1;

    CnpStringArray *sa = create_string_array(n, output_width, function_name);
    if (!sa) return NULL;

    for (int64_t i = 0; i < n; i++) {
        size_t len = strlen(a[i]);
        if (len >= requested_width) {
            size_t copied = len < output_width ? len : output_width;
            memcpy(sa->strings[i], a[i], copied);
            sa->strings[i][copied] = '\0';
        } else {
            size_t total_pad = output_width - len;
            size_t left_pad = total_pad / 2 +
                ((total_pad & output_width & 1u) != 0u);
            size_t right_pad = total_pad - left_pad;
            size_t pos = 0;
            memset(sa->strings[i] + pos, (unsigned char)fillchar, left_pad);
            pos += left_pad;
            memcpy(sa->strings[i] + pos, a[i], len);
            pos += len;
            memset(sa->strings[i] + pos, (unsigned char)fillchar, right_pad);
            pos += right_pad;
            sa->strings[i][pos] = '\0';
        }
    }
    return finish_string_array(sa, function_name);
}

/* =========================================================================
 * char_ljust - Left justify strings
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_ljust(const char **a, int64_t n, int64_t width, char fillchar) {
    const char *function_name = "cnp_char_ljust";
    if (!validate_string_inputs(a, n, function_name)) return NULL;
    if (width < 0) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "width must be non-negative");
        return NULL;
    }
    if (fillchar == '\0') {
        cnp_set_error(CNP_ERR_VALUE, function_name, "fill character cannot be NUL");
        return NULL;
    }
    size_t requested_width = (size_t)width;
    size_t output_width = width > 0 ? requested_width : 1;

    CnpStringArray *sa = create_string_array(n, output_width, function_name);
    if (!sa) return NULL;

    for (int64_t i = 0; i < n; i++) {
        size_t len = strlen(a[i]);
        size_t copied = len < output_width ? len : output_width;
        memcpy(sa->strings[i], a[i], copied);
        if (requested_width > 0) {
            memset(
                sa->strings[i] + copied, (unsigned char)fillchar,
                output_width - copied);
            sa->strings[i][output_width] = '\0';
        } else {
            sa->strings[i][copied] = '\0';
        }
    }
    return finish_string_array(sa, function_name);
}

/* =========================================================================
 * char_rjust - Right justify strings
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_rjust(const char **a, int64_t n, int64_t width, char fillchar) {
    const char *function_name = "cnp_char_rjust";
    if (!validate_string_inputs(a, n, function_name)) return NULL;
    if (width < 0) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "width must be non-negative");
        return NULL;
    }
    if (fillchar == '\0') {
        cnp_set_error(CNP_ERR_VALUE, function_name, "fill character cannot be NUL");
        return NULL;
    }
    size_t requested_width = (size_t)width;
    size_t output_width = width > 0 ? requested_width : 1;

    CnpStringArray *sa = create_string_array(n, output_width, function_name);
    if (!sa) return NULL;

    for (int64_t i = 0; i < n; i++) {
        size_t len = strlen(a[i]);
        if (len >= requested_width) {
            size_t copied = len < output_width ? len : output_width;
            memcpy(sa->strings[i], a[i], copied);
            sa->strings[i][copied] = '\0';
            continue;
        }
        size_t pad = output_width - len;
        memset(sa->strings[i], (unsigned char)fillchar, pad);
        memcpy(sa->strings[i] + pad, a[i], len + 1);
    }
    return finish_string_array(sa, function_name);
}

/* =========================================================================
 * char_zfill - Pad strings with zeros on the left
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_zfill(const char **a, int64_t n, int64_t width) {
    const char *function_name = "cnp_char_zfill";
    if (!validate_string_inputs(a, n, function_name)) return NULL;
    if (width < 0) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "width must be non-negative");
        return NULL;
    }
    size_t requested_width = (size_t)width;
    size_t output_width = width > 0 ? requested_width : 1;

    CnpStringArray *sa = create_string_array(n, output_width, function_name);
    if (!sa) return NULL;

    for (int64_t i = 0; i < n; i++) {
        size_t len = strlen(a[i]);
        if (len >= requested_width) {
            size_t copied = len < output_width ? len : output_width;
            memcpy(sa->strings[i], a[i], copied);
            sa->strings[i][copied] = '\0';
            continue;
        }
        size_t pad = output_width - len;
        size_t pos = 0;
        if (len > 0 && (a[i][0] == '+' || a[i][0] == '-')) {
            sa->strings[i][pos++] = a[i][0];
            memset(sa->strings[i] + pos, '0', pad);
            pos += pad;
            memcpy(sa->strings[i] + pos, a[i] + 1, len);
        } else {
            memset(sa->strings[i], '0', pad);
            pos = pad;
            memcpy(sa->strings[i] + pos, a[i], len);
        }
        sa->strings[i][output_width] = '\0';
    }
    return finish_string_array(sa, function_name);
}

/* =========================================================================
 * char_strlen - Return length of each string
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_strlen(const char **a, int64_t n) {
    const char *function_name = "cnp_char_strlen";
    if (!validate_string_inputs(a, n, function_name)) return NULL;

    int64_t shape[1] = {n};
    CnpArray *result = cnp_array_new(1, shape, CNP_INT, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int32_t *data = (int32_t*)result->data;
    for (int64_t i = 0; i < n; i++) {
        size_t length = strlen(a[i]);
        if (length > INT32_MAX) {
            cnp_array_decref(result);
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "string length exceeds the int32 result range");
            return NULL;
        }
        data[i] = (int32_t)length;
    }
    return result;
}

/* =========================================================================
 * char_count - Count occurrences of substring
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_count(const char **a, int64_t n, const char *sub) {
    const char *function_name = "cnp_char_count";
    if (!validate_string_inputs(a, n, function_name)) return NULL;
    if (!sub) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "substring is required");
        return NULL;
    }

    size_t sub_len = strlen(sub);
    int64_t shape[1] = {n};
    CnpArray *result = cnp_array_new(1, shape, CNP_INT, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int32_t *data = (int32_t*)result->data;
    for (int64_t i = 0; i < n; i++) {
        size_t count = 0;
        if (sub_len == 0) {
            size_t length = strlen(a[i]);
            if (length >= INT32_MAX) {
                cnp_array_decref(result);
                cnp_set_error(
                    CNP_ERR_VALUE, function_name,
                    "substring count exceeds the int32 result range");
                return NULL;
            }
            count = length + 1;
        } else {
            const char *cursor = a[i];
            const char *match;
            while ((match = strstr(cursor, sub)) != NULL) {
                ++count;
                cursor = match + sub_len;
            }
            if (count > INT32_MAX) {
                cnp_array_decref(result);
                cnp_set_error(
                    CNP_ERR_VALUE, function_name,
                    "substring count exceeds the int32 result range");
                return NULL;
            }
        }
        data[i] = (int32_t)count;
    }
    return result;
}

/* =========================================================================
 * char_find - Find substring, return -1 if not found
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_find(const char **a, int64_t n, const char *sub) {
    const char *function_name = "cnp_char_find";
    if (!validate_string_inputs(a, n, function_name)) return NULL;
    if (!sub) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "substring is required");
        return NULL;
    }

    int64_t shape[1] = {n};
    CnpArray *result = cnp_array_new(1, shape, CNP_INT, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }

    int32_t *data = (int32_t*)result->data;
    for (int64_t i = 0; i < n; i++) {
        const char *found = strstr(a[i], sub);
        if (found && (size_t)(found - a[i]) > INT32_MAX) {
            cnp_array_decref(result);
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "substring index exceeds the int32 result range");
            return NULL;
        }
        data[i] = found ? (int32_t)(found - a[i]) : -1;
    }
    return result;
}

/* =========================================================================
 * char_replace - Replace substring
 * ========================================================================= */
static size_t replacement_occurrences(
    const char *value, const char *old_string, size_t old_length,
    int64_t replacement_limit) {
    if (replacement_limit == 0) return 0;
    size_t available = 0;
    if (old_length == 0) {
        available = strlen(value) + 1;
    } else {
        const char *cursor = value;
        const char *match;
        while ((match = strstr(cursor, old_string)) != NULL) {
            ++available;
            cursor = match + old_length;
        }
    }
    if (replacement_limit < 0 || (uint64_t)replacement_limit >= available)
        return available;
    return (size_t)replacement_limit;
}

CNP_API CnpArray* CNP_CALL cnp_char_replace(const char **a, int64_t n,
                                             const char *old_str, const char *new_str, int64_t count) {
    const char *function_name = "cnp_char_replace";
    if (!validate_string_inputs(a, n, function_name)) return NULL;
    if (!old_str || !new_str) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "old and new strings are required");
        return NULL;
    }

    size_t old_len = strlen(old_str);
    size_t new_len = strlen(new_str);

    size_t max_len = 0;
    for (int64_t i = 0; i < n; i++) {
        size_t len = strlen(a[i]);
        size_t occurrences = replacement_occurrences(
            a[i], old_str, old_len, count);
        size_t new_total;
        if (new_len >= old_len) {
            size_t added;
            if (!checked_size_multiply(
                    occurrences, new_len - old_len, &added) ||
                !checked_size_add(len, added, &new_total)) {
                cnp_set_error(
                    CNP_ERR_MEMORY, function_name,
                    "replacement result is too large");
                return NULL;
            }
        } else {
            new_total = len - occurrences * (old_len - new_len);
        }
        if (new_total > max_len) max_len = new_total;
    }

    CnpStringArray *sa = create_string_array(n, max_len, function_name);
    if (!sa) return NULL;

    for (int64_t i = 0; i < n; i++) {
        const char *s = a[i];
        size_t len = strlen(s);
        size_t pos = 0;
        size_t replacements = replacement_occurrences(
            s, old_str, old_len, count);

        if (old_len == 0) {
            size_t remaining = replacements;
            for (size_t source = 0; source <= len; ++source) {
                if (remaining > 0) {
                    memcpy(sa->strings[i] + pos, new_str, new_len);
                    pos += new_len;
                    --remaining;
                }
                if (source < len) sa->strings[i][pos++] = s[source];
            }
        } else {
            size_t source = 0;
            size_t remaining = replacements;
            while (source < len) {
                if (remaining > 0 && old_len <= len - source &&
                    memcmp(s + source, old_str, old_len) == 0) {
                    memcpy(sa->strings[i] + pos, new_str, new_len);
                    pos += new_len;
                    source += old_len;
                    --remaining;
                } else {
                    sa->strings[i][pos++] = s[source++];
                }
            }
        }
        sa->strings[i][pos] = '\0';
    }
    return finish_string_array(sa, function_name);
}

/* =========================================================================
 * char_split - Legacy ABI cannot represent a list for each input string
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_split(const char **a, int64_t n, const char *sep) {
    const char *function_name = "cnp_char_split";
    if (!validate_string_inputs(a, n, function_name)) return NULL;
    if (sep && sep[0] == '\0') {
        cnp_set_error(CNP_ERR_VALUE, function_name, "empty separator is not allowed");
        return NULL;
    }
    cnp_set_error(
        CNP_ERR_NOT_IMPLEMENTED, function_name,
        "legacy result cannot represent nested tokens; use cnp_char_split_v2");
    return NULL;
}

CNP_API CnpStringListResult* CNP_CALL cnp_char_split_v2(
    const char **a, int64_t n, const char *sep, int64_t maxsplit) {
    const char *function_name = "cnp_char_split_v2";
    if (n < 0 || (n > 0 && !a)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "input strings and a non-negative count are required");
        return NULL;
    }
    if (sep && sep[0] == '\0') {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "empty separator is not allowed");
        return NULL;
    }
    if ((uint64_t)n > (SIZE_MAX / sizeof(int64_t)) - 1) {
        cnp_set_error(CNP_ERR_MEMORY, function_name, "offset table is too large");
        return NULL;
    }

    CnpStringListResult *result = (CnpStringListResult*)cnp_calloc(
        1, sizeof(CnpStringListResult));
    if (!result) {
        cnp_set_error(CNP_ERR_MEMORY, function_name, "cannot allocate result");
        return NULL;
    }
    result->outer_count = n;
    result->offsets = (int64_t*)cnp_calloc(
        (size_t)n + 1, sizeof(int64_t));
    if (!result->offsets) {
        string_list_destroy(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name, "cannot allocate result offsets");
        return NULL;
    }

    int64_t total_count = 0;
    for (int64_t row = 0; row < n; ++row) {
        if (!a[row]) {
            string_list_destroy(result);
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "input string at row %lld is null", (long long)row);
            return NULL;
        }
        int64_t row_count;
        bool count_valid = true;
        if (sep) {
            count_valid = explicit_split_count(
                a[row], sep, maxsplit, &row_count);
        } else {
            row_count = whitespace_split_count(a[row], maxsplit);
        }
        if (!count_valid || row_count > INT64_MAX - total_count) {
            string_list_destroy(result);
            cnp_set_error(
                CNP_ERR_MEMORY, function_name, "token count is too large");
            return NULL;
        }
        total_count += row_count;
        result->offsets[row + 1] = total_count;
    }
    if ((uint64_t)total_count > SIZE_MAX / sizeof(char*)) {
        string_list_destroy(result);
        cnp_set_error(
            CNP_ERR_MEMORY, function_name, "token table is too large");
        return NULL;
    }
    result->total_count = total_count;
    if (total_count > 0) {
        result->tokens = (char**)cnp_calloc(
            (size_t)total_count, sizeof(char*));
        if (!result->tokens) {
            string_list_destroy(result);
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "cannot allocate token table");
            return NULL;
        }
    }

    int64_t token_index = 0;
    for (int64_t row = 0; row < n; ++row) {
        if (!sep) {
            const unsigned char *cursor = (const unsigned char*)a[row];
            while (*cursor && char_is_space(*cursor)) ++cursor;
            int64_t splits = 0;
            while (*cursor) {
                const unsigned char *start = cursor;
                if (maxsplit >= 0 && splits >= maxsplit) {
                    if (!string_list_store_token(
                            result, token_index++, (const char*)start,
                            strlen((const char*)start))) {
                        string_list_destroy(result);
                        cnp_set_error(
                            CNP_ERR_MEMORY, function_name,
                            "cannot allocate split token");
                        return NULL;
                    }
                    break;
                }
                while (*cursor && !char_is_space(*cursor)) ++cursor;
                if (!string_list_store_token(
                        result, token_index++, (const char*)start,
                        (size_t)(cursor - start))) {
                    string_list_destroy(result);
                    cnp_set_error(
                        CNP_ERR_MEMORY, function_name,
                        "cannot allocate split token");
                    return NULL;
                }
                while (*cursor && char_is_space(*cursor)) ++cursor;
                if (*cursor) ++splits;
            }
            continue;
        }

        size_t separator_length = strlen(sep);
        const char *cursor = a[row];
        int64_t splits = 0;
        while (maxsplit < 0 || splits < maxsplit) {
            const char *match = strstr(cursor, sep);
            if (!match) break;
            if (!string_list_store_token(
                    result, token_index++, cursor,
                    (size_t)(match - cursor))) {
                string_list_destroy(result);
                cnp_set_error(
                    CNP_ERR_MEMORY, function_name,
                    "cannot allocate split token");
                return NULL;
            }
            cursor = match + separator_length;
            ++splits;
        }
        if (!string_list_store_token(
                result, token_index++, cursor, strlen(cursor))) {
            string_list_destroy(result);
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "cannot allocate split token");
            return NULL;
        }
    }
    return result;
}

CNP_API int64_t CNP_CALL cnp_string_list_outer_count(
    const CnpStringListResult *result) {
    if (!result) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_string_list_outer_count",
            "result is required");
        return -1;
    }
    return result->outer_count;
}

CNP_API int64_t CNP_CALL cnp_string_list_token_count(
    const CnpStringListResult *result, int64_t row) {
    const char *function_name = "cnp_string_list_token_count";
    if (!result) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "result is required");
        return -1;
    }
    if (row < 0 || row >= result->outer_count) {
        cnp_set_error(
            CNP_ERR_INDEX, function_name, "row is out of bounds");
        return -1;
    }
    return result->offsets[row + 1] - result->offsets[row];
}

CNP_API const char* CNP_CALL cnp_string_list_get(
    const CnpStringListResult *result, int64_t row, int64_t token) {
    const char *function_name = "cnp_string_list_get";
    if (!result) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "result is required");
        return NULL;
    }
    if (row < 0 || row >= result->outer_count) {
        cnp_set_error(
            CNP_ERR_INDEX, function_name, "row is out of bounds");
        return NULL;
    }
    int64_t start = result->offsets[row];
    int64_t count = result->offsets[row + 1] - start;
    if (token < 0 || token >= count) {
        cnp_set_error(
            CNP_ERR_INDEX, function_name, "token is out of bounds");
        return NULL;
    }
    return result->tokens[start + token];
}

CNP_API void CNP_CALL cnp_string_list_free(CnpStringListResult *result) {
    string_list_destroy(result);
}

/* =========================================================================
 * char_join_v2 - Insert a scalar separator between bytes of each string
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_char_join_v2(
    const char **a, int64_t n, const char *sep) {
    const char *function_name = "cnp_char_join_v2";
    if (!validate_string_inputs(a, n, function_name)) return NULL;
    if (!sep) sep = "";
    size_t separator_length = strlen(sep);
    size_t max_len = 0;
    for (int64_t index = 0; index < n; ++index) {
        size_t length = strlen(a[index]);
        size_t output_length = length;
        if (length > 1) {
            size_t separator_bytes;
            if (!checked_size_multiply(
                    length - 1, separator_length, &separator_bytes) ||
                !checked_size_add(
                    length, separator_bytes, &output_length)) {
                cnp_set_error(
                    CNP_ERR_MEMORY, function_name,
                    "joined result string is too large");
                return NULL;
            }
        }
        if (output_length > max_len) max_len = output_length;
    }

    CnpStringArray *strings = create_string_array(n, max_len, function_name);
    if (!strings) return NULL;
    for (int64_t index = 0; index < n; ++index) {
        size_t length = strlen(a[index]);
        size_t position = 0;
        for (size_t source = 0; source < length; ++source) {
            if (source > 0) {
                memcpy(
                    strings->strings[index] + position,
                    sep, separator_length);
                position += separator_length;
            }
            strings->strings[index][position++] = a[index][source];
        }
        strings->strings[index][position] = '\0';
    }
    return finish_string_array(strings, function_name);
}

/* =========================================================================
 * char_join - Legacy scalar sequence join convenience ABI
 * ========================================================================= */
CNP_API char* CNP_CALL cnp_char_join(const char **a, int64_t n, const char *sep) {
    const char *function_name = "cnp_char_join";
    if (!validate_string_inputs(a, n, function_name)) return NULL;
    if (!sep) sep = "";

    size_t sep_len = strlen(sep);
    size_t total_len = 0;
    for (int64_t i = 0; i < n; i++) {
        if (!checked_size_add(total_len, strlen(a[i]), &total_len) ||
            (i < n - 1 &&
             !checked_size_add(total_len, sep_len, &total_len))) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "joined result string is too large");
            return NULL;
        }
    }
    if (total_len == SIZE_MAX) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "joined result string is too large");
        return NULL;
    }

    char *result = (char*)cnp_malloc(total_len + 1);
    if (!result) {
        cnp_set_error(CNP_ERR_MEMORY, function_name, "cannot allocate joined string");
        return NULL;
    }

    size_t position = 0;
    for (int64_t i = 0; i < n; i++) {
        size_t length = strlen(a[i]);
        memcpy(result + position, a[i], length);
        position += length;
        if (i < n - 1) {
            memcpy(result + position, sep, sep_len);
            position += sep_len;
        }
    }
    result[position] = '\0';
    return result;
}

/* =========================================================================
 * Utility: Free string array result
 * ========================================================================= */
CNP_API void CNP_CALL cnp_char_free_result(CnpArray *arr) {
    cnp_array_decref(arr);
}

CNP_API void CNP_CALL cnp_char_free_string(char *str) {
    if (str) {
        cnp_free(str, strlen(str) + 1);
    }
}
