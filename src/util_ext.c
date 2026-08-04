/**
 * cnumpy utility and misc functions
 * Corresponds to numpy: seterr, geterr, min_scalar_type, common_type,
 *   emath (sqrt, log, log2, log10, power, arcsin, arccos, arctanh),
 *   array2string, set_printoptions, get_printoptions, nditer helpers,
 *   can_cast public, typename, sctype2char, isscalar, iscomplexobj, isrealobj
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>
#include <stdio.h>

/* =========================================================================
 * Error state management
 * numpy.seterr / numpy.geterr
 * ========================================================================= */
static int g_err_divide = 0;  /* 0=warn, 1=raise, 2=ignore, 3=call */
static int g_err_over = 0;
static int g_err_under = 2;
static int g_err_invalid = 0;

CNP_API void CNP_CALL cnp_seterr(int divide, int over, int under, int invalid) {
    const char *function_name = "cnp_seterr";
    if (divide < 0 || divide > 3 || over < 0 || over > 3 ||
            under < 0 || under > 3 || invalid < 0 || invalid > 3) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "error modes must be warn(0), raise(1), ignore(2), or call(3)");
        return;
    }
    g_err_divide = divide;
    g_err_over = over;
    g_err_under = under;
    g_err_invalid = invalid;
}

CNP_API void CNP_CALL cnp_geterr(int *divide, int *over, int *under, int *invalid) {
    if (divide) *divide = g_err_divide;
    if (over) *over = g_err_over;
    if (under) *under = g_err_under;
    if (invalid) *invalid = g_err_invalid;
}

/* =========================================================================
 * cnp_min_scalar_type - Return minimum scalar type to hold value
 * numpy.min_scalar_type(a)
 * ========================================================================= */
CNP_API CNP_TYPE CNP_CALL cnp_min_scalar_type(double value) {
    if (!isfinite(value)) return CNP_HALF;
    double magnitude = fabs(value);
    if (magnitude < 65000.0) return CNP_HALF;
    if (magnitude < 3.4e38) return CNP_FLOAT;
    return CNP_DOUBLE;
}

/* =========================================================================
 * cnp_common_type - Return common type for arrays
 * numpy.common_type(*arrays)
 * ========================================================================= */
CNP_API CNP_TYPE CNP_CALL cnp_common_type(int narrays, const CnpArray **arrays) {
    const char *function_name = "cnp_common_type";
    if (narrays < 0 || (narrays > 0 && !arrays)) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "array pointer list is required for a positive array count");
        return CNP_NOTYPE;
    }
    if (narrays == 0) return CNP_HALF;

    int real_precision = -1;
    int complex_precision = -1;
    bool has_integer = false;
    for (int i = 0; i < narrays; ++i) {
        if (!arrays[i] || !arrays[i]->dtype) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "array %d and its dtype are required", i);
            return CNP_NOTYPE;
        }
        CNP_TYPE type = arrays[i]->dtype->type_num;
        if (cnp_type_is_integer(type)) {
            has_integer = true;
        } else if (type == CNP_HALF) {
            if (real_precision < 0) real_precision = 0;
        } else if (type == CNP_FLOAT) {
            if (real_precision < 1) real_precision = 1;
        } else if (type == CNP_DOUBLE) {
            if (real_precision < 2) real_precision = 2;
        } else if (type == CNP_LONGDOUBLE) {
            if (real_precision < 3) real_precision = 3;
        } else if (type == CNP_CFLOAT) {
            if (complex_precision < 1) complex_precision = 1;
        } else if (type == CNP_CDOUBLE) {
            if (complex_precision < 2) complex_precision = 2;
        } else if (type == CNP_CLONGDOUBLE) {
            if (complex_precision < 3) complex_precision = 3;
        } else {
            cnp_set_error(
                CNP_ERR_TYPE, function_name,
                "array %d dtype %d is not supported by numpy.common_type",
                i, (int)type);
            return CNP_NOTYPE;
        }
    }
    if (has_integer && real_precision < 2) real_precision = 2;
    if (complex_precision >= 0) {
        int precision = complex_precision > real_precision
            ? complex_precision : real_precision;
        if (precision <= 1) return CNP_CFLOAT;
        if (precision == 2) return CNP_CDOUBLE;
        return CNP_CLONGDOUBLE;
    }
    if (real_precision <= 0) return CNP_HALF;
    if (real_precision == 1) return CNP_FLOAT;
    if (real_precision == 2) return CNP_DOUBLE;
    return CNP_LONGDOUBLE;
}

/* =========================================================================
 * NumPy emath/scimath wrappers
 *
 * NumPy first promotes a real input to the corresponding complex dtype only
 * when its represented domain requires it, then dispatches the ordinary
 * ufunc.  Reusing the core ufuncs here keeps dtype promotion, broadcasting,
 * complex branch cuts, logical strides, and special values on one path.
 * ========================================================================= */
typedef CnpArray* (CNP_CALL *CnpEmathUnaryKernel)(const CnpArray *array);

typedef enum {
    CNP_EMATH_NEGATIVE,
    CNP_EMATH_ABS_GREATER_ONE
} CnpEmathDomain;

static bool emath_numeric_type(CNP_TYPE type) {
    return type == CNP_BOOL || cnp_type_is_integer(type) ||
           cnp_type_is_float(type) || cnp_type_is_complex(type);
}

static bool emath_validate_array(
    const CnpArray *array, const char *role, const char *function_name) {
    if (!array) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name, "%s array is required", role);
        return false;
    }
    int64_t ignored_nbytes;
    if (!cnp_array_nbytes_checked(
            array, function_name, &ignored_nbytes)) return false;
    if (!array->dtype || !emath_numeric_type(array->dtype->type_num)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "%s array must have a represented numeric dtype", role);
        return false;
    }
    return true;
}

static bool emath_needs_complex(
    const CnpArray *array, CnpEmathDomain domain) {
    if (cnp_type_is_complex(array->dtype->type_num)) return false;
    for (int64_t index = 0; index < array->size; ++index) {
        double value = cnp_array_flat_get(array, index);
        if (domain == CNP_EMATH_NEGATIVE) {
            if (value < 0.0) return true;
        } else if (fabs(value) > 1.0) {
            return true;
        }
    }
    return false;
}

static CNP_TYPE emath_complex_type(CNP_TYPE source_type) {
    if (source_type == CNP_FLOAT) return CNP_CFLOAT;
    if (source_type == CNP_LONGDOUBLE) return CNP_CLONGDOUBLE;
    return CNP_CDOUBLE;
}

static CnpArray *emath_unary(
    const CnpArray *array, CnpEmathDomain domain,
    CnpEmathUnaryKernel kernel, const char *function_name) {
    if (!emath_validate_array(array, "input", function_name)) return NULL;
    CnpArray *converted = NULL;
    const CnpArray *operand = array;
    if (emath_needs_complex(array, domain)) {
        converted = cnp_astype(
            array, emath_complex_type(array->dtype->type_num), CNP_CAST_SAFE);
        if (!converted) {
            cnp_relabel_error(function_name);
            return NULL;
        }
        operand = converted;
    }
    CnpArray *result = kernel(operand);
    if (converted) cnp_array_decref(converted);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_emath_sqrt(const CnpArray *arr) {
    return emath_unary(
        arr, CNP_EMATH_NEGATIVE, cnp_sqrt, "cnp_emath_sqrt");
}

CNP_API CnpArray* CNP_CALL cnp_emath_log(const CnpArray *arr) {
    return emath_unary(
        arr, CNP_EMATH_NEGATIVE, cnp_log, "cnp_emath_log");
}

CNP_API CnpArray* CNP_CALL cnp_emath_log10(const CnpArray *arr) {
    return emath_unary(
        arr, CNP_EMATH_NEGATIVE, cnp_log10, "cnp_emath_log10");
}

CNP_API CnpArray* CNP_CALL cnp_emath_log2(const CnpArray *arr) {
    return emath_unary(
        arr, CNP_EMATH_NEGATIVE, cnp_log2, "cnp_emath_log2");
}

CNP_API CnpArray* CNP_CALL cnp_emath_arcsin(const CnpArray *arr) {
    return emath_unary(
        arr, CNP_EMATH_ABS_GREATER_ONE,
        cnp_arcsin, "cnp_emath_arcsin");
}

CNP_API CnpArray* CNP_CALL cnp_emath_arccos(const CnpArray *arr) {
    return emath_unary(
        arr, CNP_EMATH_ABS_GREATER_ONE,
        cnp_arccos, "cnp_emath_arccos");
}

CNP_API CnpArray* CNP_CALL cnp_emath_arctanh(const CnpArray *arr) {
    return emath_unary(
        arr, CNP_EMATH_ABS_GREATER_ONE,
        cnp_arctanh, "cnp_emath_arctanh");
}

CNP_API CnpArray* CNP_CALL cnp_emath_power(
    const CnpArray *base, const CnpArray *exp_arr) {
    const char *function_name = "cnp_emath_power";
    if (!emath_validate_array(base, "base", function_name) ||
        !emath_validate_array(exp_arr, "exponent", function_name))
        return NULL;

    CnpArray *converted = NULL;
    const CnpArray *operand = base;
    if (emath_needs_complex(base, CNP_EMATH_NEGATIVE)) {
        converted = cnp_astype(
            base, emath_complex_type(base->dtype->type_num), CNP_CAST_SAFE);
        if (!converted) {
            cnp_relabel_error(function_name);
            return NULL;
        }
        operand = converted;
    }
    CnpArray *result = cnp_power(operand, exp_arr);
    if (converted) cnp_array_decref(converted);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

/* =========================================================================
 * Print options
 * numpy.set_printoptions / numpy.get_printoptions
 * ========================================================================= */
static int g_precision = 8;
static int g_threshold = 1000;
static int g_edgeitems = 3;
static int g_linewidth = 75;
static int g_suppress = 0;

CNP_API void CNP_CALL cnp_set_printoptions(int precision, int threshold, int edgeitems,
                                            int linewidth, int suppress) {
    const char *function_name = "cnp_set_printoptions";
    if (precision < -1 || threshold < -1 || edgeitems < -1 ||
            linewidth < -1 || (suppress != 0 && suppress != 1)) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "print options must be nonnegative or -1 (unchanged), and suppress must be 0 or 1");
        return;
    }
    if (precision >= 0) g_precision = precision;
    if (threshold >= 0) g_threshold = threshold;
    if (edgeitems >= 0) g_edgeitems = edgeitems;
    if (linewidth >= 0) g_linewidth = linewidth;
    g_suppress = suppress;
}

CNP_API void CNP_CALL cnp_get_printoptions(int *precision, int *threshold, int *edgeitems,
                                            int *linewidth, int *suppress) {
    if (precision) *precision = g_precision;
    if (threshold) *threshold = g_threshold;
    if (edgeitems) *edgeitems = g_edgeitems;
    if (linewidth) *linewidth = g_linewidth;
    if (suppress) *suppress = g_suppress;
}

/* =========================================================================
 * cnp_array2string - Convert array to string representation
 * numpy.array2string(a)
 * ========================================================================= */
CNP_API int CNP_CALL cnp_array2string(const CnpArray *arr, char *buffer, int64_t bufsize) {
    const char *function_name = "cnp_array2string";
    if (buffer && bufsize > 0) buffer[0] = '\0';
    if (!arr || !buffer || bufsize <= 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "an array and non-empty writable buffer are required");
        return -1;
    }
    char *text = cnp_text_array_string(
        arr, NULL, g_precision, g_threshold,
        g_edgeitems, g_suppress != 0, function_name);
    if (!text) return -1;
    size_t length = strlen(text);
    if (length > INT_MAX || length + 1 > (uint64_t)bufsize) {
        cnp_free(text, length + 1);
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "output buffer needs %llu bytes but has %lld",
            (unsigned long long)(length + 1), (long long)bufsize);
        return -1;
    }
    memcpy(buffer, text, length + 1);
    cnp_free(text, length + 1);
    return (int)length;
}

/* =========================================================================
 * cnp_iscomplexobj - Check if array has complex dtype
 * numpy.iscomplexobj(x)
 * ========================================================================= */
static bool object_kind_type(
    const CnpArray *arr, const char *function_name, CNP_TYPE *type) {
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return false;
    }
    if (!arr->dtype) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array must have a dtype");
        return false;
    }
    *type = arr->dtype->type_num;
    if (*type <= CNP_NOTYPE || *type >= CNP_NTYPES) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "source array dtype %d is not a valid CNP_TYPE", (int)*type);
        return false;
    }
    return true;
}

CNP_API bool CNP_CALL cnp_iscomplexobj(const CnpArray *arr) {
    CNP_TYPE type;
    if (!object_kind_type(arr, "cnp_iscomplexobj", &type)) return false;
    return cnp_type_is_complex(type);
}

/* =========================================================================
 * cnp_isrealobj - Check if array has real dtype
 * numpy.isrealobj(x)
 * ========================================================================= */
CNP_API bool CNP_CALL cnp_isrealobj(const CnpArray *arr) {
    CNP_TYPE type;
    if (!object_kind_type(arr, "cnp_isrealobj", &type)) return false;
    return !cnp_type_is_complex(type);
}

/* =========================================================================
 * cnp_isscalar - Check whether a CnpArray argument is a scalar object
 * numpy.isscalar(x)
 * Every valid CnpArray represents an ndarray, including 0-D arrays, so the
 * NumPy-compatible result within this C ABI's input domain is always false.
 * ========================================================================= */
CNP_API bool CNP_CALL cnp_isscalar(const CnpArray *arr) {
    CNP_TYPE type;
    if (!object_kind_type(arr, "cnp_isscalar", &type)) return false;
    (void)type;
    return false;
}

/* =========================================================================
 * cnp_typename - Return type name string for dtype
 * numpy.typename(type)
 * ========================================================================= */
CNP_API const char* CNP_CALL cnp_typename(CNP_TYPE type) {
    switch (type) {
        case CNP_BOOL: return "bool";
        case CNP_BYTE: return "signed char";
        case CNP_UBYTE: return "unsigned char";
        case CNP_SHORT: return "short";
        case CNP_USHORT: return "unsigned short";
        case CNP_INT: return "integer";
        case CNP_UINT: return "unsigned integer";
        case CNP_LONG: return "long integer";
        case CNP_ULONG: return "unsigned long integer";
        case CNP_LONGLONG: return "long long integer";
        case CNP_ULONGLONG: return "unsigned long long integer";
        case CNP_FLOAT: return "single precision";
        case CNP_DOUBLE: return "double precision";
        case CNP_LONGDOUBLE: return "long precision";
        case CNP_CFLOAT: return "complex single precision";
        case CNP_CDOUBLE: return "complex double precision";
        case CNP_CLONGDOUBLE: return "complex long double precision";
        case CNP_OBJECT: return "object";
        case CNP_STRING: return "string";
        case CNP_UNICODE: return "unicode";
        case CNP_VOID: return "void";
        default:
            cnp_set_error(
                CNP_ERR_TYPE, "cnp_typename",
                "CNP_TYPE %d has no NumPy typename", (int)type);
            return NULL;
    }
}

/* =========================================================================
 * cnp_ndenumerate_next - Iterate over array elements with coordinates
 * Helper for numpy.ndenumerate pattern
 * ========================================================================= */
CNP_API bool CNP_CALL cnp_ndenumerate_next(const CnpArray *arr, int64_t *iter_state,
                                            int64_t *coords, double *value) {
    const char *function_name = "cnp_ndenumerate_next";
    if (!arr || !iter_state) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "array and iterator state must not be null");
        return false;
    }
    if (*iter_state < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "iterator state must be non-negative");
        return false;
    }
    if (*iter_state >= arr->size) return false;

    int64_t idx = *iter_state;
    if (coords) {
        int64_t tmp = idx;
        for (int d = arr->ndim - 1; d >= 0; d--) {
            coords[d] = tmp % arr->shape[d];
            tmp /= arr->shape[d];
        }
    }
    if (value) *value = cnp_array_flat_get(arr, idx);
    (*iter_state)++;
    return true;
}

/* =========================================================================
 * cnp_ndindex_next - Generate next multi-index for given shape
 * Helper for numpy.ndindex pattern
 * ========================================================================= */
CNP_API bool CNP_CALL cnp_ndindex_next(int ndim, const int64_t *shape, int64_t *coords) {
    const char *function_name = "cnp_ndindex_next";
    if (ndim <= 0 || !shape || !coords) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "positive ndim, shape, and coordinates are required");
        return false;
    }
    for (int dimension = 0; dimension < ndim; ++dimension) {
        if (shape[dimension] < 0) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "shape dimension %d must be non-negative", dimension);
            return false;
        }
        if (shape[dimension] == 0) return false;
    }
    /* Increment from last dimension */
    for (int d = ndim - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < shape[d]) return true;
        coords[d] = 0;
    }
    return false;  /* All done */
}
