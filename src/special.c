/**
 * cnumpy special math functions - gamma, beta, erf, bessel, etc.
 * Corresponds to scipy.special and numpy advanced math functions
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <limits.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef double (*CnpRealUnaryFunction)(
    double value, CNP_TYPE result_dtype);

static double special_positive_infinity(void) {
    uint64_t bits = UINT64_C(0x7ff0000000000000);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool special_is_nan(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) ==
               UINT64_C(0x7ff0000000000000) &&
           (bits & UINT64_C(0x000fffffffffffff)) != 0;
}

static bool special_is_infinite(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7fffffffffffffff)) ==
        UINT64_C(0x7ff0000000000000);
}

static bool special_real_dtype_supported(
    const CnpArray *source, const char *function_name) {
    char kind = source->dtype->kind;
    if (kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f')
        return true;
    cnp_set_error(
        CNP_ERR_TYPE, function_name,
        "source array must have a real numeric dtype");
    return false;
}

static CNP_TYPE special_real_result_dtype(CNP_TYPE source_dtype) {
    switch (source_dtype) {
        case CNP_BOOL:
        case CNP_BYTE:
        case CNP_UBYTE:
        case CNP_SHORT:
        case CNP_USHORT:
        case CNP_HALF:
        case CNP_FLOAT:
            return CNP_FLOAT;
        default:
            return CNP_DOUBLE;
    }
}

static uint64_t special_stride_magnitude(int64_t stride) {
    if (stride >= 0) return (uint64_t)stride;
    return (uint64_t)(-(stride + 1)) + 1;
}

static CnpArray *special_new_keep_order(
    const CnpArray *source, CNP_TYPE result_dtype,
    const char *function_name) {
    CNP_ORDER initial_order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, initial_order);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (source->size == 0 || source->ndim < 2 ||
            (source->flags & (CNP_ARRAY_C_CONTIGUOUS |
                              CNP_ARRAY_F_CONTIGUOUS)))
        return result;

    int axes[CNP_MAXDIMS];
    for (int axis = 0; axis < source->ndim; ++axis)
        axes[axis] = axis;
    for (int index = 1; index < source->ndim; ++index) {
        int axis = axes[index];
        uint64_t magnitude = special_stride_magnitude(
            source->strides[axis]);
        int insertion = index;
        while (insertion > 0 && special_stride_magnitude(
                source->strides[axes[insertion - 1]]) > magnitude) {
            axes[insertion] = axes[insertion - 1];
            --insertion;
        }
        axes[insertion] = axis;
    }

    int64_t stride = result->dtype->elsize;
    for (int index = 0; index < result->ndim; ++index) {
        int axis = axes[index];
        result->strides[axis] = stride;
        stride *= result->shape[axis];
    }
    result->flags &= ~(CNP_ARRAY_C_CONTIGUOUS |
                       CNP_ARRAY_F_CONTIGUOUS);
    result->flags |= cnp_compute_layout_flags(
        result->ndim, result->shape, result->strides,
        result->dtype->elsize);
    return result;
}

static int64_t special_flat_offset(
    const CnpArray *source, int64_t flat_index) {
    int64_t coordinates[CNP_MAXDIMS] = {0};
    int64_t remaining = flat_index;
    for (int dimension = source->ndim - 1; dimension >= 0; --dimension) {
        coordinates[dimension] = remaining % source->shape[dimension];
        remaining /= source->shape[dimension];
    }
    return source->offset + cnp_multi_to_offset(
        source->ndim, coordinates, source->strides);
}

static double special_real_flat_get(
    const CnpArray *source, int64_t flat_index) {
    int64_t offset = special_flat_offset(source, flat_index);
    if (source->dtype->type_num == CNP_HALF) {
        return cnp_half_to_float(
            *(const uint16_t*)((const char*)source->data + offset));
    }
    return cnp_get_element_double(
        source->data, offset, source->dtype->type_num);
}

static void special_real_store(
    CnpArray *result, int64_t offset, double value) {
    if (result->dtype->type_num == CNP_HALF) {
        *(uint16_t*)((char*)result->data + offset) =
            cnp_float_to_half(value);
        return;
    }
    cnp_set_element_double(
        result->data, offset, result->dtype->type_num, value);
}

static CnpArray *special_real_unary_as(
    const CnpArray *source, CNP_TYPE result_dtype,
    CnpRealUnaryFunction operation, const char *function_name) {
    CnpArray *result = special_new_keep_order(
        source, result_dtype, function_name);
    if (!result) return NULL;
    for (int64_t index = 0; index < source->size; ++index) {
        int64_t result_offset = special_flat_offset(result, index);
        double value = operation(
            special_real_flat_get(source, index), result_dtype);
        special_real_store(result, result_offset, value);
    }
    return result;
}

static CnpArray *special_real_unary(
    const CnpArray *source, CnpRealUnaryFunction operation,
    const char *function_name) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (!special_real_dtype_supported(source, function_name))
        return NULL;
    return special_real_unary_as(
        source, special_real_result_dtype(source->dtype->type_num),
        operation, function_name);
}

static double special_erf_scalar(double value, CNP_TYPE result_dtype) {
    (void)result_dtype;
    return erf(value);
}

static double special_erfc_scalar(double value, CNP_TYPE result_dtype) {
    (void)result_dtype;
    return erfc(value);
}

static double special_expit_scalar(
    double value, CNP_TYPE result_dtype) {
    if (result_dtype == CNP_FLOAT) {
        float input = (float)value;
        return (double)(1.0f / (1.0f + expf(-input)));
    }
    return 1.0 / (1.0 + exp(-value));
}

static double special_logit_scalar(
    double value, CNP_TYPE result_dtype) {
    (void)result_dtype;
    if (isnan(value)) return value;
    if (value == 0.0) return -special_positive_infinity();
    if (value == 1.0) return special_positive_infinity();
    if (value < 0.0 || value > 1.0) return NAN;
    return log(value) - log1p(-value);
}

typedef struct {
    double real;
    double imag;
} CnpSpecialComplex;

static CnpSpecialComplex special_complex(double real, double imag) {
    CnpSpecialComplex value;
    value.real = real;
    value.imag = imag;
    return value;
}

static CnpSpecialComplex special_complex_add(
    CnpSpecialComplex left, CnpSpecialComplex right) {
    return special_complex(left.real + right.real, left.imag + right.imag);
}

static CnpSpecialComplex special_complex_subtract(
    CnpSpecialComplex left, CnpSpecialComplex right) {
    return special_complex(left.real - right.real, left.imag - right.imag);
}

static CnpSpecialComplex special_complex_scale(
    CnpSpecialComplex value, double factor) {
    return special_complex(value.real * factor, value.imag * factor);
}

static CnpSpecialComplex special_complex_multiply(
    CnpSpecialComplex left, CnpSpecialComplex right) {
    return special_complex(
        left.real * right.real - left.imag * right.imag,
        left.real * right.imag + left.imag * right.real);
}

static CnpSpecialComplex special_complex_divide(
    CnpSpecialComplex numerator, CnpSpecialComplex denominator) {
    double real_absolute = fabs(denominator.real);
    double imaginary_absolute = fabs(denominator.imag);
    if (real_absolute >= imaginary_absolute) {
        double ratio = denominator.imag / denominator.real;
        double scale = denominator.real + denominator.imag * ratio;
        return special_complex(
            (numerator.real + numerator.imag * ratio) / scale,
            (numerator.imag - numerator.real * ratio) / scale);
    }
    {
        double ratio = denominator.real / denominator.imag;
        double scale = denominator.real * ratio + denominator.imag;
        return special_complex(
            (numerator.real * ratio + numerator.imag) / scale,
            (numerator.imag * ratio - numerator.real) / scale);
    }
}

static CnpSpecialComplex special_complex_log(CnpSpecialComplex value) {
    return special_complex(
        log(hypot(value.real, value.imag)),
        atan2(value.imag, value.real));
}

static CnpSpecialComplex special_complex_exp(CnpSpecialComplex value) {
    double magnitude = exp(value.real);
    return special_complex(
        magnitude * cos(value.imag), magnitude * sin(value.imag));
}

static CnpSpecialComplex special_complex_power(
    CnpSpecialComplex base, CnpSpecialComplex exponent) {
    return special_complex_exp(
        special_complex_multiply(exponent, special_complex_log(base)));
}

static CnpSpecialComplex special_complex_sin(CnpSpecialComplex value) {
    return special_complex(
        sin(value.real) * cosh(value.imag),
        cos(value.real) * sinh(value.imag));
}

static CnpSpecialComplex special_complex_gamma(CnpSpecialComplex value) {
    static const double coefficients[9] = {
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    };
    CnpSpecialComplex shifted;
    CnpSpecialComplex series;
    CnpSpecialComplex scale;
    CnpSpecialComplex result;
    if (!isfinite(value.real) || !isfinite(value.imag))
        return special_complex(NAN, NAN);
    if (value.imag == 0.0 && value.real <= 0.0 &&
            value.real == floor(value.real))
        return special_complex(NAN, NAN);
    if (value.real < 0.5) {
        CnpSpecialComplex reflected = special_complex_gamma(
            special_complex_subtract(special_complex(1.0, 0.0), value));
        CnpSpecialComplex sine = special_complex_sin(
            special_complex_scale(value, M_PI));
        return special_complex_divide(
            special_complex(M_PI, 0.0),
            special_complex_multiply(sine, reflected));
    }

    shifted = special_complex_subtract(value, special_complex(1.0, 0.0));
    series = special_complex(coefficients[0], 0.0);
    for (int index = 1; index < 9; ++index) {
        series = special_complex_add(
            series,
            special_complex_divide(
                special_complex(coefficients[index], 0.0),
                special_complex_add(
                    shifted, special_complex((double)index, 0.0))));
    }
    scale = special_complex_add(shifted, special_complex(7.5, 0.0));
    result = special_complex_power(
        scale,
        special_complex_add(shifted, special_complex(0.5, 0.0)));
    result = special_complex_multiply(
        result, special_complex_exp(special_complex_scale(scale, -1.0)));
    result = special_complex_multiply(result, series);
    return special_complex_scale(result, 2.50662827463100050242);
}

static double special_polynomial(
    double value, const double *coefficients, int degree) {
    double result = coefficients[0];
    for (int index = 1; index <= degree; ++index)
        result = result * value + coefficients[index];
    return result;
}

static double special_polynomial_one_leading(
    double value, const double *coefficients, int coefficient_count) {
    double result = value + coefficients[0];
    for (int index = 1; index < coefficient_count; ++index)
        result = result * value + coefficients[index];
    return result;
}

/* Match the pairwise Horner grouping selected by Boost.Math under GCC. */
static double special_polynomial_pairwise(
    double value, const double *coefficients, int coefficient_count) {
    volatile double square = value * value;
    int even_index = (coefficient_count - 1) & ~1;
    int odd_index = ((coefficient_count - 1) & 1)
        ? coefficient_count - 1 : coefficient_count - 2;
    volatile double even = coefficients[even_index];
    volatile double odd = coefficients[odd_index];
    for (even_index -= 2; even_index >= 0; even_index -= 2)
        even = even * square + coefficients[even_index];
    for (odd_index -= 2; odd_index >= 1; odd_index -= 2)
        odd = odd * square + coefficients[odd_index];
    volatile double odd_product = odd * value;
    return even + odd_product;
}

/* Preserve the separately rounded leading and correction products. */
static double special_split_sum_product(
    double scale, double leading_coefficient, double correction) {
    volatile double leading = scale * leading_coefficient;
    volatile double tail = scale * correction;
    return leading + tail;
}

/* Cephes Gamma branch structure used by the pinned SciPy 1.12 oracle. */
static double special_stirling_gamma(double value) {
    static const double coefficients[5] = {
        7.87311395793093628397e-4,
        -2.29549961613378126380e-4,
        -2.68132617805781232825e-3,
        3.47222221605458667310e-3,
        8.33333333333482257126e-2
    };
    const double maximum_gamma = 171.624376956302725;
    const double split = 143.01608;
    double inverse;
    double correction;
    double exponential;
    double result;
    if (value >= maximum_gamma) return special_positive_infinity();
    inverse = 1.0 / value;
    correction = 1.0 + inverse * special_polynomial(
        inverse, coefficients, 4);
    exponential = exp(value);
    if (value > split) {
        double power = pow(value, 0.5 * value - 0.25);
        result = power * (power / exponential);
    } else {
        result = pow(value, value - 0.5) / exponential;
    }
    return 2.50662827463100050242 * result * correction;
}

static double special_gamma_real_scalar(
    double value, CNP_TYPE result_dtype) {
    static const double numerator_coefficients[7] = {
        1.60119522476751861407e-4,
        1.19135147006586384913e-3,
        1.04213797561761569935e-2,
        4.76367800457137231464e-2,
        2.07448227648435975150e-1,
        4.94214826801497100753e-1,
        9.99999999999999996796e-1
    };
    static const double denominator_coefficients[8] = {
        -2.31581873324120129819e-5,
        5.39605580493303397842e-4,
        -4.45641913851797240494e-3,
        1.18139785222060435552e-2,
        3.58236398605498653373e-2,
        -2.34591795718243348568e-1,
        7.14304917030273074085e-2,
        1.00000000000000000320
    };
    double absolute;
    double accumulator;
    (void)result_dtype;
    if (!isfinite(value)) return value;
    absolute = fabs(value);
    if (absolute > 33.0) {
        if (value < 0.0) {
            double integer_part = floor(absolute);
            double remainder;
            double denominator;
            double sign = 1.0;
            if (integer_part == absolute)
                return special_positive_infinity();
            if (fmod(integer_part, 2.0) == 0.0) sign = -1.0;
            remainder = absolute - integer_part;
            if (remainder > 0.5) {
                integer_part += 1.0;
                remainder = absolute - integer_part;
            }
            denominator = absolute * sin(M_PI * remainder);
            if (denominator == 0.0)
                return sign * special_positive_infinity();
            denominator = fabs(denominator);
            return sign * M_PI /
                (denominator * special_stirling_gamma(absolute));
        }
        return special_stirling_gamma(value);
    }

    accumulator = 1.0;
    while (value >= 3.0) {
        value -= 1.0;
        accumulator *= value;
    }
    while (value < 0.0) {
        if (value > -1e-9) {
            if (value == 0.0) return special_positive_infinity();
            return accumulator /
                ((1.0 + 0.5772156649015329 * value) * value);
        }
        accumulator /= value;
        value += 1.0;
    }
    while (value < 2.0) {
        if (value < 1e-9) {
            if (value == 0.0) return special_positive_infinity();
            return accumulator /
                ((1.0 + 0.5772156649015329 * value) * value);
        }
        accumulator /= value;
        value += 1.0;
    }
    if (value == 2.0) return accumulator;
    value -= 2.0;
    return accumulator *
        special_polynomial(value, numerator_coefficients, 6) /
        special_polynomial(value, denominator_coefficients, 7);
}

static double special_polynomial_leading_one(
    double value, const double *coefficients, int coefficient_count) {
    double result = value + coefficients[0];
    for (int index = 1; index < coefficient_count; ++index)
        result = result * value + coefficients[index];
    return result;
}

/* Cephes log-Gamma branch structure used by the pinned SciPy 1.12 oracle. */
static double special_lgamma_with_sign(double value, int *sign) {
    static const double large_coefficients[5] = {
        8.11614167470508450300e-4,
        -5.95061904284301438324e-4,
        7.93650340457716943945e-4,
        -2.77777777730099687205e-3,
        8.33333333333331927722e-2
    };
    static const double numerator_coefficients[6] = {
        -1.37825152569120859100e3,
        -3.88016315134637840924e4,
        -3.31612992738871184744e5,
        -1.16237097492762307383e6,
        -1.72173700820839662146e6,
        -8.53555664245765465627e5
    };
    static const double denominator_coefficients[6] = {
        -3.51815701436523470549e2,
        -1.70642106651881136949e4,
        -2.20528590553854454839e5,
        -1.13933444367982507207e6,
        -2.53252307177582951285e6,
        -2.01889141433532773231e6
    };
    const double maximum_log_gamma = 2.556348e305;
    double first;
    double second;
    double accumulator;
    double correction;
    double shifted;
    *sign = 1;
    if (!isfinite(value)) return value;

    if (value < -34.0) {
        double absolute = -value;
        double reflected = special_lgamma_with_sign(absolute, sign);
        double integer_part = floor(absolute);
        double remainder;
        if (integer_part == absolute)
            return special_positive_infinity();
        *sign = fmod(integer_part, 2.0) == 0.0 ? -1 : 1;
        remainder = absolute - integer_part;
        if (remainder > 0.5) {
            integer_part += 1.0;
            remainder = integer_part - absolute;
        }
        remainder = absolute * sin(M_PI * remainder);
        if (remainder == 0.0)
            return special_positive_infinity();
        return 1.14472988584940017414 - log(remainder) - reflected;
    }

    if (value < 13.0) {
        accumulator = 1.0;
        shifted = 0.0;
        first = value;
        while (first >= 3.0) {
            shifted -= 1.0;
            first = value + shifted;
            accumulator *= first;
        }
        while (first < 2.0) {
            if (first == 0.0)
                return special_positive_infinity();
            accumulator /= first;
            shifted += 1.0;
            first = value + shifted;
        }
        if (accumulator < 0.0) {
            *sign = -1;
            accumulator = -accumulator;
        } else {
            *sign = 1;
        }
        if (first == 2.0) return log(accumulator);
        shifted -= 2.0;
        value += shifted;
        correction = value * special_polynomial(
            value, numerator_coefficients, 5) /
            special_polynomial_leading_one(
                value, denominator_coefficients, 6);
        return log(accumulator) + correction;
    }

    if (value > maximum_log_gamma)
        return special_positive_infinity();
    first = (value - 0.5) * log(value) - value +
        0.91893853320467274178;
    if (value > 1e8) return first;
    second = 1.0 / (value * value);
    if (value >= 1000.0) {
        first += ((7.9365079365079365079365e-4 * second -
            2.7777777777777777777778e-3) * second +
            8.3333333333333333333333e-2) / value;
    } else {
        first += special_polynomial(second, large_coefficients, 4) / value;
    }
    return first;
}

static double special_gammaln_scalar(
    double value, CNP_TYPE result_dtype) {
    int sign;
    (void)result_dtype;
    return special_lgamma_with_sign(value, &sign);
}

static CnpArray *special_gamma_unary(const CnpArray *source) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_gamma", "source array is required");
        return NULL;
    }
    if (source->dtype->type_num != CNP_CFLOAT &&
            source->dtype->type_num != CNP_CDOUBLE) {
        return special_real_unary(
            source, special_gamma_real_scalar, "cnp_gamma");
    }

    CNP_TYPE result_dtype = source->dtype->type_num;
    CNP_ORDER result_order =
        (source->flags & CNP_ARRAY_F_CONTIGUOUS) &&
        !(source->flags & CNP_ARRAY_C_CONTIGUOUS)
        ? CNP_ORDER_F : CNP_ORDER_C;
    CnpArray *result = cnp_array_new(
        source->ndim, source->shape, result_dtype, result_order);
    if (!result) {
        cnp_relabel_error("cnp_gamma");
        return NULL;
    }
    for (int64_t index = 0; index < source->size; ++index) {
        int64_t offset = special_flat_offset(source, index);
        const char *pointer = (const char*)source->data + offset;
        CnpSpecialComplex input;
        CnpSpecialComplex output;
        if (source->dtype->type_num == CNP_CFLOAT) {
            const cnp_cfloat *value = (const cnp_cfloat*)pointer;
            input = special_complex(value->real, value->imag);
        } else {
            const cnp_cdouble *value = (const cnp_cdouble*)pointer;
            input = special_complex(value->real, value->imag);
        }
        output = special_complex_gamma(input);
        int64_t result_offset = special_flat_offset(result, index);
        char *result_pointer = (char*)result->data + result_offset;
        if (result_dtype == CNP_CFLOAT) {
            cnp_cfloat *target = (cnp_cfloat*)result_pointer;
            target->real = (float)output.real;
            target->imag = (float)output.imag;
        } else {
            cnp_cdouble *target = (cnp_cdouble*)result_pointer;
            target->real = output.real;
            target->imag = output.imag;
        }
    }
    return result;
}

/* =========================================================================
 * Internal: Gamma function using Lanczos approximation
 * ========================================================================= */
static double gamma_lanczos(double x) {
    static const double g = 7.0;
    static const double c[9] = {
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    };

    if (x < 0.5) {
        /* Reflection formula */
        return M_PI / (sin(M_PI * x) * gamma_lanczos(1.0 - x));
    }

    x -= 1.0;
    double a = c[0];
    double t = x + g + 0.5;
    for (int i = 1; i < 9; i++) {
        a += c[i] / (x + i);
    }

    return sqrt(2.0 * M_PI) * pow(t, x + 0.5) * exp(-t) * a;
}

/* =========================================================================
 * gamma - Gamma function
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_gamma(const CnpArray *x) {
    return special_gamma_unary(x);
}

/* =========================================================================
 * gammaln - Log of absolute value of gamma function
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_gammaln(const CnpArray *x) {
    return special_real_unary(x, special_gammaln_scalar, "cnp_gammaln");
}

/* =========================================================================
 * beta - Beta function: B(a,b) = gamma(a)*gamma(b)/gamma(a+b)
 * ========================================================================= */
static double special_lgamma_sign(double value, int *sign) {
    return special_lgamma_with_sign(value, sign);
}

static double special_beta_scalar(double left, double right);

static double special_beta_negative_integer(
    double integer_argument, double other_argument) {
    int integer_other;
    if (integer_argument < INT_MIN || integer_argument > INT_MAX ||
            other_argument < INT_MIN || other_argument > INT_MAX ||
            other_argument != floor(other_argument))
        return special_positive_infinity();
    integer_other = (int)other_argument;
    if (1.0 - integer_argument - other_argument <= 0.0)
        return special_positive_infinity();
    return (integer_other % 2 == 0 ? 1.0 : -1.0) *
        special_beta_scalar(
            1.0 - integer_argument - other_argument, other_argument);
}

static double special_beta_asymptotic(
    double large_argument, double other_argument, int *sign) {
    double result = special_lgamma_sign(other_argument, sign);
    double inverse = 1.0 / large_argument;
    result -= other_argument * log(large_argument);
    result += other_argument * (1.0 - other_argument) * inverse / 2.0;
    result += other_argument * (1.0 - other_argument) *
        (1.0 - 2.0 * other_argument) * inverse * inverse / 12.0;
    result -= other_argument * other_argument *
        (1.0 - other_argument) * (1.0 - other_argument) *
        inverse * inverse * inverse / 12.0;
    return result;
}

static double special_beta_scalar(double left, double right) {
    const double maximum_direct_gamma = 171.624376956302725;
    const double asymptotic_factor = 1e6;
    double sum;
    double gamma_left;
    double gamma_right;
    double gamma_sum;
    int sign = 1;
    int gamma_sign;

    if (isnan(left) || isnan(right)) return NAN;
    if (left <= 0.0 && left == floor(left))
        return special_beta_negative_integer(left, right);
    if (right <= 0.0 && right == floor(right))
        return special_beta_negative_integer(right, left);
    if (fabs(left) < fabs(right)) {
        double swap = left;
        left = right;
        right = swap;
    }
    if (fabs(left) > asymptotic_factor * fabs(right) &&
            left > asymptotic_factor) {
        double logarithm = special_beta_asymptotic(left, right, &sign);
        return sign * exp(logarithm);
    }

    sum = left + right;
    if (fabs(sum) > maximum_direct_gamma ||
            fabs(left) > maximum_direct_gamma ||
            fabs(right) > maximum_direct_gamma) {
        double logarithm = special_lgamma_sign(sum, &gamma_sign);
        sign *= gamma_sign;
        logarithm = special_lgamma_sign(right, &gamma_sign) - logarithm;
        sign *= gamma_sign;
        logarithm = special_lgamma_sign(left, &gamma_sign) + logarithm;
        sign *= gamma_sign;
        if (logarithm > log(DBL_MAX))
            return sign * special_positive_infinity();
        return sign * exp(logarithm);
    }

    gamma_sum = special_gamma_real_scalar(sum, CNP_DOUBLE);
    gamma_left = special_gamma_real_scalar(left, CNP_DOUBLE);
    gamma_right = special_gamma_real_scalar(right, CNP_DOUBLE);
    if (gamma_sum == 0.0)
        return special_positive_infinity();
    if (fabs(fabs(gamma_left) - fabs(gamma_sum)) >
            fabs(fabs(gamma_right) - fabs(gamma_sum))) {
        return (gamma_right / gamma_sum) * gamma_left;
    }
    return (gamma_left / gamma_sum) * gamma_right;
}

static int64_t special_broadcast_offset(
    const CnpArray *source, const int64_t *coordinates, int output_ndim) {
    int dimension_offset = output_ndim - source->ndim;
    int64_t byte_offset = source->offset;
    for (int dimension = 0; dimension < source->ndim; ++dimension) {
        int64_t coordinate = coordinates[dimension_offset + dimension];
        if (source->shape[dimension] == 1) coordinate = 0;
        byte_offset += coordinate * source->strides[dimension];
    }
    return byte_offset;
}

static double special_broadcast_real_get(
    const CnpArray *source, const int64_t *coordinates, int output_ndim) {
    int64_t byte_offset = special_broadcast_offset(
        source, coordinates, output_ndim);
    if (source->dtype->type_num == CNP_HALF) {
        return cnp_half_to_float(
            *(const uint16_t*)((const char*)source->data + byte_offset));
    }
    return cnp_get_element_double(
        source->data, byte_offset, source->dtype->type_num);
}

CNP_API CnpArray* CNP_CALL cnp_beta(const CnpArray *a, const CnpArray *b) {
    if (!a || !b) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_beta", "input arrays are required");
        return NULL;
    }
    if (!special_real_dtype_supported(a, "cnp_beta") ||
            !special_real_dtype_supported(b, "cnp_beta"))
        return NULL;

    int out_ndim;
    int64_t *out_shape;
    if (cnp_broadcast_shapes(2, (const int64_t*[]){a->shape, b->shape},
                              (const int[]){a->ndim, b->ndim}, &out_ndim, &out_shape) != CNP_OK) {
        cnp_relabel_error("cnp_beta");
        return NULL;
    }
    CNP_TYPE result_dtype =
        special_real_result_dtype(a->dtype->type_num) == CNP_FLOAT &&
        special_real_result_dtype(b->dtype->type_num) == CNP_FLOAT
        ? CNP_FLOAT : CNP_DOUBLE;
    CnpArray *result = cnp_array_new(
        out_ndim, out_shape, result_dtype, CNP_ORDER_C);
    cnp_free(out_shape, out_ndim * sizeof(int64_t));
    if (!result) {
        cnp_relabel_error("cnp_beta");
        return NULL;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        double left = special_broadcast_real_get(a, coordinates, out_ndim);
        double right = special_broadcast_real_get(b, coordinates, out_ndim);
        cnp_set_element_double(
            result->data, index * result->dtype->elsize,
            result_dtype, special_beta_scalar(left, right));
        for (int dimension = out_ndim - 1; dimension >= 0; --dimension) {
            coordinates[dimension]++;
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

/* =========================================================================
 * erf - Error function
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_erf(const CnpArray *x) {
    return special_real_unary(x, special_erf_scalar, "cnp_erf");
}

/* =========================================================================
 * erfc - Complementary error function: 1 - erf(x)
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_erfc(const CnpArray *x) {
    return special_real_unary(x, special_erfc_scalar, "cnp_erfc");
}

/* =========================================================================
 * erfinv - Inverse error function
 *
 * Rational branch structure and coefficients follow the Boost.Math source
 * pinned by SciPy 1.12.0 (Copyright John Maddock 2006), distributed under
 * the Boost Software License, Version 1.0.
 * ========================================================================= */
static double special_erfinv_scalar(
    double x, CNP_TYPE result_dtype) {
    (void)result_dtype;
    if (special_is_nan(x)) return x;
    if (x < -1.0 || x > 1.0) return NAN;
    if (x == -1.0) return -special_positive_infinity();
    if (x == 1.0) return special_positive_infinity();
    if (x == 0.0) return 0.0;

    double magnitude = fabs(x);
    if (magnitude <= 0.5) {
        static const double numerator[8] = {
            -0.000508781949658280665617,
            -0.00836874819741736770379,
            0.0334806625409744615033,
            -0.0126926147662974029034,
            -0.0365637971411762664006,
            0.0219878681111168899165,
            0.00822687874676915743155,
            -0.00538772965071242932965
        };
        static const double denominator[10] = {
            1.0,
            -0.970005043303290640362,
            -1.56574558234175846809,
            1.56221558398423026363,
            0.662328840472002992063,
            -0.71228902341542847553,
            -0.0527396382340099713954,
            0.0795283687341571680018,
            -0.00233393759374190016776,
            0.000886216390456424707504
        };
        double factor = magnitude * (magnitude + 10.0);
        double correction = special_polynomial_pairwise(
            magnitude, numerator, 8) /
            special_polynomial_pairwise(magnitude, denominator, 10);
        return copysign(special_split_sum_product(
            factor, 0.0891314744949340820313f, correction), x);
    }

    double complement = 1.0 - magnitude;
    if (complement >= 0.25) {
        static const double numerator[9] = {
            -0.202433508355938759655,
            0.105264680699391713268,
            8.37050328343119927838,
            17.6447298408374015486,
            -18.8510648058714251895,
            -44.6382324441786960818,
            17.445385985570866523,
            21.1294655448340526258,
            -3.67192254707729348546
        };
        static const double denominator[9] = {
            1.0,
            6.24264124854247537712,
            3.9713437953343869095,
            -28.6608180499800029974,
            -20.1432634680485188801,
            48.5609213108739935468,
            10.8268667355460159008,
            -22.6436933413139721736,
            1.72114765761200282724
        };
        double root = sqrt(-2.0 * log(complement));
        double shifted = complement - 0.25f;
        double correction = special_polynomial_pairwise(
            shifted, numerator, 9) /
            special_polynomial_pairwise(shifted, denominator, 9);
        return copysign(
            root / (2.249481201171875f + correction), x);
    }

    double root = sqrt(-log(complement));
    if (root < 3.0) {
        static const double numerator[11] = {
            -0.131102781679951906451,
            -0.163794047193317060787,
            0.117030156341995252019,
            0.387079738972604337464,
            0.337785538912035898924,
            0.142869534408157156766,
            0.0290157910005329060432,
            0.00214558995388805277169,
            -0.679465575181126350155e-6,
            0.285225331782217055858e-7,
            -0.681149956853776992068e-9
        };
        static const double denominator[8] = {
            1.0,
            3.46625407242567245975,
            5.38168345707006855425,
            4.77846592945843778382,
            2.59301921623620271374,
            0.848854343457902036425,
            0.152264338295331783612,
            0.01105924229346489121
        };
        double shifted = root - 1.125f;
        double correction = special_polynomial_pairwise(
            shifted, numerator, 11) /
            special_polynomial_pairwise(shifted, denominator, 8);
        return copysign(special_split_sum_product(
            root, 0.807220458984375f, correction), x);
    }
    if (root < 6.0) {
        static const double numerator[9] = {
            -0.0350353787183177984712,
            -0.00222426529213447927281,
            0.0185573306514231072324,
            0.00950804701325919603619,
            0.00187123492819559223345,
            0.000157544617424960554631,
            0.460469890584317994083e-5,
            -0.230404776911882601748e-9,
            0.266339227425782031962e-11
        };
        static const double denominator[7] = {
            1.0,
            1.3653349817554063097,
            0.762059164553623404043,
            0.220091105764131249824,
            0.0341589143670947727934,
            0.00263861676657015992959,
            0.764675292302794483503e-4
        };
        double shifted = root - 3.0;
        double correction = special_polynomial_pairwise(
            shifted, numerator, 9) /
            special_polynomial_pairwise(shifted, denominator, 7);
        return copysign(special_split_sum_product(
            root, 0.93995571136474609375f, correction), x);
    }
    {
        static const double numerator[9] = {
            -0.0167431005076633737133,
            -0.00112951438745580278863,
            0.00105628862152492910091,
            0.000209386317487588078668,
            0.149624783758342370182e-4,
            0.449696789927706453732e-6,
            0.462596163522878599135e-8,
            -0.281128735628831791805e-13,
            0.99055709973310326855e-16
        };
        static const double denominator[7] = {
            1.0,
            0.591429344886417493481,
            0.138151865749083321638,
            0.0160746087093676504695,
            0.000964011807005165528527,
            0.275335474764726041141e-4,
            0.282243172016108031869e-6
        };
        double shifted = root - 6.0;
        double correction = special_polynomial_pairwise(
            shifted, numerator, 9) /
            special_polynomial_pairwise(shifted, denominator, 7);
        return copysign(special_split_sum_product(
            root, 0.98362827301025390625f, correction), x);
    }
}

CNP_API CnpArray* CNP_CALL cnp_erfinv(const CnpArray *x) {
    return special_real_unary(
        x, special_erfinv_scalar, "cnp_erfinv");
}

/* =========================================================================
 * factorial - Factorial function (returns double for large values)
 * ========================================================================= */
static double special_factorial_approx_scalar(
    double value, CNP_TYPE result_dtype) {
    if (special_is_nan(value)) return value;
    if (value < 0.0) return 0.0;
    return special_gamma_real_scalar(value + 1.0, result_dtype);
}

static bool special_integer_offset_get(
    const CnpArray *source, int64_t offset, int64_t *value) {
    if (source->dtype->type_num == CNP_ULONG ||
            source->dtype->type_num == CNP_ULONGLONG) {
        uint64_t unsigned_value;
        memcpy(
            &unsigned_value,
            (const char*)source->data + offset,
            sizeof(unsigned_value));
        if (unsigned_value > (uint64_t)INT64_MAX) return false;
        *value = (int64_t)unsigned_value;
        return true;
    }
    *value = cnp_get_element_int(
        source->data, offset, source->dtype->type_num);
    return true;
}

static bool special_integer_flat_get(
    const CnpArray *source, int64_t flat_index, int64_t *value) {
    int64_t offset = special_flat_offset(source, flat_index);
    return special_integer_offset_get(source, offset, value);
}

static bool special_broadcast_integer_get(
    const CnpArray *source, const int64_t *coordinates,
    int output_ndim, int64_t *value) {
    return special_integer_offset_get(
        source,
        special_broadcast_offset(source, coordinates, output_ndim),
        value);
}

static int64_t special_greatest_common_divisor(
    int64_t left, int64_t right) {
    while (right != 0) {
        int64_t remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

static bool special_exact_comb(
    int64_t n, int64_t k, int64_t *value) {
    if (n < 0 || k < 0 || k > n) {
        *value = 0;
        return true;
    }
    if (k > n - k) k = n - k;
    int64_t result = 1;
    for (int64_t index = 1; index <= k; ++index) {
        int64_t numerator = n - k + index;
        int64_t denominator = index;
        int64_t divisor = special_greatest_common_divisor(
            result, denominator);
        result /= divisor;
        denominator /= divisor;
        divisor = special_greatest_common_divisor(
            numerator, denominator);
        numerator /= divisor;
        denominator /= divisor;
        if (numerator != 0 && result > INT64_MAX / numerator)
            return false;
        result *= numerator;
        result /= denominator;
    }
    *value = result;
    return true;
}

static bool special_exact_perm(
    int64_t n, int64_t k, int64_t *value) {
    if (n < 0 || k < 0 || k > n) {
        *value = 0;
        return true;
    }
    int64_t result = 1;
    for (int64_t index = 0; index < k; ++index) {
        int64_t factor = n - k + 1 + index;
        if (factor != 0 && result > INT64_MAX / factor)
            return false;
        result *= factor;
    }
    *value = result;
    return true;
}

CNP_API CnpArray* CNP_CALL cnp_factorial(const CnpArray *x, bool exact) {
    if (!x) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_factorial", "source array is required");
        return NULL;
    }
    if (x->size == 0) {
        CnpArray *result = cnp_array_copy(x);
        if (!result) cnp_relabel_error("cnp_factorial");
        return result;
    }
    if (x->dtype->kind != 'i' && x->dtype->kind != 'u' &&
            x->dtype->kind != 'f') {
        cnp_set_error(
            CNP_ERR_TYPE, "cnp_factorial",
            "source array must have an integer or floating-point dtype");
        return NULL;
    }
    if (!exact) {
        return special_real_unary(
            x, special_factorial_approx_scalar, "cnp_factorial");
    }

    if (x->dtype->kind == 'f') {
        cnp_set_error(
            CNP_ERR_TYPE, "cnp_factorial",
            "exact factorial requires an integer dtype");
        return NULL;
    }
    int64_t largest = 0;
    for (int64_t index = 0; index < x->size; ++index) {
        int64_t value;
        if (!special_integer_flat_get(x, index, &value) || value > 20) {
            cnp_set_error(
                CNP_ERR_CONVERSION, "cnp_factorial",
                "exact factorial result exceeds int64 representation");
            return NULL;
        }
        if (value > largest) largest = value;
    }

    CNP_TYPE result_dtype = largest > 12 ? CNP_LONGLONG : CNP_INT;
    CnpArray *result = special_new_keep_order(
        x, result_dtype, "cnp_factorial");
    if (!result) return NULL;
    for (int64_t index = 0; index < x->size; ++index) {
        int64_t value;
        int64_t factorial = 1;
        special_integer_flat_get(x, index, &value);
        if (value < 0) {
            factorial = 0;
        } else {
            for (int64_t factor = 2; factor <= value; ++factor)
                factorial *= factor;
        }
        cnp_set_element_int(
            result->data, special_flat_offset(result, index),
            result_dtype, factorial);
    }
    return result;
}

/* =========================================================================
 * comb - Number of combinations: C(n, k)
 * ========================================================================= */
static double special_comb_approx_scalar(double n, double k) {
    if (special_is_nan(n) || special_is_nan(k)) return 0.0;
    if (n < 0.0 || k < 0.0 || k > n) return 0.0;
    if (k == 0.0) return 1.0;
    if (special_is_infinite(n)) {
        if (special_is_infinite(k)) return NAN;
        return special_positive_infinity();
    }
    if (special_is_infinite(k)) return 0.0;
    if (k == n) return 1.0;

    int sign;
    double logarithm =
        special_lgamma_with_sign(n + 1.0, &sign) -
        special_lgamma_with_sign(k + 1.0, &sign) -
        special_lgamma_with_sign(n - k + 1.0, &sign);
    return exp(logarithm);
}

CNP_API CnpArray* CNP_CALL cnp_comb(const CnpArray *n, const CnpArray *k, bool exact) {
    if (!n || !k) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_comb", "input arrays are required");
        return NULL;
    }
    if (!special_real_dtype_supported(n, "cnp_comb") ||
            !special_real_dtype_supported(k, "cnp_comb"))
        return NULL;
    if (exact && (n->dtype->kind == 'f' || k->dtype->kind == 'f')) {
        cnp_set_error(
            CNP_ERR_TYPE, "cnp_comb",
            "exact combinations require integer dtypes");
        return NULL;
    }

    int out_ndim;
    int64_t *out_shape;
    if (cnp_broadcast_shapes(2, (const int64_t*[]){n->shape, k->shape},
                             (const int[]){n->ndim, k->ndim}, &out_ndim, &out_shape) != CNP_OK) {
        cnp_relabel_error("cnp_comb");
        return NULL;
    }

    CNP_TYPE result_dtype = exact ? CNP_LONGLONG :
        (special_real_result_dtype(n->dtype->type_num) == CNP_FLOAT &&
         special_real_result_dtype(k->dtype->type_num) == CNP_FLOAT
         ? CNP_FLOAT : CNP_DOUBLE);
    CnpArray *result = cnp_array_new(
        out_ndim, out_shape, result_dtype, CNP_ORDER_C);
    cnp_free(out_shape, out_ndim * sizeof(int64_t));
    if (!result) {
        cnp_relabel_error("cnp_comb");
        return NULL;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        if (exact) {
            int64_t left;
            int64_t right;
            int64_t combination;
            if (!special_broadcast_integer_get(
                    n, coordinates, out_ndim, &left) ||
                    !special_broadcast_integer_get(
                    k, coordinates, out_ndim, &right)) {
                cnp_array_free(result);
                cnp_set_error(
                    CNP_ERR_CONVERSION, "cnp_comb",
                    "exact combination inputs exceed int64 representation");
                return NULL;
            }
            if (!special_exact_comb(left, right, &combination)) {
                cnp_array_free(result);
                cnp_set_error(
                    CNP_ERR_CONVERSION, "cnp_comb",
                    "exact combination result exceeds int64 representation");
                return NULL;
            }
            cnp_set_element_int(
                result->data, index * result->dtype->elsize,
                result_dtype, combination);
        } else {
            double left = special_broadcast_real_get(
                n, coordinates, out_ndim);
            double right = special_broadcast_real_get(
                k, coordinates, out_ndim);
            cnp_set_element_double(
                result->data, index * result->dtype->elsize,
                result_dtype, special_comb_approx_scalar(left, right));
        }
        for (int dimension = out_ndim - 1; dimension >= 0; --dimension) {
            coordinates[dimension]++;
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

/* =========================================================================
 * perm - Number of permutations: P(n, k)
 * ========================================================================= */
static double special_perm_approx_scalar(double n, double k) {
    if (special_is_nan(n) || special_is_nan(k)) return 0.0;
    if (n < 0.0 || k < 0.0 || k > n) return 0.0;
    if (k == 0.0) return 1.0;
    if (special_is_infinite(n)) {
        if (special_is_infinite(k)) return NAN;
        return special_positive_infinity();
    }
    if (special_is_infinite(k)) return 0.0;

    int sign;
    double logarithm =
        special_lgamma_with_sign(n + 1.0, &sign) -
        special_lgamma_with_sign(n - k + 1.0, &sign);
    return exp(logarithm);
}

CNP_API CnpArray* CNP_CALL cnp_perm(const CnpArray *n, const CnpArray *k, bool exact) {
    if (!n || !k) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_perm", "input arrays are required");
        return NULL;
    }
    if (!special_real_dtype_supported(n, "cnp_perm") ||
            !special_real_dtype_supported(k, "cnp_perm"))
        return NULL;
    if (exact && (n->dtype->kind == 'f' || k->dtype->kind == 'f')) {
        cnp_set_error(
            CNP_ERR_TYPE, "cnp_perm",
            "exact permutations require integer dtypes");
        return NULL;
    }

    int out_ndim;
    int64_t *out_shape;
    if (cnp_broadcast_shapes(2, (const int64_t*[]){n->shape, k->shape},
                             (const int[]){n->ndim, k->ndim}, &out_ndim, &out_shape) != CNP_OK) {
        cnp_relabel_error("cnp_perm");
        return NULL;
    }

    CNP_TYPE difference_dtype = cnp_promote_type(
        n->dtype->type_num, k->dtype->type_num);
    CNP_TYPE result_dtype = exact ? CNP_LONGLONG :
        (special_real_result_dtype(difference_dtype) == CNP_FLOAT &&
         special_real_result_dtype(k->dtype->type_num) == CNP_FLOAT
         ? CNP_FLOAT : CNP_DOUBLE);
    CnpArray *result = cnp_array_new(
        out_ndim, out_shape, result_dtype, CNP_ORDER_C);
    cnp_free(out_shape, out_ndim * sizeof(int64_t));
    if (!result) {
        cnp_relabel_error("cnp_perm");
        return NULL;
    }

    int64_t coordinates[CNP_MAXDIMS] = {0};
    for (int64_t index = 0; index < result->size; ++index) {
        if (exact) {
            int64_t left;
            int64_t right;
            int64_t permutation;
            if (!special_broadcast_integer_get(
                    n, coordinates, out_ndim, &left) ||
                    !special_broadcast_integer_get(
                        k, coordinates, out_ndim, &right)) {
                cnp_array_free(result);
                cnp_set_error(
                    CNP_ERR_CONVERSION, "cnp_perm",
                    "exact permutation inputs exceed int64 representation");
                return NULL;
            }
            if (!special_exact_perm(left, right, &permutation)) {
                cnp_array_free(result);
                cnp_set_error(
                    CNP_ERR_CONVERSION, "cnp_perm",
                    "exact permutation result exceeds int64 representation");
                return NULL;
            }
            cnp_set_element_int(
                result->data, index * result->dtype->elsize,
                result_dtype, permutation);
        } else {
            double left = special_broadcast_real_get(
                n, coordinates, out_ndim);
            double right = special_broadcast_real_get(
                k, coordinates, out_ndim);
            cnp_set_element_double(
                result->data, index * result->dtype->elsize,
                result_dtype, special_perm_approx_scalar(left, right));
        }
        for (int dimension = out_ndim - 1; dimension >= 0; --dimension) {
            coordinates[dimension]++;
            if (coordinates[dimension] < result->shape[dimension]) break;
            coordinates[dimension] = 0;
        }
    }
    return result;
}

/* Cephes/Clenshaw coefficients used by numpy.i0 in NumPy 1.25.0. */
static const double numpy_i0_a[30] = {
    -4.4153416464793395e-18,
    3.3307945188222384e-17,
    -2.431279846547955e-16,
    1.715391285555133e-15,
    -1.1685332877993451e-14,
    7.676185498604936e-14,
    -4.856446783111929e-13,
    2.95505266312964e-12,
    -1.726826291441556e-11,
    9.675809035373237e-11,
    -5.189795601635263e-10,
    2.6598237246823866e-09,
    -1.300025009986248e-08,
    6.046995022541919e-08,
    -2.670793853940612e-07,
    1.1173875391201037e-06,
    -4.4167383584587505e-06,
    1.6448448070728896e-05,
    -5.754195010082104e-05,
    1.8850288509584165e-04,
    -5.763755745385824e-04,
    1.6394756169413357e-03,
    -4.324309995050576e-03,
    1.0546460394594998e-02,
    -2.373741480589947e-02,
    4.930528423967071e-02,
    -9.490109704804764e-02,
    1.7162090152220877e-01,
    -3.046826723431984e-01,
    6.767952744094761e-01
};

static const double numpy_i0_b[25] = {
    -7.233180487874754e-18,
    -4.830504485944182e-18,
    4.46562142029676e-17,
    3.461222867697461e-17,
    -2.8276239805165836e-16,
    -3.425485619677219e-16,
    1.7725601330565263e-15,
    3.8116806693526224e-15,
    -9.554846698828307e-15,
    -4.150569347287222e-14,
    1.54008621752141e-14,
    3.8527783827421426e-13,
    7.180124451383666e-13,
    -1.7941785315068062e-12,
    -1.3215811840447713e-11,
    -3.1499165279632416e-11,
    1.1889147107846439e-11,
    4.94060238822497e-10,
    3.3962320257083865e-09,
    2.266668990498178e-08,
    2.0489185894690638e-07,
    2.8913705208347567e-06,
    6.889758346916825e-05,
    3.3691164782556943e-03,
    8.044904110141088e-01
};

#if defined(_MSC_VER)
#pragma float_control(precise, on, push)
#endif

static double special_i0_chbevl_double(
    double x, const double *coefficients, size_t count) {
    double b0 = coefficients[0];
    double b1 = 0.0;
    double b2 = 0.0;
    for (size_t index = 1; index < count; ++index) {
        b2 = b1;
        b1 = b0;
        b0 = x * b1;
        b0 = b0 - b2;
        b0 = b0 + coefficients[index];
    }
    b0 = b0 - b2;
    return 0.5 * b0;
}

static float special_i0_chbevl_float(
    float x, const double *coefficients, size_t count) {
    float b0 = (float)coefficients[0];
    float b1 = 0.0f;
    float b2 = 0.0f;
    for (size_t index = 1; index < count; ++index) {
        b2 = b1;
        b1 = b0;
        b0 = x * b1;
        b0 = b0 - b2;
        b0 = b0 + (float)coefficients[index];
    }
    b0 = b0 - b2;
    return 0.5f * b0;
}

static double special_i0_half_round(double value) {
    return cnp_half_to_float(cnp_float_to_half(value));
}

static double special_i0_chbevl_half(
    double x, const double *coefficients, size_t count) {
    double b0 = coefficients[0];
    double b1 = 0.0;
    double b2 = 0.0;
    for (size_t index = 1; index < count; ++index) {
        b2 = b1;
        b1 = b0;
        b0 = special_i0_half_round(
            x * special_i0_half_round(b1));
        b0 = special_i0_half_round(b0 - b2);
        b0 = special_i0_half_round(
            b0 + special_i0_half_round(coefficients[index]));
    }
    b0 = special_i0_half_round(b0 - b2);
    return special_i0_half_round(0.5 * b0);
}

static double special_i0_double_scalar(
    double value, CNP_TYPE result_dtype) {
    (void)result_dtype;
    double x = fabs(value);
    if (!isfinite(x)) return NAN;
    if (x <= 8.0) {
        double reduced = x / 2.0;
        reduced = reduced - 2.0;
        return exp(x) * special_i0_chbevl_double(
            reduced, numpy_i0_a, 30);
    }
    {
        double reduced = 32.0 / x;
        reduced = reduced - 2.0;
        double numerator = exp(x) * special_i0_chbevl_double(
            reduced, numpy_i0_b, 25);
        return numerator / sqrt(x);
    }
}

static double special_i0_float_scalar(
    double value, CNP_TYPE result_dtype) {
    (void)result_dtype;
    float x = fabsf((float)value);
    if (!isfinite(x)) return NAN;
    if (x <= 8.0f) {
        float reduced = x / 2.0f;
        reduced = reduced - 2.0f;
        float result = expf(x) * special_i0_chbevl_float(
            reduced, numpy_i0_a, 30);
        return (double)result;
    }
    {
        float reduced = 32.0f / x;
        reduced = reduced - 2.0f;
        float numerator = expf(x) * special_i0_chbevl_float(
            reduced, numpy_i0_b, 25);
        float result = numerator / sqrtf(x);
        return (double)result;
    }
}

static double special_i0_half_scalar(
    double value, CNP_TYPE result_dtype) {
    (void)result_dtype;
    double x = special_i0_half_round(fabs(value));
    if (!isfinite(x)) return NAN;
    if (x <= 8.0) {
        double reduced = special_i0_half_round(x / 2.0);
        reduced = special_i0_half_round(reduced - 2.0);
        double exponential = special_i0_half_round(
            (double)expf((float)x));
        double polynomial = special_i0_chbevl_half(
            reduced, numpy_i0_a, 30);
        return special_i0_half_round(exponential * polynomial);
    }
    {
        double reduced = special_i0_half_round(32.0 / x);
        reduced = special_i0_half_round(reduced - 2.0);
        double exponential = special_i0_half_round(
            (double)expf((float)x));
        double polynomial = special_i0_chbevl_half(
            reduced, numpy_i0_b, 25);
        double numerator = special_i0_half_round(
            exponential * polynomial);
        double denominator = special_i0_half_round(
            (double)sqrtf((float)x));
        return special_i0_half_round(numerator / denominator);
    }
}

#if defined(_MSC_VER)
#pragma float_control(pop)
#endif

/* Cephes Bessel kernels used by the pinned SciPy 1.12 reference. */
static double bessel_j0(double x, CNP_TYPE result_dtype) {
    static const double numerator_small[4] = {
        -4.79443220978201773821E9,
        1.95617491946556577543E12,
        -2.49248344360967716204E14,
        9.70862251047306323952E15
    };
    static const double denominator_small[8] = {
        4.99563147152651017219E2,
        1.73785401676374683123E5,
        4.84409658339962045305E7,
        1.11855537045356834862E10,
        2.11277520115489217587E12,
        3.10518229857422583814E14,
        3.18121955943204943306E16,
        1.71086294081043136091E18
    };
    static const double numerator_p[7] = {
        7.96936729297347051624E-4,
        8.28352392107440799803E-2,
        1.23953371646414299388E0,
        5.44725003058768775090E0,
        8.74716500199817011941E0,
        5.30324038235394892183E0,
        9.99999999999999997821E-1
    };
    static const double denominator_p[7] = {
        9.24408810558863637013E-4,
        8.56288474354474431428E-2,
        1.25352743901058953537E0,
        5.47097740330417105182E0,
        8.76190883237069594232E0,
        5.30605288235394617618E0,
        1.00000000000000000218E0
    };
    static const double numerator_q[8] = {
        -1.13663838898469149931E-2,
        -1.28252718670509318512E0,
        -1.95539544257735972385E1,
        -9.32060152123768231369E1,
        -1.77681167980488050595E2,
        -1.47077505154951170175E2,
        -5.14105326766599330220E1,
        -6.05014350600728481186E0
    };
    static const double denominator_q[7] = {
        6.43178256118178023184E1,
        8.56430025976980587198E2,
        3.88240183605401609683E3,
        7.24046774195652478189E3,
        5.93072701187316984827E3,
        2.06209331660327847417E3,
        2.42005740240291393179E2
    };
    const double first_zero_squared = 5.78318596294678452118E0;
    const double second_zero_squared = 3.04712623436620863991E1;
    const double square_root_two_over_pi =
        7.97884560802865355880E-1;
    (void)result_dtype;
    if (x < 0.0) x = -x;
    if (x <= 5.0) {
        double square = x * x;
        if (x < 1.0E-5) return 1.0 - square / 4.0;
        double root_product =
            (square - first_zero_squared) *
            (square - second_zero_squared);
        return root_product * special_polynomial(
            square, numerator_small, 3) /
            special_polynomial_one_leading(
                square, denominator_small, 8);
    }

    double inverse = 5.0 / x;
    double square = 25.0 / (x * x);
    double p = special_polynomial(square, numerator_p, 6) /
        special_polynomial(square, denominator_p, 6);
    double q = special_polynomial(square, numerator_q, 7) /
        special_polynomial_one_leading(
            square, denominator_q, 7);
    volatile double phase = x - 7.85398163397448309616E-1;
    return (p * cos(phase) - inverse * q * sin(phase)) *
        square_root_two_over_pi / sqrt(x);
}

static double bessel_j1(double x, CNP_TYPE result_dtype) {
    static const double numerator_small[4] = {
        -8.99971225705559398224E8,
        4.52228297998194034323E11,
        -7.27494245221818276015E13,
        3.68295732863852883286E15
    };
    static const double denominator_small[8] = {
        6.20836478118054335476E2,
        2.56987256757748830383E5,
        8.35146791431949253037E7,
        2.21511595479792499675E10,
        4.74914122079991414898E12,
        7.84369607876235854894E14,
        8.95222336184627338078E16,
        5.32278620332680085395E18
    };
    static const double numerator_p[7] = {
        7.62125616208173112003E-4,
        7.31397056940917570436E-2,
        1.12719608129684925192E0,
        5.11207951146807644818E0,
        8.42404590141772420927E0,
        5.21451598682361504063E0,
        1.00000000000000000254E0
    };
    static const double denominator_p[7] = {
        5.71323128072548699714E-4,
        6.88455908754495404082E-2,
        1.10514232634061696926E0,
        5.07386386128601488557E0,
        8.39985554327604159757E0,
        5.20982848682361821619E0,
        9.99999999999999997461E-1
    };
    static const double numerator_q[8] = {
        5.10862594750176621635E-2,
        4.98213872951233449420E0,
        7.58238284132545283818E1,
        3.66779609360150777800E2,
        7.10856304998926107277E2,
        5.97489612400613639965E2,
        2.11688757100572135698E2,
        2.52070205858023719784E1
    };
    static const double denominator_q[7] = {
        7.42373277035675149943E1,
        1.05644886038262816351E3,
        4.98641058337653607651E3,
        9.56231892404756170795E3,
        7.99704160447350683650E3,
        2.82619278517639096600E3,
        3.36093607810698293419E2
    };
    const double first_zero_squared = 1.46819706421238932572E1;
    const double second_zero_squared = 4.92184563216946036703E1;
    const double square_root_two_over_pi =
        7.97884560802865355880E-1;
    (void)result_dtype;
    if (special_is_nan(x)) return x;
    if (x < 0.0) return -bessel_j1(-x, result_dtype);
    if (x <= 5.0) {
        double square = x * x;
        double ratio = special_polynomial(
            square, numerator_small, 3) /
            special_polynomial_one_leading(
                square, denominator_small, 8);
        return ratio * x *
            (square - first_zero_squared) *
            (square - second_zero_squared);
    }

    double inverse = 5.0 / x;
    double square = inverse * inverse;
    double p = special_polynomial(square, numerator_p, 6) /
        special_polynomial(square, denominator_p, 6);
    double q = special_polynomial(square, numerator_q, 7) /
        special_polynomial_one_leading(
            square, denominator_q, 7);
    volatile double phase = x - 2.35619449019234492885E0;
    return (p * cos(phase) - inverse * q * sin(phase)) *
        square_root_two_over_pi / sqrt(x);
}

CNP_API CnpArray* CNP_CALL cnp_i0(const CnpArray *x) {
    const char *function_name = "cnp_i0";
    if (!x) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "source array is required");
        return NULL;
    }
    if (x->dtype->kind == 'c') {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "i0 not supported for complex values");
        return NULL;
    }
    if (!special_real_dtype_supported(x, function_name))
        return NULL;

    CNP_TYPE result_dtype;
    CnpRealUnaryFunction operation;
    switch (x->dtype->type_num) {
        case CNP_HALF:
            result_dtype = CNP_HALF;
            operation = special_i0_half_scalar;
            break;
        case CNP_FLOAT:
            result_dtype = CNP_FLOAT;
            operation = special_i0_float_scalar;
            break;
        case CNP_LONGDOUBLE:
            result_dtype = CNP_LONGDOUBLE;
            operation = special_i0_double_scalar;
            break;
        case CNP_DOUBLE:
            result_dtype = CNP_DOUBLE;
            operation = special_i0_double_scalar;
            break;
        default:
            result_dtype = CNP_DOUBLE;
            operation = special_i0_double_scalar;
            break;
    }
    return special_real_unary_as(
        x, result_dtype, operation, function_name);
}

CNP_API CnpArray* CNP_CALL cnp_j0(const CnpArray *x) {
    return special_real_unary(x, bessel_j0, "cnp_j0");
}

CNP_API CnpArray* CNP_CALL cnp_j1(const CnpArray *x) {
    return special_real_unary(x, bessel_j1, "cnp_j1");
}

/* =========================================================================
 * digamma - Psi function (derivative of log gamma)
 * ========================================================================= */
static double special_digamma_one_to_two(double value) {
    static const double numerator[6] = {
        -0.0020713321167745952,
        -0.045251321448739056,
        -0.28919126444774784,
        -0.65031853770896507,
        -0.32555031186804491,
        0.25479851061131551
    };
    static const double denominator[7] = {
        -0.55789841321675513e-6,
        0.0021284987017821144,
        0.054151797245674225,
        0.43593529692665969,
        1.4606242909763515,
        2.0767117023730469,
        1.0
    };
    const float leading = 0.99558162689208984f;
    const double root_first = 1569415565.0 / 1073741824.0;
    const double root_second =
        (381566830.0 / 1073741824.0) / 1073741824.0;
    const double root_third =
        0.9016312093258695918615325266959189453125e-19;
    double displacement = value - root_first;
    displacement -= root_second;
    displacement -= root_third;
    double correction = special_polynomial(
        value - 1.0, numerator, 5) /
        special_polynomial(value - 1.0, denominator, 6);
    return special_split_sum_product(
        displacement, (double)leading, correction);
}

static double special_digamma_asymptotic(double value) {
    static const double coefficients[7] = {
        8.33333333333333333333e-2,
        -2.10927960927960927961e-2,
        7.57575757575757575758e-3,
        -4.16666666666666666667e-3,
        3.96825396825396825397e-3,
        -8.33333333333333333333e-3,
        8.33333333333333333333e-2
    };
    double correction = 0.0;
    if (value < 1.0e17) {
        double inverse_square = 1.0 / (value * value);
        correction = inverse_square * special_polynomial(
            inverse_square, coefficients, 6);
    }
    return log(value) - 0.5 / value - correction;
}

static double special_hurwitz_zeta_integer(
    int exponent, double offset) {
    static const double bernoulli_denominators[12] = {
        12.0,
        -720.0,
        30240.0,
        -1209600.0,
        47900160.0,
        -1.8924375803183791606e9,
        7.47242496e10,
        -2.950130727918164224e12,
        1.1646782814350067249e14,
        -4.5979787224074726105e15,
        1.8152105401943546773e17,
        -7.1661652561756670113e18
    };
    const double machine_epsilon = 2.22044604925031308085e-16;
    double exponent_value = (double)exponent;
    double sum = pow(offset, -exponent_value);
    double argument = offset;
    double term = 0.0;
    int iteration = 0;
    while (iteration < 9 || argument <= 9.0) {
        ++iteration;
        argument += 1.0;
        term = pow(argument, -exponent_value);
        sum += term;
        if (fabs(term / sum) < machine_epsilon)
            return sum;
    }

    double power = term;
    double rising_factorial = 1.0;
    double factor_index = 0.0;
    sum += term * argument / (exponent_value - 1.0);
    sum -= 0.5 * term;
    for (int index = 0; index < 12; ++index) {
        rising_factorial *= exponent_value + factor_index;
        power /= argument;
        term = rising_factorial * power /
            bernoulli_denominators[index];
        sum += term;
        if (fabs(term / sum) < machine_epsilon)
            return sum;
        factor_index += 1.0;
        rising_factorial *= exponent_value + factor_index;
        power /= argument;
        factor_index += 1.0;
    }
    return sum;
}

static double special_digamma_root_series(
    double value, double root, double root_value) {
    const double machine_epsilon = 2.22044604925031308085e-16;
    double displacement = value - root;
    double coefficient = -1.0;
    double result = root_value;
    for (int order = 1; order < 100; ++order) {
        coefficient *= -displacement;
        double term = coefficient * special_hurwitz_zeta_integer(
            order + 1, root);
        result += term;
        if (fabs(term) < machine_epsilon * fabs(result))
            break;
    }
    return result;
}

static double special_complex_absolute(CnpSpecialComplex value) {
    return hypot(value.real, value.imag);
}

static CnpSpecialComplex special_complex_reciprocal(
    CnpSpecialComplex value) {
    return special_complex_divide(special_complex(1.0, 0.0), value);
}

static CnpSpecialComplex special_complex_sinpi(
    CnpSpecialComplex value) {
    double angle = M_PI * remainder(value.real, 2.0);
    double hyperbolic = M_PI * value.imag;
    return special_complex(
        sin(angle) * cosh(hyperbolic),
        cos(angle) * sinh(hyperbolic));
}

static CnpSpecialComplex special_complex_cospi(
    CnpSpecialComplex value) {
    double angle = M_PI * remainder(value.real, 2.0);
    double hyperbolic = M_PI * value.imag;
    return special_complex(
        cos(angle) * cosh(hyperbolic),
        -sin(angle) * sinh(hyperbolic));
}

static CnpSpecialComplex special_digamma_complex_root_series(
    CnpSpecialComplex value, double root, double root_value) {
    const double machine_epsilon = 2.22044604925031308085e-16;
    CnpSpecialComplex displacement = special_complex_subtract(
        value, special_complex(root, 0.0));
    CnpSpecialComplex coefficient = special_complex(-1.0, 0.0);
    CnpSpecialComplex result = special_complex(root_value, 0.0);
    for (int order = 1; order < 100; ++order) {
        coefficient = special_complex_multiply(
            coefficient, special_complex_scale(displacement, -1.0));
        CnpSpecialComplex term = special_complex_scale(
            coefficient,
            special_hurwitz_zeta_integer(order + 1, root));
        result = special_complex_add(result, term);
        if (special_complex_absolute(term) <
                machine_epsilon * special_complex_absolute(result))
            break;
    }
    return result;
}

static CnpSpecialComplex special_digamma_complex_asymptotic(
    CnpSpecialComplex value) {
    static const double bernoulli_even[16] = {
        0.166666666666666667,
        -0.0333333333333333333,
        0.0238095238095238095,
        -0.0333333333333333333,
        0.0757575757575757576,
        -0.253113553113553114,
        1.16666666666666667,
        -7.09215686274509804,
        54.9711779448621554,
        -529.124242424242424,
        6192.12318840579710,
        -86580.2531135531136,
        1425517.16666666667,
        -27298231.0678160920,
        601580873.900642368,
        -15116315767.0921569
    };
    const double machine_epsilon = 2.22044604925031308085e-16;
    CnpSpecialComplex reciprocal = special_complex_reciprocal(value);
    CnpSpecialComplex reciprocal_square = special_complex_multiply(
        reciprocal, reciprocal);
    CnpSpecialComplex power = special_complex(1.0, 0.0);
    CnpSpecialComplex result = special_complex_subtract(
        special_complex_log(value),
        special_complex_scale(reciprocal, 0.5));
    for (int index = 0; index < 16; ++index) {
        power = special_complex_multiply(power, reciprocal_square);
        CnpSpecialComplex term = special_complex_scale(
            power, -bernoulli_even[index] / (2.0 * (index + 1)));
        result = special_complex_add(result, term);
        if (special_complex_absolute(term) <
                machine_epsilon * special_complex_absolute(result))
            break;
    }
    return result;
}

static CnpSpecialComplex special_digamma_complex_scalar(
    CnpSpecialComplex value) {
    const double positive_root = 1.4616321449683623;
    const double positive_root_value = -9.2412655217294275e-17;
    const double negative_root = -0.504083008264455409;
    const double negative_root_value = 7.2897639029768949e-17;
    const double asymptotic_radius = 16.0;
    CnpSpecialComplex result = special_complex(0.0, 0.0);
    double absolute = special_complex_absolute(value);

    if (value.imag == 0.0 && value.real <= 0.0 &&
            ceil(value.real) == value.real)
        return special_complex(NAN, NAN);
    if (special_complex_absolute(special_complex_subtract(
            value, special_complex(negative_root, 0.0))) < 0.3)
        return special_digamma_complex_root_series(
            value, negative_root, negative_root_value);

    if (value.real < 0.0 && fabs(value.imag) < asymptotic_radius) {
        CnpSpecialComplex cotangent = special_complex_divide(
            special_complex_cospi(value),
            special_complex_sinpi(value));
        result = special_complex_subtract(
            result, special_complex_scale(cotangent, M_PI));
        value = special_complex_subtract(
            special_complex(1.0, 0.0), value);
        absolute = special_complex_absolute(value);
    }

    if (absolute < 0.5) {
        result = special_complex_subtract(
            result, special_complex_reciprocal(value));
        value = special_complex_add(value, special_complex(1.0, 0.0));
        absolute = special_complex_absolute(value);
    }

    if (special_complex_absolute(special_complex_subtract(
            value, special_complex(positive_root, 0.0))) < 0.5) {
        result = special_complex_add(
            result,
            special_digamma_complex_root_series(
                value, positive_root, positive_root_value));
    } else if (absolute > asymptotic_radius) {
        result = special_complex_add(
            result, special_digamma_complex_asymptotic(value));
    } else if (value.real >= 0.0) {
        int steps = (int)(asymptotic_radius - absolute) + 1;
        CnpSpecialComplex shifted = special_complex_add(
            value, special_complex((double)steps, 0.0));
        CnpSpecialComplex recurrence =
            special_digamma_complex_asymptotic(shifted);
        for (int step = 1; step <= steps; ++step) {
            recurrence = special_complex_subtract(
                recurrence,
                special_complex_reciprocal(special_complex_subtract(
                    shifted, special_complex((double)step, 0.0))));
        }
        result = special_complex_add(result, recurrence);
    } else {
        int steps = (int)(asymptotic_radius - absolute) - 1;
        CnpSpecialComplex shifted = special_complex_subtract(
            value, special_complex((double)steps, 0.0));
        CnpSpecialComplex recurrence =
            special_digamma_complex_asymptotic(shifted);
        for (int step = 0; step < steps; ++step) {
            recurrence = special_complex_add(
                recurrence,
                special_complex_reciprocal(special_complex_add(
                    shifted, special_complex((double)step, 0.0))));
        }
        result = special_complex_add(result, recurrence);
    }
    return result;
}

static double special_digamma_scalar(
    double value, CNP_TYPE result_dtype) {
    const double euler = 0.577215664901532860606512090082402431;
    const double positive_root = 1.4616321449683623;
    const double positive_root_value = -9.2412655217294275e-17;
    const double negative_root = -0.504083008264455409;
    const double negative_root_value = 7.2897639029768949e-17;
    double result = 0.0;
    (void)result_dtype;
    if (special_is_nan(value)) return value;
    if (special_is_infinite(value))
        return value > 0.0 ? value : NAN;
    if (fabs(value - negative_root) < 0.3)
        return special_digamma_root_series(
            value, negative_root, negative_root_value);
    if (fabs(value - positive_root) < 0.5)
        return special_digamma_root_series(
            value, positive_root, positive_root_value);
    if (value == 0.0)
        return copysign(special_positive_infinity(), -value);
    if (value < 0.0) {
        double integral;
        double fraction = modf(value, &integral);
        if (fraction == 0.0) return NAN;
        result = -M_PI / tan(M_PI * fraction);
        value = 1.0 - value;
    }

    if (value <= 10.0 && value == floor(value)) {
        int integer = (int)value;
        for (int index = 1; index < integer; ++index)
            result += 1.0 / index;
        return result - euler;
    }

    if (value < 1.0) {
        result -= 1.0 / value;
        value += 1.0;
    } else if (value < 10.0) {
        while (value > 2.0) {
            value -= 1.0;
            result += 1.0 / value;
        }
    }
    if (value >= 1.0 && value <= 2.0)
        return result + special_digamma_one_to_two(value);
    return result + special_digamma_asymptotic(value);
}

static CnpArray *special_digamma_unary(const CnpArray *source) {
    if (!source) {
        cnp_set_error(
            CNP_ERR_GENERIC, "cnp_digamma", "source array is required");
        return NULL;
    }
    if (source->dtype->type_num != CNP_CFLOAT &&
            source->dtype->type_num != CNP_CDOUBLE)
        return special_real_unary(
            source, special_digamma_scalar, "cnp_digamma");

    CNP_TYPE result_dtype = source->dtype->type_num;
    CnpArray *result = special_new_keep_order(
        source, result_dtype, "cnp_digamma");
    if (!result) return NULL;
    for (int64_t index = 0; index < source->size; ++index) {
        int64_t source_offset = special_flat_offset(source, index);
        const char *source_pointer =
            (const char*)source->data + source_offset;
        CnpSpecialComplex input;
        if (result_dtype == CNP_CFLOAT) {
            const cnp_cfloat *value = (const cnp_cfloat*)source_pointer;
            input = special_complex(value->real, value->imag);
        } else {
            const cnp_cdouble *value = (const cnp_cdouble*)source_pointer;
            input = special_complex(value->real, value->imag);
        }
        CnpSpecialComplex output = special_digamma_complex_scalar(input);
        int64_t result_offset = special_flat_offset(result, index);
        char *result_pointer = (char*)result->data + result_offset;
        if (result_dtype == CNP_CFLOAT) {
            cnp_cfloat *target = (cnp_cfloat*)result_pointer;
            target->real = (float)output.real;
            target->imag = (float)output.imag;
        } else {
            cnp_cdouble *target = (cnp_cdouble*)result_pointer;
            target->real = output.real;
            target->imag = output.imag;
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_digamma(const CnpArray *x) {
    return special_digamma_unary(x);
}

/* =========================================================================
 * zeta - Riemann zeta function
 * ========================================================================= */
static double special_riemann_zeta_positive(double value) {
    static const double integer_zeta_minus_one[31] = {
        -1.50000000000000000000e0,
        0.0,
        6.44934066848226436472e-1,
        2.02056903159594285400e-1,
        8.23232337111381915160e-2,
        3.69277551433699263314e-2,
        1.73430619844491397145e-2,
        8.34927738192282683980e-3,
        4.07735619794433937869e-3,
        2.00839282608221441785e-3,
        9.94575127818085337146e-4,
        4.94188604119464558702e-4,
        2.46086553308048298638e-4,
        1.22713347578489146752e-4,
        6.12481350587048292585e-5,
        3.05882363070204935517e-5,
        1.52822594086518717326e-5,
        7.63719763789976227360e-6,
        3.81729326499983985646e-6,
        1.90821271655393892566e-6,
        9.53962033872796113152e-7,
        4.76932986787806463117e-7,
        2.38450502727732990004e-7,
        1.19219925965311073068e-7,
        5.96081890512594796124e-8,
        2.98035035146522801861e-8,
        1.49015548283650412347e-8,
        7.45071178983542949198e-9,
        3.72533402478845705482e-9,
        1.86265972351304900640e-9,
        9.31327432419668182872e-10
    };
    static const double medium_numerator[9] = {
        5.85746514569725319540e11,
        2.57534127756102572888e11,
        4.87781159567948256438e10,
        5.15399538023885770696e9,
        3.41646073514754094281e8,
        1.60837006880656492731e7,
        5.92785467342109522998e5,
        1.51129169964938823117e4,
        2.01822444485997955865e2
    };
    static const double medium_denominator[8] = {
        3.90497676373371157516e11,
        5.22858235368272161797e10,
        5.64451517271280543351e9,
        3.39006746015350418834e8,
        1.79410371500126453702e7,
        5.66666825131384797029e5,
        1.60382976810944131506e4,
        1.96436237223387314144e2
    };
    static const double large_numerator[11] = {
        8.70728567484590192539e6,
        1.76506865670346462757e8,
        2.60889506707483264896e10,
        5.29806374009894791647e11,
        2.26888156119238241487e13,
        3.31884402932705083599e14,
        5.13778997975868230192e15,
        -1.98123688133907171455e15,
        -9.92763810039983572356e16,
        7.82905376180870586444e16,
        9.26786275768927717187e16
    };
    static const double large_denominator[10] = {
        -7.92625410563741062861e6,
        -1.60529969932920229676e8,
        -2.37669260975543221788e10,
        -4.80319584350455169857e11,
        -2.07820961754173320170e13,
        -2.96075404507272223680e14,
        -4.86299103694609136686e15,
        5.34589509675789930199e15,
        5.71464111092297631292e16,
        -1.79915597658676556828e16
    };
    static const double unit_numerator[6] = {
        -3.28717474506562731748e-1,
        1.55162528742623950834e1,
        -2.48762831680821954401e2,
        1.01050368053237678329e3,
        1.26726061410235149405e4,
        -1.11578094770515181334e5
    };
    static const double unit_denominator[5] = {
        1.95107674914060531512e1,
        3.17710311750646984099e2,
        3.03835500874445748734e3,
        2.03665876435770579345e4,
        7.43853965136767874343e4
    };
    const double machine_epsilon = 2.22044604925031308085e-16;
    double integer_part;
    double inverse;
    double term;
    double sum;

    if (value == 1.0) return special_positive_infinity();
    if (value >= 127.0) return 1.0;
    integer_part = floor(value);
    if (integer_part == value && integer_part < 31.0)
        return 1.0 + integer_zeta_minus_one[(int)integer_part];
    if (value < 1.0) {
        double denominator = (1.0 - value) *
            special_polynomial_one_leading(
                value, unit_denominator, 5);
        return 1.0 + special_polynomial(
            value, unit_numerator, 5) / denominator;
    }
    if (value <= 10.0) {
        double denominator = pow(2.0, value) * (value - 1.0);
        inverse = 1.0 / value;
        return 1.0 + value * special_polynomial(
            inverse, medium_numerator, 8) /
            (denominator * special_polynomial_one_leading(
                inverse, medium_denominator, 8));
    }
    if (value <= 50.0) {
        inverse = special_polynomial(
            value, large_numerator, 10) /
            special_polynomial_one_leading(
                value, large_denominator, 10);
        return 1.0 + exp(inverse) + pow(2.0, -value);
    }

    sum = 0.0;
    inverse = 1.0;
    do {
        inverse += 2.0;
        term = pow(inverse, -value);
        sum += term;
    } while (term / sum > machine_epsilon);
    term = pow(2.0, -value);
    return 1.0 + (sum + term) / (1.0 - term);
}

static double special_riemann_zeta_small_negative(double value) {
    static const double coefficients[10] = {
        -1.0000000009110164892,
        -1.0000000057646759799,
        -9.9999983138417361078e-1,
        -1.0000013011460139596,
        -1.000001940896320456,
        -9.9987929950057116496e-1,
        -1.000785194477042408,
        -1.0031782279542924256,
        -9.1893853320467274178e-1,
        -1.5
    };
    return 1.0 + special_polynomial(value, coefficients, 9);
}

static double special_rational_same_degree(
    double value, const double *numerator,
    const double *denominator, int degree) {
    double argument;
    double numerator_result;
    double denominator_result;
    if (fabs(value) > 1.0) {
        argument = 1.0 / value;
        numerator_result = numerator[degree];
        denominator_result = denominator[degree];
        for (int index = degree - 1; index >= 0; --index) {
            numerator_result =
                numerator_result * argument + numerator[index];
            denominator_result =
                denominator_result * argument + denominator[index];
        }
    } else {
        argument = value;
        numerator_result = numerator[0];
        denominator_result = denominator[0];
        for (int index = 1; index <= degree; ++index) {
            numerator_result =
                numerator_result * argument + numerator[index];
            denominator_result =
                denominator_result * argument + denominator[index];
        }
    }
    return numerator_result / denominator_result;
}

static double special_lanczos_sum_expg_scaled(double value) {
    static const double numerator[13] = {
        0.006061842346248906525783753964555936883222,
        0.5098416655656676188125178644804694509993,
        19.51992788247617482847860966235652136208,
        449.9445569063168119446858607650988409623,
        6955.999602515376140356310115515198987526,
        75999.29304014542649875303443598909137092,
        601859.6171681098786670226533699352302507,
        3481712.15498064590882071018964774556468,
        14605578.08768506808414169982791359218571,
        43338889.32467613834773723740590533316085,
        86363131.28813859145546927288977868422342,
        103794043.1163445451906271053616070238554,
        56906521.91347156388090791033559122686859
    };
    static const double denominator[13] = {
        1.0,
        66.0,
        1925.0,
        32670.0,
        357423.0,
        2637558.0,
        13339535.0,
        45995730.0,
        105258076.0,
        150917976.0,
        120543840.0,
        39916800.0,
        0.0
    };
    return special_rational_same_degree(
        value, numerator, denominator, 12);
}

static double special_riemann_zeta_reflection(double positive) {
    const double lanczos_g = 6.024680040776729583740234375;
    const double square_root_two_over_pi =
        0.79788456080286535587989;
    const double euler_number =
        2.718281828459045235360287471352662498;
    double half = positive / 2.0;
    double shifted;
    double small_term;
    double base;
    double large_term;
    if (half == floor(half)) return 0.0;
    shifted = fmod(positive, 4.0);
    small_term = -square_root_two_over_pi *
        sin(0.5 * M_PI * shifted);
    small_term *= special_lanczos_sum_expg_scaled(positive + 1.0) *
        special_riemann_zeta_positive(positive + 1.0);
    base = (positive + lanczos_g + 0.5) /
        (2.0 * M_PI * euler_number);
    large_term = pow(base, positive + 0.5);
    if (!special_is_infinite(large_term) &&
            !special_is_nan(large_term))
        return large_term * small_term;
    large_term = pow(base, 0.5 * positive + 0.25);
    return (large_term * small_term) * large_term;
}

static double special_riemann_zeta_scalar(
    double value, CNP_TYPE result_dtype) {
    (void)result_dtype;
    if (special_is_nan(value)) return value;
    if (special_is_infinite(value))
        return value > 0.0 ? 1.0 : NAN;
    if (value < 0.0 && value > -0.01)
        return special_riemann_zeta_small_negative(value);
    if (value < 0.0)
        return special_riemann_zeta_reflection(-value);
    return special_riemann_zeta_positive(value);
}

CNP_API CnpArray* CNP_CALL cnp_zeta(const CnpArray *x) {
    return special_real_unary(
        x, special_riemann_zeta_scalar, "cnp_zeta");
}

/* =========================================================================
 * expit - Logistic sigmoid function: 1 / (1 + exp(-x))
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_expit(const CnpArray *x) {
    return special_real_unary(x, special_expit_scalar, "cnp_expit");
}

/* =========================================================================
 * logit - Inverse sigmoid: log(x / (1-x))
 * ========================================================================= */
CNP_API CnpArray* CNP_CALL cnp_logit(const CnpArray *x) {
    return special_real_unary(x, special_logit_scalar, "cnp_logit");
}

/* =========================================================================
 * softmax/log_softmax - stable normalization along one logical axis
 * ========================================================================= */
static bool softmax_accepts_dtype(
    const CnpArray *x, const char *function_name) {
    char kind = x->dtype->kind;
    if (kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f')
        return true;
    cnp_set_error(CNP_ERR_TYPE, function_name,
                  "source array must have a real numeric dtype");
    return false;
}

static CnpArray *softmax_axis(
    const CnpArray *x, int axis, bool logarithmic,
    const char *function_name) {
    int resolved_axis;
    if (!cnp_reduction_resolve_axis(
            x, axis, false, function_name, &resolved_axis))
        return NULL;
    if (!softmax_accepts_dtype(x, function_name)) return NULL;

    CnpReductionTraversal traversal;
    cnp_reduction_traversal_init(x, resolved_axis, &traversal);
    if (traversal.axis_length == 0 &&
            traversal.outer != 0 && traversal.inner != 0) {
        cnp_set_error(CNP_ERR_GENERIC, function_name,
                      "zero-size reduction has no identity");
        return NULL;
    }

    CNP_TYPE output_type = x->dtype->type_num == CNP_FLOAT
        ? CNP_FLOAT : CNP_DOUBLE;
    CnpArray *result = cnp_array_new(
        x->ndim, x->shape, output_type, CNP_ORDER_C);
    if (!result) return NULL;

    for (int64_t outer = 0; outer < traversal.outer; ++outer) {
        for (int64_t inner = 0; inner < traversal.inner; ++inner) {
            int64_t first_offset = cnp_reduction_source_offset(
                &traversal, outer, inner, 0);
            double maximum = cnp_get_element_double(
                x->data, first_offset, x->dtype->type_num);
            for (int64_t item = 1;
                 item < traversal.axis_length; ++item) {
                int64_t source_offset = cnp_reduction_source_offset(
                    &traversal, outer, inner, item);
                double value = cnp_get_element_double(
                    x->data, source_offset, x->dtype->type_num);
                if (isnan(value)) {
                    maximum = value;
                    break;
                }
                if (value > maximum) maximum = value;
            }

            if (logarithmic && !isfinite(maximum)) maximum = 0.0;
            if (output_type == CNP_FLOAT) {
                float maximum_float = (float)maximum;
                float total = 0.0f;
                for (int64_t item = 0;
                     item < traversal.axis_length; ++item) {
                    int64_t source_offset = cnp_reduction_source_offset(
                        &traversal, outer, inner, item);
                    float shifted = (float)cnp_get_element_double(
                        x->data, source_offset, x->dtype->type_num) -
                        maximum_float;
                    float exponential = expf(shifted);
                    int64_t output_index =
                        (outer * traversal.axis_length + item) *
                        traversal.inner + inner;
                    cnp_set_element_double(
                        result->data,
                        output_index * result->dtype->elsize,
                        output_type,
                        logarithmic ? shifted : exponential);
                    total += exponential;
                }
                float adjustment = logarithmic ? logf(total) : total;
                for (int64_t item = 0;
                     item < traversal.axis_length; ++item) {
                    int64_t output_index =
                        (outer * traversal.axis_length + item) *
                        traversal.inner + inner;
                    float value = *(float*)((char*)result->data +
                        output_index * result->dtype->elsize);
                    value = logarithmic
                        ? value - adjustment : value / adjustment;
                    *(float*)((char*)result->data +
                        output_index * result->dtype->elsize) = value;
                }
            } else {
                double total = 0.0;
                for (int64_t item = 0;
                     item < traversal.axis_length; ++item) {
                    int64_t source_offset = cnp_reduction_source_offset(
                        &traversal, outer, inner, item);
                    double shifted = cnp_get_element_double(
                        x->data, source_offset, x->dtype->type_num) -
                        maximum;
                    double exponential = exp(shifted);
                    int64_t output_index =
                        (outer * traversal.axis_length + item) *
                        traversal.inner + inner;
                    double value = logarithmic ? shifted : exponential;
                    ((double*)result->data)[output_index] = value;
                    total += exponential;
                }
                double adjustment = logarithmic ? log(total) : total;
                for (int64_t item = 0;
                     item < traversal.axis_length; ++item) {
                    int64_t output_index =
                        (outer * traversal.axis_length + item) *
                        traversal.inner + inner;
                    ((double*)result->data)[output_index] = logarithmic
                        ? ((double*)result->data)[output_index] - adjustment
                        : ((double*)result->data)[output_index] / adjustment;
                }
            }
        }
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_softmax(const CnpArray *x, int axis) {
    return softmax_axis(x, axis, false, "cnp_softmax");
}

CNP_API CnpArray* CNP_CALL cnp_log_softmax(
    const CnpArray *x, int axis) {
    return softmax_axis(x, axis, true, "cnp_log_softmax");
}
