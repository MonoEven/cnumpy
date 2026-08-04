/**
 * NumPy-compatible scalar-parameter random distributions.
 *
 * The public generator is the project xoshiro256** state in random.c.  These
 * distributions deliberately share that state so cnp_random_seed controls
 * every public random routine.
 */
#include "../include/cnumpy/cnumpy_internal.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>

#define CNP_RANDOM_PI 3.14159265358979323846264338327950288

static double random_open_unit(void) {
    double value;
    do {
        value = cnp_random_double();
    } while (value == 0.0);
    return value;
}

static bool random_positive_parameter(
    double value, const char *name, const char *function_name) {
    if (isfinite(value) && value > 0.0) return true;
    cnp_set_error(
        CNP_ERR_VALUE, function_name,
        "%s must be finite and positive", name);
    return false;
}

static bool random_nonnegative_parameter(
    double value, const char *name, const char *function_name) {
    if (isfinite(value) && value >= 0.0) return true;
    cnp_set_error(
        CNP_ERR_VALUE, function_name,
        "%s must be finite and non-negative", name);
    return false;
}

static CnpArray *random_double_output(
    int ndim, const int64_t *shape, const char *function_name) {
    return cnp_random_output_new(
        ndim, shape, CNP_DOUBLE, function_name);
}

static CnpArray *random_int_output(
    int ndim, const int64_t *shape, const char *function_name) {
    return cnp_random_output_new(
        ndim, shape, CNP_INT, function_name);
}

static int64_t random_logseries_sample(double probability) {
    long double term =
        -(long double)probability / log1pl(-(long double)probability);
    long double cumulative = term;
    long double target = (long double)cnp_random_double();
    int64_t value = 1;
    while (target > cumulative) {
        ++value;
        term *= (long double)probability *
            (long double)(value - 1) / (long double)value;
        cumulative += term;
    }
    return value;
}

CNP_API CnpArray* CNP_CALL cnp_random_logseries(
    double p, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_logseries";
    if (!isfinite(p) || p <= 0.0 || p >= 1.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "probability must be finite and strictly between zero and one");
        return NULL;
    }
    CnpArray *result = random_int_output(ndim, shape, function_name);
    if (!result) return NULL;
    int32_t *output = (int32_t*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        int64_t value = random_logseries_sample(p);
        if (value > INT32_MAX) {
            cnp_array_decref(result);
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "sample exceeds NumPy's Windows int32 result range");
            return NULL;
        }
        output[index] = (int32_t)value;
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_negative_binomial(
    double n, double p, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_negative_binomial";
    if (!random_positive_parameter(n, "n", function_name)) return NULL;
    if (!isfinite(p) || p <= 0.0 || p > 1.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "probability must be finite and in (0, 1]");
        return NULL;
    }
    CnpArray *result = random_int_output(ndim, shape, function_name);
    if (!result) return NULL;
    int32_t *output = (int32_t*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        double lambda = p == 1.0
            ? 0.0 : cnp_random_gamma_sample(n) * (1.0 - p) / p;
        if (!isfinite(lambda) || lambda >
                (double)INT32_MAX - 10.0 * sqrt((double)INT32_MAX)) {
            cnp_array_decref(result);
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "sample rate exceeds NumPy's Windows int32 result range");
            return NULL;
        }
        output[index] = (int32_t)cnp_random_poisson_sample(lambda);
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_pareto(
    double a, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_pareto";
    if (!random_positive_parameter(a, "shape", function_name)) return NULL;
    CnpArray *result = random_double_output(ndim, shape, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        output[index] = expm1(
            -log1p(-cnp_random_double()) / a);
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_power(
    double a, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_power";
    if (!random_positive_parameter(a, "shape", function_name)) return NULL;
    CnpArray *result = random_double_output(ndim, shape, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        output[index] = pow(cnp_random_double(), 1.0 / a);
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_rayleigh(
    double scale, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_rayleigh";
    if (!random_nonnegative_parameter(scale, "scale", function_name))
        return NULL;
    CnpArray *result = random_double_output(ndim, shape, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        output[index] = scale * sqrt(
            -2.0 * log1p(-cnp_random_double()));
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_standard_cauchy(
    int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_standard_cauchy";
    CnpArray *result = random_double_output(ndim, shape, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        output[index] = tan(
            CNP_RANDOM_PI * (cnp_random_double() - 0.5));
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_standard_t(
    double df, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_standard_t";
    if (!random_positive_parameter(df, "degrees of freedom", function_name))
        return NULL;
    CnpArray *result = random_double_output(ndim, shape, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        double denominator = cnp_random_gamma_sample(df / 2.0) * 2.0;
        output[index] = cnp_random_gauss() /
            sqrt(denominator / df);
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_triangular(
    double left, double mode, double right,
    int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_triangular";
    if (!isfinite(left) || !isfinite(mode) || !isfinite(right) ||
            left > mode || mode > right || left == right) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "triangular parameters must satisfy left <= mode <= right with left < right");
        return NULL;
    }
    CnpArray *result = random_double_output(ndim, shape, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    double split = (mode - left) / (right - left);
    for (int64_t index = 0; index < result->size; ++index) {
        double uniform = cnp_random_double();
        output[index] = uniform < split
            ? left + sqrt(uniform * (right - left) * (mode - left))
            : right - sqrt(
                (1.0 - uniform) * (right - left) * (right - mode));
    }
    return result;
}

static double random_wrap_angle(double value) {
    double wrapped = fmod(value + CNP_RANDOM_PI, 2.0 * CNP_RANDOM_PI);
    if (wrapped < 0.0) wrapped += 2.0 * CNP_RANDOM_PI;
    return wrapped - CNP_RANDOM_PI;
}

CNP_API CnpArray* CNP_CALL cnp_random_vonmises(
    double mu, double kappa, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_vonmises";
    if (!random_nonnegative_parameter(kappa, "kappa", function_name))
        return NULL;
    CnpArray *result = random_double_output(ndim, shape, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        if (kappa <= 1e-8) {
            output[index] =
                2.0 * CNP_RANDOM_PI * cnp_random_double() - CNP_RANDOM_PI;
            continue;
        }
        double tau = 1.0 + sqrt(1.0 + 4.0 * kappa * kappa);
        double rho = (tau - sqrt(2.0 * tau)) / (2.0 * kappa);
        double ratio = (1.0 + rho * rho) / (2.0 * rho);
        double cosine;
        for (;;) {
            double z = cos(CNP_RANDOM_PI * cnp_random_double());
            cosine = (1.0 + ratio * z) / (ratio + z);
            double threshold = kappa * (ratio - cosine);
            double uniform = cnp_random_double();
            if (uniform < threshold * (2.0 - threshold) ||
                    uniform <= threshold * exp(1.0 - threshold)) break;
        }
        double angle = cnp_random_double() > 0.5
            ? acos(cosine) : -acos(cosine);
        output[index] = random_wrap_angle(mu + angle);
    }
    return result;
}

static double random_noncentral_chisquare_sample(double df, double nonc) {
    int64_t poisson = cnp_random_poisson_sample(nonc / 2.0);
    return cnp_random_gamma_sample((df + 2.0 * poisson) / 2.0) * 2.0;
}

CNP_API CnpArray* CNP_CALL cnp_random_noncentral_chisquare(
    double df, double nonc, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_noncentral_chisquare";
    if (!random_positive_parameter(df, "degrees of freedom", function_name) ||
            !random_nonnegative_parameter(nonc, "noncentrality", function_name))
        return NULL;
    CnpArray *result = random_double_output(ndim, shape, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t index = 0; index < result->size; ++index)
        output[index] = random_noncentral_chisquare_sample(df, nonc);
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_noncentral_f(
    double dfnum, double dfden, double nonc,
    int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_noncentral_f";
    if (!random_positive_parameter(
            dfnum, "numerator degrees of freedom", function_name) ||
            !random_positive_parameter(
                dfden, "denominator degrees of freedom", function_name) ||
            !random_nonnegative_parameter(
                nonc, "noncentrality", function_name)) return NULL;
    CnpArray *result = random_double_output(ndim, shape, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        double numerator = random_noncentral_chisquare_sample(dfnum, nonc);
        double denominator = cnp_random_gamma_sample(dfden / 2.0) * 2.0;
        output[index] = (numerator / dfnum) / (denominator / dfden);
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_f(
    double dfnum, double dfden, int ndim, const int64_t *shape) {
    CnpArray *result = cnp_random_noncentral_f(
        dfnum, dfden, 0.0, ndim, shape);
    if (!result) cnp_relabel_error("cnp_random_f");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_laplace(
    double loc, double scale, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_laplace";
    if (!random_nonnegative_parameter(scale, "scale", function_name))
        return NULL;
    CnpArray *result = random_double_output(ndim, shape, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        double centered = cnp_random_double() - 0.5;
        output[index] = loc - scale * copysign(
            log1p(-2.0 * fabs(centered)), centered);
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_logistic(
    double loc, double scale, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_logistic";
    if (!random_nonnegative_parameter(scale, "scale", function_name))
        return NULL;
    CnpArray *result = random_double_output(ndim, shape, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        double uniform = random_open_unit();
        output[index] = loc + scale * log(uniform / (1.0 - uniform));
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_gumbel(
    double loc, double scale, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_gumbel";
    if (!random_nonnegative_parameter(scale, "scale", function_name))
        return NULL;
    CnpArray *result = random_double_output(ndim, shape, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        output[index] = loc - scale * log(-log(random_open_unit()));
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_dirichlet(
    const double *alpha, int k, int64_t nsamples) {
    const char *function_name = "cnp_random_dirichlet";
    if (!alpha || k <= 0 || nsamples < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "alpha, a positive component count, and a non-negative sample count are required");
        return NULL;
    }
    for (int component = 0; component < k; ++component) {
        if (!isfinite(alpha[component]) || alpha[component] <= 0.0) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "every alpha component must be finite and positive");
            return NULL;
        }
    }
    int64_t shape[2] = {nsamples, k};
    CnpArray *result = random_double_output(2, shape, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t sample = 0; sample < nsamples; ++sample) {
        double sum;
        do {
            sum = 0.0;
            for (int component = 0; component < k; ++component) {
                double value = cnp_random_gamma_sample(alpha[component]);
                output[sample * k + component] = value;
                sum += value;
            }
        } while (sum == 0.0);
        for (int component = 0; component < k; ++component)
            output[sample * k + component] /= sum;
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_multinomial(
    int64_t n, const double *pvals, int k, int64_t nsamples) {
    const char *function_name = "cnp_random_multinomial";
    if (!pvals || k <= 0 || n < 0 || n > INT32_MAX || nsamples < 0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "n, probabilities, component count, or sample count is invalid");
        return NULL;
    }
    double leading_sum = 0.0;
    for (int component = 0; component < k; ++component) {
        if (!isfinite(pvals[component]) ||
                pvals[component] < 0.0 || pvals[component] > 1.0) {
            cnp_set_error(
                CNP_ERR_VALUE, function_name,
                "probabilities must be finite and in [0, 1]");
            return NULL;
        }
        if (component < k - 1) leading_sum += pvals[component];
    }
    if (leading_sum > 1.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "the probabilities before the final component sum to more than one");
        return NULL;
    }
    int64_t shape[2] = {nsamples, k};
    CnpArray *result = random_int_output(2, shape, function_name);
    if (!result) return NULL;
    int32_t *output = (int32_t*)result->data;
    for (int64_t sample = 0; sample < nsamples; ++sample) {
        int64_t remaining_n = n;
        double remaining_probability = 1.0;
        for (int component = 0; component < k - 1; ++component) {
            double conditional = remaining_probability == 0.0
                ? 0.0 : pvals[component] / remaining_probability;
            if (conditional > 1.0) conditional = 1.0;
            int64_t value = cnp_random_binomial_sample(
                remaining_n, conditional);
            output[sample * k + component] = (int32_t)value;
            remaining_n -= value;
            remaining_probability -= pvals[component];
        }
        output[sample * k + k - 1] = (int32_t)remaining_n;
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_weibull(
    double a, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_weibull";
    if (a < 0.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "shape must be non-negative");
        return NULL;
    }
    CnpArray *result = random_double_output(ndim, shape, function_name);
    if (!result) return NULL;
    double *output = (double*)result->data;
    for (int64_t index = 0; index < result->size; ++index) {
        if (a == 0.0) output[index] = 0.0;
        else output[index] = pow(
            -log1p(-cnp_random_double()), 1.0 / a);
    }
    return result;
}
