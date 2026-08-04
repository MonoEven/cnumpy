/**
 * cnumpy random number generation - PCG/xoshiro based PRNG
 */
#include "../include/cnumpy/cnumpy_internal.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Global random state */
CnpRandomState g_cnp_random_state = {{0}, 0, 0.0};

/* =========================================================================
 * xoshiro256** PRNG implementation
 * ========================================================================= */
static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t cnp_random_uint64(void) {
    uint64_t *s = g_cnp_random_state.state;
    uint64_t result = rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);

    return result;
}

/* SplitMix64 for seeding */
static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

CNP_API void CNP_CALL cnp_random_seed(uint64_t seed) {
    uint64_t sm_state = seed;
    g_cnp_random_state.state[0] = splitmix64(&sm_state);
    g_cnp_random_state.state[1] = splitmix64(&sm_state);
    g_cnp_random_state.state[2] = splitmix64(&sm_state);
    g_cnp_random_state.state[3] = splitmix64(&sm_state);
    g_cnp_random_state.has_gauss = 0;
    g_cnp_random_state.gauss = 0.0;
}

/* Generate uniform double in [0, 1) */
double cnp_random_double(void) {
    return (double)(cnp_random_uint64() >> 11) * (1.0 / 9007199254740992.0);
}

/* Generate standard normal using Box-Muller */
double cnp_random_gauss(void) {
    if (g_cnp_random_state.has_gauss) {
        g_cnp_random_state.has_gauss = 0;
        return g_cnp_random_state.gauss;
    }

    double u1, u2;
    do { u1 = cnp_random_double(); } while (u1 <= 1e-15);
    u2 = cnp_random_double();

    double mag = sqrt(-2.0 * log(u1));
    g_cnp_random_state.gauss = mag * sin(2.0 * 3.14159265358979323846 * u2);
    g_cnp_random_state.has_gauss = 1;
    return mag * cos(2.0 * 3.14159265358979323846 * u2);
}

/* =========================================================================
 * Random array generators
 * ========================================================================= */
static bool random_output_shape_valid(
    int ndim, const int64_t *shape, const char *function_name) {
    int64_t size = 1;
    if (ndim < 0 || ndim > CNP_MAXDIMS) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "output rank must be between 0 and %d", CNP_MAXDIMS);
        return false;
    }
    if (ndim > 0 && !shape) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "output shape must not be null when rank is positive");
        return false;
    }
    for (int dimension = 0; dimension < ndim; ++dimension) {
        if (shape[dimension] < 0) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "output shape must not contain negative lengths");
            return false;
        }
        if (shape[dimension] != 0 &&
                size > INT64_MAX / shape[dimension]) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "output shape is too large");
            return false;
        }
        size *= shape[dimension];
    }
    return true;
}

CnpArray *cnp_random_output_new(
    int ndim, const int64_t *shape, CNP_TYPE type,
    const char *function_name) {
    if (!random_output_shape_valid(ndim, shape, function_name)) return NULL;
    CnpArray *result = cnp_array_new(
        ndim, shape, type, CNP_ORDER_C);
    if (!result) cnp_relabel_error(function_name);
    return result;
}

static uint64_t random_bounded(uint64_t bound) {
    uint64_t threshold = (UINT64_C(0) - bound) % bound;
    uint64_t value;
    do {
        value = cnp_random_uint64();
    } while (value < threshold);
    return value % bound;
}

double cnp_random_gamma_sample(double shape_param) {
    if (shape_param < 1.0) {
        double uniform;
        do {
            uniform = cnp_random_double();
        } while (uniform == 0.0);
        return cnp_random_gamma_sample(shape_param + 1.0) *
            pow(uniform, 1.0 / shape_param);
    }

    double d = shape_param - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double normal;
        double value;
        do {
            normal = cnp_random_gauss();
            value = 1.0 + c * normal;
        } while (value <= 0.0);
        value = value * value * value;
        double uniform = cnp_random_double();
        double normal_fourth = normal * normal * normal * normal;
        if (uniform < 1.0 - 0.0331 * normal_fourth ||
                log(uniform) < 0.5 * normal * normal +
                    d * (1.0 - value + log(value))) {
            return d * value;
        }
    }
}

int64_t cnp_random_poisson_sample(double lambda) {
    if (lambda < 30.0) {
        double limit = exp(-lambda);
        double product = 1.0;
        int64_t count = 0;
        do {
            ++count;
            product *= cnp_random_double();
        } while (product > limit);
        return count - 1;
    }

    double root = sqrt(lambda);
    double b = 0.931 + 2.53 * root;
    double a = -0.059 + 0.02483 * b;
    double inverse_alpha = 1.1239 + 1.1328 / (b - 3.4);
    double squeeze = 0.9277 - 3.6224 / (b - 2.0);
    for (;;) {
        double centered = cnp_random_double() - 0.5;
        double uniform = cnp_random_double();
        double distance = 0.5 - fabs(centered);
        int64_t value = (int64_t)floor(
            (2.0 * a / distance + b) * centered + lambda + 0.43);
        if (distance >= 0.07 && uniform <= squeeze) return value;
        if (value < 0 ||
                (distance < 0.013 && uniform > distance)) continue;
        if (log(
                uniform * inverse_alpha /
                (a / (distance * distance) + b)) <=
                -lambda + value * log(lambda) - lgamma(value + 1.0)) {
            return value;
        }
    }
}

int64_t cnp_random_binomial_sample(int64_t n, double probability) {
    int64_t count = 0;
    for (int64_t trial = 0; trial < n; ++trial) {
        if (cnp_random_double() < probability) ++count;
    }
    return count;
}

static double random_positive_infinity(void) {
    uint64_t bits = UINT64_C(0x7ff0000000000000);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

CNP_API CnpArray* CNP_CALL cnp_random_random(int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_random";
    CnpArray *arr = cnp_random_output_new(
        ndim, shape, CNP_DOUBLE, function_name);
    if (!arr) return NULL;

    double *data = (double*)arr->data;
    for (int64_t i = 0; i < arr->size; i++) {
        data[i] = cnp_random_double();
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_random_uniform(double low, double high, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_uniform";
    double range = high - low;
    if (!isfinite(range)) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "uniform range exceeds valid floating-point bounds");
        return NULL;
    }
    CnpArray *arr = cnp_random_output_new(
        ndim, shape, CNP_DOUBLE, function_name);
    if (!arr) return NULL;

    double *data = (double*)arr->data;
    for (int64_t i = 0; i < arr->size; i++) {
        data[i] = low + cnp_random_double() * range;
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_random_normal(double mean, double std, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_normal";
    if (std < 0.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "normal scale must be non-negative");
        return NULL;
    }
    CnpArray *arr = cnp_random_output_new(
        ndim, shape, CNP_DOUBLE, function_name);
    if (!arr) return NULL;

    double *data = (double*)arr->data;
    for (int64_t i = 0; i < arr->size; i++) {
        data[i] = mean + std * cnp_random_gauss();
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_random_standard_normal(int ndim, const int64_t *shape) {
    CnpArray *result = cnp_random_normal(0.0, 1.0, ndim, shape);
    if (!result) cnp_relabel_error("cnp_random_standard_normal");
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_integers(int64_t low, int64_t high, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_integers";
    if (low < INT32_MIN || high > INT32_MAX || low > high) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "inclusive integer bounds are invalid for NumPy's Windows int32 result");
        return NULL;
    }
    CnpArray *arr = cnp_random_output_new(
        ndim, shape, CNP_INT, function_name);
    if (!arr) return NULL;

    uint64_t range = (uint64_t)(high - low) + 1;
    int32_t *data = (int32_t*)arr->data;
    for (int64_t i = 0; i < arr->size; i++) {
        data[i] = (int32_t)(low + (int64_t)random_bounded(range));
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_random_randint(int64_t low, int64_t high, int ndim, const int64_t *shape) {
    /* randint: [low, high) */
    const char *function_name = "cnp_random_randint";
    if (low < INT32_MIN || high > (int64_t)INT32_MAX + 1 || low >= high) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "half-open integer bounds are invalid for NumPy's Windows int32 result");
        return NULL;
    }
    CnpArray *arr = cnp_random_output_new(
        ndim, shape, CNP_INT, function_name);
    if (!arr) return NULL;

    uint64_t range = (uint64_t)(high - low);
    int32_t *data = (int32_t*)arr->data;
    for (int64_t i = 0; i < arr->size; i++) {
        data[i] = (int32_t)(low + (int64_t)random_bounded(range));
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_random_binomial(int64_t n, double p, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_binomial";
    if (n < 0 || n > INT32_MAX || !isfinite(p) || p < 0.0 || p > 1.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "binomial n and probability are outside valid bounds");
        return NULL;
    }
    CnpArray *arr = cnp_random_output_new(
        ndim, shape, CNP_INT, function_name);
    if (!arr) return NULL;

    int32_t *data = (int32_t*)arr->data;
    for (int64_t i = 0; i < arr->size; i++) {
        data[i] = (int32_t)cnp_random_binomial_sample(n, p);
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_random_poisson(double lam, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_poisson";
    double maximum_lambda = (double)INT32_MAX -
        10.0 * sqrt((double)INT32_MAX);
    if (!isfinite(lam) || lam < 0.0 || lam > maximum_lambda) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "Poisson lambda is outside valid Windows int32 bounds");
        return NULL;
    }
    CnpArray *arr = cnp_random_output_new(
        ndim, shape, CNP_INT, function_name);
    if (!arr) return NULL;

    int32_t *data = (int32_t*)arr->data;
    for (int64_t i = 0; i < arr->size; i++) {
        data[i] = (int32_t)cnp_random_poisson_sample(lam);
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_random_exponential(double scale, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_exponential";
    if (scale < 0.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "exponential scale must be non-negative");
        return NULL;
    }
    CnpArray *arr = cnp_random_output_new(
        ndim, shape, CNP_DOUBLE, function_name);
    if (!arr) return NULL;

    double *data = (double*)arr->data;
    for (int64_t i = 0; i < arr->size; i++) {
        double u;
        do { u = cnp_random_double(); } while (u <= 0.0);
        data[i] = -scale * log(u);
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_random_gamma(double shape_param, double scale, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_gamma";
    if (shape_param < 0.0 || scale < 0.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "gamma shape and scale must be non-negative");
        return NULL;
    }
    CnpArray *arr = cnp_random_output_new(
        ndim, shape, CNP_DOUBLE, function_name);
    if (!arr) return NULL;

    double *data = (double*)arr->data;
    for (int64_t i = 0; i < arr->size; i++) {
        if (isnan(shape_param) || isnan(scale)) data[i] = NAN;
        else if (shape_param == 0.0 || scale == 0.0) data[i] = 0.0;
        else if (isinf(shape_param) || isinf(scale))
            data[i] = random_positive_infinity();
        else data[i] = cnp_random_gamma_sample(shape_param) * scale;
    }
    return arr;
}

CNP_API CnpArray* CNP_CALL cnp_random_beta(double a, double b, int ndim, const int64_t *shape) {
    const char *function_name = "cnp_random_beta";
    if (a <= 0.0 || b <= 0.0) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "beta shape parameters must be positive");
        return NULL;
    }
    CnpArray *arr = cnp_random_output_new(
        ndim, shape, CNP_DOUBLE, function_name);
    if (!arr) return NULL;

    double *data = (double*)arr->data;
    for (int64_t i = 0; i < arr->size; i++) {
        if (isnan(a) || isnan(b) || isinf(a)) data[i] = NAN;
        else if (isinf(b)) data[i] = 0.0;
        else {
            double left = cnp_random_gamma_sample(a);
            double right = cnp_random_gamma_sample(b);
            data[i] = left / (left + right);
        }
    }
    return arr;
}

static bool random_choice_real_probability_type(CNP_TYPE type) {
    return type == CNP_BOOL ||
        cnp_type_is_integer(type) || cnp_type_is_float(type);
}

static bool random_choice_output_size(
    int size_ndim,
    const int64_t *size_shape,
    bool size_none,
    int64_t *output_size,
    const char *function_name) {
    int64_t count = 1;
    if (size_none) {
        if (size_ndim != 0 || size_shape != NULL) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "size=None must not include an output shape");
            return false;
        }
        *output_size = 1;
        return true;
    }
    if (size_ndim < 0 || size_ndim > CNP_MAXDIMS) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "choice output dimensions are invalid");
        return false;
    }
    if (size_ndim > 0 && !size_shape) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "choice output shape must not be null");
        return false;
    }
    for (int dimension = 0; dimension < size_ndim; dimension++) {
        int64_t length = size_shape[dimension];
        if (length < 0) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "choice output shape must not contain negative lengths");
            return false;
        }
        if (length != 0 && count > INT64_MAX / length) {
            cnp_set_error(
                CNP_ERR_SHAPE, function_name,
                "choice output shape is too large");
            return false;
        }
        count *= length;
    }
    *output_size = count;
    return true;
}

static uint64_t random_choice_bounded(uint64_t bound) {
    return random_bounded(bound);
}

static void random_choice_copy_value(
    const CnpArray *population,
    int64_t population_index,
    CnpArray *result,
    int64_t result_index) {
    const char *source =
        (const char*)population->data + population->offset +
        population_index * population->strides[0];
    char *destination =
        (char*)result->data + result_index * result->dtype->elsize;
    memcpy(destination, source, (size_t)result->dtype->elsize);
}

static double *random_choice_probabilities(
    const CnpArray *probabilities,
    int64_t population_size,
    int64_t output_size,
    bool replace,
    int64_t *positive_count,
    const char *function_name) {
    double *weights;
    double sum = 0.0;
    double compensation = 0.0;
    double tolerance;
    CNP_TYPE probability_type;

    *positive_count = 0;
    if (!probabilities) return NULL;
    if (probabilities->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "probabilities must be one-dimensional");
        return NULL;
    }
    if (probabilities->size != population_size) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "probability length must match the population length");
        return NULL;
    }
    probability_type = probabilities->dtype->type_num;
    if (!random_choice_real_probability_type(probability_type)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "probabilities must have a real numeric dtype");
        return NULL;
    }
    if ((uint64_t)population_size > SIZE_MAX / sizeof(double)) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "probability workspace is too large");
        return NULL;
    }
    weights = population_size > 0
        ? (double*)cnp_malloc((size_t)population_size * sizeof(double))
        : NULL;
    if (population_size > 0 && !weights) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "unable to allocate probability workspace");
        return NULL;
    }

    for (int64_t index = 0; index < population_size; index++) {
        int64_t offset = probabilities->offset +
            index * probabilities->strides[0];
        double value = cnp_get_element_double(
            probabilities->data, offset, probability_type);
        double adjusted;
        double next_sum;
        if (!isfinite(value)) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "probabilities must contain only finite values");
            cnp_free(weights, (size_t)population_size * sizeof(double));
            return NULL;
        }
        if (value < 0.0) {
            cnp_set_error(
                CNP_ERR_GENERIC, function_name,
                "probabilities must not contain negative values");
            cnp_free(weights, (size_t)population_size * sizeof(double));
            return NULL;
        }
        weights[index] = value;
        if (value > 0.0) (*positive_count)++;
        adjusted = value - compensation;
        next_sum = sum + adjusted;
        compensation = (next_sum - sum) - adjusted;
        sum = next_sum;
    }
    if (sum == 0.0) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "probabilities must not be all zero");
        cnp_free(weights, (size_t)population_size * sizeof(double));
        return NULL;
    }
    tolerance = probability_type == CNP_HALF ||
            probability_type == CNP_FLOAT
        ? sqrt(FLT_EPSILON) : sqrt(DBL_EPSILON);
    if (fabs(sum - 1.0) > tolerance) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "probabilities must sum to 1 within dtype precision");
        cnp_free(weights, (size_t)population_size * sizeof(double));
        return NULL;
    }
    if (!replace && output_size > *positive_count) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "fewer non-zero probability entries than requested samples");
        cnp_free(weights, (size_t)population_size * sizeof(double));
        return NULL;
    }
    for (int64_t index = 0; index < population_size; index++) {
        weights[index] /= sum;
    }
    return weights;
}

static CnpArray *random_choice_execute(
    const CnpArray *population,
    int size_ndim,
    const int64_t *size_shape,
    bool size_none,
    bool replace,
    const CnpArray *probabilities,
    const char *function_name) {
    int64_t population_size;
    int64_t output_size;
    int64_t positive_count = 0;
    double *weights = NULL;
    int64_t *indices = NULL;
    CnpArray *result = NULL;

    if (!population) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "population must not be null");
        return NULL;
    }
    if (population->ndim != 1) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "population must be one-dimensional");
        return NULL;
    }
    if (!random_choice_output_size(
            size_ndim, size_shape, size_none,
            &output_size, function_name)) {
        return NULL;
    }
    population_size = population->size;
    if (population_size == 0 && output_size > 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "population cannot be empty when samples are requested");
        return NULL;
    }
    if (!replace && output_size > population_size) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "cannot take a larger sample than the population without replacement");
        return NULL;
    }
    if (probabilities) {
        weights = random_choice_probabilities(
            probabilities, population_size, output_size,
            replace, &positive_count, function_name);
        if (!weights) return NULL;
    }
    if (!replace && !probabilities && population_size > 0) {
        if ((uint64_t)population_size > SIZE_MAX / sizeof(int64_t)) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "choice index workspace is too large");
            return NULL;
        }
        indices = (int64_t*)cnp_malloc(
            (size_t)population_size * sizeof(int64_t));
        if (!indices) {
            cnp_set_error(
                CNP_ERR_MEMORY, function_name,
                "unable to allocate choice index workspace");
            return NULL;
        }
        for (int64_t index = 0; index < population_size; index++) {
            indices[index] = index;
        }
    }

    result = cnp_array_new(
        size_none ? 0 : size_ndim,
        size_none ? NULL : size_shape,
        population->dtype->type_num,
        CNP_ORDER_C);
    if (!result) {
        if (weights) {
            cnp_free(weights, (size_t)population_size * sizeof(double));
        }
        if (indices) {
            cnp_free(indices, (size_t)population_size * sizeof(int64_t));
        }
        cnp_relabel_error(function_name);
        return NULL;
    }

    if (!probabilities && replace) {
        for (int64_t sample = 0; sample < output_size; sample++) {
            int64_t selected = (int64_t)random_choice_bounded(
                (uint64_t)population_size);
            random_choice_copy_value(
                population, selected, result, sample);
        }
    } else if (!probabilities) {
        for (int64_t sample = 0; sample < output_size; sample++) {
            int64_t selected_offset = sample +
                (int64_t)random_choice_bounded(
                    (uint64_t)(population_size - sample));
            int64_t temporary = indices[sample];
            indices[sample] = indices[selected_offset];
            indices[selected_offset] = temporary;
            random_choice_copy_value(
                population, indices[sample], result, sample);
        }
    } else if (replace) {
        double cumulative = 0.0;
        for (int64_t index = 0; index < population_size; index++) {
            cumulative += weights[index];
            weights[index] = cumulative;
        }
        weights[population_size - 1] = 1.0;
        for (int64_t sample = 0; sample < output_size; sample++) {
            double target = cnp_random_double();
            int64_t low = 0;
            int64_t high = population_size;
            while (low < high) {
                int64_t middle = low + (high - low) / 2;
                if (target < weights[middle]) {
                    high = middle;
                } else {
                    low = middle + 1;
                }
            }
            random_choice_copy_value(population, low, result, sample);
        }
    } else {
        for (int64_t sample = 0; sample < output_size; sample++) {
            double total = 0.0;
            double cumulative = 0.0;
            double target;
            int64_t selected = -1;
            for (int64_t index = 0; index < population_size; index++) {
                total += weights[index];
            }
            target = cnp_random_double() * total;
            for (int64_t index = 0; index < population_size; index++) {
                cumulative += weights[index];
                if (weights[index] > 0.0 && target < cumulative) {
                    selected = index;
                    break;
                }
            }
            if (selected < 0) {
                cnp_set_error(
                    CNP_ERR_GENERIC, function_name,
                    "weighted choice failed to select a population entry");
                cnp_array_free(result);
                cnp_free(
                    weights, (size_t)population_size * sizeof(double));
                return NULL;
            }
            random_choice_copy_value(
                population, selected, result, sample);
            weights[selected] = 0.0;
        }
    }

    if (weights) {
        cnp_free(weights, (size_t)population_size * sizeof(double));
    }
    if (indices) {
        cnp_free(indices, (size_t)population_size * sizeof(int64_t));
    }
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_random_choice_v2(
    const CnpArray *a,
    int size_ndim,
    const int64_t *size_shape,
    bool size_none,
    bool replace,
    const CnpArray *p) {
    return random_choice_execute(
        a, size_ndim, size_shape, size_none, replace, p,
        "cnp_random_choice_v2");
}

CNP_API CnpArray* CNP_CALL cnp_random_choice(
    const CnpArray *a, int64_t size, bool replace, const CnpArray *p) {
    return random_choice_execute(
        a, 1, &size, false, replace, p,
        "cnp_random_choice");
}

static bool random_validate_permutation_input(
    const CnpArray *arr, bool require_writeable,
    const char *function_name) {
    if (!arr) {
        cnp_set_error(
            CNP_ERR_GENERIC, function_name,
            "array must not be NULL");
        return false;
    }
    if (arr->ndim == 0) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "array must be at least 1-dimensional");
        return false;
    }
    if (require_writeable && !(arr->flags & CNP_ARRAY_WRITEABLE)) {
        cnp_set_error(
            CNP_ERR_TYPE, function_name,
            "array is not writeable");
        return false;
    }
    return true;
}

static int64_t random_shuffle_element_offset(
    const CnpArray *arr, int64_t row, int64_t inner_index) {
    int64_t offset = arr->offset + row * arr->strides[0];
    for (int axis = arr->ndim - 1; axis > 0; --axis) {
        int64_t coordinate = inner_index % arr->shape[axis];
        inner_index /= arr->shape[axis];
        offset += coordinate * arr->strides[axis];
    }
    return offset;
}

static CNP_STATUS random_shuffle_execute(
    CnpArray *arr, const char *function_name) {
    if (!random_validate_permutation_input(
            arr, true, function_name))
        return cnp_get_error(NULL);
    if (arr->size <= 1 || arr->shape[0] <= 1) return CNP_OK;

    int element_size = arr->dtype->elsize;
    int64_t row_count = arr->shape[0];
    int64_t row_size = arr->size / row_count;
    size_t row_bytes = (size_t)row_size * (size_t)element_size;
    bool rows_are_contiguous =
        (arr->flags & CNP_ARRAY_C_CONTIGUOUS) != 0;
    size_t temporary_bytes = rows_are_contiguous
        ? row_bytes : (size_t)element_size;
    char *temporary = (char*)cnp_malloc(temporary_bytes);
    if (!temporary) {
        cnp_set_error(
            CNP_ERR_MEMORY, function_name,
            "failed to allocate shuffle row buffer");
        return CNP_ERR_MEMORY;
    }

    for (int64_t row = row_count - 1; row > 0; --row) {
        int64_t selected = (int64_t)random_choice_bounded(
            (uint64_t)(row + 1));
        if (row == selected) continue;

        if (rows_are_contiguous) {
            char *row_pointer = (char*)arr->data + arr->offset +
                row * arr->strides[0];
            char *selected_pointer = (char*)arr->data + arr->offset +
                selected * arr->strides[0];
            memcpy(temporary, row_pointer, row_bytes);
            memcpy(row_pointer, selected_pointer, row_bytes);
            memcpy(selected_pointer, temporary, row_bytes);
            continue;
        }

        for (int64_t inner = 0; inner < row_size; ++inner) {
            int64_t row_offset = random_shuffle_element_offset(
                arr, row, inner);
            int64_t selected_offset = random_shuffle_element_offset(
                arr, selected, inner);
            char *row_pointer = (char*)arr->data + row_offset;
            char *selected_pointer = (char*)arr->data + selected_offset;
            memcpy(temporary, row_pointer, (size_t)element_size);
            memcpy(row_pointer, selected_pointer, (size_t)element_size);
            memcpy(selected_pointer, temporary, (size_t)element_size);
        }
    }
    cnp_free(temporary, temporary_bytes);
    return CNP_OK;
}

CNP_API CnpArray* CNP_CALL cnp_random_permutation(const CnpArray *arr) {
    const char *function_name = "cnp_random_permutation";
    if (!random_validate_permutation_input(
            arr, false, function_name))
        return NULL;
    CnpArray *result = cnp_array_copy(arr);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    CNP_STATUS status = random_shuffle_execute(result, function_name);
    if (status != CNP_OK) {
        cnp_array_free(result);
        return NULL;
    }
    return result;
}

CNP_API void CNP_CALL cnp_random_shuffle(CnpArray *arr) {
    (void)random_shuffle_execute(arr, "cnp_random_shuffle");
}
