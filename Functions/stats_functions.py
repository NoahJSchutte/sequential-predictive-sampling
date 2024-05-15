# This script contains functions that are related to random variables and statistical functions
# This is needed because scipy and numpy are both useful in different ways and have different conventions
import numpy as np
from scipy.stats import norm, t, expon, beta, lognorm


def get_mean(distribution, parameters):
    if distribution == 'exponential':
        return parameters[0]
    elif distribution == 'lognormal':
        m = parameters[0]
        s = parameters[1]
        return np.exp(m + 0.5*s**2)


def draw_from_distribution(distribution, parameters):
    # np.random is used for draws, so: normal: [mean, sigma], exponential: [shape], etc.
    if distribution == 'beta':
        draw = getattr(np.random, distribution)(*parameters[0:2])
        draw = parameters[2] + draw*(parameters[3] - parameters[2])
    else:
        draw = getattr(np.random, distribution)(*parameters)
    return draw


def draw_from_distribution_scipy(distribution, parameters):
    # np.random is used for draws, so: normal: [mean, sigma], exponential: [shape], etc.
    return getattr(np.random, distribution)(*parameters)


def lognorm_conditional_expectation_given_min(m, s, k):  # E[X | X >= k]
    k = max(k, 0.01)
    return np.exp(m+0.5*s**2) * norm.cdf((m+s**2-np.log(k))/s) / (1 - norm.cdf((np.log(k)-m)/s))


def get_inverse_cdf(distribution, percentile, parameters):  # uses scipy
    if distribution == 'lognormal':
        mean = parameters[0]
        stdev = parameters[1]
        return lognorm.ppf(percentile, stdev, scale=np.exp(mean))
    elif distribution == 'exponential':
        return expon.ppf(percentile, scale=parameters[0])
    elif distribution == 'beta':
        inverse_cdf_value = eval(f'{distribution}.ppf')(percentile, *parameters[0:2])
        return parameters[2] + inverse_cdf_value*(parameters[3] - parameters[2])
    else:
        return eval(f'{distribution}.ppf')(percentile, *parameters)


def get_t_bound(mean, variance, degrees, confidence, upper=True) -> float:
    probability_of_threshold = 1 - (1-confidence)**0.5
    difference_from_mean = variance**0.5 * t.ppf(probability_of_threshold, degrees)
    if upper:
        return mean + difference_from_mean
    else:  # lower
        return mean - difference_from_mean


def get_t_bounds(mean, variance, degrees, confidence) -> (float, float):
    difference_from_mean = variance**0.5 * t.ppf(confidence, degrees)
    return mean - difference_from_mean, mean + difference_from_mean


def norm_cdf(mean, variance):
    return 1 - norm.cdf(mean / variance**0.5)


