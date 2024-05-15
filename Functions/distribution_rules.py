def uniform1(mean):
    lower_bound = max(0, mean - mean**0.5)
    upper_bound = mean + mean**0.5
    return [lower_bound, upper_bound]


def uniform2(mean):
    lower_bound = 0
    upper_bound = 2*mean
    return [lower_bound, upper_bound]


def exponential1(mean):
    return [mean]


def beta1(mean):
    if mean == 0:
        alpha, beta, lb, ub = 0, 0, 0, 0
    else:
        variance = mean/3
        lb = mean/2
        ub = mean*2
        alpha = ((lb - mean)*(lb*ub - lb*mean - ub*mean + mean**2 + variance)) / (variance*(ub - lb))
        beta = -((ub - mean)*(lb*ub - lb*mean - ub*mean + mean**2 + variance)) / (variance*(ub - lb))

    return [alpha, beta, lb, ub]


def beta2(mean):
    if mean == 0:
        alpha, beta, lb, ub = 0, 0, 0, 0
    else:
        variance = mean**2/3
        lb = mean/2
        ub = mean*2
        alpha = ((lb - mean)*(lb*ub - lb*mean - ub*mean + mean**2 + variance)) / (variance*(ub - lb))
        beta = -((ub - mean)*(lb*ub - lb*mean - ub*mean + mean**2 + variance)) / (variance*(ub - lb))

    return [alpha, beta, lb, ub]



