

def FactorizedDistributionMoments(distribution, samples=250, exclude=[]):
    """Compute the mean in a memory-safe manner"""
    means = {}
    variances = {}
    for k, v in distribution.model.items():
        if k in exclude:
            continue
        if callable(v):
            raise AttributeError("Need factorized nameddistribution object")
        else:
            test_distribution = v
        try:
            mean = test_distribution.mean()
            variance = test_distribution.variance()
        except NotImplementedError:
            sum_1 = test_distribution.sample()
            sum_2 = sum_1 ** 2
            for _ in range(samples - 1):
                s = test_distribution.sample()
                sum_1 = sum_1 + s
                sum_2 = sum_2 + s ** 2
            mean = sum_1 / samples
            variance = sum_2 / samples - mean ** 2
        means[k] = mean
        variances[k] = variance
    return means, variances
