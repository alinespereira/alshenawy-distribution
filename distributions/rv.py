import numpy as np
from scipy.stats import rv_continuous


class alshenawy_gen(rv_continuous):
    "Alshenawy's A distribution"
    def _pdf(self, x, beta):
        return np.exp((1 / beta) * (1 - np.exp(beta / x)) + beta / x) / (x ** 2)

    def _cdf(self, x, beta):
        return np.exp((1 / beta) * (1 - np.exp(beta / x)))

    def _logpdf(self, x, beta):
        return beta / x - 2 * np.log(x) + (1 - np.exp(beta / x)) / beta

    def _logcdf(self, x, beta):
        return (1 / beta) * (1 - np.exp(beta / x))

alshenawy = alshenawy_gen(a=0.0, name='alshenawy')
