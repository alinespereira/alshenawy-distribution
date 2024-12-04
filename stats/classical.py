from typing import Callable

import numpy as np
from scipy import stats, optimize
import numpy.typing as npt

from .utils import ConfidenceInterval, Simulation

LogLikelihood = Callable[[npt.NDArray], float]


def simulate(
        param: float,
        negative_log_likelihood: LogLikelihood,
        initial_guess: float,
        sample_size: int,
        alpha: float,
        **optimizer_params
    ) -> Simulation:

    if "param" in optimizer_params:
        optimizer_params.pop('param')
    if "negative_log_likelihood" in optimizer_params:
        optimizer_params.pop('negative_log_likelihood')
    if "initial_guess" in optimizer_params:
        optimizer_params.pop('initial_guess')
    if "sample_size" in optimizer_params:
        optimizer_params.pop('sample_size')
    if "alpha" in optimizer_params:
        optimizer_params.pop('alpha')

    result = optimize.minimize(
        negative_log_likelihood,
        x0=initial_guess,
        **optimizer_params
    )
    estimate = result.x[0]
    # Cálculo do intervalo de confiança assintótico para o patâmetro
    observed_information = 1 / result.hess_inv.todense()[0, 0]
    estimator_distribution = stats.norm(
        loc=estimate,
        scale=1 / np.sqrt(sample_size * observed_information)
    )
    ci = ConfidenceInterval(
        lower=estimator_distribution.ppf(alpha / 2),
        upper=estimator_distribution.ppf(1 - alpha / 2)
    )
    return Simulation(
        sample_size=sample_size,
        true_param=param,
        estimated_param=estimate,
        ci=ci
    )

