from typing import TypedDict

import numpy as np
import numpy.typing as npt
import arviz as az
import pymc as pm
from pymc.distributions.continuous import PositiveContinuous

from distributions.continuous import A
from .utils import ConfidenceInterval, Simulation


class PriorParams(TypedDict):
    name: str
    alpha: float
    beta: float

class SamplerParams(TypedDict):
    draws: int
    chains: int
    tune: int
    target_accept: float
    return_inferencedata: bool
    random_seed: np.random.Generator



def simulate(
        param: float,
        sample: npt.NDArray,
        prior: PositiveContinuous,
        prior_params: PriorParams,
        alpha: float,
        sampler_params: SamplerParams
    ) -> Simulation:

    with pm.Model() as model:
        beta = prior(**prior_params)
        y = A("y", beta=beta, observed=sample)


        idata = pm.sample(**sampler_params)

    estimate = float(idata.posterior.beta.mean().to_numpy())
    lower, upper = az.hdi(idata, hdi_prob=1 - alpha).beta.to_numpy()
    ci = ConfidenceInterval(lower=lower, upper=upper)

    return Simulation(
        sample_size=len(sample),
        true_param=param,
        estimated_param=estimate,
        ci=ci
    )


