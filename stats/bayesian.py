import logging
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import arviz as az
import pymc as pm
from pymc.distributions.continuous import PositiveContinuous

from distributions.continuous import A
from .utils import ConfidenceInterval, Simulation

logger = logging.getLogger("pymc")
logger.setLevel(logging.WARNING)


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
    nuts_sampler: str
    progressbar: bool


def make_model(
        sample: npt.NDArray,
        prior: PositiveContinuous,
        prior_params: PriorParams,
    ) -> pm.Model:
    with pm.Model() as model:
        data = pm.Data("sample", sample)
        beta = prior(**prior_params)
        y = A("y", beta=beta, observed=data)

    return model

def simulate(
        param: float,
        sample: npt.NDArray,
        model: pm.Model,
        alpha: float,
        sampler_params: SamplerParams,
    ) -> Simulation:

    with model:
        model.set_data("sample", sample)

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


