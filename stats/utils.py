from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float

    def contains_value(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class Simulation:
    sample_size: int
    true_param: float
    estimated_param: float
    ci: ConfidenceInterval

    @property
    def ci_contains_true_param(self) -> bool:
        return self.ci.contains_value(self.true_param)

    @property
    def deviation(self) -> float:
        return self.estimated_param - self.true_param


@dataclass(frozen=True)
class Summary:
    true_param: float
    sample_size: int
    mean: float
    bias: float
    mse: float
    coverage_probability: float

def summarize(simulations: list[Simulation]) -> Summary:
    assert np.all([
        sim.true_param == simulations[0].true_param
        for sim in simulations
    ])
    assert np.all([
        sim.sample_size == simulations[0].sample_size
        for sim in simulations
    ])

    n_simulations = len(simulations)
    mean = np.mean([sim.estimated_param for sim in simulations])
    bias = np.mean([sim.deviation for sim in simulations])
    mse = np.square([sim.deviation for sim in simulations]).mean()
    coverage_probability = np.mean([sim.ci_contains_true_param for sim in simulations])

    return Summary(
        true_param=simulations[0].true_param,
        sample_size=simulations[0].sample_size,
        mean=mean,
        bias=bias,
        mse=mse,
        coverage_probability=coverage_probability
    )
