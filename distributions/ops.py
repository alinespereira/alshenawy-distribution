from typing import Tuple

import numpy as np
from pytensor.tensor.random.op import RandomVariable

from distributions.rv import alshenawy


class AlshenawyRV(RandomVariable):
    name: str = "alshenawy"

    # Provide a numpy-style signature for this RV, which indicates
    # the number and core dimensionality of each input and output.
    signature = "()->()"

    # The NumPy/PyTensor dtype for this RV (e.g. `"int32"`, `"int64"`).
    # The standard in the library is `"int64"` for discrete variables
    # and `"floatX"` for continuous variables
    dtype: str = "floatX"

    # A pretty text and LaTeX representation for the RV
    _print_name: Tuple[str, str] = ("A", "\\operatorname{A}")

    # If you want to add a custom signature and default values for the
    # parameters, do it like this. Otherwise this can be left out.

    #     return super().__call__(loc, scale, **kwargs)

    # This is the Python code that produces samples.  Its signature will always
    # start with a NumPy `RandomState` object, then the distribution
    # parameters, and, finally, the size.

    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        beta: np.ndarray,
        size: Tuple[int, ...],
    ) -> np.ndarray:
        return alshenawy.rvs(beta, random_state=rng, size=size)

# Create the actual `RandomVariable` `Op`...
alshenawy_rv = AlshenawyRV()
