import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.random.op import RandomVariable
from pymc.distributions.continuous import PositiveContinuous
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.shape_utils import rv_size_is_none

from distributions.ops import alshenawy_rv

# Subclassing `PositiveContinuous` will dispatch a default `log` transformation
class Alshenawy(PositiveContinuous):
    # This will be used by the metaclass `DistributionMeta` to dispatch the
    # class `logp` and `logcdf` methods to the `blah` `Op` defined in the last line of the code above.
    rv_op = alshenawy_rv

    # dist() is responsible for returning an instance of the rv_op.
    # We pass the standard parametrizations to super().dist
    @classmethod
    def dist(cls, beta, **kwargs):
        beta = pt.as_tensor_variable(beta)

        # The first value-only argument should be a list of the parameters that
        # the rv_op needs in order to be instantiated
        return super().dist([beta], **kwargs)

    # support_point returns a symbolic expression for the stable point from which to start sampling
    # the variable, given the implicit `rv`, `size` and `param1` ... `paramN`.
    # This is typically a "representative" point such as the the mean or mode.
    def support_point(rv, size, beta):
        support_point, _ = pt.broadcast_arrays(beta)
        if not rv_size_is_none(size):
            support_point = pt.full(size, support_point)
        return support_point

    # Logp returns a symbolic expression for the elementwise log-pdf or log-pmf evaluation
    # of the variable given the `value` of the variable and the parameters `param1` ... `paramN`.
    def logp(value, beta):
        logp_expression = (
            beta / value
            - 2 * pm.math.log(value)
            + (1 - pm.math.exp(beta / value)) / beta
        )

        # A switch is often used to enforce the distribution support domain
        bounded_logp_expression = pt.switch(
            pt.gt(value, 0),
            logp_expression,
            -np.inf,
        )

        # We use `check_parameters` for parameter validation. After the default expression,
        # multiple comma-separated symbolic conditions can be added.
        # Whenever a bound is invalidated, the returned expression raises an error
        # with the message defined in the optional `msg` keyword argument.
        return check_parameters(
            bounded_logp_expression,
            beta > 0,
            msg="beta > 0",
        )
