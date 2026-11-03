#########################################################################################
##
##                          FIRST-ORDER LAG (example block)
##                              (example_block.py)
##
##   Reference block shipped with the toolbox template. Replace this module
##   with your own blocks — keep the structure: a clear header, a docstring
##   documenting math / parameters / ports, input validation, and port labels.
##
#########################################################################################

# IMPORTS ==============================================================================

import numpy as np
from pathsim.blocks import DynamicalSystem

# BLOCKS ===============================================================================


class FirstOrderLag(DynamicalSystem):
    """First-order lag (low-pass filter) block.

    .. math::

        \\tau \\frac{dy}{dt} = u - y

    Parameters
    ----------
    tau : float
        Time constant [s]. Must be positive.
    y0 : float
        Initial output value. Default 0.0.

    Inputs
    ------
    u (0) : input signal

    Outputs
    -------
    y (0) : low-pass filtered signal
    """

    input_port_labels = {"u": 0}
    output_port_labels = {"y": 0}

    def __init__(self, tau=1.0, y0=0.0):
        # input validation
        if tau <= 0:
            raise ValueError(f"'tau' must be positive but is {tau}")

        # store parameters
        self.tau = float(tau)

        def _fn_d(x, u, t):
            (y,) = x
            (u0,) = u
            return np.array([(u0 - y) / self.tau])

        def _fn_a(x, u, t):
            return x.copy()

        super().__init__(
            func_dyn=_fn_d,
            func_alg=_fn_a,
            initial_value=np.array([float(y0)]),
        )
