#########################################################################################
##
##                           Stack Blocks (tritium/stack.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from typing import ClassVar

import numpy as np

from .fuel_cycle import FuelCycleComponent

# BLOCKS ================================================================================


class Stack(FuelCycleComponent):
    """Stack that accumulates the tritium released to the environment.

    The stack is the end of the fuel cycle and has no outflow. It accumulates
    what flows in, so its inventory is the cumulative release:

    .. math::

        \\frac{dI}{dt} = \\sum_i u_i - \\lambda I

    Parameters
    ----------
    decay_constant : float
        Radioactive decay constant of the released tritium [1/s]. Must be
        non-negative. Default 0.0.
    initial_value : float
        Initial cumulative release [unit]. Must be non-negative. Default 0.0.

    Inputs
    ------
    inflows (0, 1, ...) : any number of inflows, summed [unit/s]

    Outputs
    -------
    inventory (0) : cumulative release [unit]
    loss (1) : decay [unit/s]
    """

    output_port_labels: ClassVar[dict[str, int]] = {
        "inventory": 0,
        "loss": 1,
    }

    def __init__(self, decay_constant=0.0, initial_value=0.0):
        super().__init__(
            tau=np.inf,
            decay_constant=decay_constant,
            initial_value=initial_value,
        )

    def _outlet_fractions(self):
        return ()
