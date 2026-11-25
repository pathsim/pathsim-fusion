#########################################################################################
##
##                     Residence Time Fuel Cycle Component Block
##
#########################################################################################

# IMPORTS ===============================================================================

from typing import ClassVar

import numpy as np
from pathsim.blocks.dynsys import DynamicalSystem

# BLOCKS ================================================================================


class FuelCycleComponent(DynamicalSystem):
    """Lumped fuel cycle component with a residence time, losses and decay.

    Residence time model of a tritium fuel cycle component as used by
    Meschini et al., Nucl. Fusion 63, 126005 (2023). The component holds one
    inventory :math:`I`. All inputs are inflows and are summed. The inventory
    leaves towards the next component after a mean residence time
    :math:`\\tau`, a fraction :math:`\\varepsilon` of that outflow is lost on
    top of it, and the inventory decays with the constant :math:`\\lambda`:

    .. math::

        \\frac{dI}{dt} = \\sum_i u_i + S - (1 + \\varepsilon) \\frac{I}{\\tau} - \\lambda I

    The outputs are the inventory, the outflow to the next component and the
    total loss rate (non-radioactive loss plus decay):

    .. math::

        y_0 = I

    .. math::

        y_1 = \\frac{I}{\\tau}

    .. math::

        y_2 = \\varepsilon \\frac{I}{\\tau} + \\lambda I

    so that the rate of change of the inventory is the inflows plus the
    source minus ``outflow`` and ``loss``. With an infinite ``tau`` the
    component has no outflow and only accumulates, which describes a storage.

    The block does not fix the unit of the inventory (for example kg or mol),
    the inflows and the source term are in that unit per second.

    Parameters
    ----------
    tau : float
        Residence time [s]. Must be positive. Default 1.0.
    loss_fraction : float
        Non-radioactive loss as a fraction of the outflow [-]. Must be
        non-negative. Default 0.0.
    decay_constant : float
        Radioactive decay constant [1/s]. Must be non-negative. Default 0.0.
    source_term : float
        Constant generation rate inside the component [unit/s]. Default 0.0.
    initial_value : float
        Initial inventory [unit]. Must be non-negative. Default 0.0.

    Inputs
    ------
    inflows (0, 1, ...) : any number of inflows, summed [unit/s]

    Outputs
    -------
    inventory (0) : inventory of the component [unit]
    outflow (1) : flow to the next component, ``inventory / tau`` [unit/s]
    loss (2) : non-radioactive loss plus decay [unit/s]
    """

    input_port_labels = None

    output_port_labels: ClassVar[dict[str, int]] = {
        "inventory": 0,
        "outflow": 1,
        "loss": 2,
    }

    def __init__(
        self,
        tau=1.0,
        loss_fraction=0.0,
        decay_constant=0.0,
        source_term=0.0,
        initial_value=0.0,
    ):

        # input validation
        if not tau > 0:
            raise ValueError(f"'tau' must be positive but is {tau}")
        if not loss_fraction >= 0:
            raise ValueError(
                f"'loss_fraction' must be non-negative but is {loss_fraction}"
            )
        if not decay_constant >= 0:
            raise ValueError(
                f"'decay_constant' must be non-negative but is {decay_constant}"
            )

        if np.ndim(initial_value) != 0 or not initial_value >= 0:
            raise ValueError(
                f"'initial_value' must be a non-negative number but is {initial_value}"
            )

        # store parameters
        self.tau = tau
        self.loss_fraction = loss_fraction
        self.decay_constant = decay_constant
        self.source_term = source_term

        # rhs of inventory ode
        def _fn_d(x, u, t):
            removal = (1 + self.loss_fraction) / self.tau + self.decay_constant
            return np.sum(u) + self.source_term - removal * x

        # jacobian of rhs wrt x
        def _jc_d(x, u, t):
            return -(1 + self.loss_fraction) / self.tau - self.decay_constant

        # output function: inventory, the outflow to each outlet and the loss
        def _fn_a(x, u, t):
            # the state is a plain float until a solver is set
            inventory = np.ravel(x)[0]
            outflow = inventory / self.tau
            loss = self.loss_fraction * outflow + self.decay_constant * inventory
            outlets = [f * outflow for f in self._outlet_fractions()]
            return np.array([inventory, *outlets, loss])

        super().__init__(
            func_dyn=_fn_d,
            jac_dyn=_jc_d,
            func_alg=_fn_a,
            initial_value=initial_value,
        )

    def _outlet_fractions(self):
        """Fractions of the outflow that go to each outlet port, in port order.

        The components derived from this block override it to route their
        outflow to several named outlets.
        """
        return (1.0,)
