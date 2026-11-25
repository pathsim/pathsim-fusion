#########################################################################################
##
##       Heat Exchanger Permeation Blocks (tritium/heat_exchanger_permeation.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from typing import ClassVar

from .fuel_cycle import FuelCycleComponent, _check_fraction

# BLOCKS ================================================================================


class HeatExchangerPermeation(FuelCycleComponent):
    """Heat exchanger tritium holdup with a permeated fraction (lowest fidelity).

    Tritium that reaches the heat exchanger with the breeder or coolant. The
    outflow :math:`I / \\tau` is split by the permeation fraction
    :math:`f_p` into the tritium that permeates through the heat exchanger
    wall and the tritium that returns with the fluid:

    .. math::

        y_{permeated} = f_p \\frac{I}{\\tau}

    .. math::

        y_{returned} = (1 - f_p) \\frac{I}{\\tau}

    The inventory follows the residence time model of
    :class:`FuelCycleComponent`. The defaults are those of the ARC case of
    Meschini et al., Nucl. Fusion 63, 126005 (2023), where the permeated
    tritium goes to the detritiation system.

    Parameters
    ----------
    tau : float
        Residence time [s]. Must be positive. Default 1000.0.
    permeation_fraction : float
        Fraction of the outflow that permeates through the wall [-]. Must
        be between 0 and 1. Default 1e-4.
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
    inventory (0) : inventory of the heat exchanger [unit]
    permeated (1) : tritium permeating through the wall [unit/s]
    returned (2) : tritium returning with the fluid [unit/s]
    loss (3) : non-radioactive loss plus decay [unit/s]
    """

    output_port_labels: ClassVar[dict[str, int]] = {
        "inventory": 0,
        "permeated": 1,
        "returned": 2,
        "loss": 3,
    }

    def __init__(
        self,
        tau=1000.0,
        permeation_fraction=1e-4,
        loss_fraction=0.0,
        decay_constant=0.0,
        source_term=0.0,
        initial_value=0.0,
    ):

        # input validation
        _check_fraction("permeation_fraction", permeation_fraction)

        # store parameters
        self.permeation_fraction = permeation_fraction

        super().__init__(
            tau=tau,
            loss_fraction=loss_fraction,
            decay_constant=decay_constant,
            source_term=source_term,
            initial_value=initial_value,
        )

    def _outlet_fractions(self):
        return (self.permeation_fraction, 1 - self.permeation_fraction)
