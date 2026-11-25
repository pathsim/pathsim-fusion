#########################################################################################
##
##             Fusion Chamber Pump Blocks (tritium/fusion_chamber_pump.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from typing import ClassVar

from .fuel_cycle import FuelCycleComponent, _check_fraction

# BLOCKS ================================================================================


class FusionChamberPump(FuelCycleComponent):
    """Fusion chamber vacuum pump with direct internal recycling (lowest fidelity).

    Holds the unburnt fuel pumped out of the fusion chamber. The outflow
    :math:`I / \\tau` is split by the direct internal recycling fraction
    :math:`f_{DIR}` into the fuel that is recycled directly and the exhaust
    that goes to the exhaust processing:

    .. math::

        y_{recycled} = f_{DIR} \\frac{I}{\\tau}

    .. math::

        y_{exhaust} = (1 - f_{DIR}) \\frac{I}{\\tau}

    The inventory follows the residence time model of
    :class:`FuelCycleComponent`. The defaults are those of the ARC case of
    Meschini et al., Nucl. Fusion 63, 126005 (2023).

    Parameters
    ----------
    tau : float
        Residence time [s]. Must be positive. Default 600.0.
    dir_fraction : float
        Fraction of the outflow that is recycled directly [-]. Must be
        between 0 and 1. Default 0.3.
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
    inventory (0) : inventory of the pump [unit]
    recycled (1) : directly recycled fuel, ``dir_fraction * inventory / tau`` [unit/s]
    exhaust (2) : flow to the exhaust processing [unit/s]
    loss (3) : non-radioactive loss plus decay [unit/s]
    """

    output_port_labels: ClassVar[dict[str, int]] = {
        "inventory": 0,
        "recycled": 1,
        "exhaust": 2,
        "loss": 3,
    }

    def __init__(
        self,
        tau=600.0,
        dir_fraction=0.3,
        loss_fraction=0.0,
        decay_constant=0.0,
        source_term=0.0,
        initial_value=0.0,
    ):

        # input validation
        _check_fraction("dir_fraction", dir_fraction)

        # store parameters
        self.dir_fraction = dir_fraction

        super().__init__(
            tau=tau,
            loss_fraction=loss_fraction,
            decay_constant=decay_constant,
            source_term=source_term,
            initial_value=initial_value,
        )

    def _outlet_fractions(self):
        return (self.dir_fraction, 1 - self.dir_fraction)
