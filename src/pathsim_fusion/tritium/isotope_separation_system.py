#########################################################################################
##
##       Isotope Separation System Blocks (tritium/isotope_separation_system.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from typing import ClassVar

from .fuel_cycle import FuelCycleComponent, _check_fraction

# BLOCKS ================================================================================


class IsotopeSeparationSystem(FuelCycleComponent):
    """Isotope separation system as a residence time model with a waste fraction.

    Lowest fidelity model of the isotope separation. The outflow
    :math:`I / \\tau` is split by the waste fraction :math:`f_w` into the
    product that goes to the storage and the waste stream that goes to the
    detritiation:

    .. math::

        y_{product} = (1 - f_w) \\frac{I}{\\tau}

    .. math::

        y_{waste} = f_w \\frac{I}{\\tau}

    The inventory follows the residence time model of
    :class:`FuelCycleComponent`. The defaults are those of the ARC case of
    Meschini et al., Nucl. Fusion 63, 126005 (2023), Table 4.

    Parameters
    ----------
    tau : float
        Residence time [s]. Must be positive. Default 13320.0.
    waste_fraction : float
        Fraction of the outflow that goes to the detritiation [-]. Must be
        between 0 and 1. Default 0.1.
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
    inventory (0) : inventory of the isotope separation [unit]
    product (1) : separated fuel to the storage [unit/s]
    waste (2) : waste stream to the detritiation [unit/s]
    loss (3) : non-radioactive loss plus decay [unit/s]
    """

    output_port_labels: ClassVar[dict[str, int]] = {
        "inventory": 0,
        "product": 1,
        "waste": 2,
        "loss": 3,
    }

    def __init__(
        self,
        tau=13320.0,
        waste_fraction=0.1,
        loss_fraction=0.0,
        decay_constant=0.0,
        source_term=0.0,
        initial_value=0.0,
    ):

        # input validation
        _check_fraction("waste_fraction", waste_fraction)

        # store parameters
        self.waste_fraction = waste_fraction

        super().__init__(
            tau=tau,
            loss_fraction=loss_fraction,
            decay_constant=decay_constant,
            source_term=source_term,
            initial_value=initial_value,
        )

    def _outlet_fractions(self):
        return (1 - self.waste_fraction, self.waste_fraction)
