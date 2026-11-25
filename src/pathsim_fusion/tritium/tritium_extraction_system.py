#########################################################################################
##
##       Tritium Extraction System Blocks (tritium/tritium_extraction_system.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from typing import ClassVar

from .fuel_cycle import FuelCycleComponent, _check_fraction

# BLOCKS ================================================================================


class TritiumExtractionSystem(FuelCycleComponent):
    """Tritium extraction system as a residence time model with an efficiency.

    Lowest fidelity model of the extraction of tritium from the breeder. The
    outflow :math:`I / \\tau` is split by the extraction efficiency
    :math:`\\eta` into the extracted tritium and the residual that stays in
    the breeder:

    .. math::

        y_{extracted} = \\eta \\frac{I}{\\tau}

    .. math::

        y_{residual} = (1 - \\eta) \\frac{I}{\\tau}

    The inventory follows the residence time model of
    :class:`FuelCycleComponent`. The defaults are those of the ARC case of
    Meschini et al., Nucl. Fusion 63, 126005 (2023).

    Parameters
    ----------
    tau : float
        Residence time [s]. Must be positive. Default 86400.0.
    efficiency : float
        Fraction of the outflow that is extracted [-]. Must be between 0
        and 1. Default 0.95.
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
    inventory (0) : inventory of the extraction system [unit]
    extracted (1) : extracted tritium, ``efficiency * inventory / tau`` [unit/s]
    residual (2) : tritium that stays in the breeder [unit/s]
    loss (3) : non-radioactive loss plus decay [unit/s]
    """

    output_port_labels: ClassVar[dict[str, int]] = {
        "inventory": 0,
        "extracted": 1,
        "residual": 2,
        "loss": 3,
    }

    def __init__(
        self,
        tau=86400.0,
        efficiency=0.95,
        loss_fraction=0.0,
        decay_constant=0.0,
        source_term=0.0,
        initial_value=0.0,
    ):

        # input validation
        _check_fraction("efficiency", efficiency)

        # store parameters
        self.efficiency = efficiency

        super().__init__(
            tau=tau,
            loss_fraction=loss_fraction,
            decay_constant=decay_constant,
            source_term=source_term,
            initial_value=initial_value,
        )

    def _outlet_fractions(self):
        return (self.efficiency, 1 - self.efficiency)
