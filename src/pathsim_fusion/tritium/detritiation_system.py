#########################################################################################
##
##             Detritiation System Blocks (tritium/detritiation_system.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from .fuel_cycle import FuelCycleComponent

# BLOCKS ================================================================================


class DetritiationSystem(FuelCycleComponent):
    """Detritiation system as a residence time model (lowest fidelity).

    Recovers tritium from waste and permeated streams and returns it to the
    isotope separation. The inventory leaves after the residence time
    :math:`\\tau`:

    .. math::

        \\frac{dI}{dt} = \\sum_i u_i + S - (1 + \\varepsilon) \\frac{I}{\\tau} - \\lambda I

    The default residence time is that of the detritiation system in the ARC
    case of Meschini et al., Nucl. Fusion 63, 126005 (2023).

    Parameters
    ----------
    tau : float
        Residence time [s]. Must be positive. Default 3600.0.
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
    inventory (0) : inventory of the detritiation system [unit]
    outflow (1) : recovered tritium, ``inventory / tau`` [unit/s]
    loss (2) : non-radioactive loss plus decay [unit/s]
    """

    def __init__(
        self,
        tau=3600.0,
        loss_fraction=0.0,
        decay_constant=0.0,
        source_term=0.0,
        initial_value=0.0,
    ):
        super().__init__(
            tau=tau,
            loss_fraction=loss_fraction,
            decay_constant=decay_constant,
            source_term=source_term,
            initial_value=initial_value,
        )
