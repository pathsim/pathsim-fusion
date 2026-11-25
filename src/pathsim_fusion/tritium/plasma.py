#########################################################################################
##
##                          Plasma Blocks (tritium/plasma.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from .fuel_cycle import FuelCycleComponent

# BLOCKS ================================================================================


class Plasma(FuelCycleComponent):
    """Plasma as a residence time model (lowest fidelity).

    Holds the fuel inside the plasma. The fuel injected by the fuelling
    system is the inflow, the inventory leaves towards the fusion chamber
    pump after the residence time :math:`\\tau`. The burning of the fuel is
    not part of this model:

    .. math::

        \\frac{dI}{dt} = \\sum_i u_i + S - (1 + \\varepsilon) \\frac{I}{\\tau} - \\lambda I

    The default residence time is a placeholder, no reference value is set
    for this component yet.

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
    inventory (0) : inventory of the plasma [unit]
    outflow (1) : flow to the fusion chamber pump, ``inventory / tau`` [unit/s]
    loss (2) : non-radioactive loss plus decay [unit/s]
    """

    def __init__(
        self,
        tau=1.0,
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
