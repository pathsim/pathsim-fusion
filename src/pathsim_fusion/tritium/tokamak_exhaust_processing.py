#########################################################################################
##
##      Tokamak Exhaust Processing Blocks (tritium/tokamak_exhaust_processing.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from .fuel_cycle import FuelCycleComponent

# BLOCKS ================================================================================


class TokamakExhaustProcessing(FuelCycleComponent):
    """Tokamak exhaust processing as a residence time model (lowest fidelity).

    Cleans the exhaust of the fusion chamber (fuel cleanup) before the
    isotope separation. The inventory leaves after the residence time
    :math:`\\tau`:

    .. math::

        \\frac{dI}{dt} = \\sum_i u_i + S - (1 + \\varepsilon) \\frac{I}{\\tau} - \\lambda I

    The default residence time is that of the fuel cleanup in the ARC case
    of Meschini et al., Nucl. Fusion 63, 126005 (2023), Table 4.

    Parameters
    ----------
    tau : float
        Residence time [s]. Must be positive. Default 1080.0.
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
    inventory (0) : inventory of the exhaust processing [unit]
    outflow (1) : flow to the isotope separation, ``inventory / tau`` [unit/s]
    loss (2) : non-radioactive loss plus decay [unit/s]
    """

    def __init__(
        self,
        tau=1080.0,
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
