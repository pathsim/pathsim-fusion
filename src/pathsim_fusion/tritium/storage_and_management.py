#########################################################################################
##
##          Storage and Management Blocks (tritium/storage_and_management.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from typing import ClassVar

import numpy as np

from .fuel_cycle import FuelCycleComponent

# BLOCKS ================================================================================


class StorageAndManagement(FuelCycleComponent):
    """Fuel storage that accumulates its inflows and decays (lowest fidelity).

    The storage has no residence time. It accumulates what flows in, and the
    fuel taken out for fuelling is connected as a negative inflow:

    .. math::

        \\frac{dI}{dt} = \\sum_i u_i + S - \\lambda I

    Parameters
    ----------
    decay_constant : float
        Radioactive decay constant [1/s]. Must be non-negative. Default 0.0.
    source_term : float
        Constant generation rate inside the component [unit/s]. Default 0.0.
    initial_value : float
        Initial inventory, for example the startup inventory [unit]. Must be
        non-negative. Default 0.0.

    Inputs
    ------
    inflows (0, 1, ...) : any number of inflows, summed, withdrawals negative [unit/s]

    Outputs
    -------
    inventory (0) : inventory of the storage [unit]
    loss (1) : decay [unit/s]
    """

    output_port_labels: ClassVar[dict[str, int]] = {
        "inventory": 0,
        "loss": 1,
    }

    def __init__(self, decay_constant=0.0, source_term=0.0, initial_value=0.0):
        super().__init__(
            tau=np.inf,
            decay_constant=decay_constant,
            source_term=source_term,
            initial_value=initial_value,
        )

    def _outlet_fractions(self):
        return ()
