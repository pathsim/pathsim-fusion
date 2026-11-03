#########################################################################################
##
##                                      Bubbler Block
##
#########################################################################################

# IMPORTS ===============================================================================

import numbers
from typing import ClassVar

import numpy as np
from pathsim.blocks.dynsys import DynamicalSystem
from pathsim.events.schedule import ScheduleList

# BLOCK DEFINITIONS =====================================================================


class Bubbler4(DynamicalSystem):
    """Tritium bubbling system with sequential vial collection stages.

    This block models a tritium collection system used in fusion reactor blanket
    purge gas processing. The system bubbles tritium-containing gas through a series
    of liquid-filled vials to capture and concentrate tritium for measurement and
    inventory tracking.


    Physical Description
    --------------------
    The bubbler consists of two parallel processing chains:

    **Soluble Chain (Vials 1-2):**
    Tritium already in soluble forms (HTO, HT) flows sequentially through
    vials 1 and 2. Each vial has a collection efficiency :math:`\\eta_{vial}`,
    representing the fraction of tritium that dissolves into the liquid phase
    and is retained.

    **Insoluble Chain (Vials 3-4):**
    Tritium in insoluble forms (T₂, organically bound) first undergoes catalytic
    conversion to soluble forms with efficiency :math:`\\alpha_{conv}`. The
    converted tritium, along with uncaptured soluble tritium from the first chain,
    then flows through vials 3 and 4 with the same collection efficiency.


    Mathematical Formulation
    -------------------------
    The system is governed by the following differential equations for the
    vial inventories :math:`x_i`:

    .. math::

        \\begin{align}
            \\frac{dx_1}{dt} &= \\eta_{vial} \\cdot u_{sol} \\\\
            \\frac{dx_2}{dt} &= \\eta_{vial} \\cdot (1-\\eta_{vial}) \\cdot u_{sol} \\\\
            \\frac{dx_3}{dt} &= \\eta_{vial} \\cdot [\\alpha_{conv} \\cdot u_{insol} + (1-\\eta_{vial})^2 \\cdot u_{sol}] \\\\
            \\frac{dx_4}{dt} &= \\eta_{vial} \\cdot (1-\\eta_{vial}) \\cdot [\\alpha_{conv} \\cdot u_{insol} + (1-\\eta_{vial})^2 \\cdot u_{sol}]
        \\end{align}


    The sample output represents uncaptured tritium exiting the system:

    .. math::

        y_{sample} = (1-\\alpha_{conv}) \\cdot u_{insol} + (1-\\eta_{vial})^2 \\cdot [\\alpha_{conv} \\cdot u_{insol} + (1-\\eta_{vial})^2 \\cdot u_{sol}]


    Where:
        - :math:`u_{sol}` = soluble tritium input flow rate
        - :math:`u_{insol}` = insoluble tritium input flow rate
        - :math:`\\eta_{vial}` = vial collection efficiency
        - :math:`\\alpha_{conv}` = conversion efficiency from insoluble to soluble
        - :math:`x_i` = tritium inventory in vial i

    Parameters
    ----------
    conversion_efficiency : float
       Conversion efficiency from insoluble to soluble forms (:math:`\\alpha_{conv}`),
       between 0 and 1.
    vial_efficiency : float
       Collection efficiency of each vial (:math:`\\eta_{vial}`), between 0 and 1.
    replacement_times : float | list[float] | list[list[float]] | None
        Times at which each vial is replaced with a fresh one. If None, no
        replacement events are created. If a single value is provided, it is
        used for all vials. If a single list of floats is provided, it will be
        used for all vials. If a list of four lists is provided, each sublist
        corresponds to the replacement times for each vial.

    Inputs
    ------
    sample_in_soluble (0) : soluble tritium flow rate into the bubbler
    sample_in_insoluble (1) : insoluble tritium flow rate into the bubbler

    Outputs
    -------
    vial1 (0) : tritium inventory in vial 1
    vial2 (1) : tritium inventory in vial 2
    vial3 (2) : tritium inventory in vial 3
    vial4 (3) : tritium inventory in vial 4
    sample_out (4) : uncaptured tritium flow rate leaving the bubbler

    Note
    ----
    Vial replacement is modeled as instantaneous reset events that set the
    corresponding vial inventory to zero, simulating the physical replacement
    of a full vial with an empty one.
    """

    input_port_labels: ClassVar[dict[str, int]] = {
        "sample_in_soluble": 0,
        "sample_in_insoluble": 1,
    }
    output_port_labels: ClassVar[dict[str, int]] = {
        "vial1": 0,
        "vial2": 1,
        "vial3": 2,
        "vial4": 3,
        "sample_out": 4,
    }

    def __init__(
        self,
        conversion_efficiency=0.9,
        vial_efficiency=0.9,
        replacement_times=None,
    ):

        # input validation
        if not 0 <= conversion_efficiency <= 1:
            raise ValueError(
                "'conversion_efficiency' must be between 0 and 1 "
                f"but is {conversion_efficiency}"
            )
        if not 0 <= vial_efficiency <= 1:
            raise ValueError(
                f"'vial_efficiency' must be between 0 and 1 but is {vial_efficiency}"
            )

        # bubbler parameters
        self.replacement_times = replacement_times
        self.vial_efficiency = vial_efficiency
        self.conversion_efficiency = conversion_efficiency

        # dynamical component, ode rhs
        def _fn_d(x, u, t):

            # short
            ve = self.vial_efficiency
            ce = self.conversion_efficiency

            # unpack inputs
            sol, ins = u

            # compute vial content change rates
            dv1 = ve * sol
            dv2 = dv1 * (1 - ve)
            dv3 = ve * (ce * ins + (1 - ve) ** 2 * sol)
            dv4 = dv3 * (1 - ve)

            return np.array([dv1, dv2, dv3, dv4])

        # algebraic output component
        def _fn_a(x, u, t):

            # short
            ve = self.vial_efficiency
            ce = self.conversion_efficiency

            # unpack inputs
            sol, ins = u

            sample_out = (1 - ce) * ins + (1 - ve) ** 2 * (
                ce * ins + (1 - ve) ** 2 * sol
            )

            return np.hstack([x, sample_out])

        # initialization just like `DynamicalSystem` block
        super().__init__(
            func_dyn=_fn_d,
            func_alg=_fn_a,
            initial_value=np.zeros(4),
        )

        # create internal vial reset events
        self._create_reset_events()

    def _create_reset_event_vial(self, i, reset_times):
        """Define event action function and return a `ScheduleList` event
        per vial `i` that triggers at predefined `reset_times`.
        """

        def reset_vial_i(_):
            # get the full engine state
            x = self.engine.get()
            # set index 'i' to zero
            x[i] = 0.0
            # set the full engine state
            self.engine.set(x)

        return ScheduleList(times_evt=reset_times, func_act=reset_vial_i)

    def _create_reset_events(self):
        """Create reset events for all vials based on the replacement times.

        Raises
        ------
        ValueError : If reset_times is not valid.
        """

        replacement_times = self.replacement_times
        self.events = []

        # no replacement
        if replacement_times is None:
            return

        # a single time is used for all vials (numbers.Real also covers numpy scalars)
        if isinstance(replacement_times, numbers.Real):
            replacement_times = [[replacement_times]] * 4

        # a flat list of times is used for all vials
        elif isinstance(replacement_times, (list, tuple)) and all(
            isinstance(t, numbers.Real) for t in replacement_times
        ):
            replacement_times = [list(replacement_times)] * 4

        elif isinstance(replacement_times, np.ndarray) and replacement_times.ndim == 1:
            replacement_times = [replacement_times.tolist()] * 4

        # otherwise one list of times per vial
        elif len(replacement_times) != 4:
            raise ValueError(
                "replacement_times must be a single value or a list with the same length as the number of vials"
            )

        # create the internal events
        self.events = [
            self._create_reset_event_vial(i, ts)
            for i, ts in enumerate(replacement_times)
        ]
