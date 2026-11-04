#########################################################################################
##
##                            Ionisation Chamber Block
##
##        Moved from pathsim-chem (src/pathsim_chem/tritium/ionisation_chamber.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import warnings
from typing import ClassVar

from pathsim.blocks.function import Function

# BLOCKS ================================================================================


class IonisationChamber(Function):
    """Flow-through ionisation chamber for tritium detection.

    Algebraic block that models a flow-through ionisation chamber. The sample
    passes through unchanged while the chamber produces a signal proportional
    to the tritium concentration, scaled by a detection efficiency.

    Mathematical Formulation
    -------------------------
    The chamber receives a tritium flux and flow rate, computes the
    concentration, and applies the detection efficiency:

    .. math::

        c = \\frac{\\Phi_{in}}{\\dot{V}}

    .. math::

        \\text{signal} = \\varepsilon(c) \\cdot c

    .. math::

        \\Phi_{out} = \\Phi_{in}

    The efficiency is the constant ``detection_efficiency``, or
    ``func_efficiency(c)`` if given. An optional ``detection_threshold``
    :math:`c_{min}` acts as a detection limit:

    .. math::

        \\varepsilon(c) =
        \\begin{cases}
        \\varepsilon_0 \\text{ or } f(c) & , c \\geq c_{min} \\\\
        0                               & , c < c_{min}
        \\end{cases}

    For a non-positive flow rate the concentration is taken as zero.

    Parameters
    ----------
    detection_efficiency : float
        Constant detection efficiency [-]. Must be non-negative. Unused if
        ``func_efficiency`` is given. Default 1.0.
    detection_threshold : float, None
        Detection limit on the concentration, in the units of
        ``flux_in / flow_rate``. Below it the signal is zero. Must be
        non-negative. Default None (no threshold).
    func_efficiency : callable, None
        Concentration-dependent efficiency ``f(c) -> float``. Replaces
        ``detection_efficiency`` when given. Default None.

    Inputs
    ------
    flux_in (0) : tritium flux into the chamber [e.g. Bq/s or mol/s]
    flow_rate (1) : volumetric flow rate through the chamber [m³/s]

    Outputs
    -------
    flux_out (0) : tritium flux out of the chamber, equal to ``flux_in``
    signal (1) : chamber signal, efficiency times concentration
    """

    input_port_labels: ClassVar[dict[str, int]] = {
        "flux_in": 0,
        "flow_rate": 1,
    }

    output_port_labels: ClassVar[dict[str, int]] = {
        "flux_out": 0,
        "signal": 1,
    }

    def __init__(
        self, detection_efficiency=1.0, detection_threshold=None, func_efficiency=None
    ):

        # backwards compatibility: pathsim-chem accepted a callable here
        if callable(detection_efficiency):
            if func_efficiency is not None:
                raise ValueError(
                    "Pass the efficiency function as 'func_efficiency' only, "
                    "not also as 'detection_efficiency'"
                )
            warnings.warn(
                "Passing a callable as 'detection_efficiency' is deprecated, "
                "use 'func_efficiency' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            func_efficiency = detection_efficiency
            detection_efficiency = 1.0

        # input validation
        if detection_efficiency < 0:
            raise ValueError(
                f"'detection_efficiency' must be non-negative but is {detection_efficiency}"
            )
        if detection_threshold is not None and detection_threshold < 0:
            raise ValueError(
                f"'detection_threshold' must be non-negative but is {detection_threshold}"
            )
        if func_efficiency is not None and not callable(func_efficiency):
            raise ValueError(
                f"'func_efficiency' must be callable but is {func_efficiency}"
            )

        # store parameters
        self.detection_efficiency = detection_efficiency
        self.detection_threshold = detection_threshold
        self.func_efficiency = func_efficiency

        super().__init__(func=self._eval)

    def _efficiency(self, concentration):
        """Detection efficiency at a given concentration."""

        # below the detection limit nothing is detected
        if (
            self.detection_threshold is not None
            and concentration < self.detection_threshold
        ):
            return 0.0

        if self.func_efficiency is not None:
            return self.func_efficiency(concentration)

        return self.detection_efficiency

    def _eval(self, flux_in, flow_rate):
        concentration = flux_in / flow_rate if flow_rate > 0 else 0.0

        signal = self._efficiency(concentration) * concentration
        flux_out = flux_in

        return (flux_out, signal)
