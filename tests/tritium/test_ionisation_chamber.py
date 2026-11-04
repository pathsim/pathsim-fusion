########################################################################################
##
##                                  TESTS FOR
##                     'tritium.ionisation_chamber.py'
##
########################################################################################

# IMPORTS ==============================================================================

import math
import unittest

from pathsim import Connection, Simulation
from pathsim.blocks import Constant

from pathsim_fusion.tritium import IonisationChamber

# TESTS ================================================================================


class TestIonisationChamberInit(unittest.TestCase):
    """Test construction and parameter validation of the IonisationChamber block."""

    def test_init_default(self):
        """Default construction works (PathView builds blocks from defaults)."""
        ic = IonisationChamber()
        self.assertEqual(ic.detection_efficiency, 1.0)
        self.assertIsNone(ic.detection_threshold)
        self.assertIsNone(ic.func_efficiency)

    def test_info_defaults(self):
        """Block.info() reports a concrete default for the efficiency."""
        params = IonisationChamber.info()["parameters"]
        self.assertEqual(params["detection_efficiency"]["default"], 1.0)
        self.assertIn("func_efficiency", params)

    def test_init_constant_efficiency(self):
        """Test initialization with constant detection efficiency."""
        ic = IonisationChamber(detection_efficiency=0.8)
        self.assertEqual(ic.detection_efficiency, 0.8)
        self.assertIsNone(ic.detection_threshold)

    def test_init_threshold(self):
        """Test initialization with detection threshold."""
        ic = IonisationChamber(detection_threshold=10.0)
        self.assertEqual(ic.detection_threshold, 10.0)
        self.assertEqual(ic.detection_efficiency, 1.0)

    def test_init_func_efficiency(self):
        """Test initialization with an efficiency function."""
        eff = lambda c: min(c / 100.0, 1.0)
        ic = IonisationChamber(func_efficiency=eff)
        self.assertIs(ic.func_efficiency, eff)

    def test_init_callable_efficiency_deprecated(self):
        """A callable 'detection_efficiency' still works but warns."""
        eff = lambda c: min(c / 100.0, 1.0)
        with self.assertWarns(DeprecationWarning):
            ic = IonisationChamber(detection_efficiency=eff)
        self.assertIs(ic.func_efficiency, eff)
        self.assertEqual(ic.detection_efficiency, 1.0)

    def test_init_validation(self):
        """Invalid parameters raise ValueError."""
        eff = lambda c: 1.0
        for kwargs in [
            {"detection_efficiency": -0.1},
            {"detection_threshold": -1.0},
            {"func_efficiency": 0.5},
            {"detection_efficiency": eff, "func_efficiency": eff},
        ]:
            with (
                self.subTest(**{k: repr(v) for k, v in kwargs.items()}),
                self.assertRaises(ValueError),
            ):
                IonisationChamber(**kwargs)

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(IonisationChamber.input_port_labels["flux_in"], 0)
        self.assertEqual(IonisationChamber.input_port_labels["flow_rate"], 1)
        self.assertEqual(IonisationChamber.output_port_labels["flux_out"], 0)
        self.assertEqual(IonisationChamber.output_port_labels["signal"], 1)

    def test_len(self):
        """Algebraic block with direct passthrough."""
        self.assertEqual(len(IonisationChamber()), 1)


class TestIonisationChamberUpdate(unittest.TestCase):
    """Test the outputs of the IonisationChamber block."""

    def _signal(self, ic, flux_in, flow_rate):
        """Helper: set inputs, update, and return (flux_out, signal)."""
        ic.inputs[0] = flux_in
        ic.inputs[1] = flow_rate
        ic.update(None)
        return ic.outputs[0], ic.outputs[1]

    def test_passthrough(self):
        """Sample flux passes through unchanged."""
        flux_out, _ = self._signal(
            IonisationChamber(detection_efficiency=0.5), 100.0, 10.0
        )
        self.assertAlmostEqual(flux_out, 100.0)

    def test_signal_constant_efficiency(self):
        """Signal = efficiency * concentration."""
        # concentration = 200 / 10 = 20
        _, signal = self._signal(
            IonisationChamber(detection_efficiency=0.8), 200.0, 10.0
        )
        self.assertAlmostEqual(signal, 0.8 * 20.0)

    def test_signal_threshold_above(self):
        """Above threshold, signal = concentration."""
        # concentration = 10 > 5
        _, signal = self._signal(
            IonisationChamber(detection_threshold=5.0), 100.0, 10.0
        )
        self.assertAlmostEqual(signal, 10.0)

    def test_signal_threshold_at(self):
        """Exactly at the threshold the chamber detects."""
        _, signal = self._signal(
            IonisationChamber(detection_threshold=10.0), 100.0, 10.0
        )
        self.assertAlmostEqual(signal, 10.0)

    def test_signal_threshold_below(self):
        """Below threshold, signal = 0."""
        # concentration = 10 < 50
        _, signal = self._signal(
            IonisationChamber(detection_threshold=50.0), 100.0, 10.0
        )
        self.assertAlmostEqual(signal, 0.0)

    def test_signal_threshold_with_efficiency(self):
        """Above threshold the constant efficiency applies."""
        ic = IonisationChamber(detection_efficiency=0.5, detection_threshold=5.0)
        _, signal = self._signal(ic, 100.0, 10.0)
        self.assertAlmostEqual(signal, 0.5 * 10.0)

    def test_signal_func_efficiency(self):
        """Efficiency function applied to concentration."""
        # linear ramp: efficiency(50) = 0.5, signal = 0.5 * 50 = 25
        ic = IonisationChamber(func_efficiency=lambda c: min(c / 100.0, 1.0))
        _, signal = self._signal(ic, 500.0, 10.0)
        self.assertAlmostEqual(signal, 25.0)

    def test_signal_func_efficiency_overrides_constant(self):
        """'func_efficiency' takes precedence over 'detection_efficiency'."""
        ic = IonisationChamber(detection_efficiency=0.1, func_efficiency=lambda c: 0.9)
        _, signal = self._signal(ic, 100.0, 10.0)
        self.assertAlmostEqual(signal, 0.9 * 10.0)

    def test_signal_func_efficiency_below_threshold(self):
        """The threshold also gates an efficiency function."""
        ic = IonisationChamber(detection_threshold=50.0, func_efficiency=lambda c: 1.0)
        _, signal = self._signal(ic, 100.0, 10.0)
        self.assertAlmostEqual(signal, 0.0)

    def test_signal_deprecated_callable(self):
        """Deprecated callable 'detection_efficiency' gives the same signal."""
        with self.assertWarns(DeprecationWarning):
            ic = IonisationChamber(detection_efficiency=lambda c: min(c / 100.0, 1.0))
        _, signal = self._signal(ic, 500.0, 10.0)
        self.assertAlmostEqual(signal, 25.0)

    def test_runtime_parameter_change(self):
        """Changing parameters after construction affects the output."""
        ic = IonisationChamber(detection_threshold=5.0)
        ic.detection_threshold = 50.0
        ic.detection_efficiency = 0.5
        _, signal = self._signal(ic, 100.0, 10.0)
        self.assertAlmostEqual(signal, 0.0)

        ic.detection_threshold = None
        _, signal = self._signal(ic, 100.0, 10.0)
        self.assertAlmostEqual(signal, 0.5 * 10.0)

    def test_zero_flow_rate(self):
        """Zero flow rate should not crash, signal = 0."""
        flux_out, signal = self._signal(
            IonisationChamber(detection_efficiency=1.0), 100.0, 0.0
        )
        self.assertAlmostEqual(flux_out, 100.0)
        self.assertAlmostEqual(signal, 0.0)

    def test_negative_flow_rate(self):
        """Negative flow rate is treated like zero flow, signal = 0."""
        _, signal = self._signal(IonisationChamber(), 100.0, -10.0)
        self.assertAlmostEqual(signal, 0.0)

    def test_nan_flow_rate(self):
        """A NaN flow rate gives a NaN signal instead of a plausible zero, so an
        upstream error is not hidden. The flux still passes through unchanged."""
        flux_out, signal = self._signal(IonisationChamber(), 100.0, float("nan"))
        self.assertAlmostEqual(flux_out, 100.0)
        self.assertTrue(math.isnan(signal))


class TestIonisationChamberSimulation(unittest.TestCase):
    """Test the IonisationChamber block inside a full Simulation."""

    def test_constant_inputs(self):
        """Constant flux and flow give the analytic signal and passthrough."""
        flux = Constant(200.0)
        flow = Constant(10.0)
        ic = IonisationChamber(detection_efficiency=0.8)

        sim = Simulation(
            blocks=[flux, flow, ic],
            connections=[
                Connection(flux, ic["flux_in"]),
                Connection(flow, ic["flow_rate"]),
            ],
            dt=0.1,
            log=False,
        )
        sim.run(1.0)

        self.assertAlmostEqual(ic.outputs[0], 200.0)
        self.assertAlmostEqual(ic.outputs[1], 0.8 * 20.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
