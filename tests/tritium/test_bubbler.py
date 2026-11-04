########################################################################################
##
##                                  TESTS FOR
##                             'tritium.bubbler.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

import numpy as np
from pathsim import Connection, Simulation
from pathsim.blocks import Constant
from pathsim.events.schedule import ScheduleList
from pathsim.solvers import ESDIRK43, EUF, RKCK54

from pathsim_fusion.tritium import Bubbler4

# TESTS ================================================================================


class TestTritiumBubbler4(unittest.TestCase):
    """
    Test the implementation of the 'Bubbler4' block class from the fusion toolbox.
    The block inherits from `ODE` and models a 4-vial tritium collection system.
    """

    def test_init(self):
        """Test initialization with various parameter combinations"""

        # Default initialization
        B = Bubbler4()
        self.assertEqual(B.conversion_efficiency, 0.9)
        self.assertEqual(B.vial_efficiency, 0.9)
        self.assertEqual(B.replacement_times, None)
        self.assertEqual(len(B.events), 0)  # No events when replacement_times is None

        # Specific initialization
        B = Bubbler4(
            conversion_efficiency=0.8, vial_efficiency=0.7, replacement_times=[100, 200]
        )
        self.assertEqual(B.conversion_efficiency, 0.8)
        self.assertEqual(B.vial_efficiency, 0.7)
        self.assertEqual(B.replacement_times, [100, 200])

        # Set solver to check internal solver instance
        B.set_solver(EUF, parent=None)
        self.assertTrue(B.engine)
        np.testing.assert_array_equal(B.engine.initial_value, np.zeros(4))

    def test_init_replacement_times(self):
        """Test different replacement_times configurations"""

        # Single list - should replicate for all vials
        B = Bubbler4(replacement_times=[100, 200, 300])
        B.set_solver(EUF, parent=None)
        self.assertEqual(len(B.events), 4)

        # List of lists - one per vial
        times = [[100, 200], [150, 250], [120, 220], [180, 280]]
        B = Bubbler4(replacement_times=times)
        B.set_solver(EUF, parent=None)
        self.assertEqual(len(B.events), 4)
        for event in B.events:
            self.assertIsInstance(event, ScheduleList)

        # NumPy array
        B = Bubbler4(replacement_times=np.array([100, 200, 300]))
        B.set_solver(EUF, parent=None)
        self.assertEqual(len(B.events), 4)

        # Invalid case - wrong length
        with self.assertRaises(ValueError):
            B = Bubbler4(replacement_times=[[100], [200], [300]])  # Only 3 vials

    def test_update_outputs(self):
        """Test the update method and output calculations"""

        # Test with default parameters
        B = Bubbler4()
        B.set_solver(EUF, parent=None)

        # Set some test inputs
        B.inputs[0] = 10.0  # soluble input
        B.inputs[1] = 5.0  # insoluble input

        B.update(0.0)

        # Check that outputs are set (initial state is zero)
        self.assertEqual(B.outputs[0], 0.0)  # vial 1
        self.assertEqual(B.outputs[1], 0.0)  # vial 2
        self.assertEqual(B.outputs[2], 0.0)  # vial 3
        self.assertEqual(B.outputs[3], 0.0)  # vial 4

        # Calculate expected sample_out with default parameters (ve=0.9, ce=0.9)
        ve, ce = 0.9, 0.9
        sol, ins = 10.0, 5.0
        expected_sample_out = (1 - ce) * ins + (1 - ve) ** 2 * (
            ce * ins + (1 - ve) ** 2 * sol
        )
        self.assertAlmostEqual(B.outputs[4], expected_sample_out, places=10)

    def test_mass_conservation(self):
        """Test that mass is conserved in the system"""

        B = Bubbler4(conversion_efficiency=0.8, vial_efficiency=0.7)
        B.set_solver(EUF, parent=None)

        # Set test inputs
        sol_in, ins_in = 10.0, 5.0
        B.inputs[0] = sol_in
        B.inputs[1] = ins_in

        # Get rates and sample_out
        u = np.array([sol_in, ins_in])
        x = np.zeros(4)
        rates = B.func_dyn(x, u, 0.0)

        B.update(0.0)
        sample_out = B.outputs[4]

        # Total input rate
        total_in = sol_in + ins_in

        # Total accumulation rate in vials
        total_vial_rates = np.sum(rates)

        # Mass conservation: input = vial accumulation + sample output
        total_out = total_vial_rates + sample_out

        self.assertAlmostEqual(total_in, total_out, places=10)

    def test_vial_reset_functionality(self):
        """Test that vial reset events work correctly"""

        B = Bubbler4(replacement_times=[10.0])
        B.set_solver(EUF, parent=None)

        # Manually set some vial inventories
        x = np.array([1.0, 2.0, 3.0, 4.0])
        B.engine.set(x)

        # Test reset function for vial 0
        reset_event = B.events[0]
        reset_func = reset_event.func_act

        # Execute reset
        reset_func(None)

        # Check that vial 0 was reset but others remain
        x_after = B.engine.get()
        self.assertEqual(x_after[0], 0.0)
        self.assertEqual(x_after[1], 2.0)
        self.assertEqual(x_after[2], 3.0)
        self.assertEqual(x_after[3], 4.0)

        # Test reset function for vial 3
        reset_event = B.events[3]
        reset_func = reset_event.func_act

        # Execute reset
        reset_func(None)

        # Check that vial 0 was reset but others remain
        x_after = B.engine.get()
        self.assertEqual(x_after[0], 0.0)
        self.assertEqual(x_after[1], 2.0)
        self.assertEqual(x_after[2], 3.0)
        self.assertEqual(x_after[3], 0.0)


class TestTritiumBubbler4Fixes(unittest.TestCase):
    """
    Tests added when moving 'Bubbler4' from pathsim-chem: scalar replacement
    times, parameter validation, port labels and full-simulation behaviour.
    """

    def test_scalar_replacement_time(self):
        """A single replacement time creates one event per vial (all four vials)."""
        for t in [10.0, 10, np.float64(10.0), np.int64(10)]:
            with self.subTest(t=repr(t)):
                B = Bubbler4(replacement_times=t)
                self.assertEqual(len(B.events), 4)

    def test_tuple_replacement_times(self):
        """A flat tuple of times is used for all vials, like a flat list."""
        B = Bubbler4(replacement_times=(100.0, 200.0))
        self.assertEqual(len(B.events), 4)

    def test_invalid_efficiencies(self):
        """Efficiencies outside [0, 1] raise ValueError."""
        for kwargs in [
            {"vial_efficiency": 1.5},
            {"vial_efficiency": -0.1},
            {"conversion_efficiency": 1.1},
            {"conversion_efficiency": -0.5},
        ]:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                Bubbler4(**kwargs)

    def test_port_labels(self):
        """Port labels are contiguous and in the documented order."""
        self.assertEqual(
            Bubbler4.input_port_labels,
            {"sample_in_soluble": 0, "sample_in_insoluble": 1},
        )
        self.assertEqual(list(Bubbler4.output_port_labels.values()), [0, 1, 2, 3, 4])
        self.assertEqual(Bubbler4.output_port_labels["sample_out"], 4)

    def test_len(self):
        """The sample output depends directly on the inputs (passthrough)."""
        self.assertEqual(len(Bubbler4()), 1)

    def _run(self, replacement_times, t_end, ve=0.8, ce=0.7, sol=1.0, ins=2.0):
        """Helper: run a Bubbler4 with constant inputs through a full Simulation."""
        src_sol, src_ins = Constant(sol), Constant(ins)
        B = Bubbler4(
            vial_efficiency=ve,
            conversion_efficiency=ce,
            replacement_times=replacement_times,
        )
        sim = Simulation(
            blocks=[src_sol, src_ins, B],
            connections=[
                Connection(src_sol, B["sample_in_soluble"]),
                Connection(src_ins, B["sample_in_insoluble"]),
            ],
            dt=0.01,
            log=False,
        )
        sim.run(t_end)
        return B

    def test_simulation_conservation(self):
        """Vial inventories plus integrated sample outflow equal the total input."""
        t_end = 10.0
        B = self._run(None, t_end)
        vials = sum(B.outputs[i] for i in range(4))
        self.assertAlmostEqual(
            vials + B.outputs[4] * t_end, (1.0 + 2.0) * t_end, places=8
        )

    def test_simulation_scalar_replacement(self):
        """With a scalar replacement time all vials restart from zero at that time."""
        ve, ce, sol, ins = 0.8, 0.7, 1.0, 2.0
        B = self._run(5.0, t_end=7.0, ve=ve, ce=ce, sol=sol, ins=ins)

        # constant inputs -> each vial grows linearly for the 2 s since replacement
        rate_3 = ve * (ce * ins + (1 - ve) ** 2 * sol)
        expected = 2.0 * np.array(
            [ve * sol, ve * (1 - ve) * sol, rate_3, rate_3 * (1 - ve)]
        )
        actual = np.array([B.outputs[i] for i in range(4)])
        np.testing.assert_allclose(actual, expected, rtol=1e-6)


class TestTritiumBubbler4Physics(unittest.TestCase):
    """
    Physics and event checks for 'Bubbler4' against hand-derived values: every
    output with distinct inputs, efficiency limits, per-vial replacement timing,
    solver independence and independence of two instances.
    """

    def _expected_rates(self, ve, ce, sol, ins):
        """Hand-derived vial accumulation rates and sample outflow."""
        a = ce * ins + (1 - ve) ** 2 * sol
        rates = np.array([ve * sol, ve * (1 - ve) * sol, ve * a, ve * (1 - ve) * a])
        return rates, (1 - ce) * ins + (1 - ve) ** 2 * a

    def _run(self, B, t_end, sol, ins, Solver=EUF):
        """Run 'B' with constant inputs and return all five outputs."""
        src_sol, src_ins = Constant(sol), Constant(ins)
        sim = Simulation(
            blocks=[src_sol, src_ins, B],
            connections=[
                Connection(src_sol, B["sample_in_soluble"]),
                Connection(src_ins, B["sample_in_insoluble"]),
            ],
            dt=0.01,
            Solver=Solver,
            log=False,
        )
        sim.run(t_end)
        return np.array([B.outputs[i] for i in range(5)])

    def test_all_outputs_distinct_inputs(self):
        """Every output matches the hand calculation, with distinct values on each
        input and parameter so a swapped port or parameter shows up."""
        ve, ce, sol, ins, t_end = 0.6, 0.3, 3.0, 7.0, 2.0
        B = Bubbler4(vial_efficiency=ve, conversion_efficiency=ce)
        rates, sample_out = self._expected_rates(ve, ce, sol, ins)
        expected = np.hstack([rates * t_end, sample_out])
        np.testing.assert_allclose(self._run(B, t_end, sol, ins), expected, rtol=1e-10)

    def test_efficiency_limits(self):
        """At the efficiency limits tritium ends up exactly where expected, and
        the balance closes in each case."""
        sol, ins = 3.0, 7.0
        cases = {
            # (ve, ce): (vial1, vial2, vial3, vial4, sample_out)
            (0.0, 0.0): (0.0, 0.0, 0.0, 0.0, 10.0),
            (1.0, 1.0): (3.0, 0.0, 7.0, 0.0, 0.0),
            (1.0, 0.0): (3.0, 0.0, 0.0, 0.0, 7.0),
            (0.0, 1.0): (0.0, 0.0, 0.0, 0.0, 10.0),
        }
        for (ve, ce), expected in cases.items():
            with self.subTest(vial_efficiency=ve, conversion_efficiency=ce):
                B = Bubbler4(vial_efficiency=ve, conversion_efficiency=ce)
                out = self._run(B, 1.0, sol, ins)
                np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_per_vial_replacement_timing(self):
        """With one replacement time per vial, each vial restarts from zero at its
        own time and the other vials are untouched."""
        ve, t_end = 0.9, 5.0
        B = Bubbler4(
            vial_efficiency=ve,
            conversion_efficiency=0.5,
            replacement_times=[[1.0], [2.0], [3.0], [4.0]],
        )
        rates, _ = self._expected_rates(ve, 0.5, 1.0, 0.0)
        since_reset = t_end - np.array([1.0, 2.0, 3.0, 4.0])
        out = self._run(B, t_end, 1.0, 0.0)
        np.testing.assert_allclose(out[:4], rates * since_reset, rtol=1e-8)

    def test_replacement_at_end_of_run(self):
        """A replacement scheduled at the final time empties the vials."""
        B = Bubbler4(replacement_times=5.0)
        out = self._run(B, 5.0, 1.0, 2.0)
        np.testing.assert_allclose(out[:4], np.zeros(4), atol=1e-12)

    def test_unsorted_replacement_times(self):
        """Replacement times out of order are rejected rather than silently
        reordered."""
        with self.assertRaises(ValueError):
            Bubbler4(replacement_times=[3.0, 1.0])

    def test_solver_independence(self):
        """Explicit and implicit solvers give the same inventories (the rates are
        constant between replacements, so every solver should be exact)."""
        results = []
        for Solver in [EUF, RKCK54, ESDIRK43]:
            with self.subTest(Solver=Solver.__name__):
                B = Bubbler4(replacement_times=[1.5, 3.2])
                results.append(self._run(B, 4.0, 3.0, 7.0, Solver=Solver))
        for out in results[1:]:
            np.testing.assert_allclose(out, results[0], rtol=1e-8)

    def test_instances_independent(self):
        """Two bubblers in one simulation keep separate states and events."""
        B1 = Bubbler4(replacement_times=1.0)
        B2 = Bubbler4()
        src_sol, src_ins = Constant(1.0), Constant(0.0)
        sim = Simulation(
            blocks=[src_sol, src_ins, B1, B2],
            connections=[
                Connection(src_sol, B1[0], B2[0]),
                Connection(src_ins, B1[1], B2[1]),
            ],
            dt=0.01,
            log=False,
        )
        sim.run(2.0)
        self.assertAlmostEqual(B1.outputs[0], 0.9 * 1.0, places=10)
        self.assertAlmostEqual(B2.outputs[0], 0.9 * 2.0, places=10)

    def test_reset_and_rerun(self):
        """After 'sim.reset()' a second run reproduces the first, events included."""
        B = Bubbler4(replacement_times=2.0)
        src_sol, src_ins = Constant(1.0), Constant(2.0)
        sim = Simulation(
            blocks=[src_sol, src_ins, B],
            connections=[Connection(src_sol, B[0]), Connection(src_ins, B[1])],
            dt=0.01,
            log=False,
        )
        sim.run(3.0)
        first = np.array([B.outputs[i] for i in range(5)])
        sim.reset()
        sim.run(3.0)
        second = np.array([B.outputs[i] for i in range(5)])
        np.testing.assert_allclose(second, first, rtol=1e-12)


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
