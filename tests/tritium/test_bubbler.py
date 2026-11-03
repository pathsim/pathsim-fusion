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
from pathsim.solvers import EUF

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


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
