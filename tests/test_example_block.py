import unittest

import numpy as np
from pathsim import Connection, Simulation
from pathsim.blocks import Constant
from pathsim.solvers import SSPRK22

from pathsim_toolbox import FirstOrderLag


class TestFirstOrderLagInit(unittest.TestCase):
    def test_port_labels(self):
        self.assertEqual(FirstOrderLag.input_port_labels["u"], 0)
        self.assertEqual(FirstOrderLag.output_port_labels["y"], 0)

    def test_default_init(self):
        blk = FirstOrderLag()
        self.assertAlmostEqual(blk.initial_value[0], 0.0)
        self.assertAlmostEqual(blk.tau, 1.0)

    def test_custom_init(self):
        blk = FirstOrderLag(tau=2.5, y0=1.0)
        self.assertAlmostEqual(blk.tau, 2.5)
        self.assertAlmostEqual(blk.initial_value[0], 1.0)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            FirstOrderLag(tau=0.0)
        with self.assertRaises(ValueError):
            FirstOrderLag(tau=-1.0)


class TestFirstOrderLagSimulation(unittest.TestCase):
    """Physics tests run via a full PathSim Simulation."""

    def _run(self, blk, t_end, dt=0.01):
        """Helper: wire the block to a constant source and run."""
        src = Constant(blk.inputs[0])
        sim = Simulation(
            blocks=[src, blk],
            connections=[Connection(src, blk[0])],
            dt=dt,
            Solver=SSPRK22,
        )
        sim.run(t_end)
        return blk.outputs[0]

    def test_steady_state_tracks_input(self):
        """After many time constants the output approaches the input."""
        blk = FirstOrderLag(tau=1.0, y0=0.0)
        blk.inputs[0] = 5.0
        y = self._run(blk, t_end=20.0)
        self.assertAlmostEqual(y, 5.0, places=2)

    def test_input_equal_to_init_stays_constant(self):
        """Input equal to the initial output → nothing changes."""
        blk = FirstOrderLag(tau=1.0, y0=3.0)
        blk.inputs[0] = 3.0
        y = self._run(blk, t_end=10.0)
        self.assertAlmostEqual(y, 3.0, places=4)

    def test_step_response_at_one_tau(self):
        """Step response reaches ~63.2 % of the final value after one tau."""
        tau = 2.0
        blk = FirstOrderLag(tau=tau, y0=0.0)
        blk.inputs[0] = 1.0
        y = self._run(blk, t_end=tau)
        self.assertAlmostEqual(y, 1.0 - np.exp(-1.0), places=2)


if __name__ == "__main__":
    unittest.main()
