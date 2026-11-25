########################################################################################
##
##                                  TESTS FOR
##                          'tritium.fuel_cycle.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

import numpy as np
from pathsim import Connection, Simulation
from pathsim.blocks import Adder, Constant, Integrator, Scope, Splitter
from pathsim.solvers import ESDIRK43, EUF, RKCK54

import pathsim_fusion
from pathsim_fusion.tritium import FuelCycleComponent

# TESTS ================================================================================


class TestFuelCycleComponentInit(unittest.TestCase):
    """Test construction and parameter validation of the FuelCycleComponent block."""

    def test_init_default(self):
        """Default construction works (PathView builds blocks from defaults)."""
        blk = FuelCycleComponent()
        self.assertEqual(blk.tau, 1.0)
        self.assertEqual(blk.loss_fraction, 0.0)
        self.assertEqual(blk.decay_constant, 0.0)
        self.assertEqual(blk.source_term, 0.0)
        self.assertEqual(blk.initial_value, 0.0)

    def test_init_custom(self):
        """Every parameter is stored under its own name."""
        blk = FuelCycleComponent(
            tau=4.0,
            loss_fraction=0.25,
            decay_constant=0.5,
            source_term=7.0,
            initial_value=8.0,
        )
        self.assertEqual(blk.tau, 4.0)
        self.assertEqual(blk.loss_fraction, 0.25)
        self.assertEqual(blk.decay_constant, 0.5)
        self.assertEqual(blk.source_term, 7.0)
        self.assertEqual(blk.initial_value, 8.0)

    def test_info_defaults(self):
        """Block.info() reports a concrete default for every parameter."""
        params = FuelCycleComponent.info()["parameters"]
        expected = {
            "tau": 1.0,
            "loss_fraction": 0.0,
            "decay_constant": 0.0,
            "source_term": 0.0,
            "initial_value": 0.0,
        }
        self.assertEqual({k: v["default"] for k, v in params.items()}, expected)

    def test_init_validation(self):
        """Invalid parameters raise ValueError that names the parameter."""
        nan = float("nan")
        for name, value in [
            ("tau", 0.0),
            ("tau", -1.0),
            ("tau", nan),
            ("loss_fraction", -0.1),
            ("loss_fraction", nan),
            ("decay_constant", -1e-9),
            ("decay_constant", nan),
            ("initial_value", -1.0),
            ("initial_value", nan),
            ("initial_value", [1.0, 2.0]),
        ]:
            with (
                self.subTest(name=name, value=value),
                self.assertRaisesRegex(ValueError, f"'{name}'"),
            ):
                FuelCycleComponent(**{name: value})

    def test_init_boundary_values(self):
        """Zero loss, zero decay, zero inventory and an infinite tau are valid."""
        FuelCycleComponent(loss_fraction=0.0, decay_constant=0.0, initial_value=0.0)
        FuelCycleComponent(tau=np.inf)
        FuelCycleComponent(tau=2, initial_value=np.float64(3.0))

    def test_exported(self):
        """The block is reachable from the package, where PathView looks for it."""
        self.assertIs(pathsim_fusion.FuelCycleComponent, FuelCycleComponent)
        self.assertIn("FuelCycleComponent", pathsim_fusion.__all__)


class TestFuelCycleComponentUnit(unittest.TestCase):
    """Test ports, rates and outputs of a bare FuelCycleComponent block."""

    def _make(self, **kwargs):
        """Block with distinct parameter values, ready to update."""
        params = {
            "tau": 4.0,
            "loss_fraction": 0.25,
            "decay_constant": 0.5,
            "source_term": 7.0,
            "initial_value": 8.0,
        }
        params.update(kwargs)
        blk = FuelCycleComponent(**params)
        blk.set_solver(EUF, parent=None)
        return blk

    def test_port_labels(self):
        """Inputs are variable, outputs are inventory, outflow and loss in that order."""
        self.assertIsNone(FuelCycleComponent.input_port_labels)
        self.assertEqual(
            FuelCycleComponent.output_port_labels,
            {"inventory": 0, "outflow": 1, "loss": 2},
        )

    def test_len(self):
        """No output depends directly on an input, before and after a solver is set."""
        self.assertEqual(len(FuelCycleComponent()), 0)
        self.assertEqual(len(self._make()), 0)

    def test_outputs(self):
        """Outputs are I, I/tau and eps*I/tau + lambda*I, whatever the inputs."""
        blk = self._make()
        blk.inputs[0] = 3.0
        blk.inputs[1] = 11.0
        blk.update(0.0)
        self.assertAlmostEqual(blk.outputs[0], 8.0, places=12)
        self.assertAlmostEqual(blk.outputs[1], 2.0, places=12)
        self.assertAlmostEqual(blk.outputs[2], 0.25 * 2.0 + 0.5 * 8.0, places=12)
        self.assertAlmostEqual(blk["inventory"].get_outputs()[0], 8.0, places=12)
        self.assertAlmostEqual(blk["outflow"].get_outputs()[0], 2.0, places=12)
        self.assertAlmostEqual(blk["loss"].get_outputs()[0], 4.5, places=12)

    def test_rate(self):
        """dI/dt = sum(u) + S - (1 + eps) I / tau - lambda I, with every inflow summed."""
        blk = self._make()
        x = np.array([8.0])
        u = np.array([3.0, 11.0, -2.0])
        expected = (3.0 + 11.0 - 2.0) + 7.0 - 1.25 * 8.0 / 4.0 - 0.5 * 8.0
        self.assertAlmostEqual(float(blk.op_dyn(x, u, 0.0)[0]), expected, places=12)

    def test_rate_equals_inflow_minus_outputs(self):
        """The inventory changes by the inflows and source minus outflow and loss."""
        blk = self._make()
        blk.inputs[0] = 3.0
        blk.inputs[1] = 11.0
        blk.update(0.0)
        rate = float(blk.op_dyn(np.array([8.0]), np.array([3.0, 11.0]), 0.0)[0])
        self.assertAlmostEqual(
            rate, 3.0 + 11.0 + 7.0 - blk.outputs[1] - blk.outputs[2], places=12
        )

    def test_no_input(self):
        """With nothing connected the block relaxes on its own."""
        blk = self._make(source_term=0.0)
        blk.update(0.0)
        rate = float(blk.op_dyn(np.array([8.0]), blk.inputs.to_array(), 0.0)[0])
        self.assertAlmostEqual(rate, -1.25 * 8.0 / 4.0 - 0.5 * 8.0, places=12)

    def test_jacobian(self):
        """The analytic jacobian matches a central difference of the rate."""
        blk = self._make()
        u = np.array([3.0, 11.0])
        for x0 in (0.0, 0.3, 8.0, 1e3):
            with self.subTest(x0=x0):
                h = 1e-4
                up = blk.op_dyn(np.array([x0 + h]), u, 0.0)[0]
                dn = blk.op_dyn(np.array([x0 - h]), u, 0.0)[0]
                jac = np.ravel(blk.op_dyn.jac_x(np.array([x0]), u, 0.0))[0]
                self.assertAlmostEqual(jac, (up - dn) / (2 * h), places=7)
                self.assertAlmostEqual(jac, -1.25 / 4.0 - 0.5, places=12)

    def test_infinite_tau(self):
        """An infinite tau gives a storage: no outflow, only decay is lost."""
        blk = self._make(tau=np.inf)
        blk.inputs[0] = 3.0
        blk.update(0.0)
        self.assertAlmostEqual(blk.outputs[0], 8.0, places=12)
        self.assertEqual(blk.outputs[1], 0.0)
        self.assertAlmostEqual(blk.outputs[2], 0.5 * 8.0, places=12)
        rate = float(blk.op_dyn(np.array([8.0]), np.array([3.0]), 0.0)[0])
        self.assertAlmostEqual(rate, 3.0 + 7.0 - 0.5 * 8.0, places=12)

    def test_parameter_edit(self):
        """Parameters changed after construction are used by the next update."""
        blk = self._make()
        blk.tau = 2.0
        blk.loss_fraction = 0.5
        blk.decay_constant = 0.125
        blk.source_term = 1.0
        blk.update(0.0)
        self.assertAlmostEqual(blk.outputs[1], 4.0, places=12)
        self.assertAlmostEqual(blk.outputs[2], 0.5 * 4.0 + 0.125 * 8.0, places=12)
        rate = float(blk.op_dyn(np.array([8.0]), np.array([0.0]), 0.0)[0])
        self.assertAlmostEqual(rate, 1.0 - 1.5 * 8.0 / 2.0 - 0.125 * 8.0, places=12)

    def test_instances_independent(self):
        """Two blocks do not share parameters or state."""
        a = self._make()
        b = self._make(tau=2.0, initial_value=1.0)
        a.update(0.0)
        b.update(0.0)
        self.assertAlmostEqual(a.outputs[1], 2.0, places=12)
        self.assertAlmostEqual(b.outputs[1], 0.5, places=12)


class TestFuelCycleComponentSimulation(unittest.TestCase):
    """Test FuelCycleComponent in a Simulation against analytic solutions."""

    def _run(self, blocks, connections, duration, Solver=RKCK54):
        """Run a simulation with tight tolerances."""
        sim = Simulation(
            blocks,
            connections,
            Solver=Solver,
            dt=0.01,
            tolerance_lte_rel=1e-8,
            tolerance_lte_abs=1e-10,
            log=False,
        )
        sim.run(duration)
        return sim

    def _step_response(self, Solver):
        """Constant feed into one component, returns time, outputs and the exact I(t)."""
        src = Constant(3.0)
        blk = FuelCycleComponent(
            tau=2.0,
            loss_fraction=0.5,
            decay_constant=0.25,
            source_term=1.0,
            initial_value=5.0,
        )
        sco = Scope()
        connections = [Connection(src, blk)]
        connections += [Connection(blk[i], sco[i]) for i in range(3)]
        self._run([src, blk, sco], connections, 5.0, Solver)
        time, data = sco.read()

        # dI/dt = 4 - k I with k = (1 + eps)/tau + lambda
        k = 1.5 / 2.0 + 0.25
        exact = 4.0 / k + (5.0 - 4.0 / k) * np.exp(-k * time)
        return time, data, exact

    def test_step_response(self):
        """A constant feed gives the first-order response, for both solver families."""
        for Solver in (RKCK54, ESDIRK43):
            with self.subTest(Solver=Solver.__name__):
                _, data, exact = self._step_response(Solver)
                np.testing.assert_allclose(data[0], exact, atol=1e-6)
                np.testing.assert_allclose(data[1], exact / 2.0, atol=1e-6)
                np.testing.assert_allclose(
                    data[2], 0.5 * exact / 2.0 + 0.25 * exact, atol=1e-6
                )

    def test_steady_state(self):
        """The inventory settles where the feed balances outflow and loss."""
        src = Constant(3.0)
        blk = FuelCycleComponent(tau=0.5, loss_fraction=0.5, decay_constant=1.0)
        self._run([src, blk], [Connection(src, blk)], 20.0)
        inventory, outflow, loss = blk.outputs.to_array()
        self.assertAlmostEqual(inventory, 3.0 / (1.5 / 0.5 + 1.0), places=6)
        self.assertAlmostEqual(outflow + loss, 3.0, places=6)

    def test_pure_decay(self):
        """A storage without inflow decays as I0 exp(-lambda t)."""
        blk = FuelCycleComponent(tau=np.inf, decay_constant=0.3, initial_value=2.0)
        sco = Scope()
        self._run([blk, sco], [Connection(blk["inventory"], sco)], 4.0)
        time, data = sco.read()
        np.testing.assert_allclose(data[0], 2.0 * np.exp(-0.3 * time), atol=1e-6)

    def test_multiple_inflows(self):
        """Inflows on several ports add up, also when one of them is negative."""
        srcs = [Constant(3.0), Constant(11.0), Constant(-2.0)]
        blk = FuelCycleComponent(tau=np.inf)
        connections = [Connection(src, blk[i]) for i, src in enumerate(srcs)]
        self._run([*srcs, blk], connections, 2.0)
        self.assertAlmostEqual(blk.outputs[0], 12.0 * 2.0, places=6)

    def test_reset(self):
        """Resetting the simulation restores the initial inventory."""
        src = Constant(3.0)
        blk = FuelCycleComponent(tau=2.0, initial_value=5.0)
        sim = self._run([src, blk], [Connection(src, blk)], 1.0)
        self.assertNotAlmostEqual(blk.outputs[0], 5.0, places=3)
        sim.reset()
        blk.update(0.0)
        self.assertAlmostEqual(blk.outputs[0], 5.0, places=12)

    def test_closed_loop_conservation(self):
        """Without loss and decay a recycle loop of three components conserves the total."""
        a = FuelCycleComponent(tau=1.0, initial_value=6.0)
        b = FuelCycleComponent(tau=0.3)
        c = FuelCycleComponent(tau=2.5)
        spl = Splitter([0.7, 0.3])
        sco = Scope()
        connections = [
            Connection(a["outflow"], spl),
            Connection(spl[0], b[0]),
            Connection(spl[1], c[0]),
            Connection(b["outflow"], c[1]),
            Connection(c["outflow"], a[0]),
        ]
        connections += [
            Connection(blk["inventory"], sco[i]) for i, blk in enumerate((a, b, c))
        ]
        self._run([a, b, c, spl, sco], connections, 10.0)
        _, data = sco.read()
        np.testing.assert_allclose(data.sum(axis=0), 6.0, atol=1e-6)
        self.assertGreater(data[1].max(), 0.1)
        self.assertGreater(data[2].max(), 0.1)

    def test_loss_balance(self):
        """Inventories plus the integrated loss port account for all tritium fed in."""
        src = Constant(3.0)
        a = FuelCycleComponent(tau=1.0, loss_fraction=0.2, decay_constant=0.1)
        b = FuelCycleComponent(tau=np.inf, decay_constant=0.4, initial_value=2.0)
        add = Adder()
        lost = Integrator(0.0)
        connections = [
            Connection(src, a),
            Connection(a["outflow"], b),
            Connection(a["loss"], add[0]),
            Connection(b["loss"], add[1]),
            Connection(add, lost),
        ]
        self._run([src, a, b, add, lost], connections, 6.0)
        total = a.outputs[0] + b.outputs[0] + lost.outputs[0]
        self.assertAlmostEqual(total, 2.0 + 3.0 * 6.0, places=5)
        self.assertGreater(lost.outputs[0], 1.0)


if __name__ == "__main__":
    unittest.main()
