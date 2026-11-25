########################################################################################
##
##                                     TESTS FOR
##                    the fuel cycle component blocks of tritium/
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

import numpy as np
from pathsim import Connection, Simulation
from pathsim.blocks import Adder, Constant, Integrator, Scope, Splitter
from pathsim.solvers import EUF, RKCK54

from pathsim_fusion.tritium import (
    BreedingBlanket,
    DetritiationSystem,
    FusionChamberPump,
    HeatExchangerPermeation,
    IsotopeSeparationSystem,
    PlasmaFacingComponent,
    StorageAndManagement,
    TokamakExhaustProcessing,
    TritiumExtractionSystem,
)

# TESTS ================================================================================


class TestSingleOutletComponents(unittest.TestCase):
    """Test the components that keep the ports of FuelCycleComponent.

    BreedingBlanket, PlasmaFacingComponent, TokamakExhaustProcessing and
    DetritiationSystem differ from the base block only in their default tau.
    """

    # block and its default residence time
    cases = (
        (BreedingBlanket, 4500.0),
        (PlasmaFacingComponent, 1000.0),
        (TokamakExhaustProcessing, 1080.0),
        (DetritiationSystem, 3600.0),
    )

    def test_defaults(self):
        """Default construction works and gives the documented residence time."""
        for cls, tau in self.cases:
            with self.subTest(block=cls.__name__):
                blk = cls()
                self.assertEqual(blk.tau, tau)
                self.assertEqual(blk.loss_fraction, 0.0)
                self.assertEqual(blk.decay_constant, 0.0)
                self.assertEqual(cls.info()["parameters"]["tau"]["default"], tau)
                self.assertEqual(len(blk), 0)

    def test_ports(self):
        """Ports are those of the base block."""
        for cls, _ in self.cases:
            with self.subTest(block=cls.__name__):
                self.assertIsNone(cls.input_port_labels)
                self.assertEqual(
                    cls.output_port_labels, {"inventory": 0, "outflow": 1, "loss": 2}
                )

    def test_parameters_passed_on(self):
        """Every parameter reaches the residence time model."""
        for cls, _ in self.cases:
            with self.subTest(block=cls.__name__):
                blk = cls(
                    tau=4.0,
                    loss_fraction=0.25,
                    decay_constant=0.5,
                    source_term=7.0,
                    initial_value=8.0,
                )
                blk.set_solver(EUF, parent=None)
                blk.update(0.0)
                np.testing.assert_allclose(
                    blk.outputs.to_array(), [8.0, 2.0, 4.5], rtol=1e-12
                )
                rate = float(blk.op_dyn(np.array([8.0]), np.array([3.0, 11.0]), 0.0)[0])
                self.assertAlmostEqual(rate, 14.0 + 7.0 - 2.5 - 4.0, places=12)

    def test_validation(self):
        """The validation of the base block applies."""
        for cls, _ in self.cases:
            with (
                self.subTest(block=cls.__name__),
                self.assertRaisesRegex(ValueError, "'tau'"),
            ):
                cls(tau=0.0)


class TestTwoOutletComponents(unittest.TestCase):
    """Test the components that split their outflow between two named outlets.

    TritiumExtractionSystem, HeatExchangerPermeation, FusionChamberPump and
    IsotopeSeparationSystem.
    """

    # block, fraction parameter, its default, default tau, the outlet that gets
    # the fraction and the outlet that gets the rest
    cases = (
        (TritiumExtractionSystem, "efficiency", 0.95, 86400.0, "extracted", "residual"),
        (
            HeatExchangerPermeation,
            "permeation_fraction",
            1e-4,
            1000.0,
            "permeated",
            "returned",
        ),
        (FusionChamberPump, "dir_fraction", 0.3, 600.0, "recycled", "exhaust"),
        (IsotopeSeparationSystem, "waste_fraction", 0.1, 13320.0, "waste", "product"),
    )

    def _make(self, cls, name, fraction):
        """Block with distinct parameter values, ready to update."""
        blk = cls(
            tau=4.0,
            loss_fraction=0.25,
            decay_constant=0.5,
            source_term=7.0,
            initial_value=8.0,
            **{name: fraction},
        )
        blk.set_solver(EUF, parent=None)
        return blk

    def test_defaults(self):
        """Default construction works and gives the documented defaults."""
        for cls, name, fraction, tau, _, _ in self.cases:
            with self.subTest(block=cls.__name__):
                blk = cls()
                self.assertEqual(blk.tau, tau)
                self.assertEqual(getattr(blk, name), fraction)
                params = cls.info()["parameters"]
                self.assertEqual(params["tau"]["default"], tau)
                self.assertEqual(params[name]["default"], fraction)
                self.assertEqual(len(blk), 0)

    def test_ports(self):
        """Outputs are the inventory, the two outlets and the loss, contiguous."""
        expected = {
            TritiumExtractionSystem: ["inventory", "extracted", "residual", "loss"],
            HeatExchangerPermeation: ["inventory", "permeated", "returned", "loss"],
            FusionChamberPump: ["inventory", "recycled", "exhaust", "loss"],
            IsotopeSeparationSystem: ["inventory", "product", "waste", "loss"],
        }
        for cls, labels in expected.items():
            with self.subTest(block=cls.__name__):
                self.assertIsNone(cls.input_port_labels)
                self.assertEqual(
                    cls.output_port_labels, {k: i for i, k in enumerate(labels)}
                )

    def test_outputs(self):
        """Each outlet gets its share of I/tau, the loss is that of the base block."""
        for cls, name, _, _, part, rest in self.cases:
            with self.subTest(block=cls.__name__):
                blk = self._make(cls, name, 0.2)
                blk.inputs[0] = 3.0
                blk.update(0.0)
                self.assertAlmostEqual(
                    blk["inventory"].get_outputs()[0], 8.0, places=12
                )
                self.assertAlmostEqual(blk[part].get_outputs()[0], 0.2 * 2.0, places=12)
                self.assertAlmostEqual(blk[rest].get_outputs()[0], 0.8 * 2.0, places=12)
                self.assertAlmostEqual(blk["loss"].get_outputs()[0], 4.5, places=12)

    def test_rate_equals_inflow_minus_outputs(self):
        """The split does not change the inventory balance."""
        for cls, name, _, _, part, rest in self.cases:
            with self.subTest(block=cls.__name__):
                blk = self._make(cls, name, 0.2)
                blk.update(0.0)
                out = blk[part].get_outputs()[0] + blk[rest].get_outputs()[0]
                rate = float(blk.op_dyn(np.array([8.0]), np.array([3.0, 11.0]), 0.0)[0])
                self.assertAlmostEqual(
                    rate, 14.0 + 7.0 - out - blk["loss"].get_outputs()[0], places=12
                )

    def test_fraction_limits(self):
        """Fractions 0 and 1 send the whole outflow to one outlet."""
        for cls, name, _, _, part, rest in self.cases:
            for fraction in (0.0, 1.0):
                with self.subTest(block=cls.__name__, fraction=fraction):
                    blk = self._make(cls, name, fraction)
                    blk.update(0.0)
                    self.assertAlmostEqual(
                        blk[part].get_outputs()[0], fraction * 2.0, places=12
                    )
                    self.assertAlmostEqual(
                        blk[rest].get_outputs()[0], (1 - fraction) * 2.0, places=12
                    )

    def test_fraction_validation(self):
        """A fraction outside [0, 1] raises ValueError that names the parameter."""
        for cls, name, _, _, _, _ in self.cases:
            for value in (-0.1, 1.1, float("nan")):
                with (
                    self.subTest(block=cls.__name__, value=value),
                    self.assertRaisesRegex(ValueError, f"'{name}'"),
                ):
                    cls(**{name: value})

    def test_base_validation(self):
        """The validation of the base block applies."""
        for cls, _, _, _, _, _ in self.cases:
            with (
                self.subTest(block=cls.__name__),
                self.assertRaisesRegex(ValueError, "'tau'"),
            ):
                cls(tau=-1.0)

    def test_fraction_edit(self):
        """A fraction changed after construction is used by the next update."""
        for cls, name, _, _, part, rest in self.cases:
            with self.subTest(block=cls.__name__):
                blk = self._make(cls, name, 0.2)
                setattr(blk, name, 0.6)
                blk.update(0.0)
                self.assertAlmostEqual(blk[part].get_outputs()[0], 0.6 * 2.0, places=12)
                self.assertAlmostEqual(blk[rest].get_outputs()[0], 0.4 * 2.0, places=12)


class TestStorageAndManagement(unittest.TestCase):
    """Test the StorageAndManagement block."""

    def test_defaults(self):
        """Default construction works, the storage has no residence time parameter."""
        blk = StorageAndManagement()
        self.assertEqual(blk.decay_constant, 0.0)
        self.assertEqual(blk.source_term, 0.0)
        self.assertEqual(blk.initial_value, 0.0)
        self.assertEqual(
            list(StorageAndManagement.info()["parameters"]),
            ["decay_constant", "source_term", "initial_value"],
        )
        self.assertEqual(len(blk), 0)

    def test_ports(self):
        """Outputs are the inventory and the loss only."""
        self.assertIsNone(StorageAndManagement.input_port_labels)
        self.assertEqual(
            StorageAndManagement.output_port_labels, {"inventory": 0, "loss": 1}
        )

    def test_outputs_and_rate(self):
        """The storage accumulates the inflows and source, and loses only the decay."""
        blk = StorageAndManagement(
            decay_constant=0.5, source_term=7.0, initial_value=8.0
        )
        blk.set_solver(EUF, parent=None)
        blk.inputs[0] = 3.0
        blk.update(0.0)
        np.testing.assert_allclose(blk.outputs.to_array(), [8.0, 4.0], rtol=1e-12)
        rate = float(blk.op_dyn(np.array([8.0]), np.array([3.0, -11.0]), 0.0)[0])
        self.assertAlmostEqual(rate, 3.0 - 11.0 + 7.0 - 4.0, places=12)

    def test_validation(self):
        """Negative decay constant and initial inventory raise ValueError."""
        with self.assertRaisesRegex(ValueError, "'decay_constant'"):
            StorageAndManagement(decay_constant=-1.0)
        with self.assertRaisesRegex(ValueError, "'initial_value'"):
            StorageAndManagement(initial_value=-1.0)

    def test_withdrawal(self):
        """A negative inflow withdraws fuel, the inventory follows the net flow."""
        feed, draw = Constant(3.0), Constant(-1.0)
        blk = StorageAndManagement(initial_value=5.0)
        sim = Simulation(
            [feed, draw, blk],
            [Connection(feed, blk[0]), Connection(draw, blk[1])],
            Solver=RKCK54,
            dt=0.01,
            log=False,
        )
        sim.run(4.0)
        self.assertAlmostEqual(blk.outputs[0], 5.0 + 2.0 * 4.0, places=6)
        self.assertEqual(blk.outputs[1], 0.0)


class TestFuelCycleLoop(unittest.TestCase):
    """Test the component blocks wired into a closed fuel cycle."""

    def test_conservation(self):
        """Inventories plus integrated losses stay equal to the initial inventory.

        Every component block once, in a loop without sources: the storage
        feeds the pump at a constant rate, everything returns to the storage.
        """
        kw = {"loss_fraction": 0.1, "decay_constant": 0.02}
        sto = StorageAndManagement(decay_constant=0.02, initial_value=10.0)
        vp = FusionChamberPump(tau=0.5, dir_fraction=0.3, **kw)
        tep = TokamakExhaustProcessing(tau=0.8, **kw)
        iss = IsotopeSeparationSystem(tau=1.5, waste_fraction=0.2, **kw)
        ds = DetritiationSystem(tau=0.7, **kw)
        pfc = PlasmaFacingComponent(tau=0.4, **kw)
        bb = BreedingBlanket(tau=1.1, **kw)
        tes = TritiumExtractionSystem(tau=2.0, efficiency=0.6, **kw)
        hx = HeatExchangerPermeation(tau=0.3, permeation_fraction=0.25, **kw)
        comps = [sto, vp, tep, iss, ds, pfc, bb, tes, hx]

        fuelling = Constant(0.5)
        withdrawal = Constant(-0.5)
        to_pfc = Splitter([0.9, 0.1])
        loss_rate, lost, sco = Adder(), Integrator(0.0), Scope()

        connections = [
            Connection(withdrawal, sto[0]),
            Connection(fuelling, to_pfc),
            Connection(to_pfc[0], vp[0]),
            Connection(to_pfc[1], pfc[0]),
            Connection(vp["recycled"], sto[1]),
            Connection(vp["exhaust"], tep[0]),
            Connection(tep["outflow"], iss[0]),
            Connection(iss["product"], sto[2]),
            Connection(iss["waste"], ds[0]),
            Connection(ds["outflow"], iss[1]),
            Connection(pfc["outflow"], bb[0]),
            Connection(bb["outflow"], tes[0]),
            Connection(tes["extracted"], sto[3]),
            Connection(tes["residual"], hx[0]),
            Connection(hx["permeated"], ds[1]),
            Connection(hx["returned"], bb[1]),
            Connection(loss_rate, lost),
            Connection(lost, sco[len(comps)]),
        ]
        for i, blk in enumerate(comps):
            connections.append(Connection(blk["inventory"], sco[i]))
            connections.append(Connection(blk["loss"], loss_rate[i]))

        sim = Simulation(
            [*comps, fuelling, withdrawal, to_pfc, loss_rate, lost, sco],
            connections,
            Solver=RKCK54,
            dt=0.01,
            tolerance_lte_rel=1e-8,
            tolerance_lte_abs=1e-10,
            log=False,
        )
        sim.run(8.0)
        _, data = sco.read()
        np.testing.assert_allclose(data.sum(axis=0), 10.0, atol=1e-6)

        # every component took part, and a real share was lost
        self.assertTrue(np.all(data[:-1].max(axis=1) > 1e-3))
        self.assertGreater(data[-1, -1], 0.5)


if __name__ == "__main__":
    unittest.main()
