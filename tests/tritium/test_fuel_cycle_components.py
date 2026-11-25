########################################################################################
##
##                                     TESTS FOR
##                    the fuel cycle component blocks of tritium/
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

import numpy as np
from pathsim.solvers import EUF

from pathsim_fusion.tritium import (
    BreedingBlanket,
    DetritiationSystem,
    PlasmaFacingComponent,
    TokamakExhaustProcessing,
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


if __name__ == "__main__":
    unittest.main()
