########################################################################################
##
##                                  TESTS FOR
##                  the ARC fuel cycle built from 'tritium' components
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
from pathlib import Path

import numpy as np
from pathsim import Connection, Simulation
from pathsim.blocks import Adder, Amplifier, Integrator, PulseSource, Scope, Splitter
from pathsim.solvers import RKBS32

from pathsim_fusion.tritium import (
    BreedingBlanket,
    DetritiationSystem,
    FuelCycleComponent,
    FusionChamberPump,
    HeatExchangerPermeation,
    IsotopeSeparationSystem,
    PlasmaFacingComponent,
    StorageAndManagement,
    TokamakExhaustProcessing,
    TritiumExtractionSystem,
)

# REFERENCE ============================================================================

# Inventories from the Simulink model of Meschini et al., Nucl. Fusion 63, 126005 (2023),
# first four days. Source and license in the header of the file.
REFERENCE = Path(__file__).parent / "data" / "arc_meschini_inventories.csv"

HOUR = 3600.0
DAY = 24 * HOUR

# parameters of the model that produced the reference data (as in the ARC example
# notebook); the fuel cleanup and isotope separation residence times and the burn rate
# differ from the tables of the paper
TAU_BZ = 1.25 * HOUR
TAU_TES = 24 * HOUR
TAU_FW = 1000.0
TAU_DIV = 1000.0
TAU_HX = 1000.0
TAU_DS = 1 * HOUR
TAU_VP = 600.0
TAU_FC = 1 * HOUR
TAU_ISS = 3 * HOUR
TAU_MEM = 100.0

DECAY_CONSTANT = 1.73e-9
LOSS_FRACTION = 1e-4

F_P_FW = 1e-4
F_P_DIV = 1e-4
F_HX_BZ = 0.33
F_HX_FW = 0.33
F_HX_DS = 1e-4
F_HX_DIV = 1 - F_HX_BZ - F_HX_FW - F_HX_DS
F_ISS_DS = 0.1
F_DIR = 0.3
ETA_TES = 0.95

BURN_RATE = 8.99e-7
TBE = 0.02
TBR = 1.067
PULSE_PERIOD = 1800.0
AVAILABILITY = 0.7

STARTUP_INVENTORY = 1.14

# long enough to pass the lowest storage inventory, after about 3.3 days
DURATION = 4 * DAY

# HELPERS ==============================================================================


def build_arc():
    """Build the ARC fuel cycle; returns the simulation, its scope and the labels."""

    common = {"loss_fraction": LOSS_FRACTION, "decay_constant": DECAY_CONSTANT}

    bz = BreedingBlanket(tau=TAU_BZ, **common)
    tes = TritiumExtractionSystem(tau=TAU_TES, efficiency=ETA_TES, **common)
    fw = PlasmaFacingComponent(tau=TAU_FW, decay_constant=DECAY_CONSTANT)
    div = PlasmaFacingComponent(tau=TAU_DIV, decay_constant=DECAY_CONSTANT)
    hx = HeatExchangerPermeation(tau=TAU_HX, permeation_fraction=F_HX_DS, **common)
    ds = DetritiationSystem(tau=TAU_DS, **common)
    vp = FusionChamberPump(tau=TAU_VP, dir_fraction=F_DIR, **common)
    fc = TokamakExhaustProcessing(tau=TAU_FC, **common)
    iss = IsotopeSeparationSystem(tau=TAU_ISS, waste_fraction=F_ISS_DS, **common)
    mem = FuelCycleComponent(tau=TAU_MEM, **common)
    sto = StorageAndManagement(
        decay_constant=DECAY_CONSTANT, initial_value=STARTUP_INVENTORY
    )

    names = ["BZ", "TES", "FW", "DIV", "HX", "DS", "VP", "FC", "ISS", "MEM", "STO"]
    components = [bz, tes, fw, div, hx, ds, vp, fc, iss, mem, sto]

    pulse = PulseSource(amplitude=1.0, T=PULSE_PERIOD, duty=AVAILABILITY)
    fuelling = Amplifier(-BURN_RATE / TBE)
    breeding = Amplifier(TBR * BURN_RATE)
    to_fw = Amplifier(F_P_FW * BURN_RATE / TBE)
    to_div = Amplifier(F_P_DIV * BURN_RATE / TBE)
    to_vp = Amplifier((1 - TBE - F_P_FW - F_P_DIV) * BURN_RATE / TBE)

    returned = 1 - F_HX_DS
    split_hx = Splitter([F_HX_BZ / returned, F_HX_FW / returned, F_HX_DIV / returned])

    # the inventories, then the tritium that left the cycle by losses and decay
    sco = Scope(sampling_period=60.0, labels=[*names, "lost"])
    loss_rate = Adder()
    lost = Integrator(0.0)

    connections = [
        Connection(pulse, fuelling, breeding, to_fw, to_div, to_vp),
        Connection(breeding, bz[0]),
        Connection(to_fw, fw[0]),
        Connection(to_div, div[0]),
        Connection(to_vp, vp[0]),
        Connection(fuelling, sto[0]),
        Connection(bz["outflow"], tes[0]),
        Connection(tes["extracted"], mem[0]),
        Connection(tes["residual"], hx[0]),
        Connection(mem["outflow"], sto[1]),
        Connection(hx["returned"], split_hx),
        Connection(split_hx[0], bz[1]),
        Connection(split_hx[1], fw[1]),
        Connection(split_hx[2], div[1]),
        Connection(hx["permeated"], ds[0]),
        Connection(fw["outflow"], bz[2]),
        Connection(div["outflow"], bz[3]),
        Connection(vp["recycled"], sto[2]),
        Connection(vp["exhaust"], fc[0]),
        Connection(fc["outflow"], iss[0]),
        Connection(ds["outflow"], iss[1]),
        Connection(iss["product"], sto[3]),
        Connection(iss["waste"], ds[1]),
        Connection(loss_rate, lost),
        Connection(lost, sco[len(components)]),
    ]
    for i, component in enumerate(components):
        connections.append(Connection(component["inventory"], sco[i]))
        connections.append(Connection(component["loss"], loss_rate[i]))

    blocks = [
        *components,
        pulse,
        fuelling,
        breeding,
        to_fw,
        to_div,
        to_vp,
        split_hx,
        loss_rate,
        lost,
        sco,
    ]

    # looser than the example notebook to keep the test fast; the difference to the
    # reference stays at the precision of the reference data
    sim = Simulation(
        blocks,
        connections,
        Solver=RKBS32,
        dt=1.0,
        dt_max=PULSE_PERIOD,
        tolerance_lte_rel=1e-4,
        tolerance_lte_abs=1e-9,
        log=False,
    )
    return sim, sco, names


# TESTS ================================================================================


class TestARCBaseline(unittest.TestCase):
    """Reproduce the ARC fuel cycle of Meschini et al. (2023) from the components.

    The model runs once for the first four days of operation; the tests compare it
    with the inventories of the reference Simulink model and check the tritium balance.
    """

    @classmethod
    def setUpClass(cls):
        sim, sco, names = build_arc()
        sim.run(DURATION)
        cls.time, data = sco.read()
        cls.inventory = dict(zip(names, data[:-1]))
        cls.inventories = data[:-1]
        cls.lost = data[-1]
        cls.reference = np.loadtxt(REFERENCE, delimiter=",", comments="#")

    def test_inventories_match_reference(self):
        """Breeding zone, extraction, isotope separation and storage follow the reference.

        The reference values mostly carry 7 to 9 significant digits and the two models agree
        to about 6e-5. The bound of 1e-3 leaves room for solver and interpolation noise
        but fails on any change to a residence time, a flow fraction or a port.
        Compared after the first hour, while the inventories still start from zero.
        """
        ref = self.reference
        later = ref[:, 0] > HOUR
        for column, name in [(1, "BZ"), (2, "TES"), (3, "ISS"), (4, "STO")]:
            with self.subTest(component=name):
                ours = np.interp(ref[later, 0], self.time, self.inventory[name])
                np.testing.assert_allclose(ours, ref[later, column], rtol=1e-3)

    def test_lowest_storage_inventory(self):
        """The storage reaches its lowest inventory, 0.7157 kg, after about 3.3 days.

        This is the startup inventory the cycle needs beyond its working inventory, the
        headline result of the reference model. The time must agree to one pulse.
        """
        ref_sto = self.reference[:, 4]
        sto = self.inventory["STO"]
        self.assertAlmostEqual(sto.min(), ref_sto.min(), delta=1e-3 * ref_sto.min())
        self.assertAlmostEqual(
            self.time[sto.argmin()],
            self.reference[ref_sto.argmin(), 0],
            delta=PULSE_PERIOD,
        )

    def test_tritium_balance(self):
        """All inventories plus the lost tritium equal startup plus net breeding.

        The net breeding is (TBR - 1) times the burn rate times the time the plasma
        was on. The residual is set by the solver tolerance, far below the inventories.
        """
        on_time = PULSE_PERIOD * AVAILABILITY
        t = self.time
        burn_time = (t // PULSE_PERIOD) * on_time + np.minimum(
            t % PULSE_PERIOD, on_time
        )
        expected = STARTUP_INVENTORY + (TBR - 1) * BURN_RATE * burn_time
        total = self.inventories.sum(axis=0) + self.lost
        np.testing.assert_allclose(total, expected, rtol=0, atol=1e-6)


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
