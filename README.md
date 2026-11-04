<p align="center">
  <strong>Fusion fuel cycle and balance-of-plant blocks for PathSim</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/pathsim-fusion/"><img src="https://img.shields.io/pypi/v/pathsim-fusion" alt="PyPI"></a>
  <img src="https://img.shields.io/github/license/pathsim/pathsim-fusion" alt="License">
</p>

<p align="center">
  <a href="https://pathsim.org">Homepage</a> &bull;
  <a href="https://docs.pathsim.org/fusion">Documentation</a> &bull;
  <a href="https://github.com/pathsim/pathsim-fusion">GitHub</a>
</p>

---

PathSim-Fusion extends the [PathSim](https://github.com/pathsim/pathsim)
simulation framework with specialized blocks. All blocks follow the standard
PathSim block interface and can be connected into simulation diagrams.

## Blocks

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| `IonisationChamber` | Flow-through tritium ionisation chamber (signal = ε(c)·c, flux passes through) | `detection_efficiency`, `func_efficiency`, `detection_threshold` |
| `Bubbler4` | Four-vial tritium sampling bubbler (soluble + insoluble chains, catalytic conversion, scheduled vial replacement) | `conversion_efficiency`, `vial_efficiency`, `replacement_times` |
| `GLC` | Counter-current bubble-column gas–liquid contactor for tritium extraction from LiPb (Malara 1995, on `BVP1D`) | `P_in`, `L`, `D`, `T`, `BCs` |

## Install

```bash
pip install pathsim-fusion
```

## Quick Example

```python
from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Source, Scope
from pathsim_fusion import IonisationChamber

flux = Source(lambda t: 1e-3 * float(t > 1))  # tritium flux, step at t = 1
flow = Constant(0.5)  # carrier gas flow rate
ic = IonisationChamber(detection_efficiency=0.9)
sco = Scope(labels=["flux", "signal"])

sim = Simulation(
    blocks=[flux, flow, ic, sco],
    connections=[
        Connection(flux, ic["flux_in"], sco[0]),
        Connection(flow, ic["flow_rate"]),
        Connection(ic["signal"], sco[1]),
    ],
    dt=0.01,
)
sim.run(5)
sco.plot()
```

## Development

```bash
pip install -e ".[test,lint]" mypy
ruff check .                # lint
ruff format .               # format
mypy src/pathsim_fusion    # type check
pytest tests/ -v            # run tests
```

## License

MIT
