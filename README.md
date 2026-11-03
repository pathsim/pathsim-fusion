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
| `FirstOrderLag` | First-order lag / low-pass filter | `tau`, `y0` |
| `IonisationChamber` | Flow-through tritium ionisation chamber (signal = ε(c)·c, flux passes through) | `detection_efficiency`, `detection_threshold` |
| `Bubbler4` | Four-vial tritium sampling bubbler (soluble + insoluble chains, catalytic conversion, scheduled vial replacement) | `conversion_efficiency`, `vial_efficiency`, `replacement_times` |
| `GLC` | Counter-current bubble-column gas–liquid contactor for tritium extraction from LiPb (Malara 1995, on `BVP1D`) | `P_in`, `L`, `D`, `T`, `BCs` |

## Install

```bash
pip install pathsim-fusion
```

## Quick Example

```python
from pathsim import Simulation, Connection
from pathsim.blocks import Source, Scope
from pathsim_fusion import FirstOrderLag

src = Source(lambda t: float(t > 1))  # unit step at t = 1
lag = FirstOrderLag(tau=2.0)  # first-order lag
sco = Scope(labels=["input", "output"])

sim = Simulation(
    blocks=[src, lag, sco],
    connections=[
        Connection(src, lag[0], sco[0]),
        Connection(lag, sco[1]),
    ],
    dt=0.01,
)
sim.run(20)
sco.plot()
```

## Development

```bash
pip install -e ".[test]" mypy
mypy src/pathsim_fusion    # type check
pytest tests/ -v            # run tests
```

## License

MIT
