<!-- TEMPLATE:START -->
> **This is the PathSim toolbox template.**
> The README below still contains placeholders. To turn this template into a
> real toolbox, create a repository from it, then run the initializer:
>
> ```bash
> python init.py <name> --description "<one-line description>" --label <Label>
> ```
>
> It renames the package, rewrites every reference, and removes this banner.
> See [`TEMPLATE.md`](TEMPLATE.md) for the full setup, including registering
> the toolbox with the documentation build.

---
<!-- TEMPLATE:END -->

<p align="center">
  <strong>A toolbox for PathSim</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/pathsim-toolbox/"><img src="https://img.shields.io/pypi/v/pathsim-toolbox" alt="PyPI"></a>
  <img src="https://img.shields.io/github/license/pathsim/pathsim-toolbox" alt="License">
</p>

<p align="center">
  <a href="https://pathsim.org">Homepage</a> &bull;
  <a href="https://docs.pathsim.org/toolbox">Documentation</a> &bull;
  <a href="https://github.com/pathsim/pathsim-toolbox">GitHub</a>
</p>

---

PathSim-Toolbox extends the [PathSim](https://github.com/pathsim/pathsim)
simulation framework with specialized blocks. All blocks follow the standard
PathSim block interface and can be connected into simulation diagrams.

## Blocks

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| `FirstOrderLag` | First-order lag / low-pass filter | `tau`, `y0` |

## Install

```bash
pip install pathsim-toolbox
```

## Quick Example

```python
from pathsim import Simulation, Connection
from pathsim.blocks import Source, Scope
from pathsim_toolbox import FirstOrderLag

src = Source(lambda t: float(t > 1))   # unit step at t = 1
lag = FirstOrderLag(tau=2.0)           # first-order lag
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
mypy src/pathsim_toolbox    # type check
pytest tests/ -v            # run tests
```

## License

MIT
