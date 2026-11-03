"""
PathSim-Fusion: Fusion fuel cycle and balance-of-plant blocks for PathSim

Specialized simulation blocks for the PathSim framework. All blocks follow
the standard PathSim block interface and can be connected into simulation
diagrams.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# Pure-Python blocks — eager.
from .example_block import FirstOrderLag
from .tritium import GLC, Bubbler4, IonisationChamber

__all__ = ["GLC", "Bubbler4", "FirstOrderLag", "IonisationChamber", "__version__"]

# If you add blocks that depend on a native package (e.g. pybamm, jsbsim,
# casadi, …) put them in their own submodule and re-export them defensively
# here so the toolbox still imports when the heavy dependency is missing
# (e.g. in Pyodide). Pair this with a PEP 508 environment marker in
# pyproject.toml — see the comment there for an example.
#
# try:
#     from .native_block import HeavyBlock
#     __all__ += ["HeavyBlock"]
# except ImportError:
#     pass
