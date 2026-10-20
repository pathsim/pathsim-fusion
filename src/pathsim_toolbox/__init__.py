"""
PathSim-Toolbox: A toolbox for PathSim

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

__all__ = ["__version__", "FirstOrderLag"]

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
