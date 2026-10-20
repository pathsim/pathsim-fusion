# Using this template

This repository is a **GitHub template** for new PathSim toolboxes. It gives
every toolbox the same layout, CI, packaging and docs contract, so they don't
drift apart.

> This file is for the person setting up a new toolbox. The `init.py` script
> deletes it once initialization is done.

## 1. Create the repo

On GitHub, click **Use this template → Create a new repository**. Name it
`pathsim-<name>` and put it in the `pathsim` organization.

## 2. Run the initializer

Clone the new repo and run:

```bash
python init.py <name> --description "<one-line description>" --label <Label>
```

- `<name>` — short, lowercase, e.g. `rf`, `chem`, `vehicle`. Becomes the
  package `pathsim_<name>`, the distribution `pathsim-<name>`, and the docs
  slug `docs.pathsim.org/<name>`.
- `--description` — one-line project description for `pyproject.toml` / README.
- `--label` — display capitalization for `PathSim-<Label>` (defaults to the
  capitalized name; pass `RF` explicitly for acronyms).

The script renames `src/pathsim_toolbox/`, rewrites every reference across the
repo, removes the template banner from `README.md`, and finally deletes
`TEMPLATE.md` and `init.py`. Commit the result.

## 3. Replace the placeholder content

- `src/pathsim_<name>/example_block.py` — replace `FirstOrderLag` with your
  real blocks. Keep the structure: header comment, docstring documenting
  math / parameters / ports, input validation, `*_port_labels`.
- `tests/` — mirror the structure of `src/`. Tests run with both `unittest`
  and `pytest`; physics is verified through a full `Simulation`.
- `docs/source/examples/` — example notebooks. The placeholder is a copied
  PathSim core example; replace it with notebooks that demonstrate your
  blocks. Every `*.ipynb` here is executed and rendered by the docs build.

## 4. Stay compatible with the web runtime

PathView ships a browser build that runs Python through Pyodide. Pure-Python
toolboxes load there out of the box, but dependencies that ship native code
(`pybamm`, `jsbsim`, `casadi`, anything that needs a compiled wheel) fail to
install in the browser.

Default `pip install pathsim-<name>` on Linux/macOS/Windows should still
install everything — no extras required. To get there:

1. **Mark heavy dependencies in `pyproject.toml`** with a PEP 508 environment
   marker so micropip (Pyodide) silently skips them:

   ```toml
   dependencies = [
       "pathsim>=0.22",
       "numpy>=1.15",
       "pybamm>=25.12; sys_platform != 'emscripten'",
   ]
   ```

2. **Guard the re-export in `src/pathsim_<name>/__init__.py`** so the rest of
   the toolbox stays importable when the heavy dep is missing:

   ```python
   try:
       from .cells import CellElectrical
       __all__ += ["CellElectrical"]
   except ImportError:
       pass
   ```

In PathView's web Toolbox Manager, introspection then finds only the blocks
whose submodules imported successfully; the desktop install sees everything.

## 5. Register with the docs build

The documentation site (`docs.pathsim.org`) is built by the
[`pathsim/docs`](https://github.com/pathsim/docs) repository. It clones every
toolbox listed in `scripts/lib/config.py` and builds API docs + notebooks per
git tag.

To make the new toolbox appear, open a PR on `pathsim/docs` adding an entry to
`PACKAGES` and `MIN_SUPPORTED_VERSIONS` in `scripts/lib/config.py`:

```python
"<name>": {
    "repo": ROOT_DIR / "pathsim-<name>",
    "source": ROOT_DIR / "pathsim-<name>" / "src",
    "notebooks": ROOT_DIR / "pathsim-<name>" / "docs" / "source" / "examples",
    "figures": ROOT_DIR / "pathsim-<name>" / "docs" / "source" / "examples" / "figures",
    "display_name": "PathSim-<Label>",
    "griffe_package": "pathsim_<name>",
    "github_repo": "pathsim/pathsim-<name>",
    "root_modules": ["pathsim_<name>"],
    "required": False,
},
```

The repo-side docs contract is just: an importable package under `src/` and
`*.ipynb` files under `docs/source/examples/`. No Sphinx — the docs build owns
all rendering.

## 6. Releasing

`.github/workflows/publish.yml` publishes to PyPI via trusted publishing when
a GitHub Release is published. Versions come from git tags via `setuptools-scm`,
so tag releases as `vX.Y.Z`. Configure a
[PyPI trusted publisher](https://docs.pypi.org/trusted-publishers/) for the
project pointing at this repo and the `publish.yml` workflow.
