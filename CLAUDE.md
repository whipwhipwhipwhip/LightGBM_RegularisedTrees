# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is a fork (`lightgbm-org/LightGBM` upstream, remote `whipwhipwhipwhip/LightGBM_RegularisedTrees`) of LightGBM, a gradient boosting framework using tree-based learning. The core engine is C++; Python and R packages wrap a shared library (`lib_lightgbm`) via a C API. There is also a CLI executable and SWIG-generated bindings (Java/.NET).

## Build commands

### Core C++ library + CLI (Linux/macOS)

```sh
cmake -B build -S .
cmake --build build -j4
```

Useful CMake options (pass as `-DOPTION=ON/OFF`):

- `BUILD_CLI` (default ON) — build the `lightgbm` CLI executable
- `BUILD_CPP_TEST` — build the Google Test suite (target `testlightgbm`)
- `BUILD_STATIC_LIB` — static instead of shared library
- `USE_OPENMP` (default ON), `USE_MPI`, `USE_GPU`, `USE_CUDA`, `USE_ROCM`, `USE_SWIG`
- `USE_DEBUG` — disable optimizations, enable extra internal checks
- `USE_SANITIZER` + `-DENABLED_SANITIZERS="address;leak;undefined"` (not combinable with `thread`)
- `USE_HOMEBREW_FALLBACK` (macOS only, default ON) — look in `brew --prefix` for deps like OpenMP

On macOS, `libomp` is required for the default (OpenMP) build: `brew install cmake libomp`.

### C++ unit tests

```sh
cmake -B build -S . -DBUILD_CPP_TEST=ON
cmake --build build --target testlightgbm -j4
./testlightgbm   # or build/testlightgbm depending on platform/generator
```

Tests live in `tests/cpp_tests/` and use Google Test. Prefer building with sanitizers (see above) when debugging memory issues.

### Python package

```sh
sh ./build-python.sh install              # compile lib_lightgbm and install the Python package
sh ./build-python.sh install --precompile # install using an existing lib_lightgbm.{so,dll,dylib} at repo root
sh ./build-python.sh sdist                # build sdist into dist/
sh ./build-python.sh bdist_wheel          # build wheel into dist/
```

Run `sh ./build-python.sh --help`-style comments at the top of `build-python.sh` for the full option list (Boost/OpenCL/CUDA/MPI flags, `--bit32`, `--time-costs`, etc.).

Python source lives in `python-package/lightgbm/`; tests in `tests/python_package_test/` (pytest). Run a single test, e.g.:

```sh
pytest tests/python_package_test/test_basic.py -k test_name
```

### R package

See `R-package/` and `build_r.R` / `build-cran-package.sh`.

### pixi environments

`pixi.toml` defines a workspace (`default`, `py310`) used mainly for CI on end-of-life Python versions; not required for normal local development.

## Linting / pre-commit

Many static checks run via `pre-commit`:

```sh
pre-commit run --all-files
```

This covers `cpplint`, `cmakelint`, `yamllint`, `ruff` (check + format, configured via `python-package/pyproject.toml`), `mypy` (against `python-package/`), `biome` (JS/JSON), `shellcheck`, `typos`, `rstcheck`, and a local `check-omp-pragmas` and `parameter-generator` regeneration hook.

If you add or change a training parameter, regenerate generated parameter docs/bindings with `.ci/parameter-generator.py` (this is also enforced by pre-commit).

## Architecture

The C++ core is organized so that each major concept has a header in `include/LightGBM/` and an implementation under a matching `src/` subdirectory:

| Concept | Header | Implementation dir |
|---|---|---|
| Training/prediction entrypoint | `application.h` | `src/application/` |
| Boosting algorithms (GBDT, DART, ...) | `boosting.h` | `src/boosting/` |
| Dataset, Bin, Config, Tree, IO | `dataset.h`, `bin.h`, `config.h`, `tree.h`, `dataset_loader.h`, `feature_group.h` | `src/io/` |
| Evaluation metrics | `metric.h` | `src/metric/` |
| Distributed communication | `network.h` | `src/network/` |
| Objective/loss functions | `objective_function.h` | `src/objective/` |
| Tree learners (serial, GPU, feature/data-parallel) | `tree_learner.h` | `src/treelearner/` |
| CUDA-accelerated paths | — | `src/cuda/` |
| Arrow/PyCapsule interop | `arrow.h` | `src/arrow/` |
| Common utilities | `include/LightGBM/utils/` | `src/utils/` |

The public C API (used by every language binding) is declared in `include/LightGBM/c_api.h`; its doc comments are the source for the generated C API docs.

The Python (`python-package/lightgbm/`) and R (`R-package/`) packages are thin wrappers that load the compiled `lib_lightgbm` shared library and call into the C API — `python-package/lightgbm/basic.py` holds the ctypes bindings, `engine.py`/`sklearn.py` build the training loop / scikit-learn-style estimators on top, and `dask.py` adds distributed training support.

Generated/derived files to be aware of: `R-package/configure`, `R-package/inst/Makevars*`, and `R-package/man/*.Rd` are auto-generated and excluded from pre-commit checks — don't hand-edit them. Parameter documentation/bindings generated from `.ci/parameter-generator.py` should be regenerated, not hand-edited.
