#!/usr/bin/env bash
# Fast dev loop for the regularised-trees fork.
#
# Builds ONLY the C++ shared library and drops it straight into an existing
# venv's installed lightgbm package, replacing lib_lightgbm.dylib. Skips the
# wheel build and `uv sync` entirely: an incremental C++ change goes from edit
# to runnable in seconds rather than minutes.
#
# This is safe because the LightGBM Python package is a thin ctypes wrapper --
# it locates the library at import time via lightgbm/libpath.py and does not
# validate parameter names itself, so adding a C++ config field needs no
# Python-side change at all.
#
# Usage:
#   ./rebuild-dev.sh                      # install into the default target venv below
#   ./rebuild-dev.sh /path/to/other/.venv
#
# Re-run `uv sync --reinstall-package lightgbm` from the consuming project only
# when you want a real wheel again (e.g. before sharing the build).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$REPO_DIR/build-dev"
TARGET_VENV="${1:-/Users/cono/code/trees_testing/.venv}"

case "$(uname -s)" in
  Darwin) LIB_NAME="lib_lightgbm.dylib" ;;
  *)      LIB_NAME="lib_lightgbm.so" ;;
esac

PKG_LIB_DIR="$(echo "$TARGET_VENV"/lib/python*/site-packages/lightgbm/lib)"
if [[ ! -d "$PKG_LIB_DIR" ]]; then
  echo "error: no installed lightgbm package under $TARGET_VENV" >&2
  echo "       run 'uv sync' in the consuming project first" >&2
  exit 1
fi

# cmake is not installed system-wide here; uv can supply it on demand.
if command -v cmake >/dev/null 2>&1; then
  CMAKE=(cmake)
else
  CMAKE=(uv run --no-project --with cmake --with ninja cmake)
fi

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  echo ">>> configuring (first run only)"
  "${CMAKE[@]}" -S "$REPO_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_CLI=OFF \
    -DUSE_OPENMP=ON
fi

echo ">>> building _lightgbm"
"${CMAKE[@]}" --build "$BUILD_DIR" --target _lightgbm -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

echo ">>> installing $LIB_NAME -> $PKG_LIB_DIR"
# CMakeLists sets LIBRARY_OUTPUT_DIRECTORY to the repo root, not the build dir.
cp "$REPO_DIR/$LIB_NAME" "$PKG_LIB_DIR/$LIB_NAME"

# On Apple Silicon, overwriting a dylib in place invalidates its signature and the
# kernel SIGKILLs any process that loads it (python dies with exit 137, no output).
# Re-sign ad-hoc to make the replacement loadable.
if [[ "$(uname -s)" == "Darwin" ]]; then
  codesign -f -s - "$PKG_LIB_DIR/$LIB_NAME" 2>/dev/null
fi

echo ">>> done"
