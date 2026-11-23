"""Reproduce output comparisons between local routines and pyrfu.

Run with ``.venv/bin/python scripts/compare_pyrfu_parity.py`` while pyrfu is
installed in the environment. The app itself does not import pyrfu.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from myhelpers.transformation.filt import filt
from myhelpers.transformation.pyrf_local import (
    compress_cwt,
    mva,
    wavelet,
)


def load_upstream(name: str):
    """Load the upstream module without executing pyrfu's package initializer.

    pyrfu's initializer imports geopack and attempts to load IGRF coefficient
    files. Direct module loading keeps this comparison focused on the four
    routines while avoiding that unrelated initialization side effect.
    """
    pyrf_dir = Path(importlib.metadata.distribution("pyrfu").locate_file("pyrfu/pyrf"))
    if "pyrfu" not in sys.modules:
        package = types.ModuleType("pyrfu")
        package.__path__ = [str(pyrf_dir.parent)]
        sys.modules["pyrfu"] = package
    if "pyrfu.pyrf" not in sys.modules:
        package = types.ModuleType("pyrfu.pyrf")
        package.__path__ = [str(pyrf_dir)]
        sys.modules["pyrfu.pyrf"] = package
    if name == "wavelet" and "pyrfu.pyrf.calc_fs" not in sys.modules:
        load_upstream("calc_fs")
    module_name = f"pyrfu.pyrf.{name}"
    path = pyrf_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load installed pyrfu source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def close(actual: object, expected: object, label: str) -> None:
    if np.issubdtype(np.asarray(actual).dtype, np.datetime64):
        np.testing.assert_array_equal(actual, expected, err_msg=label)
        return
    np.testing.assert_allclose(
        actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True, err_msg=label
    )


def main() -> None:
    assert "pyrfu" not in sys.modules, (
        "local myhelpers imports unexpectedly loaded pyrfu"
    )
    upstream_filt = load_upstream("filt").filt
    upstream_mva = load_upstream("mva").mva
    upstream_wavelet = load_upstream("wavelet").wavelet
    upstream_compress_cwt = load_upstream("compress_cwt").compress_cwt
    rng = np.random.default_rng(42)
    n = 2048
    time = pd.date_range("2024-01-01", periods=n, freq="20ms")
    data = rng.normal(size=(n, 3))
    data += np.column_stack(
        [np.sin(2 * np.pi * f * np.arange(n) * 0.02) for f in (0.7, 1.3, 2.1)]
    )
    field = xr.DataArray(
        data,
        dims=("time", "comp"),
        coords={"time": time, "comp": ["x", "y", "z"]},
        name="B",
    )

    local_filtered = filt(field, 0, 4, 3)
    upstream_filtered = upstream_filt(field, 0, 4, 3)
    close(local_filtered.data, upstream_filtered.data, "filt")

    nan_field = field.copy()
    nan_field.data[50:53, 1] = np.nan
    local_nan_filtered = filt(nan_field, 0, 4, 3)
    assert np.isnan(local_nan_filtered.data[50:53, 1]).all()
    assert np.isfinite(local_nan_filtered.data[:, 0]).all()

    local_mva = mva(field)
    upstream_mva_result = upstream_mva(field)
    for local, upstream, label in zip(
        local_mva, upstream_mva_result, ("mva data", "mva values", "mva vectors")
    ):
        close(
            local.data if isinstance(local, xr.DataArray) else local,
            upstream.data if isinstance(upstream, xr.DataArray) else upstream,
            label,
        )

    local_cwt = wavelet(field, f=[0.2, 15], n_freqs=48)
    upstream_cwt = upstream_wavelet(field, f=[0.2, 15], n_freqs=48)
    assert isinstance(local_cwt, xr.Dataset) and isinstance(upstream_cwt, xr.Dataset)
    assert list(local_cwt.data_vars) == list(upstream_cwt.data_vars)
    for var in local_cwt.data_vars:
        close(local_cwt[var].data, upstream_cwt[var].data, f"wavelet {var}")

    local_nan_compressed = compress_cwt(local_cwt, nc=16)
    upstream_nan_compressed = upstream_compress_cwt(upstream_cwt, nc=16)
    for i, (local, upstream) in enumerate(
        zip(local_nan_compressed, upstream_nan_compressed)
    ):
        close(local, upstream, f"compress_cwt NaN result {i}")

    finite_cwt = xr.Dataset(
        {
            name: (("time", "frequency"), rng.normal(size=(n, 48)))
            for name in ("x", "y", "z")
        },
        coords={"time": time, "frequency": np.arange(48.0)},
    )
    local_compressed = compress_cwt(finite_cwt, nc=16)
    upstream_compressed = upstream_compress_cwt(finite_cwt, nc=16)
    for i, (local, upstream) in enumerate(zip(local_compressed, upstream_compressed)):
        close(local, upstream, f"compress_cwt result {i}")

    print("All pyrfu parity comparisons passed.")


if __name__ == "__main__":
    main()
