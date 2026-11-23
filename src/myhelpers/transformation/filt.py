#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NaN-safe replacement for pyrfu.pyrf.filt.

This module provides a drop-in replacement of the `filt` function from
`pyrfu.pyrf.filt` which handles NaN values by temporarily interpolating
over them, performing the filtering, and then restoring NaNs at the original
time indices so the output has the same shape and NaN locations as the input.

Functions included (copied / adapted from upstream):
- _ellip_coefficients
- filt

Usage:
>>> from pyrf_filt_nan_safe import filt
>>> out = filt(inp, f_min=0.0, f_max=1.0, order=-1)

The function expects an xarray.DataArray with a `time` coordinate.
"""

# 3rd party imports
from typing import Any

import numpy as np
import xarray as xr
from scipy import signal

signal: Any

__author__ = "Adapted by Jonathan from Louis Richard"
__email__ = ""
__copyright__ = ""
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Prototype"


def _ellip_coefficients(
    f_min: float, f_max: float, order: int
) -> tuple[Any, Any, Any, Any]:
    """Compute numerator and denominator coefficients for elliptic filters.

    Args:
        f_min: Normalized lower cutoff frequency.
        f_max: Normalized upper cutoff frequency.
        order: Filter order, or ``-1`` to choose the order automatically.

    Returns:
        Numerator and denominator arrays for the low/high pass stages.
    """
    num1, den1, num2, den2 = [None] * 4

    # fact defines the width between stopband and passband
    r_p, r_s, fact = 0.5, 60, 1.1

    if f_min == 0:
        if order == -1:
            order, f_max = signal.ellipord(
                f_max,
                np.min([f_max * fact, 0.9999]),
                r_p,
                r_s,
            )

        num1, den1 = signal.ellip(order, r_p, r_s, f_max, btype="lowpass")
    elif f_max == 0:
        if order == -1:
            order, f_min = signal.ellipord(
                f_min,
                np.min([f_min * fact, 0.9999]),
                r_p,
                r_s,
            )

        num1, den1 = signal.ellip(order, r_p, r_s, f_min, btype="highpass")
    else:
        if order == -1:
            order1, f_max = signal.ellipord(
                f_max,
                np.min([f_max * 1.3, 0.9999]),
                0.5,
                60,
            )
            order2, f_min = signal.ellipord(f_min, f_min * 0.75, 0.5, 60)
        else:
            order1, order2 = [order, order]

        num1, den1 = signal.ellip(order1, 0.5, 60, f_max)
        num2, den2 = signal.ellip(order2, 0.5, 60, f_min)

    return num1, den1, num2, den2


def _interpolate_nans_1d(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linearly interpolate missing values in a one-dimensional array.

    Args:
        y: One-dimensional signal values, potentially containing NaNs.
        x: Monotonically increasing coordinates corresponding to ``y``.

    Returns:
        A copy of ``y`` with missing values interpolated where possible.
    """
    if y.ndim != 1:
        raise ValueError("_interpolate_nans_1d expects a 1D array")

    mask = np.isnan(y)
    if not mask.any():
        return y.copy()
    if mask.all():
        # Nothing to interpolate
        return y.copy()

    # Good points
    x_good = x[~mask]
    y_good = y[~mask]

    # np.interp requires x to be increasing (which time should be)
    # It will use edge values for extrapolation; that's fine because
    # we will restore NaNs at the same positions after filtering.
    y_interp = y.copy()
    y_interp[mask] = np.interp(x[mask], x_good, y_good)
    return y_interp


def filt(
    inp: xr.DataArray, f_min: float = 0.0, f_max: float = 1.0, order: int = -1
) -> xr.DataArray:
    """Filter a time series while preserving its original NaN positions.

    Args:
        inp: Time series DataArray with a ``time`` coordinate.
        f_min: Lower passband cutoff frequency in hertz.
        f_max: Upper passband cutoff frequency in hertz.
        order: Elliptic filter order, or ``-1`` for automatic selection.

    Returns:
        Filtered DataArray with the input coordinates, dimensions, attributes,
        and NaN locations preserved.
    """
    assert isinstance(inp, xr.DataArray), "inp must be a xarray.DataArray"
    
    f_samp: Any = 1 / (np.median(np.diff(inp.time.astype(np.int64) * 1e-9)))

    # make sure we operate on floats
    inp_data = inp.data.astype(np.float64)

    assert isinstance(f_min, (int, float)), "f_min must be int or float"
    assert isinstance(f_max, (int, float)), "f_max must be int or float"
    assert isinstance(order, (int, float)), "order must be int or float"

    f_min, f_max = [f_min / (f_samp / 2.0), f_max / (f_samp / 2.0)]

    f_max = np.min([f_max, 1.0])
    
    # Parameters of the elliptic filter. fact defines the width between
    # stopband and passband
    # r_pass, r_stop, fact = [0.5, 60, 1.1]
    num1, den1, num2, den2 = _ellip_coefficients(f_min, f_max, order)

    # Ensure 2D array (time x components)
    squeezed = False
    if len(inp_data.shape) == 1:
        inp_data = inp_data[:, np.newaxis]
        squeezed = True

    # Prepare output
    out_data = np.zeros(inp_data.shape, dtype=inp_data.dtype)

    # time coordinate to use for interpolation
    # convert to seconds since epoch (float) for interpolation
    time_seconds = inp.time.astype(np.int64) * 1e-9
    time_seconds = time_seconds.astype(np.float64)

    for i_col in range(inp_data.shape[1]):
        col = inp_data[:, i_col]
        nan_mask = np.isnan(col)

        if nan_mask.all():
            # keep column as all-NaN
            out_data[:, i_col] = col.copy()
            continue

        # interpolate over NaNs to produce a continuous signal for filtering
        col_interp = _interpolate_nans_1d(col, time_seconds)

        # Now apply filtering exactly as original code
        padtype = "odd"
        padlen = 3 * (max(len(num1), len(den1)) - 1)
        try:
            filtered = signal.filtfilt(
                num1, den1, col_interp, padtype=padtype, padlen=padlen
            )
        except Exception:
            # In case filtfilt fails (very short series, etc.), fall back to
            # returning NaNs in that column to avoid producing misleading data.
            out_data[:, i_col] = np.full(col.shape, np.nan)
            continue

        if num2 is not None and den2 is not None:
            padlen = 3 * (max(len(num2), len(den2)) - 1)
            try:
                filtered = signal.filtfilt(
                    num2, den2, filtered, padtype=padtype, padlen=padlen
                )
            except Exception:
                out_data[:, i_col] = np.full(col.shape, np.nan)
                continue

        # restore original NaN positions
        filtered[nan_mask] = np.nan
        out_data[:, i_col] = filtered

    if squeezed:
        out_data = out_data[:, 0]

    out = xr.DataArray(
        out_data,
        coords=inp.coords,
        dims=inp.dims,
        attrs=inp.attrs,
    )

    return out
