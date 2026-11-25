"""Local implementations of the pyrfu routines used by myhelpers.

These functions are adapted from pyrfu 2.4.21 (MIT; copyright (c) 2020-2026
L. Richard). See ``THIRD_PARTY_NOTICES.md`` at the project root.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

import numpy as np
import xarray as xr
from scipy import fft
from scipy import signal

from myhelpers.transformation.filt import filt

logger = logging.getLogger(__name__)


def mva(
    inp: xr.DataArray, flag: str = "mvar"
) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """Compute pyrfu-compatible minimum-variance coordinates (default MVAR)."""
    assert flag.lower() in ["mvar", "<bn>=0", "td"], "invalid method!!"
    data = inp.data
    n_t = data.shape[0]
    idx_1, idx_2 = [[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    moments = np.mean(data[:, idx_1] * data[:, idx_2], axis=0)
    if flag.lower() in ["mvar", "<bn>=0"]:
        moments -= np.mean(data, axis=0)[idx_1] * np.mean(data, axis=0)[idx_2]
    matrix = np.array([moments[[0, 3, 4]], moments[[3, 1, 5]], moments[[4, 5, 2]]])
    values, vectors = np.linalg.eig(matrix)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    vectors[:, 2] = np.cross(vectors[:, 0], vectors[:, 1])

    if flag.lower() in ["<bn>=0", "td"]:
        # These constrained modes are retained for API compatibility with
        # pyrfu's implementation.
        if flag.lower() == "<bn>=0":
            mean_projection = np.mean(
                np.sum(
                    np.tile(data, (3, 1, 1))
                    * np.transpose(np.tile(vectors, (n_t, 1, 1)), (2, 0, 1)),
                    axis=1,
                ),
                axis=1,
            )
            coeffs = np.zeros(3)
            coeffs[0] = np.sum(mean_projection**2)
            coeffs[1] = -(values[1] + values[2]) * mean_projection[0] ** 2
            coeffs[1] -= (values[0] + values[2]) * mean_projection[1] ** 2
            coeffs[1] -= (values[0] + values[1]) * mean_projection[2] ** 2
            coeffs[2] = values[1] * values[2] * mean_projection[0] ** 2
            coeffs[2] += values[0] * values[2] * mean_projection[1] ** 2
            coeffs[2] += values[0] * values[1] * mean_projection[2] ** 2
            l_min = np.min(np.roots(coeffs))
            normal = mean_projection / (values - l_min)
            normal /= np.linalg.norm(normal, keepdims=True)
            normal = vectors @ normal
        else:
            l_min = values[2]
            normal = vectors[:, 2]

        projected = np.sum(data * np.tile(normal, (n_t, 1)), axis=1)
        residual = data - np.tile(projected, (3, 1)).T * np.tile(normal, (n_t, 1))
        residual_mean = np.mean(residual, axis=0)
        moments = np.mean(residual[:, idx_1] * residual[:, idx_2], axis=0)
        moments -= residual_mean[idx_1] * residual_mean[idx_2]
        matrix = np.array([moments[[0, 3, 4]], moments[[3, 1, 5]], moments[[4, 5, 2]]])
        values, vectors = np.linalg.eig(matrix)
        order = values.argsort()[::-1]
        values, vectors = values[order], vectors[:, order]
        if flag.lower() == "td":
            values[2], vectors[:, 2] = l_min, np.cross(vectors[:, 0], vectors[:, 1])

    out = xr.DataArray((vectors.T @ data.T).T, coords=inp.coords, dims=inp.dims)
    return out, values, vectors


def _sampling_frequency(inp: xr.DataArray) -> float:
    times = inp.time.data.astype("datetime64[ns]").astype(np.int64)
    return 1e9 / np.median(np.diff(times))


def psd(
    inp: xr.DataArray,
    n_fft: int = 256,
    n_overlap: int | None = 128,
    window: str = "hamming",
    d_flag: str = "constant",
    scaling: str = "density",
) -> xr.DataArray:
    """Estimate a time series' power spectral density using Welch's method.

    The first dimension of ``inp`` must be time. Welch's estimate is computed
    independently for each value in any remaining dimensions.

    Args:
        inp: Time series of scalar, vector, or tensor measurements.
        n_fft: FFT length. It is increased when shorter than a segment.
        n_overlap: Number of samples overlapped between segments. The segment
            length is twice this value. ``None`` uses 256-sample segments with
            128 samples of overlap.
        window: Window passed to :func:`scipy.signal.welch`.
        d_flag: Detrending method passed to :func:`scipy.signal.welch`.
        scaling: Either ``"density"`` or ``"spectrum"``.

    Returns:
        A DataArray with frequency as its first dimension, followed by the
        input's remaining dimensions. NaNs propagate into the corresponding
        spectra.

    Notes:
        If you do not want your data to be split into different windows, use n_fft = n_samples, n_overlap = n_samples // 2, where n_samples is the input data length.
    """
    if not isinstance(inp, xr.DataArray):
        raise TypeError("inp must be an xarray.DataArray")
    if inp.ndim < 1 or inp.dims[0] != "time":
        raise ValueError("the first input dimension must be 'time'")

    if n_overlap is None:
        n_persegs = 256
        n_overlap = n_persegs // 2
    else:
        n_persegs = 2 * n_overlap

    if n_fft < n_persegs:
        logger.warning("nfft < n_persegs. set to n_persegs")
        n_fft = n_persegs

    freqs, p_xx = signal.welch(
        inp.data,
        nfft=n_fft,
        fs=_sampling_frequency(inp),
        window=window,
        noverlap=n_overlap,
        detrend=d_flag,
        nperseg=n_persegs,
        scaling=scaling,
        return_onesided=True,
        axis=0,
    )

    coords = [freqs, *[inp.coords[dim].data for dim in inp.dims[1:]]]
    dims = ["f", *inp.dims[1:]]
    attrs: dict[str, str] = {}
    units = inp.attrs.get("UNITS", inp.attrs.get("units"))
    if units:
        power_units = f"{units}^2"
        attrs["UNITS"] = f"{power_units}/Hz" if scaling == "density" else power_units

    return xr.DataArray(p_xx, coords=coords, dims=dims, attrs=attrs)


def wavelet(
    inp: xr.DataArray,
    f_s: float | None = None,
    f: list[float] | None = None,
    n_freqs: int | None = None,
    linear: float | bool | None = None,
    wavelet_width: float | None = None,
    cut_edge: bool | None = True,
    return_power: bool | None = True,
) -> xr.DataArray | xr.Dataset:
    """Compute pyrfu-compatible FFT Morlet wavelet power or coefficients."""
    if not isinstance(inp, xr.DataArray):
        raise TypeError("Input must be a DataArray")
    f_s = _sampling_frequency(inp) if f_s is None else f_s
    n_freqs = 200 if n_freqs is None else n_freqs
    wavelet_width = 5.36 if wavelet_width is None else wavelet_width
    if linear is not None:
        if isinstance(linear, float):
            delta_f, linear_df = linear, True
        elif isinstance(linear, bool) and linear:
            delta_f, linear_df = 100.0, True
        else:
            raise TypeError("linear keyword argument must be bool or float")
    else:
        delta_f, linear_df = 100.0, False

    f_nyq = f_s / 2
    sigma = wavelet_width / f_nyq
    f_min, f_max = (f_nyq / 1e2, f_nyq / 1e-2) if f is None else sorted(f)
    if linear_df:
        scale_number = int(np.floor(f_nyq / delta_f))
        scales = f_nyq / np.linspace(scale_number * delta_f, delta_f, scale_number)
    else:
        scale_number = n_freqs
        scales = np.logspace(
            np.log10(f_nyq / f_max), np.log10(f_nyq / f_min), scale_number
        )

    if len(inp.time.data) % 2:
        time, data = inp.time.data[:-1], inp.data[:-1, ...].astype(np.float64)
    else:
        time, data = inp.time.data, inp.data.astype(np.float64)
    scales[np.isnan(scales)] = 0.0
    freq = f_nyq * np.arange(1, 1 + len(data) / 2) / (len(data) / 2)
    freqs_fft = np.hstack([0, freq, -np.flip(freq[:-1])])
    freqs_cwt = f_nyq / scales
    if data.ndim not in (1, 2):
        raise ValueError("Input data must be 1D or 2D")
    if data.ndim == 1:
        data = data[:, np.newaxis]
    out_dict: dict[str, Any] = {}
    for i in range(data.shape[1]):
        spectrum = fft.fft(data[:, i], workers=os.cpu_count())
        wave = spectrum[:, None] * np.exp(
            -sigma * sigma * ((scales[None, :] * freqs_fft[:, None] - f_nyq) ** 2) / 2
        )
        power = fft.ifft(wave, axis=0, workers=os.cpu_count())
        if return_power:
            result = np.abs((2 * np.pi) * np.conj(power) * power / freqs_cwt[None, :])
        else:
            result = np.sqrt(np.abs((2 * np.pi) / freqs_cwt[None, :])) * power
        if cut_edge:
            censure = np.floor(2 * scales).astype(int)
            for j in range(scale_number):
                result[: censure[j], j] = np.nan
                result[len(data[:, i]) - censure[j] :, j] = np.nan
        result = np.fliplr(result)
        if inp.ndim == 2:
            out_dict[str(inp.comp.data[i])] = (["time", "frequency"], result)
        else:
            single = result
    if inp.ndim == 1:
        return xr.DataArray(
            single, coords=[time, np.flip(freqs_cwt)], dims=["time", "frequency"]
        )
    return xr.Dataset(out_dict, coords={"time": time, "frequency": np.flip(freqs_cwt)})


def _compress_cwt_1d(cwt: np.ndarray, nc: int = 100) -> np.ndarray:
    nf = cwt.shape[1]
    indices = np.arange(int(nc / 2), len(cwt) - int(nc / 2), step=nc, dtype=np.int64)
    compressed = np.zeros((len(indices), nf))
    for i, index in enumerate(indices):
        for j in range(nf):
            compressed[i, j] = np.nanmean(
                cwt[index - int(nc / 2) : index + int(nc / 2), j]
            )
    return compressed


@lru_cache(maxsize=1)
def _numba_compressor() -> Any:
    """Create pyrfu's compiled reducer only when compressing NaN data."""
    import numba

    return numba.jit(cache=True, fastmath=True, nopython=True, parallel=True)(
        _compress_cwt_1d
    )


def compress_cwt(cwt: xr.Dataset, nc: int = 100) -> tuple[np.ndarray, ...]:
    """Average CWT samples in pyrfu-compatible windows."""
    assert isinstance(cwt, xr.Dataset), "cwt must be an xarray.Dataset"
    indices = np.arange(
        int(nc / 2), len(cwt.time.data) - int(nc / 2), step=nc, dtype=np.int64
    )
    compressor = _compress_cwt_1d
    if any(np.isnan(cwt[name].data).any() for name in ("x", "y", "z")):
        compressor = _numba_compressor()
    return (
        cwt.time.data[indices],
        compressor(cwt.x.data, nc),
        compressor(cwt.y.data, nc),
        compressor(cwt.z.data, nc),
    )


__all__ = ["compress_cwt", "filt", "mva", "psd", "wavelet"]
