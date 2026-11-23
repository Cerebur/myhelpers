"""SPICE ephemeris and instrument geometry helpers for BepiColombo."""

from typing import Any, Sequence

import numpy as np
import spiceypy as spice
import xarray as xr

from myhelpers.parameters.paths import DATA_PTH_REMOTE

DATA_PTH = DATA_PTH_REMOTE
VENUS_RADIUS = 6052  # km


def find_position_bc(timeDeltas: Sequence[Any]) -> np.ndarray:
    """Return BepiColombo positions in Venus radii in the BC_VSO frame.

    Args:
        timeDeltas: Datetime-like values at which to evaluate the spacecraft state.

    Returns:
        An array of position vectors, one row per requested time.
    """
    pos = []
    posval: Any
    for tDt in timeDeltas:
        et0 = spice.str2et(
            tDt.strftime("%FT%T.%f")
        )  # returns the kernel oberserver epoch based on a calender datetime input
        posval, _ = spice.spkezr(
            targ="MMO_SPACECRAFT", et=et0, ref="BC_VSO", abcorr="LT+S", obs="VENUS"
        )
        # The positions obtained this way are as seen by the observer (earth??)
        pos.append(posval[:3] / VENUS_RADIUS)
    return np.array(pos)


def determine_orientation_hga_mmo(
    time_interval: Sequence[Any], resolution: str = "min"
) -> xr.DataArray:
    """Calculate the Mio high-gain antenna orientation in the BC_VSO frame.

    Args:
        time_interval: Start and end times in UTC.
        resolution: Sampling cadence, either ``"min"`` or ``"sec"``.

    Returns:
        An orientation DataArray with time and Cartesian component dimensions.
    """
    assert resolution in ("min", "sec")

    start, end = time_interval

    # ---- Build numpy datetime64 time axis directly ----
    if resolution == "min":
        step = np.timedelta64(1, "m")
    else:
        step = np.timedelta64(1, "s")

    times = np.arange(np.datetime64(start), np.datetime64(end), step)

    # ---- Convert times to SPICE ephemeris time in one shot ----
    # SPICE wants ISO-8601 strings
    time_strings = times.astype("datetime64[us]").astype(str)
    ets: Any
    ets = spice.str2et(time_strings)

    # ---- Fixed spacecraft-frame antenna vector ----
    a_sc = np.array([0.0, 0.0, 1.0])

    # ---- Loop only over pxform (C-coded, fast) ----
    a_vso = np.empty((len(ets), 3), dtype=np.float64)

    for i, et in enumerate(ets):
        rot_sc_to_vso = spice.pxform("MMO_SPACECRAFT", "BC_VSO", et)
        a_vso[i] = rot_sc_to_vso @ a_sc

    # ---- Return xarray DataArray ----
    da = xr.DataArray(
        a_vso,
        coords=dict(time=times.astype("datetime64[ns]"), comp=["x", "y", "z"]),
        dims=["time", "comp"],
        name="HGA_orientation",
    )

    return da


def get_ids(sc_name: str, instrument_name: str) -> int:
    """Return the SPICE identifier for a supported BepiColombo instrument.

    Args:
        sc_name: Spacecraft name; currently ``"mmo"`` is supported.
        instrument_name: Instrument key such as ``"mea1"`` or ``"mia"``.

    Returns:
        The NAIF instrument ID.
    """
    inst_dict_mmo = {
        "mea1": -68310,
        "mea1_cruise": -68311,
        "mea2": -68320,
        "mea2_cruise": -68321,
        "mia": -68350,
        "mia_cruise": -68351,
        "mipa": -121520,
        "mipa_cruise": -121520,
    }

    if sc_name.lower() == "mmo":
        if instrument_name in inst_dict_mmo.keys():
            return inst_dict_mmo[instrument_name]
        else:
            raise ValueError(
                f"Instrument not supported. Select one of {inst_dict_mmo.keys()}"
            )
    else:
        raise NotImplementedError("Spacecraft is not supported")
