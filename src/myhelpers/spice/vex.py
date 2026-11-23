"""SPICE ephemeris helpers for Venus Express."""

from typing import Any, Sequence

import numpy as np
import spiceypy as spice

from myhelpers.parameters.paths import DATA_PTH_REMOTE

DATA_PTH = DATA_PTH_REMOTE
VENUS_RADIUS = 6052  # km


def find_position_vex(timeDeltas: Sequence[Any]) -> np.ndarray:
    """Return Venus Express positions in Venus radii in the VSO frame.

    Args:
        timeDeltas: Datetime-like values at which to evaluate the spacecraft state.

    Returns:
        An array of position vectors, one row per requested time.
    """
    pos = []
    for tDt in timeDeltas:
        et0 = spice.str2et(
            tDt.strftime("%FT%T.%f")
        )  # returns the kernel oberserver epoch based on a calender datetime input
        posval, _ = spice.spkpos(
            targ="VEX", et=et0, ref="VSO", abcorr="LT+S", obs="VENUS"
        )
        # The positions obtained this way are as seen by the observer (earth??)
        pos.append(posval[:3] / VENUS_RADIUS)
    return np.array(pos)
