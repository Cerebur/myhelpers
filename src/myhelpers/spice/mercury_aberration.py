"""Calculate the solar-wind aberration angle at Mercury."""

from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd
from spiceypy import cyice

SOLAR_WIND_SPEED_KM_S = 400.0

DateTimeValue = datetime | pd.Timestamp | np.datetime64 | str
DateTimeInput = (
    DateTimeValue | Sequence[DateTimeValue] | pd.Series | pd.DatetimeIndex | np.ndarray
)


def mercury_aberration_angle(datetime: DateTimeInput) -> list[float]:
    """Calculate the solar-wind aberration angle at Mercury.

    The angle is calculated from Mercury's heliocentric velocity and a radial
    solar wind speed of 400 km/s. The apparent solar wind velocity in Mercury's
    frame is the radial solar wind velocity minus Mercury's orbital velocity.
    The returned angle is the angle between that apparent velocity and the
    Sun-Mercury radial direction.

    Args:
        datetime: One datetime-like value or a sequence of datetime-like
            values. Accepted values are Python or pandas datetimes,
            ``numpy.datetime64`` values, and datetime strings. Naive times are
            interpreted as UTC; timezone-aware times are converted to UTC.

    Returns:
        A list of aberration angles in degrees, in the same order as the input.
        A scalar input returns a one-element list.

    Raises:
        ValueError: If any input cannot be parsed as a datetime or is ``NaT``.

    Note:
        SPICE kernels covering the requested times must already be loaded.
    """
    datetime_array = np.asarray(datetime, dtype=object)
    datetime_flat = datetime_array.reshape(-1)

    if datetime_flat.size == 0:
        return []

    timestamps = pd.to_datetime(datetime_flat, utc=True, errors="raise")
    if np.any(timestamps.isna()):
        raise ValueError("datetime input contains NaT values.")

    utc_strings = np.asarray(timestamps.strftime("%Y-%m-%d %H:%M:%S.%f UTC"), dtype=str)
    et = cyice.str2et(utc_strings)

    # State is Mercury relative to the Sun in the inertial J2000 frame.
    state, _ = cyice.spkezr("MERCURY", et, "J2000", "NONE", "SUN")
    position = state[..., :3]
    velocity = state[..., 3:]

    radial_unit = position / np.linalg.norm(position, axis=-1, keepdims=True)
    radial_velocity = np.sum(velocity * radial_unit, axis=-1)
    transverse_velocity = velocity - radial_velocity[..., np.newaxis] * radial_unit
    transverse_speed = np.linalg.norm(transverse_velocity, axis=-1)

    # In the Mercury frame, the radial solar wind speed is reduced by
    # Mercury's outward radial velocity.
    angle_radians = np.arctan2(
        transverse_speed, SOLAR_WIND_SPEED_KM_S - radial_velocity
    )
    return np.degrees(angle_radians).tolist()
