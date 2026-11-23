"""Calculates the heliocentric distance between Sun and planets."""

import numpy as np
import pandas as pd
from spiceypy import cyice


def planet_heliocentric_distance(
    datetime: pd.Timestamp | pd.DatetimeIndex | np.ndarray | list,
    planet: str,
    unit: str = "AU",
) -> float | np.ndarray:
    """Calculate the heliocentric distance of a planet.

    Supports individual timestamps and arrays of arbitrary shape.

    Parameters
    ----------
    datetime : pd.Timestamp or array-like
        Observation time(s). Naive timestamps are assumed to be UTC.
        Timezone-aware timestamps are converted to UTC.
    planet : str
        Planet whose distance to the Sun should be calculated.
    unit : str
        Output unit: "AU" or "km".

    Returns:
    -------
    float or np.ndarray
        Heliocentric distance. Array output has the same shape as the
        input datetime array.
    """
    # Validate output unit
    unit = unit.upper()
    if unit not in ("AU", "KM"):
        raise ValueError("unit must be either 'AU' or 'km'")

    # Convert input to NumPy array and preserve original shape
    datetime_array = np.asarray(datetime, dtype=object)
    original_shape = datetime_array.shape
    is_scalar = datetime_array.ndim == 0

    # Flatten for vectorized Cyice functions
    datetime_flat = datetime_array.reshape(-1)

    # Convert to UTC timestamps
    timestamps = pd.to_datetime(datetime_flat, utc=True)

    if np.any(timestamps.isna()):
        raise ValueError("datetime input contains NaT values.")

    # Convert pandas timestamps to SPICE-compatible UTC strings
    utc_strings = np.asarray(
        timestamps.strftime("%Y-%m-%d %H:%M:%S.%f UTC"),
        dtype=str,
    )

    # Convert UTC to SPICE ephemeris time (ET/TDB)
    et = cyice.str2et(utc_strings)

    # Calculate planetary positions relative to the Sun
    position, _ = cyice.spkpos(
        planet.upper(),
        et,
        "J2000",
        "NONE",
        "SUN",
    )

    # Calculate Euclidean distances
    distance_km = np.linalg.norm(position, axis=-1)

    # Convert units
    if unit == "AU":
        distance = distance_km / 149597870.7
    else:
        distance = distance_km

    # Restore original input shape
    distance = distance.reshape(original_shape)

    # Return scalar for scalar input
    if is_scalar:
        return float(distance)

    return distance

