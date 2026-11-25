"""Shared MESSENGER orbit and aberration helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from myhelpers.parameters.paths import DATA_PTH_LOCAL
from myhelpers.transformation.aberration import transform_to_aberrated_coordinates

ORB_DIR = Path(DATA_PTH_LOCAL) / "messenger" / "data" / "orbnum"
ABERRATION_ANGLES_FILE = (
    Path(DATA_PTH_LOCAL) / "derived" / "messenger_orbit_aberration.csv"
)

_ORB_LINE = re.compile(r"^\s*(\d+)\s+(\d{4}\s+[A-Z]{3}\s+\d{2}\s+\d{2}:\d{2}:\d{2})")
_MONTHS = {
    month: f"{number:02d}"
    for number, month in enumerate(
        (
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ),
        start=1,
    )
}


@lru_cache(maxsize=1)
def load_orbit_boundaries() -> pd.DataFrame:
    """Read apoapsis times and orbit numbers from the MESSENGER ``.orb`` files."""
    rows: list[tuple[pd.Timestamp, int]] = []
    for orbit_file in sorted(ORB_DIR.glob("*.orb")):
        with orbit_file.open(encoding="ascii") as stream:
            for line in stream:
                match = _ORB_LINE.match(line)
                if match is None:
                    continue
                orbit_number = int(match.group(1))
                year, month, day, clock = match.group(2).split()
                timestamp = pd.to_datetime(
                    f"{year}-{_MONTHS[month]}-{day} {clock}", utc=True
                )
                rows.append((timestamp, orbit_number))

    if not rows:
        raise FileNotFoundError(f"No orbit-number records found in {ORB_DIR}")
    return (
        pd.DataFrame(rows, columns=["apoapsis", "orbit_number"])
        .drop_duplicates(subset="apoapsis", keep="last")
        .sort_values("apoapsis")
        .reset_index(drop=True)
    )


def get_orbit_numbers(times: pd.Series) -> pd.Series:
    """Find the apoapsis-to-apoapsis orbit containing each UTC timestamp."""
    orbit_boundaries = load_orbit_boundaries()
    requested = pd.DataFrame(
        {
            "datetime": pd.to_datetime(times, utc=True).array,
            "_position": range(len(times)),
        }
    )
    merged = pd.merge_asof(
        requested.sort_values("datetime"),
        orbit_boundaries,
        left_on="datetime",
        right_on="apoapsis",
        direction="backward",
    )
    merged = merged.sort_values("_position")
    return pd.Series(pd.array(merged["orbit_number"], dtype="Int64"), index=times.index)


@lru_cache(maxsize=1)
def _load_aberration_angles() -> pd.Series:
    """Load the aberration angle table once per process."""
    angles = pd.read_csv(ABERRATION_ANGLES_FILE)
    required_columns = {"orbit_number", "aberration_angle_deg"}
    if not required_columns.issubset(angles.columns):
        missing = ", ".join(sorted(required_columns - set(angles.columns)))
        raise ValueError(
            f"Aberration-angle file {ABERRATION_ANGLES_FILE} is missing columns: {missing}."
        )

    return angles.set_index("orbit_number")["aberration_angle_deg"]


def aberrate_vectors(
    vectors: np.ndarray,
    orbit_numbers: Sequence[int] | np.ndarray | pd.Series,
) -> np.ndarray:
    """Rotate MSO vectors using known MESSENGER orbit numbers.

    This is useful when orbit numbers are already available, avoiding a fresh
    lookup against all MESSENGER orbit-boundary files.
    """
    vector_array = np.asarray(vectors, dtype=float)
    orbit_array = pd.Series(orbit_numbers)
    if orbit_array.size != vector_array.shape[0]:
        raise ValueError("orbit_numbers must contain one value per vector.")
    orbit_angles = orbit_array.map(_load_aberration_angles()).to_numpy(
        dtype=float, na_value=np.nan
    )
    return transform_to_aberrated_coordinates(vector_array, orbit_angles)


def aberrate_positions(
    frame: pd.DataFrame,
    position_columns: tuple[str, str, str] = ("X", "Y", "Z"),
) -> pd.DataFrame:
    """Rotate MSO positions using the frame's known orbit numbers."""
    vectors = frame[list(position_columns)].to_numpy(dtype=float)
    transformed = aberrate_vectors(vectors, frame["orbit_number"])

    result = frame.copy()
    result[list(position_columns)] = transformed
    return result


__all__ = [
    "ABERRATION_ANGLES_FILE",
    "ORB_DIR",
    "aberrate_vectors",
    "aberrate_positions",
    "get_orbit_numbers",
    "load_orbit_boundaries",
]
