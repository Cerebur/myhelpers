"""Load Mercury boundary crossing catalogues into a common table."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from spiceypy import cyice

from myhelpers.parameters import DATA_PTH_LOCAL, DATA_PTH_REMOTE
from myhelpers.spice.load_metakernel import load_messenger_kernel
from myhelpers.transformation.aberration import transform_to_aberrated_coordinates
from myhelpers.transformation.coords_trafo_to_msm import mso_to_msm

CTLG_DIR_REMOTE = Path(DATA_PTH_REMOTE) / "derived" / "Mercury_crossing_catalogues"
CTLG_DIR_LOCAL = Path(DATA_PTH_LOCAL) / "derived" / "crossing_catalogues"

_POSSIBLE_DATASETS: dict[str, str | list[str]] = {
    "Winslow2013": "crossings_winslow2013.txt",
    "Philpott2019": "crossings_philpott2019.csv",
    "Hollman2026": "hollman_2026_crossing_list.csv",
    "Zhong2024": [
        "BS_in_Zhong.dat",
        "BS_out_Zhong.dat",
        "MP_in_Zhong.dat",
        "MP_out_Zhong.dat",
    ],
    "Sun2026": [
        "Bow_Shock_In_Time_Duration_public_version_WeijieSun_20260113.txt",
        "Bow_Shock_Out_Time_Duration_public_version_WeijieSun_20260113.txt",
        "MagPause_In_Time_Duration_public_version_WeijieSun_20260113.txt",
        "MagPause_Out_Time_Duration_public_version_WeijieSun_20260113.txt",
    ],
}

_ORB_DIR = Path(DATA_PTH_LOCAL) / "messenger" / "data" / "orbnum"
_ABERRATION_ANGLES_FILE = (
    Path(DATA_PTH_LOCAL) / "derived" / "messenger_orbit_aberration.csv"
)
_CACHE_VERSION = 2
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


def _load_orbit_boundaries() -> pd.DataFrame:
    """Read apoapsis times and orbit numbers from the MESSENGER ``.orb`` files."""
    rows: list[tuple[pd.Timestamp, int]] = []
    for orbit_file in sorted(_ORB_DIR.glob("*.orb")):
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
        raise FileNotFoundError(f"No orbit-number records found in {_ORB_DIR}")
    return (
        pd.DataFrame(rows, columns=["apoapsis", "orbit_number"])
        .drop_duplicates(subset="apoapsis", keep="last")
        .sort_values("apoapsis")
        .reset_index(drop=True)
    )


def _get_orbit_numbers(times: pd.Series) -> pd.Series:
    """Find the apoapsis-to-apoapsis orbit containing each UTC timestamp."""
    orbit_boundaries = _load_orbit_boundaries()
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


def _load_spacecraft_positions(times: pd.Series) -> pd.DataFrame:
    """Calculate MSO positions from SPICE for the requested UTC timestamps."""
    columns = ["x_mso", "y_mso", "z_mso"]
    timestamps = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    positions = np.full((len(timestamps), 3), np.nan, dtype=float)
    valid_indices = np.flatnonzero(~timestamps.isna())
    valid_timestamps = timestamps.take(valid_indices)
    for year in valid_timestamps.year.unique():
        year_mask = valid_timestamps.year == year
        row_indices = valid_indices[year_mask]
        year_times = valid_timestamps[year_mask]
        load_messenger_kernel(int(year))
        utc_strings = np.asarray(
            year_times.strftime("%Y-%m-%d %H:%M:%S.%f UTC"), dtype=str
        )
        et = cyice.str2et(utc_strings)
        year_positions, _ = cyice.spkpos("MESSENGER", et, "MSGR_MSO", "NONE", "MERCURY")
        positions[row_indices] = year_positions

    return pd.DataFrame(positions, columns=columns, index=times.index)


def _aberrate_positions(frame: pd.DataFrame) -> pd.DataFrame:
    """Rotate MSO positions using the aberration angle for each orbit."""
    angles = pd.read_csv(_ABERRATION_ANGLES_FILE)
    required_columns = {"orbit_number", "aberration_angle_deg"}
    if not required_columns.issubset(angles.columns):
        missing = ", ".join(sorted(required_columns - set(angles.columns)))
        raise ValueError(
            f"Aberration-angle file {_ABERRATION_ANGLES_FILE} is missing columns: {missing}."
        )

    angle_by_orbit = angles.set_index("orbit_number")["aberration_angle_deg"]
    orbit_angles = frame["orbit_number"].map(angle_by_orbit).to_numpy(
        dtype=float, na_value=np.nan
    )
    vectors = frame[["x_mso", "y_mso", "z_mso"]].to_numpy(dtype=float)
    transformed = transform_to_aberrated_coordinates(vectors, orbit_angles)

    result = frame.copy()
    result[["x_mso", "y_mso", "z_mso"]] = transformed
    return result


def _pair_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign shared pair identifiers to matching crossing indices per orbit."""
    result = frame.copy()
    pair_ids = pd.Series(pd.NA, index=result.index, dtype="Int64")
    next_pair_id = 1
    for first_index, second_index in ((1, 2), (3, 4), (5, 6), (7, 8)):
        first_rows = result.loc[result["index"].eq(first_index)]
        second_rows = result.loc[result["index"].eq(second_index)]
        for orbit_number, first_group in first_rows.groupby(
            "orbit_number", dropna=False
        ):
            second_group = second_rows.loc[second_rows["orbit_number"].eq(orbit_number)]
            for first_row, second_row in zip(first_group.index, second_group.index):
                pair_ids.loc[[first_row, second_row]] = next_pair_id
                next_pair_id += 1
    result["pair_id"] = pair_ids
    return result


def _paired_timestamp_rows(
    first_time: pd.Timestamp,
    last_time: pd.Timestamp | None,
    crossing_index: int,
    next_pair_id: int,
) -> tuple[list[dict[str, object]], int]:
    """Expand the first/last times from a source record into crossing rows."""
    if pd.isna(first_time):
        return [], next_pair_id
    first_utc = pd.Timestamp(first_time).tz_convert("UTC")
    rows: list[dict[str, object]] = [
        {"index": crossing_index, "datetime": first_utc, "pair_id": pd.NA}
    ]
    if last_time is not None and not pd.isna(last_time):
        last_utc = pd.Timestamp(last_time).tz_convert("UTC")
        if last_utc != first_utc:
            rows[0]["pair_id"] = next_pair_id
            rows.append(
                {
                    "index": crossing_index + 1,
                    "datetime": last_utc,
                    "pair_id": next_pair_id,
                }
            )
            next_pair_id += 1
    return rows, next_pair_id


def _read_winslow(path: Path) -> pd.DataFrame:
    """Read Winslow timestamps, keeping the catalogue's crossing indices."""
    raw = pd.read_csv(path, sep="\t")
    datetimes = (
        pd.to_datetime(
            raw["Year"].astype(str) + "-" + raw["DOY"].astype(str).str.zfill(3),
            format="%Y-%j",
            utc=True,
        )
        + pd.to_timedelta(raw["Hour"], unit="h")
        + pd.to_timedelta(raw["Minute"], unit="m")
        + pd.to_timedelta(raw["Second"], unit="s")
    )
    frame = pd.DataFrame(
        {
            "index": raw["Index"].astype(int),
            "datetime": datetimes,
            "orbit_number": raw["Orbit#"],
        }
    )
    frame["orbit_number"] = frame["orbit_number"].astype("Int64")
    return _pair_rows(frame)


def _read_philpott(path: Path) -> pd.DataFrame:
    """Read Philpott timestamps, including its data-gap boundary records."""
    raw = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    datetimes = (
        pd.to_datetime(
            raw["Year"].astype(str) + "-" + raw["Day of year"].astype(str).str.zfill(3),
            format="%Y-%j",
            utc=True,
        )
        + pd.to_timedelta(raw["Hour"], unit="h")
        + pd.to_timedelta(raw["Minute"], unit="m")
        + pd.to_timedelta(raw["Second"], unit="s")
    )
    frame = pd.DataFrame(
        {
            "index": raw["Boundary number"].astype(int),
            "datetime": datetimes,
            "orbit_number": raw["Orbit number"].astype("Int64"),
        }
    )
    return _pair_rows(frame)


def _read_hollman(path: Path) -> pd.DataFrame:
    """Read the Hollman catalogue, which does not specify crossing pairs."""
    raw = pd.read_csv(path)
    label_to_index = {"BS_IN": 1, "BS_OUT": 7, "MP_IN": 3, "MP_OUT": 5}
    frame = pd.DataFrame(
        {
            "index": raw["Label"].map(label_to_index).astype("Int64"),
            "datetime": pd.to_datetime(raw["Time"], utc=True),
            "pair_id": pd.Series(pd.NA, index=raw.index, dtype="Int64"),
        }
    )
    return frame


def _read_zhong(directory: Path) -> pd.DataFrame:
    """Read Zhong's first/last crossing timestamps, preserving source pairing."""
    index_by_file = {
        "BS_in_Zhong.dat": 1,
        "BS_out_Zhong.dat": 7,
        "MP_in_Zhong.dat": 3,
        "MP_out_Zhong.dat": 5,
    }
    rows: list[dict[str, object]] = []
    next_pair_id = 1
    for filename, crossing_index in index_by_file.items():
        raw = pd.read_csv(
            directory / filename, sep=r"\s+", header=None, names=["first", "last"]
        )
        for first_text, last_text in raw.itertuples(index=False, name=None):
            first_time = pd.to_datetime(
                first_text, format="%Y-%m-%d/%H:%M:%S", errors="coerce", utc=True
            )
            last_time = pd.to_datetime(
                last_text, format="%Y-%m-%d/%H:%M:%S", errors="coerce", utc=True
            )
            expanded, next_pair_id = _paired_timestamp_rows(
                first_time,
                None if pd.isna(last_time) else last_time,
                crossing_index,
                next_pair_id,
            )
            rows.extend(expanded)
    return pd.DataFrame(rows, columns=["index", "datetime", "pair_id"])


def _read_sun(directory: Path) -> pd.DataFrame:
    """Read Sun's first/last crossing timestamps, preserving source pairing."""
    files_and_indices = {
        "Bow_Shock_In_Time_Duration_public_version_WeijieSun_20260113.txt": 1,
        "Bow_Shock_Out_Time_Duration_public_version_WeijieSun_20260113.txt": 7,
        "MagPause_In_Time_Duration_public_version_WeijieSun_20260113.txt": 3,
        "MagPause_Out_Time_Duration_public_version_WeijieSun_20260113.txt": 5,
    }
    rows: list[dict[str, object]] = []
    next_pair_id = 1
    for filename, crossing_index in files_and_indices.items():
        raw = pd.read_csv(
            directory / filename, sep=r"\s+", header=None, engine="python"
        )
        for record in raw.itertuples(index=False, name=None):
            first_text = " ".join(str(item) for item in record[:6])
            last_text = " ".join(str(item) for item in record[6:12])
            first_time = pd.to_datetime(
                first_text, format="%Y %m %d %H %M %S.%f", errors="coerce", utc=True
            )
            last_time = pd.to_datetime(
                last_text, format="%Y %m %d %H %M %S.%f", errors="coerce", utc=True
            )
            expanded, next_pair_id = _paired_timestamp_rows(
                first_time,
                None if pd.isna(last_time) else last_time,
                crossing_index,
                next_pair_id,
            )
            rows.extend(expanded)
    return pd.DataFrame(rows, columns=["index", "datetime", "pair_id"])


def load_crossing_catalogue(
    dataset: str,
    local: bool = True,
    resave: bool = False,
    aberrated: bool = False,
    coords: str = "mso",
) -> pd.DataFrame:
    """Load a Mercury crossing catalogue in a consistent tabular format.

    The returned table has one row per crossing and the columns ``index``,
    ``datetime``, ``orbit_number``, ``pair_id``, ``x_mso``, ``y_mso``, and
    ``z_mso``. Datetimes are UTC-aware. Position columns contain MSO
    coordinates by default, or MSM coordinates when ``coords="msm"``. Positions
    are missing when the archive does not cover a crossing time.
    ``index`` follows the Winslow/Philpott convention: 1/2 are inbound bow
    shock, 3/4 inbound magnetopause, 5/6 outbound magnetopause, and 7/8
    outbound bow shock (outer/inner ordering follows the source index scheme).
    ``pair_id`` is shared only by two crossings belonging to the same pair.
    Hollman records have no pair information and therefore use missing values.
    If ``aberrated`` is ``True``, the MSO positions are rotated using the
    orbit-specific angles in ``derived/messenger_orbit_aberration.csv``.
    ``coords`` selects the position coordinate system and accepts ``"MSO"``
    or ``"MSM"`` (case-insensitive). The position column names are retained
    when returning MSM values.

    Args:
        dataset: Catalogue name: ``Winslow2013``, ``Philpott2019``,
            ``Hollman2026``, ``Zhong2024``, or ``Sun2026``.
        local: If ``True``, read the catalogue from the local data directory;
            otherwise read it from the remote catalogue directory.
        resave: If ``True``, rebuild and overwrite the cached Parquet dataframe.
            By default an existing cache is loaded; if no cache exists, it is
            created before returning the dataframe.
        aberrated: If ``True``, rotate the MSO positions into aberrated
            coordinates using the angle associated with each row's orbit number.
        coords: Coordinate system for the returned positions: ``"mso"`` or
            ``"msm"``. The default is ``"mso"``.

    Returns:
        A DataFrame with the catalogue columns and spacecraft position in the
        requested coordinate system. Position columns retain the ``x_mso``,
        ``y_mso``, and ``z_mso`` names. Missing orbit and pair numbers are
        nullable integers.

    Raises:
        ValueError: If ``dataset`` or ``coords`` is unsupported, or the
            aberration-angle file does not have the expected columns.
        FileNotFoundError: If a source catalogue, orbit-number file, or
            aberration-angle file is absent.
    """
    if dataset not in _POSSIBLE_DATASETS:
        supported = ", ".join(_POSSIBLE_DATASETS)
        raise ValueError(f"Unknown catalogue {dataset!r}. Choose from: {supported}.")
    coordinate_system = coords.lower()
    if coordinate_system not in {"mso", "msm"}:
        raise ValueError("coords must be either 'MSO' or 'MSM'.")

    directory = CTLG_DIR_LOCAL if local else CTLG_DIR_REMOTE
    cache_path = directory / f"{dataset}_v{_CACHE_VERSION}.parquet"
    if cache_path.is_file() and not resave:
        frame = pd.read_parquet(cache_path)
    else:
        files = _POSSIBLE_DATASETS[dataset]
        if isinstance(files, str):
            path = directory / files
            if dataset == "Winslow2013":
                frame = _read_winslow(path)
            elif dataset == "Philpott2019":
                frame = _read_philpott(path)
            else:
                frame = _read_hollman(path)
        elif dataset == "Zhong2024":
            frame = _read_zhong(directory)
        else:
            frame = _read_sun(directory)

        if "orbit_number" not in frame:
            frame["orbit_number"] = _get_orbit_numbers(frame["datetime"])
        else:
            missing_orbits = frame["orbit_number"].isna()
            if missing_orbits.any():
                frame.loc[missing_orbits, "orbit_number"] = _get_orbit_numbers(
                    frame.loc[missing_orbits, "datetime"]
                )
        if "pair_id" not in frame:
            frame["pair_id"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")

        frame["index"] = frame["index"].astype("Int64")
        frame["orbit_number"] = frame["orbit_number"].astype("Int64")
        frame["pair_id"] = frame["pair_id"].astype("Int64")
        frame = frame[["index", "datetime", "orbit_number", "pair_id"]].sort_values(
            "datetime", ignore_index=True
        )
        frame = pd.concat([frame, _load_spacecraft_positions(frame["datetime"])], axis=1)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache_path = cache_path.with_suffix(".tmp.parquet")
        frame.to_parquet(temporary_cache_path, index=False)
        temporary_cache_path.replace(cache_path)

    if aberrated:
        frame = _aberrate_positions(frame)
    if coordinate_system == "msm":
        frame[["x_mso", "y_mso", "z_mso"]] = mso_to_msm(
            frame[["x_mso", "y_mso", "z_mso"]].to_numpy()
        )
    return frame
