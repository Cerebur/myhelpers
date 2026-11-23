"""Download MESSENGER MAG-3-CDR and FIPS-derived data."""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from myhelpers.parameters.paths import DATA_PTH_REMOTE

DateInput: TypeAlias = (
    str
    | datetime
    | date
    | pd.Timestamp
    | pd.DatetimeIndex
    | list[str | datetime | date | pd.Timestamp]
)

Instrument = Literal["MAG", "FIPS"]
MAGProduct = Literal["SCI", "LAC"]
FIPSProduct = Literal[
    "ERPCHANG",
    "ESPEC",
    "FLUXMAP",
    "NOBS",
    "NTP",
    "PCHANG",
    "ROTMSO",
]
MAGCoordinates = Literal[
    "J2000",
    "MBF",
    "MSO",
    "RTN",
    "SPACECRAFT",
    "VSO",
]

MAG_BASE_URL = (
    "https://pds-ppi.igpp.ucla.edu/data/MESS-E_V_H_SW-MAG-3-CDR-CALIBRATED-V1.0/DATA"
)

FIPS_BASE_URL = "https://pds-ppi.igpp.ucla.edu/data/mess-epps-fips-derived"

MAG_COORDINATE_PREFIXES = {
    "J2000": "J2K",
    "MBF": "MBF",
    "MSO": "MSO",
    "RTN": "RTN",
    "SPACECRAFT": "SC",
    "VSO": "VSO",
}

DEFAULT_DOWNLOAD_PATH = Path(DATA_PTH_REMOTE) / "MESSENGER" / "data"


def _normalize_dates(values: DateInput) -> list[pd.Timestamp]:
    """Convert supported date inputs into UTC-normalized timestamps."""
    if isinstance(values, pd.DatetimeIndex):
        timestamps = values
    else:
        if isinstance(values, (str, datetime, date, pd.Timestamp)):
            values = [values]

        timestamps = pd.DatetimeIndex([pd.Timestamp(value) for value in values])

    if timestamps.tz is not None:
        timestamps = timestamps.tz_convert("UTC").tz_localize(None)

    return list(timestamps)


def _month_directory(timestamp: pd.Timestamp) -> str:
    """Return the PDS directory name for a given month."""
    year = timestamp.year
    month = timestamp.month

    first_day = date(year, month, 1)
    last_day = date(year, month, calendar.monthrange(year, month)[1])

    first_doy = first_day.timetuple().tm_yday
    last_doy = last_day.timetuple().tm_yday
    month_name = calendar.month_abbr[month].upper()

    return f"{first_doy:03d}_{last_doy:03d}_{month_name}"


def _download_file(url: str, destination: Path) -> None:
    """Download a file atomically, removing incomplete downloads."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = destination.with_suffix(".TAB.part")

    request = Request(
        url,
        headers={"User-Agent": "Python MESSENGER MAG data downloader"},
    )

    try:
        with urlopen(request, timeout=60) as response:
            with temporary_file.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)

        temporary_file.replace(destination)

    except (HTTPError, URLError, OSError):
        temporary_file.unlink(missing_ok=True)
        raise


class _DirectoryParser(HTMLParser):
    """Extract links from an Apache-style directory listing."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def _list_remote_tab_files(url: str) -> list[str]:
    """Return TAB filenames available in a remote directory."""
    request = Request(
        url,
        headers={"User-Agent": "Python MESSENGER data downloader"},
    )

    with urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8", errors="replace")

    parser = _DirectoryParser()
    parser.feed(html)

    return [
        link
        for link in parser.links
        if link.upper().endswith(".TAB") and "/" not in link
    ]


def _get_day_from_filename(
    filename: str,
    day: pd.Timestamp,
    instrument: Instrument,
) -> bool:
    """Check whether a filename contains the requested observation day."""
    if instrument == "MAG":
        date_code = f"{day.year % 100:02d}{day.dayofyear:03d}"
    else:
        date_code = f"{day.year:04d}{day.dayofyear:03d}"

    return bool(
        re.search(
            rf"(?<!\d){date_code}(?!\d)",
            filename,
        )
    )


def _get_latest_version(filenames: list[str]) -> str:
    """Return the highest-version product filename."""

    def version(filename: str) -> int:
        match = re.search(r"_V(\d+)\.TAB$", filename, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    return max(filenames, key=version)


def _get_remote_directory(
    day: pd.Timestamp,
    instrument: Instrument,
    data_product: str,
    coordinate_system: str | None,
) -> tuple[str, Path]:
    """Construct the remote directory and local relative directory."""
    year = str(day.year)
    month = _month_directory(day)

    if instrument == "MAG":
        assert coordinate_system is not None

        remote = f"{MAG_BASE_URL}/{coordinate_system}/{year}/{month}"

        relative = Path(
            "MAG-3-CDR-CALIBRATED",
            coordinate_system,
            data_product,
            year,
            month,
        )

    else:
        product_directory = (
            "rotmo" if data_product == "ROTMSO" else data_product.lower()
        )

        remote = f"{FIPS_BASE_URL}/data-fips-{product_directory}/{year}/{month}"

        relative = Path(
            "FIPS-DDR",
            data_product,
            year,
            month,
        )

    return remote, relative


def _get_required_files(
    day: pd.Timestamp,
    instrument: Instrument,
    data_product: str,
    coordinate_system: str | None,
) -> list[tuple[str, Path]]:
    """Find available data files for one UTC calendar day."""
    remote_directory, relative_directory = _get_remote_directory(
        day,
        instrument,
        data_product,
        coordinate_system,
    )

    try:
        filenames = _list_remote_tab_files(remote_directory + "/")
    except HTTPError as exc:
        if exc.code == 404:
            return []
        raise

    if instrument == "MAG":
        if coordinate_system is None:
            coordinate_system = "MSO"
        prefix = MAG_COORDINATE_PREFIXES[coordinate_system]
        product_pattern = re.compile(
            rf"^MAG{prefix}_?{data_product}",
            re.IGNORECASE,
        )

        matching_files = [
            filename
            for filename in filenames
            if product_pattern.search(filename)
            and _get_day_from_filename(filename, day, instrument)
        ]
    else:
        matching_files = [
            filename
            for filename in filenames
            if _get_day_from_filename(filename, day, instrument)
        ]

    if not matching_files:
        return []

    # Avoid downloading multiple archive versions of the same product.
    filename = _get_latest_version(matching_files)

    url = f"{remote_directory}/{filename}"
    local_path = relative_directory / filename

    return [(url, local_path)]


def download_messenger_data(
    start: DateInput,
    end: DateInput,
    instrument: Instrument = "MAG",
    data_product: MAGProduct | FIPSProduct = "SCI",
    coordinate_system: MAGCoordinates | None = None,
    download_path: str | Path = DEFAULT_DOWNLOAD_PATH,
    overwrite: bool = False,
) -> list[Path]:
    """Download MESSENGER MAG or FIPS science data from PDS PPI.

    Downloads daily data products for specified time intervals. Supports
    the MAG-3-CDR calibrated bundle and the FIPS-derived DDR bundle.

    MAG supports SCI and LAC products in J2000, MBF, MSO, RTN,
    SPACECRAFT and VSO coordinates.

    FIPS supports the ERPCHANG, ESPEC, FLUXMAP, NOBS, NTP, PCHANG
    and ROTMSO derived data collections. ROTMSO is the only
    coordinate-specific FIPS collection.

    Only files overlapping the requested UTC calendar days are considered.
    The remote directory is inspected to identify files that actually exist.
    Existing local files are skipped unless ``overwrite=True``.

    If more than five files need to be downloaded for an individual
    start-end pair, terminal confirmation is requested.

    Args:
        start: Start datetime or sequence of start datetimes.
            Accepts strings, Python datetime/date objects, pandas
            Timestamps, DatetimeIndex or lists of supported values.
        end: End datetime or sequence of end datetimes. Must contain
            the same number of entries as ``start``.
        instrument: Instrument/data bundle to download: MAG or FIPS.
        data_product: MAG product (SCI or LAC) or FIPS derived
            product (ERPCHANG, ESPEC, FLUXMAP, NOBS, NTP, PCHANG,
            ROTMSO).
        coordinate_system: MAG coordinate system. Defaults to MSO
            when instrument="MAG". For FIPS, leave as None.
        download_path: Root directory for downloaded files.
            Defaults to DATA_PTH_REMOTE/MESSENGER/data.
        overwrite: If True, download files again even if they exist.

    Returns:
        List of paths to all downloaded or already existing files
        corresponding to the requested intervals.

    Raises:
        ValueError: If the datetime intervals or product selections
            are invalid.
        HTTPError: If the PDS server returns an HTTP error other
            than 404.
        URLError: If the PDS server cannot be reached.
    """
    starts = _normalize_dates(start)
    ends = _normalize_dates(end)

    if len(starts) != len(ends):
        raise ValueError("start and end must contain the same number of datetimes.")

    if not starts:
        return []

    capital_instrument: str = instrument.upper()

    if capital_instrument not in ("MAG", "FIPS"):
        raise ValueError("instrument must be 'MAG' or 'FIPS'.")

    selected_instrument: Instrument = "MAG" if capital_instrument == "MAG" else "FIPS"

    if selected_instrument == "MAG":
        if data_product not in ("SCI", "LAC"):
            raise ValueError("MAG data_product must be 'SCI' or 'LAC'.")

        assigned_coordinate_system = (coordinate_system or "MSO").upper()

        if assigned_coordinate_system not in MAG_COORDINATE_PREFIXES:
            raise ValueError(
                f"Unsupported MAG coordinate system: {assigned_coordinate_system}."
            )

    else:
        if data_product not in (
            "ERPCHANG",
            "ESPEC",
            "FLUXMAP",
            "NOBS",
            "NTP",
            "PCHANG",
            "ROTMSO",
        ):
            raise ValueError("Unsupported FIPS data product.")

        if coordinate_system is not None:
            raise ValueError(
                "FIPS derived products do not accept a separate "
                "coordinate_system argument. ROTMSO is a distinct "
                "data product."
            )
        else:
            assigned_coordinate_system = coordinate_system

    for start_time, end_time in zip(starts, ends):
        if start_time > end_time:
            raise ValueError(
                f"Start datetime {start_time} is after end datetime {end_time}."
            )

    download_path = Path(download_path).expanduser()
    download_path.mkdir(parents=True, exist_ok=True)

    downloaded_files: list[Path] = []

    for start_time, end_time in zip(starts, ends):
        days = pd.date_range(
            start=start_time.normalize(),
            end=end_time.normalize(),
            freq="D",
        )

        required_files: list[tuple[str, Path]] = []

        for day in days:
            required_files.extend(
                _get_required_files(
                    day=day,
                    instrument=instrument,
                    data_product=data_product,
                    coordinate_system=assigned_coordinate_system,
                )
            )

        # Convert relative paths to absolute local paths.
        required_files = [(url, download_path / path) for url, path in required_files]

        # Remove duplicates while preserving order.
        required_files = list(
            dict.fromkeys((url, path) for url, path in required_files)
        )

        pending_files = [
            (url, path)
            for url, path in required_files
            if overwrite or not path.is_file()
        ]

        if len(pending_files) > 5:
            print(
                f"\nInstrument: {instrument}"
                f"\nProduct: {data_product}"
                f"\nCoordinate system: {assigned_coordinate_system or 'N/A'}"
                f"\nInterval: {start_time} -- {end_time}"
                f"\nFiles to download: {len(pending_files)}"
            )

            response = input("Proceed with download? [y/N]: ").strip().lower()

            if response not in {"y", "yes"}:
                print("Skipping this interval.")
                continue

        for url, path in pending_files:
            print(f"Downloading {path.name} ...")
            _download_file(url, path)

        downloaded_files.extend(path for _, path in required_files if path.is_file())

    return list(dict.fromkeys(downloaded_files))
