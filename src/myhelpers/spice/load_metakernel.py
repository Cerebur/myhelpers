"""Loads the correct meta-kernel for a given timestamp."""

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import spiceypy as spice

from myhelpers.parameters import SPICE_PTH_REMOTE

DATA_PTH = SPICE_PTH_REMOTE


def load_bepi_kernel() -> Path:
    """Load the bc_ops kernel of bepi.

    Args:
        None.

    Returns:
        The path to the loaded kernel
    """
    path = Path(DATA_PTH + "bepi/kernels/mk/bc_ops.tm")
    spice.furnsh(str(path))
    return path


def load_vex_kernel() -> Path:
    """Load the VEX_OPS kernel of Venus Express.

    Args:
        None.

    Returns:
        The path to the loaded kernel
    """
    path = Path(DATA_PTH + "vex/kernels/mk/VEX_OPS.tm")
    spice.furnsh(str(path))
    return path


def load_messenger_kernel(
    date_or_year: int | date | datetime | pd.Timestamp,
    mk_dir: str | Path = DATA_PTH + "MESSENGER/extras/mk",
) -> Path:
    """Load the latest available yearly MESSENGER meta-kernel.

    Parameters
    ----------
    date_or_year : int, datetime.date, datetime.datetime or pd.Timestamp
        Year or date for which to load the meta-kernel.
    mk_dir : str or Path
        Directory containing the yearly MESSENGER meta-kernels.

    Returns:
    -------
    Path
        Path to the loaded meta-kernel.
    """
    # Extract year
    if isinstance(date_or_year, int):
        year = date_or_year
    elif isinstance(date_or_year, (date, datetime, pd.Timestamp)):
        year = date_or_year.year
    else:
        raise TypeError("date_or_year must be an integer year or a datetime object.")

    # Find available meta-kernels
    mk_dir = Path(mk_dir)

    if not mk_dir.is_dir():
        raise FileNotFoundError(f"Meta-kernel directory not found: {mk_dir}")

    pattern = re.compile(rf"msgr_{year}_v(\d+)\.tm")

    candidates = []

    for path in mk_dir.glob("*.tm"):
        match = pattern.fullmatch(path.name)
        if match:
            version = int(match.group(1))
            candidates.append((version, path))

    if not candidates:
        raise FileNotFoundError(
            f"No MESSENGER meta-kernel found for {year} in {mk_dir}"
        )

    # Select highest version
    version, kernel_path = max(candidates, key=lambda item: item[0])

    # Load kernel
    kernel_path = kernel_path.resolve()
    print("Loading MESSENGER meta-kernel...")
    spice.furnsh(str(kernel_path))

    print(f"Loaded MESSENGER meta-kernel: {kernel_path.name} (v{version:02d})")

    return kernel_path
