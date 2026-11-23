"""Load Venus Express Electron Spectrometer observations."""

import glob
import re
from datetime import datetime
from typing import Any

import pandas as pd
import xarray as xr

from myhelpers.parameters.paths import DATA_PTH_LOCAL, DATA_PTH_REMOTE

data_local = False
if data_local:
    DATA_PTH = DATA_PTH_LOCAL
else:
    DATA_PTH = DATA_PTH_REMOTE


def load_ELS_data_VEx(time_interval: list[Any]) -> Any:
    """Load Venus Express ELS flux data over a time interval.

    Args:
        time_interval: Mutable list containing the interval start and end times.

    Returns:
        The selected ``data_3d`` DataArray and paths of the source files.
    """
    if (
        type(time_interval[0]) is not pd.Timestamp
        or type(time_interval[1]) is not pd.Timestamp
    ):
        print("Not a datetime format, will convert")
        time_interval[0] = pd.to_datetime(time_interval[0])
        time_interval[1] = pd.to_datetime(time_interval[1])

    all_files_ele = glob.glob(DATA_PTH + "ASPERA4/vex_els_flux__*.nc")

    pattern = re.compile(r"__(\d{8})_(\d{6})_(\d{8})_(\d{6})")

    selected_files_ele = []

    for f in all_files_ele:
        match = pattern.search(f)
        if not match:
            continue

        start_str = match.group(1) + match.group(2)
        end_str = match.group(3) + match.group(4)

        file_start = datetime.strptime(start_str, "%Y%m%d%H%M%S")
        file_end = datetime.strptime(end_str, "%Y%m%d%H%M%S")

        # overlap condition:
        if file_start <= time_interval[1] and file_end >= time_interval[0]:
            selected_files_ele.append(f)
            print("Loading file: ", f)

    datasets_ele = [xr.open_dataset(name) for name in selected_files_ele]
    ds_ele: Any = xr.concat(datasets_ele, dim="epoch")
    ds_ele["epoch"] = pd.to_datetime(ds_ele["epoch"].values)
    ds_ele = ds_ele.where(
        (ds_ele["epoch"] >= time_interval[0]) & (ds_ele["epoch"] <= time_interval[1]),
        drop=True,
    )
    da_ele = ds_ele["data_3d"]
    da_ele = da_ele.rename({"epoch": "time", "energy_3d": "energy_3d"})

    return da_ele, selected_files_ele
