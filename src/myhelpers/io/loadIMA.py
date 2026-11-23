"""Load Venus Express ASPERA-4 Ion Mass Analyzer data."""

import glob
import re
from datetime import datetime
from typing import Any

import cdflib.xarray
import irfpy.vima.massring
import numpy as np
import pandas as pd
import xarray as xr

from myhelpers.parameters.paths import DATA_PTH_LOCAL, DATA_PTH_REMOTE

data_local = False
if data_local:
    DATA_PTH = DATA_PTH_LOCAL
else:
    DATA_PTH = DATA_PTH_REMOTE


m_per_q = {"H1": 1, "O1": 16}


def load_IMA_data_VEx(
    time_interval: list[Any],
    ion_type: str | list[str],
    level: str = "L1",
    **kwargs: Any,
) -> Any:
    """Load Venus Express IMA observations at level L1 or L2.

    Args:
        time_interval: Mutable list containing the interval start and end times.
        ion_type: Ion selector or list of L1 ion species (for example ``["H1"]``).
        level: Data product level, ``"L1"`` or ``"L2"``.
        **kwargs: Optional product controls. ``resolve_polar`` selects polar-angle
            resolved output for L2 data.

    Returns:
        The selected ion data and source file paths (L1 returns a list of arrays).
    """
    if (
        type(time_interval[0]) is not pd.Timestamp
        or type(time_interval[1]) is not pd.Timestamp
    ):
        print("Not a datetime format, will convert")
        time_interval[0] = pd.to_datetime(time_interval[0])
        time_interval[1] = pd.to_datetime(time_interval[1])

    if level == "L1":
        all_files_ion = glob.glob(DATA_PTH + "ASPERA4/vex_ima_*_m25*_*.cdf")
        pattern = re.compile(r"_(\d{8})(\d{6})")
    elif level == "L2":
        if not isinstance(ion_type, str):
            raise ValueError("ion_type variable should be a string, e.g. H1")
        all_files_ion = glob.glob(DATA_PTH + f"ASPERA4/vex_L2_ima_*{ion_type}__*.cdf")
        pattern = re.compile(r"__(\d{8})_(\d{6})_(\d{8})_(\d{6})")
    else:
        raise NotImplementedError("The requested data level is not supported.")

    selected_files_ion = []

    for f in all_files_ion:
        match = pattern.search(f)
        if not match:
            print(f)
            continue

        start_str = match.group(1) + match.group(2)
        file_start = datetime.strptime(start_str, "%Y%m%d%H%M%S")

        if level == "L1":
            condition = file_start <= time_interval[1]
        elif level == "L2":
            end_str = match.group(3) + match.group(4)
            file_end = datetime.strptime(end_str, "%Y%m%d%H%M%S")
            condition = (
                (file_start <= time_interval[0] and file_end >= time_interval[0])
                or (file_start >= time_interval[0] and file_end <= time_interval[1])
                or (file_start <= time_interval[1] and file_end >= time_interval[1])
            )

        # overlap condition:
        if condition:  # and file_end >= time_interval[0]:
            selected_files_ion.append(f)
            print("Loading file: ", f)

    datasets_ion = [cdflib.xarray.cdf_to_xarray(name) for name in selected_files_ion]
    ds_ion: Any = xr.concat(datasets_ion, dim="Epoch")
    ds_ion["Epoch"] = pd.to_datetime(ds_ion["Epoch"].values)
    ds_ion = ds_ion.where(
        (ds_ion["Epoch"] >= time_interval[0]) & (ds_ion["Epoch"] <= time_interval[1]),
        drop=True,
    )

    # Convert the dataset into an array
    if level == "L1":
        da_ion = ds_ion[
            "Ima_Counts"
        ].mean(
            axis=2
        )  # This is averaging over the azimuthal angles which should be recorded simultaneously with the energy sweep
        # Original shapes
        n_epoch = da_ion.sizes["Epoch"]
        n_elev = da_ion.sizes["Elevations_Tab"]

        # 1. Repeat the Epochs for each Elevation
        old_epochs = da_ion["Epoch"].values
        # Each original epoch is repeated n_elev times
        new_epochs = np.repeat(old_epochs, n_elev)

        # 2. Add 12-second offsets for each Elevation
        offsets = np.tile(np.arange(n_elev) * 24, n_epoch)  # in seconds
        new_epochs = pd.to_datetime(new_epochs) + pd.to_timedelta(offsets, unit="s")

        # 3. Expand the DataArray
        # Swap Elevations_Tab into Epoch dimension
        # First, reshape the data
        new_shape = (
            n_epoch * n_elev,
            da_ion.sizes["Masses_Tab"],
            da_ion.sizes["Energies_Tab"],
        )
        da_reshaped = da_ion.transpose(
            "Epoch", "Elevations_Tab", "Masses_Tab", "Energies_Tab"
        ).values
        da_reshaped = da_reshaped.reshape(new_shape)

        if not isinstance(ion_type, list):
            print(type(ion_type))
            raise ValueError("ion_type should be a list, e.g. [H1,O1]")
        das_ion = []
        for ion in ion_type:
            if ion == "all":
                kernel = np.ones(da_reshaped.shape[1:]).T
                da_new = xr.DataArray(
                    (da_reshaped * kernel.T).sum(axis=1),
                    coords={
                        "time": new_epochs,
                        #'Masses_Tab': da_ion['Masses_Tab'],
                        "Energies_Tab": da_ion["Energies_Tab"],
                    },
                    dims=["time", "Energies_Tab"],
                )

                das_ion.append(da_new)
                continue

            if ion not in m_per_q.keys():
                raise NotImplementedError(
                    "Only the following ions have specific mass values specified: ",
                    m_per_q.keys(),
                )
            mq = m_per_q[ion]
            # In oder to extract only the ions specified, we need to integrate the counts via a kernel
            pacc = ds_ion["Post_Acc_High_Level_Ref"].values.mean().astype(int)

            _, rm = irfpy.vima.massring.massline(
                mq, pacc, enestep=ds_ion["Energies_Tab"].values
            )
            dm = np.array(
                [
                    irfpy.vima.massring.dm(e, mq, pacc)
                    for e in ds_ion["Energies_Tab"].values
                ]
            )

            rm = rm[:, None]
            dm = dm[:, None]
            mass_rings = ds_ion["Masses_Tab"].values[None, :]
            x = mass_rings - rm

            p = 2
            sigma = dm / ((2 * np.log(2)) ** (1 / (2 * p)))
            kernel = np.exp(-0.5 * np.abs(x / sigma) ** (2.0 * p))
            kernel /= kernel.sum(axis=1, keepdims=True)

            # 4. Create new DataArray
            da_new = xr.DataArray(
                (da_reshaped * kernel.T).sum(axis=1),
                coords={
                    "time": new_epochs,
                    #'Masses_Tab': da_ion['Masses_Tab'],
                    "Energies_Tab": da_ion["Energies_Tab"],
                },
                dims=["time", "Energies_Tab"],
            )

            das_ion.append(da_new)
        return das_ion, selected_files_ion

    elif level == "L2":
        mask = ds_ion["DiffFlux"].data < 0
        new_ima_data = ds_ion["DiffFlux"].data
        new_ima_data[mask] = 0
        da_ion = ds_ion["DiffFlux"].mean(axis=2)

        return ds_ion

        resolve_polar = False
        if "resolve_polar" in kwargs.keys():
            if kwargs["resolve_polar"]:
                resolve_polar = True

        if resolve_polar:
            n_epoch = da_ion.sizes["Epoch"]
            n_elev = da_ion.sizes["Elevation"]

            # 1. Repeat the Epochs for each Elevation
            old_epochs = da_ion["Epoch"].values
            # Each original epoch is repeated n_elev times
            new_epochs = np.repeat(old_epochs, n_elev)

            # 2. Add offsets for each Elevation
            time_per_elev = np.diff(ds_ion["Epoch"].data).mean() / (
                np.timedelta64(1, "s") * n_elev
            )

            offsets = np.tile(np.arange(n_elev) * time_per_elev, n_epoch)
            new_epochs = pd.to_datetime(new_epochs) + pd.to_timedelta(offsets, unit="s")

            # 3. Expand the DataArray
            # Swap Elevations_Tab into Epoch dimension
            # First, reshape the data
            new_shape = (
                n_epoch * n_elev,
                da_ion.sizes["Energy"],
            )
            da_reshaped = da_ion.transpose("Epoch", "Elevation", "Energy").values
            da_reshaped = da_reshaped.reshape(new_shape)
            da_ion = da_reshaped
            epochs = new_epochs
            da_ion = xr.DataArray(
                da_ion,
                coords={
                    "epoch": epochs,
                    "energy": da_ion["Energy"].data,
                },
                dims=["epoch", "energy"],
            )

        else:
            da_ion = da_ion.mean(axis=1)
            epochs = da_ion["Epoch"].data

            da_ion = xr.DataArray(
                da_ion,
                coords={"epoch": epochs, "energy": da_ion["Energy"].data},
                dims=["epoch", "energy"],
            )

        return da_ion, selected_files_ion
