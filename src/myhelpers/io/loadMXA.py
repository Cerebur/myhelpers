"""Load BepiColombo MPPE MIA and MEA data products."""

import glob
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from myhelpers.parameters.paths import DATA_PTH_LOCAL, DATA_PTH_REMOTE
from myhelpers.transformation import repair_time_piecewise


def _load_mxa_data(instrument: str, DATA_PTH: str, utc: bool = False) -> Any:
    """Load an MPPE instrument's NetCDF data product.

    Args:
        instrument: Instrument name, either ``"MIA"`` or ``"MEA"``.
        DATA_PTH: Root directory containing MPPE files.
        utc: Whether to parse timestamps as UTC-aware values.

    Returns:
        A time-by-energy DataArray and matching source file paths.
    """
    assert instrument == "MIA" or instrument == "MEA"
    # Open a netCDF4 file
    file_name = f"BC_{instrument}*_data_VFB1.nc"
    file_path = glob.glob(DATA_PTH + "/MPPE/" + file_name)
    mea_data = xr.open_dataset(
        file_path[0]
    )  # epochs, duration, energy is all identical
    epoch = mea_data["epoch"].data
    duration = mea_data["duration"].data
    if utc:
        time_str = np.array(["".join(e.astype(str)) for e in epoch])
        time = pd.to_datetime(time_str, utc=True)
    else:
        # Check whether the timestrings have a utc tag
        time_str = ["".join(e.astype(str)) for e in epoch]
        if time_str[0][-1] == "Z":
            time_str = np.array(["".join(e[:-1]) for e in time_str])
        else:
            time_str = np.array(["".join(e.astype(str)) for e in epoch])
        time = pd.to_datetime(time_str)

    time = time + pd.to_timedelta(duration[:] / 2.0, unit="s")

    data3d = mea_data["data_3d"].data  # dimensions of epoch,theta,psi,energy_channels
    data3d = data3d[:, 0, 0, :]  # theta and psi remained constant ?
    energy_per_channel = mea_data["energy_table"].data
    if instrument == "MIA":
        # There are four data files for MIA, we will sum them
        for i, file in enumerate(file_path):
            if i == 0:
                # We have already included this file in the original data3d
                continue
            ds = xr.open_dataset(file)
            data3d_file = ds["data_3d"].data[:, 0, 0, :]
            data3d += data3d_file
        # data3d /= 4

    spctr = xr.DataArray(
        data3d,
        coords=dict(time=time, energy=energy_per_channel[0]),
        dims=["time", "energy"],
    )

    return spctr, file_path


def load_mia_data(
    data_local: bool = False,
    resolution: int = 4,
    utc: bool = False,
    repair_missing_data: bool = True,
) -> Any:
    """Load BepiColombo MPPE MIA energy spectra.

    Args:
        data_local: Whether to read from the local data directory.
        resolution: Product cadence in seconds; supported values are 1 and 4.
        utc: Whether output timestamps should be UTC-aware.
        repair_missing_data: Repair irregular time sampling in the 1-second data.

    Returns:
        A time-by-energy DataArray and paths of the source files.
    """
    if data_local:
        DATA_PTH = DATA_PTH_LOCAL
    else:
        DATA_PTH = DATA_PTH_REMOTE

    if resolution != 1 and resolution != 4:
        raise ValueError("The loaded resolution can only be 1s or 4s.")
    if resolution == 4:
        # Load the Electron data
        spectr, file_path = _load_mxa_data(instrument="MIA", DATA_PTH=DATA_PTH, utc=utc)
    else:
        file_path_ispectr = glob.glob(DATA_PTH + "MPPE/mia_vfb1_q1s_eflux*.txt")
        if len(file_path_ispectr) != 1:
            raise FileNotFoundError(
                "The file path to the 1 second resolution file could not be determined. File path: ",
                file_path_ispectr,
            )
        file_path_ienergy = glob.glob(DATA_PTH + "MPPE/mia_vfb1_q1s_energy*.txt")
        if len(file_path_ienergy) != 1:
            raise FileNotFoundError(
                "The file path to the 1 second resolution file could not be determined. File path: ",
                file_path_ienergy,
            )

        # Load the first column as strings (timestamps) and the rest as floats
        df = pd.read_csv(
            file_path_ispectr[0],
            sep=r"\s+",  # whitespace
            header=None,
            parse_dates=[0],  # first column as datetime
            na_values=["NaN"],
        )
        if utc:
            # Force UTC timezone after parsing
            timestamps = pd.to_datetime(
                df.iloc[:, 0], utc=True
            )  # ensures datetime dtype
        else:
            timestamps = pd.to_datetime(
                df.iloc[:, 0], utc=False
            )  # ensures datetime dtype
        timestamps = pd.DatetimeIndex(timestamps, name="time")
        numeric_spectrum_data = df.iloc[:, 1:].to_numpy(dtype=float)

        channel_data = np.genfromtxt(
            file_path_ienergy[0],
            dtype=np.float64,
            encoding="utf-8",
            delimiter=None,
            autostrip=True,
        )

        differences = np.where(~np.isclose(channel_data, channel_data[0]))[0]
        if len(differences) != 0:
            raise ValueError("The energies of the channel are changing unexpectedly")

        spectr = xr.DataArray(
            numeric_spectrum_data,
            coords=dict(time=timestamps, energy=channel_data[0]),
            dims=["time", "energy"],
        )

        file_path = file_path_ispectr

        if repair_missing_data:
            spectr = repair_time_piecewise(spectr)

    return spectr, file_path


def load_mea_data(
    data_local: bool = False,
    utc: bool = False,
    secondaries_removed: bool = False,
    unit: str = "counts",
    density: bool = False,
    temperature: bool = False,
    repair_missing_data: bool = True,
) -> Any:
    """Load BepiColombo MPPE MEA spectra or derived plasma quantities.

    Args:
        data_local: Whether to read from the local data directory.
        utc: Whether output timestamps should be UTC-aware.
        secondaries_removed: Load the differential flux product with secondaries removed.
        unit: Spectral units, either ``"counts"`` or ``"je"``.
        density: Return electron density instead of a spectrum.
        temperature: Return electron temperature instead of a spectrum.
        repair_missing_data: Repair irregular sampling in spectral data.

    Returns:
        A DataArray and the path or paths of the source data.
    """
    if data_local:
        DATA_PTH = DATA_PTH_LOCAL
    else:
        DATA_PTH = DATA_PTH_REMOTE

    if density and temperature:
        raise ValueError("Can not return both density and temperature. Choose one")

    if density:
        file_path = DATA_PTH + "from_Mathias/Ne.csv"
        ne_df = pd.read_csv(file_path, header=0)
        ne_df["time (UTC)"] = pd.to_datetime(ne_df["time (UTC)"])
        for column in ne_df.columns[1:]:
            ne_df[column] = ne_df[column].astype(float)

        ne = xr.DataArray(
            ne_df.drop("time (UTC)", axis=1).values.T[0],
            coords={"time": ne_df["time (UTC)"].values},
            dims=["time"],
        )
        return ne, file_path

    elif temperature:
        file_path = DATA_PTH + "from_Mathias/Te.csv"
        te_df = pd.read_csv(file_path, header=0)
        te_df["time (UTC)"] = pd.to_datetime(te_df["time (UTC)"])
        for column in te_df.columns[1:]:
            te_df[column] = te_df[column].astype(float)

        te = xr.DataArray(
            te_df.drop("time (UTC)", axis=1).values.T[0],
            coords={"time": te_df["time (UTC)"].values},
            dims=["time"],
        )

        return te, file_path

    if unit.lower() == "counts" and not secondaries_removed:
        # Load the Electron data
        spectr, file_path = _load_mxa_data(instrument="MEA", DATA_PTH=DATA_PTH, utc=utc)
    elif secondaries_removed:
        file_path = DATA_PTH + "from_Mathias/JE_MEA1_wo_sec.csv"
        je_wos_mea1 = pd.read_csv(file_path, header=0)
        je_wos_mea1["time (UTC)"] = pd.to_datetime(
            je_wos_mea1["time (UTC)"],
        )
        for column in je_wos_mea1.columns[1:]:
            je_wos_mea1[column] = je_wos_mea1[column].astype(float)
        spectra_wos = je_wos_mea1.drop("time (UTC)", axis=1).values
        spectr = xr.DataArray(
            spectra_wos,
            coords={
                "time": je_wos_mea1["time (UTC)"].values,
                "energies": je_wos_mea1.columns[1:].astype(float),
            },
            dims=["time", "energies"],
        )
    elif unit.lower() == "je":
        file_path = DATA_PTH + "from_Mathias/JE_MEA1.csv"
        je_mea1 = pd.read_csv(file_path, header=0)
        je_mea1["time (UTC)"] = pd.to_datetime(
            je_mea1["time (UTC)"],
        )
        for column in je_mea1.columns[1:]:
            je_mea1[column] = je_mea1[column].astype(float)
        spectra = je_mea1.drop("time (UTC)", axis=1).values
        spectr = xr.DataArray(
            spectra,
            coords={
                "time": je_mea1["time (UTC)"].values,
                "energies": je_mea1.columns[1:].astype(float),
            },
            dims=["time", "energies"],
        )
    else:
        raise NotImplementedError(
            "Either unit is not supported (counts or je) or secondaries_removed was requested with unit of counts. Only differential energy flux (je) is allowed in that case."
        )

    if repair_missing_data:
        spectr = repair_time_piecewise(spectr)

    return spectr, file_path
