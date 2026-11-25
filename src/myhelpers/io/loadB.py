"""Load magnetic field observations and wavelet transforms."""

import glob
import io
import os
import re
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from myhelpers.io.download_MESS_data import download_messenger_data
from myhelpers.mission_info.messenger import (
    aberrate_positions as _aberrate_positions,
    get_orbit_numbers as _get_orbit_numbers,
)
from myhelpers.parameters.paths import DATA_PTH_LOCAL, DATA_PTH_REMOTE
from myhelpers.parameters.planetary_parameters import planetary_radii
from myhelpers.transformation import mso_to_msm


def load_mag_data(
    time_interval: list[Any],
    coordinates: str = "vso",
    data_local: bool = False,
    spacecraft: str = "bepi",
    **kwargs: Any,
) -> Any:
    """Load magnetic field data for BepiColombo or Venus Express.

    Args:
        time_interval: Mutable list containing the interval start and end times.
        coordinates: Requested coordinate frame; currently only ``"vso"`` and ``"mso"`` is supported.
        data_local: Whether to search the local data directory.
        spacecraft: Spacecraft selector, either ``"bepi"`` or ``"vex"`` or ``"messenger"``.
        **kwargs: Additional loader options passed to the selected loader.
            For MESSENGER, ``resolution`` may be ``"high"`` or ``"sec"`` and ``aberrated`` may be ``True`` or ``False``.

    Returns:
        Magnetic field data and the paths of the files that were loaded. If spacecraft == "bepi" or "messenger", a dataset containing magnetic field and position will be returned.
    """
    possible_spacecrafts = ["bepi", "vex", "messenger"]

    assert spacecraft in possible_spacecrafts
    if spacecraft == "bepi":
        return load_mag_data_bepi(time_interval, coordinates, data_local)
    elif spacecraft == "vex":
        return load_mag_data_vex(time_interval, coordinates, data_local, **kwargs)
    elif spacecraft == "messenger":
        return load_mag_data_messenger(time_interval, coordinates, data_local, **kwargs)


def load_mag_data_vex(
    time_interval: list[Any],
    coordinates: str = "vso",
    data_local: bool = False,
    **kwargs: Any,
) -> Any:
    """Load Venus Express magnetic field data from NetCDF files.

    Args:
        time_interval: Mutable list containing the interval start and end times.
        coordinates: Requested coordinate frame; only ``"vso"`` is supported.
        data_local: Whether to search the local data directory.
        **kwargs: Set ``array`` to return the magnetic field DataArray; by default
            only the magnetic field variable is returned.

    Returns:
        The selected magnetic data and a list of source file paths.
    """
    if data_local:
        DATA_PTH = DATA_PTH_LOCAL
    else:
        DATA_PTH = DATA_PTH_REMOTE

    if coordinates != "vso":
        raise NotImplementedError("Data is always returned in a VSO coordinate system.")

    if (
        type(time_interval[0]) is not pd.Timestamp
        or type(time_interval[1]) is not pd.Timestamp
    ):
        print("Not a datetime format, will convert")
        time_interval[0] = pd.to_datetime(time_interval[0])
        time_interval[1] = pd.to_datetime(time_interval[1])

    all_files_ele = glob.glob(DATA_PTH + "ASPERA4/vex_mag__*.nc")

    pattern = re.compile(r"__(\d{8})_(\d{6})_(\d{8})_(\d{6}).nc")

    selected_files_mag = []

    for f in all_files_ele:
        match = pattern.search(f)
        if not match:
            continue

        start_str = match.group(1) + match.group(2)
        end_str = match.group(3) + match.group(4)

        file_start = datetime.strptime(start_str, "%Y%m%d%H%M%S")
        file_end = datetime.strptime(end_str, "%Y%m%d%H%M%S")
        condition = (
            (file_start <= time_interval[0] and file_end >= time_interval[0])
            or (file_start >= time_interval[0] and file_end <= time_interval[1])
            or (file_start <= time_interval[1] and file_end >= time_interval[1])
        )

        # overlap condition:
        if condition:
            selected_files_mag.append(f)
            print("Loading file: ", f)

    datasets_mag = [xr.open_dataset(name) for name in selected_files_mag]
    ds_mag: Any = xr.concat(datasets_mag, dim="epoch")
    ds_mag["epoch"] = pd.to_datetime(ds_mag["epoch"].values)
    ds_mag = ds_mag.where(
        (ds_mag["epoch"] >= time_interval[0]) & (ds_mag["epoch"] <= time_interval[1]),
        drop=True,
    )

    if len(kwargs) == 0:
        kwargs = {"array": True}
    if kwargs["array"]:
        da_mag = ds_mag["mag"]
        da_mag = da_mag.rename({"epoch": "time", "components_mag": "comp"})
        da_mag = da_mag.where((da_mag <= 1000))
        return da_mag, selected_files_mag

    return ds_mag, selected_files_mag


def load_mag_data_bepi(
    time_interval: list[Any], coordinates: str = "vso", data_local: bool = False
) -> Any:
    """Load BepiColombo MPO-MAG observations from daily Parquet files.

    Args:
        time_interval: Mutable list containing the interval start and end times.
        coordinates: Requested frame; only ``"vso"`` is supported.
        data_local: Whether to search the local data directory.

    Returns:
        A VSO magnetic field DataArray and the paths of the source files.
    """
    if data_local:
        DATA_PTH = DATA_PTH_LOCAL
    else:
        DATA_PTH = DATA_PTH_REMOTE

    if coordinates != "vso":
        raise NotImplementedError("Data is always returned in a VSO coordinate system.")
    if (
        type(time_interval[0]) is not pd.Timestamp
        or type(time_interval[1]) is not pd.Timestamp
    ):
        print("Not a datetime format, will convert")
        time_interval[0] = pd.to_datetime(time_interval[0])
        time_interval[1] = pd.to_datetime(time_interval[1])

    # Open a magnetic data file saved as parquet
    # Check that tmin and tmax are on the same day or import all days in between
    dates = pd.date_range(time_interval[0], time_interval[1], freq="D")
    date_strings = dates.strftime("%Y%m%d").tolist()

    file_paths = []
    for date_string in date_strings:
        file_paths.append(
            glob.glob(
                DATA_PTH
                + f"/MPO-MAG/*_cal_*_ob_*_{coordinates}_*_{date_string}.parquet"
            )[0]
        )

    dfs = [pd.read_parquet(file_path_mag) for file_path_mag in file_paths]
    df = pd.concat(dfs, ignore_index=True)

    # Drop the individual components of time and housekeeping data N1-N5
    df = df[["datetime", "X", "Y", "Z", "Bx", "By", "Bz"]]

    # Filter for the time region we are interested in
    df = df.loc[
        (df["datetime"] >= time_interval[0]) & (df["datetime"] <= time_interval[1])
    ]

    # Convert the position coordinates from km to planetary radius
    df[["X", "Y", "Z"]] = df[["X", "Y", "Z"]] / planetary_radii("venus")

    # Create xarray inputs for the local transformation routines
    b_vso = xr.DataArray(
        df[["Bx", "By", "Bz"]],
        coords=dict(time=df["datetime"], comp=["x", "y", "z"]),
        dims=["time", "comp"],
        attrs={"Coordinate System": "VSO"},
    )
    pos_vso = xr.DataArray(
        df[["X", "Y", "Z"]],
        coords=dict(time=df["datetime"], comp=["x", "y", "z"]),
        dims=["time", "comp"],
        attrs={"Coordinate System": "VSO", "Units": "Venus radii"},
    )
    ds = xr.Dataset({"B": b_vso, "pos": pos_vso})

    return ds, file_paths


def _messenger_file_version(filename: str) -> int:
    """Return the archive version encoded in a MESSENGER TAB filename."""
    match = re.search(r"_V(\d+)\.TAB$", filename)
    return int(match.group(1)) if match else 0


def _messenger_csv_time_offset(
    csv_path: Path,
    target: pd.Timestamp,
    *,
    after: bool = False,
) -> int:
    """Find a row byte offset in the time-sorted MESSENGER seconds CSV."""
    file_size = csv_path.stat().st_size
    with csv_path.open("rb") as csv_file:
        csv_file.readline()  # Skip the header.
        low = csv_file.tell()
        high = file_size

        while low < high:
            midpoint = (low + high) // 2
            row_start = midpoint
            while row_start > low:
                csv_file.seek(row_start - 1)
                if csv_file.read(1) == b"\n":
                    break
                row_start -= 1
            csv_file.seek(row_start)
            line = csv_file.readline()
            if not line:
                high = row_start
                continue

            row_end = csv_file.tell()
            row_time = pd.Timestamp(line.rsplit(b",", 1)[-1].strip().decode())
            is_before_target = row_time < target or (after and row_time == target)
            if is_before_target:
                low = row_end
            else:
                high = row_start

    return low


def load_mag_data_messenger(
    time_interval: list[Any],
    coordinates: str = "mso",
    data_local: bool = False,
    resolution: str = "high",
    aberrated: bool = False,
) -> Any:
    """Load MESSENGER MAG observations from daily TAB or second-cadence CSV data.

    Args:
        time_interval: Mutable list containing the interval start and end times.
        coordinates: Requested frame; only ``"mso"`` and ``"msm"`` is supported.
        data_local: Whether to search the local data directory.
        resolution: Data source, either daily ``"high"`` cadence files or the
            ``"sec"`` cadence CSV.
        aberrated: Whether the coordinates should be aberration corrected

    Returns:
        A MSO magnetic field DataArray and the paths of the source files.
    """
    data_root = Path(DATA_PTH_LOCAL if data_local else DATA_PTH_REMOTE)

    if coordinates.lower() not in ["mso", "msm"]:
        raise NotImplementedError(
            "Data is always returned in an MSO or MSM coordinate system."
        )
    if resolution not in ("high", "sec"):
        raise ValueError("MESSENGER resolution must be 'high' or 'sec'.")
    # MESSENGER timestamps in both supported products are UTC values stored
    # without timezone metadata. Convert aware inputs to UTC, then make them
    # naive so they can be compared with the archive timestamps.
    requested_start = pd.Timestamp(time_interval[0])
    requested_end = pd.Timestamp(time_interval[1])
    if requested_start.tz is not None:
        requested_start = requested_start.tz_convert("UTC").tz_localize(None)
    if requested_end.tz is not None:
        requested_end = requested_end.tz_convert("UTC").tz_localize(None)

    messenger_data_path = data_root / "MESSENGER" / "data"
    if resolution == "high":
        high_resolution_path = (
            messenger_data_path / "MAG-3-CDR-CALIBRATED" / "MSO" / "SCI"
        )
        dates = pd.date_range(requested_start.normalize(), requested_end.normalize())
        date_strings = dates.strftime("%y%j").tolist()

        files_by_date: dict[str, str] = {}
        missing_dates: list[str] = []
        for date_string in date_strings:
            year = pd.to_datetime(date_string, format="%y%j").year
            matches = glob.glob(
                str(
                    high_resolution_path
                    / str(year)
                    / "*"
                    / f"MAGMSOSCI{date_string}_V*.TAB"
                )
            )
            if matches:
                files_by_date[date_string] = max(
                    matches,
                    key=_messenger_file_version,
                )
            else:
                missing_dates.append(date_string)

        if missing_dates:
            print(
                "Downloading missing MESSENGER high-resolution files for: "
                + ", ".join(missing_dates)
            )
            downloaded_paths = download_messenger_data(
                start=requested_start,
                end=requested_end,
                instrument="MAG",
                data_product="SCI",
                coordinate_system="MSO",
                download_path=data_root / "MESSENGER" / "data",
            )
            for downloaded_path in downloaded_paths:
                path = Path(downloaded_path)
                for date_string in missing_dates:
                    if date_string in path.name and path.suffix.upper() == ".TAB":
                        files_by_date[date_string] = str(path)
            for date_string in missing_dates:
                year = pd.to_datetime(date_string, format="%y%j").year
                matches = glob.glob(
                    str(
                        high_resolution_path
                        / str(year)
                        / "*"
                        / f"MAGMSOSCI{date_string}_V*.TAB"
                    )
                )
                if matches:
                    files_by_date[date_string] = max(
                        matches, key=_messenger_file_version
                    )

        file_paths = [
            files_by_date[date_string]
            for date_string in date_strings
            if date_string in files_by_date
        ]
        if not file_paths:
            raise FileNotFoundError(
                f"No MESSENGER high-resolution files found for "
                f"{requested_start} through {requested_end} "
                f"in {high_resolution_path}"
            )

        dfs = [
            pd.read_csv(
                file_path,
                delimiter=r"\s+",
                names=[
                    "Year",
                    "DOY",
                    "Hour",
                    "Minute",
                    "Second",
                    "Timetag",
                    "x_mso",
                    "y_mso",
                    "z_mso",
                    "bx_mso",
                    "by_mso",
                    "bz_mso",
                ],
            )
            for file_path in file_paths
        ]
        df = pd.concat(dfs, ignore_index=True)
        df["datetime"] = pd.to_datetime(
            df["Year"].astype(str) + df["DOY"].astype(str).str.zfill(3),
            format="%Y%j",
        ) + pd.to_timedelta(
            df["Hour"] * 3600 + df["Minute"] * 60 + df["Second"], unit="s"
        )
        df = df[
            ["datetime", "x_mso", "y_mso", "z_mso", "bx_mso", "by_mso", "bz_mso"]
        ].rename(
            columns={
                "x_mso": "X",
                "y_mso": "Y",
                "z_mso": "Z",
                "bx_mso": "Bx",
                "by_mso": "By",
                "bz_mso": "Bz",
            }
        )
        df = df.loc[
            (df["datetime"] >= requested_start) & (df["datetime"] <= requested_end)
        ]
    else:
        if not data_local:
            raise Warning(
                "Do not set data_local to False when trying to load the csv file. It takes Aaages."
            )
            raise
        csv_path = messenger_data_path / "MAG" / "MESSENGER_sec.csv"
        start_offset = _messenger_csv_time_offset(csv_path, requested_start)
        end_offset = _messenger_csv_time_offset(
            csv_path,
            requested_end,
            after=True,
        )
        with csv_path.open("rb") as csv_file:
            csv_file.seek(start_offset)
            selected_bytes = csv_file.read(max(0, end_offset - start_offset))

        if selected_bytes:
            columns = [
                "X_MSO",
                "Y_MSO",
                "Z_MSO",
                "BX_MSO",
                "BY_MSO",
                "BZ_MSO",
                "Time",
            ]
            df = pd.read_csv(
                io.BytesIO(selected_bytes),
                header=None,
                names=columns,
                usecols=columns,
                parse_dates=["Time"],
                date_format="%Y-%m-%dT%H:%M:%S.%f",
            ).rename(
                columns={
                    "Time": "datetime",
                    "X_MSO": "X",
                    "Y_MSO": "Y",
                    "Z_MSO": "Z",
                    "BX_MSO": "Bx",
                    "BY_MSO": "By",
                    "BZ_MSO": "Bz",
                }
            )
        else:
            df = pd.DataFrame(columns=["datetime", "X", "Y", "Z", "Bx", "By", "Bz"])
        file_paths = [str(csv_path)]

    # If aberrated coordinates, convert them
    if aberrated:
        df["orbit_number"] = _get_orbit_numbers(df["datetime"])
        df = _aberrate_positions(df)
    # If MSM coordinates, convert them
    if coordinates == "msm":
        df[["X", "Y", "Z"]] = mso_to_msm(df[["X", "Y", "Z"]].to_numpy())

    # Convert the position coordinates from km to planetary radius
    df[["X", "Y", "Z"]] = df[["X", "Y", "Z"]] / planetary_radii("mercury")

    # Create xarray inputs for the local transformation routines
    b_mso = xr.DataArray(
        df[["Bx", "By", "Bz"]],
        coords=dict(time=df["datetime"], comp=["x", "y", "z"]),
        dims=["time", "comp"],
        attrs={"Coordinate System": "MSO"},
    )
    pos_mso = xr.DataArray(
        df[["X", "Y", "Z"]],
        coords=dict(time=df["datetime"], comp=["x", "y", "z"]),
        dims=["time", "comp"],
        attrs={"Coordinate System": "MSO", "Units": "Mercury radii"},
    )
    ds = xr.Dataset({"B": b_mso, "pos": pos_mso})
    return ds, file_paths


def load_Bcwt_data(
    time_interval: Sequence[Any],
    minimum_resolution: int,
    fmin_b: float,
    fmax_b: float,
    coord_system: str = "vso",
    data_local: bool = False,
    load: bool = True,
    save: bool = False,
    **kwargs: Any,
) -> Any:
    """Load or calculate a magnetic field continuous wavelet transform.

    Args:
        time_interval: Start and end times used to select or identify a transform.
        minimum_resolution: Minimum output time resolution in nanoseconds.
        fmin_b: Lower transform frequency in hertz.
        fmax_b: Upper transform frequency in hertz.
        coord_system: Magnetic field coordinate system (``"vso"`` or ``"fac"``).
        data_local: Whether to use the local data directory.
        load: Search for and load a previously saved transform when true.
        save: Save a newly calculated transform when true and ``load`` is false.
        **kwargs: Input field and source metadata required for calculation or lookup.
            Calculation expects ``b_vso`` or ``b_fac``, ``min_segment_length``,
            ``file_path_mag``, and ``spacecraft``; lookup needs the latter two.

    Returns:
        The wavelet Dataset, or ``None`` when loading finds no suitable file.
    """
    if data_local:
        DATA_PTH = DATA_PTH_LOCAL
    else:
        DATA_PTH = DATA_PTH_REMOTE

    if load:
        # Check whether a suitable wavelet has been saved
        sc_specifier = ""
        if kwargs["spacecraft"] == "vex":
            sc_specifier = "_vex"
        elif kwargs["spacecraft"] == "bepi":
            pass
        file_path_mag = kwargs["file_path_mag"]
        file_parts = Path(file_path_mag).parts
        idx = file_parts.index("Master_thesis")
        if file_parts[idx + 1] == "data":
            subdir = file_parts[idx + 2]
        else:
            raise FileNotFoundError(
                "The given file path has a different structure to what is expected (/Volumes/MyHarddrive/Science_projects/Master_thesis/data/*)"
            )

        cwts_saved = glob.glob(
            DATA_PTH + subdir + f"/mag{sc_specifier}_*CWT*_{coord_system}*.nc"
        )
        found_files = []
        for file in cwts_saved:
            filename = file.split("/")[-1]
            file_tint: list[Any] = filename.split("CWT")[-1].split("_")[1].split("-")
            file_tint[0] = pd.to_datetime(file_tint[0])
            file_tint[1] = pd.to_datetime(file_tint[1])
            if (
                file_tint[0] - pd.to_timedelta(minimum_resolution, "ns")
                > time_interval[0]
            ):
                continue
            if (
                file_tint[1] + pd.to_timedelta(minimum_resolution, "ns")
                < time_interval[0]
            ):
                continue

            file_fint = filename.split("CWT")[-1].split("_")[3].split("-")
            if float(file_fint[0][:-2]) > fmin_b:
                continue
            if float(file_fint[1][:-2]) < fmax_b:
                continue

            file_res = filename.split("CWT")[-1].split("_")[4][3:-4]
            if float(file_res) > minimum_resolution:
                continue

            found_files.append(file)

        if len(found_files) > 0:
            # Load saved cwt data
            b_cwt = xr.open_dataset(found_files[0])
            #            if coord_system == "fac":
            #                b_cwt_t = b_cwt.time.data
            #                b_cwt_para = b_cwt.Bpara
            #                b_cwt_perp = b_cwt.Bperp
            #            elif coord_system == "vso":
            #                b_cwt_t = b_cwt.time.data
            #                b_cwt_x = b_cwt.Bx
            #                b_cwt_y = b_cwt.By
            #                b_cwt_z = b_cwt.Bz
            print("Wavelet loaded successfully: " + found_files[0])
            return b_cwt
        else:
            print("No saved files found!")
            return None
    else:
        print("Computing Wavelet")

        from myhelpers.transformation.pyrf_local import compress_cwt, wavelet

        nf = 100  # number of frequencies to be resolved
        if coord_system == "fac":
            b_name = "b_fac"
        elif coord_system == "vso":
            b_name = "b_vso"
        else:
            raise NotImplementedError("This coordinate system is not supported")
        # If the dataarray contains nan values, the cwt will return only nans. Therefore, we will interpolate over the nan values and set those times to nans later
        b = kwargs[b_name]
        min_segment_length = kwargs["min_segment_length"]
        b_data = b.data
        time = b.time

        # Identify rows where ANY component is NaN
        nan_rows = np.any(np.isnan(b_data), axis=-1)

        # Prepare output container (filled with NaNs initially)
        # We determine output shape using a small dummy computation
        dummy_idx = np.where(~nan_rows)[0]
        if len(dummy_idx) == 0:
            raise ValueError("All values are NaN.")

        # Compute once on small valid slice to get output shape
        test_slice = b.isel(time=slice(dummy_idx[0], dummy_idx[0] + min_segment_length))
        test_cwt_ds = wavelet(test_slice, f=[fmin_b, fmax_b], n_freqs=nf)

        # Create full NaN arrays for each variable
        cwt_full = {}
        for var in test_cwt_ds.data_vars:
            shape = (len(time),) + test_cwt_ds[var].data.shape[1:]
            cwt_full[var] = np.full(shape, np.nan, dtype=test_cwt_ds[var].dtype)

        # Find contiguous valid segments
        valid_mask = ~nan_rows
        change_points = np.where(np.diff(valid_mask.astype(int)) != 0)[0] + 1
        segments = np.split(np.arange(len(time)), change_points)

        for seg in segments:
            if not valid_mask[seg[0]]:
                # This is a NaN segment → leave as NaN
                continue

            if len(seg) < min_segment_length:
                # Too short → leave as NaN
                continue

            # Extract segment as xarray
            segment_da = b.isel(time=seg)

            # Compute wavelet
            seg_cwt_ds = wavelet(segment_da, f=[fmin_b, fmax_b], n_freqs=nf)

            # Align each variable
            for var in seg_cwt_ds.data_vars:
                seg_data = seg_cwt_ds[var].data
                orig_len = len(seg)
                seg_len = seg_data.shape[0]

                if seg_len < orig_len:
                    # Pad at end with NaNs
                    pad_shape = (orig_len - seg_len,) + seg_data.shape[1:]
                    seg_data = np.concatenate(
                        [seg_data, np.full(pad_shape, np.nan, dtype=seg_data.dtype)],
                        axis=0,
                    )
                elif seg_len > orig_len:
                    # Trim to fit
                    seg_data = seg_data[:orig_len]

                # Assign back
                cwt_full[var][seg, ...] = seg_data
        # Build final dataset with original coordinates
        coords: dict[Hashable, Any] = {"time": time}
        for var in test_cwt_ds.data_vars:
            for c in test_cwt_ds[var].coords:
                if c != "time":
                    coords[c] = test_cwt_ds[var].coords[c]

        b_cwt = xr.Dataset(
            {var: (test_cwt_ds[var].dims, cwt_full[var]) for var in cwt_full},
            coords=coords,
        )
        print(
            "Wavelet transform computed with",
            nf,
            "frequencies in range:",
            fmin_b,
            "-",
            fmax_b,
            "Hz",
        )

        # Compress the wavelet transform
        time_resol = np.diff(b_cwt.time.data).mean()  # time resolution of cwt in ns

        if time_resol >= np.timedelta64(minimum_resolution, "ns"):
            print(
                f"No further compression, time resolution is {time_resol} while specified minimum resolution is {minimum_resolution}"
            )
            # No further compression needed
            b_cwt_t = b_cwt.time.data
            b_cwt_x = b_cwt.Bx
            b_cwt_y = b_cwt.By
            b_cwt_z = b_cwt.Bz

            if coord_system == "fac":
                # If the coordinate system is fac, we would like to combine the perpendicular components
                options: Any
                options = dict(
                    coords=[b_cwt_t, b_cwt.frequency], dims=["time", "frequency"]
                )
                b_cwt_perp = xr.DataArray(b_cwt_x + b_cwt_y, **options)
                b_cwt_para = xr.DataArray(b_cwt_z, **options)
                b_cwt = xr.Dataset(
                    {"Bperp": b_cwt_perp, "Bpara": b_cwt_para},
                    attrs={"Coordinate System": "fac"},
                )
            elif coord_system == "vso":
                b_cwt = xr.Dataset(
                    {"Bx": b_cwt_x, "By": b_cwt_y, "Bz": b_cwt_z},
                    attrs={"Coordinate System": "VSO"},
                )
                pass

        else:
            nc = int(
                minimum_resolution / (time_resol.astype(int))
            )  # nc is the number of timesteps to average over
            b_cwt_t, b_cwt_x, b_cwt_y, b_cwt_z = compress_cwt(b_cwt, nc)

            options = dict(
                coords=[b_cwt_t, b_cwt.frequency], dims=["time", "frequency"]
            )
            if coord_system == "vso":
                b_cwt_x = xr.DataArray(b_cwt_x, **options)
                b_cwt_y = xr.DataArray(b_cwt_y, **options)
                b_cwt_z = xr.DataArray(b_cwt_z, **options)

                b_cwt = xr.Dataset(
                    {"Bx": b_cwt_x, "By": b_cwt_y, "Bz": b_cwt_z},
                    attrs={"Coordinate System": "vso"},
                )
            elif coord_system == "fac":
                # If the coordinate system is fac, we would like to combine the perpendicular components
                options = dict(
                    coords=[b_cwt_t, b_cwt.frequency], dims=["time", "frequency"]
                )
                b_cwt_perp = xr.DataArray(b_cwt_x + b_cwt_y, **options)
                b_cwt_para = xr.DataArray(b_cwt_z, **options)
                b_cwt = xr.Dataset(
                    {"Bperp": b_cwt_perp, "Bpara": b_cwt_para},
                    attrs={"Coordinate System": "fac"},
                )

        print("Wavelet transform compressed.")
        if save:
            file_path_mag = kwargs["file_path_mag"]
            file_parts = Path(file_path_mag).parts
            idx = file_parts.index("Master_thesis")
            if file_parts[idx + 1] == "data":
                subdir = file_parts[idx + 2]
            else:
                raise FileNotFoundError(
                    "The given file path has a different structure to what is expected (/Volumes/MyHarddrive/Science_projects/Master_thesis/data/*)"
                )

            if not os.path.exists(DATA_PTH + subdir):
                os.makedirs(DATA_PTH + subdir)
            time_resol = (
                np.diff(b_cwt_t).mean().astype(float) * 1e-9
            )  # time resolution of cwt in s
            tint_save = [
                pd.to_datetime(b_cwt_t[0]).strftime("%Y%m%d%H%M%S"),
                pd.to_datetime(b_cwt_t[-1]).strftime("%Y%m%d%H%M%S"),
            ]
            sc_specifier = ""
            if kwargs["spacecraft"] == "vex":
                sc_specifier = "_vex"
            elif kwargs["spacecraft"] == "bepi":
                pass

            save_path = (
                DATA_PTH
                + subdir
                + f"/mag{sc_specifier}_CWT_{tint_save[0]}-{tint_save[-1]}_{coord_system}_{fmin_b}Hz-{fmax_b}Hz_Res{time_resol * 1e-9:.4f}s.nc"
            )
            b_cwt.to_netcdf(save_path)
            print("Saved cwt under " + save_path)
        return b_cwt


if __name__ == "__main__":
    load_mag_data(
        [pd.to_datetime("2020-10-15T00:00"), pd.to_datetime("2020-10-15T15:00")],
        data_local=True,
    )
