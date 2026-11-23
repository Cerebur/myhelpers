"""Repair irregular sampling in time-indexed xarray data."""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def repair_time_piecewise(ds: xr.Dataset | xr.DataArray, tol: float = 2) -> xr.Dataset | xr.DataArray:
    """Repair irregular time sampling by resampling affected segments.

    Args:
        ds: Time-indexed xarray object with a ``time`` coordinate.
        tol: Relative deviation from the median positive time step that marks
            a sampling interval as irregular.

    Returns:
        An xarray object with regularized time coordinates; repaired gaps are
        represented by missing values.
    """
    # 1. Sort and drop duplicate timestamps to prevent InvalidIndexError
    ds = ds.sortby("time")
    _, unique_idx = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=unique_idx)
    
    # Standardize time precision to nanoseconds
    ds = ds.assign_coords(time=ds.time.values.astype("datetime64[ns]"))

    t = ds.time.values
    dt = np.diff(t)
    
    # Edge case: Skip if time series is too short
    if len(dt) == 0:
        return ds
        
    dt = np.concatenate([[dt[0]], dt])

    # 2. Determine reference positive time difference
    dt_pos = dt[dt > np.timedelta64(0, "ns")]
    if len(dt_pos) == 0:
        return ds
        
    # Use pandas Timedelta for safe multiplication with floats later
    dt_ref: Any = pd.Timedelta(np.median(dt_pos))

    # 3. Identify deviated steps securely
    dt_ref_np = dt_ref.to_timedelta64()
    tol_np = (dt_ref * tol).to_timedelta64()
    
    bad_step = (dt == np.timedelta64(0, "ns")) | (np.abs(dt - dt_ref_np) > tol_np)

    # Expand the 'bad' mask to include the adjacent 'good' points. 
    # This guarantees that the resampled bad segments will perfectly overlap/anchor 
    # with the edges of the neighboring good segments.
    bad = bad_step.copy()
    if len(bad) > 1:
        bad[:-1] |= bad_step[1:]
        bad[1:] |= bad_step[:-1]

    # 4. Assign indicators natively to avoid pd.Index.get_indexer
    is_good = ~bad
    ds = ds.assign_coords(is_good=("time", is_good))

    # Build segment IDs
    seg_id = xr.DataArray(
        np.cumsum(np.r_[True, bad[1:] != bad[:-1]]),
        dims="time",
        coords={"time": ds.time}
    )
    ds = ds.assign_coords(seg_id=seg_id)

    pieces = []
    
    # Safely convert dt_ref to an exact string alias for xarray's resample
    freq_str = f"{int(dt_ref_np.astype(np.int64))}ns"

    for _, g in ds.groupby("seg_id", squeeze=False):
        # Because seg_id chunks by contiguous blocks, all items in 'g' 
        # share the exact same 'is_good' status. We only need to check the first one.
        segment_is_good = g.is_good.values[0]
        
        # Drop the structural coordinates before appending/resampling to prevent NaN propagation
        g = g.drop_vars(["seg_id", "is_good"])

        if segment_is_good:
            pieces.append(g)
        else:
            # Resample bad segments to regularize them and fill missing data with NaNs
            g2 = g.resample(time=freq_str).asfreq()
            pieces.append(g2)

    # 5. Recombine the dataset
    ds_fixed = xr.concat(pieces, dim="time").sortby("time")
    
    # Because our expanded 'bad' regions explicitly overlap with the edges of 'good' regions
    # to maintain alignment, the concat step will create duplicate edge points. Drop them.
    _, final_unique_idx = np.unique(ds_fixed.time.values, return_index=True)
    ds_fixed = ds_fixed.isel(time=final_unique_idx)

    return ds_fixed
