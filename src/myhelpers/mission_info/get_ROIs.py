"""Reference regions of interest for selected Venus encounters."""

from typing import Any

import pandas as pd

roi_data = [
    {
        "name": "whole orbit",
        "sc": "VEX",
        "orbit": 696,
        "start": "2008-03-16T15:00:00",
        "end": "2008-03-17T15:30:00",
    },
    {
        "name": "orbit_narrow",
        "sc": "VEX",
        "orbit": 696,
        "start": "2008-03-17T03:59:41",
        "end": "2008-03-17T06:00:27",
    },
    {
        "name": "magnetosheath",
        "sc": "VEX",
        "orbit": 696,
        "start": "2008-03-17T03:59:41",
        "end": "2008-03-17T04:11:35",
    },
    {
        "name": "ionosphere",
        "sc": "VEX",
        "orbit": 696,
        "start": "2008-03-17T04:11:35",
        "end": "2008-03-17T04:18:15",
    },
    {
        "name": "magnetosphere",
        "sc": "VEX",
        "orbit": 696,
        "start": "2008-03-17T04:18:15",
        "end": "2008-03-17T06:00:27",
    },
    {
        "name": "bursty flow",
        "sc": "VEX",
        "orbit": 696,
        "start": "2008-03-17T04:25:00",
        "end": "2008-03-17T04:50:00",
    },
    {
        "name": "bursty flow narrow",
        "sc": "VEX",
        "orbit": 696,
        "start": "2008-03-17T04:33:00",
        "end": "2008-03-17T04:43:00",
    },
    {
        "name": "magnetosphere",
        "sc": "VEX",
        "orbit": 582,
        "start": "2007-11-24T00:00:00",
        "end": "2007-11-24T04:00:00",
    },
    {
        "name": "bursty flow",
        "sc": "VEX",
        "orbit": 582,
        "start": "2007-11-24T01:30:00",
        "end": "2007-11-24T02:20:00",
    },
    {
        "name": "magnetosphere",
        "sc": "VEX",
        "orbit": 576,
        "start": "2007-11-17T23:30:00",
        "end": "2007-11-18T04:30:00",
    },
    {
        "name": "bursty flow",
        "sc": "VEX",
        "orbit": 576,
        "start": "2007-11-18T01:00:00",
        "end": "2007-11-18T01:35:00",
    },
    {
        "name": "magnetosphere",
        "sc": "VEX",
        "orbit": 346,
        "start": "2007-04-02T03:30:00",
        "end": "2007-04-02T07:30:00",
    },
    {
        "name": "bursty flow",
        "sc": "VEX",
        "orbit": 346,
        "start": "2007-04-02T05:30:00",
        "end": "2007-04-02T06:30:00",
    },
    {
        "name": "magnetosphere",
        "sc": "VEX",
        "orbit": 247,
        "start": "2006-12-24T07:00:00",
        "end": "2006-12-24T10:00:00",
    },
    {
        "name": "bursty flow",
        "sc": "VEX",
        "orbit": 247,
        "start": "2006-12-24T08:15:00",
        "end": "2006-12-24T08:50:00",
    },
    {
        "name": "whole flyby",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T03:00:00",
        "end": "2020-10-15T15:00:00",
    },
    {
        "name": "magnetosphere",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T04:07:36",
        "end": "2020-10-15T14:00:00",
    },
    {
        "name": "bursty flow",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:40:27",
        "end": "2020-10-15T06:02:56",
    },
    {
        "name": "bursty flow narrow",  # needed for fleet position plotting
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:40:27",
        "end": "2020-10-15T06:02:56",
    },
    {
        "name": "bursty flow region 1",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:40:27",
        "end": "2020-10-15T05:45:54",
    },
    {
        "name": "bursty flow region 2",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:45:54",
        "end": "2020-10-15T05:49:46",
    },
    {
        "name": "bursty flow region 3",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:49:46",
        "end": "2020-10-15T05:51:57",
    },
    {
        "name": "bursty flow region 4",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:51:57",
        "end": "2020-10-15T06:02:56",
    },
    {
        "name": "bursty increased E flux region 1",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:40:40",
        "end": "2020-10-15T05:40:48",
    },
    {
        "name": "bursty increased E flux region 2",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:41:00",
        "end": "2020-10-15T05:41:45",
    },
    {
        "name": "bursty increased E flux region 3",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:43:09",
        "end": "2020-10-15T05:43:29",
    },
    {
        "name": "bursty increased E flux region 4",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:43:38",
        "end": "2020-10-15T05:43:54",
    },
    {
        "name": "bursty increased E flux region 5",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:44:07",
        "end": "2020-10-15T05:44:36",
    },
    {
        "name": "bursty increased E flux region 6",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:44:36",
        "end": "2020-10-15T05:46:02",
    },
    {
        "name": "bursty increased E flux region 7",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:47:42",
        "end": "2020-10-15T05:47:58",
    },
    {
        "name": "bursty increased E flux region 8",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:48:07",
        "end": "2020-10-15T05:48:19",
    },
    {
        "name": "bursty increased E flux region 9",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:49:17",
        "end": "2020-10-15T05:49:21",
    },
    {
        "name": "bursty increased E flux region 10",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:49:29",
        "end": "2020-10-15T05:49:33",
    },
    {
        "name": "bursty increased E flux region 11",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:49:50",
        "end": "2020-10-15T05:51:50",
    },
    {
        "name": "bursty increased E flux region 12",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:51:59",
        "end": "2020-10-15T05:52:07",
    },
    {
        "name": "bursty increased E flux region 13",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:52:40",
        "end": "2020-10-15T05:52:48",
    },
    {
        "name": "bursty increased E flux region 14",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:40:40",
        "end": "2020-10-15T05:40:48",
    },
    {
        "name": "bursty increased E flux region 15",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:55:14",
        "end": "2020-10-15T05:56:06",
    },
    {
        "name": "bursty increased E flux region 16",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:56:23",
        "end": "2020-10-15T05:57:21",
    },
    {
        "name": "bursty increased E flux region 17",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:59:04",
        "end": "2020-10-15T06:02:02",
    },
    {
        "name": "magnetosheath_v21",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T04:14:00",
        "end": "2020-10-15T04:44:00",
    },
    {
        "name": "magnetosheath",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T04:07:36",
        "end": "2020-10-15T04:46:18",
    },
    {
        "name": "neutral_sheet_prox_v21",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T05:23:00",
        "end": "2020-10-15T06:08:00",
    },
    {
        "name": "neutral_sheet_xing_v21",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T06:08:00",
        "end": "2020-10-15T06:15:00",
    },
    {
        "name": "neutral_sheet_flap_v21",
        "sc": "BEPI",
        "orbit": None,
        "start": "2020-10-15T06:46:00",
        "end": "2020-10-15T07:01:00",
    },
    {
        "name": "FTE_rain_DOY11101",
        "sc": "MESSENGER",
        "orbit": None,
        "start": "2011-04-11T00:00:00",
        "end": "2011-04-12T00:00:00",
    },
    {
        "name": "FTE_rain_DOY11327",
        "sc": "MESSENGER",
        "orbit": None,
        "start": "2011-11-23T08:00:00",
        "end": "2011-11-23T11:00:00",
    },
    {
        "name": "KTH_match_DOY11327",
        "sc": "MESSENGER",
        "orbit": None,
        "start": "2011-11-23T20:00:00",
        "end": "2011-11-23T23:00:00",
    },
]

df_roi = pd.DataFrame(roi_data)
# Convert strings to actual datetime objects for time-range filtering
df_roi["start"] = pd.to_datetime(df_roi["start"])
df_roi["end"] = pd.to_datetime(df_roi["end"])
# Make sure that orbit is a int
df_roi["orbit"] = df_roi["orbit"].astype("Int64")


def get_rois(
    spacecraft: Any = None,
    orbit: Any = None,
    start_time: Any = None,
    stop_time: Any = None,
    name: Any = None,
) -> Any:
    """Select reference regions of interest by spacecraft, orbit, time, or name.

    Args:
        spacecraft: Spacecraft name or collection of names.
        orbit: Orbit number or collection of orbit numbers.
        start_time: Optional beginning of a time window.
        stop_time: Optional end of a time window.
        name: ROI name or collection of names.

    Returns:
        A copy of the ROI table containing rows matching all supplied filters.
    """
    query_df: Any = df_roi.copy()

    # Helper function to ensure inputs are iterable lists
    def to_list(val: Any) -> Any:
        """Normalize a scalar filter to a one-item collection."""
        if val is None:
            return None
        # If it's already a list, tuple, or set, return it. Otherwise, wrap in a list.
        if isinstance(val, (list, tuple, set)):
            return val
        return [val]

    # Normalize inputs
    sc_list = to_list(spacecraft)
    orbit_list = to_list(orbit)
    name_list = to_list(name)

    # 1. Filter by Spacecraft
    if sc_list:
        query_df = query_df[query_df["sc"].isin(sc_list)]

    # 2. Filter by Orbit
    if orbit_list:
        query_df = query_df[query_df["orbit"].isin(orbit_list)]

    # 3. Filter by Name
    if name_list:
        query_df = query_df[query_df["name"].isin(name_list)]

    # 4. Filter by Timeframe (Overlap Logic)
    if start_time and stop_time:
        start_t = pd.to_datetime(start_time)
        stop_t = pd.to_datetime(stop_time)

        # Logic: ROI overlaps with window if it starts before window ends
        # AND ends after window starts.
        mask = (query_df["start"] <= stop_t) & (query_df["end"] >= start_t)
        query_df = query_df[mask]

    return query_df
