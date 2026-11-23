"""Annotate plot x-axis ticks with time-varying spacecraft coordinates."""

from typing import Any, Sequence


def add_ephemeris(
    fig: Any, ax: Any, coords: Any, labels: Sequence[str] = ["X", "Y", "Z"],
    units: str = "km", y_offset: float = 0.12, row_spacing: float = 0.22
) -> None:
    """Add interpolated ephemeris values below the x-axis tick labels.

    Args:
        fig: Matplotlib figure to annotate.
        ax: Axes whose major x ticks determine annotation positions.
        coords: Time-indexed xarray coordinates with a ``time`` dimension.
        labels: Coordinate component labels to display.
        units: Unit string shown beside each component label.
        y_offset: Vertical gap between ticks and the first ephemeris row.
        row_spacing: Vertical gap between successive component rows.
    """
    import numpy as np
    import pandas as pd

    # Ensure coordinates are sorted for interpolation
    coords = coords.sortby("time")

    # Force a draw to ensure ticks are updated
    fig.canvas.draw()

    # Infer font size from existing xticks
    tick_labels = ax.xaxis.get_ticklabels()
    fs = tick_labels[0].get_fontsize() if tick_labels else 10

    ticks = ax.get_xticks()
    # Convert matplotlib numeric dates to pandas datetime
    tick_times_raw = pd.to_datetime(ticks, unit="D", origin="1970-01-01")

    # Clip tick times to coordinate range to avoid precision-induced NaNs
    # at the very start/end of the axis.
    t_min, t_max = coords.time.min().values, coords.time.max().values
    tick_times = tick_times_raw.values.clip(t_min, t_max)

    # Add extrapolation support to interp
    pos_at_ticks = coords.interp(
        time=tick_times, method="nearest", kwargs={"fill_value": "extrapolate"}
    )

    bbox = ax.get_position()
    y_start = bbox.y0 - y_offset
    inv_fig = fig.transFigure.inverted()

    for i, label in enumerate(labels):
        # Adjusted row_spacing (suggested 0.04 instead of 0.22)
        y_pos = y_start - (i * row_spacing)

        # Row Label
        fig.text(
            bbox.x0 - 0.01,
            y_pos,
            f"{label} [{units}]",
            va="top",
            ha="right",
            fontsize=fs,
        )

        # Values aligned with each major tick
        for t_idx, val in enumerate(pos_at_ticks.data):
            # FIX 3: Robustly handle remaining NaNs (don't print "nan")
            if np.isnan(val[i]):
                continue

            v_text = f"{val[i]:.2f}"

            # Horizontal centering
            x_pos_data = ticks[t_idx]
            x_pos_fig = ax.get_xaxis_transform().transform((x_pos_data, 0))
            x_fig = inv_fig.transform(x_pos_fig)[0]

            # Only plot if within the horizontal axis bounds
            if bbox.x0 - 0.001 <= x_fig <= bbox.x1 + 0.001:
                fig.text(
                    x_fig,
                    y_pos,
                    v_text,
                    va="top",
                    ha="center",
                    fontsize=fs,
                )
