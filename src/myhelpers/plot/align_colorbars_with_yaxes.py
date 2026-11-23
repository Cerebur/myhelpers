"""Position colorbar axes relative to alternating y-axis labels."""

from typing import Any


def align_colorbars_with_yaxes(
    fig: Any,
    axs: Any,
    cbar_axes: Any,
    cbar_parents: Any,
    *,
    pad: float = 0.01,
    cbar_width: float = 0.015,
    shrink: float = 0.9,
    start_left: bool = True,
) -> None:
    """Align colorbar axes to parent plots and alternate them by side.

    Args:
        fig: Matplotlib figure containing the axes.
        axs: Ordered parent plot axes used to determine label sides.
        cbar_axes: Colorbar axes to reposition.
        cbar_parents: Parent axes corresponding to each colorbar axis.
        pad: Gap between each parent axis and its colorbar.
        cbar_width: Width of each colorbar in figure coordinates.
        shrink: Height multiplier applied to each colorbar.
        start_left: Place the first associated colorbar to the right when true.
    """
    fig.canvas.draw()

    # Map parent axis → index
    ax_to_index = {ax: i for i, ax in enumerate(axs)}
    for ax, cax in zip(cbar_parents, cbar_axes):
        i = ax_to_index[ax]
        # Determine side WITHOUT relying on current label state
        is_left = (i % 2 == 0) if start_left else (i % 2 != 0)

        bbox = ax.get_position()

        # Shrink height and center vertically
        new_height = bbox.height * shrink
        y_center = bbox.y0 + bbox.height / 2
        y0 = y_center - new_height / 2

        if is_left:
            # Place colorbar on RIGHT
            x0 = bbox.x1 + pad
        else:
            # Place colorbar on LEFT
            x0 = bbox.x0 - pad - cbar_width

        new_pos = [
            x0,
            y0,
            cbar_width,
            new_height,
        ]

        cax.set_position(new_pos)

        # Align colorbar label and ticks with layout
        cax.yaxis.set_label_position("right" if is_left else "left")

        cax.tick_params(
            axis="y",
            labelright=is_left,
            labelleft=not is_left,
        )

