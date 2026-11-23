"""Arrange and align y-axis labels across Matplotlib axes."""

from typing import Any

import numpy as np


def format_yaxes(
    fig: Any,
    axs: Any,
    *,
    cbar_axes: Any = None,
    cbar_parents: Any = None,
    alternate: bool = True,
    start_left: bool = True,
    align_labels: bool = True,
    label_offset: float = 0.015,
) -> None:
    """Place y-axis labels on alternating sides and optionally align them.

    Args:
        fig: Matplotlib figure containing the axes.
        axs: One axes object or a sequence/array of axes to arrange.
        cbar_axes: Optional colorbar axes to include in label alignment.
        cbar_parents: Parent axes corresponding to ``cbar_axes``.
        alternate: Alternate label sides across axes.
        start_left: Put the first axis label on the left when true.
        align_labels: Align labels to the widest tick labels on each side.
        label_offset: Figure-relative spacing between tick labels and axis labels.
    """
    if not isinstance(axs, (list, tuple, np.ndarray)):
        axs = [axs]

    left_axes = []
    right_axes = []

    ax_to_index = {ax: i for i, ax in enumerate(axs)}

    for i, ax in enumerate(axs):
        is_left = (i % 2 == 0) if start_left else (i % 2 != 0) if alternate else True
        if is_left:
            ax.yaxis.set_label_position("left")
            ax.tick_params(
                axis="y",
                which="both",
                left=True,
                right=True,
                labelleft=True,
                labelright=False,
            )
            left_axes.append(ax)
        else:
            ax.yaxis.set_label_position("right")
            ax.tick_params(
                axis="y",
                which="both",
                left=True,
                right=True,
                labelleft=False,
                labelright=True,
            )
            right_axes.append(ax)

    if cbar_axes is not None and cbar_parents is not None:
        for cax, ax in zip(cbar_axes, cbar_parents):
            i = ax_to_index[ax]
            is_left = (i % 2 == 0) if start_left else (i % 2 != 0)
            if is_left:
                right_axes.append(cax)
            else:
                left_axes.append(cax)

    if align_labels:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv_fig = fig.transFigure.inverted()

        # --- Align Left Side ---
        if left_axes:
            min_fig_x = 1.0
            for ax in left_axes:
                # Temporarily hide the label to measure just the ticks
                lbl = ax.yaxis.get_label()
                orig_viz = lbl.get_visible()
                lbl.set_visible(False)

                bbox = ax.yaxis.get_tightbbox(renderer)
                if bbox is not None:
                    fig_bbox = inv_fig.transform(bbox)
                    if fig_bbox[0, 0] < min_fig_x:
                        min_fig_x = fig_bbox[0, 0]

                lbl.set_visible(orig_viz)

            # Place all labels exactly label_offset% figure width to the left of the widest tick
            label_fig_x = min_fig_x - label_offset

            for ax in left_axes:
                ax_bbox = ax.get_position()
                ax_x = (label_fig_x - ax_bbox.x0) / ax_bbox.width
                ax.yaxis.set_label_coords(ax_x, 0.5)

        # --- Align Right Side ---
        if right_axes:
            max_fig_x = 0.0
            for ax in right_axes:
                lbl = ax.yaxis.get_label()
                orig_viz = lbl.get_visible()
                lbl.set_visible(False)

                bbox = ax.yaxis.get_tightbbox(renderer)
                if bbox is not None:
                    fig_bbox = inv_fig.transform(bbox)
                    if fig_bbox[1, 0] > max_fig_x:
                        max_fig_x = fig_bbox[1, 0]

                lbl.set_visible(orig_viz)

            # Place all labels exactly label_offset% figure width to the right of the widest tick
            label_fig_x = max_fig_x + label_offset

            for ax in right_axes:
                ax_bbox = ax.get_position()
                ax_x = (label_fig_x - ax_bbox.x0) / ax_bbox.width
                ax.yaxis.set_label_coords(ax_x, 0.5)
