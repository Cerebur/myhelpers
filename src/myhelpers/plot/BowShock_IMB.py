"""Plot and calculate Venus bow shock and ion-boundary curves."""

from typing import Any

import numpy as np


def BowShock_IMB(
    x0: float, x1: float, half: bool = False, color: Any = "k", lw: float = 2
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    """Calculate the Venus IMB and bow shock curves over an x range.

    Args:
        x0: Beginning of the x-coordinate range in Venus radii.
        x1: End of the x-coordinate range in Venus radii.
        half: Draw the half-circle arc when true.
        color: Matplotlib color used for the returned IMB arc.
        lw: Line width assigned to the IMB arc.

    Returns:
        IMB radial values, an IMB Arc patch, bow shock coordinates, and x values.
    """
    import matplotlib.patches as patch
    import numpy as np

    # IMB calculation
    x_lims = np.arange(x0 - 0.01, x1 + 0.01, 0.01)
    IMB = []
    for x in x_lims:
        k = -0.097
        d = 1.109
        if x <= 0.0:
            r = k * x + d
        elif 0.0 < x <= d:
            # A circle with radius 1.109
            r = np.sqrt(d * d - x * x)
        else:
            r = 0.0
        IMB.append(r)
    IMB = np.array(IMB)
    if not half:
        IMB_circle = patch.Arc(
            (0, 0),
            2 * 1.109,
            2 * 1.109,
            angle=0.0,
            theta1=0.0,
            theta2=90.0,
            color=color,
            lw=lw,
            ls="--",
        )
    else:
        IMB_circle = patch.Arc(
            (0, 0),
            2 * 1.109,
            2 * 1.109,
            angle=0.0,
            theta1=-90.0,
            theta2=90.0,
            color=color,
            lw=lw,
            ls="--",
        )

    # Bow Shock calculation
    l = 1.303
    e = 1.056
    xref = 0.788
    angle = np.arange(0, 180, 0.01) * np.pi / 180
    BS_radius = l / (1 + e * np.cos(angle))
    BS_x = BS_radius * np.cos(angle) + xref
    BS_y = BS_radius * np.sin(angle)
    BS = np.array([BS_x[BS_x < BS_radius + xref], BS_y[BS_x < BS_radius + xref]])

    return IMB, IMB_circle, BS, x_lims


def plot_venus_bs_imb(
    ax: Any, x_lims: Any = -6, color: Any = "k", lw: float = 2
) -> Any:
    """Draw the Venus IMB and bow shock in an x-y Matplotlib axes.

    Args:
        ax: Matplotlib axes on which to draw the boundaries.
        x_lims: Negative x limit used to define the plotted range.
        color: Single color or mapping with ``icb`` and ``bs`` colors.
        lw: Base line width for the boundaries.

    Returns:
        Tuples containing the two IMB line artists and two bow shock artists.
    """
    if isinstance(color, dict):
        if "icb" in color.keys():
            color_icb = color["icb"]
        else:
            color_icb = "k"
        if "bs" in color.keys():
            color_bs = color["bs"]
        else:
            color_bs = "k"
    else:
        color_icb = color
        color_bs = color

    # circle = plt.Circle((0, 0), 1, color=color, fill=False, lw=2)
    # ax.add_artist(circle)

    ICB_line, ICB_circle, BS, x_lims = BowShock_IMB(
        x_lims,
        0,
        half=True,
        color=color_icb,
        lw=lw,
    )
    (icb_line1,) = ax.plot(x_lims, ICB_line, color=color_icb, linestyle="--", lw=lw)
    (bs_line1,) = ax.plot(
        BS[0, :], BS[1, :], color=color_bs, linestyle=(0, (3, 1, 1, 1)), lw=lw + 2
    )
    ax.add_artist(ICB_circle)

    (icb_line2,) = ax.plot(x_lims, -ICB_line, color=color_icb, linestyle="--", lw=lw)
    (bs_line2,) = ax.plot(
        BS[0, :], -BS[1, :], color=color_bs, lw=lw + 2, linestyle=(0, (3, 1, 1, 1))
    )
    ax.add_artist(ICB_circle)

    return (icb_line1, icb_line2), (bs_line1, bs_line2)


def plot_venus_bs_imb_yz(ax: Any) -> None:
    """Draw circular Venus, IMB, and bow shock boundaries in a y-z view.

    Args:
        ax: Matplotlib axes on which to draw the circles.
    """
    import matplotlib.pyplot as plt
    plt: Any

    # Plot the Venus surface
    circle_venus = plt.Circle((0, 0), 1, color="k", fill=False, lw=2)
    ax.add_artist(circle_venus)

    # Plot the Bow shock and IMB at the terminator
    IMB_circle = plt.Circle((0, 0), 1.109, color="k", ls="--", fill=False, lw=2)
    BS_circle = plt.Circle((0, 0), 2.091, color="k", ls="-", fill=False, lw=2)
    ax.add_artist(IMB_circle)
    ax.add_artist(BS_circle)
