"""Configure shared Matplotlib styles for spacecraft plots."""

import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc
from cycler import cycler

cset = tc.high_contrast

tol_bright_components = [
    cset.blue,  # blue
    cset.yellow,  # green
    cset.red,  # red
    "k",
]


def use_my_style(
    use_tex: bool = False, aspect: str | None = None, theme: str = "light",
    use_tol: bool = True, fontsize: str = "normal"
) -> None:
    """Set global Matplotlib style parameters used by this package.

    Args:
        use_tex: Enable LaTeX text rendering.
        aspect: Set figure width and height equally when ``"equal"``.
        theme: Figure theme, either ``"light"`` or ``"dark"``.
        use_tol: Use the Tol high-contrast color cycle.
        fontsize: Font preset, either ``"normal"`` or ``"small"``.
    """
    fig_width_pt = 347.5 * 1.6
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / (2.0)  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inlw=2ches
    if aspect == "equal":
        fig_height = fig_width
    else:
        fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    if use_tex:
        preamble = (
            r"\usepackage{amsmath}"
            + "\n"
            + r"\usepackage{amssymb}"
            + "\n"
            + r"\usepackage{siunitx}"
        )
        plt.rcParams["text.latex.preamble"] = preamble
        font = "Computer Modern Roman"
    else:
        font = "DejaVu Sans"

    if theme == "dark":
        bg_color = "black"
        fg_color = "white"
    elif theme == "light":
        bg_color = "white"
        fg_color = "black"
    else:
        raise NotImplementedError("The requested theme is not implemented")
    if fontsize == "normal":
        label = 24
        fontsize_var = 24
        legend = 22
        labelxtick = 24
        labelytick = 24
    elif fontsize == "small":
        label = 16
        fontsize_var = 16
        legend = 13
        labelytick = 16
        labelxtick = 16
    else:
        raise NotImplementedError(
            f"Requested font is {fontsize}, but only normal and small are supported"
        )
    params = {
        # "backend": "pdf",
        "text.usetex": use_tex,
        "font.weight": "normal",
        "axes.linewidth": 1.5,
        "axes.labelsize": label,
        "font.size": fontsize_var,
        "legend.fontsize": legend,
        "xtick.labelsize": labelxtick,
        "ytick.labelsize": labelytick,
        "font.family": font,
        "figure.figsize": fig_size,
        # ---- COLORS ----
        "figure.facecolor": bg_color,
        "axes.facecolor": bg_color,
        "axes.edgecolor": fg_color,
        "axes.labelcolor": fg_color,
        "text.color": fg_color,
        "xtick.color": fg_color,
        "ytick.color": fg_color,
        "grid.color": fg_color,
        "savefig.facecolor": bg_color,  # important when saving!
        # ---- TICKS ----
        "xtick.top": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.top": True,
        "xtick.minor.bottom": True,
        "ytick.minor.left": True,
        "ytick.minor.right": True,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.minor.width": 1,
        "ytick.minor.width": 1,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.pad": 8,
        "xtick.minor.pad": 8,
    }

    plt.rcParams.update(params)
    if use_tol:
        plt.rcParams["axes.prop_cycle"] = cycler(color=tol_bright_components)
