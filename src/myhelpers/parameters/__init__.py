"""Functions for loading path variables."""

from .paths import DATA_PTH_LOCAL, DATA_PTH_REMOTE, DIR_PTH, PLOT_PTH, SPICE_PTH_REMOTE
from .planetary_parameters import planetary_radii

__all__ = [
    "DATA_PTH_LOCAL",
    "DATA_PTH_REMOTE",
    "DIR_PTH",
    "PLOT_PTH",
    "SPICE_PTH_REMOTE",
    "planetary_radii",
]
