"""Coordinate transformations to the Mercury Solar Magnetic frame."""

import numpy as np

from myhelpers.parameters import mercury_dipole_offset


def mso_to_msm(coordinates: np.ndarray) -> np.ndarray:
    """Translate MSO position coordinates into the MSM frame.

    MSO and MSM axes are parallel; the MSM origin is displaced northward
    along ``+Z`` by Mercury's dipole offset. The offset is given in kilometers,
    so input coordinates must also be in kilometers.

    Args:
        coordinates: A position vector with shape ``(3,)`` or an array of
            position vectors with shape ``(N, 3)`` in MSO coordinates. Unit must be km

    Returns:
        A floating-point array in MSM coordinates with the same shape as the
        input. The ``Z`` component is reduced by the dipole offset. Units will be km

    Raises:
        ValueError: If ``coordinates`` does not have shape ``(3,)`` or
            ``(N, 3)``.
    """
    positions = np.asarray(coordinates, dtype=float)
    if positions.ndim == 1:
        if positions.shape[0] != 3:
            raise ValueError("A single position vector must have shape (3,).")
    elif positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Position vectors must have shape (3,) or (N, 3).")

    transformed = positions.copy()
    transformed[..., 2] -= mercury_dipole_offset()
    return transformed
