"""Coordinate transformations for solar-wind aberration."""

import numpy as np


def transform_to_aberrated_coordinates(
    vectors: np.ndarray, aberration_angles: np.ndarray | float
) -> np.ndarray:
    """Rotate vectors into the aberrated coordinate frame.

    The transformation is a passive rotation of the coordinate frame about the
    positive z axis. Angles are in degrees, with positive angles rotating the
    frame counterclockwise when viewed from the positive z direction. The z
    component is unchanged.

    Args:
        vectors: Array of shape ``(N, 3)`` containing the input x, y, and z
            components.
        aberration_angles: Aberration angle in degrees for each vector, as a
            one-dimensional array of shape ``(N,)``. For ``N == 1``, a single
            float may be supplied.

    Returns:
        A floating-point array of shape ``(N, 3)`` in aberrated coordinates.

    Raises:
        ValueError: If ``vectors`` does not have shape ``(N, 3)`` or the angle
            input does not match the number of vectors.
    """
    vector_array = np.asarray(vectors, dtype=float)
    if vector_array.ndim != 2 or vector_array.shape[1] != 3:
        raise ValueError("vectors must have shape (N, 3).")

    angle_array = np.asarray(aberration_angles, dtype=float)
    if angle_array.ndim == 0:
        if vector_array.shape[0] != 1:
            raise ValueError("A scalar aberration angle can only be used when N == 1.")
        angle_array = angle_array.reshape(1)
    elif angle_array.ndim != 1 or angle_array.shape[0] != vector_array.shape[0]:
        raise ValueError("aberration_angles must have shape (N,).")

    angle_radians = np.deg2rad(angle_array)
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)

    transformed = np.empty_like(vector_array)
    transformed[:, 0] = cosine * vector_array[:, 0] - sine * vector_array[:, 1]
    transformed[:, 1] = sine * vector_array[:, 0] + cosine * vector_array[:, 1]
    transformed[:, 2] = vector_array[:, 2]
    return transformed
