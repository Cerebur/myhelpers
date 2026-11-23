"""Defining planetary parameters."""


def planetary_radii(planet: str) -> float:
    """Returns the planetary radius of the specified planet.

    Args:
        planet: The planet in question

    Returns:
        Radius in km
    """
    radius_dict = {"venus": 6051.8, "mercury": 2439.7}

    return radius_dict[planet.lower()]
