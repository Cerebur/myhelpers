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


def mercury_dipole_offset() -> float:
    """Returns the dipole offset of Mercury in km.

    Mercury's dipole is offset northward from the planet center along the spin axis.

    Args:
        None

    Returns:
        Dipole offset in km
    """
    return 479
