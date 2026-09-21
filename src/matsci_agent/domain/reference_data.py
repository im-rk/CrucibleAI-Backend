"""Small offline table of well-known experimental band gaps.

Used only when no Materials Project key is configured, so the pipeline and demos still work.
These are literature room-temperature experimental values, NOT Materials Project (DFT) data,
and every record built from them says so.
"""

from __future__ import annotations

OFFLINE_SOURCE = "Offline reference (experimental literature value; not Materials Project)"

# formula -> (band gap in eV, direct gap?, crystal system)
OFFLINE_BANDGAPS: dict[str, tuple[float, bool, str]] = {
    "Si": (1.12, False, "cubic"),
    "Ge": (0.66, False, "cubic"),
    "GaAs": (1.42, True, "cubic"),
    "InP": (1.34, True, "cubic"),
    "GaP": (2.26, False, "cubic"),
    "CdTe": (1.5, True, "cubic"),
    "GaN": (3.4, True, "hexagonal"),
    "AlN": (6.2, True, "hexagonal"),
    "ZnO": (3.37, True, "hexagonal"),
    "SnO2": (3.6, True, "tetragonal"),
    "Cu2O": (2.1, True, "cubic"),
    "TiO2": (3.2, False, "tetragonal"),  # anatase
    "Al2O3": (8.8, True, "trigonal"),
    "MgO": (7.8, True, "cubic"),
    "BaTiO3": (3.2, False, "tetragonal"),
    "SrTiO3": (3.25, False, "cubic"),
    "CsPbI3": (1.73, True, "cubic"),  # alpha (black) phase
    "CsPbBr3": (2.3, True, "cubic"),
    "MAPbI3": (1.55, True, "tetragonal"),
}
