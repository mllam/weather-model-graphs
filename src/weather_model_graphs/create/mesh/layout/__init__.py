"""
Mesh layout modules.

Each layout module defines how mesh node coordinates are placed in space
(coordinate creation step).  The resulting undirected primitive graphs are
then consumed by the connectivity modules to produce directed mesh graphs.

Available layouts:

- ``rectilinear``: nodes placed on a uniform rectangular grid.
- ``triangular``: nodes placed on a regular (equilateral) triangular lattice.
"""

from . import rectilinear, triangular

__all__ = ["rectilinear", "triangular"]
