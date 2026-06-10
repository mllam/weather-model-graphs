"""
Mesh layout modules.

Each layout module defines how mesh node coordinates are placed in space
(coordinate creation step).  The resulting undirected primitive graphs are
then consumed by the connectivity modules to produce directed mesh graphs.

Available layouts:

- ``rectilinear``: Uniform rectangular grid (``grid_2d_graph``).
- ``triangular``: Regular triangular lattice (``triangular_lattice_graph``).
"""

from . import rectilinear, triangular

__all__ = ["rectilinear", "triangular"]
