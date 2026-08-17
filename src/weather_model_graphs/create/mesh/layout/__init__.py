"""
Mesh layout modules.

Each layout module defines how mesh node coordinates are placed in space
(coordinate creation step).  The resulting undirected primitive graphs are
then consumed by the connectivity modules to produce directed mesh graphs.

Available layouts:

- ``rectilinear``: nodes placed on a uniform rectangular grid.
- ``triangular``: nodes placed on a regular (equilateral) triangular lattice.
- ``prebuilt``: nodes taken from a user-provided mesh graph (edge-less node
  clouds; mesh edges are built in the connectivity step).
"""

from . import prebuilt, rectilinear, triangular

__all__ = ["prebuilt", "rectilinear", "triangular"]
