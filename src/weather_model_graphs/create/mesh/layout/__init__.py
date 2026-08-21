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

#: Names of the mesh layouts supported by
#: :func:`weather_model_graphs.create.create_all_graph_components`.
#:
#: This is the single source of truth for which layouts exist: adding a new
#: layout module means adding its name here, and every consumer (argument
#: validation, error messages, and downstream tools such as neural-lam's
#: ``create_graph_with_wmg`` CLI) picks it up automatically.  Re-exported as
#: ``weather_model_graphs.create.MESH_LAYOUT_OPTIONS``.
MESH_LAYOUT_OPTIONS = ("rectilinear", "triangular")

__all__ = ["MESH_LAYOUT_OPTIONS", "rectilinear", "triangular"]
