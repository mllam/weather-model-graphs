"""
Backward-compatibility re-exports from ``layout.rectilinear``.

All coordinate creation functions have been moved to
``wmg.create.mesh.layout.rectilinear``.  This module re-exports them
so that existing imports continue to work.
"""

from .connectivity.general import create_directed_mesh_graph
from .layout.rectilinear import (
    create_multirange_2d_mesh_graphs,
    create_multirange_2d_mesh_primitives,
    create_single_level_2d_mesh_graph,
    create_single_level_2d_mesh_primitive,
)

__all__ = [
    "create_directed_mesh_graph",
    "create_multirange_2d_mesh_graphs",
    "create_multirange_2d_mesh_primitives",
    "create_single_level_2d_mesh_graph",
    "create_single_level_2d_mesh_primitive",
]
