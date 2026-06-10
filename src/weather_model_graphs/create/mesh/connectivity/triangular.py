"""
Triangular mesh connectivity functions.

Coordinate creation (primitives) lives in ``layout.triangular``.
This module contains only triangular-specific *connectivity* logic
that cannot be handled by the generic connectivity functions in
``flat.py`` or ``hierarchical.py``.

The generic ``create_hierarchical_from_coordinates`` already works with
triangular primitives, so no triangular-specific hierarchical function
is needed here.
"""

from typing import List

import networkx
import numpy as np
import scipy.spatial

from ....networkx_utils import prepend_node_index
from ..layout.triangular import (
    create_multirange_2d_triangular_mesh_primitives,
    create_single_level_2d_triangular_mesh_graph,
    create_single_level_2d_triangular_mesh_primitive,
)
from .general import create_directed_mesh_graph

# Re-export layout functions for backward compatibility
__all__ = [
    "create_single_level_2d_triangular_mesh_primitive",
    "create_multirange_2d_triangular_mesh_primitives",
    "create_single_level_2d_triangular_mesh_graph",
    "create_flat_multiscale_from_triangular_coordinates",
]


def create_flat_multiscale_from_triangular_coordinates(
    G_coords_list: List[networkx.Graph],
    pattern: str = "4-star",
) -> networkx.DiGraph:
    """
    Create flat multiscale mesh graph from a list of triangular coordinate
    graphs.

    Unlike the rectilinear variant (``create_flat_multiscale_from_coordinates``)
    which relies on grid-index-based coincident-node detection, this function
    uses position-based matching.  For each coarser level, any node whose
    position coincides (within floating-point tolerance) with an existing finer
    level node is merged with it, so that multi-resolution edges share the
    same node identity.

    Parameters
    ----------
    G_coords_list : list[networkx.Graph]
        One undirected triangular mesh primitive per level.
    pattern : str
        Connectivity pattern: ``"4-star"`` or ``"8-star"`` (default ``"4-star"``).

    Returns
    -------
    networkx.DiGraph
        Flat multiscale triangular mesh graph.
    """
    # Convert each level to directed graph
    G_directed = [create_directed_mesh_graph(g, pattern=pattern) for g in G_coords_list]

    # Prepend level index to make node labels unique across levels
    G_directed = [
        prepend_node_index(g, level_i) for level_i, g in enumerate(G_directed)
    ]

    # Build merged graph, starting from finest level
    G_tot = G_directed[0]

    for lev in range(1, len(G_directed)):
        G_coarse = G_directed[lev]

        # KDTree of existing (finer) nodes for position matching
        fine_nodes = list(G_tot.nodes())
        fine_positions = np.array([G_tot.nodes[n]["pos"] for n in fine_nodes])
        kdt = scipy.spatial.KDTree(fine_positions)

        # Find which coarse nodes coincide with existing fine nodes
        relabel_map = {}
        for node in G_coarse.nodes():
            pos = G_coarse.nodes[node]["pos"]
            dist, idx = kdt.query(pos)
            if dist < 1e-8:
                relabel_map[node] = fine_nodes[idx]

        if relabel_map:
            G_coarse = networkx.relabel_nodes(G_coarse, relabel_map)

        G_tot = networkx.compose(G_tot, G_coarse)

    # Re-index to sequential (0, i) labels
    G_tot = prepend_node_index(G_tot, 0)

    # Preserve dx/dy as per-level dicts
    G_tot.graph["dx"] = {i: g.graph["dx"] for i, g in enumerate(G_directed)}
    G_tot.graph["dy"] = {i: g.graph["dy"] for i, g in enumerate(G_directed)}

    return G_tot
