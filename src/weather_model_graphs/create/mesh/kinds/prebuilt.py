"""
Functions for handling user-provided (prebuilt) mesh graphs.

Supports two modes:

Nodes+edges mode
    User provides a complete ``nx.DiGraph`` with all node and edge attributes.
    Validated and used as-is.

Nodes-only mode
    User provides an ``nx.Graph`` with only nodes (having ``pos`` and ``type``
    attributes). The library builds connectivity edges using Delaunay
    triangulation.

This enables use of arbitrary mesh topologies (ICON icosahedral, MPAS Voronoi,
custom observation networks, etc.) without the library needing to generate them.
"""

import networkx
import numpy as np
import scipy.spatial


def validate_prebuilt_nodes(mesh_graph):
    """
    Validate that all nodes in a prebuilt mesh graph have the required
    attributes for nodes-only mode.

    Each node must have:

    - ``pos``: ``np.ndarray`` of shape ``(2,)`` with ``(x, y)`` coordinates
    - ``type``: ``str``, should be ``"mesh"``

    Parameters
    ----------
    mesh_graph : networkx.Graph or networkx.DiGraph
        Graph with nodes to validate.

    Raises
    ------
    ValueError
        If the graph has no nodes or any node is missing required attributes.
    """
    if mesh_graph.number_of_nodes() == 0:
        raise ValueError(
            "mesh_layout='prebuilt' requires a mesh_graph with at least one "
            "node."
        )

    for node, data in mesh_graph.nodes(data=True):
        if "pos" not in data:
            raise ValueError(
                f"Node {node} is missing required 'pos' attribute. "
                "All nodes in a prebuilt mesh must have 'pos' as "
                "np.ndarray of shape (2,)."
            )
        pos = data["pos"]
        if not hasattr(pos, "shape") or pos.shape != (2,):
            raise ValueError(
                f"Node {node} has 'pos' with invalid shape. "
                f"Expected shape (2,), got {getattr(pos, 'shape', 'N/A')}."
            )
        if "type" not in data:
            raise ValueError(
                f"Node {node} is missing required 'type' attribute. "
                "All nodes in a prebuilt mesh must have type='mesh'."
            )


def validate_prebuilt_nodes_with_levels(mesh_graph):
    """
    Validate prebuilt nodes that include level attributes for multi-level modes.

    In addition to the base node requirements (``pos``, ``type``), each node
    must have an integer ``level`` attribute.

    Parameters
    ----------
    mesh_graph : networkx.Graph or networkx.DiGraph
        Graph with nodes to validate.

    Raises
    ------
    ValueError
        If any node is missing the ``level`` attribute.
    """
    validate_prebuilt_nodes(mesh_graph)

    for node, data in mesh_graph.nodes(data=True):
        if "level" not in data:
            raise ValueError(
                f"Node {node} is missing required 'level' attribute. "
                "For multi-level prebuilt meshes, all nodes must have "
                "an integer 'level' attribute (0 = finest)."
            )


def validate_prebuilt_mesh_edges(mesh_graph, require_levels=False):
    """
    Validate that a complete prebuilt DiGraph has all required node and edge
    attributes for nodes+edges mode.

    Parameters
    ----------
    mesh_graph : networkx.DiGraph
        Directed graph to validate.
    require_levels : bool
        If True, also check that nodes have ``level`` attributes and that
        at least some edges have a ``level`` attribute (required for
        hierarchical connectivity where level-based graph splitting is used).

    Raises
    ------
    TypeError
        If the graph is not a ``networkx.DiGraph``.
    ValueError
        If any required attribute is missing.
    """
    if not isinstance(mesh_graph, networkx.DiGraph):
        raise TypeError(
            "Prebuilt mesh with edges must be a networkx.DiGraph, "
            f"got {type(mesh_graph).__name__}."
        )

    if require_levels:
        validate_prebuilt_nodes_with_levels(mesh_graph)
    else:
        validate_prebuilt_nodes(mesh_graph)

    if mesh_graph.number_of_edges() == 0:
        raise ValueError(
            "Provided mesh_graph has nodes but no edges in nodes+edges mode. "
            "For nodes+edges mode, the graph must contain edges with "
            "'len' and 'vdiff' attributes."
        )

    for u, v, data in mesh_graph.edges(data=True):
        if "len" not in data:
            raise ValueError(
                f"Edge ({u}, {v}) is missing required 'len' attribute."
            )
        if "vdiff" not in data:
            raise ValueError(
                f"Edge ({u}, {v}) is missing required 'vdiff' attribute."
            )

    if require_levels:
        has_level_edge = any(
            "level" in data for _, _, data in mesh_graph.edges(data=True)
        )
        if not has_level_edge:
            raise ValueError(
                "For hierarchical prebuilt meshes, at least some edges must "
                "have a 'level' attribute (needed for level-based graph "
                "splitting)."
            )


def _build_edges_delaunay(G_nodes):
    """
    Build directed edges from node positions using Delaunay triangulation.

    For graphs with fewer than 3 nodes, creates a complete graph (since
    Delaunay triangulation requires at least 3 points).

    Parameters
    ----------
    G_nodes : networkx.Graph
        Input graph with nodes having ``pos`` attributes.

    Returns
    -------
    networkx.DiGraph
        Directed graph with bidirectional edges, each having ``len`` and
        ``vdiff`` attributes.
    """
    node_list = list(G_nodes.nodes())
    positions = np.array([G_nodes.nodes[n]["pos"] for n in node_list])

    dg = networkx.DiGraph()
    dg.add_nodes_from(G_nodes.nodes(data=True))
    dg.graph.update(G_nodes.graph)

    if len(node_list) < 2:
        # Single node or empty: no edges possible
        return dg

    if len(node_list) < 3:
        # Two nodes: create complete graph (bidirectional edge)
        for i, u in enumerate(node_list):
            for j, v in enumerate(node_list):
                if i != j:
                    d = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
                    dg.add_edge(
                        u, v, len=d, vdiff=positions[i] - positions[j]
                    )
        return dg

    # Delaunay triangulation for >= 3 nodes
    try:
        tri = scipy.spatial.Delaunay(positions)
    except scipy.spatial.QhullError:
        # Degenerate case (e.g. collinear points): fall back to connecting
        # each node to its nearest neighbour via KDTree so we still
        # return a connected graph.
        kdt = scipy.spatial.KDTree(positions)
        edges_set = set()
        for idx in range(len(node_list)):
            # k=2 because the first result is the point itself
            _, neigh = kdt.query(positions[idx], k=min(2, len(node_list)))
            if len(node_list) >= 2:
                neigh_idx = neigh[1] if np.ndim(neigh) > 0 else neigh
                a, b = idx, int(neigh_idx)
                edges_set.add((min(a, b), max(a, b)))

        for idx_i, idx_j in edges_set:
            u = node_list[idx_i]
            v = node_list[idx_j]
            pos_u = positions[idx_i]
            pos_v = positions[idx_j]
            d = np.sqrt(np.sum((pos_u - pos_v) ** 2))
            dg.add_edge(u, v, len=d, vdiff=pos_u - pos_v)
            dg.add_edge(v, u, len=d, vdiff=pos_v - pos_u)
        return dg

    # Extract unique edges from triangulation simplices
    edges_set = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                edge = (min(a, b), max(a, b))
                edges_set.add(edge)

    # Create bidirectional edges with len and vdiff
    for idx_i, idx_j in edges_set:
        u = node_list[idx_i]
        v = node_list[idx_j]
        pos_u = positions[idx_i]
        pos_v = positions[idx_j]
        d = np.sqrt(np.sum((pos_u - pos_v) ** 2))
        dg.add_edge(u, v, len=d, vdiff=pos_u - pos_v)
        dg.add_edge(v, u, len=d, vdiff=pos_v - pos_u)

    return dg


def create_prebuilt_flat_from_nodes(G_nodes):
    """
    Create a flat directed mesh graph from prebuilt nodes using Delaunay
    triangulation.

    This is used when ``mesh_layout="prebuilt"`` and
    ``m2m_connectivity="flat"`` with a nodes-only graph (no edges provided).

    Parameters
    ----------
    G_nodes : networkx.Graph
        Undirected graph with nodes having ``pos`` and ``type`` attributes.

    Returns
    -------
    networkx.DiGraph
        Directed flat mesh graph with ``len`` and ``vdiff`` edge attributes.
    """
    validate_prebuilt_nodes(G_nodes)
    return _build_edges_delaunay(G_nodes)


def create_prebuilt_flat_multiscale_from_nodes(G_nodes):
    """
    Create a flat multiscale directed mesh graph from prebuilt nodes with
    level attributes.

    Splits nodes by level, builds Delaunay connectivity per level,
    then merges all levels into a single flat graph.

    Parameters
    ----------
    G_nodes : networkx.Graph
        Graph with nodes having ``pos``, ``type``, and ``level`` attributes.

    Returns
    -------
    networkx.DiGraph
        Merged flat multiscale directed mesh graph.
    """
    validate_prebuilt_nodes_with_levels(G_nodes)

    levels = sorted(set(G_nodes.nodes[n]["level"] for n in G_nodes.nodes()))

    G_all_levels = []
    for lev in levels:
        # Get nodes for this level
        level_nodes = [
            n for n in G_nodes.nodes() if G_nodes.nodes[n]["level"] == lev
        ]
        G_level = G_nodes.subgraph(level_nodes).copy()
        G_level.graph["level"] = lev

        # Build connectivity for this level using Delaunay
        G_directed = _build_edges_delaunay(G_level)
        G_directed.graph["level"] = lev

        # Add level attribute to edges
        for u, v in G_directed.edges():
            G_directed.edges[u, v]["level"] = lev

        G_all_levels.append(G_directed)

    # Merge all levels into a single flat graph
    G_tot = networkx.compose_all(G_all_levels)
    return G_tot


def create_prebuilt_hierarchical_from_nodes(G_nodes, inter_level=None):
    """
    Create a hierarchical directed mesh graph from prebuilt nodes with
    level attributes.

    Splits nodes by level, builds Delaunay connectivity per level for
    intra-level edges, then builds inter-level up/down connections using
    KDTree nearest-neighbour search.

    Parameters
    ----------
    G_nodes : networkx.Graph
        Graph with nodes having ``pos``, ``type``, and ``level`` attributes.
    inter_level : dict or None
        Inter-level connectivity options:

        - ``k``: int, number of nearest neighbours for inter-level
          connections (default: 1)

    Returns
    -------
    networkx.DiGraph
        Hierarchical directed mesh graph with intra-level
        (direction="same"), inter-level down (direction="down"), and
        inter-level up (direction="up") edges.

    Raises
    ------
    ValueError
        If fewer than 2 levels are provided.
    """
    validate_prebuilt_nodes_with_levels(G_nodes)

    if inter_level is None:
        inter_level = {"pattern": "nearest", "k": 1}
    inter_k = inter_level.get("k", 1)

    # Split nodes by level
    levels = sorted(set(G_nodes.nodes[n]["level"] for n in G_nodes.nodes()))

    if len(levels) < 2:
        raise ValueError(
            "At least two mesh levels are required for hierarchical mesh "
            "graph."
        )

    # Build intra-level connectivity using Delaunay
    Gs_all_levels = []
    for lev in levels:
        level_nodes = [
            n for n in G_nodes.nodes() if G_nodes.nodes[n]["level"] == lev
        ]
        G_level = G_nodes.subgraph(level_nodes).copy()
        G_level.graph["level"] = lev

        G_directed = _build_edges_delaunay(G_level)
        G_directed.graph["level"] = lev
        Gs_all_levels.append(G_directed)

    # Relabel nodes with level index prefix: node -> (level_idx, node)
    for i, G in enumerate(Gs_all_levels):
        mapping = {node: (i, node) for node in G.nodes}
        Gs_all_levels[i] = networkx.relabel_nodes(G, mapping, copy=True)

    # Add direction and level attributes to intra-level edges
    for i, G in enumerate(Gs_all_levels):
        for u, v in G.edges:
            G.edges[u, v]["direction"] = "same"
            G.edges[u, v]["level"] = i

    # Build inter-level connections (coarser level -> finer level)
    up_graphs = []
    down_graphs = []
    for G_from, G_to in zip(Gs_all_levels[1:], Gs_all_levels[:-1]):
        from_level = G_from.graph["level"]
        to_level = G_to.graph["level"]

        # Start from coarser level graph (without edges)
        G_down = G_from.copy()
        G_down.clear_edges()
        G_down = networkx.DiGraph(G_down)
        G_down.add_nodes_from(G_to.nodes(data=True))

        # Build KDTree from coarser level node positions
        v_to_list = list(G_to.nodes)
        v_from_list = list(G_from.nodes)
        v_from_xy = np.array(
            [G_from.nodes[n]["pos"] for n in v_from_list]
        )
        kdt_m = scipy.spatial.KDTree(v_from_xy)

        # Clamp k to the number of available coarser-level nodes
        effective_k = min(inter_k, len(v_from_list))

        # Connect finer level nodes to nearest coarser level nodes
        for v in v_to_list:
            neigh_idx = kdt_m.query(
                G_down.nodes[v]["pos"], effective_k
            )[1]
            if effective_k == 1:
                neigh_idx = [neigh_idx]

            for idx in neigh_idx:
                u = v_from_list[idx]
                d = np.sqrt(
                    np.sum(
                        (G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"])
                        ** 2
                    )
                )
                G_down.add_edge(u, v)
                G_down.edges[u, v]["len"] = d
                G_down.edges[u, v]["vdiff"] = (
                    G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]
                )
                G_down.edges[u, v]["levels"] = f"{from_level}>{to_level}"
                G_down.edges[u, v]["direction"] = "down"

        # Create up edges (reverse of down)
        G_up = networkx.DiGraph()
        G_up.add_nodes_from(G_down.nodes(data=True))
        for u, v, data in G_down.edges(data=True):
            data = data.copy()
            data["levels"] = f"{to_level}>{from_level}"
            data["direction"] = "up"
            G_up.add_edge(v, u, **data)

        up_graphs.append(G_up)
        down_graphs.append(G_down)

    # Compose all level and inter-level graphs
    G_up_all = networkx.compose_all(up_graphs)
    G_down_all = networkx.compose_all(down_graphs)
    G_all_levels = networkx.compose_all(Gs_all_levels)

    G_m2m = networkx.compose_all([G_all_levels, G_up_all, G_down_all])

    return G_m2m
