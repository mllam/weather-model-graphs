"""
Prebuilt mesh layout: coordinate creation from user-provided mesh nodes.

Unlike the generated layouts (``rectilinear``, ``triangular``), the prebuilt
layout does not place mesh nodes itself -- the user supplies their own node
positions (e.g. ICON grid vertices, an observation-station network, or any
custom point set) and the library builds the graph around them.

This is the coordinate creation step in the two-step mesh creation process:

1. **Coordinate creation** (this module) -> edge-less ``nx.Graph`` (a "node
   cloud") with ``pos`` and ``type`` node attributes.  No adjacency edges are
   created here: how a point cloud gets connected is a *connectivity*
   decision, so edge construction happens in the connectivity step (see
   ``create_directed_mesh_graph`` in ``connectivity.general``, which builds
   directed edges directly from the node positions with
   ``method="delaunay"``).
2. **Connectivity creation** -> ``nx.DiGraph`` with ``len`` and ``vdiff``
   edge attributes.

Currently only *nodes-only* input is supported: the input graph must not
contain any edges.  Support for user-provided edges (using them as the mesh
adjacency) is planned as a follow-up -- see the design discussion in
https://github.com/mllam/weather-model-graphs/issues/79.

The user input contract:

- an ``nx.Graph`` (or edge-less ``nx.DiGraph``) whose nodes carry:

  - ``pos``: ``np.ndarray`` of shape ``(2,)`` -- the node position, **in the
    same coordinate system as the grid coordinates** passed to
    ``create_all_graph_components``
  - ``type``: ``str``, must be ``"mesh"``
  - ``level``: ``int``, only for hierarchical meshes (lowest value = finest
    level); must be present on either all nodes or none

- or, for convenience, a bare ``np.ndarray`` of shape ``[N, 2]`` with node
  positions (a nodes-only, single-level mesh).
"""

from typing import List, Union

import networkx
import numpy as np
import scipy.spatial
from loguru import logger


def validate_prebuilt_mesh_nodes(
    mesh_graph: networkx.Graph, require_levels: bool = False
) -> None:
    """
    Validate that a user-provided mesh graph satisfies the prebuilt nodes-only
    input contract.

    Every node must have a ``pos`` attribute (``np.ndarray`` of shape ``(2,)``
    with finite values) and a ``type`` attribute equal to ``"mesh"``.  Node
    positions must be unique.  The graph must not contain any edges
    (user-provided edges are not yet supported, see issue #79).

    Parameters
    ----------
    mesh_graph : networkx.Graph
        User-provided graph to validate.
    require_levels : bool
        If True, additionally require an integer ``level`` attribute on every
        node, with at least two distinct level values (needed for hierarchical
        meshes).

    Raises
    ------
    ValueError
        If the graph is empty, a node attribute is missing or malformed,
        positions are duplicated, or level attributes are inconsistent.
    NotImplementedError
        If the graph contains edges (nodes+edges input is not yet supported).
    """
    if mesh_graph.number_of_nodes() == 0:
        raise ValueError(
            "mesh_layout='prebuilt' requires a mesh_graph with at least one node."
        )

    if mesh_graph.number_of_edges() > 0:
        raise NotImplementedError(
            "mesh_layout='prebuilt' currently only supports nodes-only input, "
            f"but the given mesh_graph has {mesh_graph.number_of_edges()} "
            "edge(s). Mesh connectivity is built in the connectivity step "
            "(method='delaunay' by default). Support for user-provided edges "
            "is planned -- see "
            "https://github.com/mllam/weather-model-graphs/issues/79."
        )

    n_with_level = 0
    positions = []
    for node, data in mesh_graph.nodes(data=True):
        if "pos" not in data:
            raise ValueError(
                f"Node {node!r} is missing the required 'pos' attribute. All "
                "nodes in a prebuilt mesh must have 'pos' as an np.ndarray of "
                "shape (2,)."
            )
        pos = np.asarray(data["pos"])
        if pos.shape != (2,):
            raise ValueError(
                f"Node {node!r} has 'pos' with shape {pos.shape}, expected "
                "(2,). All nodes in a prebuilt mesh must have 'pos' as an "
                "np.ndarray of shape (2,)."
            )
        if not np.all(np.isfinite(pos.astype(float))):
            raise ValueError(
                f"Node {node!r} has a non-finite 'pos' value ({pos}). Node "
                "positions must be finite numbers."
            )
        if data.get("type") != "mesh":
            raise ValueError(
                f"Node {node!r} has type={data.get('type')!r}, expected "
                "'mesh'. All nodes in a prebuilt mesh must have the 'type' "
                "attribute set to 'mesh' (the 'grid' node type is reserved "
                "for the grid nodes created from the `coords` argument)."
            )
        if "level" in data:
            n_with_level += 1
            if not isinstance(data["level"], (int, np.integer)):
                raise ValueError(
                    f"Node {node!r} has a non-integer 'level' attribute "
                    f"({data['level']!r}). Mesh levels must be integers "
                    "(lowest value = finest level)."
                )
        positions.append(pos.astype(float))

    n_nodes = mesh_graph.number_of_nodes()
    if 0 < n_with_level < n_nodes:
        raise ValueError(
            f"Only {n_with_level} of {n_nodes} nodes have a 'level' "
            "attribute. For a hierarchical prebuilt mesh every node must "
            "have a 'level'; for a flat mesh no node should have one."
        )

    positions_arr = np.stack(positions)
    n_unique = np.unique(positions_arr, axis=0).shape[0]
    if n_unique < n_nodes:
        raise ValueError(
            f"The mesh_graph contains duplicate node positions ({n_nodes} "
            f"nodes but only {n_unique} unique positions). Duplicate "
            "positions would produce zero-length mesh edges."
        )

    if require_levels:
        if n_with_level == 0:
            raise ValueError(
                "Hierarchical prebuilt meshes require an integer 'level' "
                "attribute on every node (lowest value = finest level), but "
                "no node has one."
            )
        levels = {int(data["level"]) for _, data in mesh_graph.nodes(data=True)}
        if len(levels) < 2:
            raise ValueError(
                "At least two distinct mesh levels are required for a "
                f"hierarchical prebuilt mesh, but only level(s) "
                f"{sorted(levels)} were found."
            )


def _as_node_cloud_graph(
    mesh_graph: Union[networkx.Graph, np.ndarray]
) -> networkx.Graph:
    """Normalize prebuilt-mesh input to an undirected node-cloud graph.

    Accepts either a graph (undirected or directed -- direction is the
    library's to assign, so an edge-less DiGraph is treated as its
    undirected node set) or a bare ``[N, 2]`` coordinate array.
    """
    if isinstance(mesh_graph, np.ndarray):
        xy = np.asarray(mesh_graph, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(
                "A prebuilt mesh given as an array must have shape "
                f"[N_mesh_nodes, 2], got {xy.shape}."
            )
        g = networkx.Graph()
        for i, pos in enumerate(xy):
            g.add_node(i, pos=pos, type="mesh")
        return g
    if isinstance(mesh_graph, networkx.Graph):  # includes DiGraph
        return networkx.Graph(mesh_graph)
    raise TypeError(
        "mesh_graph must be a networkx.Graph (or edge-less DiGraph) or an "
        f"np.ndarray of shape [N, 2], got {type(mesh_graph).__name__}."
    )


def _estimate_node_spacing(positions: np.ndarray) -> float:
    """Median nearest-neighbour distance -- the characteristic node spacing.

    Used to fill the ``dx``/``dy`` graph attributes that generated layouts
    derive from their lattice spacing (needed e.g. by the hierarchical
    connectivity step and relative-distance grid connection methods).
    """
    if positions.shape[0] < 2:
        return 0.0
    kdt = scipy.spatial.KDTree(positions)
    # k=2: the nearest neighbour that isn't the node itself
    dists, _ = kdt.query(positions, k=2)
    return float(np.median(dists[:, 1]))


def _node_cloud_primitive(g_cloud: networkx.Graph) -> networkx.Graph:
    """Build one edge-less mesh primitive from a validated node cloud.

    Node labels are replaced by ``(i,)`` integer tuples (insertion order) so
    they sort against the grid node labels and support the level-index
    prepending used by hierarchical connectivity.  Only the contract
    attributes (``pos``, ``type``) are carried over.
    """
    g = networkx.Graph()
    positions = []
    for i, (_, data) in enumerate(g_cloud.nodes(data=True)):
        pos = np.asarray(data["pos"], dtype=float)
        g.add_node((i,), pos=pos, type="mesh")
        positions.append(pos)
    spacing = _estimate_node_spacing(np.stack(positions))
    g.graph["dx"] = spacing
    g.graph["dy"] = spacing
    return g


def create_single_level_prebuilt_mesh_primitive(
    mesh_graph: Union[networkx.Graph, np.ndarray]
) -> networkx.Graph:
    """
    Create a single-level mesh primitive from user-provided mesh nodes.

    This is the coordinate creation step for ``mesh_layout="prebuilt"`` with
    flat connectivity.  The result is an *edge-less* undirected graph (a node
    cloud): mesh adjacency for a point cloud is built in the connectivity
    step (``method="delaunay"`` by default).

    Parameters
    ----------
    mesh_graph : networkx.Graph or np.ndarray
        User-provided mesh nodes (see the module docstring for the input
        contract).  If nodes carry a ``level`` attribute it is ignored (with
        a warning) -- use ``m2m_connectivity="hierarchical"`` to build a
        hierarchical mesh from the levels.

    Returns
    -------
    networkx.Graph
        Edge-less mesh primitive.  Node attributes: ``pos``
        (np.ndarray of shape ``(2,)``), ``type`` (``"mesh"``).  Graph
        attributes: ``dx``, ``dy`` (median nearest-neighbour node spacing).
    """
    g_cloud = _as_node_cloud_graph(mesh_graph)
    validate_prebuilt_mesh_nodes(g_cloud)
    if any("level" in d for _, d in g_cloud.nodes(data=True)):
        logger.warning(
            "The prebuilt mesh_graph nodes carry 'level' attributes but a "
            "single-level (flat) mesh was requested; the levels are ignored. "
            "Use m2m_connectivity='hierarchical' to build a hierarchical "
            "mesh from them."
        )
    return _node_cloud_primitive(g_cloud)


def create_multi_level_prebuilt_mesh_primitives(
    mesh_graph: Union[networkx.Graph, np.ndarray]
) -> List[networkx.Graph]:
    """
    Create per-level mesh primitives from user-provided mesh nodes with
    ``level`` attributes.

    This is the coordinate creation step for ``mesh_layout="prebuilt"`` with
    hierarchical connectivity.  Nodes are split by their integer ``level``
    attribute (lowest value = finest level) into one *edge-less* primitive
    per level; intra-level adjacency is built per level in the connectivity
    step (``intra_level=dict(method="delaunay")`` by default) and inter-level
    up/down edges by nearest-neighbour search (``inter_level``).

    Parameters
    ----------
    mesh_graph : networkx.Graph
        User-provided mesh nodes with ``pos``, ``type`` and ``level``
        attributes on every node (see the module docstring).

    Returns
    -------
    list[networkx.Graph]
        Edge-less mesh primitives, one per level, ordered finest first.
        Each carries the graph attributes ``level`` (0-based level index),
        ``dx`` and ``dy`` (median nearest-neighbour spacing of that level).
    """
    g_cloud = _as_node_cloud_graph(mesh_graph)
    validate_prebuilt_mesh_nodes(g_cloud, require_levels=True)

    user_levels = sorted({int(d["level"]) for _, d in g_cloud.nodes(data=True)})

    primitives = []
    for level_index, user_level in enumerate(user_levels):
        level_nodes = [
            n for n, d in g_cloud.nodes(data=True) if int(d["level"]) == user_level
        ]
        g_level = _node_cloud_primitive(g_cloud.subgraph(level_nodes))
        for node in g_level.nodes:
            g_level.nodes[node]["level"] = level_index
        g_level.graph["level"] = level_index
        primitives.append(g_level)

    return primitives
