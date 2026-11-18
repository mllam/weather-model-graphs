import networkx
import numpy as np
import scipy.spatial


def create_directed_mesh_graph(
    G_undirected: networkx.Graph, pattern: str = None, method: str = None
) -> networkx.DiGraph:
    """
    Convert an undirected mesh primitive graph to a directed mesh graph
    (nx.DiGraph).

    This is the second step in the two-step mesh creation process:
    1. Coordinate creation (mesh layout module) -> nx.Graph
    2. Connectivity creation (this function) -> nx.DiGraph

    Two kinds of mesh primitive are supported:

    - **Primitives with adjacency edges** (the generated layouts,
      ``rectilinear``/``triangular``): the edges carry an ``adjacency_type``
      attribute and the optional ``pattern`` argument selects a subset of
      them. When no ``pattern`` is given, every edge the layout produced is
      used. When a ``pattern`` is given it must match the adjacency types
      present in the primitive, otherwise a ``ValueError`` is raised listing
      what is available (rather than silently producing an empty mesh).
    - **Edge-less primitives** (the ``prebuilt`` layout's node clouds): there
      is no adjacency to select from, so the directed edges are built
      directly from the node positions using ``method`` (currently only
      ``"delaunay"``, the default: Delaunay triangulation of the node
      positions). ``pattern`` does not apply to node clouds.

    Parameters
    ----------
    G_undirected : networkx.Graph
        Undirected mesh primitive graph. Expected node attributes:
        - ``"pos"``: np.ndarray of shape [2,], spatial coordinates.
        Expected edge attributes (only when the primitive has edges):
        - ``"adjacency_type"``: str, e.g. ``"cardinal"`` or ``"diagonal"``.
        Additional edge attributes (e.g. ``"level"``) are preserved in the
        output directed graph.
    pattern : str, optional
        Connectivity pattern for primitives with adjacency edges. Options:
        - ``None`` (default): use every edge the layout produced
        - ``"4-star"``: only cardinal edges (horizontal/vertical neighbours)
        - ``"8-star"``: all edges (cardinal + diagonal neighbours)
    method : str, optional
        Edge construction method for edge-less primitives (node clouds).
        Options:
        - ``None`` (default): resolves to ``"delaunay"`` for node clouds
        - ``"delaunay"``: Delaunay triangulation of the node positions

    Returns
    -------
    networkx.DiGraph
        Directed graph with bidirectional edges, each having ``"len"`` and
        ``"vdiff"`` attributes. All original node, edge and graph attributes
        from the primitive graph are preserved.

    Raises
    ------
    ValueError
        If ``pattern`` does not match the adjacency types present in the
        primitive, if ``pattern`` is given for an edge-less primitive, if
        ``method`` is given for a primitive that already has adjacency
        edges, or if the node positions are degenerate (e.g. all collinear)
        so that no triangulation exists.
    NotImplementedError
        If an unknown ``method`` is requested.
    """
    if G_undirected.number_of_edges() == 0 and G_undirected.number_of_nodes() > 0:
        return _create_directed_mesh_graph_from_node_cloud(
            G_undirected, pattern=pattern, method=method
        )

    if method is not None:
        raise ValueError(
            f"method='{method}' was given, but the mesh primitive already "
            "has adjacency edges (created by the mesh layout). The 'method' "
            "argument only applies to edge-less primitives (node clouds "
            "from mesh_layout='prebuilt')."
        )

    if pattern is None:
        # Use every edge the layout produced
        edges_to_use = list(G_undirected.edges(data=True))
    elif pattern == "4-star":
        # Filter to only cardinal edges, preserving edge data
        edges_to_use = [
            (u, v, d)
            for u, v, d in G_undirected.edges(data=True)
            if d.get("adjacency_type") == "cardinal"
        ]
        if len(edges_to_use) == 0:
            available = sorted(
                {
                    str(d.get("adjacency_type"))
                    for _, _, d in G_undirected.edges(data=True)
                }
            )
            raise ValueError(
                "pattern='4-star' selects edges with "
                "adjacency_type='cardinal', but the mesh primitive has no "
                f"such edges (available adjacency types: {available}). "
                "Omit 'pattern' to use every edge the layout produced."
            )
    elif pattern == "8-star":
        # Use all edges with their data
        edges_to_use = list(G_undirected.edges(data=True))
    else:
        raise ValueError(
            f"Unknown connectivity pattern: '{pattern}'. "
            "Choose '4-star', '8-star', or omit 'pattern' to use every "
            "edge the layout produced."
        )

    # Create filtered undirected graph with only selected edges (preserving attrs)
    g_filtered = networkx.Graph()
    g_filtered.add_nodes_from(G_undirected.nodes(data=True))
    g_filtered.add_edges_from(edges_to_use)

    # Convert to directed graph (creates edges in both directions)
    dg = networkx.DiGraph(g_filtered)
    for u, v in g_filtered.edges():
        d = np.sqrt(
            np.sum((G_undirected.nodes[u]["pos"] - G_undirected.nodes[v]["pos"]) ** 2)
        )
        dg.edges[u, v]["len"] = d
        dg.edges[u, v]["vdiff"] = (
            G_undirected.nodes[u]["pos"] - G_undirected.nodes[v]["pos"]
        )
        # Ensure reverse edge exists and has attributes
        dg.edges[v, u]["len"] = d
        dg.edges[v, u]["vdiff"] = (
            G_undirected.nodes[v]["pos"] - G_undirected.nodes[u]["pos"]
        )

    # Preserve graph-level attributes (dx, dy, level, etc.)
    dg.graph.update(G_undirected.graph)

    return dg


def _create_directed_mesh_graph_from_node_cloud(
    G_nodes: networkx.Graph, pattern: str = None, method: str = None
) -> networkx.DiGraph:
    """Build a directed mesh graph directly from an edge-less node cloud.

    The directed edges are constructed straight from the node positions
    (no intermediate undirected adjacency graph is built).
    """
    # A single-node primitive has no edges under any semantics; don't reject
    # a 'pattern' that a generated-layout code path may have passed along.
    if pattern is not None and G_nodes.number_of_nodes() > 1:
        raise ValueError(
            f"pattern='{pattern}' was given, but the mesh primitive has no "
            "adjacency edges to select from (it is a node cloud from "
            "mesh_layout='prebuilt'). Use method='delaunay' (the default) "
            "to control how edges are constructed from the node positions."
        )
    if method is None:
        method = "delaunay"
    if method != "delaunay":
        raise NotImplementedError(
            f"method='{method}' is not implemented for building mesh edges "
            "from node positions. Currently supported: 'delaunay'."
        )

    dg = networkx.DiGraph()
    dg.add_nodes_from(G_nodes.nodes(data=True))
    dg.graph.update(G_nodes.graph)

    nodes = list(G_nodes.nodes)
    positions = np.array(
        [np.asarray(G_nodes.nodes[n]["pos"], dtype=float) for n in nodes]
    )
    n_nodes = len(nodes)

    # Delaunay triangulation needs >= 3 non-collinear points; smaller node
    # clouds get the only sensible connectivity directly.
    if n_nodes == 1:
        return dg
    if n_nodes == 2:
        undirected_pairs = {(0, 1)}
    else:
        try:
            triangulation = scipy.spatial.Delaunay(positions)
        except scipy.spatial.QhullError as exc:
            raise ValueError(
                "Delaunay triangulation of the prebuilt mesh nodes failed "
                f"({n_nodes} nodes). This typically means the node positions "
                "are degenerate (e.g. all collinear). Provide at least 3 "
                "non-collinear mesh node positions."
            ) from exc
        undirected_pairs = set()
        for simplex in triangulation.simplices:
            for i in range(3):
                a, b = int(simplex[i]), int(simplex[(i + 1) % 3])
                undirected_pairs.add((min(a, b), max(a, b)))

    for ia, ib in sorted(undirected_pairs):
        u, v = nodes[ia], nodes[ib]
        vdiff = positions[ia] - positions[ib]
        d = float(np.sqrt(np.sum(vdiff**2)))
        dg.add_edge(u, v, len=d, vdiff=vdiff)
        dg.add_edge(v, u, len=d, vdiff=-vdiff)

    return dg
