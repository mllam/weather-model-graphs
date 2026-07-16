import networkx
import numpy as np

from ....spatial import SpatialCoordinateValuesSelector


def create_directed_mesh_graph(
    G_undirected: networkx.Graph,
    pattern: str = "8-star",
    distance_metric: str = "euclidean",
) -> networkx.DiGraph:
    """
    Convert an undirected mesh primitive graph with spatial adjacency edges to a
    directed mesh graph (nx.DiGraph) based on the specified connectivity pattern.

    This is the second step in the two-step mesh creation process:
    1. Coordinate creation (create_single_level_2d_mesh_primitive) -> nx.Graph
    2. Connectivity creation (this function) -> nx.DiGraph

    The ``pattern`` argument defines the spatial neighbourhood connectivity:
    - ``"4-star"``: only cardinal directions (horizontal and vertical neighbours)
    - ``"8-star"``: cardinal directions plus diagonals (all 8 surrounding neighbours)

    Parameters
    ----------
    G_undirected : networkx.Graph
        Undirected mesh primitive graph. Expected node attributes:
        - ``"pos"``: np.ndarray of shape [2,], spatial coordinates.
        Expected edge attributes:
        - ``"adjacency_type"``: str, either ``"cardinal"`` or ``"diagonal"``.
        Additional edge attributes (e.g. ``"level"``) are preserved in the
        output directed graph.
    pattern : str
        Connectivity pattern. Options:
        - ``"4-star"``: only cardinal edges (horizontal/vertical neighbours)
        - ``"8-star"``: all edges (cardinal + diagonal neighbours)
    distance_metric : str
        Distance metric for edge length computation. Options:
        - ``"euclidean"``: standard Cartesian distance (default)
        - ``"haversine"``: great-circle distance for lon/lat in degrees

    Returns
    -------
    networkx.DiGraph
        Directed graph with bidirectional edges, each having ``"len"`` and
        ``"vdiff"`` attributes. All original edge attributes from the
        primitive graph are preserved.
    """
    if pattern == "4-star":
        # Filter to only cardinal edges, preserving edge data
        edges_to_use = [
            (u, v, d)
            for u, v, d in G_undirected.edges(data=True)
            if d.get("adjacency_type") == "cardinal"
        ]
    elif pattern == "8-star":
        # Use all edges with their data
        edges_to_use = list(G_undirected.edges(data=True))
    else:
        raise ValueError(
            f"Unknown connectivity pattern: '{pattern}'. "
            "Choose '4-star' or '8-star'."
        )

    # Create filtered undirected graph with only selected edges (preserving attrs)
    g_filtered = networkx.Graph()
    g_filtered.add_nodes_from(G_undirected.nodes(data=True))
    g_filtered.add_edges_from(edges_to_use)

    # Convert to directed graph (creates edges in both directions)
    dg = networkx.DiGraph(g_filtered)

    pos = {n: G_undirected.nodes[n]["pos"] for n in G_undirected.nodes}
    all_positions = np.array(list(pos.values()))
    selector = SpatialCoordinateValuesSelector(distance_metric, all_positions)

    for u, v in g_filtered.edges():
        d = selector.distance_between(pos[u], pos[v])
        dg.edges[u, v]["len"] = d
        dg.edges[u, v]["vdiff"] = pos[u] - pos[v]
        # Ensure reverse edge exists and has attributes
        dg.edges[v, u]["len"] = d
        dg.edges[v, u]["vdiff"] = pos[v] - pos[u]

    # Preserve graph-level attributes (dx, dy, level, etc.)
    dg.graph.update(G_undirected.graph)

    return dg
