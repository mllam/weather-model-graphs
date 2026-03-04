import networkx
import numpy as np
from loguru import logger


def create_single_level_2d_mesh_primitive(xy: np.ndarray, nx: int, ny: int):
    """
    Create an undirected mesh primitive graph (nx.Graph) with node positions
    and spatial adjacency edges, representing the coordinate creation step.

    A mesh primitive is an undirected graph that encodes all potential
    neighbourhood connectivity edges. It serves as a blueprint from which
    directed connectivity graphs can later be built by selecting a subset
    of edges (e.g. 4-star or 8-star pattern).

    This produces a graph where:
    - Nodes have a ``"pos"`` attribute (np.ndarray of shape [2,] with x and y
      coordinates) and a ``"type"`` attribute (str, always ``"mesh"``).
    - Edges have an ``"adjacency_type"`` attribute (str): ``"cardinal"`` for
      horizontal/vertical neighbours (4-star connectivity) or ``"diagonal"``
      for diagonal neighbours (additional edges in 8-star connectivity).

    This is the first step in the two-step mesh creation process:
    1. Coordinate creation (this function) -> nx.Graph with spatial adjacency
    2. Connectivity creation (create_directed_mesh_graph) -> nx.DiGraph

    Parameters
    ----------
    xy : np.ndarray
        Grid point coordinates, shaped [N_grid_points, 2], with first column
        representing x coordinates and second column y coordinates.
    nx : int
        Number of nodes in x direction
    ny : int
        Number of nodes in y direction

    Returns
    -------
    networkx.Graph
        Undirected mesh primitive graph with node positions and annotated
        spatial adjacency edges.
    """
    xm, xM = np.amin(xy[:, 0]), np.amax(xy[:, 0])
    ym, yM = np.amin(xy[:, 1]), np.amax(xy[:, 1])

    # avoid nodes on border
    dx = (xM - xm) / nx
    dy = (yM - ym) / ny
    lx = np.linspace(xm + dx / 2, xM - dx / 2, nx)
    ly = np.linspace(ym + dy / 2, yM - dy / 2, ny)

    mg = np.meshgrid(lx, ly)
    g = networkx.grid_2d_graph(len(lx), len(ly))

    # Node name and `pos` attribute takes form (x, y)
    for node in g.nodes:
        node_xi, node_yi = node  # Extract x and y index from node to index mx
        g.nodes[node]["pos"] = np.array(
            [mg[0][node_yi, node_xi], mg[1][node_yi, node_xi]]
        )
        g.nodes[node]["type"] = "mesh"

    # Mark existing grid_2d_graph edges as cardinal (4-star adjacency)
    for u, v in g.edges():
        g.edges[u, v]["adjacency_type"] = "cardinal"

    # Add diagonal edges (8-star adjacency)
    diagonal_edges = [
        ((x, y), (x + 1, y + 1)) for x in range(nx - 1) for y in range(ny - 1)
    ] + [((x + 1, y), (x, y + 1)) for x in range(nx - 1) for y in range(ny - 1)]
    g.add_edges_from(diagonal_edges)
    for u, v in diagonal_edges:
        g.edges[u, v]["adjacency_type"] = "diagonal"

    g.graph["dx"] = dx
    g.graph["dy"] = dy

    return g


def create_directed_mesh_graph(
    G_undirected: networkx.Graph, pattern: str = "8-star"
):
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
    for u, v in g_filtered.edges():
        d = np.sqrt(
            np.sum(
                (G_undirected.nodes[u]["pos"] - G_undirected.nodes[v]["pos"]) ** 2
            )
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


def create_single_level_2d_mesh_graph(xy, nx, ny):
    """
    Create directed graph with nx * ny nodes representing a 2D grid with
    positions spanning the range of xy coordinate values (first dimension
    is assumed to be x and y coordinate values respectively). Each nodes is
    connected to its eight nearest neighbours, both horizontally, vertically
    and diagonally as directed edges (which means that the graph contains two
    edges between each pair of connected nodes).

    The nodes contain a "pos" attribute with the x and y
    coordinates of the node, and an "type" attribute with the
    type of the node (i.e. "mesh" for mesh nodes).

    The edges contain a "len" attribute with the length of the edge
    and a "vdiff" attribute with the vector difference between the
    nodes.

    Internally, this uses the two-step process:
    1. create_single_level_2d_mesh_primitive (coordinate creation)
    2. create_directed_mesh_graph (connectivity creation, pattern="8-star")

    Parameters
    ----------
    xy : np.ndarray [N_grid_points, 2]
        Grid point coordinates, with first column representing
        x coordinates and second column y coordinates. N_grid_points is the
        total number of grid points.
    nx : int
        Number of nodes in x direction
    ny : int
        Number of nodes in y direction

    Returns
    -------
    networkx.DiGraph
        Graph representing the 2D grid
    """
    G_coords = create_single_level_2d_mesh_primitive(xy, nx, ny)
    return create_directed_mesh_graph(G_coords, pattern="8-star")


def create_multirange_2d_mesh_primitives(
    max_num_levels, xy, mesh_node_spacing=3, interlevel_refinement_factor=3
):
    """
    Create a list of undirected mesh primitive graphs (nx.Graph) representing
    different levels of mesh resolution spanning the spatial domain of the
    xy coordinates.

    This is the coordinate creation step for multi-level and hierarchical mesh
    graphs. Each returned graph contains nodes with spatial positions and edges
    annotated with adjacency type (``"cardinal"`` or ``"diagonal"``).

    The graphs can be consumed by connectivity creation functions to produce
    directed mesh graphs for flat_multiscale or hierarchical architectures.

    Parameters
    ----------
    max_num_levels : int
        Number of edge-distance levels in mesh graph
    xy : np.ndarray
        Grid point coordinates, shaped [N_grid_points, 2]
    mesh_node_spacing : float
        Distance (in x- and y-direction) between created mesh nodes,
        in coordinate system of xy
    interlevel_refinement_factor : float
        Refinement factor between grid points and bottom level of mesh hierarchy

    Returns
    -------
    G_all_levels : list of networkx.Graph
        List of undirected mesh primitive graphs for each level, each with
        node positions and annotated spatial adjacency edges.
        Each graph has ``"level"`` and ``"interlevel_refinement_factor"``
        graph attributes.
    """
    # Compute the size along x and y direction of area to cover with graph
    # This is measured in the Cartesian coordinates of xy
    coord_extent = np.ptp(xy, axis=0)
    # Number of nodes that would fit on bottom level of hierarchy,
    # in both directions
    max_nodes_bottom = (coord_extent / mesh_node_spacing).astype(int)

    # Find the number of mesh levels possible in x- and y-direction,
    # and the number of leaf nodes that would correspond to
    # max_nodes_bottom/(interlevel_refinement_factor^mesh_levels) = 1
    max_mesh_levels_float = np.log(max_nodes_bottom) / np.log(
        interlevel_refinement_factor
    )

    max_mesh_levels = max_mesh_levels_float.astype(int)  # (2,)
    nleaf = interlevel_refinement_factor**max_mesh_levels
    # leaves at the bottom in each direction, if using max_mesh_levels

    # As we can not instantiate different number of mesh levels in each
    # direction, create mesh levels corresponding to the minimum of the two
    mesh_levels_to_create = max_mesh_levels.min()

    if max_num_levels:
        # Limit the levels in mesh graph
        mesh_levels_to_create = min(mesh_levels_to_create, max_num_levels)

    logger.debug(f"mesh_levels: {mesh_levels_to_create}, nleaf: {nleaf}")

    # multi resolution tree levels
    G_all_levels = []
    for lev in range(mesh_levels_to_create):  # 0-index mesh levels
        # Compute number of nodes on level separate for each direction
        nodes_x, nodes_y = (
            nleaf / (interlevel_refinement_factor**lev)
        ).astype(int)
        g = create_single_level_2d_mesh_primitive(xy, nodes_x, nodes_y)
        # Add level information to nodes, edges and full graph
        for node in g.nodes:
            g.nodes[node]["level"] = lev
        for edge in g.edges:
            g.edges[edge]["level"] = lev
        g.graph["level"] = lev
        # Store refinement factor so connectivity step can use it
        g.graph["interlevel_refinement_factor"] = interlevel_refinement_factor
        G_all_levels.append(g)

    return G_all_levels


def create_multirange_2d_mesh_graphs(
    max_num_levels, xy, mesh_node_distance=3, level_refinement_factor=3
):
    """
    Create a list of 2D grid mesh graphs representing different levels of edge-length
    scales spanning the spatial domain of the xy coordinates.
    This list of graphs can then later be for example a) flattened into single graph
    containing multiple ranges of connections or b) combined into a hierarchical graph.

    Each graph in the list contains a "level" attribute with the level index of the graph.

    Internally uses the two-step process:
    1. create_multirange_2d_mesh_primitives (coordinate creation)
    2. create_directed_mesh_graph (connectivity creation, pattern="8-star")

    Parameters
    ----------
    max_num_levels : int
        Number of edge-distance levels in mesh graph
    xy : np.ndarray
        Grid point coordinates, shaped [N_grid_points, 2]
    mesh_node_distance: float
        Distance (in x- and y-direction) between created mesh nodes,
        in coordinate system of xy
    level_refinement_factor: float
        Refinement factor between grid points and bottom level of mesh hierarchy

    Returns
    -------
    G_all_levels : list of networkx.Graph
        List of networkx graphs for each level representing the connectivity
        of the mesh within each level
    """
    G_coords_list = create_multirange_2d_mesh_primitives(
        max_num_levels=max_num_levels,
        xy=xy,
        mesh_node_spacing=mesh_node_distance,
        interlevel_refinement_factor=level_refinement_factor,
    )

    G_all_levels = []
    for g_coords in G_coords_list:
        g_directed = create_directed_mesh_graph(g_coords, pattern="8-star")
        G_all_levels.append(g_directed)

    return G_all_levels
