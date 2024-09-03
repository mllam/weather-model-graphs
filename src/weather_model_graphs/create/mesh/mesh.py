import networkx
import numpy as np
from loguru import logger


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

    Parameters
    ----------
    xy : np.ndarray [2, N, M]
        Grid point coordinates, with first dimension representing
        x and y coordinates respectively. N and M are the number
        of grid points in the y and x direction respectively
    nx : int
        Number of nodes in x direction
    ny : int
        Number of nodes in y direction

    Returns
    -------
    networkx.DiGraph
        Graph representing the 2D grid
    """
    xm, xM = np.amin(xy[0][0, :]), np.amax(xy[0][0, :])
    ym, yM = np.amin(xy[1][:, 0]), np.amax(xy[1][:, 0])

    # avoid nodes on border
    dx = (xM - xm) / nx
    dy = (yM - ym) / ny
    lx = np.linspace(xm + dx / 2, xM - dx / 2, nx)
    ly = np.linspace(ym + dy / 2, yM - dy / 2, ny)

    mg = np.meshgrid(lx, ly)
    g = networkx.grid_2d_graph(len(lx), len(ly))

    for node in g.nodes:
        node_xi, node_yi = node # Extract x and y index from node to index mx
        g.nodes[node]["pos"] = np.array([
            mg[0][node_yi, node_xi],
            mg[1][node_yi, node_xi]
        ])
        g.nodes[node]["type"] = "mesh"

    # add diagonal edges
    g.add_edges_from(
        [((x, y), (x + 1, y + 1)) for x in range(nx - 1) for y in range(ny - 1)]
        + [((x + 1, y), (x, y + 1)) for x in range(nx - 1) for y in range(ny - 1)]
    )

    # turn into directed graph
    dg = networkx.DiGraph(g)
    for u, v in g.edges():
        d = np.sqrt(np.sum((g.nodes[u]["pos"] - g.nodes[v]["pos"]) ** 2))
        dg.edges[u, v]["len"] = d
        dg.edges[u, v]["vdiff"] = g.nodes[u]["pos"] - g.nodes[v]["pos"]
        dg.add_edge(v, u)
        dg.edges[v, u]["len"] = d
        dg.edges[v, u]["vdiff"] = g.nodes[v]["pos"] - g.nodes[u]["pos"]

    dg.graph["dx"] = dx
    dg.graph["dy"] = dy

    return dg


def create_multirange_2d_mesh_graphs(max_num_levels, xy, refinement_factor=3):
    """
    Create a list of 2D grid mesh graphs representing different levels of edge-length
    scales spanning the spatial domain of the xy coordinates.
    This list of graphs can then later be for example a) flattened into single graph
    containing multiple ranges of connections or b) combined into a hierarchical graph.

    Each graph in the list contains a "level" attribute with the level index of the graph.

    Parameters
    ----------
    max_num_levels : int
        Number of edge-distance levels in mesh graph
    xy : np.ndarray
        Grid point coordinates
    refinement_factor : int
        Degree of refinement between successive mesh graphs, the number of nodes
        grows by refinement_factor**2 between successive mesh graphs

    Returns
    -------
    G_all_levels : list of networkx.Graph
        List of networkx graphs for each level representing the connectivity
        of the mesh within each level
    """
    nlev = int(np.log(max(xy.shape)) / np.log(refinement_factor))
    nleaf = refinement_factor**nlev  # leaves at the bottom = nleaf**2

    mesh_levels = nlev - 1
    if max_num_levels:
        # Limit the levels in mesh graph
        mesh_levels = min(mesh_levels, max_num_levels)

    logger.debug(f"nlev: {nlev}, nleaf: {nleaf}, mesh_levels: {mesh_levels}")

    # multi resolution tree levels
    G_all_levels = []
    for lev in range(mesh_levels): # 0-index mesh levels
        n = int(nleaf / (refinement_factor**(lev + 1)))
        g = create_single_level_2d_mesh_graph(xy, n, n)
        # Add level information to nodes, edges and full graph
        for node in g.nodes:
            g.nodes[node]["level"] = lev
        for edge in g.edges:
            g.edges[edge]["level"] = lev
        g.graph["level"] = lev
        G_all_levels.append(g)

    return G_all_levels
