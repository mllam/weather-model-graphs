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
    xy : np.ndarray [2, M, N]
        Grid point coordinates, with first dimension representing
        x and y coordinates respectively. M and N are the number
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

    # Node name and `pos` attribute takes form (x, y)
    for node in g.nodes:
        node_xi, node_yi = node  # Extract x and y index from node to index mx
        g.nodes[node]["pos"] = np.array(
            [mg[0][node_yi, node_xi], mg[1][node_yi, node_xi]]
        )
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


def create_multirange_2d_mesh_graphs(
    max_num_levels, xy, grid_refinement_factor=3, level_refinement_factor=3
):
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
        Grid point coordinates, shaped [2, M, N]
    refinement_factor : int
        Degree of refinement between successive mesh graphs, the number of nodes
        grows by approximately refinement_factor**2 between successive
        mesh graphs.

    Returns
    -------
    G_all_levels : list of networkx.Graph
        List of networkx graphs for each level representing the connectivity
        of the mesh within each level
    """
    # Compute the size (grid nodes) along x and y direction of area
    # to cover with graph
    coord_extent = np.array((xy.shape[2], xy.shape[1]))

    # Find the number of mesh levels possible in x- and y-direction,
    # and the number of leaf nodes that would correspond to
    # max_coord/(grid_refinement_factor*level_refinement_factor^mesh_levels) = 1
    max_mesh_levels_float = (
        np.log(coord_extent) - np.log(grid_refinement_factor)
    ) / np.log(level_refinement_factor)

    # Need to add a small epsilon before flooring to int, due to numerical
    # issues with the computation above
    eps = 1e-8
    max_mesh_levels = (max_mesh_levels_float + eps).astype(int)  # (2,)
    nleaf = grid_refinement_factor * (
        level_refinement_factor**max_mesh_levels
    )  # leaves at the bottom in each direction, if using max_mesh_levels

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
            nleaf / (grid_refinement_factor * (level_refinement_factor**lev))
        ).astype(int)
        g = create_single_level_2d_mesh_graph(xy, nodes_x, nodes_y)
        # Add level information to nodes, edges and full graph
        for node in g.nodes:
            g.nodes[node]["level"] = lev
        for edge in g.edges:
            g.edges[edge]["level"] = lev
        g.graph["level"] = lev
        G_all_levels.append(g)

    return G_all_levels
