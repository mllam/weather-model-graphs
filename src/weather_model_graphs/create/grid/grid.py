import networkx
import numpy as np

from ...networkx_utils import prepend_node_index


def create_grid_graph_nodes(xy, level_id=-1):
    """
    Create a networkx.Graph comprising only nodes for each (x,y)-point in the `xy` coordinate
    array (the attribute `pos` giving the (x,y)-coordinate value) and with
    node label (level_id, i, j)

    Each node contains the following attributes:
    - "pos": the (x,y)-coordinate value of the node
    - "level": the level id of the node (default to -1 for grid nodes)
    - "type": the type of the node (i.e. "grid" for grid nodes)

    Parameters
    ----------
    xy : np.ndarray [2, Ny, Nx]
        Grid point coordinates, with first dimension representing
        x and y coordinates respectively. Ny and Nx are the number
        of grid points in the y and x direction respectively
    level_id : int
        Level id of the nodes (default to -1)

    Returns
    -------
    networkx.Graph
        Graph representing the grid nodes
    """
    if len(xy.shape) != 3:
        raise NotImplementedError(
            "Mesh coordinates are assumed to lie on a regular grid so that "
            "the coordinates values are given with an array of shape [2, nx, ny]"
        )

    # grid nodes
    Ny, Nx = xy.shape[1:]

    G_grid = networkx.grid_2d_graph(Ny, Nx)
    G_grid.clear_edges()

    # vg features (only pos introduced here)
    for node in G_grid.nodes:
        # pos is in feature but here explicit for convenience
        G_grid.nodes[node]["pos"] = np.array([xy[0][node], xy[1][node]])
        G_grid.nodes[node]["level"] = level_id
        G_grid.nodes[node]["type"] = "grid"

    # add `level_id` (default to 1000) to node key to separate grid nodes (1000,i,j) from mesh nodes
    # (i,j) and impose sorting order such that vm are the first nodes
    G_grid = prepend_node_index(G_grid, level_id)

    return G_grid
