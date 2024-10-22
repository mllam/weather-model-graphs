import networkx

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
    xy : np.ndarray [N_grid_points, 2]
        Grid point coordinates, with first column representing
        x coordinates and second column y coordinates. N_grid_points is the
        total number of grid points.
    level_id : int
        Level id of the nodes (default to -1)

    Returns
    -------
    networkx.Graph
        Graph representing the grid nodes
    """
    assert (
        len(xy.shape) == 2 and xy.shape[1] == 2
    ), "Grid node coordinates should be given as an array of shape [num_grid_nodes, 2]."

    # Create graph
    G_grid = networkx.Graph()

    # Add grid nodes
    # vg features (only pos introduced here)
    for i, pos in enumerate(xy):
        # pos is in feature but here explicit for convenience
        G_grid.add_node((i,), pos=pos, level=level_id, type="grid")

    # add `level_id` (default to -1) to node key to separate grid nodes (-1,i) from mesh nodes
    # (i,) and impose sorting order such that vm are the first nodes
    G_grid = prepend_node_index(G_grid, level_id)

    return G_grid
