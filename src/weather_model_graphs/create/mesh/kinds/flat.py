import networkx
import numpy as np

from ....networkx_utils import prepend_node_index
from .. import mesh as mesh_graph


def create_flat_multiscale_mesh_graph(
    xy, mesh_node_distance: float, level_refinement_factor: int, max_num_levels: int
):
    """
    Create flat mesh graph by merging the single-level mesh
    graphs across all levels in `G_all_levels`.

    Parameters
    ----------
    xy : np.ndarray [N_grid_points, 2]
        Grid point coordinates, with first column representing
        x coordinates and second column y coordinates. N_grid_points is the
        total number of grid points.
    mesh_node_distance: float
        Distance (in x- and y-direction) between created mesh nodes,
        in coordinate system of xy
    level_refinement_factor: int
        Refinement factor between grid points and bottom level of mesh hierarchy
        NOTE: Must be an odd integer >1 to create proper multiscale graph
    max_num_levels : int
        Maximum number of levels in the multi-scale graph
    Returns
    -------
    G_tot : networkx.Graph
        The merged mesh graph
    """
    # Check that level_refinement_factor is an odd integer
    if (
        int(level_refinement_factor) != level_refinement_factor
        or level_refinement_factor % 2 != 1
    ):
        raise ValueError(
            "The `level_refinement_factor` must be an odd integer. "
            f"Given value: {level_refinement_factor}."
        )

    G_all_levels: list[networkx.DiGraph] = mesh_graph.create_multirange_2d_mesh_graphs(
        max_num_levels=max_num_levels,
        xy=xy,
        mesh_node_distance=mesh_node_distance,
        level_refinement_factor=level_refinement_factor,
    )

    # combine all levels to one graph
    G_tot = G_all_levels[0]
    # First node at level l+1 share position with node (offset, offset) at level l
    level_offset = level_refinement_factor // 2

    first_level_nodes = list(G_all_levels[0].nodes)
    # Last nodes in first layer has pos (nx-1, ny-1)
    num_nodes_x = first_level_nodes[-1][0] + 1
    num_nodes_y = first_level_nodes[-1][1] + 1

    for lev in range(1, len(G_all_levels)):
        nodes = list(G_all_levels[lev - 1].nodes)
        ij = (
            np.array(nodes)
            .reshape((num_nodes_x, num_nodes_y, 2))[
                level_offset::level_refinement_factor,
                level_offset::level_refinement_factor,
                :,
            ]
            .reshape(int(num_nodes_x * num_nodes_y / (level_refinement_factor**2)), 2)
        )
        ij = [tuple(x) for x in ij]
        G_all_levels[lev] = networkx.relabel_nodes(
            G_all_levels[lev], dict(zip(G_all_levels[lev].nodes, ij))
        )
        G_tot = networkx.compose(G_tot, G_all_levels[lev])

        # Update number of nodes in x- and y-direction for next iteraion
        num_nodes_x //= level_refinement_factor
        num_nodes_y //= level_refinement_factor

    # Relabel mesh nodes to start with 0
    G_tot = prepend_node_index(G_tot, 0)

    # add dx and dy to graph
    G_tot.graph["dx"] = {i: g.graph["dx"] for i, g in enumerate(G_all_levels)}
    G_tot.graph["dy"] = {i: g.graph["dy"] for i, g in enumerate(G_all_levels)}

    return G_tot


def create_flat_singlescale_mesh_graph(xy, mesh_node_distance: float):
    """
    Create flat mesh graph of single level

    Parameters
    ----------
    xy : np.ndarray [N_grid_points, 2]
        Grid point coordinates, with first column representing
        x coordinates and second column y coordinates. N_grid_points is the
        total number of grid points.
    mesh_node_distance: float
        Distance (in x- and y-direction) between created mesh nodes,
        in coordinate system of xy
    Returns
    -------
    G_flat : networkx.Graph
        The flat mesh graph
    """
    # Compute number of mesh nodes in x and y dimensions
    range_x, range_y = np.ptp(xy, axis=0)
    nx = int(range_x / mesh_node_distance)
    ny = int(range_y / mesh_node_distance)

    if nx == 0 or ny == 0:
        raise ValueError(
            "The given `mesh_node_distance` is too large for the provided coordinates. "
            f"Got mesh_node_distance={mesh_node_distance}, but the x-range is {range_x} "
            f"and y-range is {range_y}. Maybe you want to decrease the `mesh_node_distance`"
            " so that the mesh nodes are spaced closer together?"
        )

    return mesh_graph.create_single_level_2d_mesh_graph(xy=xy, nx=nx, ny=ny)
