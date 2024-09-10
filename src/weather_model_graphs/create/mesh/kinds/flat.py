import networkx
import numpy as np

from ....networkx_utils import prepend_node_index
from .. import mesh as mesh_graph


def create_flat_multiscale_mesh_graph(
    xy, grid_refinement_factor: float, level_refinement_factor: int, max_num_levels: int
):
    """
    Create flat mesh graph by merging the single-level mesh
    graphs across all levels in `G_all_levels`.

    Parameters
    ----------
    xy : np.ndarray [2, M, N]
        Grid point coordinates, with first dimension representing
        x and y coordinates respectively. M and N are the number
        of grid points in the y and x direction respectively
    grid_refinement_factor: float
        Refinement factor between grid points and bottom level of mesh hierarchy
    level_refinement_factor: int
        Refinement factor between grid points and bottom level of mesh hierarchy
        NOTE: Must be an odd integer >1 to create proper multiscale graph
    max_num_levels : int
        Maximum number of levels in the multi-scale graph
    Returns
    -------
    m2m_graphs : list
        List of PyTorch geometric graphs for each level
    G_bottom_mesh : networkx.Graph
        Graph representing the bottom mesh level
    all_mesh_nodes : networkx.NodeView
        All mesh nodes
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
        grid_refinement_factor=grid_refinement_factor,
        level_refinement_factor=level_refinement_factor,
    )

    # combine all levels to one graph
    G_tot = G_all_levels[0]
    # First node at level l+1 share position with node (offset, offset) at level l
    level_offset = level_refinement_factor // 2
    for lev in range(1, len(G_all_levels)):
        nodes = list(G_all_levels[lev - 1].nodes)
        # Last nodes always has pos (nx-1, ny-1)
        num_nodes_x = nodes[-1][0] + 1
        num_nodes_y = nodes[-1][1] + 1
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

    # Relabel mesh nodes to start with 0
    G_tot = prepend_node_index(G_tot, 0)

    # add dx and dy to graph
    G_tot.graph["dx"] = {i: g.graph["dx"] for i, g in enumerate(G_all_levels)}
    G_tot.graph["dy"] = {i: g.graph["dy"] for i, g in enumerate(G_all_levels)}

    return G_tot
