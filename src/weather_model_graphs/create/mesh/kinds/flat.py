import networkx
import numpy as np

from ....networkx_utils import prepend_node_index
from .. import mesh as mesh_graph


def create_flat_multiscale_mesh_graph(xy, max_num_levels: int):
    """
    Create flat mesh graph by merging the single-level mesh
    graphs across all levels in `G_all_levels`.

    Parameters
    ----------
    xy : np.ndarray [2, N, M]
        Grid point coordinates, with first dimension representing
        x and y coordinates respectively. N and M are the number
        of grid points in the x and y direction respectively
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
    # 3 is the only refinement factor possible for multiscale graph
    refinement_factor = 3
    G_all_levels: list[networkx.DiGraph] = mesh_graph.create_multirange_2d_mesh_graphs(
        max_num_levels=max_num_levels,
        xy=xy,
        refinement_factor=refinement_factor,
    )

    # combine all levels to one graph
    G_tot = G_all_levels[0]
    for lev in range(1, len(G_all_levels)):
        nodes = list(G_all_levels[lev - 1].nodes)
        n = int(np.sqrt(len(nodes)))
        ij = (
            np.array(nodes)
            .reshape((n, n, 2))[1::refinement_factor, 1::refinement_factor, :]
            .reshape(int(n / refinement_factor) ** 2, 2)
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
