import networkx
import numpy as np
import scipy

from ....networkx_utils import prepend_node_index
from .. import mesh as mesh_graph


def create_hierarchical_multiscale_mesh_graph(
    xy,
    grid_refinement_factor: float,
    level_refinement_factor: float,
    max_num_levels: int,
):
    """
    Create a hierarchical multiscale mesh graph with nearest neighbour
    connections within each level (horizontally, vertically, and diagonally), and
    connections between levels (coarse to fine and fine to coarse) using the
    nearest neighbour connection.

    Parameters
    ----------
    xy: np.ndarray
        2D array of mesh point positions.
    refinement_factor: int
        Refinement factor between mesh levels, the reduction factor in the number of
        mesh points between levels (in both x and y directions).
    max_num_levels: int
        The number of levels in the hierarchical mesh graph.

    Returns
    -------
    dict
        A dictionary containing the hierarchical mesh graph, the mesh down graph, and
        the mesh up graph, with keys "m2m", "mesh_down", and "mesh_up" respectively.
    """
    Gs_all_levels: list[networkx.DiGraph] = mesh_graph.create_multirange_2d_mesh_graphs(
        max_num_levels=max_num_levels,
        xy=xy,
        grid_refinement_factor=grid_refinement_factor,
        level_refinement_factor=level_refinement_factor,
    )
    n_mesh_levels = len(Gs_all_levels)

    if n_mesh_levels < 2:
        raise ValueError(
            "At least two mesh levels are required for hierarchical mesh graph. "
            f"You may need to reduce the refinement factors"
            f"or increase the max number of levels {max_num_levels} "
            f"or number of grid points {xy.shape[1:]}."
        )

    # Relabel nodes of each level with level index first

    Gs_all_levels = [
        prepend_node_index(graph, level_i)
        for level_i, graph in enumerate(Gs_all_levels)
    ]

    # add `direction` attribute to all edges with value `same``
    for i, G in enumerate(Gs_all_levels):
        for u, v in G.edges:
            G.edges[u, v]["direction"] = "same"
            G.edges[u, v]["level"] = i

    # Create inter-level mesh edges
    up_graphs = []
    down_graphs = []
    for G_from, G_to in zip(
        Gs_all_levels[1:],
        Gs_all_levels[:-1],
    ):
        from_level = G_from.graph["level"]
        to_level = G_to.graph["level"]

        # start out from graph at from level
        G_down = G_from.copy()
        G_down.clear_edges()
        G_down = networkx.DiGraph(G_down)

        # Add nodes of to level
        G_down.add_nodes_from(G_to.nodes(data=True))

        # build kd tree for mesh point pos
        # order in vm should be same as in vm_xy
        v_to_list = list(G_to.nodes)
        v_from_list = list(G_from.nodes)
        v_from_xy = np.array([xy for _, xy in G_from.nodes.data("pos")])
        kdt_m = scipy.spatial.KDTree(v_from_xy)

        # add edges from mesh to grid
        for v in v_to_list:
            # find 1(?) nearest neighbours (index to vm_xy)
            neigh_idx = kdt_m.query(G_down.nodes[v]["pos"], 1)[1]
            u = v_from_list[neigh_idx]

            # add edge from mesh to grid
            G_down.add_edge(u, v)
            d = np.sqrt(np.sum((G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]) ** 2))
            G_down.edges[u, v]["len"] = d
            G_down.edges[u, v]["vdiff"] = (
                G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]
            )
            G_down.edges[u, v]["levels"] = f"{from_level}>{to_level}"
            G_down.edges[u, v]["direction"] = "down"

        G_up = networkx.DiGraph()
        G_up.add_nodes_from(G_down.nodes(data=True))
        for u, v, data in G_down.edges(data=True):
            data = data.copy()
            data["levels"] = f"{to_level}>{from_level}"
            data["direction"] = "up"
            G_up.add_edge(v, u, **data)

        up_graphs.append(G_up)
        down_graphs.append(G_down)

    G_up_all = networkx.compose_all(up_graphs)
    G_down_all = networkx.compose_all(down_graphs)
    G_all_levels = networkx.compose_all(Gs_all_levels)

    G_m2m = networkx.compose_all([G_all_levels, G_up_all, G_down_all])

    # add dx and dy to graph
    for prop in ("dx", "dy"):
        G_m2m.graph[prop] = {i: g.graph[prop] for i, g in enumerate(Gs_all_levels)}

    return G_m2m
