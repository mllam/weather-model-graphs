import networkx
import numpy as np
import scipy

from ....networkx_utils import prepend_node_index
from .. import mesh as mesh_graph


def create_hierarchical_from_coordinates(
    G_coords_list,
    intra_level=None,
    inter_level=None,
):
    """
    Create a hierarchical multiscale mesh graph from a list of coordinate graphs.

    This is the connectivity creation step for hierarchical meshes.
    It takes undirected coordinate graphs (one per level) and produces a
    directed mesh graph with intra-level connectivity and inter-level
    up/down connections.

    Parameters
    ----------
    G_coords_list : list of networkx.Graph
        List of undirected coordinate graphs, one per level. Each should have
        nodes with "pos" and "type" attributes, and edges with "adjacency_type"
        attributes. Created by create_multirange_2d_mesh_coordinates.
    intra_level : dict or None
        Intra-level connectivity options. Supports:
        - pattern: str, "4-star" or "8-star" (default: "8-star")
    inter_level : dict or None
        Inter-level connectivity options. Supports:
        - pattern: str, "nearest" (default: "nearest")
        - k: int, number of nearest neighbours (default: 1)

    Returns
    -------
    networkx.DiGraph
        A directed graph containing the hierarchical mesh with intra-level
        edges (direction="same"), inter-level down edges (direction="down"),
        and inter-level up edges (direction="up").
    """
    if intra_level is None:
        intra_level = {"pattern": "8-star"}
    if inter_level is None:
        inter_level = {"pattern": "nearest", "k": 1}

    intra_pattern = intra_level.get("pattern", "8-star")
    inter_pattern = inter_level.get("pattern", "nearest")
    inter_k = inter_level.get("k", 1)

    if inter_pattern != "nearest":
        raise NotImplementedError(
            f"Inter-level pattern '{inter_pattern}' is not yet supported "
            "for hierarchical graphs. Only 'nearest' is currently implemented."
        )

    # Convert each level's coordinate graph to directed graph with chosen pattern
    Gs_all_levels = [
        mesh_graph.create_directed_mesh_graph(g_coords, pattern=intra_pattern)
        for g_coords in G_coords_list
    ]

    n_mesh_levels = len(Gs_all_levels)

    if n_mesh_levels < 2:
        raise ValueError(
            "At least two mesh levels are required for hierarchical mesh graph. "
            "You may need to reduce the level refinement factor "
            f"or increase the max number of levels "
            f"or number of grid points."
        )

    # Relabel nodes of each level with level index first
    Gs_all_levels = [
        prepend_node_index(graph, level_i)
        for level_i, graph in enumerate(Gs_all_levels)
    ]

    # add `direction` attribute to all edges with value `same`
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

        # add edges from coarser to finer level
        for v in v_to_list:
            # find k nearest neighbours (index to vm_xy)
            neigh_idx = kdt_m.query(G_down.nodes[v]["pos"], inter_k)[1]
            if inter_k == 1:
                neigh_idx = [neigh_idx]

            for idx in neigh_idx:
                u = v_from_list[idx]

                # add edge from coarser to finer
                G_down.add_edge(u, v)
                d = np.sqrt(
                    np.sum(
                        (G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]) ** 2
                    )
                )
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


def create_hierarchical_multiscale_mesh_graph(
    xy,
    mesh_node_distance: float,
    level_refinement_factor: float,
    max_num_levels: int,
):
    """
    Create a hierarchical multiscale mesh graph with nearest neighbour
    connections within each level (horizontally, vertically, and diagonally), and
    connections between levels (coarse to fine and fine to coarse) using the
    nearest neighbour connection.

    Internally uses the two-step process:
    1. create_multirange_2d_mesh_coordinates (coordinate creation)
    2. create_hierarchical_from_coordinates (connectivity creation)

    Parameters
    ----------
    xy: np.ndarray
        2D array of mesh point positions.
        Distance (in x- and y-direction) between created mesh nodes in bottom level,
        in coordinate system of xy
    mesh_node_distance: float
        Distance (in x- and y-direction) between created mesh nodes in bottom level,
        in coordinate system of xy
    level_refinement_factor: float
        Refinement factor between grid points and bottom level of mesh hierarchy
    max_num_levels: int
        The number of levels in the hierarchical mesh graph.

    Returns
    -------
    networkx.DiGraph
        A directed graph containing the hierarchical mesh with intra-level,
        up, and down edges.
    """
    G_coords_list = mesh_graph.create_multirange_2d_mesh_coordinates(
        max_num_levels=max_num_levels,
        xy=xy,
        grid_spacing=mesh_node_distance,
        interlevel_refinement_factor=level_refinement_factor,
    )

    return create_hierarchical_from_coordinates(
        G_coords_list,
        intra_level={"pattern": "8-star"},
        inter_level={"pattern": "nearest", "k": 1},
    )
