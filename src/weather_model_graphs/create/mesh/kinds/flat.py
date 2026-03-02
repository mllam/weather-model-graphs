import networkx
import numpy as np

from ....networkx_utils import prepend_node_index
from .. import mesh as mesh_graph


def create_flat_multiscale_from_coordinates(
    G_coords_list,
    pattern="8-star",
):
    """
    Create flat multiscale mesh graph from a list of coordinate graphs.

    This is the connectivity creation step for flat multiscale meshes.
    It takes undirected coordinate graphs (one per level) and produces a
    single directed mesh graph where all levels are merged into one flat graph.

    In a flat multiscale graph, coarser levels are merged into the finer level
    by coincident node positions (no separate inter-level connectivity needed).
    The ``pattern`` controls the intra-level edge connectivity for each level.

    Parameters
    ----------
    G_coords_list : list of networkx.Graph
        List of undirected coordinate graphs, one per level. Each should have
        nodes with "pos" and "type" attributes, and edges with "adjacency_type"
        attributes. Created by create_multirange_2d_mesh_coordinates.
    pattern : str
        Connectivity pattern for intra-level edges: "4-star" or "8-star"
        (default: "8-star")

    Returns
    -------
    G_tot : networkx.DiGraph
        The merged flat multiscale mesh graph
    """

    # Retrieve interlevel_refinement_factor from graph attributes
    interlevel_refinement_factor = G_coords_list[0].graph.get(
        "interlevel_refinement_factor", 3
    )

    # Check that interlevel_refinement_factor is an odd integer
    if (
        int(interlevel_refinement_factor) != interlevel_refinement_factor
        or interlevel_refinement_factor % 2 != 1
    ):
        raise ValueError(
            "The `interlevel_refinement_factor` must be an odd integer. "
            f"Given value: {interlevel_refinement_factor}."
        )

    # Convert each level's coordinate graph to directed graph with chosen pattern
    G_all_levels = [
        mesh_graph.create_directed_mesh_graph(g_coords, pattern=pattern)
        for g_coords in G_coords_list
    ]

    # combine all levels to one graph
    G_tot = G_all_levels[0]
    # First node at level l+1 share position with node (offset, offset) at level l
    level_offset = interlevel_refinement_factor // 2

    first_level_nodes = list(G_all_levels[0].nodes)
    # Last nodes in first layer has pos (nx-1, ny-1)
    num_nodes_x = first_level_nodes[-1][0] + 1
    num_nodes_y = first_level_nodes[-1][1] + 1

    for lev in range(1, len(G_all_levels)):
        nodes = list(G_all_levels[lev - 1].nodes)
        ij = (
            np.array(nodes)
            .reshape((num_nodes_x, num_nodes_y, 2))[
                level_offset::interlevel_refinement_factor,
                level_offset::interlevel_refinement_factor,
                :,
            ]
            .reshape(
                int(
                    num_nodes_x
                    * num_nodes_y
                    / (interlevel_refinement_factor**2)
                ),
                2,
            )
        )
        ij = [tuple(x) for x in ij]
        G_all_levels[lev] = networkx.relabel_nodes(
            G_all_levels[lev], dict(zip(G_all_levels[lev].nodes, ij))
        )
        G_tot = networkx.compose(G_tot, G_all_levels[lev])

        # Update number of nodes in x- and y-direction for next iteration
        num_nodes_x //= interlevel_refinement_factor
        num_nodes_y //= interlevel_refinement_factor

    # Relabel mesh nodes to start with 0
    G_tot = prepend_node_index(G_tot, 0)

    # add dx and dy to graph
    G_tot.graph["dx"] = {i: g.graph["dx"] for i, g in enumerate(G_all_levels)}
    G_tot.graph["dy"] = {i: g.graph["dy"] for i, g in enumerate(G_all_levels)}

    return G_tot


def create_flat_singlescale_from_coordinates(G_coords, pattern="8-star"):
    """
    Create a flat single-scale directed mesh graph from a coordinate graph.

    This is the connectivity creation step for flat single-scale meshes.
    It converts an undirected coordinate graph to a directed mesh graph
    using the specified connectivity pattern.

    Parameters
    ----------
    G_coords : networkx.Graph
        Undirected coordinate graph with nodes having "pos" attributes and
        edges having "adjacency_type" attributes. Created by
        create_single_level_2d_mesh_coordinates.
    pattern : str
        Connectivity pattern: "4-star" or "8-star" (default: "8-star")

    Returns
    -------
    networkx.DiGraph
        The flat single-scale directed mesh graph
    """
    return mesh_graph.create_directed_mesh_graph(G_coords, pattern=pattern)


def create_flat_multiscale_mesh_graph(
    xy, mesh_node_distance: float, level_refinement_factor: int, max_num_levels: int
):
    """
    Create flat mesh graph by merging the single-level mesh
    graphs across all levels in `G_all_levels`.

    Internally uses the two-step process:
    1. create_multirange_2d_mesh_coordinates (coordinate creation)
    2. create_flat_multiscale_from_coordinates (connectivity creation)

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
    G_coords_list = mesh_graph.create_multirange_2d_mesh_coordinates(
        max_num_levels=max_num_levels,
        xy=xy,
        grid_spacing=mesh_node_distance,
        interlevel_refinement_factor=level_refinement_factor,
    )

    return create_flat_multiscale_from_coordinates(
        G_coords_list,
        pattern="8-star",
    )


def create_flat_singlescale_mesh_graph(xy, mesh_node_distance: float):
    """
    Create flat mesh graph of single level

    Internally uses the two-step process:
    1. create_single_level_2d_mesh_coordinates (coordinate creation)
    2. create_directed_mesh_graph (connectivity creation, pattern="8-star")

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
