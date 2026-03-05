"""
Functions for creating regular triangular mesh graphs.

Uses ``networkx.triangular_lattice_graph`` to produce an equilateral-triangle
lattice with 6-connectivity (each interior node has 6 neighbours).  This
mirrors the rectilinear mesh functions (which use ``networkx.grid_2d_graph``
with 8-connectivity) and plugs into the same two-step process:

1. **Coordinate creation** (this module) → ``nx.Graph`` with ``pos``, ``type``,
   and ``adjacency_type`` attributes.
2. **Connectivity creation** (``create_directed_mesh_graph``) → ``nx.DiGraph``
   with ``len`` and ``vdiff`` edge attributes.

Supports flat, flat_multiscale, and hierarchical ``m2m_connectivity`` modes.
"""

import networkx
import numpy as np
import scipy.spatial
from loguru import logger

from ....networkx_utils import prepend_node_index
from .. import coords as mesh_coords


def create_single_level_2d_triangular_mesh_primitive(
    xy: np.ndarray, nx: int, ny: int
):
    """
    Create an undirected triangular mesh primitive graph (``nx.Graph``) with
    node positions and spatial adjacency edges.

    This is analogous to ``create_single_level_2d_mesh_primitive`` but uses
    ``networkx.triangular_lattice_graph`` instead of ``grid_2d_graph``.

    In a triangular lattice, each interior node has 6 neighbours (vs. 8 for
    the rectilinear lattice with diagonals), providing more isotropic message
    passing.

    The nodes form a grid of ``(ny + 1)`` rows and ``(nx + 1) // 2`` columns,
    with odd-row nodes shifted horizontally.  Positions are scaled and offset
    so that the mesh spans the coordinate domain of *xy* (with nodes inset
    from the border by half a cell width in each direction).

    Parameters
    ----------
    xy : np.ndarray
        Grid point coordinates, shaped ``[N_grid_points, 2]``.
    nx : int
        Number of triangle columns (passed as *n* to
        ``triangular_lattice_graph``).
    ny : int
        Number of triangle rows (passed as *m* to
        ``triangular_lattice_graph``).

    Returns
    -------
    networkx.Graph
        Undirected mesh primitive graph.  Node attributes: ``pos``
        (np.ndarray[2,]), ``type`` (``"mesh"``).  Edge attributes:
        ``adjacency_type`` (always ``"cardinal"`` — triangular lattices have
        only one class of edge).  Graph attributes: ``dx``, ``dy``.
    """
    xm, xM = np.amin(xy[:, 0]), np.amax(xy[:, 0])
    ym, yM = np.amin(xy[:, 1]), np.amax(xy[:, 1])

    # Create the raw triangular lattice
    g_raw = networkx.triangular_lattice_graph(ny, nx, with_positions=True)

    if g_raw.number_of_nodes() == 0:
        raise ValueError(
            f"triangular_lattice_graph({ny}, {nx}) produced 0 nodes.  "
            "Increase nx/ny or decrease mesh_node_spacing."
        )

    # Gather raw positions to compute extent
    raw_positions = np.array([g_raw.nodes[n]["pos"] for n in g_raw.nodes()])
    raw_xmin, raw_ymin = raw_positions.min(axis=0)
    raw_xmax, raw_ymax = raw_positions.max(axis=0)
    raw_extent_x = raw_xmax - raw_xmin
    raw_extent_y = raw_ymax - raw_ymin

    # Domain extent with half-cell inset
    domain_x = xM - xm
    domain_y = yM - ym

    # Scale factors — map raw lattice extent to domain extent (inset by half
    # a cell in each direction, mirroring the rectilinear approach)
    if raw_extent_x > 0:
        scale_x = domain_x / (raw_extent_x + 1.0)  # +1 for inset
    else:
        scale_x = domain_x  # single column
    if raw_extent_y > 0:
        scale_y = domain_y / (raw_extent_y + np.sqrt(3) / 2)  # +row_h for inset
    else:
        scale_y = domain_y  # single row

    # Effective dx/dy for graph attributes
    dx = scale_x
    dy = scale_y * (np.sqrt(3) / 2)

    # Offset so mesh is centred within domain
    offset_x = xm + (domain_x - raw_extent_x * scale_x) / 2
    offset_y = ym + (domain_y - raw_extent_y * scale_y) / 2

    # Build output graph with scaled positions
    g = networkx.Graph()
    for node in g_raw.nodes():
        raw_pos = g_raw.nodes[node]["pos"]
        pos = np.array([
            offset_x + (raw_pos[0] - raw_xmin) * scale_x,
            offset_y + (raw_pos[1] - raw_ymin) * scale_y,
        ])
        g.add_node(node, pos=pos, type="mesh")

    for u, v in g_raw.edges():
        g.add_edge(u, v, adjacency_type="cardinal")

    g.graph["dx"] = dx
    g.graph["dy"] = dy

    return g


def create_multirange_2d_triangular_mesh_primitives(
    max_num_levels,
    xy,
    mesh_node_spacing=3,
    interlevel_refinement_factor=3,
):
    """
    Create a list of undirected triangular mesh primitive graphs representing
    different levels of mesh resolution.

    Mirrors ``create_multirange_2d_mesh_primitives`` but uses triangular
    lattice topology at each level.

    Parameters
    ----------
    max_num_levels : int
        Maximum number of levels in the multi-scale graph.
    xy : np.ndarray
        Grid point coordinates, shaped ``[N_grid_points, 2]``.
    mesh_node_spacing : float
        Distance between mesh nodes at the finest level, in coordinate units.
    interlevel_refinement_factor : float
        Factor by which mesh node count decreases per level.

    Returns
    -------
    list[networkx.Graph]
        Triangular mesh primitive graphs, one per level.
    """
    coord_extent = np.ptp(xy, axis=0)
    # For triangular lattice, ny accounts for row spacing of sqrt(3)/2
    max_nx = int(coord_extent[0] / mesh_node_spacing)
    max_ny = int(coord_extent[1] / (mesh_node_spacing * np.sqrt(3) / 2))

    max_nodes_bottom = np.array([max_nx, max_ny])

    max_mesh_levels_float = np.log(max_nodes_bottom) / np.log(
        interlevel_refinement_factor
    )
    max_mesh_levels = max_mesh_levels_float.astype(int)
    nleaf = interlevel_refinement_factor ** max_mesh_levels

    mesh_levels_to_create = max_mesh_levels.min()
    if max_num_levels:
        mesh_levels_to_create = min(mesh_levels_to_create, max_num_levels)

    logger.debug(
        f"triangular mesh_levels: {mesh_levels_to_create}, nleaf: {nleaf}"
    )

    G_all_levels = []
    for lev in range(mesh_levels_to_create):
        nodes_x, nodes_y = (
            nleaf / (interlevel_refinement_factor ** lev)
        ).astype(int)
        g = create_single_level_2d_triangular_mesh_primitive(
            xy, nodes_x, nodes_y
        )
        for node in g.nodes:
            g.nodes[node]["level"] = lev
        for edge in g.edges:
            g.edges[edge]["level"] = lev
        g.graph["level"] = lev
        g.graph["interlevel_refinement_factor"] = interlevel_refinement_factor
        G_all_levels.append(g)

    return G_all_levels


# ── Convenience wrapper functions (mirror flat.py) ──


def create_single_level_2d_triangular_mesh_graph(xy, nx, ny):
    """
    Create a directed triangular mesh graph from coordinates.

    Internally uses the two-step process:
    1. ``create_single_level_2d_triangular_mesh_primitive`` (coordinate creation)
    2. ``create_directed_mesh_graph`` (connectivity creation)

    For triangular lattices, *all* edges are ``"cardinal"`` so patterns
    ``"4-star"`` and ``"8-star"`` produce the same result (6-connectivity).

    Parameters
    ----------
    xy : np.ndarray
        Grid point coordinates, shaped ``[N_grid_points, 2]``.
    nx : int
        Number of triangle columns.
    ny : int
        Number of triangle rows.

    Returns
    -------
    networkx.DiGraph
        Directed triangular mesh graph.
    """
    G_coords = create_single_level_2d_triangular_mesh_primitive(xy, nx, ny)
    return mesh_coords.create_directed_mesh_graph(G_coords, pattern="4-star")


def create_flat_singlescale_triangular_mesh_graph(xy, mesh_node_distance: float):
    """
    Create a flat single-scale triangular mesh graph.

    Mirrors ``create_flat_singlescale_mesh_graph`` but with triangular
    lattice topology (6-connectivity).

    Parameters
    ----------
    xy : np.ndarray
        Grid point coordinates, shaped ``[N_grid_points, 2]``.
    mesh_node_distance : float
        Approximate side length of equilateral triangles, in coordinate units.

    Returns
    -------
    networkx.DiGraph
        Flat single-scale directed triangular mesh graph.
    """
    range_x, range_y = np.ptp(xy, axis=0)
    nx = int(range_x / mesh_node_distance)
    ny = int(range_y / (mesh_node_distance * np.sqrt(3) / 2))

    if nx == 0 or ny == 0:
        raise ValueError(
            "The given `mesh_node_distance` is too large for the provided "
            f"coordinates. Got mesh_node_distance={mesh_node_distance}, but "
            f"the x-range is {range_x} and y-range is {range_y}. Maybe you "
            "want to decrease the `mesh_node_distance` so that the mesh "
            "nodes are spaced closer together?"
        )

    return create_single_level_2d_triangular_mesh_graph(xy, nx, ny)


def create_flat_multiscale_from_triangular_coordinates(
    G_coords_list,
    pattern="4-star",
):
    """
    Create flat multiscale mesh graph from a list of triangular coordinate
    graphs.

    Unlike the rectilinear variant (``create_flat_multiscale_from_coordinates``)
    which relies on grid-index-based coincident-node detection, this function
    uses position-based matching.  For each coarser level, any node whose
    position coincides (within floating-point tolerance) with an existing finer
    level node is merged with it, so that multi-resolution edges share the
    same node identity.

    Parameters
    ----------
    G_coords_list : list[networkx.Graph]
        One undirected triangular mesh primitive per level.
    pattern : str
        Connectivity pattern: ``"4-star"`` or ``"8-star"`` (default ``"4-star"``).

    Returns
    -------
    networkx.DiGraph
        Flat multiscale triangular mesh graph.
    """
    # Convert each level to directed graph
    G_directed = [
        mesh_coords.create_directed_mesh_graph(g, pattern=pattern)
        for g in G_coords_list
    ]

    # Prepend level index to make node labels unique across levels
    G_directed = [
        prepend_node_index(g, level_i)
        for level_i, g in enumerate(G_directed)
    ]

    # Build merged graph, starting from finest level
    G_tot = G_directed[0]

    for lev in range(1, len(G_directed)):
        G_coarse = G_directed[lev]

        # KDTree of existing (finer) nodes for position matching
        fine_nodes = list(G_tot.nodes())
        fine_positions = np.array(
            [G_tot.nodes[n]["pos"] for n in fine_nodes]
        )
        kdt = scipy.spatial.KDTree(fine_positions)

        # Find which coarse nodes coincide with existing fine nodes
        relabel_map = {}
        for node in G_coarse.nodes():
            pos = G_coarse.nodes[node]["pos"]
            dist, idx = kdt.query(pos)
            if dist < 1e-8:
                relabel_map[node] = fine_nodes[idx]

        if relabel_map:
            G_coarse = networkx.relabel_nodes(G_coarse, relabel_map)

        G_tot = networkx.compose(G_tot, G_coarse)

    # Re-index to sequential (0, i) labels
    G_tot = prepend_node_index(G_tot, 0)

    # Preserve dx/dy as per-level dicts
    G_tot.graph["dx"] = {i: g.graph["dx"] for i, g in enumerate(G_directed)}
    G_tot.graph["dy"] = {i: g.graph["dy"] for i, g in enumerate(G_directed)}

    return G_tot


def create_flat_multiscale_triangular_mesh_graph(
    xy,
    mesh_node_distance: float,
    level_refinement_factor: int,
    max_num_levels: int,
):
    """
    Create a flat multiscale triangular mesh graph.

    Mirrors ``create_flat_multiscale_mesh_graph`` but with triangular lattice
    topology at each level.

    Parameters
    ----------
    xy : np.ndarray
        Grid point coordinates, shaped ``[N_grid_points, 2]``.
    mesh_node_distance : float
        Approximate side length of equilateral triangles at finest level.
    level_refinement_factor : int
        Refinement factor between levels (must be an odd integer > 1).
    max_num_levels : int
        Maximum number of levels in the multiscale graph.

    Returns
    -------
    networkx.DiGraph
        Flat multiscale directed triangular mesh graph.
    """
    G_coords_list = create_multirange_2d_triangular_mesh_primitives(
        max_num_levels=max_num_levels,
        xy=xy,
        mesh_node_spacing=mesh_node_distance,
        interlevel_refinement_factor=level_refinement_factor,
    )

    return create_flat_multiscale_from_triangular_coordinates(
        G_coords_list, pattern="4-star"
    )


def create_hierarchical_triangular_mesh_graph(
    xy,
    mesh_node_distance: float,
    level_refinement_factor: float,
    max_num_levels: int,
    intra_level=None,
    inter_level=None,
):
    """
    Create a hierarchical multiscale triangular mesh graph.

    Mirrors ``create_hierarchical_multiscale_mesh_graph`` but with triangular
    lattice topology at each level.

    Parameters
    ----------
    xy : np.ndarray
        Grid point coordinates, shaped ``[N_grid_points, 2]``.
    mesh_node_distance : float
        Approximate side length of equilateral triangles at finest level.
    level_refinement_factor : float
        Refinement factor between levels.
    max_num_levels : int
        Maximum number of levels.
    intra_level : dict, optional
        Intra-level connectivity config.  Default: ``{"pattern": "4-star"}``.
    inter_level : dict, optional
        Inter-level connectivity config.  Default: ``{"pattern": "nearest", "k": 1}``.

    Returns
    -------
    networkx.DiGraph
        Hierarchical directed triangular mesh graph.
    """
    from .hierarchical import create_hierarchical_from_coordinates

    if intra_level is None:
        intra_level = {"pattern": "4-star"}
    if inter_level is None:
        inter_level = {"pattern": "nearest", "k": 1}

    G_coords_list = create_multirange_2d_triangular_mesh_primitives(
        max_num_levels=max_num_levels,
        xy=xy,
        mesh_node_spacing=mesh_node_distance,
        interlevel_refinement_factor=level_refinement_factor,
    )

    return create_hierarchical_from_coordinates(
        G_coords_list,
        intra_level=intra_level,
        inter_level=inter_level,
    )
