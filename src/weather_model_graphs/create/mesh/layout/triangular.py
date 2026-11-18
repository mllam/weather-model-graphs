"""
Triangular mesh layout: coordinate creation for regular triangular lattices.

Uses ``networkx.triangular_lattice_graph`` to produce an equilateral-triangle
lattice with 6-connectivity (each interior node has 6 neighbours).  This
mirrors the rectilinear layout (which uses ``networkx.grid_2d_graph``
with 8-connectivity) and plugs into the same two-step process:

1. **Coordinate creation** (this module) -> ``nx.Graph`` with ``pos``, ``type``,
   and ``adjacency_type`` attributes.
2. **Connectivity creation** (``create_directed_mesh_graph`` in
   ``connectivity.general``) -> ``nx.DiGraph`` with ``len`` and ``vdiff``
   edge attributes.

The lattice is scaled by a single factor in both directions, so the triangles
stay equilateral whatever the aspect ratio of the coordinate domain.  When
``mesh_node_spacing`` is given it is therefore the true distance between
neighbouring mesh nodes, and the lattice is sized to cover the domain (which
means the outermost nodes can sit just outside it).  Scaling x and y
independently would fit the domain exactly but skew the triangles, which
would defeat the purpose of using a triangular lattice.
"""

from typing import List, Optional, Tuple

import networkx
import numpy as np
from loguru import logger

# Vertical distance between rows of a triangular lattice with unit edge length
_ROW_HEIGHT = np.sqrt(3) / 2


def _raw_lattice_span(m: int, n: int) -> Tuple[float, float]:
    """
    Extent of the unit-edge lattice produced by
    ``networkx.triangular_lattice_graph(m, n)``.

    Parameters
    ----------
    m : int
        Number of triangle rows.
    n : int
        Number of triangle columns.

    Returns
    -------
    tuple[float, float]
        Extent in the x- and y-direction, in units of the lattice edge length.
    """
    return (n + 1) / 2, m * _ROW_HEIGHT


def _lattice_counts_covering(
    domain_x: float, domain_y: float, mesh_node_spacing: float
) -> Tuple[int, int]:
    """
    Smallest lattice counts whose extent covers the domain at a given edge length.

    Parameters
    ----------
    domain_x, domain_y : float
        Extent of the coordinate domain.
    mesh_node_spacing : float
        Distance between neighbouring mesh nodes (the lattice edge length).

    Returns
    -------
    tuple[int, int]
        ``(m, n)``, the row and column counts for
        ``networkx.triangular_lattice_graph``.
    """
    n = max(int(np.ceil(2 * domain_x / mesh_node_spacing - 1)), 1)
    m = max(int(np.ceil(domain_y / (mesh_node_spacing * _ROW_HEIGHT))), 1)
    return m, n


def create_single_level_2d_mesh_primitive(
    xy: np.ndarray,
    nx: int = None,
    ny: int = None,
    *,
    mesh_node_spacing: float = None,
    origin: Optional[np.ndarray] = None,
) -> networkx.Graph:
    """
    Create an undirected triangular mesh primitive graph (``nx.Graph``) with
    node positions and spatial adjacency edges.

    This is analogous to ``create_single_level_2d_mesh_primitive`` in the
    rectilinear layout but uses ``networkx.triangular_lattice_graph`` instead
    of ``grid_2d_graph``.

    In a triangular lattice each interior node has 6 neighbours (vs. 8 for the
    rectilinear lattice with diagonals), all at the same distance, which gives
    more isotropic message passing.  The lattice is scaled by the same factor
    in both directions so that this property holds for any domain shape.

    Either provide ``mesh_node_spacing``, or provide ``nx`` and ``ny``
    directly:

    - with ``mesh_node_spacing`` the lattice edge length *is* the requested
      spacing and the node counts are chosen so the mesh covers the domain,
      so the outermost nodes may lie just outside it.
    - with ``nx``/``ny`` the node counts are fixed and the lattice is scaled
      to the largest size that still fits inside the domain, so the edge
      length follows from the counts.

    Parameters
    ----------
    xy : np.ndarray
        Grid point coordinates, shaped ``[N_grid_points, 2]``.
    nx : int, optional
        Number of triangle columns (passed as *n* to
        ``triangular_lattice_graph``). If not given, computed from
        ``mesh_node_spacing``.
    ny : int, optional
        Number of triangle rows (passed as *m* to
        ``triangular_lattice_graph``). If not given, computed from
        ``mesh_node_spacing``.
    mesh_node_spacing : float, optional
        Distance between neighbouring mesh nodes, in coordinate units.
    origin : np.ndarray, optional
        Position of the lattice's lower-left corner node, shaped ``[2,]``.
        If not given the lattice is centred on the domain.  Multi-level
        meshes pass a shared origin so that coarser levels place their nodes
        on top of finer-level nodes.

    Returns
    -------
    networkx.Graph
        Undirected mesh primitive graph.  Node attributes: ``pos``
        (np.ndarray[2,]), ``type`` (``"mesh"``).  Edge attributes:
        ``adjacency_type`` (always ``"cardinal"`` -- triangular lattices have
        only one class of edge).  Graph attributes: ``dx`` (distance between
        neighbouring nodes in a row) and ``dy`` (distance between rows).
    """
    xm, xM = np.amin(xy[:, 0]), np.amax(xy[:, 0])
    ym, yM = np.amin(xy[:, 1]), np.amax(xy[:, 1])
    domain_x = xM - xm
    domain_y = yM - ym

    if mesh_node_spacing is not None:
        # The requested spacing is the lattice edge length; pick counts that
        # cover the domain.
        ny, nx = _lattice_counts_covering(domain_x, domain_y, mesh_node_spacing)
        scale = mesh_node_spacing
    elif nx is None or ny is None:
        raise ValueError(
            "Either provide both `nx` and `ny`, or provide "
            "`mesh_node_spacing` to compute them automatically."
        )
    else:
        # Counts are fixed: use the largest uniform scale that keeps the
        # lattice inside the domain (a single factor, so triangles stay
        # equilateral).
        raw_span_x, raw_span_y = _raw_lattice_span(ny, nx)
        scale_x = domain_x / raw_span_x if raw_span_x > 0 else domain_x
        scale_y = domain_y / raw_span_y if raw_span_y > 0 else domain_y
        scale = min(scale_x, scale_y)

    # Create the raw triangular lattice (unit edge length)
    g_raw = networkx.triangular_lattice_graph(ny, nx, with_positions=True)

    if g_raw.number_of_nodes() == 0:
        raise ValueError(
            f"triangular_lattice_graph({ny}, {nx}) produced 0 nodes.  "
            "Increase nx/ny or decrease mesh_node_spacing."
        )

    raw_positions = np.array([g_raw.nodes[n]["pos"] for n in g_raw.nodes()])
    raw_min = raw_positions.min(axis=0)
    raw_extent = raw_positions.max(axis=0) - raw_min

    if origin is None:
        # Centre the (scaled) lattice on the domain
        origin = np.array(
            [
                xm + (domain_x - raw_extent[0] * scale) / 2,
                ym + (domain_y - raw_extent[1] * scale) / 2,
            ]
        )
    else:
        origin = np.asarray(origin, dtype=float)

    # Build output graph with scaled positions
    g = networkx.Graph()
    for node in g_raw.nodes():
        pos = origin + (np.asarray(g_raw.nodes[node]["pos"]) - raw_min) * scale
        g.add_node(node, pos=pos, type="mesh")

    for u, v in g_raw.edges():
        g.add_edge(u, v, adjacency_type="cardinal")

    # Distance between neighbouring nodes within a row, and between rows
    g.graph["dx"] = scale
    g.graph["dy"] = scale * _ROW_HEIGHT

    return g


def _centred_origin(xy: np.ndarray, mesh_node_spacing: float) -> np.ndarray:
    """
    Lower-left corner of the lattice that covers the domain, centred on it.

    Multi-level meshes anchor every level to this same point so that coarser
    levels (whose spacing is a multiple of the finest) place their nodes on
    top of finer-level nodes.
    """
    xm, xM = np.amin(xy[:, 0]), np.amax(xy[:, 0])
    ym, yM = np.amin(xy[:, 1]), np.amax(xy[:, 1])
    domain_x, domain_y = xM - xm, yM - ym
    m, n = _lattice_counts_covering(domain_x, domain_y, mesh_node_spacing)
    span_x, span_y = _raw_lattice_span(m, n)
    return np.array(
        [
            xm + (domain_x - span_x * mesh_node_spacing) / 2,
            ym + (domain_y - span_y * mesh_node_spacing) / 2,
        ]
    )


def create_multirange_2d_mesh_primitives(
    xy: np.ndarray,
    *,
    max_num_levels: int = None,
    mesh_node_spacing: float = 3,
    interlevel_refinement_factor: int = 3,
) -> List[networkx.Graph]:
    """
    Create a list of undirected triangular mesh primitive graphs representing
    different levels of mesh resolution.

    Mirrors ``create_multirange_2d_mesh_primitives`` in the rectilinear layout
    but uses triangular lattice topology at each level.  Level ``l`` has a node
    spacing of ``mesh_node_spacing * interlevel_refinement_factor**l``, and all
    levels are anchored to the same origin, so that (for an odd refinement
    factor) coarser-level nodes coincide with finer-level nodes.  Multiscale
    connectivity relies on that coincidence to merge the levels.

    Parameters
    ----------
    xy : np.ndarray
        Grid point coordinates, shaped ``[N_grid_points, 2]``.
    max_num_levels : int, optional
        Maximum number of levels in the multi-scale graph. If None (default),
        as many levels are created as the domain allows.
    mesh_node_spacing : float
        Distance between mesh nodes at the finest level, in coordinate units.
    interlevel_refinement_factor : int
        Factor by which the mesh node spacing grows per level.

    Returns
    -------
    list[networkx.Graph]
        Triangular mesh primitive graphs, one per level, finest first.
    """
    coord_extent = np.ptp(xy, axis=0)
    # Node counts at the finest level, used to work out how many levels the
    # domain can support before the coarsest mesh runs out of nodes.
    max_ny, max_nx = _lattice_counts_covering(
        coord_extent[0], coord_extent[1], mesh_node_spacing
    )
    max_nodes_bottom = np.array([max_nx, max_ny])

    max_mesh_levels = (
        np.log(max_nodes_bottom) / np.log(interlevel_refinement_factor)
    ).astype(int)

    mesh_levels_to_create = max(int(max_mesh_levels.min()), 1)
    if max_num_levels:
        mesh_levels_to_create = min(mesh_levels_to_create, max_num_levels)

    logger.debug(f"triangular mesh_levels: {mesh_levels_to_create}")

    # All levels share the finest level's origin so their nodes line up
    origin = _centred_origin(xy, mesh_node_spacing)

    G_all_levels = []
    for lev in range(mesh_levels_to_create):
        level_spacing = mesh_node_spacing * interlevel_refinement_factor**lev
        g = create_single_level_2d_mesh_primitive(
            xy, mesh_node_spacing=level_spacing, origin=origin
        )
        for node in g.nodes:
            g.nodes[node]["level"] = lev
        for edge in g.edges:
            g.edges[edge]["level"] = lev
        g.graph["level"] = lev
        g.graph["interlevel_refinement_factor"] = interlevel_refinement_factor
        G_all_levels.append(g)

    return G_all_levels
