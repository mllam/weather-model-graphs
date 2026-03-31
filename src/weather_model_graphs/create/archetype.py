from typing import Iterable

import pyproj

from .base import create_all_graph_components


def create_keisler_graph(
    coords,
    mesh_node_distance=3,
    coords_crs: pyproj.crs.CRS | None = None,
    graph_crs: pyproj.crs.CRS | None = None,
    decode_mask: Iterable[bool] | None = None,
    return_components: bool = False,
):
    """
    Create a flat LAM graph from Oskarsson et al (2023, https://arxiv.org/abs/2309.17370)
    This graph setup is inspired by the global graph used by Keisler (2022, https://arxiv.org/abs/2202.07575).

    This graph is a flat single scale graph with nearest neighbour connectivity
    (8 neighbours) within the mesh.

    The grid to mesh connectivity connects each mesh node to grid nodes withing
    distance 0.51d, where d is the length of diagonal edges between neighbouring
    mesh nodes. The choice of 0.51 makes sure that all grid node positions will
    be connected to at least one mesh node (see
    https://www.desmos.com/calculator/sqqz0ka4ho for a visualization).
    The mesh to grid connectivity connects each grid point to the 4 nearest mesh nodes.

    Parameters
    ----------
    coords: np.ndarray
        2D array of grid point positions, in coordinate CRS
    mesh_node_distance: float
        Distance (in x- and y-direction) between created mesh nodes,
        in coordinate system of coords
    coords_crs: pyproj.crs.CRS or None
        Coordinate Reference System (CRS) of the input `coords`. Must be a
        pyproj.crs.CRS or cartopy.crs.CRS.
    graph_crs: pyproj.crs.CRS or None
        CRS where the graph creation should take place. If both `coords_crs`
        and `graph_crs` are given, coords are transformed from `coords_crs` to
        `graph_crs`. If any one of them is None, the original coords are used.
    decode_mask: Iterable[bool] or None
        An Iterable of booleans, masking which grid positions should be decoded
        to (included in the m2g subgraph), i.e., which positions should be output.
        Must have the same length as `coords`. If None (default), all grid nodes
        are included.
    return_components: bool, default False
        If True, the function returns a dict with m2g, m2m, and g2m as separate
        graphs. If False, returns one combined graph.

    Returns
    -------
    networkx.DiGraph or dict[networkx.DiGraph]
        The graph or graph components.
    """
    return create_all_graph_components(
        coords=coords,
        m2m_connectivity="flat",
        m2m_connectivity_kwargs=dict(mesh_node_distance=mesh_node_distance),
        g2m_connectivity="within_radius",
        m2g_connectivity="nearest_neighbours",
        g2m_connectivity_kwargs=dict(
            rel_max_dist=0.51,
        ),
        m2g_connectivity_kwargs=dict(
            max_num_neighbours=4,
        ),
        coords_crs=coords_crs,
        graph_crs=graph_crs,
        decode_mask=decode_mask,
        return_components=return_components,
    )


def create_graphcast_graph(
    coords,
    mesh_node_distance=3,
    level_refinement_factor=3,
    max_num_levels=None,
    coords_crs: pyproj.crs.CRS | None = None,
    graph_crs: pyproj.crs.CRS | None = None,
    decode_mask: Iterable[bool] | None = None,
    return_components: bool = False,
):
    """
    Create a multiscale LAM graph from Oskarsson et al (2023, https://arxiv.org/abs/2309.17370)
    This graph setup is inspired by the global GraphCast graph used by Lam et al (2023, https://arxiv.org/abs/2212.12794)

    This graph is a flat multiscale graph with neighbour connectivity and longer multi-scale edges.

    The grid to mesh connectivity connects each mesh node to grid nodes withing
    distance 0.51d, where d is the length of diagonal edges between neighbouring
    mesh nodes. The choice of 0.51 makes sure that all grid node positions will
    be connected to at least one mesh node (see
    https://www.desmos.com/calculator/sqqz0ka4ho for a visualization).
    The mesh to grid connectivity connects each grid point to the 4 nearest mesh nodes.

    Parameters
    ----------
    coords: np.ndarray
        2D array of grid point positions, in coordinate CRS
    mesh_node_distance: float
        Distance (in x- and y-direction) between created mesh nodes,
        in coordinate system of coords
    level_refinement_factor: int
        Refinement factor between grid points and bottom level of mesh hierarchy
        NOTE: Must be an odd integer >1 to create proper multiscale graph
    max_num_levels: int
        The number of levels of longer-range connections in the mesh graph.
    coords_crs: pyproj.crs.CRS or None
        Coordinate Reference System (CRS) of the input `coords`. Must be a
        pyproj.crs.CRS or cartopy.crs.CRS.
    graph_crs: pyproj.crs.CRS or None
        CRS where the graph creation should take place. If both `coords_crs`
        and `graph_crs` are given, coords are transformed from `coords_crs` to
        `graph_crs`. If any one of them is None, the original coords are used.
    decode_mask: Iterable[bool] or None
        An Iterable of booleans, masking which grid positions should be decoded
        to (included in the m2g subgraph), i.e., which positions should be output.
        Must have the same length as `coords`. If None (default), all grid nodes
        are included.
    return_components: bool, default False
        If True, the function returns a dict with m2g, m2m, and g2m as separate
        graphs. If False, returns one combined graph.

    Returns
    -------
    networkx.DiGraph or dict[networkx.DiGraph]
        The graph or graph components.
    """
    return create_all_graph_components(
        coords=coords,
        m2m_connectivity="flat_multiscale",
        m2m_connectivity_kwargs=dict(
            mesh_node_distance=mesh_node_distance,
            level_refinement_factor=level_refinement_factor,
            max_num_levels=max_num_levels,
        ),
        g2m_connectivity="within_radius",
        m2g_connectivity="nearest_neighbours",
        g2m_connectivity_kwargs=dict(
            rel_max_dist=0.51,
        ),
        m2g_connectivity_kwargs=dict(
            max_num_neighbours=4,
        ),
        coords_crs=coords_crs,
        graph_crs=graph_crs,
        decode_mask=decode_mask,
        return_components=return_components,
    )


def create_oskarsson_hierarchical_graph(
    coords,
    mesh_node_distance=3,
    level_refinement_factor=3,
    max_num_levels=None,
    coords_crs: pyproj.crs.CRS | None = None,
    graph_crs: pyproj.crs.CRS | None = None,
    decode_mask: Iterable[bool] | None = None,
    return_components: bool = False,
):
    """
    Create a LAM graph following Oskarsson et al (2023, https://arxiv.org/abs/2309.17370)
    hierarchical architecture.

    The mesh graph in this architecture is hierarchical in that each refinement of
    longer-range edges are split into different levels. In addition to these same-level
    connections the mesh graph contains nearest neighbour connections between
    levels (up and down). To distinguish between these these three types of
    edge connections each edge has a `direction` attribute (with value "up",
    "down", or "same"). In addition, the `levels` attribute indicates which two levels
    are connected for cross-level edges (e.g. "1>2" for edges between level 1 and 2).

    The grid to mesh connectivity connects each mesh node to grid nodes withing
    distance 0.51d, where d is the length of diagonal edges between neighbouring
    mesh nodes. The choice of 0.51 makes sure that all grid node positions will
    be connected to at least one mesh node (see
    https://www.desmos.com/calculator/sqqz0ka4ho for a visualization).
    The mesh to grid connectivity connects each grid point to the 4 nearest mesh nodes.

    Parameters
    ----------
    coords: np.ndarray
        2D array of grid point positions, in coordinate CRS
    mesh_node_distance: float
        Distance (in x- and y-direction) between created mesh nodes in bottom level,
        in coordinate system of coords
    level_refinement_factor: float
        Refinement factor between grid points and bottom level of mesh hierarchy
    max_num_levels: int
        The number of levels of longer-range connections in the mesh graph.
    coords_crs: pyproj.crs.CRS or None
        Coordinate Reference System (CRS) of the input `coords`. Must be a
        pyproj.crs.CRS or cartopy.crs.CRS.
    graph_crs: pyproj.crs.CRS or None
        CRS where the graph creation should take place. If both `coords_crs`
        and `graph_crs` are given, coords are transformed from `coords_crs` to
        `graph_crs`. If any one of them is None, the original coords are used.
    decode_mask: Iterable[bool] or None
        An Iterable of booleans, masking which grid positions should be decoded
        to (included in the m2g subgraph), i.e., which positions should be output.
        Must have the same length as `coords`. If None (default), all grid nodes
        are included.
    return_components: bool, default False
        If True, the function returns a dict with m2g, m2m, and g2m as separate
        graphs. If False, returns one combined graph.

    Returns
    -------
    networkx.DiGraph or dict[networkx.DiGraph]
        The graph or graph components.
    """
    return create_all_graph_components(
        coords=coords,
        m2m_connectivity="hierarchical",
        m2m_connectivity_kwargs=dict(
            mesh_node_distance=mesh_node_distance,
            level_refinement_factor=level_refinement_factor,
            max_num_levels=max_num_levels,
        ),
        g2m_connectivity="within_radius",
        m2g_connectivity="nearest_neighbours",
        g2m_connectivity_kwargs=dict(
            rel_max_dist=0.51,
        ),
        m2g_connectivity_kwargs=dict(
            max_num_neighbours=4,
        ),
        coords_crs=coords_crs,
        graph_crs=graph_crs,
        decode_mask=decode_mask,
        return_components=return_components,
    )
