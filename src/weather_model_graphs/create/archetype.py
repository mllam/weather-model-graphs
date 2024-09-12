from .base import create_all_graph_components


def create_keisler_graph(xy_grid, grid_refinement_factor=3):
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
    xy_grid: np.ndarray
        2D array of grid point positions.
    grid_refinement_factor: float
        Refinement factor between grid points and mesh

    Returns
    -------
    networkx.DiGraph or dict[networkx.DiGraph]
        The graph or graph components.
    """
    return create_all_graph_components(
        xy=xy_grid,
        m2m_connectivity="flat",
        m2m_connectivity_kwargs=dict(grid_refinement_factor=grid_refinement_factor),
        g2m_connectivity="within_radius",
        m2g_connectivity="nearest_neighbours",
        g2m_connectivity_kwargs=dict(
            rel_max_dist=0.51,
        ),
        m2g_connectivity_kwargs=dict(
            max_num_neighbours=4,
        ),
    )


def create_graphcast_graph(
    xy_grid, grid_refinement_factor=3, level_refinement_factor=3, max_num_levels=None
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
    xy_grid: np.ndarray
        2D array of grid point positions.
    grid_refinement_factor: float
        Refinement factor between grid points and bottom level of mesh hierarchy
    level_refinement_factor: int
        Refinement factor between grid points and bottom level of mesh hierarchy
        NOTE: Must be an odd integer >1 to create proper multiscale graph
    max_num_levels: int
        The number of levels of longer-range connections in the mesh graph.

    Returns
    -------
    networkx.DiGraph or dict[networkx.DiGraph]
        The graph or graph components.
    """
    return create_all_graph_components(
        xy=xy_grid,
        m2m_connectivity="flat_multiscale",
        m2m_connectivity_kwargs=dict(
            grid_refinement_factor=grid_refinement_factor,
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
    )


def create_oskarsson_hierarchical_graph(
    xy_grid, grid_refinement_factor=3, level_refinement_factor=3, max_num_levels=None
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
    xy_grid: np.ndarray
        2D array of grid point positions.
    grid_refinement_factor: float
        Refinement factor between grid points and bottom level of mesh hierarchy
    level_refinement_factor: float
        Refinement factor between grid points and bottom level of mesh hierarchy

    Returns
    -------
    networkx.DiGraph or dict[networkx.DiGraph]
        The graph or graph components.
    """
    return create_all_graph_components(
        xy=xy_grid,
        m2m_connectivity="hierarchical",
        m2m_connectivity_kwargs=dict(
            grid_refinement_factor=grid_refinement_factor,
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
    )
