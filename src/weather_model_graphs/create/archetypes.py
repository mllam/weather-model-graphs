from . import create_all_graph_components


def create_keissler_graph(xy_grid, refinement_factor=3, merge_components=True):
    return create_all_graph_components(
        xy=xy_grid,
        merge_components=merge_components,
        m2m_connectivity="flat",
        m2m_connectivity_kwargs=dict(refinement_factor=refinement_factor),
        m2g_connectivity="nearest_neighbour",
        g2m_connectivity="nearest_neighbours",
        g2m_connectivity_kwargs=dict(
            max_num_neighbours=4,
        )
    )

def create_graphcast_graph(xy_grid, refinement_factor=2, max_num_levels=None, merge_components=True):
    return create_all_graph_components(
        xy=xy_grid,
        merge_components=merge_components,
        m2m_connectivity="flat_multiscale",
        m2m_connectivity_kwargs=dict(refinement_factor=refinement_factor, max_num_levels=max_num_levels),
        m2g_connectivity="nearest_neighbour",
        g2m_connectivity="nearest_neighbours",
        g2m_connectivity_kwargs=dict(
            max_num_neighbours=4,
        )
    )

def create_oscarsson_hierarchical_graph(xy_grid, merge_components=True):
    return create_all_graph_components(
        xy=xy_grid,
        merge_components=merge_components,
        m2m_connectivity="hierarchical",
        m2m_connectivity_kwargs=dict(refinement_factor=2, max_num_levels=3),
        m2g_connectivity="nearest_neighbour",
        g2m_connectivity="nearest_neighbours",
        g2m_connectivity_kwargs=dict(
            max_num_neighbours=4,
        )
    )
