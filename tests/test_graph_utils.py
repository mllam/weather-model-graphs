import numpy as np

import weather_model_graphs as wmg


def _create_fake_xy(N=10):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xy = np.meshgrid(x, y)
    xy = np.stack(xy, axis=0)
    return xy


def test_graph_splitting():
    xy = _create_fake_xy(N=64)
    graph = wmg.create.archetype.create_oskarsson_hierarchical_graph(xy_grid=xy)

    graph_components = wmg.split_graph_by_edge_attribute(graph=graph, attr="component")

    # split the m2m graph into the different parts that create the up, in-level
    # and down connections respectively
    G_m2m = graph_components["m2m"]

    m2m_graph_components = wmg.split_graph_by_edge_attribute(
        graph=G_m2m, attr="direction"
    )
    assert len(m2m_graph_components) == 3
