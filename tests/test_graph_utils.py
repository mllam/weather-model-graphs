import tests.utils as test_utils
import weather_model_graphs as wmg


def test_graph_splitting():
    xy = test_utils.create_fake_xy(N=64)
    graph = wmg.create.archetype.create_oskarsson_hierarchical_graph(xy=xy)

    graph_components = wmg.split_graph_by_edge_attribute(graph=graph, attr="component")

    # split the m2m graph into the different parts that create the up, in-level
    # and down connections respectively
    G_m2m = graph_components["m2m"]

    m2m_graph_components = wmg.split_graph_by_edge_attribute(
        graph=G_m2m, attr="direction"
    )
    assert len(m2m_graph_components) == 3
