import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest

import tests.utils as test_utils
import weather_model_graphs as wmg


def test_plot():
    xy = test_utils.create_fake_xy(10)

    graph = wmg.create.create_all_graph_components(
        m2m_connectivity="flat_multiscale",
        coords=xy,
        m2m_connectivity_kwargs=dict(
            max_num_levels=3,
            mesh_node_distance=2,
            level_refinement_factor=3,
        ),
        g2m_connectivity="nearest_neighbour",
        m2g_connectivity="nearest_neighbour",
    )

    graph_components = wmg.split_graph_by_edge_attribute(graph=graph, attr="component")

    def _is_ndarray(val):
        return isinstance(val, np.ndarray)

    def _is_valid_color_attr(val):
        return isinstance(val, (int, float, str))

    fig, ax = plt.subplots()
    for graph in graph_components.values():
        node_attrs = list(list(graph.nodes(data=True))[0][1].keys())
        edge_attrs = list(list(graph.edges(data=True))[0][2].keys())

        for edge_attr in edge_attrs + []:
            for node_attr in node_attrs + []:
                should_raise = None
                if not _is_valid_color_attr(
                    list(graph.edges(data=True))[0][2][edge_attr]
                ):
                    should_raise = NotImplementedError
                elif not _is_valid_color_attr(
                    list(graph.nodes(data=True))[0][1][node_attr]
                ):
                    should_raise = NotImplementedError

                def fn():
                    wmg.visualise.nx_draw_with_pos_and_attr(
                        graph,
                        ax=ax,
                        edge_color_attr=edge_attr,
                        node_color_attr=node_attr,
                    )

                if should_raise is not None:
                    with pytest.raises(should_raise):
                        fn()
                else:
                    fn()

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        fig.savefig(f.name)


def test_plot_nodes_only_graph_with_edge_color_attr():
    """Plotting a graph with nodes but no edges and edge_color_attr should
    not crash (previously raised IndexError)."""
    import networkx as nx

    G = nx.DiGraph()
    G.add_node(0, pos=np.array([1.0, 2.0]), type="mesh")
    G.add_node(1, pos=np.array([3.0, 4.0]), type="grid")

    # Should silently skip edge colouring, not raise IndexError
    ax = wmg.visualise.nx_draw_with_pos_and_attr(G, edge_color_attr="len")
    assert ax is not None
    plt.close("all")


def test_plot_empty_graph_with_node_color_attr():
    """Plotting a completely empty graph with node_color_attr should not crash."""
    import networkx as nx

    G = nx.DiGraph()

    ax = wmg.visualise.nx_draw_with_pos_and_attr(G, node_color_attr="type")
    assert ax is not None
    plt.close("all")


def test_get_graph_attr_values_raises_on_empty_edges():
    """_get_graph_attr_values should raise a clear ValueError for empty edges."""
    import networkx as nx

    from weather_model_graphs.visualise.plot_2d import _get_graph_attr_values

    G = nx.DiGraph()
    G.add_node(0, pos=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="no edges"):
        _get_graph_attr_values(G, "len", component="edges")


def test_get_graph_attr_values_raises_on_empty_nodes():
    """_get_graph_attr_values should raise a clear ValueError for empty nodes."""
    import networkx as nx

    from weather_model_graphs.visualise.plot_2d import _get_graph_attr_values

    G = nx.DiGraph()

    with pytest.raises(ValueError, match="no nodes"):
        _get_graph_attr_values(G, "type", component="nodes")
