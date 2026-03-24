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
