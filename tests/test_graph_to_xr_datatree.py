import numpy as np
import pytest
import xarray as xr
from loguru import logger

import tests.utils as test_utils
import weather_model_graphs as wmg
from weather_model_graphs.load import collect_datasets


@pytest.mark.parametrize("kind", ["graphcast", "keisler", "oskarsson_hierarchical"])
def test_create_graph_archetype(kind):
    xy = test_utils.create_fake_xy(N=64)
    fn_name = f"create_{kind}_graph"
    fn = getattr(wmg.create.archetype, fn_name)

    graph = fn(coords=xy)

    split_edges_by = wmg.split.DEFAULT_EDGE_SPLITS[kind]
    split_nodes_by = wmg.split.DEFAULT_NODE_SPLITS[kind]

    dt = wmg.save.graph_to_datatree(
        graph=graph, split_edges_by=split_edges_by, split_nodes_by=split_nodes_by
    )

    # check that we can merge all the datasets across the datatree that
    # represents all the subgraphs. This is only possible if the edge-indexes
    # are indexes are unique across the datasets
    ds_global = xr.merge(collect_datasets(dt))

    # check that the merged dataset has the same number of edges as the original graph
    assert ds_global.sizes["edge_index"] == len(graph.edges)

    graph_reconstructed = wmg.load.datatree_to_graph(dt)

    # check that the reconstructed graph has the same number of edges as the original graph
    assert len(graph_reconstructed.edges) == len(graph.edges)

    # check that the edges of the reconstructed graph are the same as the original graph
    for edge in graph.edges:
        assert edge in graph_reconstructed.edges

    # check that the edge attributes of the reconstructed graph are the same as the original graph
    for edge in graph.edges:
        for edge_feature in wmg.save.DEFAULT_EDGE_FEATURES:
            np.testing.assert_equal(
                graph.edges[edge].get(edge_feature),
                graph_reconstructed.edges[edge].get(edge_feature),
            )

    # check that the node attributes of the reconstructed graph are the same as the original graph
    for node in graph.nodes:
        for node_feature in wmg.save.DEFAULT_NODE_FEATURES:
            np.testing.assert_equal(
                graph.nodes[node].get(node_feature),
                graph_reconstructed.nodes[node].get(node_feature),
            )

    import ipdb

    ipdb.set_trace()


@logger.catch(reraise=True)
def main():
    for kind in ["oskarsson_hierarchical", "keisler", "graphcast"]:
        logger.info(f"Testing {kind} graph")
        test_create_graph_archetype(kind)


if __name__ == "__main__":
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        main()
