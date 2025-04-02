import pytest
import xarray as xr

import tests.utils as test_utils
import weather_model_graphs as wmg

DEFAULT_GRAPH_SPLITS = {
    "keisler": "component",
    "graphcast": "component",
    "oskarsson_hierarchical": {
        "component": {
            "m2m": {"direction": {"same": "level", "up": "levels", "down": "levels"}},
        }
    },
}


def collect_datasets(tree):
    datasets = []
    if tree.ds is not None:
        datasets.append(tree.ds)
    for child in tree.children.values():
        datasets.extend(collect_datasets(child))
    return datasets


@pytest.mark.parametrize("kind", ["graphcast", "keisler", "oskarsson_hierarchical"])
def test_create_graph_archetype(kind):
    xy = test_utils.create_fake_xy(N=64)
    fn_name = f"create_{kind}_graph"
    fn = getattr(wmg.create.archetype, fn_name)

    graph = fn(coords=xy)

    split_by = DEFAULT_GRAPH_SPLITS[kind]

    dt = wmg.save.graph_to_datatree(graph=graph, split_by=split_by)

    # check that we can merge all the datasets across the datatree that
    # represents all the subgraphs. This is only possible if the edge-indexes
    # are indexes are unique across the datasets
    ds_global = xr.merge(collect_datasets(dt))

    # check that the merged dataset has the same number of edges as the original graph
    assert ds_global.dims["edge_index"] == len(graph.edges)

    # TODO: add test for the graph having the same edges and edge features as the original graph


if __name__ == "__main__":
    test_create_graph_archetype("keisler")
