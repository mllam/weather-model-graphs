import pickle
import tempfile
from pathlib import Path

import networkx as nx

import pytest
from loguru import logger

import tests.utils as test_utils
import weather_model_graphs as wmg
from weather_model_graphs.save import HAS_PYG


@pytest.mark.parametrize("list_from_attribute", [None, "level"])
def test_save_to_pyg(list_from_attribute):
    if not HAS_PYG:
        logger.warning(
            "Skipping test_save_to_pyg because weather-model-graphs[pytorch] is not installed."
        )
        return

    xy = test_utils.create_fake_xy(N=64)
    graph = wmg.create.archetype.create_oskarsson_hierarchical_graph(coords=xy)

    graph_components = wmg.split_graph_by_edge_attribute(graph=graph, attr="component")

    # split the m2m graph into the different parts that create the up, in-level and down connections respectively
    # this is how the graphs is stored in the neural-lam codebase
    m2m_graph = graph_components.pop("m2m")
    m2m_graph_components = wmg.split_graph_by_edge_attribute(
        graph=m2m_graph, attr="direction"
    )
    m2m_graph_components = {
        f"m2m_{name}": graph for name, graph in m2m_graph_components.items()
    }
    graph_components.update(m2m_graph_components)

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, graph in graph_components.items():
            wmg.save.to_pyg(
                graph=graph,
                output_directory=tmpdir,
                name=name,
                list_from_attribute=list_from_attribute,
            )


def test_save_to_pickle():
    """
    Test that to_pickle safely exports a graph to disk and allows
    reloading without structural loss.
    """
    xy = test_utils.create_fake_xy(N=64)
    graph = wmg.create.archetype.create_oskarsson_hierarchical_graph(coords=xy)

    with tempfile.TemporaryDirectory() as tmpdir:
        name = "test_pickle_graph"
        wmg.save.to_pickle(graph=graph, output_directory=tmpdir, name=name)

        expected_path = Path(tmpdir) / f"{name}.pickle"
        assert expected_path.exists()

        with open(expected_path, "rb") as f:
            loaded_graph = pickle.load(f)

        assert isinstance(loaded_graph, nx.DiGraph)
        assert len(loaded_graph.nodes) == len(graph.nodes)
        assert len(loaded_graph.edges) == len(graph.edges)
