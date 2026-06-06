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
    reloading without structural or data loss.
    """
    xy = test_utils.create_fake_xy(N=64)
    graph = wmg.create.archetype.create_oskarsson_hierarchical_graph(coords=xy)

    # Add a custom attribute to ensure it's preserved
    graph.graph["test_attr"] = "preserved"

    with tempfile.TemporaryDirectory() as tmpdir:
        name = "test_pickle_graph"
        wmg.save.to_pickle(graph=graph, output_directory=tmpdir, name=name)

        expected_path = Path(tmpdir) / f"{name}.pickle"
        assert expected_path.exists()

        with open(expected_path, "rb") as f:
            loaded_graph = pickle.load(f)

        # 1. Structural check
        assert isinstance(loaded_graph, nx.DiGraph)
        assert len(loaded_graph.nodes) == len(graph.nodes)
        assert len(loaded_graph.edges) == len(graph.edges)

        # 2. Graph attribute check
        assert loaded_graph.graph["test_attr"] == "preserved"

        # 3. Node attribute check: verify all attributes are preserved for a sample
        for node in list(graph.nodes)[:10]:
            for attr, value in graph.nodes[node].items():
                assert attr in loaded_graph.nodes[node]
                if isinstance(value, (int, float, str)):
                    assert loaded_graph.nodes[node][attr] == value
                # For numpy arrays (like 'pos'), use .all()
                elif hasattr(value, "all"):
                    assert (loaded_graph.nodes[node][attr] == value).all()

        # 4. Edge attribute check: verify all attributes are preserved for a sample
        for edge in list(graph.edges)[:10]:
            for attr, value in graph.edges[edge].items():
                assert attr in loaded_graph.edges[edge]
                if isinstance(value, (int, float, str)):
                    assert loaded_graph.edges[edge][attr] == value
                elif hasattr(value, "all"):
                    assert (loaded_graph.edges[edge][attr] == value).all()
