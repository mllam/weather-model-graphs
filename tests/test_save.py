import tempfile
from pathlib import Path
from types import SimpleNamespace

import networkx
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


def test_to_pyg_does_not_mutate_node_attributes(monkeypatch, tmp_path):
    graph = networkx.DiGraph()
    graph.add_node(0, pos=[0.0, 0.0], unexported="keep me")
    graph.add_node(1, pos=[1.0, 0.0], unexported="keep me too")
    graph.add_edge(0, 1, len=1.0, vdiff=0.0)

    original_node_attrs = {node: attrs.copy() for node, attrs in graph.nodes(data=True)}
    converted_graphs = []

    class FakeTensor:
        ndim = 1

        def unsqueeze(self, dim):
            return self

        def to(self, dtype):
            return self

    class FakePygGraph:
        edge_index = FakeTensor()

        def __getitem__(self, key):
            return FakeTensor()

    class FakeTorch:
        Tensor = FakeTensor
        float32 = "float32"

        @staticmethod
        def cat(values, dim):
            return FakeTensor()

        @staticmethod
        def save(value, path):
            Path(path).write_text("saved")

    def fake_from_networkx(converted_graph):
        converted_graphs.append(converted_graph)
        return FakePygGraph()

    monkeypatch.setattr(wmg.save, "HAS_PYG", True)
    monkeypatch.setattr(wmg.save, "torch", FakeTorch, raising=False)
    monkeypatch.setattr(
        wmg.save,
        "pyg_convert",
        SimpleNamespace(from_networkx=fake_from_networkx),
        raising=False,
    )

    wmg.save.to_pyg(
        graph=graph,
        output_directory=tmp_path,
        name="graph",
        node_features=["pos"],
    )

    assert {
        node: attrs for node, attrs in graph.nodes(data=True)
    } == original_node_attrs
    assert converted_graphs
    assert all(
        "unexported" not in attrs
        for converted_graph in converted_graphs
        for _, attrs in converted_graph.nodes(data=True)
    )
