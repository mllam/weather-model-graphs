"""Tests for the deprecated pyg save path (``wmg.save.to_pyg``).

``to_pyg`` is deprecated in favour of ``to_torch_tensors_on_disk`` (see
``tests/test_save_torch_tensors.py`` for the tests of the new wmg to
neural-lam bridge). These tests are expected to be deleted together with
the deprecated functionality.
"""

import tempfile

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


def test_to_pyg_emits_deprecation_warning():
    """to_pyg is retained for back-compat but must warn that it is deprecated."""
    if not HAS_PYG:
        pytest.skip("weather-model-graphs[pytorch] not installed")
    xy = test_utils.create_fake_xy(N=64)
    graph = wmg.create.archetype.create_keisler_graph(coords=xy)
    g2m_graph = wmg.split_graph_by_edge_attribute(graph=graph, attr="component")["g2m"]
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.warns(DeprecationWarning, match="to_torch_tensors_on_disk"):
            wmg.save.to_pyg(graph=g2m_graph, output_directory=tmpdir, name="g2m")
