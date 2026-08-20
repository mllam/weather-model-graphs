import networkx
import numpy as np
import pytest

from weather_model_graphs.create.mesh.kinds.hierarchical import (
    create_hierarchical_multiscale_mesh_graph,
)
from weather_model_graphs.create.mesh.mesh import create_single_level_2d_mesh_graph

xy_data = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])


def _patch_pos_to_type(monkeypatch, pos_type):
    """Coerce `pos` attributes to mimic backend-returned Python containers."""
    original_digraph = networkx.DiGraph

    def mock_digraph(g=None, *args, **kwargs):
        dg = original_digraph(g, *args, **kwargs)
        if g is not None:
            for node in dg.nodes:
                if "pos" in dg.nodes[node]:
                    dg.nodes[node]["pos"] = pos_type(dg.nodes[node]["pos"])
        return dg

    monkeypatch.setattr(networkx, "DiGraph", mock_digraph)


@pytest.mark.parametrize("pos_type", [tuple, list])
def test_flat_mesh_supports_non_array_positions(monkeypatch, pos_type):
    """Regression test for tuple/list `pos` values seen with alternate backends."""
    _patch_pos_to_type(monkeypatch, pos_type)

    dg = create_single_level_2d_mesh_graph(xy_data, nx=2, ny=2)

    for _, _, data in dg.edges(data=True):
        assert "len" in data
        assert "vdiff" in data
        assert isinstance(data["len"], (float, np.floating))
        assert isinstance(data["vdiff"], (np.ndarray, list, tuple))


def test_flat_mesh_supports_numpy_positions():
    """Sanity check for the default path (numpy positions)."""
    dg = create_single_level_2d_mesh_graph(xy_data, nx=2, ny=2)
    assert len(dg.nodes) > 0


def test_hierarchical_mesh_supports_tuple_positions(monkeypatch):
    """The hierarchical builder should also tolerate tuple-backed `pos` values."""
    _patch_pos_to_type(monkeypatch, tuple)

    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    xx, yy = np.meshgrid(x, y)
    xy_large = np.column_stack([xx.ravel(), yy.ravel()])

    components = create_hierarchical_multiscale_mesh_graph(
        xy_large,
        mesh_node_distance=10.0,
        level_refinement_factor=3.0,
        max_num_levels=3,
    )
    assert len(components.nodes) > 0
