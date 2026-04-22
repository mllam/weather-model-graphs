# tests/test_connect_nodes_vectorized.py
import numpy as np
import pytest

import tests.utils as test_utils
import weather_model_graphs as wmg
from weather_model_graphs.create.base import connect_nodes_across_graphs


def _make_source_target():
    """Small deterministic mesh source + grid target for fast tests."""
    xy = test_utils.create_fake_xy(N=10)
    source = wmg.create.mesh.create_single_level_2d_mesh_graph(xy=xy, nx=5, ny=5)
    target = wmg.create.grid.create_grid_graph_nodes(xy)
    return source, target


def _assert_edge_attrs_valid(G, G_source, G_target):
    """Common invariants for all connection methods."""
    assert set(G.nodes) == set(G_source.nodes) | set(G_target.nodes)

    for u, v, data in G.edges(data=True):
        assert u in G_source.nodes, f"Edge source {u} not in G_source"
        assert v in G_target.nodes, f"Edge target {v} not in G_target"
        assert "len" in data, "Missing 'len' attribute"
        assert "vdiff" in data, "Missing 'vdiff' attribute"

        pos_u = np.array(G.nodes[u]["pos"])
        pos_v = np.array(G.nodes[v]["pos"])
        expected_vdiff = pos_u - pos_v
        expected_len = np.linalg.norm(expected_vdiff)

        assert np.isclose(data["len"], expected_len, atol=1e-10), (
            f"Edge ({u},{v}): len={data['len']}, expected={expected_len}"
        )
        assert np.allclose(data["vdiff"], expected_vdiff, atol=1e-10), (
            f"Edge ({u},{v}): vdiff={data['vdiff']}, expected={expected_vdiff}"
        )


def test_nearest_neighbour_one_edge_per_target():
    source, target = _make_source_target()
    G = connect_nodes_across_graphs(source, target, method="nearest_neighbour")

    for node in target.nodes:
        preds = list(G.predecessors(node))
        assert len(preds) == 1, f"Target {node} has {len(preds)} predecessors, expected 1"
        assert preds[0] in source.nodes

    _assert_edge_attrs_valid(G, source, target)


@pytest.mark.parametrize("k", [1, 3, 4, 8])
def test_nearest_neighbours_at_most_k_per_target(k):
    source, target = _make_source_target()
    G = connect_nodes_across_graphs(
        source, target, method="nearest_neighbours", max_num_neighbours=k
    )

    for node in target.nodes:
        preds = list(G.predecessors(node))
        # source has 25 nodes >= max k=8, so every target always gets exactly k neighbours
        assert len(preds) == k, (
            f"Target {node} has {len(preds)} predecessors, expected exactly {k}"
        )
        for p in preds:
            assert p in source.nodes

    _assert_edge_attrs_valid(G, source, target)


@pytest.mark.parametrize("max_dist", [1.5, 3.0, 6.0])
def test_within_radius_all_edges_within_dist(max_dist):
    source, target = _make_source_target()
    G = connect_nodes_across_graphs(
        source, target, method="within_radius", max_dist=max_dist
    )
    assert G.number_of_edges() > 0, f"Expected edges with max_dist={max_dist}, got 0"

    for u, v, data in G.edges(data=True):
        assert data["len"] <= max_dist + 1e-10, (
            f"Edge ({u},{v}) len={data['len']} exceeds max_dist={max_dist}"
        )

    _assert_edge_attrs_valid(G, source, target)


def test_within_radius_rel_max_dist():
    source, target = _make_source_target()
    G = connect_nodes_across_graphs(
        source, target, method="within_radius", rel_max_dist=1.0
    )
    _assert_edge_attrs_valid(G, source, target)


def test_containing_rectangle_nodes_preserved():
    source, target = _make_source_target()
    G = connect_nodes_across_graphs(source, target, method="containing_rectangle")
    _assert_edge_attrs_valid(G, source, target)
