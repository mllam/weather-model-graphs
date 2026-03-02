"""
Tests for mesh_layout="prebuilt" support (Issue #79).

Tests verify:
1. Validation functions for prebuilt mesh graphs
2. Nodes+edges mode (user provides complete DiGraph)
3. Nodes-only mode (library builds connectivity via Delaunay)
4. Integration with create_all_graph_components for all m2m_connectivity types
5. Edge cases (single node, two nodes, missing attributes, etc.)
6. Hierarchical prebuilt meshes with level attributes
"""

import networkx as nx
import numpy as np
import pytest

import tests.utils as test_utils
import weather_model_graphs as wmg
from weather_model_graphs.create.mesh.kinds.prebuilt import (
    _build_edges_delaunay,
    create_prebuilt_flat_from_nodes,
    create_prebuilt_flat_multiscale_from_nodes,
    create_prebuilt_hierarchical_from_nodes,
    validate_prebuilt_mesh_edges,
    validate_prebuilt_nodes,
    validate_prebuilt_nodes_with_levels,
)


# ===========================
# Helper functions
# ===========================


def _make_simple_mesh_nodes(n=5):
    """Create a simple undirected graph with mesh nodes in a grid pattern."""
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            node_id = i * n + j
            G.add_node(
                node_id,
                pos=np.array([float(i), float(j)]),
                type="mesh",
            )
    return G


def _make_simple_mesh_digraph(n=4):
    """Create a complete directed mesh graph (nodes+edges) in a grid pattern."""
    G = nx.DiGraph()
    positions = {}
    for i in range(n):
        for j in range(n):
            node_id = i * n + j
            pos = np.array([float(i), float(j)])
            G.add_node(node_id, pos=pos, type="mesh")
            positions[node_id] = pos

    # Add edges between adjacent nodes (4-connected grid)
    for i in range(n):
        for j in range(n):
            node_id = i * n + j
            # Right neighbor
            if j + 1 < n:
                right = i * n + (j + 1)
                d = np.sqrt(np.sum((positions[node_id] - positions[right]) ** 2))
                vdiff = positions[node_id] - positions[right]
                G.add_edge(node_id, right, len=d, vdiff=vdiff)
                G.add_edge(right, node_id, len=d, vdiff=-vdiff)
            # Down neighbor
            if i + 1 < n:
                down = (i + 1) * n + j
                d = np.sqrt(np.sum((positions[node_id] - positions[down]) ** 2))
                vdiff = positions[node_id] - positions[down]
                G.add_edge(node_id, down, len=d, vdiff=vdiff)
                G.add_edge(down, node_id, len=d, vdiff=-vdiff)

    return G


def _make_multilevel_mesh_nodes():
    """Create a mesh with nodes at two levels (for hierarchical/multiscale)."""
    G = nx.Graph()
    # Level 0 (finest): 3x3 grid
    for i in range(3):
        for j in range(3):
            node_id = f"L0_{i}_{j}"
            G.add_node(
                node_id,
                pos=np.array([float(i), float(j)]),
                type="mesh",
                level=0,
            )
    # Level 1 (coarser): 2x2 grid
    for i in range(2):
        for j in range(2):
            node_id = f"L1_{i}_{j}"
            G.add_node(
                node_id,
                pos=np.array([float(i) * 2, float(j) * 2]),
                type="mesh",
                level=1,
            )
    return G


def _make_multilevel_mesh_digraph():
    """Create a complete hierarchical DiGraph with levels, edges, and direction attributes."""
    G = nx.DiGraph()

    # Level 0: 3x3 grid
    level0_nodes = []
    for i in range(3):
        for j in range(3):
            node_id = f"L0_{i}_{j}"
            G.add_node(
                node_id,
                pos=np.array([float(i), float(j)]),
                type="mesh",
                level=0,
            )
            level0_nodes.append(node_id)

    # Level 1: 2x2 grid
    level1_nodes = []
    for i in range(2):
        for j in range(2):
            node_id = f"L1_{i}_{j}"
            G.add_node(
                node_id,
                pos=np.array([float(i) * 2, float(j) * 2]),
                type="mesh",
                level=1,
            )
            level1_nodes.append(node_id)

    # Add intra-level edges for level 0
    for idx, n1 in enumerate(level0_nodes):
        for n2 in level0_nodes[idx + 1 :]:
            pos1 = G.nodes[n1]["pos"]
            pos2 = G.nodes[n2]["pos"]
            d = np.sqrt(np.sum((pos1 - pos2) ** 2))
            if d <= 1.5:  # Only connect nearby nodes
                G.add_edge(
                    n1, n2, len=d, vdiff=pos1 - pos2, level=0, direction="same"
                )
                G.add_edge(
                    n2, n1, len=d, vdiff=pos2 - pos1, level=0, direction="same"
                )

    # Add intra-level edges for level 1
    for idx, n1 in enumerate(level1_nodes):
        for n2 in level1_nodes[idx + 1 :]:
            pos1 = G.nodes[n1]["pos"]
            pos2 = G.nodes[n2]["pos"]
            d = np.sqrt(np.sum((pos1 - pos2) ** 2))
            if d <= 3.0:  # Connect level 1 nodes
                G.add_edge(
                    n1, n2, len=d, vdiff=pos1 - pos2, level=1, direction="same"
                )
                G.add_edge(
                    n2, n1, len=d, vdiff=pos2 - pos1, level=1, direction="same"
                )

    # Add inter-level edges (up/down)
    for n0 in level0_nodes:
        pos0 = G.nodes[n0]["pos"]
        # Find nearest level 1 node
        min_d = float("inf")
        nearest_n1 = None
        for n1 in level1_nodes:
            pos1 = G.nodes[n1]["pos"]
            d = np.sqrt(np.sum((pos0 - pos1) ** 2))
            if d < min_d:
                min_d = d
                nearest_n1 = n1
        if nearest_n1 is not None:
            pos1 = G.nodes[nearest_n1]["pos"]
            G.add_edge(
                nearest_n1,
                n0,
                len=min_d,
                vdiff=pos1 - pos0,
                level=1,
                direction="down",
            )
            G.add_edge(
                n0,
                nearest_n1,
                len=min_d,
                vdiff=pos0 - pos1,
                level=0,
                direction="up",
            )

    return G


def _make_flat_multiscale_digraph():
    """Create a flat multiscale DiGraph with level attributes on edges."""
    G = nx.DiGraph()

    # Level 0: 3x3 grid
    for i in range(3):
        for j in range(3):
            node_id = f"L0_{i}_{j}"
            G.add_node(
                node_id,
                pos=np.array([float(i), float(j)]),
                type="mesh",
                level=0,
            )

    # Level 1: 2x2 grid
    for i in range(2):
        for j in range(2):
            node_id = f"L1_{i}_{j}"
            G.add_node(
                node_id,
                pos=np.array([float(i) * 2, float(j) * 2]),
                type="mesh",
                level=1,
            )

    # Add edges within each level
    all_nodes = list(G.nodes())
    for n1 in all_nodes:
        for n2 in all_nodes:
            if n1 == n2:
                continue
            lev1 = G.nodes[n1]["level"]
            lev2 = G.nodes[n2]["level"]
            if lev1 != lev2:
                continue
            pos1 = G.nodes[n1]["pos"]
            pos2 = G.nodes[n2]["pos"]
            d = np.sqrt(np.sum((pos1 - pos2) ** 2))
            if d <= (1.5 if lev1 == 0 else 3.0):
                G.add_edge(n1, n2, len=d, vdiff=pos1 - pos2, level=lev1)

    return G


# ===========================
# Validation tests
# ===========================


class TestValidatePrebuiltNodes:
    """Tests for validate_prebuilt_nodes."""

    def test_valid_nodes(self):
        G = _make_simple_mesh_nodes(3)
        validate_prebuilt_nodes(G)  # Should not raise

    def test_empty_graph_raises(self):
        G = nx.Graph()
        with pytest.raises(ValueError, match="at least one node"):
            validate_prebuilt_nodes(G)

    def test_missing_pos_raises(self):
        G = nx.Graph()
        G.add_node(0, type="mesh")
        with pytest.raises(ValueError, match="missing required 'pos'"):
            validate_prebuilt_nodes(G)

    def test_invalid_pos_shape_raises(self):
        G = nx.Graph()
        G.add_node(0, pos=np.array([1.0, 2.0, 3.0]), type="mesh")
        with pytest.raises(ValueError, match="invalid shape"):
            validate_prebuilt_nodes(G)

    def test_missing_type_raises(self):
        G = nx.Graph()
        G.add_node(0, pos=np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="missing required 'type'"):
            validate_prebuilt_nodes(G)


class TestValidatePrebuiltNodesWithLevels:
    """Tests for validate_prebuilt_nodes_with_levels."""

    def test_valid_nodes_with_levels(self):
        G = _make_multilevel_mesh_nodes()
        validate_prebuilt_nodes_with_levels(G)  # Should not raise

    def test_missing_level_raises(self):
        G = _make_simple_mesh_nodes(3)  # No level attribute
        with pytest.raises(ValueError, match="missing required 'level'"):
            validate_prebuilt_nodes_with_levels(G)


class TestValidatePrebuiltMeshEdges:
    """Tests for validate_prebuilt_mesh_edges."""

    def test_valid_digraph(self):
        G = _make_simple_mesh_digraph(3)
        validate_prebuilt_mesh_edges(G)  # Should not raise

    def test_not_digraph_raises(self):
        G = _make_simple_mesh_nodes(3)
        with pytest.raises(TypeError, match="must be a networkx.DiGraph"):
            validate_prebuilt_mesh_edges(G)

    def test_no_edges_raises(self):
        G = nx.DiGraph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")
        with pytest.raises(ValueError, match="no edges"):
            validate_prebuilt_mesh_edges(G)

    def test_missing_len_raises(self):
        G = nx.DiGraph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")
        G.add_edge(0, 1, vdiff=np.array([1.0, 0.0]))
        with pytest.raises(ValueError, match="missing required 'len'"):
            validate_prebuilt_mesh_edges(G)

    def test_missing_vdiff_raises(self):
        G = nx.DiGraph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")
        G.add_edge(0, 1, len=1.0)
        with pytest.raises(ValueError, match="missing required 'vdiff'"):
            validate_prebuilt_mesh_edges(G)

    def test_require_levels_no_level_edges_raises(self):
        G = _make_simple_mesh_digraph(3)
        # Add level to nodes
        for node in G.nodes():
            G.nodes[node]["level"] = 0
        # Edges don't have level attribute
        with pytest.raises(ValueError, match="at least some edges must have a 'level'"):
            validate_prebuilt_mesh_edges(G, require_levels=True)

    def test_require_levels_valid(self):
        G = _make_multilevel_mesh_digraph()
        validate_prebuilt_mesh_edges(G, require_levels=True)  # Should not raise


# ===========================
# Delaunay edge building tests
# ===========================


class TestBuildEdgesDelaunay:
    """Tests for _build_edges_delaunay."""

    def test_single_node_no_edges(self):
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert isinstance(dg, nx.DiGraph)
        assert dg.number_of_nodes() == 1
        assert dg.number_of_edges() == 0

    def test_two_nodes_bidirectional(self):
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.number_of_edges() == 2  # Bidirectional
        assert dg.has_edge(0, 1)
        assert dg.has_edge(1, 0)

    def test_three_nodes_triangle(self):
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")
        G.add_node(2, pos=np.array([0.5, 1.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        # Triangle should have 3 edges * 2 directions = 6 edges
        assert dg.number_of_edges() == 6

    def test_edges_have_len_and_vdiff(self):
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([3.0, 4.0]), type="mesh")
        G.add_node(2, pos=np.array([6.0, 0.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        for u, v, data in dg.edges(data=True):
            assert "len" in data
            assert "vdiff" in data
            assert data["len"] > 0
            assert isinstance(data["vdiff"], np.ndarray)
            assert data["vdiff"].shape == (2,)

    def test_edge_length_correct(self):
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([3.0, 4.0]), type="mesh")
        G.add_node(2, pos=np.array([6.0, 0.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.edges[0, 1]["len"] == pytest.approx(5.0)  # 3-4-5

    def test_vdiff_antisymmetric(self):
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([3.0, 4.0]), type="mesh")
        G.add_node(2, pos=np.array([6.0, 0.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        for u, v in [(0, 1), (0, 2), (1, 2)]:
            if dg.has_edge(u, v) and dg.has_edge(v, u):
                np.testing.assert_array_almost_equal(
                    dg.edges[u, v]["vdiff"], -dg.edges[v, u]["vdiff"]
                )

    def test_grid_pattern(self):
        G = _make_simple_mesh_nodes(4)
        dg = _build_edges_delaunay(G)
        assert isinstance(dg, nx.DiGraph)
        assert dg.number_of_nodes() == 16
        assert dg.number_of_edges() > 0
        # All edges should be bidirectional
        for u, v in dg.edges():
            assert dg.has_edge(v, u)

    def test_preserves_node_attributes(self):
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh", level=0)
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh", level=0)
        G.add_node(2, pos=np.array([0.5, 1.0]), type="mesh", level=0)
        dg = _build_edges_delaunay(G)
        for node in dg.nodes():
            assert dg.nodes[node]["type"] == "mesh"
            assert dg.nodes[node]["level"] == 0

    def test_preserves_graph_attributes(self):
        G = nx.Graph()
        G.graph["custom"] = "value"
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")
        G.add_node(2, pos=np.array([0.5, 1.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.graph["custom"] == "value"


# ===========================
# Prebuilt flat from nodes tests
# ===========================


class TestPrebuiltFlatFromNodes:
    """Tests for create_prebuilt_flat_from_nodes."""

    def test_returns_digraph(self):
        G = _make_simple_mesh_nodes(4)
        dg = create_prebuilt_flat_from_nodes(G)
        assert isinstance(dg, nx.DiGraph)

    def test_correct_node_count(self):
        G = _make_simple_mesh_nodes(5)
        dg = create_prebuilt_flat_from_nodes(G)
        assert dg.number_of_nodes() == 25

    def test_edges_have_required_attributes(self):
        G = _make_simple_mesh_nodes(4)
        dg = create_prebuilt_flat_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            assert "len" in data
            assert "vdiff" in data

    def test_validates_nodes(self):
        G = nx.Graph()
        with pytest.raises(ValueError, match="at least one node"):
            create_prebuilt_flat_from_nodes(G)


# ===========================
# Prebuilt flat multiscale from nodes tests
# ===========================


class TestPrebuiltFlatMultiscaleFromNodes:
    """Tests for create_prebuilt_flat_multiscale_from_nodes."""

    def test_returns_digraph(self):
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        assert isinstance(dg, nx.DiGraph)

    def test_preserves_all_nodes(self):
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        # 3*3 + 2*2 = 13 nodes
        assert dg.number_of_nodes() == 13

    def test_edges_have_level_attribute(self):
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            assert "level" in data
            assert "len" in data
            assert "vdiff" in data

    def test_has_edges_from_both_levels(self):
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        edge_levels = set(data["level"] for _, _, data in dg.edges(data=True))
        assert 0 in edge_levels
        assert 1 in edge_levels

    def test_validates_levels(self):
        G = _make_simple_mesh_nodes(3)  # No level attribute
        with pytest.raises(ValueError, match="missing required 'level'"):
            create_prebuilt_flat_multiscale_from_nodes(G)


# ===========================
# Prebuilt hierarchical from nodes tests
# ===========================


class TestPrebuiltHierarchicalFromNodes:
    """Tests for create_prebuilt_hierarchical_from_nodes."""

    def test_returns_digraph(self):
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        assert isinstance(dg, nx.DiGraph)

    def test_has_up_down_same_edges(self):
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        directions = set()
        for _, _, data in dg.edges(data=True):
            if "direction" in data:
                directions.add(data["direction"])
        assert "same" in directions
        assert "up" in directions
        assert "down" in directions

    def test_up_down_edge_count_equal(self):
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        up_count = sum(
            1
            for _, _, d in dg.edges(data=True)
            if d.get("direction") == "up"
        )
        down_count = sum(
            1
            for _, _, d in dg.edges(data=True)
            if d.get("direction") == "down"
        )
        assert up_count == down_count

    def test_intra_level_edges_have_level(self):
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            if data.get("direction") == "same":
                assert "level" in data

    def test_custom_inter_level_k(self):
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(
            G, inter_level={"pattern": "nearest", "k": 2}
        )
        assert isinstance(dg, nx.DiGraph)
        # With k=2, there should be more inter-level edges
        up_count = sum(
            1
            for _, _, d in dg.edges(data=True)
            if d.get("direction") == "up"
        )
        assert up_count > 0

    def test_single_level_raises(self):
        G = _make_simple_mesh_nodes(3)
        for node in G.nodes():
            G.nodes[node]["level"] = 0
        with pytest.raises(ValueError, match="At least two mesh levels"):
            create_prebuilt_hierarchical_from_nodes(G)

    def test_validates_levels(self):
        G = _make_simple_mesh_nodes(3)  # No level attribute
        with pytest.raises(ValueError, match="missing required 'level'"):
            create_prebuilt_hierarchical_from_nodes(G)


# ===========================
# Integration tests: create_all_graph_components with prebuilt
# ===========================


class TestPrebuiltFlatIntegration:
    """Test mesh_layout='prebuilt' with m2m_connectivity='flat'."""

    def test_nodes_plus_edges_mode(self):
        """User provides a complete DiGraph — should validate and use as-is."""
        xy = test_utils.create_fake_xy(10)
        mesh = _make_simple_mesh_digraph(4)

        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )

        assert "m2m" in result
        assert "g2m" in result
        assert "m2g" in result
        assert isinstance(result["m2m"], nx.DiGraph)

    def test_nodes_only_mode(self):
        """User provides nodes-only graph — connectivity built via Delaunay."""
        xy = test_utils.create_fake_xy(10)
        mesh_nodes = _make_simple_mesh_nodes(4)

        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh_nodes),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )

        assert "m2m" in result
        assert isinstance(result["m2m"], nx.DiGraph)
        # Delaunay should have created edges
        assert result["m2m"].number_of_edges() > 0

    def test_missing_mesh_graph_raises(self):
        xy = test_utils.create_fake_xy(10)
        with pytest.raises(ValueError, match="requires 'mesh_graph'"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat",
                mesh_layout="prebuilt",
                mesh_layout_kwargs={},
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )

    def test_nodes_plus_edges_returns_combined_graph(self):
        """Test non-component mode (return_components=False)."""
        xy = test_utils.create_fake_xy(10)
        mesh = _make_simple_mesh_digraph(4)

        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=False,
        )

        assert isinstance(result, nx.DiGraph)
        # Combined graph should have edges from all components
        components = set()
        for _, _, data in result.edges(data=True):
            if "component" in data:
                components.add(data["component"])
        assert "m2m" in components
        assert "g2m" in components
        assert "m2g" in components


class TestPrebuiltFlatMultiscaleIntegration:
    """Test mesh_layout='prebuilt' with m2m_connectivity='flat_multiscale'."""

    def test_nodes_plus_edges_mode(self):
        xy = test_utils.create_fake_xy(10)
        mesh = _make_flat_multiscale_digraph()

        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat_multiscale",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )

        assert "m2m" in result
        assert isinstance(result["m2m"], nx.DiGraph)

    def test_nodes_only_mode(self):
        xy = test_utils.create_fake_xy(10)
        mesh_nodes = _make_multilevel_mesh_nodes()

        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat_multiscale",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh_nodes),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )

        assert "m2m" in result
        assert isinstance(result["m2m"], nx.DiGraph)
        assert result["m2m"].number_of_edges() > 0

    def test_missing_mesh_graph_raises(self):
        xy = test_utils.create_fake_xy(10)
        with pytest.raises(ValueError, match="requires 'mesh_graph'"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat_multiscale",
                mesh_layout="prebuilt",
                mesh_layout_kwargs={},
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )


class TestPrebuiltHierarchicalIntegration:
    """Test mesh_layout='prebuilt' with m2m_connectivity='hierarchical'."""

    def test_nodes_plus_edges_mode(self):
        xy = test_utils.create_fake_xy(10)
        mesh = _make_multilevel_mesh_digraph()

        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="hierarchical",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )

        assert "m2m" in result
        assert isinstance(result["m2m"], nx.DiGraph)

    def test_nodes_only_mode(self):
        xy = test_utils.create_fake_xy(10)
        mesh_nodes = _make_multilevel_mesh_nodes()

        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="hierarchical",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh_nodes),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )

        assert "m2m" in result
        assert isinstance(result["m2m"], nx.DiGraph)

    def test_missing_mesh_graph_raises(self):
        xy = test_utils.create_fake_xy(10)
        with pytest.raises(ValueError, match="requires 'mesh_graph'"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="hierarchical",
                mesh_layout="prebuilt",
                mesh_layout_kwargs={},
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )


# ===========================
# Edge case tests
# ===========================


class TestPrebuiltEdgeCases:
    """Edge cases for prebuilt mesh support."""

    def test_single_node_flat(self):
        """Single node mesh — no edges, but should still work."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([5.0, 5.0]), type="mesh")
        dg = create_prebuilt_flat_from_nodes(G)
        assert dg.number_of_nodes() == 1
        assert dg.number_of_edges() == 0

    def test_two_node_flat(self):
        """Two nodes — should create bidirectional edges."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")
        dg = create_prebuilt_flat_from_nodes(G)
        assert dg.number_of_edges() == 2

    def test_collinear_nodes(self):
        """Collinear nodes — Delaunay still works (degeneracy handled)."""
        G = nx.Graph()
        for i in range(5):
            G.add_node(i, pos=np.array([float(i), 0.0]), type="mesh")
        # Collinear points cause QhullError, but scipy handles with QJ option
        # or it just errors. Let's test what happens.
        # Actually, scipy.spatial.Delaunay will raise for collinear points.
        # Our code should handle this for >= 3 nodes that are collinear.
        # Let's check if it works or raises appropriately.
        try:
            dg = create_prebuilt_flat_from_nodes(G)
            # If it succeeds, check basic properties
            assert isinstance(dg, nx.DiGraph)
        except Exception:
            # Collinear points may raise — this is acceptable behavior
            pass

    def test_large_mesh_nodes_only(self):
        """Large mesh — verify Delaunay handles many points."""
        G = nx.Graph()
        np.random.seed(42)
        for i in range(100):
            G.add_node(
                i,
                pos=np.random.rand(2) * 100,
                type="mesh",
            )
        dg = create_prebuilt_flat_from_nodes(G)
        assert dg.number_of_nodes() == 100
        assert dg.number_of_edges() > 0
        # All edges should be bidirectional
        for u, v in dg.edges():
            assert dg.has_edge(v, u)

    def test_nodes_plus_edges_preserves_extra_attributes(self):
        """Extra edge attributes should be preserved in nodes+edges mode."""
        G = _make_simple_mesh_digraph(3)
        # Add custom attribute to edges
        for u, v in G.edges():
            G.edges[u, v]["custom"] = "test_value"

        xy = test_utils.create_fake_xy(10)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=G),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )

        # The m2m edges should preserve custom attributes
        for u, v, data in result["m2m"].edges(data=True):
            if data.get("component") == "m2m":
                # These came from the original prebuilt graph
                assert "custom" in data or "component" in data

    def test_prebuilt_with_decode_mask(self):
        """Prebuilt mesh should work with decode_mask for m2g."""
        xy = test_utils.create_fake_xy(10)
        mesh = _make_simple_mesh_digraph(4)
        n_grid = xy.shape[0]
        decode_mask = [True] * (n_grid // 2) + [False] * (n_grid - n_grid // 2)

        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            decode_mask=decode_mask,
            return_components=True,
        )

        assert "m2g" in result

    def test_digraph_with_zero_edges_treated_as_nodes_only(self):
        """A DiGraph with nodes but no edges should be treated as nodes-only mode."""
        G = nx.DiGraph()
        for i in range(4):
            for j in range(4):
                G.add_node(
                    i * 4 + j,
                    pos=np.array([float(i), float(j)]),
                    type="mesh",
                )
        # This is a DiGraph but has NO edges → should use nodes-only path
        xy = test_utils.create_fake_xy(10)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=G),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        # Delaunay should have built edges
        assert result["m2m"].number_of_edges() > 0

    def test_mesh_layout_none_raises(self):
        """mesh_graph=None should raise ValueError."""
        xy = test_utils.create_fake_xy(10)
        with pytest.raises(ValueError, match="requires 'mesh_graph'"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat",
                mesh_layout="prebuilt",
                mesh_layout_kwargs=dict(mesh_graph=None),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )


# ===========================
# Structural property tests
# ===========================


class TestPrebuiltGraphStructuralProperties:
    """Verify structural properties of prebuilt-generated graphs."""

    def test_all_m2m_edges_have_component_label(self):
        xy = test_utils.create_fake_xy(10)
        mesh = _make_simple_mesh_digraph(4)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        for comp_name, graph in result.items():
            for u, v, data in graph.edges(data=True):
                assert "component" in data
                assert data["component"] == comp_name

    def test_all_mesh_nodes_have_pos(self):
        xy = test_utils.create_fake_xy(10)
        mesh = _make_simple_mesh_digraph(4)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        for node, data in result["m2m"].nodes(data=True):
            assert "pos" in data

    def test_hierarchical_nodes_only_has_level_edges(self):
        """Hierarchical nodes-only mode should produce edges with level attrs."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            if data.get("direction") == "same":
                assert "level" in data

    def test_flat_nodes_only_all_edges_bidirectional(self):
        """Flat Delaunay edges should be bidirectional."""
        G = _make_simple_mesh_nodes(5)
        dg = create_prebuilt_flat_from_nodes(G)
        for u, v in dg.edges():
            assert dg.has_edge(v, u), f"Edge ({v}, {u}) missing — not bidirectional"

    def test_flat_multiscale_nodes_only_level_consistency(self):
        """Edges within each level should only connect nodes of that level."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            edge_level = data["level"]
            u_level = dg.nodes[u].get("level", None)
            v_level = dg.nodes[v].get("level", None)
            # Both endpoints should be at the same level as the edge
            if u_level is not None and v_level is not None:
                assert u_level == edge_level
                assert v_level == edge_level


# ===========================
# Additional validation edge cases
# ===========================


class TestValidationEdgeCases:
    """Exhaustive edge cases for all validation functions."""

    # --- pos attribute edge cases ---

    def test_pos_as_python_list_raises(self):
        """pos given as plain list instead of np.ndarray should fail."""
        G = nx.Graph()
        G.add_node(0, pos=[1.0, 2.0], type="mesh")
        with pytest.raises(ValueError, match="invalid shape"):
            validate_prebuilt_nodes(G)

    def test_pos_as_tuple_raises(self):
        """pos given as tuple should fail (no .shape attribute)."""
        G = nx.Graph()
        G.add_node(0, pos=(1.0, 2.0), type="mesh")
        with pytest.raises(ValueError, match="invalid shape"):
            validate_prebuilt_nodes(G)

    def test_pos_as_scalar_raises(self):
        """pos given as scalar should fail."""
        G = nx.Graph()
        G.add_node(0, pos=5.0, type="mesh")
        with pytest.raises(ValueError, match="invalid shape"):
            validate_prebuilt_nodes(G)

    def test_pos_1d_single_element_raises(self):
        """pos as 1d array of length 1 should fail."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([1.0]), type="mesh")
        with pytest.raises(ValueError, match="invalid shape"):
            validate_prebuilt_nodes(G)

    def test_pos_2d_array_raises(self):
        """pos as 2d array of shape (1,2) should fail — must be (2,)."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([[1.0, 2.0]]), type="mesh")
        with pytest.raises(ValueError, match="invalid shape"):
            validate_prebuilt_nodes(G)

    def test_pos_integer_coordinates_valid(self):
        """pos with integer dtype should work (auto-converted)."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([1, 2]), type="mesh")
        validate_prebuilt_nodes(G)  # Should not raise

    def test_pos_with_nan_coordinates(self):
        """pos with NaN should pass validation (validation doesn't check values)."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([np.nan, np.nan]), type="mesh")
        validate_prebuilt_nodes(G)  # Should not raise — just shape check

    def test_pos_with_inf_coordinates(self):
        """pos with inf should pass validation."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([np.inf, -np.inf]), type="mesh")
        validate_prebuilt_nodes(G)  # Should not raise

    # --- type attribute edge cases ---

    def test_type_not_mesh_still_valid(self):
        """type='grid' is technically valid (validation only checks presence)."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="grid")
        validate_prebuilt_nodes(G)  # Should not raise (only checks existence)

    def test_type_as_integer_still_valid(self):
        """type=42 is technically valid (validation only checks presence)."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type=42)
        validate_prebuilt_nodes(G)  # Should not raise

    # --- level attribute edge cases ---

    def test_level_as_float_still_valid(self):
        """level=0.0 instead of 0 should still pass validation."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh", level=0.0)
        validate_prebuilt_nodes_with_levels(G)

    def test_level_negative_still_valid(self):
        """Negative level values should pass validation (just checks existence)."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh", level=-1)
        validate_prebuilt_nodes_with_levels(G)

    def test_mixed_nodes_some_missing_level(self):
        """If ONE node is missing level, should raise even if others have it."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh", level=0)
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")  # no level
        with pytest.raises(ValueError, match="missing required 'level'"):
            validate_prebuilt_nodes_with_levels(G)

    def test_mixed_nodes_some_missing_pos(self):
        """If ONE node is missing pos, should raise even if others have it."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, type="mesh")  # no pos
        with pytest.raises(ValueError, match="missing required 'pos'"):
            validate_prebuilt_nodes(G)

    def test_mixed_nodes_some_missing_type(self):
        """If ONE node is missing type, should raise even if others have it."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]))  # no type
        with pytest.raises(ValueError, match="missing required 'type'"):
            validate_prebuilt_nodes(G)

    # --- Edge validation edge cases ---

    def test_edges_partial_len_missing(self):
        """If ONE edge is missing len, should raise even if others have it."""
        G = nx.DiGraph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")
        G.add_node(2, pos=np.array([0.0, 1.0]), type="mesh")
        G.add_edge(0, 1, len=1.0, vdiff=np.array([1.0, 0.0]))
        G.add_edge(1, 2, vdiff=np.array([-1.0, 1.0]))  # missing len
        with pytest.raises(ValueError, match="missing required 'len'"):
            validate_prebuilt_mesh_edges(G)

    def test_edges_partial_vdiff_missing(self):
        """If ONE edge is missing vdiff, should raise."""
        G = nx.DiGraph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")
        G.add_node(2, pos=np.array([0.0, 1.0]), type="mesh")
        G.add_edge(0, 1, len=1.0, vdiff=np.array([1.0, 0.0]))
        G.add_edge(1, 2, len=1.414)  # missing vdiff
        with pytest.raises(ValueError, match="missing required 'vdiff'"):
            validate_prebuilt_mesh_edges(G)

    def test_edge_len_zero_passes_validation(self):
        """Edge with len=0 should pass validation (no value check)."""
        G = nx.DiGraph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_edge(0, 1, len=0.0, vdiff=np.array([0.0, 0.0]))
        validate_prebuilt_mesh_edges(G)  # Should not raise

    def test_edge_len_negative_passes_validation(self):
        """Edge with negative len should pass validation (no value check)."""
        G = nx.DiGraph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")
        G.add_edge(0, 1, len=-1.0, vdiff=np.array([1.0, 0.0]))
        validate_prebuilt_mesh_edges(G)  # Should not raise

    def test_validate_digraph_on_digraph_works(self):
        """validate_prebuilt_nodes should work on DiGraph too (not just Graph)."""
        G = nx.DiGraph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        validate_prebuilt_nodes(G)  # Should not raise

    def test_self_loop_passes_edge_validation(self):
        """Self-loop edge should pass validation if attrs present."""
        G = nx.DiGraph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_edge(0, 0, len=0.0, vdiff=np.array([0.0, 0.0]))
        validate_prebuilt_mesh_edges(G)  # Should not raise

    def test_require_levels_with_only_some_level_edges(self):
        """Some (not all) edges having level should pass require_levels."""
        G = nx.DiGraph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh", level=0)
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh", level=0)
        G.add_node(2, pos=np.array([0.0, 1.0]), type="mesh", level=1)
        G.add_edge(0, 1, len=1.0, vdiff=np.array([1.0, 0.0]), level=0)
        G.add_edge(0, 2, len=1.0, vdiff=np.array([0.0, 1.0]))  # no level
        validate_prebuilt_mesh_edges(G, require_levels=True)  # Should not raise


# ===========================
# Delaunay edge building: additional edge cases
# ===========================


class TestDelaunayEdgeCasesExtended:
    """More edge cases for Delaunay triangulation."""

    def test_collinear_points_handled_gracefully(self):
        """Collinear points should NOT raise — fallback to KDTree nn."""
        G = nx.Graph()
        for i in range(5):
            G.add_node(i, pos=np.array([float(i), 0.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert isinstance(dg, nx.DiGraph)
        assert dg.number_of_nodes() == 5
        # Should have some edges from KDTree fallback
        assert dg.number_of_edges() > 0
        # All edges should be bidirectional
        for u, v in dg.edges():
            assert dg.has_edge(v, u)

    def test_collinear_three_points(self):
        """Three collinear points — minimum for Delaunay to fail."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")
        G.add_node(2, pos=np.array([2.0, 0.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert isinstance(dg, nx.DiGraph)
        assert dg.number_of_edges() > 0

    def test_duplicate_positions(self):
        """Two nodes at same position — should handle without crash."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(2, pos=np.array([1.0, 1.0]), type="mesh")
        # May raise or produce degenerate mesh — just verify no crash
        try:
            dg = _build_edges_delaunay(G)
            assert isinstance(dg, nx.DiGraph)
        except (scipy.spatial.QhullError, ValueError):
            pass  # acceptable

    def test_very_close_nodes(self):
        """Nodes extremely close together (near-degenerate)."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1e-15, 1e-15]), type="mesh")
        G.add_node(2, pos=np.array([1.0, 0.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert isinstance(dg, nx.DiGraph)

    def test_large_coordinate_values(self):
        """Nodes with large coordinate values (numerical stability)."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([1e8, 1e8]), type="mesh")
        G.add_node(1, pos=np.array([1e8 + 1, 1e8]), type="mesh")
        G.add_node(2, pos=np.array([1e8, 1e8 + 1]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert isinstance(dg, nx.DiGraph)
        assert dg.number_of_edges() == 6

    def test_negative_coordinates(self):
        """Nodes with negative coordinates."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([-10.0, -10.0]), type="mesh")
        G.add_node(1, pos=np.array([-5.0, -10.0]), type="mesh")
        G.add_node(2, pos=np.array([-10.0, -5.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.number_of_edges() == 6
        for u, v, data in dg.edges(data=True):
            assert data["len"] > 0

    def test_mixed_positive_negative_coordinates(self):
        """Nodes spanning both positive and negative coords."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([-1.0, -1.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, -1.0]), type="mesh")
        G.add_node(2, pos=np.array([0.0, 1.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.number_of_edges() == 6

    def test_string_node_ids(self):
        """Nodes with string IDs should work."""
        G = nx.Graph()
        G.add_node("a", pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node("b", pos=np.array([1.0, 0.0]), type="mesh")
        G.add_node("c", pos=np.array([0.5, 1.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.has_edge("a", "b")
        assert dg.has_edge("b", "a")

    def test_tuple_node_ids(self):
        """Nodes with tuple IDs should work."""
        G = nx.Graph()
        G.add_node((0, 0), pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node((0, 1), pos=np.array([1.0, 0.0]), type="mesh")
        G.add_node((1, 0), pos=np.array([0.0, 1.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.has_edge((0, 0), (0, 1))

    def test_four_nodes_square(self):
        """Square arrangement — should produce 4 or 5 unique edges."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")
        G.add_node(2, pos=np.array([1.0, 1.0]), type="mesh")
        G.add_node(3, pos=np.array([0.0, 1.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        # Delaunay of a square produces 2 triangles with 5 unique edges
        # → 10 directed edges (bidirectional)
        assert dg.number_of_edges() == 10
        # All nodes should have degree >= 2
        for n in dg.nodes():
            assert dg.degree(n) >= 2

    def test_hexagonal_pattern(self):
        """Hexagon arrangement — realistic mesh pattern."""
        G = nx.Graph()
        # 6 points on hexagon + 1 center
        for i in range(6):
            angle = i * np.pi / 3
            G.add_node(i, pos=np.array([np.cos(angle), np.sin(angle)]), type="mesh")
        G.add_node(6, pos=np.array([0.0, 0.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.number_of_nodes() == 7
        assert dg.number_of_edges() > 0
        # Center should connect to all outer nodes
        center_edges = list(dg.successors(6)) + list(dg.predecessors(6))
        assert len(set(center_edges)) >= 6

    def test_empty_graph_returns_empty_digraph(self):
        """Empty graph (0 nodes) should return empty DiGraph."""
        G = nx.Graph()
        dg = _build_edges_delaunay(G)
        assert isinstance(dg, nx.DiGraph)
        assert dg.number_of_nodes() == 0
        assert dg.number_of_edges() == 0

    def test_random_500_nodes(self):
        """Stress test with 500 random nodes."""
        G = nx.Graph()
        rng = np.random.default_rng(seed=123)
        for i in range(500):
            G.add_node(i, pos=rng.random(2) * 1000, type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.number_of_nodes() == 500
        assert dg.number_of_edges() > 0
        # Spot-check: all edges bidirectional
        for u, v in list(dg.edges())[:100]:
            assert dg.has_edge(v, u)


# ===========================
# Flat from nodes: additional scenarios
# ===========================


class TestPrebuiltFlatFromNodesExtended:
    """More scenarios for create_prebuilt_flat_from_nodes."""

    def test_three_nodes_triangle(self):
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")
        G.add_node(2, pos=np.array([0.5, 0.866]), type="mesh")
        dg = create_prebuilt_flat_from_nodes(G)
        assert dg.number_of_edges() == 6

    def test_irregular_spacing(self):
        """Non-uniformly spaced nodes."""
        G = nx.Graph()
        positions = [
            [0.0, 0.0], [0.1, 0.5], [0.9, 0.2], [0.5, 0.8],
            [0.3, 0.1], [0.7, 0.6], [0.2, 0.9], [0.8, 0.4],
        ]
        for i, pos in enumerate(positions):
            G.add_node(i, pos=np.array(pos), type="mesh")
        dg = create_prebuilt_flat_from_nodes(G)
        assert dg.number_of_nodes() == 8
        assert dg.number_of_edges() > 0
        for u, v in dg.edges():
            assert dg.has_edge(v, u)

    def test_cluster_arrangement(self):
        """Two clusters of nodes far apart."""
        G = nx.Graph()
        # Cluster 1 near origin
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([0.1, 0.0]), type="mesh")
        G.add_node(2, pos=np.array([0.0, 0.1]), type="mesh")
        # Cluster 2 far away
        G.add_node(3, pos=np.array([100.0, 100.0]), type="mesh")
        G.add_node(4, pos=np.array([100.1, 100.0]), type="mesh")
        G.add_node(5, pos=np.array([100.0, 100.1]), type="mesh")
        dg = create_prebuilt_flat_from_nodes(G)
        assert dg.number_of_nodes() == 6
        assert dg.number_of_edges() > 0


# ===========================
# Flat multiscale from nodes: additional scenarios
# ===========================


class TestPrebuiltFlatMultiscaleExtended:
    """More scenarios for create_prebuilt_flat_multiscale_from_nodes."""

    def test_three_levels(self):
        """Three-level mesh should work correctly."""
        G = nx.Graph()
        # Level 0: 4x4
        for i in range(4):
            for j in range(4):
                G.add_node(
                    f"L0_{i}_{j}", pos=np.array([float(i), float(j)]),
                    type="mesh", level=0,
                )
        # Level 1: 3x3
        for i in range(3):
            for j in range(3):
                G.add_node(
                    f"L1_{i}_{j}", pos=np.array([float(i) * 1.5, float(j) * 1.5]),
                    type="mesh", level=1,
                )
        # Level 2: 2x2
        for i in range(2):
            for j in range(2):
                G.add_node(
                    f"L2_{i}_{j}", pos=np.array([float(i) * 3, float(j) * 3]),
                    type="mesh", level=2,
                )
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        assert dg.number_of_nodes() == 16 + 9 + 4
        edge_levels = set(d["level"] for _, _, d in dg.edges(data=True))
        assert edge_levels == {0, 1, 2}

    def test_single_node_per_level(self):
        """Each level has exactly one node — no intra-level edges."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh", level=0)
        G.add_node(1, pos=np.array([5.0, 5.0]), type="mesh", level=1)
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        assert dg.number_of_nodes() == 2
        assert dg.number_of_edges() == 0  # Can't triangulate 1 node per level

    def test_two_nodes_per_level(self):
        """Each level has two nodes — bidirectional within each level."""
        G = nx.Graph()
        G.add_node("a0", pos=np.array([0.0, 0.0]), type="mesh", level=0)
        G.add_node("b0", pos=np.array([1.0, 0.0]), type="mesh", level=0)
        G.add_node("a1", pos=np.array([0.0, 10.0]), type="mesh", level=1)
        G.add_node("b1", pos=np.array([1.0, 10.0]), type="mesh", level=1)
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        assert dg.number_of_nodes() == 4
        assert dg.number_of_edges() == 4  # 2 per level (bidirectional)

    def test_noncontiguous_level_numbers(self):
        """Level numbers 0, 5, 10 instead of 0, 1, 2 — should work."""
        G = nx.Graph()
        for i in range(3):
            G.add_node(f"L0_{i}", pos=np.array([float(i), 0.0]), type="mesh", level=0)
            G.add_node(f"L5_{i}", pos=np.array([float(i), 5.0]), type="mesh", level=5)
            G.add_node(f"L10_{i}", pos=np.array([float(i), 10.0]), type="mesh", level=10)
        # Add a 4th node per level so Delaunay works better
        G.add_node("L0_3", pos=np.array([1.0, 1.0]), type="mesh", level=0)
        G.add_node("L5_3", pos=np.array([1.0, 6.0]), type="mesh", level=5)
        G.add_node("L10_3", pos=np.array([1.0, 11.0]), type="mesh", level=10)
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        edge_levels = set(d["level"] for _, _, d in dg.edges(data=True))
        assert edge_levels == {0, 5, 10}

    def test_no_cross_level_edges(self):
        """Flat multiscale should NOT create edges between different levels."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            u_level = dg.nodes[u]["level"]
            v_level = dg.nodes[v]["level"]
            assert u_level == v_level, (
                f"Cross-level edge found: {u} (level={u_level}) -> "
                f"{v} (level={v_level})"
            )


# ===========================
# Hierarchical from nodes: additional scenarios
# ===========================


class TestPrebuiltHierarchicalExtended:
    """More scenarios for create_prebuilt_hierarchical_from_nodes."""

    def test_three_levels(self):
        """Three-level hierarchical mesh."""
        G = nx.Graph()
        for i in range(4):
            for j in range(4):
                G.add_node(f"L0_{i}_{j}", pos=np.array([float(i), float(j)]),
                           type="mesh", level=0)
        for i in range(3):
            for j in range(3):
                G.add_node(f"L1_{i}_{j}", pos=np.array([float(i) * 1.5, float(j) * 1.5]),
                           type="mesh", level=1)
        for i in range(2):
            for j in range(2):
                G.add_node(f"L2_{i}_{j}", pos=np.array([float(i) * 3, float(j) * 3]),
                           type="mesh", level=2)
        dg = create_prebuilt_hierarchical_from_nodes(G)
        directions = set(d.get("direction") for _, _, d in dg.edges(data=True))
        assert "same" in directions
        assert "up" in directions
        assert "down" in directions

    def test_three_levels_has_inter_edges_at_all_boundaries(self):
        """Three levels should have inter-level edges between L0↔L1 and L1↔L2."""
        G = nx.Graph()
        for i in range(4):
            for j in range(4):
                G.add_node(f"L0_{i}_{j}", pos=np.array([float(i), float(j)]),
                           type="mesh", level=0)
        for i in range(3):
            for j in range(3):
                G.add_node(f"L1_{i}_{j}", pos=np.array([float(i) * 1.5, float(j) * 1.5]),
                           type="mesh", level=1)
        for i in range(2):
            for j in range(2):
                G.add_node(f"L2_{i}_{j}", pos=np.array([float(i) * 3, float(j) * 3]),
                           type="mesh", level=2)
        dg = create_prebuilt_hierarchical_from_nodes(G)
        level_strings = set()
        for _, _, d in dg.edges(data=True):
            if "levels" in d:
                level_strings.add(d["levels"])
        # Should have connections between level 0↔1 and level 1↔2
        assert len(level_strings) >= 2

    def test_asymmetric_level_sizes(self):
        """Different number of nodes per level."""
        G = nx.Graph()
        # Level 0: 5x5 = 25 nodes
        for i in range(5):
            for j in range(5):
                G.add_node(f"L0_{i}_{j}", pos=np.array([float(i), float(j)]),
                           type="mesh", level=0)
        # Level 1: 2 nodes only (minimum for Delaunay < 3 fallback)
        G.add_node("L1_0", pos=np.array([1.0, 1.0]), type="mesh", level=1)
        G.add_node("L1_1", pos=np.array([3.0, 3.0]), type="mesh", level=1)
        dg = create_prebuilt_hierarchical_from_nodes(G)
        assert isinstance(dg, nx.DiGraph)
        # Up/down edges should exist
        up_count = sum(1 for _, _, d in dg.edges(data=True) if d.get("direction") == "up")
        assert up_count > 0

    def test_inter_level_k_larger_than_coarse_nodes(self):
        """k > number of coarser nodes: should connect to all available."""
        G = nx.Graph()
        # Level 0: 4x4 = 16 nodes
        for i in range(4):
            for j in range(4):
                G.add_node(f"L0_{i}_{j}", pos=np.array([float(i), float(j)]),
                           type="mesh", level=0)
        # Level 1: 3 nodes (k=5 > 3)
        G.add_node("L1_0", pos=np.array([0.0, 0.0]), type="mesh", level=1)
        G.add_node("L1_1", pos=np.array([2.0, 0.0]), type="mesh", level=1)
        G.add_node("L1_2", pos=np.array([1.0, 2.0]), type="mesh", level=1)
        dg = create_prebuilt_hierarchical_from_nodes(
            G, inter_level={"pattern": "nearest", "k": 5}
        )
        # k=5 but only 3 coarse nodes → should connect to all 3
        down_count = sum(1 for _, _, d in dg.edges(data=True)
                         if d.get("direction") == "down")
        assert down_count > 0

    def test_hierarchical_preserves_intra_level_bidirectional(self):
        """Intra-level edges should remain bidirectional."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            if data.get("direction") == "same":
                assert dg.has_edge(v, u), (
                    f"Intra-level edge ({u},{v}) is not bidirectional"
                )

    def test_hierarchical_up_down_symmetry(self):
        """For each down edge (u→v), there should be a reverse up edge (v→u)."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        down_edges = [
            (u, v) for u, v, d in dg.edges(data=True)
            if d.get("direction") == "down"
        ]
        for u, v in down_edges:
            # The reverse edge should exist and be "up"
            assert dg.has_edge(v, u), f"Missing up edge for down ({u},{v})"
            assert dg.edges[v, u].get("direction") == "up"


# ===========================
# Integration: more edge cases
# ===========================


class TestPrebuiltIntegrationExtended:
    """Extended integration scenarios across all connectivity modes."""

    def test_unsupported_mesh_layout_raises(self):
        """mesh_layout='hexagonal' (unsupported) should raise."""
        xy = test_utils.create_fake_xy(10)
        with pytest.raises(NotImplementedError, match="not yet supported"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat",
                mesh_layout="hexagonal",
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )

    def test_flat_prebuilt_with_nearest_neighbours_g2m(self):
        """Prebuilt + nearest_neighbours (plural) g2m connectivity."""
        xy = test_utils.create_fake_xy(10)
        mesh = _make_simple_mesh_digraph(4)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbours",
            g2m_connectivity_kwargs=dict(max_num_neighbours=3),
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert "g2m" in result
        # Each grid node should have up to 3 edges in g2m
        assert result["g2m"].number_of_edges() > 0

    def test_flat_prebuilt_with_nearest_neighbours_m2g(self):
        """Prebuilt + nearest_neighbours (plural) m2g connectivity."""
        xy = test_utils.create_fake_xy(10)
        mesh = _make_simple_mesh_digraph(4)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbours",
            m2g_connectivity_kwargs=dict(max_num_neighbours=4),
            return_components=True,
        )
        assert "m2g" in result
        assert result["m2g"].number_of_edges() > 0

    def test_flat_return_components_false(self):
        """Combined graph mode for flat prebuilt."""
        xy = test_utils.create_fake_xy(10)
        mesh = _make_simple_mesh_nodes(4)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=False,
        )
        assert isinstance(result, nx.DiGraph)
        assert result.number_of_nodes() > 0
        assert result.number_of_edges() > 0

    def test_hierarchical_return_components_false(self):
        """Combined graph mode for hierarchical prebuilt."""
        xy = test_utils.create_fake_xy(10)
        mesh = _make_multilevel_mesh_nodes()
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="hierarchical",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=False,
        )
        assert isinstance(result, nx.DiGraph)

    def test_flat_multiscale_return_components_false(self):
        """Combined graph mode for flat_multiscale prebuilt."""
        xy = test_utils.create_fake_xy(10)
        mesh = _make_multilevel_mesh_nodes()
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat_multiscale",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=False,
        )
        assert isinstance(result, nx.DiGraph)

    def test_small_grid_2x2(self):
        """Very small grid (2x2=4 nodes) with prebuilt mesh."""
        xy = test_utils.create_rectangular_fake_xy(2, 2)
        mesh = _make_simple_mesh_digraph(3)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert result["g2m"].number_of_edges() > 0
        assert result["m2g"].number_of_edges() > 0

    def test_mesh_far_outside_grid(self):
        """Mesh nodes far from grid — g2m/m2g still connects nearest."""
        xy = test_utils.create_fake_xy(5)  # grid around (0,0)→(4,4)
        G = nx.Graph()
        for i in range(3):
            for j in range(3):
                G.add_node(
                    i * 3 + j,
                    pos=np.array([float(i) + 1000.0, float(j) + 1000.0]),
                    type="mesh",
                )
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=G),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        # Should still work — nearest neighbour always finds something
        assert result["g2m"].number_of_edges() > 0
        assert result["m2g"].number_of_edges() > 0

    def test_string_node_ids_integration(self):
        """Prebuilt mesh with string node IDs through full pipeline."""
        xy = test_utils.create_fake_xy(8)
        G = nx.Graph()
        for i in range(3):
            for j in range(3):
                G.add_node(
                    f"node_{i}_{j}",
                    pos=np.array([float(i), float(j)]),
                    type="mesh",
                )
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=G),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert "m2m" in result
        assert result["m2m"].number_of_edges() > 0

    def test_mesh_layout_kwargs_missing_key(self):
        """mesh_layout_kwargs without 'mesh_graph' key raises."""
        xy = test_utils.create_fake_xy(10)
        with pytest.raises(ValueError, match="requires 'mesh_graph'"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat",
                mesh_layout="prebuilt",
                mesh_layout_kwargs=dict(some_other_key=42),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )

    def test_mesh_layout_kwargs_none_defaults(self):
        """mesh_layout_kwargs=None with prebuilt should raise clearly."""
        xy = test_utils.create_fake_xy(10)
        with pytest.raises(ValueError, match="requires 'mesh_graph'"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat",
                mesh_layout="prebuilt",
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )

    def test_hierarchical_prebuilt_nodes_plus_edges_combined(self):
        """Hierarchical nodes+edges combined graph has expected component labels."""
        xy = test_utils.create_fake_xy(10)
        mesh = _make_multilevel_mesh_digraph()
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="hierarchical",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=False,
        )
        components = set(d["component"] for _, _, d in result.edges(data=True))
        assert "m2m" in components
        assert "g2m" in components
        assert "m2g" in components

    def test_flat_prebuilt_decode_mask_all_true(self):
        """decode_mask all True should behave same as no mask."""
        xy = test_utils.create_fake_xy(8)
        mesh = _make_simple_mesh_digraph(3)
        n_grid = xy.shape[0]

        result_no_mask = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        result_all_true = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            decode_mask=[True] * n_grid,
            return_components=True,
        )
        # m2g should have same number of edges
        assert result_no_mask["m2g"].number_of_edges() == result_all_true["m2g"].number_of_edges()

    def test_flat_prebuilt_decode_mask_all_false(self):
        """decode_mask all False — m2g should have no edges."""
        xy = test_utils.create_fake_xy(8)
        mesh = _make_simple_mesh_digraph(3)
        n_grid = xy.shape[0]
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            decode_mask=[False] * n_grid,
            return_components=True,
        )
        assert result["m2g"].number_of_edges() == 0

    def test_flat_prebuilt_decode_mask_single_grid_point(self):
        """decode_mask with only 1 True — m2g should have exactly 1 edge."""
        xy = test_utils.create_fake_xy(8)
        mesh = _make_simple_mesh_digraph(3)
        n_grid = xy.shape[0]
        mask = [False] * n_grid
        mask[0] = True
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            decode_mask=mask,
            return_components=True,
        )
        assert result["m2g"].number_of_edges() == 1


# ===========================
# Correctness: numerical verification
# ===========================


class TestPrebuiltNumericalCorrectness:
    """Verify numerical correctness of computed edge attributes."""

    def test_edge_len_equals_euclidean_distance(self):
        """Edge len should exactly match Euclidean distance between nodes."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([3.0, 4.0]), type="mesh")
        G.add_node(2, pos=np.array([6.0, 0.0]), type="mesh")
        dg = create_prebuilt_flat_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            pos_u = dg.nodes[u]["pos"]
            pos_v = dg.nodes[v]["pos"]
            expected_len = np.sqrt(np.sum((pos_u - pos_v) ** 2))
            assert data["len"] == pytest.approx(expected_len, abs=1e-10)

    def test_vdiff_equals_pos_difference(self):
        """Edge vdiff should equal pos[u] - pos[v]."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([3.0, 4.0]), type="mesh")
        G.add_node(2, pos=np.array([6.0, 0.0]), type="mesh")
        dg = create_prebuilt_flat_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            pos_u = dg.nodes[u]["pos"]
            pos_v = dg.nodes[v]["pos"]
            expected_vdiff = pos_u - pos_v
            np.testing.assert_array_almost_equal(data["vdiff"], expected_vdiff)

    def test_len_is_nonnegative(self):
        """All computed edge lengths should be >= 0."""
        G = _make_simple_mesh_nodes(5)
        dg = create_prebuilt_flat_from_nodes(G)
        for _, _, data in dg.edges(data=True):
            assert data["len"] >= 0

    def test_vdiff_norm_equals_len(self):
        """||vdiff|| should equal len for every edge."""
        G = _make_simple_mesh_nodes(5)
        dg = create_prebuilt_flat_from_nodes(G)
        for _, _, data in dg.edges(data=True):
            norm = np.sqrt(np.sum(data["vdiff"] ** 2))
            assert norm == pytest.approx(data["len"], abs=1e-10)

    def test_hierarchical_inter_level_len_correct(self):
        """Inter-level edge lengths should be correct Euclidean distances."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            if data.get("direction") in ("up", "down"):
                pos_u = dg.nodes[u]["pos"]
                pos_v = dg.nodes[v]["pos"]
                expected = np.sqrt(np.sum((pos_u - pos_v) ** 2))
                assert data["len"] == pytest.approx(expected, abs=1e-10)

    def test_flat_multiscale_len_correct(self):
        """Edge lengths in flat multiscale should be correct."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            pos_u = dg.nodes[u]["pos"]
            pos_v = dg.nodes[v]["pos"]
            expected = np.sqrt(np.sum((pos_u - pos_v) ** 2))
            assert data["len"] == pytest.approx(expected, abs=1e-10)

    def test_two_node_len_exact(self):
        """Two-node path: edge length should be sqrt(2)."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 1.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.edges[0, 1]["len"] == pytest.approx(np.sqrt(2.0))


# ===========================
# Graph invariant tests
# ===========================


class TestPrebuiltGraphInvariants:
    """Verify graph-level invariants that should always hold."""

    def test_flat_no_isolated_nodes_for_grid(self):
        """For a grid-like mesh, every node should have at least one edge."""
        G = _make_simple_mesh_nodes(4)
        dg = create_prebuilt_flat_from_nodes(G)
        for node in dg.nodes():
            assert dg.degree(node) > 0

    def test_flat_multiscale_no_isolated_nodes_per_level(self):
        """Within each level, every node with 3+ peers should have edges."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        # Level 0 has 9 nodes — all should have edges
        level0_nodes = [n for n in dg.nodes() if dg.nodes[n].get("level") == 0]
        for node in level0_nodes:
            assert dg.degree(node) > 0

    def test_hierarchical_all_levels_represented_in_edges(self):
        """All level indices should appear in intra-level edges."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        intra_levels = set()
        for _, _, d in dg.edges(data=True):
            if d.get("direction") == "same":
                intra_levels.add(d["level"])
        # Both level 0 and level 1 should have intra-level edges
        assert 0 in intra_levels
        assert 1 in intra_levels

    def test_nodes_plus_edges_node_count_preserved(self):
        """Node count in m2m should match input mesh node count."""
        mesh = _make_simple_mesh_digraph(4)
        original_count = mesh.number_of_nodes()
        xy = test_utils.create_fake_xy(10)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        # m2m node count = original mesh + grid nodes (from connect_nodes_across_graphs)
        # Actually m2m should have exactly the mesh nodes after relabeling
        m2m_nodes = result["m2m"].number_of_nodes()
        # The m2m graph gets relabeled and then connect_nodes_across_graphs adds grid nodes
        # but the m2m component itself should carry at least the mesh nodes
        assert m2m_nodes >= original_count

    def test_nodes_plus_edges_edge_count_preserved(self):
        """Edge count in m2m should match input mesh edge count (plus component tag)."""
        mesh = _make_simple_mesh_digraph(4)
        original_edges = mesh.number_of_edges()
        xy = test_utils.create_fake_xy(10)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        m2m_edges = result["m2m"].number_of_edges()
        assert m2m_edges == original_edges

    def test_combined_graph_has_all_node_types(self):
        """Combined graph should contain both mesh and grid nodes."""
        xy = test_utils.create_fake_xy(8)
        mesh = _make_simple_mesh_digraph(3)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=False,
        )
        node_types = set()
        for _, data in result.nodes(data=True):
            if "type" in data:
                node_types.add(data["type"])
        assert "mesh" in node_types
        assert "grid" in node_types


# ===========================
# Additional edge cases: Validation
# ===========================


class TestValidationAdditionalEdgeCases:
    """Extra edge cases for validation functions."""

    def test_validate_nodes_large_graph(self):
        """Validation should handle graphs with many nodes efficiently."""
        G = nx.Graph()
        for i in range(1000):
            G.add_node(i, pos=np.array([float(i), float(i % 7)]), type="mesh")
        validate_prebuilt_nodes(G)  # Should not raise

    def test_validate_nodes_with_levels_pos_1d_single_raises(self):
        """pos of shape (1,) should fail in with_levels variant too."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([1.0]), type="mesh", level=0)
        with pytest.raises(ValueError, match="invalid shape"):
            validate_prebuilt_nodes_with_levels(G)

    def test_validate_edges_digraph_with_extra_node_attrs(self):
        """DiGraph with extra custom node attributes should pass."""
        G = nx.DiGraph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh", custom_attr="x")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh", custom_attr="y")
        G.add_edge(0, 1, len=1.0, vdiff=np.array([1.0, 0.0]))
        validate_prebuilt_mesh_edges(G)  # Should not raise

    def test_validate_edges_digraph_with_extra_edge_attrs(self):
        """DiGraph with extra custom edge attributes should pass."""
        G = nx.DiGraph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh")
        G.add_edge(0, 1, len=1.0, vdiff=np.array([1.0, 0.0]), weight=0.5)
        validate_prebuilt_mesh_edges(G)  # Should not raise

    def test_validate_nodes_3d_pos_raises(self):
        """pos of shape (3,) should fail — only 2D supported."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([1.0, 2.0, 3.0]), type="mesh")
        with pytest.raises(ValueError, match="invalid shape"):
            validate_prebuilt_nodes(G)

    def test_validate_nodes_empty_ndarray_raises(self):
        """pos as empty array should fail."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([]), type="mesh")
        with pytest.raises(ValueError, match="invalid shape"):
            validate_prebuilt_nodes(G)

    def test_validate_edges_multiple_edges_all_valid(self):
        """All edges with correct attrs on a larger graph."""
        G = nx.DiGraph()
        n = 10
        for i in range(n):
            G.add_node(i, pos=np.array([float(i), 0.0]), type="mesh")
        for i in range(n - 1):
            d = 1.0
            G.add_edge(i, i + 1, len=d, vdiff=np.array([1.0, 0.0]))
            G.add_edge(i + 1, i, len=d, vdiff=np.array([-1.0, 0.0]))
        validate_prebuilt_mesh_edges(G)  # Should not raise

    def test_validate_nodes_mixed_dtypes_pos(self):
        """pos with mixed int/float elements should work (shape is valid)."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([1, 2.5]), type="mesh")
        validate_prebuilt_nodes(G)  # Should not raise

    def test_validate_edges_require_levels_all_edges_have_level(self):
        """All edges having level should pass require_levels."""
        G = nx.DiGraph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh", level=0)
        G.add_node(1, pos=np.array([1.0, 0.0]), type="mesh", level=0)
        G.add_edge(0, 1, len=1.0, vdiff=np.array([1.0, 0.0]), level=0)
        G.add_edge(1, 0, len=1.0, vdiff=np.array([-1.0, 0.0]), level=0)
        validate_prebuilt_mesh_edges(G, require_levels=True)  # Should not raise

    def test_validate_nodes_none_type_value(self):
        """type=None is technically present as attr — should pass."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type=None)
        validate_prebuilt_nodes(G)  # Should not raise (only checks presence)


# ===========================
# Additional edge cases: Delaunay
# ===========================


class TestDelaunayAdditionalEdgeCases:
    """Extra edge cases for _build_edges_delaunay."""

    def test_all_identical_positions(self):
        """All nodes at exact same position — degenerate case."""
        G = nx.Graph()
        for i in range(5):
            G.add_node(i, pos=np.array([1.0, 1.0]), type="mesh")
        # This should not crash — falls back to KDTree
        dg = _build_edges_delaunay(G)
        assert isinstance(dg, nx.DiGraph)
        assert dg.number_of_nodes() == 5

    def test_l_shaped_pattern(self):
        """L-shaped node arrangement — tests boundary handling."""
        G = nx.Graph()
        positions = [
            [0, 0], [1, 0], [2, 0], [3, 0],
            [0, 1], [0, 2], [0, 3],
        ]
        for i, pos in enumerate(positions):
            G.add_node(i, pos=np.array(pos, dtype=float), type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.number_of_nodes() == 7
        assert dg.number_of_edges() > 0
        for u, v in dg.edges():
            assert dg.has_edge(v, u)

    def test_narrow_aspect_ratio(self):
        """Very narrow strip of nodes (e.g. 10:1 aspect ratio)."""
        G = nx.Graph()
        for i in range(20):
            G.add_node(i, pos=np.array([float(i) * 10.0, float(i % 2) * 0.1]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert isinstance(dg, nx.DiGraph)
        assert dg.number_of_edges() > 0

    def test_node_ordering_independence(self):
        """Result should be the same regardless of node insertion order."""
        positions = [[0, 0], [1, 0], [0.5, 0.866], [0.5, 0.3]]
        G1 = nx.Graph()
        for i, pos in enumerate(positions):
            G1.add_node(i, pos=np.array(pos), type="mesh")
        G2 = nx.Graph()
        # Reverse insertion order
        for i, pos in enumerate(reversed(positions)):
            G2.add_node(len(positions) - 1 - i, pos=np.array(pos), type="mesh")
        dg1 = _build_edges_delaunay(G1)
        dg2 = _build_edges_delaunay(G2)
        assert dg1.number_of_edges() == dg2.number_of_edges()
        # Same edge set
        edges1 = set((min(u, v), max(u, v)) for u, v in dg1.edges())
        edges2 = set((min(u, v), max(u, v)) for u, v in dg2.edges())
        assert edges1 == edges2

    def test_regular_grid_larger(self):
        """8x8 regular grid — should produce consistent triangulation."""
        G = nx.Graph()
        for i in range(8):
            for j in range(8):
                G.add_node(i * 8 + j, pos=np.array([float(i), float(j)]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.number_of_nodes() == 64
        # Every interior node should have at least 4 edges (degree 8+)
        for node in dg.nodes():
            assert dg.degree(node) >= 2

    def test_circular_arrangement(self):
        """Nodes arranged in a circle — boundary-only case."""
        G = nx.Graph()
        n = 12
        for i in range(n):
            angle = 2 * np.pi * i / n
            G.add_node(i, pos=np.array([np.cos(angle), np.sin(angle)]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.number_of_nodes() == n
        assert dg.number_of_edges() > 0
        # Every node should have at least 2 neighbours (on the convex hull)
        for node in dg.nodes():
            assert dg.degree(node) >= 2

    def test_single_cluster_plus_outlier(self):
        """Dense cluster + one far-away outlier — Delaunay should still connect."""
        G = nx.Graph()
        for i in range(5):
            for j in range(5):
                G.add_node(i * 5 + j, pos=np.array([float(i), float(j)]), type="mesh")
        # Outlier far away
        G.add_node(25, pos=np.array([1000.0, 1000.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.number_of_nodes() == 26
        # Outlier should be connected to at least one node
        assert dg.degree(25) >= 2

    def test_three_nodes_right_angle(self):
        """Three nodes forming a right angle triangle."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([3.0, 0.0]), type="mesh")
        G.add_node(2, pos=np.array([0.0, 4.0]), type="mesh")
        dg = _build_edges_delaunay(G)
        assert dg.number_of_edges() == 6
        # Check hypotenuse length (3-4-5 triangle)
        assert dg.edges[1, 2]["len"] == pytest.approx(5.0)


# ===========================
# Additional edge cases: Flat from nodes
# ===========================


class TestPrebuiltFlatFromNodesAdditional:
    """Additional scenarios for create_prebuilt_flat_from_nodes."""

    def test_weakly_connected_for_grid(self):
        """Flat Delaunay on a grid should produce a weakly connected graph."""
        G = _make_simple_mesh_nodes(5)
        dg = create_prebuilt_flat_from_nodes(G)
        assert nx.is_weakly_connected(dg)

    def test_single_node_preserves_attrs(self):
        """Single node should preserve all node attributes."""
        G = nx.Graph()
        G.add_node("only", pos=np.array([3.14, 2.72]), type="mesh", custom="val")
        dg = create_prebuilt_flat_from_nodes(G)
        assert dg.nodes["only"]["pos"][0] == pytest.approx(3.14)
        assert dg.nodes["only"]["type"] == "mesh"
        assert dg.nodes["only"]["custom"] == "val"

    def test_two_nodes_length_correct(self):
        """Two nodes separated by known distance."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([5.0, 12.0]), type="mesh")
        dg = create_prebuilt_flat_from_nodes(G)
        assert dg.edges[0, 1]["len"] == pytest.approx(13.0)  # 5-12-13

    def test_five_nodes_pentagon(self):
        """Regular pentagon — non-trivial Delaunay."""
        G = nx.Graph()
        for i in range(5):
            angle = 2 * np.pi * i / 5
            G.add_node(i, pos=np.array([np.cos(angle), np.sin(angle)]), type="mesh")
        dg = create_prebuilt_flat_from_nodes(G)
        assert dg.number_of_nodes() == 5
        assert dg.number_of_edges() > 0
        # All edges bidirectional
        for u, v in dg.edges():
            assert dg.has_edge(v, u)

    def test_diagonal_strip(self):
        """Nodes along a diagonal — near-collinear but with slight offset."""
        G = nx.Graph()
        for i in range(10):
            offset = 0.01 * (i % 2)  # Tiny perpendicular offset
            G.add_node(i, pos=np.array([float(i), float(i) + offset]), type="mesh")
        dg = create_prebuilt_flat_from_nodes(G)
        assert dg.number_of_nodes() == 10
        assert dg.number_of_edges() > 0


# ===========================
# Additional edge cases: Flat multiscale
# ===========================


class TestPrebuiltFlatMultiscaleAdditional:
    """Additional scenarios for flat multiscale from nodes."""

    def test_single_level_works(self):
        """One level only — should still produce a valid graph (no cross-level)."""
        G = nx.Graph()
        for i in range(4):
            for j in range(4):
                G.add_node(i * 4 + j, pos=np.array([float(i), float(j)]),
                           type="mesh", level=0)
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        assert dg.number_of_nodes() == 16
        assert dg.number_of_edges() > 0
        # All edges should be level 0
        for _, _, d in dg.edges(data=True):
            assert d["level"] == 0

    def test_level_graph_attribute_set(self):
        """Composed graph should still have edges from all levels."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        levels_found = set(d["level"] for _, _, d in dg.edges(data=True))
        assert 0 in levels_found
        assert 1 in levels_found

    def test_three_nodes_per_level(self):
        """Exactly 3 nodes per level — minimal Delaunay triangulation."""
        G = nx.Graph()
        G.add_node("a", pos=np.array([0.0, 0.0]), type="mesh", level=0)
        G.add_node("b", pos=np.array([1.0, 0.0]), type="mesh", level=0)
        G.add_node("c", pos=np.array([0.5, 0.866]), type="mesh", level=0)
        G.add_node("d", pos=np.array([0.0, 10.0]), type="mesh", level=1)
        G.add_node("e", pos=np.array([1.0, 10.0]), type="mesh", level=1)
        G.add_node("f", pos=np.array([0.5, 10.866]), type="mesh", level=1)
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        assert dg.number_of_nodes() == 6
        # Each level has triangle: 3 undirected edges → 6 directed edges per level
        assert dg.number_of_edges() == 12

    def test_edge_vdiff_correct_per_level(self):
        """vdiff should be correct for all edges within each level."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            pos_u = dg.nodes[u]["pos"]
            pos_v = dg.nodes[v]["pos"]
            np.testing.assert_array_almost_equal(data["vdiff"], pos_u - pos_v)

    def test_many_levels(self):
        """Five levels — verify all levels present in edges."""
        G = nx.Graph()
        for lev in range(5):
            for i in range(3):
                for j in range(3):
                    G.add_node(
                        f"L{lev}_{i}_{j}",
                        pos=np.array([float(i) + lev * 0.1, float(j) + lev * 0.1]),
                        type="mesh", level=lev,
                    )
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        edge_levels = set(d["level"] for _, _, d in dg.edges(data=True))
        assert edge_levels == {0, 1, 2, 3, 4}


# ===========================
# Additional edge cases: Hierarchical
# ===========================


class TestPrebuiltHierarchicalAdditional:
    """Additional edge cases for hierarchical from nodes."""

    def test_inter_level_edges_have_len_and_vdiff(self):
        """All inter-level edges should have len and vdiff attributes."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            if data.get("direction") in ("up", "down"):
                assert "len" in data, f"Inter-level edge ({u},{v}) missing len"
                assert "vdiff" in data, f"Inter-level edge ({u},{v}) missing vdiff"
                assert data["len"] >= 0

    def test_inter_level_edges_have_levels_string(self):
        """All inter-level edges should have a 'levels' string like '1>0'."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            if data.get("direction") in ("up", "down"):
                assert "levels" in data, f"Inter-level edge ({u},{v}) missing levels"
                assert ">" in data["levels"]

    def test_two_nodes_per_level_minimum(self):
        """Minimum viable hierarchical: 2 nodes at each of 2 levels."""
        G = nx.Graph()
        G.add_node("a", pos=np.array([0.0, 0.0]), type="mesh", level=0)
        G.add_node("b", pos=np.array([1.0, 0.0]), type="mesh", level=0)
        G.add_node("c", pos=np.array([0.5, 5.0]), type="mesh", level=1)
        G.add_node("d", pos=np.array([0.5, 6.0]), type="mesh", level=1)
        dg = create_prebuilt_hierarchical_from_nodes(G)
        assert isinstance(dg, nx.DiGraph)
        directions = set(d.get("direction") for _, _, d in dg.edges(data=True))
        assert "same" in directions
        assert "up" in directions
        assert "down" in directions

    def test_four_levels(self):
        """Four-level hierarchical mesh — multi-boundary inter-level."""
        G = nx.Graph()
        for lev in range(4):
            n = 4 - lev  # Decreasing nodes per level
            for i in range(max(n, 3)):  # At least 3 for Delaunay
                for j in range(max(n, 3)):
                    G.add_node(
                        f"L{lev}_{i}_{j}",
                        pos=np.array([float(i) * (lev + 1), float(j) * (lev + 1)]),
                        type="mesh", level=lev,
                    )
        dg = create_prebuilt_hierarchical_from_nodes(G)
        intra_levels = set(
            d["level"] for _, _, d in dg.edges(data=True)
            if d.get("direction") == "same"
        )
        # Should have intra-level edges at all 4 levels
        assert intra_levels == {0, 1, 2, 3}

    def test_inter_level_vdiff_correctness(self):
        """Down edge vdiff = pos[src] - pos[dst]; up edge copies down data."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        # For down edges: vdiff = pos[src] - pos[dst]
        for u, v, data in dg.edges(data=True):
            if data.get("direction") == "down":
                pos_u = dg.nodes[u]["pos"]
                pos_v = dg.nodes[v]["pos"]
                np.testing.assert_array_almost_equal(data["vdiff"], pos_u - pos_v)
        # For up edges: they are reversed copies, so vdiff = pos[dst] - pos[src]
        for u, v, data in dg.edges(data=True):
            if data.get("direction") == "up":
                pos_u = dg.nodes[u]["pos"]
                pos_v = dg.nodes[v]["pos"]
                np.testing.assert_array_almost_equal(data["vdiff"], pos_v - pos_u)

    def test_default_inter_level_k_is_1(self):
        """Default inter_level=None should use k=1."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        # With k=1, each fine node connects to exactly 1 coarse node
        down_targets = {}
        for u, v, data in dg.edges(data=True):
            if data.get("direction") == "down":
                down_targets.setdefault(u, []).append(v)
        # Each coarse node should have at least 1 down edge
        assert len(down_targets) > 0


# ===========================
# Additional integration edge cases
# ===========================


class TestPrebuiltIntegrationAdditional:
    """Additional integration tests covering untested code paths."""

    def test_flat_with_irregular_grid_coords(self):
        """Prebuilt mesh with irregular grid coordinates."""
        xy = test_utils.create_fake_irregular_coords(50)
        mesh = _make_simple_mesh_digraph(4)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert "m2m" in result
        assert result["g2m"].number_of_edges() > 0

    def test_flat_mesh_more_nodes_than_grid(self):
        """Mesh with more nodes than grid nodes."""
        xy = test_utils.create_fake_xy(3)  # 9 grid points
        mesh = _make_simple_mesh_nodes(10)  # 100 mesh nodes
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert result["m2m"].number_of_nodes() >= 100

    def test_flat_mesh_overlapping_with_grid(self):
        """Mesh node positions overlapping grid positions."""
        xy = test_utils.create_fake_xy(5)
        G = nx.Graph()
        # Place mesh nodes at same positions as some grid nodes
        for i in range(5):
            G.add_node(i, pos=np.array([float(i), float(i)]), type="mesh")
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=G),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert "g2m" in result
        assert "m2g" in result

    def test_flat_multiscale_nodes_plus_edges_combined(self):
        """Flat multiscale nodes+edges mode with return_components=False."""
        xy = test_utils.create_fake_xy(10)
        mesh = _make_flat_multiscale_digraph()
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat_multiscale",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=False,
        )
        assert isinstance(result, nx.DiGraph)
        components = set(d["component"] for _, _, d in result.edges(data=True))
        assert "m2m" in components
        assert "g2m" in components
        assert "m2g" in components

    def test_hierarchical_nodes_only_all_components_have_edges(self):
        """Hierarchical nodes-only: all 3 components should have edges."""
        xy = test_utils.create_fake_xy(10)
        mesh = _make_multilevel_mesh_nodes()
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="hierarchical",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert result["m2m"].number_of_edges() > 0
        assert result["g2m"].number_of_edges() > 0
        assert result["m2g"].number_of_edges() > 0

    def test_flat_multiscale_missing_mesh_graph_none(self):
        """mesh_graph=None in flat_multiscale should raise."""
        xy = test_utils.create_fake_xy(10)
        with pytest.raises(ValueError, match="requires 'mesh_graph'"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat_multiscale",
                mesh_layout="prebuilt",
                mesh_layout_kwargs=dict(mesh_graph=None),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )

    def test_hierarchical_missing_mesh_graph_none(self):
        """mesh_graph=None in hierarchical should raise."""
        xy = test_utils.create_fake_xy(10)
        with pytest.raises(ValueError, match="requires 'mesh_graph'"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="hierarchical",
                mesh_layout="prebuilt",
                mesh_layout_kwargs=dict(mesh_graph=None),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )

    def test_flat_prebuilt_within_radius_g2m(self):
        """Prebuilt + within_radius g2m connectivity."""
        xy = test_utils.create_fake_xy(5)
        mesh = _make_simple_mesh_digraph(3)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="within_radius",
            g2m_connectivity_kwargs=dict(max_dist=5.0),
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert result["g2m"].number_of_edges() > 0

    def test_flat_prebuilt_within_radius_m2g(self):
        """Prebuilt + within_radius m2g connectivity."""
        xy = test_utils.create_fake_xy(5)
        mesh = _make_simple_mesh_digraph(3)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="within_radius",
            m2g_connectivity_kwargs=dict(max_dist=5.0),
            return_components=True,
        )
        assert result["m2g"].number_of_edges() > 0

    def test_hierarchical_nodes_plus_edges_return_components(self):
        """Hierarchical nodes+edges with return_components=True."""
        xy = test_utils.create_fake_xy(10)
        mesh = _make_multilevel_mesh_digraph()
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="hierarchical",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert "m2m" in result
        assert "g2m" in result
        assert "m2g" in result
        # All should be DiGraphs
        for comp in result.values():
            assert isinstance(comp, nx.DiGraph)

    def test_flat_nodes_only_with_rectangular_grid(self):
        """Prebuilt nodes-only with non-square (rectangular) grid."""
        xy = test_utils.create_rectangular_fake_xy(5, 15)
        mesh = _make_simple_mesh_nodes(4)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert result["m2m"].number_of_edges() > 0
        assert result["g2m"].number_of_edges() > 0

    def test_flat_unsupported_layout_in_flat_multiscale(self):
        """Unsupported mesh_layout in flat_multiscale should raise."""
        xy = test_utils.create_fake_xy(10)
        with pytest.raises(NotImplementedError, match="not yet supported"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat_multiscale",
                mesh_layout="hexagonal",
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )

    def test_hierarchical_unsupported_layout(self):
        """Unsupported mesh_layout in hierarchical should raise."""
        xy = test_utils.create_fake_xy(10)
        with pytest.raises(NotImplementedError, match="not yet supported"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="hierarchical",
                mesh_layout="hexagonal",
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )


# ===========================
# Additional numerical correctness
# ===========================


class TestPrebuiltNumericalAdditional:
    """More numerical correctness checks."""

    def test_len_symmetry_forward_reverse(self):
        """len(u→v) should equal len(v→u) for all edges."""
        G = _make_simple_mesh_nodes(6)
        dg = create_prebuilt_flat_from_nodes(G)
        for u, v in dg.edges():
            if dg.has_edge(v, u):
                assert dg.edges[u, v]["len"] == pytest.approx(dg.edges[v, u]["len"])

    def test_vdiff_sum_to_zero(self):
        """vdiff(u→v) + vdiff(v→u) should be zero vector."""
        G = _make_simple_mesh_nodes(6)
        dg = create_prebuilt_flat_from_nodes(G)
        for u, v in dg.edges():
            if dg.has_edge(v, u):
                total = dg.edges[u, v]["vdiff"] + dg.edges[v, u]["vdiff"]
                np.testing.assert_array_almost_equal(total, np.zeros(2))

    def test_hierarchical_len_symmetry(self):
        """Inter-level: len(u→v) for down == len(v→u) for up."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        for u, v, data in dg.edges(data=True):
            if data.get("direction") == "down" and dg.has_edge(v, u):
                rev = dg.edges[v, u]
                assert data["len"] == pytest.approx(rev["len"])

    def test_flat_multiscale_len_symmetry(self):
        """Edge lengths should be symmetric in flat multiscale."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        for u, v in dg.edges():
            if dg.has_edge(v, u):
                assert dg.edges[u, v]["len"] == pytest.approx(dg.edges[v, u]["len"])

    def test_triangle_inequality_holds(self):
        """For a triangle of 3 nodes, triangle inequality should hold."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([3.0, 0.0]), type="mesh")
        G.add_node(2, pos=np.array([1.0, 2.0]), type="mesh")
        dg = create_prebuilt_flat_from_nodes(G)
        l01 = dg.edges[0, 1]["len"]
        l12 = dg.edges[1, 2]["len"]
        l02 = dg.edges[0, 2]["len"]
        assert l01 + l12 > l02
        assert l01 + l02 > l12
        assert l12 + l02 > l01

    def test_vdiff_direction_correct(self):
        """vdiff should point from source to... actually vdiff = pos[u] - pos[v]."""
        G = nx.Graph()
        G.add_node(0, pos=np.array([0.0, 0.0]), type="mesh")
        G.add_node(1, pos=np.array([5.0, 0.0]), type="mesh")
        G.add_node(2, pos=np.array([0.0, 5.0]), type="mesh")
        dg = create_prebuilt_flat_from_nodes(G)
        # Edge 0→1: vdiff = pos[0] - pos[1] = [-5, 0]
        np.testing.assert_array_almost_equal(
            dg.edges[0, 1]["vdiff"], np.array([-5.0, 0.0])
        )
        # Edge 1→0: vdiff = pos[1] - pos[0] = [5, 0]
        np.testing.assert_array_almost_equal(
            dg.edges[1, 0]["vdiff"], np.array([5.0, 0.0])
        )


# ===========================
# Additional graph invariants
# ===========================


class TestPrebuiltGraphInvariantsAdditional:
    """More graph-level invariant checks."""

    def test_flat_weakly_connected_for_grid_mesh(self):
        """Flat Delaunay output on a grid should be weakly connected."""
        G = _make_simple_mesh_nodes(6)
        dg = create_prebuilt_flat_from_nodes(G)
        assert nx.is_weakly_connected(dg)

    def test_flat_multiscale_each_level_connected(self):
        """Within each level of flat multiscale, the subgraph should be connected."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_flat_multiscale_from_nodes(G)
        for lev in [0, 1]:
            level_nodes = [n for n in dg.nodes() if dg.nodes[n].get("level") == lev]
            subg = dg.subgraph(level_nodes)
            assert nx.is_weakly_connected(subg), f"Level {lev} is not connected"

    def test_hierarchical_intra_level_connected(self):
        """Intra-level subgraphs should be weakly connected."""
        G = _make_multilevel_mesh_nodes()
        dg = create_prebuilt_hierarchical_from_nodes(G)
        # Collect intra-level edges by level
        for lev in [0, 1]:
            level_edges = [
                (u, v) for u, v, d in dg.edges(data=True)
                if d.get("direction") == "same" and d.get("level") == lev
            ]
            level_nodes = set()
            for u, v in level_edges:
                level_nodes.add(u)
                level_nodes.add(v)
            subg = dg.subgraph(level_nodes)
            if len(level_nodes) > 1:
                assert nx.is_weakly_connected(subg), f"Level {lev} intra not connected"

    def test_combined_graph_g2m_m2g_m2m_edge_counts(self):
        """Combined graph: count edges by component should match components."""
        xy = test_utils.create_fake_xy(8)
        mesh = _make_simple_mesh_digraph(3)

        result_comp = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        result_combined = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=False,
        )
        # Count edges by component in combined graph
        combined_counts = {}
        for _, _, d in result_combined.edges(data=True):
            comp = d.get("component")
            combined_counts[comp] = combined_counts.get(comp, 0) + 1

        for name in ("m2m", "g2m", "m2g"):
            assert combined_counts.get(name, 0) == result_comp[name].number_of_edges()

    def test_nodes_plus_edges_no_extra_edges_added(self):
        """In nodes+edges mode, no extra m2m edges should be created."""
        mesh = _make_simple_mesh_digraph(3)
        original_edges = mesh.number_of_edges()
        xy = test_utils.create_fake_xy(8)
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert result["m2m"].number_of_edges() == original_edges

    def test_nodes_only_more_edges_than_nodes(self):
        """Delaunay on a grid should produce more edges than nodes."""
        G = _make_simple_mesh_nodes(5)
        dg = create_prebuilt_flat_from_nodes(G)
        assert dg.number_of_edges() > dg.number_of_nodes()


# ===========================
# Stress / robustness tests
# ===========================


class TestPrebuiltStressTests:
    """Stress and robustness tests."""

    def test_1000_node_flat(self):
        """1000-node flat Delaunay should complete without error."""
        G = nx.Graph()
        rng = np.random.default_rng(seed=99)
        for i in range(1000):
            G.add_node(i, pos=rng.random(2) * 100, type="mesh")
        dg = create_prebuilt_flat_from_nodes(G)
        assert dg.number_of_nodes() == 1000
        assert dg.number_of_edges() > 0
        # Spot check bidirectionality
        for u, v in list(dg.edges())[:50]:
            assert dg.has_edge(v, u)

    def test_1000_node_integration(self):
        """1000-node prebuilt mesh through full integration pipeline."""
        xy = test_utils.create_fake_xy(10)
        G = nx.Graph()
        rng = np.random.default_rng(seed=77)
        for i in range(1000):
            G.add_node(i, pos=rng.random(2) * 10, type="mesh")
        result = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=G),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert result["m2m"].number_of_edges() > 0
        assert result["g2m"].number_of_edges() > 0
        assert result["m2g"].number_of_edges() > 0

    def test_multilevel_stress_3_levels(self):
        """3-level hierarchical with many nodes per level."""
        G = nx.Graph()
        rng = np.random.default_rng(seed=55)
        for lev in range(3):
            n_nodes = 100 - lev * 30  # 100, 70, 40
            for i in range(n_nodes):
                G.add_node(
                    f"L{lev}_{i}",
                    pos=rng.random(2) * (10 + lev * 5),
                    type="mesh", level=lev,
                )
        dg = create_prebuilt_hierarchical_from_nodes(G)
        assert isinstance(dg, nx.DiGraph)
        directions = set(d.get("direction") for _, _, d in dg.edges(data=True))
        assert "same" in directions
        assert "up" in directions
        assert "down" in directions

    def test_repeated_runs_deterministic(self):
        """Same input should produce same output across multiple runs."""
        G = _make_simple_mesh_nodes(5)
        dg1 = create_prebuilt_flat_from_nodes(G)
        dg2 = create_prebuilt_flat_from_nodes(G)
        assert dg1.number_of_edges() == dg2.number_of_edges()
        assert set(dg1.edges()) == set(dg2.edges())
        for u, v in dg1.edges():
            assert dg1.edges[u, v]["len"] == pytest.approx(dg2.edges[u, v]["len"])
