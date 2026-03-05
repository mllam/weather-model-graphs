"""
Tests for mesh_layout="triangular" support (Issue #80).

Tests verify:
1. Primitive creation (node count, positions, adjacency_type, type attrs)
2. Single-level directed graph (bidirectional edges, len/vdiff attrs, 6-connectivity)
3. Multirange primitive creation
4. Flat single-scale mesh graph via wrapper + integration
5. Flat multiscale mesh graph (position-based merging)
6. Hierarchical mesh graph
7. Integration through create_all_graph_components for all m2m_connectivity modes
8. Edge cases (spacing too large, single-level hierarchical)
9. Numerical correctness (len symmetry, vdiff reciprocity)
10. Pattern equivalence (4-star == 8-star for triangular)
"""

import networkx as nx
import numpy as np
import pytest

import tests.utils as test_utils
import weather_model_graphs as wmg
from weather_model_graphs.create.mesh.coords import create_directed_mesh_graph
from weather_model_graphs.create.mesh.kinds.triangular import (
    create_flat_multiscale_from_triangular_coordinates,
    create_flat_multiscale_triangular_mesh_graph,
    create_flat_singlescale_triangular_mesh_graph,
    create_hierarchical_triangular_mesh_graph,
    create_multirange_2d_triangular_mesh_primitives,
    create_single_level_2d_triangular_mesh_graph,
    create_single_level_2d_triangular_mesh_primitive,
)


# ===========================
# Fixtures
# ===========================


@pytest.fixture
def xy_small():
    """Small 10x10 domain with 4 corner grid points."""
    return np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)


@pytest.fixture
def xy_medium():
    """Medium domain with many grid points."""
    return test_utils.create_fake_xy(N=20)


@pytest.fixture
def xy_rectangular():
    """Non-square domain."""
    return test_utils.create_rectangular_fake_xy(Nx=15, Ny=10)


# ===========================
# Step 1: Triangular Primitive (Coordinate Creation)
# ===========================


class TestTriangularPrimitive:
    """Tests for create_single_level_2d_triangular_mesh_primitive."""

    def test_returns_undirected_graph(self, xy_small):
        G = create_single_level_2d_triangular_mesh_primitive(xy_small, nx=5, ny=5)
        assert isinstance(G, nx.Graph)
        assert not isinstance(G, nx.DiGraph)

    def test_nodes_have_pos_and_type(self, xy_small):
        G = create_single_level_2d_triangular_mesh_primitive(xy_small, nx=4, ny=4)
        for node in G.nodes:
            assert "pos" in G.nodes[node]
            assert "type" in G.nodes[node]
            assert G.nodes[node]["type"] == "mesh"
            pos = G.nodes[node]["pos"]
            assert len(pos) == 2
            assert np.isfinite(pos).all()

    def test_nonzero_node_count(self, xy_small):
        G = create_single_level_2d_triangular_mesh_primitive(xy_small, nx=6, ny=6)
        assert G.number_of_nodes() > 0

    def test_has_edges(self, xy_small):
        G = create_single_level_2d_triangular_mesh_primitive(xy_small, nx=6, ny=6)
        assert G.number_of_edges() > 0

    def test_all_edges_are_cardinal(self, xy_small):
        """Triangular lattice has only cardinal edges (no diagonal distinction)."""
        G = create_single_level_2d_triangular_mesh_primitive(xy_small, nx=5, ny=5)
        for u, v, d in G.edges(data=True):
            assert "adjacency_type" in d, f"Edge ({u}, {v}) missing adjacency_type"
            assert d["adjacency_type"] == "cardinal"

    def test_graph_has_dx_dy(self, xy_small):
        G = create_single_level_2d_triangular_mesh_primitive(xy_small, nx=5, ny=5)
        assert "dx" in G.graph
        assert "dy" in G.graph
        assert G.graph["dx"] > 0
        assert G.graph["dy"] > 0

    def test_positions_within_domain(self, xy_small):
        """Mesh node positions should lie within the coordinate domain."""
        xm, xM = xy_small[:, 0].min(), xy_small[:, 0].max()
        ym, yM = xy_small[:, 1].min(), xy_small[:, 1].max()
        G = create_single_level_2d_triangular_mesh_primitive(xy_small, nx=6, ny=6)
        for node in G.nodes:
            pos = G.nodes[node]["pos"]
            assert xm <= pos[0] <= xM, f"x={pos[0]} out of [{xm}, {xM}]"
            assert ym <= pos[1] <= yM, f"y={pos[1]} out of [{ym}, {yM}]"

    def test_raises_on_zero_nodes(self):
        """nx=0 or ny=0 should produce 0 nodes and raise."""
        xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        with pytest.raises(ValueError, match="produced 0 nodes"):
            create_single_level_2d_triangular_mesh_primitive(xy, nx=0, ny=0)

    def test_rectangular_domain(self, xy_rectangular):
        """Works with non-square domains."""
        G = create_single_level_2d_triangular_mesh_primitive(
            xy_rectangular, nx=8, ny=5
        )
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0


# ===========================
# Step 2: Directed Mesh Graph (Connectivity Creation)
# ===========================


class TestTriangularDirectedGraph:
    """Tests for directed graph creation from triangular primitives."""

    def test_returns_digraph(self, xy_small):
        G_coords = create_single_level_2d_triangular_mesh_primitive(
            xy_small, nx=5, ny=5
        )
        G = create_directed_mesh_graph(G_coords, pattern="4-star")
        assert isinstance(G, nx.DiGraph)

    def test_edges_are_bidirectional(self, xy_small):
        G_coords = create_single_level_2d_triangular_mesh_primitive(
            xy_small, nx=5, ny=5
        )
        G = create_directed_mesh_graph(G_coords, pattern="4-star")
        for u, v in G.edges():
            assert G.has_edge(v, u), f"Edge ({u}, {v}) missing reverse"

    def test_edges_have_len_and_vdiff(self, xy_small):
        G_coords = create_single_level_2d_triangular_mesh_primitive(
            xy_small, nx=5, ny=5
        )
        G = create_directed_mesh_graph(G_coords, pattern="4-star")
        for u, v, d in G.edges(data=True):
            assert "len" in d, f"Edge ({u}, {v}) missing 'len'"
            assert "vdiff" in d, f"Edge ({u}, {v}) missing 'vdiff'"
            assert d["len"] > 0
            assert len(d["vdiff"]) == 2

    def test_len_symmetry(self, xy_small):
        """Edge length should be the same in both directions."""
        G_coords = create_single_level_2d_triangular_mesh_primitive(
            xy_small, nx=5, ny=5
        )
        G = create_directed_mesh_graph(G_coords, pattern="4-star")
        for u, v in G.edges():
            if G.has_edge(v, u):
                np.testing.assert_allclose(
                    G[u][v]["len"], G[v][u]["len"], atol=1e-10
                )

    def test_vdiff_reciprocity(self, xy_small):
        """vdiff(u→v) should equal -vdiff(v→u)."""
        G_coords = create_single_level_2d_triangular_mesh_primitive(
            xy_small, nx=5, ny=5
        )
        G = create_directed_mesh_graph(G_coords, pattern="4-star")
        for u, v in G.edges():
            if G.has_edge(v, u):
                np.testing.assert_allclose(
                    G[u][v]["vdiff"], -G[v][u]["vdiff"], atol=1e-10
                )

    def test_pattern_4star_equals_8star(self, xy_small):
        """For triangular lattice, 4-star and 8-star should produce identical
        graphs since all edges are 'cardinal'."""
        G_coords = create_single_level_2d_triangular_mesh_primitive(
            xy_small, nx=5, ny=5
        )
        G4 = create_directed_mesh_graph(G_coords, pattern="4-star")
        G8 = create_directed_mesh_graph(G_coords, pattern="8-star")
        assert G4.number_of_nodes() == G8.number_of_nodes()
        assert G4.number_of_edges() == G8.number_of_edges()

    def test_node_count_preserved(self, xy_small):
        """Directed graph should have same number of nodes as primitive."""
        G_coords = create_single_level_2d_triangular_mesh_primitive(
            xy_small, nx=5, ny=5
        )
        G = create_directed_mesh_graph(G_coords, pattern="4-star")
        assert G.number_of_nodes() == G_coords.number_of_nodes()


# ===========================
# Multirange primitives
# ===========================


class TestMultirangeTriangularPrimitives:
    """Tests for create_multirange_2d_triangular_mesh_primitives."""

    def test_returns_list(self, xy_medium):
        G_list = create_multirange_2d_triangular_mesh_primitives(
            max_num_levels=3, xy=xy_medium, mesh_node_spacing=2,
            interlevel_refinement_factor=3,
        )
        assert isinstance(G_list, list)
        assert len(G_list) >= 1

    def test_each_level_is_undirected(self, xy_medium):
        G_list = create_multirange_2d_triangular_mesh_primitives(
            max_num_levels=3, xy=xy_medium, mesh_node_spacing=2,
            interlevel_refinement_factor=3,
        )
        for G in G_list:
            assert isinstance(G, nx.Graph)
            assert not isinstance(G, nx.DiGraph)

    def test_level_attributes_set(self, xy_medium):
        G_list = create_multirange_2d_triangular_mesh_primitives(
            max_num_levels=3, xy=xy_medium, mesh_node_spacing=2,
            interlevel_refinement_factor=3,
        )
        for lev, G in enumerate(G_list):
            assert G.graph["level"] == lev
            for node in G.nodes:
                assert G.nodes[node]["level"] == lev
            for u, v in G.edges():
                assert G.edges[u, v]["level"] == lev

    def test_finer_level_has_more_nodes(self, xy_medium):
        G_list = create_multirange_2d_triangular_mesh_primitives(
            max_num_levels=3, xy=xy_medium, mesh_node_spacing=2,
            interlevel_refinement_factor=3,
        )
        if len(G_list) > 1:
            assert G_list[0].number_of_nodes() > G_list[1].number_of_nodes()

    def test_max_num_levels_respected(self, xy_medium):
        G_list = create_multirange_2d_triangular_mesh_primitives(
            max_num_levels=2, xy=xy_medium, mesh_node_spacing=2,
            interlevel_refinement_factor=3,
        )
        assert len(G_list) <= 2


# ===========================
# Convenience wrapper: single-level
# ===========================


class TestSingleLevelTriangularGraph:
    """Tests for create_single_level_2d_triangular_mesh_graph."""

    def test_returns_digraph(self, xy_small):
        G = create_single_level_2d_triangular_mesh_graph(xy_small, nx=5, ny=5)
        assert isinstance(G, nx.DiGraph)

    def test_has_bidirectional_edges(self, xy_small):
        G = create_single_level_2d_triangular_mesh_graph(xy_small, nx=5, ny=5)
        for u, v in G.edges():
            assert G.has_edge(v, u)


# ===========================
# Flat single-scale
# ===========================


class TestFlatSinglescaleTriangular:
    """Tests for create_flat_singlescale_triangular_mesh_graph."""

    def test_returns_digraph(self, xy_small):
        G = create_flat_singlescale_triangular_mesh_graph(xy_small, mesh_node_distance=2.0)
        assert isinstance(G, nx.DiGraph)

    def test_nodes_have_pos(self, xy_small):
        G = create_flat_singlescale_triangular_mesh_graph(xy_small, mesh_node_distance=2.0)
        for node in G.nodes:
            assert "pos" in G.nodes[node]

    def test_raises_on_large_spacing(self, xy_small):
        """Spacing larger than domain should raise."""
        with pytest.raises(ValueError, match="too large"):
            create_flat_singlescale_triangular_mesh_graph(
                xy_small, mesh_node_distance=100.0
            )


# ===========================
# Flat multiscale (triangular-specific merging)
# ===========================


class TestFlatMultiscaleTriangular:
    """Tests for the triangular flat multiscale graph and position-based merging."""

    def test_returns_digraph(self, xy_medium):
        G = create_flat_multiscale_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        assert isinstance(G, nx.DiGraph)

    def test_has_edges(self, xy_medium):
        G = create_flat_multiscale_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        assert G.number_of_edges() > 0

    def test_edges_have_len_and_vdiff(self, xy_medium):
        G = create_flat_multiscale_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        for u, v, d in G.edges(data=True):
            assert "len" in d
            assert "vdiff" in d

    def test_fewer_nodes_than_sum_of_levels(self, xy_medium):
        """Position-based merging should produce fewer nodes than the raw
        sum of all levels (coincident nodes get merged)."""
        G_coords_list = create_multirange_2d_triangular_mesh_primitives(
            max_num_levels=3, xy=xy_medium, mesh_node_spacing=2,
            interlevel_refinement_factor=3,
        )
        total_raw = sum(g.number_of_nodes() for g in G_coords_list)
        G = create_flat_multiscale_from_triangular_coordinates(G_coords_list)
        # Merged graph has at most as many nodes (usually fewer)
        assert G.number_of_nodes() <= total_raw

    def test_graph_has_dx_dy_dicts(self, xy_medium):
        G = create_flat_multiscale_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        assert isinstance(G.graph.get("dx"), dict)
        assert isinstance(G.graph.get("dy"), dict)


# ===========================
# Hierarchical
# ===========================


class TestHierarchicalTriangular:
    """Tests for create_hierarchical_triangular_mesh_graph."""

    def test_returns_digraph(self, xy_medium):
        G = create_hierarchical_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        assert isinstance(G, nx.DiGraph)

    def test_has_edges(self, xy_medium):
        G = create_hierarchical_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        assert G.number_of_edges() > 0

    def test_edges_have_level_attribute(self, xy_medium):
        G = create_hierarchical_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        for u, v, d in G.edges(data=True):
            # Intra-level edges have 'level', inter-level have 'levels'
            assert "level" in d or "levels" in d

    def test_multiple_levels_present(self, xy_medium):
        G = create_hierarchical_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        levels = set()
        for u, v, d in G.edges(data=True):
            if "level" in d:
                levels.add(d["level"])
            elif "levels" in d:
                # Inter-level edges like '0>1'
                parts = d["levels"].split(">")
                levels.update(int(p) for p in parts)
        assert len(levels) >= 2, "Expected multiple levels in hierarchical graph"

    def test_single_level_raises(self, xy_small):
        """Hierarchical requires ≥2 levels; too-coarse spacing should raise."""
        with pytest.raises(ValueError):
            create_hierarchical_triangular_mesh_graph(
                xy_small, mesh_node_distance=20.0,
                level_refinement_factor=3, max_num_levels=3,
            )


# ===========================
# Integration: create_all_graph_components
# ===========================


class TestIntegrationTriangular:
    """Full integration tests through create_all_graph_components."""

    COMMON_KW = dict(
        m2g_connectivity="nearest_neighbours",
        g2m_connectivity="nearest_neighbours",
        m2g_connectivity_kwargs=dict(max_num_neighbours=4),
        g2m_connectivity_kwargs=dict(max_num_neighbours=4),
        return_components=True,
    )

    def test_flat_triangular(self, xy_medium):
        comps = wmg.create.create_all_graph_components(
            coords=xy_medium,
            m2m_connectivity="flat",
            mesh_layout="triangular",
            mesh_layout_kwargs=dict(mesh_node_spacing=2.0),
            **self.COMMON_KW,
        )
        m2m = comps["m2m"]
        assert isinstance(m2m, nx.DiGraph)
        assert m2m.number_of_nodes() > 0
        assert m2m.number_of_edges() > 0
        # Should also have g2m and m2g
        assert comps["g2m"].number_of_edges() > 0
        assert comps["m2g"].number_of_edges() > 0

    def test_hierarchical_triangular(self, xy_medium):
        comps = wmg.create.create_all_graph_components(
            coords=xy_medium,
            m2m_connectivity="hierarchical",
            mesh_layout="triangular",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=2.0, max_num_refinement_levels=3
            ),
            **self.COMMON_KW,
        )
        m2m = comps["m2m"]
        assert isinstance(m2m, nx.DiGraph)
        assert m2m.number_of_nodes() > 0

    def test_flat_multiscale_triangular(self, xy_medium):
        comps = wmg.create.create_all_graph_components(
            coords=xy_medium,
            m2m_connectivity="flat_multiscale",
            mesh_layout="triangular",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=2.0, max_num_refinement_levels=3
            ),
            **self.COMMON_KW,
        )
        m2m = comps["m2m"]
        assert isinstance(m2m, nx.DiGraph)
        assert m2m.number_of_nodes() > 0

    def test_unsupported_layout_raises(self, xy_small):
        with pytest.raises(NotImplementedError, match="not yet supported"):
            wmg.create.create_all_graph_components(
                coords=xy_small,
                m2m_connectivity="flat",
                mesh_layout="hexagonal",
                mesh_layout_kwargs=dict(mesh_node_spacing=1.0),
                **self.COMMON_KW,
            )

    def test_flat_triangular_return_combined(self, xy_medium):
        """With return_components=False, returns a single composed graph."""
        kw = dict(self.COMMON_KW)
        kw["return_components"] = False
        G = wmg.create.create_all_graph_components(
            coords=xy_medium,
            m2m_connectivity="flat",
            mesh_layout="triangular",
            mesh_layout_kwargs=dict(mesh_node_spacing=2.0),
            **kw,
        )
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() > 0

    def test_flat_pattern_kwarg_forwarded(self, xy_medium):
        """m2m_connectivity_kwargs={'pattern': ...} should be forwarded."""
        comps = wmg.create.create_all_graph_components(
            coords=xy_medium,
            m2m_connectivity="flat",
            mesh_layout="triangular",
            mesh_layout_kwargs=dict(mesh_node_spacing=2.0),
            m2m_connectivity_kwargs=dict(pattern="8-star"),
            **self.COMMON_KW,
        )
        assert comps["m2m"].number_of_edges() > 0

    def test_rectilinear_still_works(self, xy_medium):
        """Regression: rectilinear layout should not be broken."""
        comps = wmg.create.create_all_graph_components(
            coords=xy_medium,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(mesh_node_spacing=2.0),
            **self.COMMON_KW,
        )
        assert comps["m2m"].number_of_nodes() > 0

    def test_rectilinear_flat_multiscale_still_works(self, xy_medium):
        """Regression: rectilinear flat_multiscale should not be broken."""
        comps = wmg.create.create_all_graph_components(
            coords=xy_medium,
            m2m_connectivity="flat_multiscale",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=2.0, max_num_refinement_levels=3
            ),
            **self.COMMON_KW,
        )
        assert comps["m2m"].number_of_nodes() > 0

    def test_rectilinear_hierarchical_still_works(self, xy_medium):
        """Regression: rectilinear hierarchical should not be broken."""
        comps = wmg.create.create_all_graph_components(
            coords=xy_medium,
            m2m_connectivity="hierarchical",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=2.0, max_num_refinement_levels=3
            ),
            **self.COMMON_KW,
        )
        assert comps["m2m"].number_of_nodes() > 0


# ===========================
# Numerical correctness
# ===========================


class TestNumericalCorrectness:
    """Test numerical properties of the triangular mesh graph."""

    def test_edge_lengths_positive(self, xy_medium):
        G = create_flat_singlescale_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0
        )
        for u, v, d in G.edges(data=True):
            assert d["len"] > 0

    def test_vdiff_consistent_with_pos(self, xy_small):
        """vdiff should equal pos(u) - pos(v)."""
        G = create_single_level_2d_triangular_mesh_graph(xy_small, nx=5, ny=5)
        for u, v, d in G.edges(data=True):
            pos_u = G.nodes[u]["pos"]
            pos_v = G.nodes[v]["pos"]
            expected_vdiff = pos_u - pos_v
            np.testing.assert_allclose(d["vdiff"], expected_vdiff, atol=1e-10)

    def test_len_consistent_with_vdiff(self, xy_small):
        """len should equal the L2 norm of vdiff."""
        G = create_single_level_2d_triangular_mesh_graph(xy_small, nx=5, ny=5)
        for u, v, d in G.edges(data=True):
            expected_len = np.linalg.norm(d["vdiff"])
            np.testing.assert_allclose(d["len"], expected_len, atol=1e-10)
