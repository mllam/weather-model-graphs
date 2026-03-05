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


@pytest.fixture
def xy_offset():
    """Domain not starting at origin."""
    return np.array([[5, 3], [15, 3], [5, 13], [15, 13]], dtype=float)


@pytest.fixture
def xy_large():
    """Larger domain with many grid points."""
    return test_utils.create_fake_xy(N=50)


@pytest.fixture
def xy_wide():
    """Very wide, short domain."""
    return test_utils.create_rectangular_fake_xy(Nx=30, Ny=5)


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

    def test_minimal_lattice(self, xy_small):
        """Smallest valid lattice (nx=1, ny=1) should produce nodes and edges."""
        G = create_single_level_2d_triangular_mesh_primitive(xy_small, nx=1, ny=1)
        assert G.number_of_nodes() >= 2
        assert G.number_of_edges() >= 1

    def test_large_lattice(self, xy_small):
        """Large nx/ny values should produce many nodes."""
        G = create_single_level_2d_triangular_mesh_primitive(xy_small, nx=20, ny=20)
        assert G.number_of_nodes() > 100

    def test_asymmetric_nx_ny(self, xy_small):
        """Very different nx and ny should still work."""
        G = create_single_level_2d_triangular_mesh_primitive(xy_small, nx=15, ny=3)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_offset_domain(self, xy_offset):
        """Domain not starting at origin: positions should still be within bounds."""
        G = create_single_level_2d_triangular_mesh_primitive(xy_offset, nx=5, ny=5)
        xm, xM = 5.0, 15.0
        ym, yM = 3.0, 13.0
        for node in G.nodes:
            pos = G.nodes[node]["pos"]
            assert xm <= pos[0] <= xM
            assert ym <= pos[1] <= yM

    def test_positions_are_numpy_arrays(self, xy_small):
        """Node positions should be numpy arrays."""
        G = create_single_level_2d_triangular_mesh_primitive(xy_small, nx=4, ny=4)
        for node in G.nodes:
            assert isinstance(G.nodes[node]["pos"], np.ndarray)

    def test_no_self_loops(self, xy_small):
        """Primitive graph should have no self-loops."""
        G = create_single_level_2d_triangular_mesh_primitive(xy_small, nx=6, ny=6)
        for u, v in G.edges():
            assert u != v

    def test_wide_domain(self, xy_wide):
        """Very wide, short domain should still produce valid mesh."""
        G = create_single_level_2d_triangular_mesh_primitive(xy_wide, nx=10, ny=3)
        assert G.number_of_nodes() > 0
        for node in G.nodes:
            pos = G.nodes[node]["pos"]
            assert np.isfinite(pos).all()


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

    def test_edge_count_is_twice_undirected(self, xy_small):
        """Directed graph should have exactly 2x the undirected edge count."""
        G_coords = create_single_level_2d_triangular_mesh_primitive(
            xy_small, nx=5, ny=5
        )
        G = create_directed_mesh_graph(G_coords, pattern="4-star")
        assert G.number_of_edges() == 2 * G_coords.number_of_edges()

    def test_no_self_loops_directed(self, xy_small):
        """Directed graph should have no self-loops."""
        G_coords = create_single_level_2d_triangular_mesh_primitive(
            xy_small, nx=5, ny=5
        )
        G = create_directed_mesh_graph(G_coords, pattern="4-star")
        for u, v in G.edges():
            assert u != v

    def test_pos_preserved_after_direction(self, xy_small):
        """Node positions should be preserved after converting to directed."""
        G_coords = create_single_level_2d_triangular_mesh_primitive(
            xy_small, nx=5, ny=5
        )
        G = create_directed_mesh_graph(G_coords, pattern="4-star")
        for node in G.nodes:
            assert "pos" in G.nodes[node]
            assert len(G.nodes[node]["pos"]) == 2

    def test_interior_node_degree_six(self, xy_small):
        """Interior nodes of a triangular lattice should have degree 6
        (6 in-edges + 6 out-edges = 12 total in directed graph)."""
        G_coords = create_single_level_2d_triangular_mesh_primitive(
            xy_small, nx=8, ny=8
        )
        G = create_directed_mesh_graph(G_coords, pattern="4-star")
        # At least one interior node should have degree 12 (6 in + 6 out)
        max_deg = max(dict(G.degree()).values())
        assert max_deg == 12

    def test_minimal_lattice_directed(self, xy_small):
        """Minimal lattice (nx=1, ny=1) should still produce a valid directed graph."""
        G = create_single_level_2d_triangular_mesh_graph(xy_small, nx=1, ny=1)
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() >= 2
        assert G.number_of_edges() >= 2  # at least one bidirectional edge


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

    def test_single_level(self, xy_medium):
        """max_num_levels=1 should produce exactly 1 level."""
        G_list = create_multirange_2d_triangular_mesh_primitives(
            max_num_levels=1, xy=xy_medium, mesh_node_spacing=2,
            interlevel_refinement_factor=3,
        )
        assert len(G_list) == 1
        assert G_list[0].graph["level"] == 0

    def test_refinement_factor_2(self, xy_medium):
        """Different refinement factor should still produce valid graphs."""
        G_list = create_multirange_2d_triangular_mesh_primitives(
            max_num_levels=3, xy=xy_medium, mesh_node_spacing=2,
            interlevel_refinement_factor=2,
        )
        assert len(G_list) >= 1
        for G in G_list:
            assert G.number_of_nodes() > 0

    def test_all_levels_cover_same_domain(self, xy_medium):
        """All levels should span approximately the same coordinate domain."""
        G_list = create_multirange_2d_triangular_mesh_primitives(
            max_num_levels=3, xy=xy_medium, mesh_node_spacing=2,
            interlevel_refinement_factor=3,
        )
        if len(G_list) < 2:
            pytest.skip("Only one level created")
        # Check centers are roughly the same across levels
        centers = []
        for G in G_list:
            positions = np.array([G.nodes[n]["pos"] for n in G.nodes])
            centers.append(positions.mean(axis=0))
        for c in centers[1:]:
            np.testing.assert_allclose(c, centers[0], atol=2.0)

    def test_interlevel_refinement_factor_preserved(self, xy_medium):
        """Each level should have the refinement factor as a graph attribute."""
        G_list = create_multirange_2d_triangular_mesh_primitives(
            max_num_levels=3, xy=xy_medium, mesh_node_spacing=2,
            interlevel_refinement_factor=3,
        )
        for G in G_list:
            assert G.graph["interlevel_refinement_factor"] == 3

    def test_all_levels_have_edges(self, xy_medium):
        """Every level should have at least some edges."""
        G_list = create_multirange_2d_triangular_mesh_primitives(
            max_num_levels=3, xy=xy_medium, mesh_node_spacing=2,
            interlevel_refinement_factor=3,
        )
        for G in G_list:
            assert G.number_of_edges() > 0


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

    def test_edges_have_attributes(self, xy_small):
        """Directed graph edges should have len and vdiff."""
        G = create_single_level_2d_triangular_mesh_graph(xy_small, nx=5, ny=5)
        for u, v, d in G.edges(data=True):
            assert "len" in d
            assert "vdiff" in d

    def test_with_rectangular_domain(self, xy_rectangular):
        """Should work correctly on non-square domains."""
        G = create_single_level_2d_triangular_mesh_graph(xy_rectangular, nx=8, ny=5)
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_edges() > 0

    def test_minimal_grid(self, xy_small):
        """Minimal grid (nx=1, ny=1) should produce a valid graph."""
        G = create_single_level_2d_triangular_mesh_graph(xy_small, nx=1, ny=1)
        assert G.number_of_nodes() >= 2


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

    def test_edges_are_bidirectional(self, xy_small):
        """All edges should have a reverse."""
        G = create_flat_singlescale_triangular_mesh_graph(xy_small, mesh_node_distance=2.0)
        for u, v in G.edges():
            assert G.has_edge(v, u)

    def test_smaller_spacing_more_nodes(self, xy_small):
        """Smaller mesh_node_distance should produce more nodes."""
        G_coarse = create_flat_singlescale_triangular_mesh_graph(
            xy_small, mesh_node_distance=3.0
        )
        G_fine = create_flat_singlescale_triangular_mesh_graph(
            xy_small, mesh_node_distance=1.5
        )
        assert G_fine.number_of_nodes() > G_coarse.number_of_nodes()

    def test_rectangular_domain(self, xy_rectangular):
        """Should work with non-square domains."""
        G = create_flat_singlescale_triangular_mesh_graph(
            xy_rectangular, mesh_node_distance=2.0
        )
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() > 0

    def test_no_nan_positions(self, xy_small):
        """No node should have NaN or Inf positions."""
        G = create_flat_singlescale_triangular_mesh_graph(xy_small, mesh_node_distance=2.0)
        for node in G.nodes:
            pos = G.nodes[node]["pos"]
            assert np.isfinite(pos).all()

    def test_spacing_just_fits(self):
        """Spacing that just fits one cell should work."""
        xy = np.array([[0, 0], [5, 0], [0, 5], [5, 5]], dtype=float)
        G = create_flat_singlescale_triangular_mesh_graph(xy, mesh_node_distance=4.0)
        assert G.number_of_nodes() >= 2


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

    def test_bidirectional_edges(self, xy_medium):
        """All edges should have a reverse."""
        G = create_flat_multiscale_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        for u, v in G.edges():
            assert G.has_edge(v, u), f"Edge ({u},{v}) no reverse"

    def test_nodes_have_pos(self, xy_medium):
        """All nodes should have pos attribute."""
        G = create_flat_multiscale_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        for node in G.nodes:
            assert "pos" in G.nodes[node]
            assert np.isfinite(G.nodes[node]["pos"]).all()

    def test_single_level_multiscale(self):
        """When domain only supports 1 level, flat_multiscale should still work."""
        xy = np.array([[0, 0], [3, 0], [0, 3], [3, 3]], dtype=float)
        G = create_flat_multiscale_triangular_mesh_graph(
            xy, mesh_node_distance=1.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() > 0

    def test_refinement_factor_2(self, xy_medium):
        """Refinement factor of 2 should work."""
        G = create_flat_multiscale_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=2, max_num_levels=3,
        )
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_edges() > 0

    def test_no_self_loops(self, xy_medium):
        """No self-loops in flat multiscale graph."""
        G = create_flat_multiscale_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        for u, v in G.edges():
            assert u != v

    def test_more_nodes_than_coarsest_level(self, xy_large):
        """Multiscale should have more nodes than the coarsest single level."""
        G_coords_list = create_multirange_2d_triangular_mesh_primitives(
            max_num_levels=3, xy=xy_large, mesh_node_spacing=2,
            interlevel_refinement_factor=3,
        )
        if len(G_coords_list) < 2:
            pytest.skip("Only one level created")
        coarsest_nodes = G_coords_list[-1].number_of_nodes()
        G_multi = create_flat_multiscale_from_triangular_coordinates(G_coords_list)
        assert G_multi.number_of_nodes() > coarsest_nodes


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

    def test_nodes_have_pos(self, xy_medium):
        """All nodes should have pos attribute."""
        G = create_hierarchical_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        for node in G.nodes:
            assert "pos" in G.nodes[node]
            assert np.isfinite(G.nodes[node]["pos"]).all()

    def test_custom_intra_level(self, xy_medium):
        """Custom intra_level pattern should be accepted."""
        G = create_hierarchical_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
            intra_level={"pattern": "8-star"},
        )
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_edges() > 0

    def test_custom_inter_level(self, xy_medium):
        """Custom inter_level config should be accepted."""
        G = create_hierarchical_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
            inter_level={"pattern": "nearest", "k": 3},
        )
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_edges() > 0

    def test_no_self_loops(self, xy_medium):
        """Hierarchical graph should have no self-loops."""
        G = create_hierarchical_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        for u, v in G.edges():
            assert u != v

    def test_has_inter_level_edges(self, xy_medium):
        """Should have inter-level edges connecting different levels."""
        G = create_hierarchical_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        inter_level_count = sum(
            1 for _, _, d in G.edges(data=True) if "levels" in d
        )
        assert inter_level_count > 0

    def test_inter_level_edges_have_direction(self, xy_medium):
        """Inter-level edges should have 'direction' attribute (up/down)."""
        G = create_hierarchical_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0,
            level_refinement_factor=3, max_num_levels=3,
        )
        for u, v, d in G.edges(data=True):
            if "levels" in d:
                assert "direction" in d
                assert d["direction"] in ("up", "down")


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

    def test_flat_triangular_with_within_radius(self, xy_medium):
        """Triangular flat with within_radius g2m/m2g connectivity."""
        comps = wmg.create.create_all_graph_components(
            coords=xy_medium,
            m2m_connectivity="flat",
            mesh_layout="triangular",
            mesh_layout_kwargs=dict(mesh_node_spacing=2.0),
            m2g_connectivity="within_radius",
            g2m_connectivity="within_radius",
            m2g_connectivity_kwargs=dict(max_dist=5.0),
            g2m_connectivity_kwargs=dict(max_dist=5.0),
            return_components=True,
        )
        assert comps["m2m"].number_of_nodes() > 0
        assert comps["g2m"].number_of_edges() > 0
        assert comps["m2g"].number_of_edges() > 0

    def test_flat_triangular_with_nearest_neighbour(self, xy_medium):
        """Triangular flat with nearest_neighbour (singular) connectivity."""
        comps = wmg.create.create_all_graph_components(
            coords=xy_medium,
            m2m_connectivity="flat",
            mesh_layout="triangular",
            mesh_layout_kwargs=dict(mesh_node_spacing=2.0),
            m2g_connectivity="nearest_neighbour",
            g2m_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert comps["m2m"].number_of_nodes() > 0
        assert comps["g2m"].number_of_edges() > 0

    def test_flat_no_mesh_node_spacing_raises(self, xy_small):
        """Missing mesh_node_spacing should raise ValueError."""
        with pytest.raises(ValueError, match="mesh_node_spacing"):
            wmg.create.create_all_graph_components(
                coords=xy_small,
                m2m_connectivity="flat",
                mesh_layout="triangular",
                mesh_layout_kwargs=dict(),
                **self.COMMON_KW,
            )

    def test_flat_multiscale_no_mesh_node_spacing_raises(self, xy_small):
        """Missing mesh_node_spacing in flat_multiscale should raise ValueError."""
        with pytest.raises(ValueError, match="mesh_node_spacing"):
            wmg.create.create_all_graph_components(
                coords=xy_small,
                m2m_connectivity="flat_multiscale",
                mesh_layout="triangular",
                mesh_layout_kwargs=dict(max_num_refinement_levels=3),
                **self.COMMON_KW,
            )

    def test_hierarchical_no_mesh_node_spacing_raises(self, xy_small):
        """Missing mesh_node_spacing in hierarchical should raise ValueError."""
        with pytest.raises(ValueError, match="mesh_node_spacing"):
            wmg.create.create_all_graph_components(
                coords=xy_small,
                m2m_connectivity="hierarchical",
                mesh_layout="triangular",
                mesh_layout_kwargs=dict(max_num_refinement_levels=3),
                **self.COMMON_KW,
            )

    def test_all_components_have_nodes(self, xy_medium):
        """All three components (g2m, m2m, m2g) should have nodes."""
        comps = wmg.create.create_all_graph_components(
            coords=xy_medium,
            m2m_connectivity="flat",
            mesh_layout="triangular",
            mesh_layout_kwargs=dict(mesh_node_spacing=2.0),
            **self.COMMON_KW,
        )
        for key in ("g2m", "m2m", "m2g"):
            assert comps[key].number_of_nodes() > 0
            assert comps[key].number_of_edges() > 0

    def test_large_domain_triangular(self, xy_large):
        """Large domain with small spacing should produce a big graph."""
        comps = wmg.create.create_all_graph_components(
            coords=xy_large,
            m2m_connectivity="flat",
            mesh_layout="triangular",
            mesh_layout_kwargs=dict(mesh_node_spacing=3.0),
            **self.COMMON_KW,
        )
        assert comps["m2m"].number_of_nodes() > 50


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

    def test_no_nan_in_edge_attrs(self, xy_small):
        """Edge attributes should contain no NaN or Inf."""
        G = create_single_level_2d_triangular_mesh_graph(xy_small, nx=6, ny=6)
        for u, v, d in G.edges(data=True):
            assert np.isfinite(d["len"])
            assert np.isfinite(d["vdiff"]).all()

    def test_no_zero_length_edges(self, xy_small):
        """All edges should have strictly positive length."""
        G = create_single_level_2d_triangular_mesh_graph(xy_small, nx=6, ny=6)
        for u, v, d in G.edges(data=True):
            assert d["len"] > 1e-12

    def test_edge_lengths_roughly_uniform_for_interior(self, xy_small):
        """For a uniform triangular lattice, all edges should have similar
        length (within a narrow tolerance, accounting for scaling)."""
        G = create_single_level_2d_triangular_mesh_graph(xy_small, nx=8, ny=8)
        lengths = [d["len"] for _, _, d in G.edges(data=True)]
        # In a uniformly scaled equilateral mesh, all edges should be
        # within ~50% of each other (accounting for aspect ratio scaling)
        max_len = max(lengths)
        min_len = min(lengths)
        assert min_len > 0
        ratio = max_len / min_len
        # For equilateral triangles with potentially different x/y scaling,
        # the ratio should still be reasonable
        assert ratio < 3.0, f"Edge length ratio {ratio} too large"

    def test_scaled_domain_produces_scaled_lengths(self):
        """Doubling the domain should roughly double edge lengths."""
        xy1 = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
        xy2 = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=float)
        G1 = create_single_level_2d_triangular_mesh_graph(xy1, nx=5, ny=5)
        G2 = create_single_level_2d_triangular_mesh_graph(xy2, nx=5, ny=5)
        avg_len1 = np.mean([d["len"] for _, _, d in G1.edges(data=True)])
        avg_len2 = np.mean([d["len"] for _, _, d in G2.edges(data=True)])
        np.testing.assert_allclose(avg_len2 / avg_len1, 2.0, rtol=0.1)

    def test_no_nan_in_positions(self, xy_medium):
        """No node should have NaN in positions."""
        G = create_flat_singlescale_triangular_mesh_graph(
            xy_medium, mesh_node_distance=2.0
        )
        for node in G.nodes:
            pos = G.nodes[node]["pos"]
            assert isinstance(pos, np.ndarray)
            assert np.isfinite(pos).all()
