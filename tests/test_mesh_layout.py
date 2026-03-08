"""
Tests for the mesh_layout parameter and two-step coordinate/connectivity
architecture introduced in Issue #78.

These tests verify:
1. The new API (mesh_layout, mesh_layout_kwargs with refinement_factor and
   max_num_refinement_levels, m2m_connectivity_kwargs with pattern for flat/
   flat_multiscale and intra_level/inter_level sub-dicts for hierarchical)
2. The two-step process (coordinate creation → connectivity creation)
3. The 4-star vs 8-star pattern functionality
4. Backward compatibility with old-style kwargs
5. Edge annotations on coordinate graphs
6. Error handling for invalid inputs
"""

import io
import warnings

import networkx as nx
import numpy as np
import pytest
from loguru import logger

import tests.utils as test_utils
import weather_model_graphs as wmg
from weather_model_graphs.create.mesh.coords import (
    create_directed_mesh_graph,
    create_multirange_2d_mesh_primitives,
    create_single_level_2d_mesh_primitive,
)
from weather_model_graphs.create.mesh.kinds.flat import (
    create_flat_multiscale_from_coordinates,
    create_flat_singlescale_from_coordinates,
)
from weather_model_graphs.create.mesh.kinds.hierarchical import (
    create_hierarchical_from_coordinates,
)


# ====================
# Step 1: Coordinate creation tests
# ====================


class TestSingleLevelCoordinateCreation:
    """Tests for create_single_level_2d_mesh_primitive."""

    def test_returns_undirected_graph(self):
        xy = test_utils.create_fake_xy(N=10)
        G = create_single_level_2d_mesh_primitive(xy, nx=5, ny=5)
        assert isinstance(G, nx.Graph)
        assert not isinstance(G, nx.DiGraph)

    def test_nodes_have_pos_and_type(self):
        xy = test_utils.create_fake_xy(N=10)
        G = create_single_level_2d_mesh_primitive(xy, nx=4, ny=4)
        for node in G.nodes:
            assert "pos" in G.nodes[node]
            assert "type" in G.nodes[node]
            assert G.nodes[node]["type"] == "mesh"
            assert len(G.nodes[node]["pos"]) == 2

    def test_correct_number_of_nodes(self):
        xy = test_utils.create_fake_xy(N=10)
        G = create_single_level_2d_mesh_primitive(xy, nx=5, ny=4)
        assert len(G.nodes) == 5 * 4

    def test_edges_have_adjacency_type(self):
        xy = test_utils.create_fake_xy(N=10)
        G = create_single_level_2d_mesh_primitive(xy, nx=5, ny=5)
        cardinal_count = 0
        diagonal_count = 0
        for u, v, d in G.edges(data=True):
            assert "adjacency_type" in d, f"Edge ({u}, {v}) missing adjacency_type"
            assert d["adjacency_type"] in ("cardinal", "diagonal")
            if d["adjacency_type"] == "cardinal":
                cardinal_count += 1
            else:
                diagonal_count += 1
        # For a 5x5 grid: cardinal = 2*(5*4) = 40, diagonal = 2*(4*4) = 32
        assert cardinal_count == 2 * (5 * 4)
        assert diagonal_count == 2 * (4 * 4)

    def test_graph_has_dx_dy(self):
        xy = test_utils.create_fake_xy(N=10)
        G = create_single_level_2d_mesh_primitive(xy, nx=5, ny=5)
        assert "dx" in G.graph
        assert "dy" in G.graph
        assert G.graph["dx"] > 0
        assert G.graph["dy"] > 0


class TestMultirangeCoordinateCreation:
    """Tests for create_multirange_2d_mesh_primitives."""

    def test_returns_list_of_undirected_graphs(self):
        xy = test_utils.create_fake_xy(N=30)
        G_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        assert isinstance(G_list, list)
        assert len(G_list) > 0
        for G in G_list:
            assert isinstance(G, nx.Graph)
            assert not isinstance(G, nx.DiGraph)

    def test_each_level_has_level_attribute(self):
        xy = test_utils.create_fake_xy(N=30)
        G_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        for i, G in enumerate(G_list):
            assert G.graph["level"] == i
            for node in G.nodes:
                assert G.nodes[node]["level"] == i

    def test_interlevel_refinement_factor_stored(self):
        xy = test_utils.create_fake_xy(N=30)
        G_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        for G in G_list:
            assert G.graph["interlevel_refinement_factor"] == 3

    def test_edges_have_adjacency_type(self):
        xy = test_utils.create_fake_xy(N=30)
        G_list = create_multirange_2d_mesh_primitives(
            max_num_levels=2, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        for G in G_list:
            for u, v, d in G.edges(data=True):
                assert "adjacency_type" in d

    def test_coarser_levels_have_fewer_nodes(self):
        xy = test_utils.create_fake_xy(N=30)
        G_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        if len(G_list) >= 2:
            for i in range(len(G_list) - 1):
                assert len(G_list[i].nodes) > len(G_list[i + 1].nodes)


# ====================
# Step 2: Connectivity creation tests
# ====================


class TestDirectedMeshGraph:
    """Tests for create_directed_mesh_graph."""

    def test_returns_directed_graph(self):
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=4, ny=4)
        G_directed = create_directed_mesh_graph(G_coords, pattern="8-star")
        assert isinstance(G_directed, nx.DiGraph)

    def test_4star_has_fewer_edges_than_8star(self):
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=5, ny=5)
        G_4star = create_directed_mesh_graph(G_coords, pattern="4-star")
        G_8star = create_directed_mesh_graph(G_coords, pattern="8-star")
        assert len(G_4star.edges) < len(G_8star.edges)

    def test_4star_only_cardinal_edges(self):
        """4-star should only include cardinal (horizontal/vertical) edges."""
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=4, ny=4)
        G_4star = create_directed_mesh_graph(G_coords, pattern="4-star")

        # In a 4x4 grid, 4-star adjacency means each node connects only to
        # horizontal/vertical neighbours
        # For a 4x4 grid: 2 * (4*3 + 3*4) = 2 * 24 = 48 directed edges
        expected_edges = 2 * (4 * 3 + 3 * 4)
        assert len(G_4star.edges) == expected_edges

    def test_8star_includes_diagonal_edges(self):
        """8-star should include both cardinal and diagonal edges."""
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=4, ny=4)
        G_8star = create_directed_mesh_graph(G_coords, pattern="8-star")

        # Cardinal: 2 * (4*3 + 3*4) = 48
        # Diagonal: 2 * 2 * (3*3) = 36
        expected_edges = 48 + 36
        assert len(G_8star.edges) == expected_edges

    def test_edges_have_len_and_vdiff(self):
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=4, ny=4)
        G_directed = create_directed_mesh_graph(G_coords, pattern="8-star")
        for u, v, d in G_directed.edges(data=True):
            assert "len" in d
            assert "vdiff" in d
            assert d["len"] > 0

    def test_invalid_pattern_raises_error(self):
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=4, ny=4)
        with pytest.raises(ValueError, match="Unknown connectivity pattern"):
            create_directed_mesh_graph(G_coords, pattern="6-star")

    def test_preserves_graph_attributes(self):
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=4, ny=4)
        G_directed = create_directed_mesh_graph(G_coords, pattern="8-star")
        assert "dx" in G_directed.graph
        assert "dy" in G_directed.graph

    def test_bidirectional_edges(self):
        """Each undirected edge should produce two directed edges."""
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=3, ny=3)
        G_directed = create_directed_mesh_graph(G_coords, pattern="4-star")
        for u, v in G_directed.edges():
            assert G_directed.has_edge(v, u), f"Missing reverse edge ({v}, {u})"


class TestFlatSinglescaleFromCoordinates:
    """Tests for create_flat_singlescale_from_coordinates."""

    def test_basic_creation(self):
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=5, ny=5)
        G = create_flat_singlescale_from_coordinates(G_coords, pattern="8-star")
        assert isinstance(G, nx.DiGraph)

    def test_4star_pattern(self):
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=5, ny=5)
        G = create_flat_singlescale_from_coordinates(G_coords, pattern="4-star")
        assert isinstance(G, nx.DiGraph)
        # Fewer edges than 8-star
        G_8 = create_flat_singlescale_from_coordinates(G_coords, pattern="8-star")
        assert len(G.edges) < len(G_8.edges)


class TestFlatMultiscaleFromCoordinates:
    """Tests for create_flat_multiscale_from_coordinates."""

    def test_basic_creation(self):
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        G = create_flat_multiscale_from_coordinates(G_coords_list)
        assert isinstance(G, nx.DiGraph)

    def test_pattern_argument(self):
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        G_4star = create_flat_multiscale_from_coordinates(
            G_coords_list,
            pattern="4-star",
        )
        G_8star = create_flat_multiscale_from_coordinates(
            G_coords_list,
            pattern="8-star",
        )
        assert len(G_4star.edges) < len(G_8star.edges)


class TestHierarchicalFromCoordinates:
    """Tests for create_hierarchical_from_coordinates."""

    def test_basic_creation(self):
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        G = create_hierarchical_from_coordinates(G_coords_list)
        assert isinstance(G, nx.DiGraph)

    def test_has_up_down_same_edges(self):
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        G = create_hierarchical_from_coordinates(G_coords_list)
        directions = set()
        for u, v, d in G.edges(data=True):
            if "direction" in d:
                directions.add(d["direction"])
        assert "same" in directions
        assert "up" in directions
        assert "down" in directions

    def test_intra_level_pattern(self):
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        G_4star = create_hierarchical_from_coordinates(
            G_coords_list,
            intra_level={"pattern": "4-star"},
        )
        G_8star = create_hierarchical_from_coordinates(
            G_coords_list,
            intra_level={"pattern": "8-star"},
        )
        assert len(G_4star.edges) < len(G_8star.edges)

    def test_inter_level_k_parameter(self):
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        G_k1 = create_hierarchical_from_coordinates(
            G_coords_list,
            inter_level={"pattern": "nearest", "k": 1},
        )
        G_k3 = create_hierarchical_from_coordinates(
            G_coords_list,
            inter_level={"pattern": "nearest", "k": 3},
        )
        # More neighbours → more inter-level edges
        assert len(G_k3.edges) > len(G_k1.edges)

    def test_invalid_inter_level_pattern_raises(self):
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        with pytest.raises(NotImplementedError, match="Inter-level pattern"):
            create_hierarchical_from_coordinates(
                G_coords_list,
                inter_level={"pattern": "some_unknown"},
            )


# ====================
# New API via create_all_graph_components tests
# ====================


class TestNewAPIFlat:
    """Tests for create_all_graph_components with new mesh_layout API (flat)."""

    def test_flat_with_new_api(self):
        xy = test_utils.create_fake_xy(N=32)
        graph = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(mesh_node_spacing=3),
            m2m_connectivity_kwargs=dict(pattern="8-star"),
            g2m_connectivity="within_radius",
            m2g_connectivity="nearest_neighbours",
            g2m_connectivity_kwargs=dict(rel_max_dist=0.51),
            m2g_connectivity_kwargs=dict(max_num_neighbours=4),
        )
        assert isinstance(graph, nx.DiGraph)

    def test_flat_4star_pattern(self):
        xy = test_utils.create_fake_xy(N=32)
        graph_4 = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(mesh_node_spacing=3),
            m2m_connectivity_kwargs=dict(pattern="4-star"),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        graph_8 = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(mesh_node_spacing=3),
            m2m_connectivity_kwargs=dict(pattern="8-star"),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        assert isinstance(graph_4, nx.DiGraph)
        assert isinstance(graph_8, nx.DiGraph)
        assert len(graph_4.edges) < len(graph_8.edges)

    def test_missing_mesh_node_spacing_raises(self):
        xy = test_utils.create_fake_xy(N=32)
        with pytest.raises(ValueError, match="mesh_node_spacing"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat",
                mesh_layout="rectilinear",
                mesh_layout_kwargs={},
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )


class TestNewAPIFlatMultiscale:
    """Tests for create_all_graph_components with new API (flat_multiscale)."""

    def test_flat_multiscale_with_new_api(self):
        xy = test_utils.create_fake_xy(N=32)
        graph = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat_multiscale",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            m2m_connectivity_kwargs=dict(
                pattern="8-star",
            ),
            g2m_connectivity="within_radius",
            m2g_connectivity="nearest_neighbours",
            g2m_connectivity_kwargs=dict(rel_max_dist=0.51),
            m2g_connectivity_kwargs=dict(max_num_neighbours=4),
        )
        assert isinstance(graph, nx.DiGraph)

    def test_flat_multiscale_4star_pattern(self):
        xy = test_utils.create_fake_xy(N=32)
        graph = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat_multiscale",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            m2m_connectivity_kwargs=dict(
                pattern="4-star",
            ),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        assert isinstance(graph, nx.DiGraph)


class TestNewAPIHierarchical:
    """Tests for create_all_graph_components with new API (hierarchical)."""

    def test_hierarchical_with_new_api(self):
        xy = test_utils.create_fake_xy(N=32)
        graph = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="hierarchical",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            m2m_connectivity_kwargs=dict(
                intra_level=dict(pattern="8-star"),
                inter_level=dict(pattern="nearest", k=1),
            ),
            g2m_connectivity="within_radius",
            m2g_connectivity="nearest_neighbours",
            g2m_connectivity_kwargs=dict(rel_max_dist=0.51),
            m2g_connectivity_kwargs=dict(max_num_neighbours=4),
        )
        assert isinstance(graph, nx.DiGraph)

    def test_hierarchical_4star_intra(self):
        xy = test_utils.create_fake_xy(N=32)
        graph = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="hierarchical",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            m2m_connectivity_kwargs=dict(
                intra_level=dict(pattern="4-star"),
                inter_level=dict(pattern="nearest", k=1),
            ),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        assert isinstance(graph, nx.DiGraph)

    def test_hierarchical_k3_nearest(self):
        xy = test_utils.create_fake_xy(N=32)
        graph = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="hierarchical",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            m2m_connectivity_kwargs=dict(
                intra_level=dict(pattern="8-star"),
                inter_level=dict(pattern="nearest", k=3),
            ),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        assert isinstance(graph, nx.DiGraph)


# ====================
# Backward compatibility tests
# ====================


class TestBackwardCompatibility:
    """Tests that old-style kwargs still work with deprecation warnings."""

    def test_old_style_flat_with_mesh_node_distance(self):
        xy = test_utils.create_fake_xy(N=32)
        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{message}", level="WARNING")
        try:
            graph = wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat",
                mesh_layout="rectilinear",
                m2m_connectivity_kwargs=dict(mesh_node_distance=3),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )
        finally:
            logger.remove(handler_id)
        assert "mesh_node_distance" in log_output.getvalue()
        assert isinstance(graph, nx.DiGraph)

    def test_old_style_flat_multiscale(self):
        xy = test_utils.create_fake_xy(N=32)
        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{message}", level="WARNING")
        try:
            graph = wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat_multiscale",
                mesh_layout="rectilinear",
                m2m_connectivity_kwargs=dict(
                    mesh_node_distance=3,
                    level_refinement_factor=3,
                    max_num_levels=3,
                ),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )
        finally:
            logger.remove(handler_id)
        log_text = log_output.getvalue()
        assert "mesh_node_distance" in log_text
        assert "level_refinement_factor" in log_text
        assert "max_num_levels" in log_text
        assert isinstance(graph, nx.DiGraph)

    def test_old_style_hierarchical(self):
        xy = test_utils.create_fake_xy(N=32)
        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{message}", level="WARNING")
        try:
            graph = wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="hierarchical",
                mesh_layout="rectilinear",
                m2m_connectivity_kwargs=dict(
                    mesh_node_distance=3,
                    level_refinement_factor=3,
                    max_num_levels=3,
                ),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )
        finally:
            logger.remove(handler_id)
        log_text = log_output.getvalue()
        assert "mesh_node_distance" in log_text
        assert "level_refinement_factor" in log_text
        assert "max_num_levels" in log_text
        assert isinstance(graph, nx.DiGraph)

    def test_kwargs_dict_not_mutated(self):
        """Verify that passing dict kwargs doesn't mutate the caller's dict."""
        xy = test_utils.create_fake_xy(N=32)
        original_kwargs = dict(
            mesh_node_distance=3,
            level_refinement_factor=3,
            max_num_levels=3,
        )
        kwargs_copy = original_kwargs.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="hierarchical",
                mesh_layout="rectilinear",
                m2m_connectivity_kwargs=original_kwargs,
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )

        # The original dict should not be modified
        assert original_kwargs == kwargs_copy


# ====================
# Error handling tests
# ====================


class TestErrorHandling:
    """Tests for proper error handling."""

    def test_unsupported_mesh_layout_raises(self):
        xy = test_utils.create_fake_xy(N=32)
        with pytest.raises(NotImplementedError, match="not yet supported"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat",
                mesh_layout="triangular",
                mesh_layout_kwargs=dict(mesh_node_spacing=3),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )

    def test_unsupported_m2m_connectivity_raises(self):
        xy = test_utils.create_fake_xy(N=32)
        with pytest.raises(NotImplementedError, match="not implemented"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="some_unknown",
                mesh_layout="rectilinear",
                mesh_layout_kwargs=dict(mesh_node_spacing=3),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )

    def test_mesh_node_spacing_too_large_raises(self):
        xy = test_utils.create_fake_xy(N=10)
        with pytest.raises(ValueError, match="too large"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat",
                mesh_layout="rectilinear",
                mesh_layout_kwargs=dict(mesh_node_spacing=100),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )


# ====================
# Equivalence tests: new API == old wrappers
# ====================


class TestEquivalence:
    """Verify that the new API produces equivalent results to the old wrappers."""

    def test_keisler_archetype_matches_new_api(self):
        """The keisler archetype function should produce the same result as
        calling create_all_graph_components with the new API directly."""
        xy = test_utils.create_fake_xy(N=32)

        graph_archetype = wmg.create.archetype.create_keisler_graph(
            coords=xy, mesh_node_distance=3
        )

        graph_new_api = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(mesh_node_spacing=3),
            m2m_connectivity_kwargs=dict(pattern="8-star"),
            g2m_connectivity="within_radius",
            m2g_connectivity="nearest_neighbours",
            g2m_connectivity_kwargs=dict(rel_max_dist=0.51),
            m2g_connectivity_kwargs=dict(max_num_neighbours=4),
        )

        assert len(graph_archetype.nodes) == len(graph_new_api.nodes)
        assert len(graph_archetype.edges) == len(graph_new_api.edges)

    def test_graphcast_archetype_matches_new_api(self):
        xy = test_utils.create_fake_xy(N=32)

        graph_archetype = wmg.create.archetype.create_graphcast_graph(
            coords=xy,
            mesh_node_distance=3,
            level_refinement_factor=3,
            max_num_levels=3,
        )

        graph_new_api = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat_multiscale",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            m2m_connectivity_kwargs=dict(
                pattern="8-star",
            ),
            g2m_connectivity="within_radius",
            m2g_connectivity="nearest_neighbours",
            g2m_connectivity_kwargs=dict(rel_max_dist=0.51),
            m2g_connectivity_kwargs=dict(max_num_neighbours=4),
        )

        assert len(graph_archetype.nodes) == len(graph_new_api.nodes)
        assert len(graph_archetype.edges) == len(graph_new_api.edges)

    def test_oskarsson_archetype_matches_new_api(self):
        xy = test_utils.create_fake_xy(N=32)

        graph_archetype = wmg.create.archetype.create_oskarsson_hierarchical_graph(
            coords=xy,
            mesh_node_distance=3,
            level_refinement_factor=3,
            max_num_levels=3,
        )

        graph_new_api = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="hierarchical",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            m2m_connectivity_kwargs=dict(
                intra_level=dict(pattern="8-star"),
                inter_level=dict(pattern="nearest", k=1),
            ),
            g2m_connectivity="within_radius",
            m2g_connectivity="nearest_neighbours",
            g2m_connectivity_kwargs=dict(rel_max_dist=0.51),
            m2g_connectivity_kwargs=dict(max_num_neighbours=4),
        )

        assert len(graph_archetype.nodes) == len(graph_new_api.nodes)
        assert len(graph_archetype.edges) == len(graph_new_api.edges)


# ====================
# Edge case tests
# ====================


class TestCoordinateCreationEdgeCases:
    """Edge cases for coordinate creation step."""

    def test_minimum_grid_2x2(self):
        """Smallest possible grid: 2x2 nodes."""
        xy = test_utils.create_fake_xy(N=10)
        G = create_single_level_2d_mesh_primitive(xy, nx=2, ny=2)
        assert len(G.nodes) == 4
        # 2x2 grid: cardinal edges = 2*(2*1) = 4, diagonal edges = 2*(1*1) = 2
        cardinal = sum(
            1 for _, _, d in G.edges(data=True) if d["adjacency_type"] == "cardinal"
        )
        diagonal = sum(
            1 for _, _, d in G.edges(data=True) if d["adjacency_type"] == "diagonal"
        )
        assert cardinal == 4
        assert diagonal == 2

    def test_single_row_grid(self):
        """Grid with only 1 row (nx=5, ny=1)."""
        xy = test_utils.create_fake_xy(N=10)
        G = create_single_level_2d_mesh_primitive(xy, nx=5, ny=1)
        assert len(G.nodes) == 5
        # 5x1 grid: only horizontal cardinal edges, no diagonals
        cardinal = sum(
            1 for _, _, d in G.edges(data=True) if d["adjacency_type"] == "cardinal"
        )
        diagonal = sum(
            1 for _, _, d in G.edges(data=True) if d["adjacency_type"] == "diagonal"
        )
        assert cardinal == 4  # 5-1 = 4 horizontal edges
        assert diagonal == 0

    def test_single_column_grid(self):
        """Grid with only 1 column (nx=1, ny=5)."""
        xy = test_utils.create_fake_xy(N=10)
        G = create_single_level_2d_mesh_primitive(xy, nx=1, ny=5)
        assert len(G.nodes) == 5
        cardinal = sum(
            1 for _, _, d in G.edges(data=True) if d["adjacency_type"] == "cardinal"
        )
        diagonal = sum(
            1 for _, _, d in G.edges(data=True) if d["adjacency_type"] == "diagonal"
        )
        assert cardinal == 4  # 5-1 = 4 vertical edges
        assert diagonal == 0

    def test_1x1_grid_no_edges(self):
        """Grid with a single node (1x1): should have no edges."""
        xy = test_utils.create_fake_xy(N=10)
        G = create_single_level_2d_mesh_primitive(xy, nx=1, ny=1)
        assert len(G.nodes) == 1
        assert len(G.edges) == 0

    def test_large_grid(self):
        """Larger grid should still work correctly."""
        xy = test_utils.create_fake_xy(N=50)
        G = create_single_level_2d_mesh_primitive(xy, nx=10, ny=10)
        assert len(G.nodes) == 100
        expected_cardinal = 2 * (10 * 9)  # 180
        expected_diagonal = 2 * (9 * 9)  # 162
        cardinal = sum(
            1 for _, _, d in G.edges(data=True) if d["adjacency_type"] == "cardinal"
        )
        diagonal = sum(
            1 for _, _, d in G.edges(data=True) if d["adjacency_type"] == "diagonal"
        )
        assert cardinal == expected_cardinal
        assert diagonal == expected_diagonal

    def test_node_positions_within_bounds(self):
        """Node positions should be within the xy bounds."""
        xy = test_utils.create_fake_xy(N=20)
        G = create_single_level_2d_mesh_primitive(xy, nx=5, ny=5)
        x_min, y_min = np.amin(xy, axis=0)
        x_max, y_max = np.amax(xy, axis=0)
        for node in G.nodes:
            pos = G.nodes[node]["pos"]
            assert pos[0] >= x_min and pos[0] <= x_max
            assert pos[1] >= y_min and pos[1] <= y_max

    def test_multirange_with_max_levels_1(self):
        """Multi-range with max_num_levels=1 should return single-level list."""
        xy = test_utils.create_fake_xy(N=30)
        G_list = create_multirange_2d_mesh_primitives(
            max_num_levels=1, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        assert len(G_list) == 1
        assert G_list[0].graph["level"] == 0

    def test_multirange_with_none_max_levels(self):
        """max_num_levels=None should auto-compute levels."""
        xy = test_utils.create_fake_xy(N=30)
        G_list = create_multirange_2d_mesh_primitives(
            max_num_levels=None, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        assert isinstance(G_list, list)
        assert len(G_list) >= 1

    def test_multirange_refinement_factor_5(self):
        """Test with a different refinement factor."""
        xy = test_utils.create_fake_xy(N=50)
        G_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=5
        )
        if len(G_list) >= 2:
            for i in range(len(G_list) - 1):
                assert len(G_list[i].nodes) > len(G_list[i + 1].nodes)


class TestConnectivityCreationEdgeCases:
    """Edge cases for connectivity creation step."""

    def test_directed_graph_from_1x1(self):
        """Creating directed graph from a single-node coordinate graph."""
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=1, ny=1)
        G_directed = create_directed_mesh_graph(G_coords, pattern="8-star")
        assert isinstance(G_directed, nx.DiGraph)
        assert len(G_directed.nodes) == 1
        assert len(G_directed.edges) == 0

    def test_directed_graph_from_2x1(self):
        """Creating directed graph from a 2x1 grid."""
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=2, ny=1)
        G_4star = create_directed_mesh_graph(G_coords, pattern="4-star")
        G_8star = create_directed_mesh_graph(G_coords, pattern="8-star")
        # 2x1: 1 edge, both patterns should have same (no diagonals possible)
        assert len(G_4star.edges) == 2  # bidirectional
        assert len(G_8star.edges) == 2

    def test_4star_is_subset_of_8star(self):
        """All edges in 4-star should exist in 8-star."""
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=5, ny=5)
        G_4star = create_directed_mesh_graph(G_coords, pattern="4-star")
        G_8star = create_directed_mesh_graph(G_coords, pattern="8-star")
        for u, v in G_4star.edges():
            assert G_8star.has_edge(u, v), f"4-star edge ({u},{v}) missing from 8-star"

    def test_edge_lengths_are_positive(self):
        """All edge lengths should be positive."""
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=4, ny=4)
        G = create_directed_mesh_graph(G_coords, pattern="8-star")
        for u, v, d in G.edges(data=True):
            assert d["len"] > 0, f"Edge ({u},{v}) has non-positive length"

    def test_vdiff_antisymmetric(self):
        """vdiff(u,v) should be -vdiff(v,u)."""
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=4, ny=4)
        G = create_directed_mesh_graph(G_coords, pattern="8-star")
        for u, v in G.edges():
            if G.has_edge(v, u):
                np.testing.assert_allclose(
                    G.edges[u, v]["vdiff"], -G.edges[v, u]["vdiff"],
                    err_msg=f"vdiff not antisymmetric for ({u},{v})"
                )

    def test_edge_len_matches_vdiff_norm(self):
        """Edge length should equal the norm of vdiff."""
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_primitive(xy, nx=4, ny=4)
        G = create_directed_mesh_graph(G_coords, pattern="8-star")
        for u, v, d in G.edges(data=True):
            expected_len = np.sqrt(np.sum(d["vdiff"] ** 2))
            np.testing.assert_allclose(
                d["len"], expected_len,
                err_msg=f"Edge ({u},{v}) len doesn't match vdiff norm"
            )

    def test_flat_multiscale_single_level_input(self):
        """Flat multiscale with a single-level list should work (degenerate case)."""
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_primitives(
            max_num_levels=1, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        G = create_flat_multiscale_from_coordinates(G_coords_list, pattern="8-star")
        assert isinstance(G, nx.DiGraph)
        assert len(G.nodes) > 0

    def test_flat_multiscale_4star_vs_8star_edge_count(self):
        """4-star flat_multiscale should have fewer edges than 8-star."""
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        G_4 = create_flat_multiscale_from_coordinates(G_coords_list, pattern="4-star")
        G_8 = create_flat_multiscale_from_coordinates(G_coords_list, pattern="8-star")
        assert len(G_4.edges) < len(G_8.edges)

    def test_hierarchical_single_level_raises(self):
        """Hierarchical with only 1 level should raise ValueError."""
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_primitives(
            max_num_levels=1, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        with pytest.raises(ValueError, match="At least two mesh levels"):
            create_hierarchical_from_coordinates(G_coords_list)

    def test_hierarchical_edge_direction_attributes(self):
        """Every edge in hierarchical graph must have a 'direction' attribute."""
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        G = create_hierarchical_from_coordinates(G_coords_list)
        for u, v, d in G.edges(data=True):
            assert "direction" in d, f"Edge ({u},{v}) missing 'direction'"
            assert d["direction"] in ("same", "up", "down")

    def test_hierarchical_up_down_symmetry(self):
        """For each 'down' edge (u,v), there should be an 'up' edge (v,u)."""
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_primitives(
            max_num_levels=3, xy=xy, mesh_node_spacing=3, interlevel_refinement_factor=3
        )
        G = create_hierarchical_from_coordinates(G_coords_list)
        for u, v, d in G.edges(data=True):
            if d.get("direction") == "down":
                assert G.has_edge(v, u), f"Missing 'up' edge for 'down' ({u},{v})"
                assert G.edges[v, u]["direction"] == "up"


class TestAPIEdgeCases:
    """Edge cases for the public create_all_graph_components API."""

    def test_flat_default_pattern_is_8star(self):
        """When no m2m_connectivity_kwargs given, flat should default to 8-star."""
        xy = test_utils.create_fake_xy(N=32)
        graph_default = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(mesh_node_spacing=3),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        graph_8star = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(mesh_node_spacing=3),
            m2m_connectivity_kwargs=dict(pattern="8-star"),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        assert len(graph_default.edges) == len(graph_8star.edges)

    def test_flat_multiscale_default_pattern_is_8star(self):
        """When no m2m_connectivity_kwargs given, flat_multiscale defaults to 8-star."""
        xy = test_utils.create_fake_xy(N=32)
        graph_default = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat_multiscale",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        graph_8star = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat_multiscale",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            m2m_connectivity_kwargs=dict(pattern="8-star"),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        assert len(graph_default.edges) == len(graph_8star.edges)

    def test_hierarchical_default_kwargs(self):
        """When no m2m_connectivity_kwargs given, hierarchical has sensible defaults."""
        xy = test_utils.create_fake_xy(N=32)
        graph_default = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="hierarchical",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        graph_explicit = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="hierarchical",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            m2m_connectivity_kwargs=dict(
                intra_level=dict(pattern="8-star"),
                inter_level=dict(pattern="nearest", k=1),
            ),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        assert len(graph_default.edges) == len(graph_explicit.edges)

    def test_mesh_layout_is_required(self):
        """When mesh_layout not specified, it should raise TypeError."""
        xy = test_utils.create_fake_xy(N=32)
        with pytest.raises(TypeError, match="mesh_layout"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat",
                mesh_layout_kwargs=dict(mesh_node_spacing=3),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )

    def test_return_components_flat(self):
        """return_components=True should return dict with g2m, m2m, m2g."""
        xy = test_utils.create_fake_xy(N=32)
        components = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(mesh_node_spacing=3),
            m2m_connectivity_kwargs=dict(pattern="8-star"),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert isinstance(components, dict)
        assert "g2m" in components
        assert "m2m" in components
        assert "m2g" in components
        for name, g in components.items():
            assert isinstance(g, nx.DiGraph)

    def test_return_components_hierarchical(self):
        """return_components=True for hierarchical should contain 3 components."""
        xy = test_utils.create_fake_xy(N=32)
        components = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="hierarchical",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            m2m_connectivity_kwargs=dict(
                intra_level=dict(pattern="8-star"),
                inter_level=dict(pattern="nearest", k=1),
            ),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert isinstance(components, dict)
        assert "g2m" in components
        assert "m2m" in components
        assert "m2g" in components

    def test_return_components_flat_multiscale(self):
        """return_components=True for flat_multiscale."""
        xy = test_utils.create_fake_xy(N=32)
        components = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat_multiscale",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            m2m_connectivity_kwargs=dict(pattern="8-star"),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert isinstance(components, dict)
        assert set(components.keys()) == {"g2m", "m2m", "m2g"}

    def test_flat_multiscale_no_sub_dicts_interface(self):
        """Ensure flat_multiscale accepts a simple pattern arg, not sub-dicts."""
        xy = test_utils.create_fake_xy(N=32)
        graph = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat_multiscale",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            m2m_connectivity_kwargs=dict(pattern="8-star"),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        assert isinstance(graph, nx.DiGraph)

    def test_decode_mask_with_new_api(self):
        """decode_mask should work correctly with the new API."""
        xy = test_utils.create_fake_xy(N=32)
        n_points = len(xy)
        mask = [True] * (n_points // 2) + [False] * (n_points - n_points // 2)
        graph = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(mesh_node_spacing=3),
            m2m_connectivity_kwargs=dict(pattern="8-star"),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            decode_mask=mask,
        )
        assert isinstance(graph, nx.DiGraph)


class TestBackwardCompatEdgeCases:
    """Advanced backward compatibility edge cases."""

    def test_old_kwargs_with_flat_multiscale_compat(self):
        """Old-style flat_multiscale kwargs should trigger deprecation warnings
        (via loguru) and be migrated to the new names."""
        xy = test_utils.create_fake_xy(N=32)
        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{message}", level="WARNING")
        try:
            graph = wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat_multiscale",
                mesh_layout="rectilinear",
                m2m_connectivity_kwargs=dict(
                    mesh_node_distance=3,
                    level_refinement_factor=3,
                    max_num_levels=3,
                ),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )
        finally:
            logger.remove(handler_id)
        log_text = log_output.getvalue()
        assert "mesh_node_spacing" in log_text
        assert "refinement_factor" in log_text
        assert "max_num_refinement_levels" in log_text
        assert isinstance(graph, nx.DiGraph)


class TestGraphStructuralProperties:
    """Tests verifying structural properties of generated graphs."""

    def test_all_mesh_nodes_have_pos(self):
        """Every node in the final graph should have 'pos' attribute."""
        xy = test_utils.create_fake_xy(N=32)
        graph = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(mesh_node_spacing=3),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        for node in graph.nodes:
            assert "pos" in graph.nodes[node], f"Node {node} missing 'pos'"

    def test_all_edges_have_component(self):
        """Every edge should have a 'component' attribute ('g2m', 'm2m', 'm2g')."""
        xy = test_utils.create_fake_xy(N=32)
        graph = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(mesh_node_spacing=3),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        for u, v, d in graph.edges(data=True):
            assert "component" in d, f"Edge ({u},{v}) missing 'component'"
            assert d["component"] in ("g2m", "m2m", "m2g")

    def test_all_edges_have_len_and_vdiff(self):
        """Every edge should have 'len' and 'vdiff' attributes."""
        xy = test_utils.create_fake_xy(N=32)
        graph = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(mesh_node_spacing=3),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        for u, v, d in graph.edges(data=True):
            assert "len" in d, f"Edge ({u},{v}) missing 'len'"
            assert "vdiff" in d, f"Edge ({u},{v}) missing 'vdiff'"

    def test_graph_is_directed(self):
        """Final graph should always be a DiGraph."""
        xy = test_utils.create_fake_xy(N=32)
        for connectivity in ["flat"]:
            graph = wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity=connectivity,
                mesh_layout="rectilinear",
                mesh_layout_kwargs=dict(mesh_node_spacing=3),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )
            assert isinstance(graph, nx.DiGraph)

    def test_flat_4star_strictly_fewer_m2m_edges(self):
        """4-star flat should have strictly fewer m2m edges than 8-star."""
        xy = test_utils.create_fake_xy(N=32)
        components_4 = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(mesh_node_spacing=3),
            m2m_connectivity_kwargs=dict(pattern="4-star"),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        components_8 = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(mesh_node_spacing=3),
            m2m_connectivity_kwargs=dict(pattern="8-star"),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        # m2m component specifically should differ
        assert len(components_4["m2m"].edges) < len(components_8["m2m"].edges)
        # g2m and m2g should be the same (same grid spacing, same connectivity)
        assert len(components_4["g2m"].edges) == len(components_8["g2m"].edges)
        assert len(components_4["m2g"].edges) == len(components_8["m2g"].edges)

    def test_hierarchical_has_same_up_down_edge_count(self):
        """Hierarchical graph should have equal number of up and down edges."""
        xy = test_utils.create_fake_xy(N=32)
        graph = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="hierarchical",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                mesh_node_spacing=3,
                refinement_factor=3,
                max_num_refinement_levels=3,
            ),
            m2m_connectivity_kwargs=dict(
                intra_level=dict(pattern="8-star"),
                inter_level=dict(pattern="nearest", k=1),
            ),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        m2m = graph["m2m"]
        up_count = sum(
            1 for _, _, d in m2m.edges(data=True) if d.get("direction") == "up"
        )
        down_count = sum(
            1 for _, _, d in m2m.edges(data=True) if d.get("direction") == "down"
        )
        assert up_count == down_count, (
            f"Up edges ({up_count}) != Down edges ({down_count})"
        )
        assert up_count > 0, "Should have at least some up/down edges"
