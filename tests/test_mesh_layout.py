"""
Tests for the mesh_layout parameter and two-step coordinate/connectivity
architecture introduced in Issue #78.

These tests verify:
1. The new API (mesh_layout, mesh_layout_kwargs, m2m_connectivity_kwargs with
   intra_level/inter_level sub-dicts)
2. The two-step process (coordinate creation → connectivity creation)
3. The 4-star vs 8-star pattern functionality
4. Backward compatibility with old-style kwargs
5. Edge annotations on coordinate graphs
6. Error handling for invalid inputs
"""

import warnings

import networkx as nx
import numpy as np
import pytest

import tests.utils as test_utils
import weather_model_graphs as wmg
from weather_model_graphs.create.mesh.mesh import (
    create_directed_mesh_graph,
    create_multirange_2d_mesh_coordinates,
    create_single_level_2d_mesh_coordinates,
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
    """Tests for create_single_level_2d_mesh_coordinates."""

    def test_returns_undirected_graph(self):
        xy = test_utils.create_fake_xy(N=10)
        G = create_single_level_2d_mesh_coordinates(xy, nx=5, ny=5)
        assert isinstance(G, nx.Graph)
        assert not isinstance(G, nx.DiGraph)

    def test_nodes_have_pos_and_type(self):
        xy = test_utils.create_fake_xy(N=10)
        G = create_single_level_2d_mesh_coordinates(xy, nx=4, ny=4)
        for node in G.nodes:
            assert "pos" in G.nodes[node]
            assert "type" in G.nodes[node]
            assert G.nodes[node]["type"] == "mesh"
            assert len(G.nodes[node]["pos"]) == 2

    def test_correct_number_of_nodes(self):
        xy = test_utils.create_fake_xy(N=10)
        G = create_single_level_2d_mesh_coordinates(xy, nx=5, ny=4)
        assert len(G.nodes) == 5 * 4

    def test_edges_have_adjacency_type(self):
        xy = test_utils.create_fake_xy(N=10)
        G = create_single_level_2d_mesh_coordinates(xy, nx=5, ny=5)
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
        G = create_single_level_2d_mesh_coordinates(xy, nx=5, ny=5)
        assert "dx" in G.graph
        assert "dy" in G.graph
        assert G.graph["dx"] > 0
        assert G.graph["dy"] > 0


class TestMultirangeCoordinateCreation:
    """Tests for create_multirange_2d_mesh_coordinates."""

    def test_returns_list_of_undirected_graphs(self):
        xy = test_utils.create_fake_xy(N=30)
        G_list = create_multirange_2d_mesh_coordinates(
            max_num_levels=3, xy=xy, grid_spacing=3, interlevel_refinement_factor=3
        )
        assert isinstance(G_list, list)
        assert len(G_list) > 0
        for G in G_list:
            assert isinstance(G, nx.Graph)
            assert not isinstance(G, nx.DiGraph)

    def test_each_level_has_level_attribute(self):
        xy = test_utils.create_fake_xy(N=30)
        G_list = create_multirange_2d_mesh_coordinates(
            max_num_levels=3, xy=xy, grid_spacing=3, interlevel_refinement_factor=3
        )
        for i, G in enumerate(G_list):
            assert G.graph["level"] == i
            for node in G.nodes:
                assert G.nodes[node]["level"] == i

    def test_interlevel_refinement_factor_stored(self):
        xy = test_utils.create_fake_xy(N=30)
        G_list = create_multirange_2d_mesh_coordinates(
            max_num_levels=3, xy=xy, grid_spacing=3, interlevel_refinement_factor=3
        )
        for G in G_list:
            assert G.graph["interlevel_refinement_factor"] == 3

    def test_edges_have_adjacency_type(self):
        xy = test_utils.create_fake_xy(N=30)
        G_list = create_multirange_2d_mesh_coordinates(
            max_num_levels=2, xy=xy, grid_spacing=3, interlevel_refinement_factor=3
        )
        for G in G_list:
            for u, v, d in G.edges(data=True):
                assert "adjacency_type" in d

    def test_coarser_levels_have_fewer_nodes(self):
        xy = test_utils.create_fake_xy(N=30)
        G_list = create_multirange_2d_mesh_coordinates(
            max_num_levels=3, xy=xy, grid_spacing=3, interlevel_refinement_factor=3
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
        G_coords = create_single_level_2d_mesh_coordinates(xy, nx=4, ny=4)
        G_directed = create_directed_mesh_graph(G_coords, pattern="8-star")
        assert isinstance(G_directed, nx.DiGraph)

    def test_4star_has_fewer_edges_than_8star(self):
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_coordinates(xy, nx=5, ny=5)
        G_4star = create_directed_mesh_graph(G_coords, pattern="4-star")
        G_8star = create_directed_mesh_graph(G_coords, pattern="8-star")
        assert len(G_4star.edges) < len(G_8star.edges)

    def test_4star_only_cardinal_edges(self):
        """4-star should only include cardinal (horizontal/vertical) edges."""
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_coordinates(xy, nx=4, ny=4)
        G_4star = create_directed_mesh_graph(G_coords, pattern="4-star")

        # In a 4x4 grid, 4-star adjacency means each node connects only to
        # horizontal/vertical neighbours
        # For a 4x4 grid: 2 * (4*3 + 3*4) = 2 * 24 = 48 directed edges
        expected_edges = 2 * (4 * 3 + 3 * 4)
        assert len(G_4star.edges) == expected_edges

    def test_8star_includes_diagonal_edges(self):
        """8-star should include both cardinal and diagonal edges."""
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_coordinates(xy, nx=4, ny=4)
        G_8star = create_directed_mesh_graph(G_coords, pattern="8-star")

        # Cardinal: 2 * (4*3 + 3*4) = 48
        # Diagonal: 2 * 2 * (3*3) = 36
        expected_edges = 48 + 36
        assert len(G_8star.edges) == expected_edges

    def test_edges_have_len_and_vdiff(self):
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_coordinates(xy, nx=4, ny=4)
        G_directed = create_directed_mesh_graph(G_coords, pattern="8-star")
        for u, v, d in G_directed.edges(data=True):
            assert "len" in d
            assert "vdiff" in d
            assert d["len"] > 0

    def test_invalid_pattern_raises_error(self):
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_coordinates(xy, nx=4, ny=4)
        with pytest.raises(ValueError, match="Unknown connectivity pattern"):
            create_directed_mesh_graph(G_coords, pattern="6-star")

    def test_preserves_graph_attributes(self):
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_coordinates(xy, nx=4, ny=4)
        G_directed = create_directed_mesh_graph(G_coords, pattern="8-star")
        assert "dx" in G_directed.graph
        assert "dy" in G_directed.graph

    def test_bidirectional_edges(self):
        """Each undirected edge should produce two directed edges."""
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_coordinates(xy, nx=3, ny=3)
        G_directed = create_directed_mesh_graph(G_coords, pattern="4-star")
        for u, v in G_directed.edges():
            assert G_directed.has_edge(v, u), f"Missing reverse edge ({v}, {u})"


class TestFlatSinglescaleFromCoordinates:
    """Tests for create_flat_singlescale_from_coordinates."""

    def test_basic_creation(self):
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_coordinates(xy, nx=5, ny=5)
        G = create_flat_singlescale_from_coordinates(G_coords, pattern="8-star")
        assert isinstance(G, nx.DiGraph)

    def test_4star_pattern(self):
        xy = test_utils.create_fake_xy(N=10)
        G_coords = create_single_level_2d_mesh_coordinates(xy, nx=5, ny=5)
        G = create_flat_singlescale_from_coordinates(G_coords, pattern="4-star")
        assert isinstance(G, nx.DiGraph)
        # Fewer edges than 8-star
        G_8 = create_flat_singlescale_from_coordinates(G_coords, pattern="8-star")
        assert len(G.edges) < len(G_8.edges)


class TestFlatMultiscaleFromCoordinates:
    """Tests for create_flat_multiscale_from_coordinates."""

    def test_basic_creation(self):
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_coordinates(
            max_num_levels=3, xy=xy, grid_spacing=3, interlevel_refinement_factor=3
        )
        G = create_flat_multiscale_from_coordinates(G_coords_list)
        assert isinstance(G, nx.DiGraph)

    def test_intra_level_pattern(self):
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_coordinates(
            max_num_levels=3, xy=xy, grid_spacing=3, interlevel_refinement_factor=3
        )
        G_4star = create_flat_multiscale_from_coordinates(
            G_coords_list,
            intra_level={"pattern": "4-star"},
            inter_level={"pattern": "coincident"},
        )
        G_8star = create_flat_multiscale_from_coordinates(
            G_coords_list,
            intra_level={"pattern": "8-star"},
            inter_level={"pattern": "coincident"},
        )
        assert len(G_4star.edges) < len(G_8star.edges)

    def test_invalid_inter_level_pattern_raises(self):
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_coordinates(
            max_num_levels=3, xy=xy, grid_spacing=3, interlevel_refinement_factor=3
        )
        with pytest.raises(NotImplementedError, match="Inter-level pattern"):
            create_flat_multiscale_from_coordinates(
                G_coords_list,
                inter_level={"pattern": "some_unknown"},
            )


class TestHierarchicalFromCoordinates:
    """Tests for create_hierarchical_from_coordinates."""

    def test_basic_creation(self):
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_coordinates(
            max_num_levels=3, xy=xy, grid_spacing=3, interlevel_refinement_factor=3
        )
        G = create_hierarchical_from_coordinates(G_coords_list)
        assert isinstance(G, nx.DiGraph)

    def test_has_up_down_same_edges(self):
        xy = test_utils.create_fake_xy(N=30)
        G_coords_list = create_multirange_2d_mesh_coordinates(
            max_num_levels=3, xy=xy, grid_spacing=3, interlevel_refinement_factor=3
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
        G_coords_list = create_multirange_2d_mesh_coordinates(
            max_num_levels=3, xy=xy, grid_spacing=3, interlevel_refinement_factor=3
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
        G_coords_list = create_multirange_2d_mesh_coordinates(
            max_num_levels=3, xy=xy, grid_spacing=3, interlevel_refinement_factor=3
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
        G_coords_list = create_multirange_2d_mesh_coordinates(
            max_num_levels=3, xy=xy, grid_spacing=3, interlevel_refinement_factor=3
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
            mesh_layout_kwargs=dict(grid_spacing=3),
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
            mesh_layout_kwargs=dict(grid_spacing=3),
            m2m_connectivity_kwargs=dict(pattern="4-star"),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        graph_8 = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(grid_spacing=3),
            m2m_connectivity_kwargs=dict(pattern="8-star"),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        assert isinstance(graph_4, nx.DiGraph)
        assert isinstance(graph_8, nx.DiGraph)
        assert len(graph_4.edges) < len(graph_8.edges)

    def test_missing_grid_spacing_raises(self):
        xy = test_utils.create_fake_xy(N=32)
        with pytest.raises(ValueError, match="grid_spacing"):
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
                grid_spacing=3,
                interlevel_refinement_factor=3,
                max_num_levels=3,
            ),
            m2m_connectivity_kwargs=dict(
                intra_level=dict(pattern="8-star"),
                inter_level=dict(pattern="coincident"),
            ),
            g2m_connectivity="within_radius",
            m2g_connectivity="nearest_neighbours",
            g2m_connectivity_kwargs=dict(rel_max_dist=0.51),
            m2g_connectivity_kwargs=dict(max_num_neighbours=4),
        )
        assert isinstance(graph, nx.DiGraph)

    def test_flat_multiscale_4star_intra(self):
        xy = test_utils.create_fake_xy(N=32)
        graph = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat_multiscale",
            mesh_layout="rectilinear",
            mesh_layout_kwargs=dict(
                grid_spacing=3,
                interlevel_refinement_factor=3,
                max_num_levels=3,
            ),
            m2m_connectivity_kwargs=dict(
                intra_level=dict(pattern="4-star"),
                inter_level=dict(pattern="coincident"),
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
                grid_spacing=3,
                interlevel_refinement_factor=3,
                max_num_levels=3,
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
                grid_spacing=3,
                interlevel_refinement_factor=3,
                max_num_levels=3,
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
                grid_spacing=3,
                interlevel_refinement_factor=3,
                max_num_levels=3,
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
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            graph = wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat",
                m2m_connectivity_kwargs=dict(mesh_node_distance=3),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )
            # Should have deprecation warning
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "mesh_node_distance" in str(deprecation_warnings[0].message)
        assert isinstance(graph, nx.DiGraph)

    def test_old_style_flat_multiscale(self):
        xy = test_utils.create_fake_xy(N=32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            graph = wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat_multiscale",
                m2m_connectivity_kwargs=dict(
                    mesh_node_distance=3,
                    level_refinement_factor=3,
                    max_num_levels=3,
                ),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 3  # 3 migrated kwargs
        assert isinstance(graph, nx.DiGraph)

    def test_old_style_hierarchical(self):
        xy = test_utils.create_fake_xy(N=32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            graph = wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="hierarchical",
                m2m_connectivity_kwargs=dict(
                    mesh_node_distance=3,
                    level_refinement_factor=3,
                    max_num_levels=3,
                ),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 3
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
                mesh_layout_kwargs=dict(grid_spacing=3),
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
                mesh_layout_kwargs=dict(grid_spacing=3),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )

    def test_grid_spacing_too_large_raises(self):
        xy = test_utils.create_fake_xy(N=10)
        with pytest.raises(ValueError, match="too large"):
            wmg.create.create_all_graph_components(
                coords=xy,
                m2m_connectivity="flat",
                mesh_layout="rectilinear",
                mesh_layout_kwargs=dict(grid_spacing=100),
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
            mesh_layout_kwargs=dict(grid_spacing=3),
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
                grid_spacing=3,
                interlevel_refinement_factor=3,
                max_num_levels=3,
            ),
            m2m_connectivity_kwargs=dict(
                intra_level=dict(pattern="8-star"),
                inter_level=dict(pattern="coincident"),
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
                grid_spacing=3,
                interlevel_refinement_factor=3,
                max_num_levels=3,
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
