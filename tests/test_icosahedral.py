from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial import KDTree

import tests.utils as test_utils
from weather_model_graphs.create import create_all_graph_components
from weather_model_graphs.create.mesh.layouts.icosahedral import (  # refinement_level_from_grid_spacing,
    cartesian_to_lat_lon,
    compute_max_edge_length,
    connect_grid_to_mesh,
    connect_mesh_to_grid,
    create_flat_icosahedral_mesh_graph,
    create_hierarchical_icosahedral_mesh_graph,
    create_hierarchy_of_icosahedral_meshes,
    find_containing_triangle,
    generate_icosahedral_mesh,
    lat_lon_to_cartesian,
)

# try:
#     import trimesh
# except ImportError:
#     pytest.skip("trimesh not installed", allow_module_level=True)
pytest.importorskip("trimesh")


@pytest.fixture
def sample_grid_1deg():
    lats = np.arange(-90, 91, 1)
    lons = np.arange(-180, 181, 1)
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    return np.column_stack([grid_lat.ravel(), grid_lon.ravel()])


@pytest.fixture
def sample_grid_5deg():
    lats = np.arange(-90, 91, 5)
    lons = np.arange(-180, 181, 5)
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    return np.column_stack([grid_lat.ravel(), grid_lon.ravel()])


@pytest.fixture
def sample_grid_10deg():
    lats = np.arange(-90, 91, 10)
    lons = np.arange(-180, 181, 10)
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    return np.column_stack([grid_lat.ravel(), grid_lon.ravel()])


@pytest.fixture
def geographic_crs():
    return "EPSG:4326"


@pytest.fixture
def projected_crs():
    return "EPSG:32633"


class TestIcosahedralMeshGeneration:
    def test_generate_base_icosahedron(self):
        vertices, faces = generate_icosahedral_mesh(refinement_level=0)
        assert len(vertices) == 12
        assert len(faces) == 20
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)
        assert faces.shape[1] == 3

    def test_generate_refined_icosahedron(self):
        for subdivisions in [1, 2, 3]:
            vertices, faces = generate_icosahedral_mesh(refinement_level=subdivisions)
            expected_vertices = 2 + 10 * 4**subdivisions
            assert len(vertices) == expected_vertices
            expected_faces = 20 * 4**subdivisions
            assert len(faces) == expected_faces
            norms = np.linalg.norm(vertices, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-10)

    def test_create_hierarchy(self):
        max_subdivisions = 3
        mesh_list = create_hierarchy_of_icosahedral_meshes(max_subdivisions)
        assert len(mesh_list) == max_subdivisions + 1
        for level, (vertices, faces) in enumerate(mesh_list):
            expected_vertices = 2 + 10 * 4**level
            expected_faces = 20 * 4**level
            assert len(vertices) == expected_vertices
            assert len(faces) == expected_faces


class TestIcosahedralMeshGraphs:
    def test_flat_icosahedral_graph(self):
        subdivisions = 2
        G = create_flat_icosahedral_mesh_graph(subdivisions=subdivisions)
        expected_nodes = 2 + 10 * 4**subdivisions
        assert len(G.nodes) == expected_nodes
        for node, data in G.nodes(data=True):
            assert "pos" in data
            assert "pos3d" in data
            assert data["type"] == "mesh"
            assert data["level"] is None
            pos3d = data["pos3d"]
            assert np.allclose(np.linalg.norm(pos3d), 1.0, atol=1e-10)
        for u, v, data in G.edges(data=True):
            assert "len" in data
            assert "vdiff" in data
            assert data["level"] is None
            assert data["len"] > 0
        assert G.graph["is_hierarchical"] is False

    def test_hierarchical_icosahedral_graph(self):
        max_subdivisions = 2
        G = create_hierarchical_icosahedral_mesh_graph(
            max_subdivisions=max_subdivisions
        )
        total_nodes = sum(2 + 10 * 4**level for level in range(max_subdivisions + 1))
        assert len(G.nodes) == total_nodes
        level_counts = {}
        for node, data in G.nodes(data=True):
            level = data["level"]
            level_counts[level] = level_counts.get(level, 0) + 1
        for level in range(max_subdivisions + 1):
            expected_nodes = 2 + 10 * 4**level
            assert level_counts[level] == expected_nodes
        inter_level_edges = [
            (u, v, data)
            for u, v, data in G.edges(data=True)
            if isinstance(data.get("level"), str) and "_to_" in data["level"]
        ]
        assert len(inter_level_edges) > 0
        assert "mesh_vertices_by_level" in G.graph
        assert "mesh_faces_by_level" in G.graph
        assert len(G.graph["mesh_vertices_by_level"]) == max_subdivisions + 1

    def test_graph_directed(self):
        subdivisions = 1
        G = create_flat_icosahedral_mesh_graph(subdivisions=subdivisions)
        edges = set(G.edges())
        for u, v in list(edges):
            assert (v, u) in edges

    def test_vdiff_is_tangential_2d_for_icosahedral(self):
        G = create_flat_icosahedral_mesh_graph(subdivisions=1)
        for u, v, data in G.edges(data=True):
            vdiff = data["vdiff"]
            assert vdiff.shape == (2,), f"Expected vdiff shape (2,), got {vdiff.shape}"
            src_pos3d = G.nodes[u]["pos3d"]
            dst_pos3d = G.nodes[v]["pos3d"]
            outward_normal = src_pos3d / np.linalg.norm(src_pos3d)
            raw_displacement = src_pos3d - dst_pos3d
            normal_component = np.dot(raw_displacement, outward_normal)
            tangential_displacement = (
                raw_displacement - normal_component * outward_normal
            )
            tangential_magnitude = np.linalg.norm(tangential_displacement)
            recovered_magnitude = np.linalg.norm(vdiff)
            assert np.isclose(tangential_magnitude, recovered_magnitude, atol=1e-8)

    def test_vdiff_is_2d_for_rectilinear(self):
        from weather_model_graphs.create.base import connect_nodes_across_graphs
        from weather_model_graphs.create.grid import create_grid_graph_nodes
        from weather_model_graphs.create.mesh import create_single_level_2d_mesh_graph

        xy = test_utils.create_fake_xy(N=16)
        G = create_single_level_2d_mesh_graph(xy=xy, nx=4, ny=4)
        G_grid = create_grid_graph_nodes(xy)
        G_connect = connect_nodes_across_graphs(
            G_source=G, G_target=G_grid, method="nearest_neighbour"
        )
        for u, v, data in G_connect.edges(data=True):
            vdiff = data["vdiff"]
            assert vdiff.shape == (2,), f"Expected vdiff shape (2,), got {vdiff.shape}"


class TestCoordinateConversions:
    def test_lat_lon_to_cartesian(self):
        points = [
            (0, 0, [1, 0, 0]),
            (0, 90, [0, 1, 0]),
            (0, 180, [-1, 0, 0]),
            (0, -90, [0, -1, 0]),
        ]
        for lat, lon, expected in points:
            result = lat_lon_to_cartesian(np.array([lat]), np.array([lon]))[0]
            assert np.allclose(result, expected, atol=1e-10)
        north_pole = lat_lon_to_cartesian(np.array([90]), np.array([0]))[0]
        assert np.allclose(north_pole, [0, 0, 1], atol=1e-10)
        south_pole = lat_lon_to_cartesian(np.array([-90]), np.array([0]))[0]
        assert np.allclose(south_pole, [0, 0, -1], atol=1e-10)

    def test_cartesian_to_lat_lon(self):
        points = [
            ([1, 0, 0], [0, 0]),
            ([0, 1, 0], [0, 90]),
            ([-1, 0, 0], [0, 180]),
            ([0, -1, 0], [0, -90]),
            ([0, 0, 1], [90, 0]),
            ([0, 0, -1], [-90, 0]),
        ]
        for cart, expected in points:
            result = cartesian_to_lat_lon(np.array([cart]))[0]
            assert np.allclose(result, expected, atol=1e-10)

    def test_cartesian_to_lat_lon_clipping(self):
        vertices = np.array(
            [
                [1.0000001, 0, 0],
                [0, 1.0, 0],
                [0, 0, 1.0000001],
                [0, 0, -1.0000001],
            ]
        )
        with pytest.warns(UserWarning, match="Clipped.*values outside"):
            lat_lon = cartesian_to_lat_lon(vertices)
        assert lat_lon.shape == (4, 2)
        assert np.all(lat_lon[:, 0] >= -90) and np.all(lat_lon[:, 0] <= 90)
        assert np.all(lat_lon[:, 1] >= -180) and np.all(lat_lon[:, 1] <= 180)


class TestGridToMeshConnectivity:
    def test_connect_mesh_to_grid_basic(self, sample_grid_10deg):
        vertices, faces = generate_icosahedral_mesh(refinement_level=2)

        edge_index, weights = connect_mesh_to_grid(
            vertices, faces, sample_grid_10deg, fallback_to_nearest=True
        )

        assert isinstance(edge_index, np.ndarray)
        assert isinstance(weights, np.ndarray)
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0, "Expected some connections"
        assert len(weights) == edge_index.shape[1]

        grid_points = np.unique(edge_index[1])
        assert len(grid_points) == len(sample_grid_10deg)

        # Weights are clamped to [0, 1] and non-negative
        assert np.all(weights >= 0.0), f"Negative weights found: {weights[weights < 0]}"
        assert np.all(
            weights <= 1.0 + 1e-9
        ), f"Weights > 1 found: {weights[weights > 1.0 + 1e-9]}"

    def test_connect_grid_to_mesh_radius_factor(self, sample_grid_5deg):
        vertices, faces = generate_icosahedral_mesh(refinement_level=2)
        edges_small = connect_grid_to_mesh(
            sample_grid_5deg, vertices, faces, radius_factor=0.3
        )
        edges_large = connect_grid_to_mesh(
            sample_grid_5deg, vertices, faces, radius_factor=0.9
        )
        assert edges_small.shape[1] <= edges_large.shape[1]

    def test_max_edge_length_computation(self):
        vertices, faces = generate_icosahedral_mesh(refinement_level=1)
        max_len = compute_max_edge_length(vertices, faces)
        assert max_len > 0
        assert max_len < 2.0


class TestMeshToGridConnectivity:
    def test_find_containing_triangle_basic(self):
        vertices, faces = generate_icosahedral_mesh(refinement_level=0)
        test_point = np.array([0.5, 0.5, 0.5])
        test_point = test_point / np.linalg.norm(test_point)
        face_centroids = vertices[faces].mean(axis=1)
        centroid_tree = KDTree(face_centroids)
        face_idx, weights = find_containing_triangle(
            test_point,
            vertices,
            faces,
            face_centroids=face_centroids,
            centroid_tree=centroid_tree,
        )
        if face_idx is not None:
            assert weights is not None
            assert len(weights) == 3
            assert np.allclose(np.sum(weights), 1.0, atol=1e-5)
            assert np.all(weights >= -0.01)

    def test_connect_mesh_to_grid_weights_attribute(self, sample_grid_10deg):
        vertices, faces = generate_icosahedral_mesh(refinement_level=2)
        edge_index, weights = connect_mesh_to_grid(
            vertices, faces, sample_grid_10deg, fallback_to_nearest=True
        )
        # Check all weights are non-negative and at most 1
        assert np.all(weights >= 0.0), "Negative weights found"
        assert np.all(weights <= 1.0 + 1e-9), "Weights exceed 1.0"

        grid_to_weights = {}
        for col in range(edge_index.shape[1]):
            grid_idx = edge_index[1, col]
            weight = weights[col]
            if grid_idx not in grid_to_weights:
                grid_to_weights[grid_idx] = []
            grid_to_weights[grid_idx].append(weight)

        for grid_idx, weight_list in grid_to_weights.items():
            # Triangle hit: 3 weights summing to 1; fallback: 1 weight == 1
            # Also allow 1 or 3 connections summing to 1 (normalised triangles or fallback)
            weight_sum = sum(weight_list)
            assert np.isclose(
                weight_sum, 1.0, atol=1e-4
            ), f"Grid point {grid_idx}: weights {weight_list} sum to {weight_sum}, expected 1.0"


class TestGridSpacing:
    def refinement_level_from_grid_spacing(
        self, grid_spacing_deg: float, radius: float = 1.0
    ) -> int:
        import warnings

        level_to_spacing = {
            0: 63.4,
            1: 31.7,
            2: 15.8,
            3: 7.9,
            4: 3.95,
            5: 1.98,
        }
        closest_level = min(
            level_to_spacing.keys(),
            key=lambda lvl: abs(level_to_spacing[lvl] - grid_spacing_deg),
        )
        actual_spacing = level_to_spacing[closest_level]
        if abs(actual_spacing - grid_spacing_deg) / grid_spacing_deg > 0.3:
            warnings.warn(
                f"Requested grid spacing {grid_spacing_deg}° is significantly different "
                f"from available mesh spacing {actual_spacing:.1f}° at level {closest_level}. "
                f"Consider using a different resolution.",
                UserWarning,
            )
        return closest_level

    def test_grid_spacing_in_create_all_graph_components(
        self, sample_grid_5deg, geographic_crs
    ):
        G = create_all_graph_components(
            coords=sample_grid_5deg,
            mesh_layout="icosahedral",
            mesh_layout_kwargs={
                "grid_spacing": 5.0,
                "hierarchical": False,
                "radius": 1.0,
            },
            m2m_connectivity="flat",
            m2g_connectivity="within_radius",
            m2g_connectivity_kwargs={"max_dist": 0.5},
            g2m_connectivity="within_radius",
            g2m_connectivity_kwargs={"max_dist": 0.5},
            coords_crs=geographic_crs,
            graph_crs=geographic_crs,
        )
        assert G is not None
        mesh_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "mesh"]
        assert len(mesh_nodes) in [162, 642]

    def test_grid_spacing_conflict_error(self, sample_grid_5deg, geographic_crs):
        with pytest.raises(
            ValueError, match="Cannot specify both grid_spacing and subdivisions"
        ):
            create_all_graph_components(
                coords=sample_grid_5deg,
                mesh_layout="icosahedral",
                mesh_layout_kwargs={
                    "grid_spacing": 5.0,
                    "subdivisions": 2,
                },
                m2m_connectivity="flat",
                m2g_connectivity="within_radius",
                m2g_connectivity_kwargs={"max_dist": 0.5},
                g2m_connectivity="within_radius",
                g2m_connectivity_kwargs={"max_dist": 0.5},
                coords_crs=geographic_crs,
                graph_crs=geographic_crs,
            )


class TestContainingTriangleDispatch:
    def test_containing_triangle_with_icosahedral(
        self, sample_grid_10deg, geographic_crs
    ):
        G = create_all_graph_components(
            coords=sample_grid_10deg,
            mesh_layout="icosahedral",
            mesh_layout_kwargs={"subdivisions": 2},
            m2m_connectivity="flat",
            m2g_connectivity="containing_triangle",
            m2g_connectivity_kwargs={"fallback_to_nearest": True},
            g2m_connectivity="within_radius",
            g2m_connectivity_kwargs={"rel_max_dist": 0.6},
            coords_crs=geographic_crs,
            graph_crs=geographic_crs,
        )
        assert G is not None
        m2g_edges = [
            (u, v, d) for u, v, d in G.edges(data=True) if d.get("component") == "m2g"
        ]
        assert len(m2g_edges) > 0
        for _, _, data in m2g_edges:
            assert "barycentric_weight" in data
            assert data["barycentric_weight"] > 0

    def test_containing_triangle_with_rectilinear_error(
        self, sample_grid_10deg, geographic_crs
    ):
        with pytest.raises(
            ValueError,
            match="containing_triangle method is only valid for mesh_layout='icosahedral'",
        ):
            create_all_graph_components(
                coords=sample_grid_10deg,
                mesh_layout="rectilinear",
                m2m_connectivity="flat",
                mesh_layout_kwargs={"mesh_node_distance": 5.0},
                m2m_connectivity_kwargs={"nx": 4, "ny": 4},
                m2g_connectivity="containing_triangle",
                g2m_connectivity="within_radius",
                coords_crs=geographic_crs,
                graph_crs=geographic_crs,
            )


class TestIntegrationWithCreateAllGraphComponents:
    def test_icosahedral_in_create_all_graph_components(
        self, sample_grid_5deg, geographic_crs
    ):
        G = create_all_graph_components(
            coords=sample_grid_5deg,
            mesh_layout="icosahedral",
            mesh_layout_kwargs={
                "subdivisions": 2,
                "hierarchical": False,
                "radius": 1.0,
            },
            m2m_connectivity="flat",
            m2g_connectivity="within_radius",
            m2g_connectivity_kwargs={"max_dist": 0.5},
            g2m_connectivity="within_radius",
            g2m_connectivity_kwargs={"max_dist": 0.5},
            coords_crs=geographic_crs,
            graph_crs=geographic_crs,
        )
        assert G is not None
        assert len(G.nodes) > 0
        mesh_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "mesh"]
        grid_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "grid"]
        assert len(mesh_nodes) > 0
        assert len(grid_nodes) == len(sample_grid_5deg)

    def test_hierarchical_icosahedral(self, sample_grid_5deg, geographic_crs):
        G = create_all_graph_components(
            coords=sample_grid_5deg,
            mesh_layout="icosahedral",
            mesh_layout_kwargs={
                "max_subdivisions": 2,
                "hierarchical": True,
                "radius": 1.0,
            },
            m2m_connectivity="hierarchical",
            m2g_connectivity="within_radius",
            m2g_connectivity_kwargs={"max_dist": 0.5},
            g2m_connectivity="within_radius",
            g2m_connectivity_kwargs={"max_dist": 0.5},
            coords_crs=geographic_crs,
            graph_crs=geographic_crs,
        )
        assert G is not None
        levels = set()
        for node, data in G.nodes(data=True):
            if data.get("type") == "mesh":
                levels.add(data.get("level", -1))
        assert len(levels) > 1

    def test_icosahedral_with_projected_crs_warning(
        self, sample_grid_5deg, projected_crs
    ):
        with pytest.warns(
            UserWarning, match="Icosahedral mesh is designed for geographic coordinates"
        ):
            create_all_graph_components(
                coords=sample_grid_5deg,
                mesh_layout="icosahedral",
                mesh_layout_kwargs={"subdivisions": 2},
                m2m_connectivity="flat",
                m2g_connectivity="within_radius",
                m2g_connectivity_kwargs={"max_dist": 0.5},
                g2m_connectivity="within_radius",
                g2m_connectivity_kwargs={"max_dist": 0.5},
                coords_crs=projected_crs,
                graph_crs=projected_crs,
            )


class TestEdgeCases:
    def test_trimesh_import_error_direct(self):
        with patch("trimesh.creation.icosphere") as mock_icosphere:
            mock_icosphere.side_effect = ImportError("No module named 'trimesh'")
            with pytest.raises(
                ImportError, match="trimesh is required for icosahedral mesh"
            ):
                generate_icosahedral_mesh(refinement_level=1)

    def test_trimesh_import_error_module_level(self):
        """Test graceful handling when trimesh is not available at module level."""
        import sys

        # Temporarily hide trimesh
        trimesh_backup = sys.modules.get("trimesh")
        sys.modules["trimesh"] = None  # Simulate missing module

        try:
            with pytest.raises((ImportError, TypeError)) as excinfo:
                from importlib import reload

                import weather_model_graphs.create.mesh.layouts.icosahedral as icosahedral_module

                reload(icosahedral_module)
                # Calling generate_icosahedral_mesh will hit the import inside the function
                icosahedral_module.generate_icosahedral_mesh(refinement_level=1)

            err_msg = str(excinfo.value).lower()
            assert "trimesh" in err_msg or "nonetype" in err_msg or "import" in err_msg
        finally:
            # Restore trimesh
            if trimesh_backup is not None:
                sys.modules["trimesh"] = trimesh_backup
            elif "trimesh" in sys.modules:
                del sys.modules["trimesh"]

    def test_invalid_subdivisions(self):
        with pytest.raises(ValueError, match="subdivisions must be non-negative"):
            generate_icosahedral_mesh(refinement_level=-1)

    def test_empty_grid_connections(self):
        vertices, faces = generate_icosahedral_mesh(refinement_level=1)
        empty_grid = np.array([]).reshape(0, 2)
        edges = connect_grid_to_mesh(empty_grid, vertices, faces)
        assert edges.shape[1] == 0

    def test_pole_connections(self, sample_grid_1deg):
        vertices, faces = generate_icosahedral_mesh(refinement_level=2)
        pole_indices = np.where(np.abs(sample_grid_1deg[:, 0]) >= 89.5)[0]
        pole_grid = sample_grid_1deg[pole_indices]
        edges = connect_grid_to_mesh(pole_grid, vertices, faces)
        assert edges.shape[1] > 0

    def test_unknown_mesh_layout_error(self, sample_grid_5deg, geographic_crs):
        with pytest.raises(ValueError, match="Unknown mesh_layout 'invalid'"):
            create_all_graph_components(
                coords=sample_grid_5deg,
                mesh_layout="invalid",
                m2m_connectivity="flat",
                m2g_connectivity="within_radius",
                g2m_connectivity="within_radius",
                coords_crs=geographic_crs,
                graph_crs=geographic_crs,
            )
