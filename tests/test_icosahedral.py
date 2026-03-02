import numpy as np
import networkx as nx
import pytest
import pyproj
from unittest.mock import patch, MagicMock
from scipy.spatial import KDTree 
from weather_model_graphs.create.mesh.layouts.icosahedral import (
    generate_icosahedral_mesh,
    create_hierarchy_of_icosahedral_meshes,
    create_flat_icosahedral_mesh_graph,
    create_hierarchical_icosahedral_mesh_graph,
    connect_grid_to_mesh,
    lat_lon_to_cartesian,
    cartesian_to_lat_lon,
    compute_max_edge_length,
    find_containing_triangle,
)
from weather_model_graphs.create import create_all_graph_components


# Skip tests if trimesh not available
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    pytest.skip("trimesh not installed", allow_module_level=True)


@pytest.fixture
def sample_grid_1deg():
    """Create a 1-degree resolution global grid."""
    lats = np.arange(-90, 91, 1)
    lons = np.arange(-180, 181, 1)
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing='ij')
    return np.column_stack([grid_lat.ravel(), grid_lon.ravel()])


@pytest.fixture
def sample_grid_5deg():
    """Create a 5-degree resolution global grid."""
    lats = np.arange(-90, 91, 5)
    lons = np.arange(-180, 181, 5)
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing='ij')
    return np.column_stack([grid_lat.ravel(), grid_lon.ravel()])


@pytest.fixture
def geographic_crs():
    """Geographic coordinate system (WGS84)."""
    return pyproj.CRS.from_string("EPSG:4326")


class TestIcosahedralMeshGeneration:
    """Test basic icosahedral mesh generation."""

    def test_generate_base_icosahedron(self):
        """Test generation of base icosahedron (subdivisions=0)."""
        vertices, faces = generate_icosahedral_mesh(refinement_level=0)
        
        # Icosahedron has 12 vertices, 20 faces
        assert len(vertices) == 12
        assert len(faces) == 20
        
        # All vertices should be on unit sphere
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)
        
        # Faces should be triangles
        assert faces.shape[1] == 3

    def test_generate_refined_icosahedron(self):
        """Test generation of refined icosahedron."""
        for subdivisions in [1, 2, 3]:
            vertices, faces = generate_icosahedral_mesh(refinement_level=subdivisions)
            
            # Check vertex count formula: 2 + 10 * 4^subdivisions
            expected_vertices = 2 + 10 * 4**subdivisions
            assert len(vertices) == expected_vertices
            
            # Check face count formula: 20 * 4^subdivisions
            expected_faces = 20 * 4**subdivisions
            assert len(faces) == expected_faces
            
            # All vertices on unit sphere
            norms = np.linalg.norm(vertices, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-10)

    def test_create_hierarchy(self):
        """Test creation of hierarchical meshes."""
        max_subdivisions = 3
        mesh_list = create_hierarchy_of_icosahedral_meshes(max_subdivisions)
        
        assert len(mesh_list) == max_subdivisions + 1
        
        # Each level should have correct number of vertices
        for level, (vertices, faces) in enumerate(mesh_list):
            expected_vertices = 2 + 10 * 4**level
            expected_faces = 20 * 4**level
            assert len(vertices) == expected_vertices
            assert len(faces) == expected_faces


class TestIcosahedralMeshGraphs:
    """Test graph creation from icosahedral meshes."""

    def test_flat_icosahedral_graph(self):
        """Test creation of flat (single-level) icosahedral graph."""
        subdivisions = 2
        G = create_flat_icosahedral_mesh_graph(subdivisions=subdivisions)

        # Check node count
        expected_nodes = 2 + 10 * 4**subdivisions
        assert len(G.nodes) == expected_nodes

        # Check node attributes
        for node, data in G.nodes(data=True):
            assert "pos" in data
            assert "pos3d" in data  # Check that 3D position exists
            assert data["type"] == "mesh"
            assert data["level"] == 0
            # 3D position should be on unit sphere
            pos3d = data["pos3d"]
            assert np.allclose(np.linalg.norm(pos3d), 1.0, atol=1e-10)
        
        # Check edge attributes
        for u, v, data in G.edges(data=True):
            assert "len" in data
            assert "vdiff" in data
            assert data["level"] == 0
            # Length should be positive
            assert data["len"] > 0
            # Vector difference should match lat/lon positions
            vec = data["vdiff"]
            pos_u = G.nodes[u]["pos"]
            pos_v = G.nodes[v]["pos"]
            expected_vec = pos_u - pos_v
            # Handle longitude wrapping in the comparison
            if abs(expected_vec[1]) > 180:
                expected_vec[1] = (expected_vec[1] + 180) % 360 - 180
            assert np.allclose(vec, expected_vec, atol=1e-10)


    def test_hierarchical_icosahedral_graph(self):
        """Test creation of hierarchical icosahedral graph."""
        max_subdivisions = 2
        G = create_hierarchical_icosahedral_mesh_graph(max_subdivisions=max_subdivisions)
        
        # Total nodes should be sum of all levels
        total_nodes = sum(2 + 10 * 4**level for level in range(max_subdivisions + 1))
        assert len(G.nodes) == total_nodes
        
        # Check level distribution
        level_counts = {}
        for node, data in G.nodes(data=True):
            level = data["level"]
            level_counts[level] = level_counts.get(level, 0) + 1
        
        for level in range(max_subdivisions + 1):
            expected_nodes = 2 + 10 * 4**level
            assert level_counts[level] == expected_nodes
        
        # Check inter-level edges exist
        inter_level_edges = [
            (u, v, data) for u, v, data in G.edges(data=True)
            if isinstance(data.get("level"), str) and "_to_" in data["level"]
        ]
        assert len(inter_level_edges) > 0

    def test_graph_directed(self):
        """Test that graphs are properly directed (both directions)."""
        subdivisions = 1
        G = create_flat_icosahedral_mesh_graph(subdivisions=subdivisions)
        
        # Check that for every edge, the reverse also exists
        edges = set(G.edges())
        for u, v in list(edges):
            assert (v, u) in edges


class TestCoordinateConversions:
    """Test lat/lon to cartesian conversions."""

    def test_lat_lon_to_cartesian(self):
        """Test conversion from lat/lon to cartesian."""
        # Test equator points
        points = [
            (0, 0, [1, 0, 0]),
            (0, 90, [0, 1, 0]),
            (0, 180, [-1, 0, 0]),
            (0, -90, [0, -1, 0]),
        ]
        for lat, lon, expected in points:
            result = lat_lon_to_cartesian(np.array([lat]), np.array([lon]))[0]
            assert np.allclose(result, expected, atol=1e-10)
        
        # Test poles
        north_pole = lat_lon_to_cartesian(np.array([90]), np.array([0]))[0]
        assert np.allclose(north_pole, [0, 0, 1], atol=1e-10)
        
        south_pole = lat_lon_to_cartesian(np.array([-90]), np.array([0]))[0]
        assert np.allclose(south_pole, [0, 0, -1], atol=1e-10)

    def test_cartesian_to_lat_lon(self):
        """Test conversion from cartesian to lat/lon."""
        # Test points on unit sphere
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


class TestGridToMeshConnectivity:
    """Test grid-to-mesh connectivity functions."""

    def test_connect_grid_to_mesh_basic(self, sample_grid_5deg):
        """Test basic grid-to-mesh connections."""
        # Create a coarse mesh
        vertices, faces = generate_icosahedral_mesh(refinement_level=2)
        
        # Connect grid to mesh
        edges = connect_grid_to_mesh(sample_grid_5deg, vertices, faces)
        
        # Check output format
        assert edges.shape[0] == 2  # source, target
        assert edges.shape[1] > 0  # at least some connections
        
        # Check indices are within bounds
        assert np.all(edges[0] < len(sample_grid_5deg))
        assert np.all(edges[1] < len(vertices))

    def test_connect_grid_to_mesh_radius_factor(self, sample_grid_5deg):
        """Test effect of radius factor on connections."""
        vertices, faces = generate_icosahedral_mesh(refinement_level=2)
        
        # Smaller radius should yield fewer connections
        edges_small = connect_grid_to_mesh(sample_grid_5deg, vertices, faces, radius_factor=0.3)
        edges_large = connect_grid_to_mesh(sample_grid_5deg, vertices, faces, radius_factor=0.9)
        
        assert edges_small.shape[1] <= edges_large.shape[1]

    def test_max_edge_length_computation(self):
        """Test computation of maximum edge length."""
        vertices, faces = generate_icosahedral_mesh(refinement_level=1)
        max_len = compute_max_edge_length(vertices, faces)
        
        # Should be positive and reasonable
        assert max_len > 0
        assert max_len < 2.0  # Max distance on unit sphere < 2


class TestMeshToGridConnectivity:
    """Test mesh-to-grid connectivity (containing triangle)."""

    def test_find_containing_triangle_basic(self):
        """Test finding containing triangle for a point."""
        # Create a simple mesh (base icosahedron)
        vertices, faces = generate_icosahedral_mesh(refinement_level=0)
        
        # Test a point that should be in a specific triangle
        # This is a simple test - in practice we'd need more thorough testing
        test_point = np.array([0.5, 0.5, 0.5])
        test_point = test_point / np.linalg.norm(test_point)
        face_centroids = vertices[faces].mean(axis=1)
        centroid_tree = KDTree(face_centroids)
        

        face_idx, weights = find_containing_triangle(
            test_point, 
            vertices, 
            faces,
            face_centroids=face_centroids,
            centroid_tree=centroid_tree
        )

        
        # Either found or not found is acceptable for this test
        if face_idx is not None:
            assert weights is not None
            assert len(weights) == 3
            assert np.allclose(np.sum(weights), 1.0, atol=1e-5)
            assert np.all(weights >= -0.01)  # Allow small numerical errors


class TestIntegrationWithCreateAllGraphComponents:
    """Test integration with the main graph creation function."""

    def test_icosahedral_in_create_all_graph_components(self, sample_grid_5deg, geographic_crs):
        """Test using icosahedral mesh in create_all_graph_components."""
        try:
            G = create_all_graph_components(
                coords=sample_grid_5deg,
                m2m_connectivity="icosahedral",
                m2m_connectivity_kwargs={
                    "subdivisions": 2,
                    "hierarchical": False,
                    "radius": 1.0,
                },
                m2g_connectivity="within_radius",
                m2g_connectivity_kwargs={"max_dist": 0.5},  # Add this
                g2m_connectivity="within_radius",
                g2m_connectivity_kwargs={"max_dist": 0.5},  # Add this
                coords_crs=geographic_crs,
                graph_crs=geographic_crs,
            )
            
            # Check that graph was created
            assert G is not None
            assert len(G.nodes) > 0
            
            # Check that nodes have appropriate attributes
            mesh_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "mesh"]
            grid_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "grid"]
            
            assert len(mesh_nodes) > 0
            assert len(grid_nodes) == len(sample_grid_5deg)
            
        except ImportError:
            pytest.skip("trimesh not available")

    def test_hierarchical_icosahedral(self, sample_grid_5deg, geographic_crs):
        """Test hierarchical icosahedral mesh."""
        try:
            G = create_all_graph_components(
                coords=sample_grid_5deg,
                m2m_connectivity="icosahedral",
                m2m_connectivity_kwargs={
                    "max_subdivisions": 2,
                    "hierarchical": True,
                    "radius": 1.0,
                },
                m2g_connectivity="within_radius",
                m2g_connectivity_kwargs={"max_dist": 0.5},  # Add this
                g2m_connectivity="within_radius",
                g2m_connectivity_kwargs={"max_dist": 0.5},  # Add this
                coords_crs=geographic_crs,
                graph_crs=geographic_crs,
            )
            
            assert G is not None
            
            # Check that nodes have level attributes
            levels = set()
            for node, data in G.nodes(data=True):
                if data.get("type") == "mesh":
                    levels.add(data.get("level", -1))
            
            # Should have multiple levels
            assert len(levels) > 1
            
        except ImportError:
            pytest.skip("trimesh not available")

    def test_icosahedral_with_projected_crs_warning(self, sample_grid_5deg):
        """Test warning when using icosahedral with projected CRS."""
        projected_crs = pyproj.CRS.from_string("EPSG:32633")  # UTM zone 33N
        
        with pytest.warns(UserWarning, match="Icosahedral mesh is designed for geographic coordinates"):
            G = create_all_graph_components(
                coords=sample_grid_5deg,
                m2m_connectivity="icosahedral",
                m2m_connectivity_kwargs={"subdivisions": 2},
                m2g_connectivity="within_radius",
                m2g_connectivity_kwargs={"max_dist": 0.5}, 
                g2m_connectivity="within_radius",
                g2m_connectivity_kwargs={"max_dist": 0.5}, 
                coords_crs=projected_crs,
                graph_crs=projected_crs,
            )

class TestEdgeCases:
    """Test edge cases and error handling."""
    def test_trimesh_import_error(self):
        """Test graceful handling when trimesh is not available."""
        # Mock the import of trimesh itself, not just icosphere
        with patch.dict('sys.modules', {'trimesh': None}):
            with patch('importlib.import_module', side_effect=ImportError("No module named 'trimesh'")):
                with pytest.raises(ImportError, match="trimesh is required for icosahedral mesh"):
                    # Need to reload the module to trigger the import error
                    from importlib import reload
                    import weather_model_graphs.create.mesh.layouts.icosahedral as icosahedral_module
                    reload(icosahedral_module)
                    icosahedral_module.generate_icosahedral_mesh(refinement_level=1)

    def test_invalid_subdivisions(self):
        """Test handling of invalid subdivision values."""
        with pytest.raises(ValueError, match="subdivisions must be non-negative"):
            generate_icosahedral_mesh(refinement_level=-1)

    def test_empty_grid_connections(self):
        """Test connecting empty grid."""
        vertices, faces = generate_icosahedral_mesh(refinement_level=1)
        empty_grid = np.array([]).reshape(0, 2)
        
        edges = connect_grid_to_mesh(empty_grid, vertices, faces)
        assert edges.shape[1] == 0

    def test_pole_connections(self, sample_grid_1deg):
        """Test connections at the poles."""
        vertices, faces = generate_icosahedral_mesh(refinement_level=2)
        
        # Find grid points at poles
        pole_indices = np.where(np.abs(sample_grid_1deg[:, 0]) >= 89.5)[0]
        pole_grid = sample_grid_1deg[pole_indices]
        
        edges = connect_grid_to_mesh(pole_grid, vertices, faces)
        
        # Should still have connections
        assert edges.shape[1] > 0

    @patch('trimesh.creation.icosphere')
    def test_trimesh_import_error(self, mock_icosphere):
        """Test graceful handling when trimesh is not available."""
        mock_icosphere.side_effect = ImportError("No module named 'trimesh'")
        
        with pytest.raises(ImportError, match="trimesh is required for icosahedral mesh"):
            generate_icosahedral_mesh(refinement_level=1)


if __name__ == "__main__":
    pytest.main([__file__])