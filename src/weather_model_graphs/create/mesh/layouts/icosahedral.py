"""Icosahedral mesh layout for global graphs."""

import numpy as np
import networkx as nx
import trimesh
from scipy.spatial import KDTree  # Will replace with haversine later

def create_icosahedral_mesh(subdivisions=3):
    """
    Generate icosahedral mesh hierarchy using trimesh.
    
    This is Mandeep's part - we'll use his output format.
    Returns list of (vertices, faces) for each level.
    """
    mesh = trimesh.creation.icosphere(subdivisions=0)
    vertices, faces = mesh.vertices, mesh.faces
    
    mesh_list = [(vertices, faces)]
    
    for level in range(1, subdivisions + 1):
        # Subdivide and project to sphere
        vertices, faces = trimesh.remesh.subdivide(*mesh_list[-1])
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / norms
        mesh_list.append((vertices, faces))
    
    return mesh_list  # Coarsest to finest

def connect_grid_to_mesh(grid_lat_lon, mesh_vertices, mesh_faces, radius_factor=0.6):
    """
    Grid → Mesh connections (g2m) adapted from create_global_mesh.py lines 224-242.
    
    Args:
        grid_lat_lon: (N_grid, 2) array of [lat, lon] in degrees
        mesh_vertices: (N_mesh, 3) cartesian coordinates
        mesh_faces: (M, 3) face indices
        radius_factor: multiplier for max edge distance
    
    Returns:
        edge_index: (2, E) array of [grid_node, mesh_node] connections
    """
    # Convert grid to cartesian for distance computation
    grid_cartesian = lat_lon_to_cartesian(grid_lat_lon[:, 0], grid_lat_lon[:, 1])
    
    # Compute max edge distance in mesh
    max_edge_len = compute_max_edge_length(mesh_vertices, mesh_faces)
    query_radius = radius_factor * max_edge_len
    
    # KD-tree for mesh vertices
    tree = KDTree(mesh_vertices)
    
    # Query for each grid point
    grid_indices, mesh_indices = [], []
    for i, point in enumerate(grid_cartesian):
        neighbors = tree.query_ball_point(point, query_radius)
        if neighbors:
            grid_indices.extend([i] * len(neighbors))
            mesh_indices.extend(neighbors)
    
    return np.array([grid_indices, mesh_indices])

def connect_mesh_to_grid(mesh_vertices, mesh_faces, grid_lat_lon):
    """
    Mesh → Grid connections (m2g) adapted from create_global_mesh.py lines 245-259.
    
    For each grid point, find containing mesh triangle.
    """
    # This is more complex - need triangle containment in 3D
    # We'll implement this after basic g2m works
    pass

def lat_lon_to_cartesian(lat, lon):
    """Convert lat/lon degrees to cartesian coordinates on unit sphere."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    return np.column_stack([x, y, z])

def compute_max_edge_length(vertices, faces):
    """Compute longest edge in mesh."""
    max_len = 0
    for face in faces:
        for i, j in [(0,1), (1,2), (2,0)]:
            dist = np.linalg.norm(vertices[face[i]] - vertices[face[j]])
            max_len = max(max_len, dist)
    return max_len

def generate_icosahedral_mesh(refinement_level: int, radius: float = 1.0):
    """
    Generates a spherical icosahedral mesh using Trimesh.
    This fulfills the mesh_layout='icosahedral' requirement.
    
    Args:
        refinement_level (int): Number of subdivisions. 
                                Level 0 is a base icosahedron (12 nodes).
        radius (float): Radius of the sphere (default 1.0 for unit sphere).
        
    Returns:
        nodes (np.ndarray): Shape (N, 3) Cartesian coordinates (x, y, z).
        faces (np.ndarray): Shape (M, 3) Triangular faces connecting the nodes.
    """
    # Create the base icosphere with the specified refinement level
    mesh = trimesh.creation.icosphere(subdivisions=refinement_level, radius=radius)
    
    # Extract nodes and faces
    nodes = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    return nodes, faces