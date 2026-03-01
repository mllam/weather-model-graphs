"""Icosahedral mesh layout for global graphs."""

import numpy as np
import networkx as nx
import trimesh
from scipy.spatial import KDTree
import warnings


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


def generate_icosahedral_mesh(refinement_level: int, radius: float = 1.0):
    """
    Generates a spherical icosahedral mesh using Trimesh.
    
    Args:
        refinement_level (int): Number of subdivisions (0 = base icosahedron, 12 nodes)
        radius (float): Radius of the sphere (default 1.0 for unit sphere)
        
    Returns:
        nodes (np.ndarray): Shape (N, 3) Cartesian coordinates (x, y, z)
        faces (np.ndarray): Shape (M, 3) Triangular faces connecting the nodes
    """
    mesh = trimesh.creation.icosphere(subdivisions=refinement_level, radius=radius)
    return np.array(mesh.vertices), np.array(mesh.faces)


def create_hierarchy_of_icosahedral_meshes(max_subdivisions: int, radius: float = 1.0):
    """
    Create a list of icosahedral meshes at different refinement levels.
    
    Args:
        max_subdivisions (int): Maximum number of subdivisions
        radius (float): Radius of the sphere
        
    Returns:
        list of (vertices, faces) tuples for each level from coarsest to finest
    """
    mesh_list = []
    for level in range(max_subdivisions + 1):
        vertices, faces = generate_icosahedral_mesh(level, radius)
        mesh_list.append((vertices, faces))
    return mesh_list  # Coarsest to finest


def create_flat_icosahedral_mesh_graph(
    subdivisions: int = 3,
    radius: float = 1.0,
    add_edge_length: bool = True,
    add_edge_vector: bool = True,
):
    """
    Create a flat (single-level) icosahedral mesh graph.
    
    Args:
        subdivisions (int): Number of mesh subdivisions (0 = base icosahedron)
        radius (float): Sphere radius
        add_edge_length (bool): Add 'len' attribute to edges with Euclidean distance
        add_edge_vector (bool): Add 'vdiff' attribute with vector difference
        
    Returns:
        networkx.DiGraph: Directed graph with mesh nodes and edges
    """
    vertices, faces = generate_icosahedral_mesh(subdivisions, radius)
    lat_lon = cartesian_to_lat_lon(vertices)  # (N, 2) -> [lat, lon] in degrees

    G = nx.Graph()
    for i, (x, y, z) in enumerate(vertices):
        G.add_node(
            i,
            pos=lat_lon[i],        # 2D (lat, lon) for KDTree compatibility
            pos3d=np.array([x, y, z]),  # 3D for distance calculations
            type="mesh",
            level=0,
        )

    for face in faces:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = face[i], face[j]
                if not G.has_edge(u, v):
                    G.add_edge(u, v)

    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes(data=True))

    for u, v in G.edges():
        vec = DG.nodes[u]["pos"] - DG.nodes[v]["pos"]   # 2D diff for vdiff
        dist = np.linalg.norm(DG.nodes[u]["pos3d"] - DG.nodes[v]["pos3d"])  # 3D dist

        DG.add_edge(u, v, len=dist, vdiff=vec, level=0)
        DG.add_edge(v, u, len=dist, vdiff=-vec, level=0)

    DG.graph["mesh_layout"] = "icosahedral"
    DG.graph["subdivisions"] = subdivisions
    DG.graph["radius"] = radius

    return DG



def create_hierarchical_icosahedral_mesh_graph(
    max_subdivisions: int = 3,
    radius: float = 1.0,
    add_edge_length: bool = True,
    add_edge_vector: bool = True,
):
    """
    Create a hierarchical icosahedral mesh graph with multiple refinement levels.
    
    Args:
        max_subdivisions (int): Maximum number of subdivisions
        radius (float): Sphere radius
        add_edge_length (bool): Add 'len' attribute to edges
        add_edge_vector (bool): Add 'vdiff' attribute to edges
        
    Returns:
        networkx.DiGraph: Combined graph with all levels and inter-level edges
    """
    mesh_list = create_hierarchy_of_icosahedral_meshes(max_subdivisions, radius)
    
    DG = create_flat_icosahedral_mesh_graph(max_subdivisions, radius,
                                            add_edge_length, add_edge_vector)
    
    node_offset = len(DG.nodes)
    
    for level, (vertices, faces) in enumerate(mesh_list[:-1]):
        lat_lon = cartesian_to_lat_lon(vertices)  # <-- convert to 2D
        level_nodes = []
        
        for i, (x, y, z) in enumerate(vertices):
            node_id = node_offset + i
            DG.add_node(
                node_id,
                pos=lat_lon[i],              # 2D for KDTree compatibility
                pos3d=np.array([x, y, z]),   # 3D for reference
                type="mesh",
                level=level,
            )
            level_nodes.append(node_id)
        
        for face in faces:
            for i in range(3):
                for j in range(i+1, 3):
                    u, v = node_offset + face[i], node_offset + face[j]
                    if not DG.has_edge(u, v):
                        vec = DG.nodes[u]["pos"] - DG.nodes[v]["pos"]   # 2D diff
                        dist = np.linalg.norm(DG.nodes[u]["pos3d"] - DG.nodes[v]["pos3d"])  # 3D dist
                        DG.add_edge(u, v, len=dist, vdiff=vec, level=level)
                        DG.add_edge(v, u, len=dist, vdiff=-vec, level=level)
        
        # inter-level edges
        if level < max_subdivisions - 1:
            finer_vertices = mesh_list[level + 1][0]
            tree = KDTree(finer_vertices)
            
            for i, coarse_node in enumerate(level_nodes):
                coarse_pos = vertices[i]
                max_edge_len = compute_max_edge_length(finer_vertices, mesh_list[level + 1][1])
                radius_query = 1.1 * max_edge_len
                
                fine_indices = tree.query_ball_point(coarse_pos, radius_query)
                if fine_indices:
                    for fine_idx in fine_indices:
                        vec3d = coarse_pos - finer_vertices[fine_idx]
                        dist = np.linalg.norm(vec3d)
                        fine_node = node_offset - len(finer_vertices) + fine_idx
                        coarse_lat_lon = DG.nodes[coarse_node]["pos"]
                        fine_lat_lon = DG.nodes[fine_node]["pos"]
                        vec2d = coarse_lat_lon - fine_lat_lon

                        DG.add_edge(fine_node, coarse_node,
                                    len=dist, vdiff=vec2d, level=f"{level}_to_{level+1}")
                        DG.add_edge(coarse_node, fine_node,
                                    len=dist, vdiff=-vec2d, level=f"{level+1}_to_{level}")
        
        node_offset += len(vertices)
    
    DG.graph["mesh_layout"] = "icosahedral_hierarchical"
    DG.graph["max_subdivisions"] = max_subdivisions
    DG.graph["radius"] = radius
    
    return DG


def connect_grid_to_mesh(grid_lat_lon, mesh_vertices, mesh_faces, radius_factor=0.6):
    """
    Grid to Mesh connections (g2m) adapted from create_global_mesh.py lines 224-242.
    
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


def cartesian_to_lat_lon(vertices):
    """Convert cartesian coordinates to lat/lon degrees."""
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Convert to spherical coordinates
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    
    return np.column_stack([lat, lon])


def compute_max_edge_length(vertices, faces):
    """Compute longest edge in mesh."""
    max_len = 0
    for face in faces:
        for i, j in [(0,1), (1,2), (2,0)]:
            dist = np.linalg.norm(vertices[face[i]] - vertices[face[j]])
            max_len = max(max_len, dist)
    return max_len


def find_containing_triangle(
    point_cartesian: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
):
    """
    Find the triangle containing a point on the sphere.
    Used for mesh-to-grid (m2g) connectivity.
    
    Args:
        point_cartesian: (3,) cartesian point on sphere
        mesh_vertices: (N_mesh, 3) mesh vertices
        mesh_faces: (M, 3) face indices
        
    Returns:
        tuple: (face_index, barycentric_weights) or (None, None) if not found
    """
    # Normalize point (should already be on sphere, but just in case)
    point_norm = point_cartesian / np.linalg.norm(point_cartesian)
    
    best_face = None
    best_weights = None
    best_sum = float('inf')
    
    for face_idx, face in enumerate(mesh_faces):
        a, b, c = mesh_vertices[face]
        
        # Check if point is in spherical triangle using barycentric coordinates
        # Convert to 3D barycentric with normalization
        matrix = np.column_stack([a, b, c])
        try:
            weights = np.linalg.solve(matrix, point_norm)
            # Weights should be positive and sum to ~1
            if np.all(weights >= -0.01) and np.all(weights <= 1.01):
                weight_sum = np.sum(weights)
                if abs(weight_sum - 1.0) < abs(best_sum - 1.0):
                    best_sum = weight_sum
                    best_face = face_idx
                    best_weights = weights / weight_sum  # Normalize
        except np.linalg.LinAlgError:
            continue
    
    return best_face, best_weights


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