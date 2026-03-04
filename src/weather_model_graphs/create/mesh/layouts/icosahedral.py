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
    lat_lon = cartesian_to_lat_lon(vertices)

    G = nx.Graph()
    for i, (x, y, z) in enumerate(vertices):
        G.add_node(
            i,
            pos=lat_lon[i],
            pos3d=np.array([x, y, z]),
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
        # Use 3D positions for both distance and direction
        vec = DG.nodes[u]["pos3d"] - DG.nodes[v]["pos3d"]
        dist = np.linalg.norm(vec)

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
    # mesh_list[0] = coarsest (12 nodes), mesh_list[-1] = finest (642 nodes at level 3)
    level_offsets = {}
    current_offset = 0
    # First, calculate offsets for all levels (from finest to coarsest)
    for level in range(max_subdivisions, -1, -1):
        level_offsets[level] = current_offset
        vertices = mesh_list[level][0]
        current_offset += len(vertices)
        # Create empty directed graph
    DG = nx.DiGraph()
    
    # Add all nodes from all levels
    for level in range(max_subdivisions + 1):
        vertices, faces = mesh_list[level]
        lat_lon = cartesian_to_lat_lon(vertices)
        offset = level_offsets[level]
        
        for i, (x, y, z) in enumerate(vertices):
            node_id = offset + i
            DG.add_node(
                node_id,
                pos=lat_lon[i],
                pos3d=np.array([x, y, z]),
                type="mesh",
                level=level,
            )
    
    # Add intra-level edges for each level
    for level in range(max_subdivisions + 1):
        vertices, faces = mesh_list[level]
        offset = level_offsets[level]
        
        for face in faces:
            for i in range(3):
                for j in range(i + 1, 3):
                    u = offset + face[i]
                    v = offset + face[j]
                    if not DG.has_edge(u, v):
                        vec = DG.nodes[u]["pos3d"] - DG.nodes[v]["pos3d"]
                        dist = np.linalg.norm(vec)
                        DG.add_edge(u, v, len=dist, vdiff=vec, level=level)
                        DG.add_edge(v, u, len=dist, vdiff=-vec, level=level)
    
    # Add inter-level edges between consecutive levels
    for coarse_level in range(max_subdivisions):
        fine_level = coarse_level + 1
        
        coarse_vertices, coarse_faces = mesh_list[coarse_level]
        fine_vertices, fine_faces = mesh_list[fine_level]
        
        coarse_offset = level_offsets[coarse_level]
        fine_offset = level_offsets[fine_level]
        
        # Build KD-tree for fine level vertices
        tree = KDTree(fine_vertices)
        max_edge_len = compute_max_edge_length(fine_vertices, fine_faces)
        radius_query = 1.1 * max_edge_len
        
        # Connect each coarse node to nearby fine nodes
        for i, coarse_pos in enumerate(coarse_vertices):
            coarse_node = coarse_offset + i
            fine_indices = tree.query_ball_point(coarse_pos, radius_query)
            
            for fine_idx in fine_indices:
                fine_node = fine_offset + fine_idx
                vec = coarse_pos - fine_vertices[fine_idx]
                dist = np.linalg.norm(vec)
                
                # Add bidirectional edges
                DG.add_edge(fine_node, coarse_node,
                           len=dist, vdiff=vec,
                           level=f"{fine_level}_to_{coarse_level}")
                DG.add_edge(coarse_node, fine_node,
                           len=dist, vdiff=-vec,
                           level=f"{coarse_level}_to_{fine_level}")
    
    DG.graph["mesh_layout"] = "icosahedral_hierarchical"
    DG.graph["max_subdivisions"] = max_subdivisions
    DG.graph["radius"] = radius
    DG.graph["level_offsets"] = level_offsets
    DG.graph["vertices_by_level"] = [v for v, _ in mesh_list]
    DG.graph["faces_by_level"] = [f for _, f in mesh_list]
    
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
    Mesh to Grid connections (m2g).
    For each grid point, find containing mesh triangle and return
    barycentric weights for interpolation.

    Args:
        mesh_vertices: (N_mesh, 3) cartesian coordinates
        mesh_faces: (M, 3) face indices
        grid_lat_lon: (N_grid, 2) array of [lat, lon] in degrees

    Returns:
        edge_index: (2, E) array of [mesh_node, grid_node] connections
        weights: (E,) barycentric weights for each edge
    """
    # Precompute face centroids and KDTree
    grid_cartesian = lat_lon_to_cartesian(grid_lat_lon[:, 0], grid_lat_lon[:, 1])

    face_centroids = mesh_vertices[mesh_faces].mean(axis=1)
    centroid_tree = KDTree(face_centroids)
    
    mesh_indices, grid_indices, weights = [], [], []
    
    for grid_idx, point in enumerate(grid_cartesian):
        face_idx, bary_weights = find_containing_triangle(
            point, mesh_vertices, mesh_faces, 
            face_centroids, centroid_tree, k_candidates=10
        )
        if face_idx is not None:
            for mesh_idx, w in zip(mesh_faces[face_idx], bary_weights):
                mesh_indices.append(mesh_idx)
                grid_indices.append(grid_idx)
                weights.append(w)
    
    return np.array([mesh_indices, grid_indices]), np.array(weights)



def lat_lon_to_cartesian(lat, lon):
    """Convert lat/lon degrees to cartesian coordinates on unit sphere."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    return np.column_stack([x, y, z])


def cartesian_to_lat_lon(vertices):
    """Convert cartesian coordinates to lat/lon degrees.
    
    Args:
        vertices: (N, 3) array of (x, y, z) coordinates on unit sphere
        
    Returns:
        (N, 2) array of (latitude, longitude) in degrees
        Latitude range: [-90, 90]
        Longitude range: [-180, 180]
    """
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Convert to spherical coordinates
    # atan2(y, x) gives range which maps to [-180°, 180°]
    lon = np.degrees(np.arctan2(y, x))
    
    # For points on unit sphere, radius = 1, so we can compute latitude directly
    lat = np.degrees(np.arcsin(z))  # arcsin gives range 
    
    # Alternative safer computation (works even if not perfectly normalized):
    # r = np.sqrt(x**2 + y**2 + z**2)
    # lat = np.degrees(np.arcsin(z / r))
    
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
    face_centroids: np.ndarray = None,  # Precomputed
    centroid_tree: KDTree = None,  #Precomputed
    k_candidates: int = 10,
):
    """
    Optimized triangle containment using spatial indexing.
    
    Args:
        point_cartesian: (3,) cartesian point on sphere
        mesh_vertices: (N_mesh, 3) mesh vertices
        mesh_faces: (M, 3) face indices
        face_centroids: Precomputed centroids of faces
        centroid_tree: Precomputed KDTree on centroids
        k_candidates: Number of candidate faces to check
        
    Returns:
        tuple: (face_index, barycentric_weights) or (None, None) if not found
    """

    point_norm = point_cartesian / np.linalg.norm(point_cartesian)
    # Fallback: compute centroids and tree if not provided
    if face_centroids is None or centroid_tree is None:
        face_centroids = mesh_vertices[mesh_faces].mean(axis=1)
        centroid_tree = KDTree(face_centroids)
    

    # Get candidate faces from spatial index
    candidate_indices = centroid_tree.query(point_norm, k=k_candidates)[1]
    
    best_face = None
    best_weights = None
    best_sum = float('inf')
    
    # Only check candidate faces
    for face_idx in candidate_indices:
        face = mesh_faces[face_idx]
        a, b, c = mesh_vertices[face]
        
        # Optional: Use barycentric with cross products instead of solve
        # This is even faster than np.linalg.solve
        weights = barycentric_coordinates(point_norm, a, b, c)
        
        if weights is not None and np.all(weights >= -0.01) and np.all(weights <= 1.01):
            weight_sum = np.sum(weights)
            if abs(weight_sum - 1.0) < 1e-6:
                return face_idx, weights / weight_sum
            else:
                if abs(weight_sum - 1.0) < abs(best_sum - 1.0):
                    best_sum = weight_sum
                    best_face = face_idx
                    best_weights = weights / weight_sum
    
    return best_face, best_weights

def barycentric_coordinates(p, a, b, c):
    """
    Compute barycentric coordinates using cross products (no matrix solve).
    Much faster than np.linalg.solve.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return None
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    return np.array([u, v, w])


def generate_icosahedral_mesh(refinement_level: int, radius: float = 1.0):
    """
    Generates a spherical icosahedral mesh using Trimesh.
    
    Args:
        refinement_level (int): Number of subdivisions. Must be non-negative.
        radius (float): Radius of the sphere (default 1.0 for unit sphere).
        
    Returns:
        nodes (np.ndarray): Shape (N, 3) Cartesian coordinates (x, y, z).
        faces (np.ndarray): Shape (M, 3) Triangular faces connecting the nodes.
        
    Raises:
        ValueError: If refinement_level is negative.
        ImportError: If trimesh is not available.
    """
    if refinement_level < 0:
        raise ValueError("subdivisions must be non-negative")
    
    # Try to import trimesh
    try:
        import trimesh
    except ImportError as e:
        raise ImportError("trimesh is required for icosahedral mesh generation. "
                         "Please install it with: pip install trimesh") from e
    
    try:
        mesh = trimesh.creation.icosphere(subdivisions=refinement_level, radius=radius)
    except ImportError as e:
        # re-raise with our custom message
        raise ImportError("trimesh is required for icosahedral mesh generation. "
                         "Please install it with: pip install trimesh") from e
    
    # Extract nodes and faces
    nodes = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    return nodes, faces
