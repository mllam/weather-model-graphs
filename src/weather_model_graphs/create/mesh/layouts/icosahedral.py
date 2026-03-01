"""Icosahedral mesh layout for global graphs."""

import numpy as np
import networkx as nx
import trimesh
from scipy.spatial import KDTree


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
    
    # Create undirected graph from faces
    G = nx.Graph()
    
    # Add nodes with positions
    for i, (x, y, z) in enumerate(vertices):
        G.add_node(i, pos=np.array([x, y, z]), type="mesh", level=0)
    
    # Add edges from faces
    for face in faces:
        for i in range(3):
            for j in range(i+1, 3):
                u, v = face[i], face[j]
                if not G.has_edge(u, v):
                    G.add_edge(u, v)
    
    # Convert to directed graph with both directions
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes(data=True))
    
    for u, v in G.edges():
        vec = G.nodes[u]["pos"] - G.nodes[v]["pos"]
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
    
    # Create base graph with finest level
    DG = create_flat_icosahedral_mesh_graph(max_subdivisions, radius, 
                                            add_edge_length, add_edge_vector)
    
    # Add coarser levels as additional nodes with level attribute
    node_offset = len(DG.nodes)
    
    for level, (vertices, faces) in enumerate(mesh_list[:-1]):  # Skip finest (already added)
        level_nodes = []
        
        # Add nodes for this level
        for i, (x, y, z) in enumerate(vertices):
            node_id = node_offset + i
            DG.add_node(node_id, pos=np.array([x, y, z]), type="mesh", level=level)
            level_nodes.append(node_id)
        
        # Add intra-level edges
        for face in faces:
            for i in range(3):
                for j in range(i+1, 3):
                    u, v = node_offset + face[i], node_offset + face[j]
                    if not DG.has_edge(u, v):
                        vec = DG.nodes[u]["pos"] - DG.nodes[v]["pos"]
                        dist = np.linalg.norm(vec)
                        DG.add_edge(u, v, len=dist, vdiff=vec, level=level)
                        DG.add_edge(v, u, len=dist, vdiff=-vec, level=level)
        
        # Add inter-level edges (coarse-to-fine)
        if level < max_subdivisions - 1:
            # Connect this level to next finer level
            finer_vertices = mesh_list[level + 1][0]
            tree = KDTree(finer_vertices)
            
            # For each coarse node, find closest fine nodes
            for i, coarse_node in enumerate(level_nodes):
                coarse_pos = vertices[i]
                # Find fine nodes within radius (heuristic: 1.1 * max edge length)
                max_edge_len = compute_max_edge_length(finer_vertices, mesh_list[level + 1][1])
                radius_query = 1.1 * max_edge_len
                
                fine_indices = tree.query_ball_point(coarse_pos, radius_query)
                if fine_indices:
                    # Add edges in both directions with appropriate attributes
                    for fine_idx in fine_indices:
                        vec = coarse_pos - finer_vertices[fine_idx]
                        dist = np.linalg.norm(vec)
                        
                        # Up edge (fine -> coarse)
                        DG.add_edge(
                            node_offset - len(finer_vertices) + fine_idx, 
                            coarse_node,
                            len=dist, vdiff=vec, level=f"{level}_to_{level+1}"
                        )
                        # Down edge (coarse -> fine)
                        DG.add_edge(
                            coarse_node,
                            node_offset - len(finer_vertices) + fine_idx,
                            len=dist, vdiff=-vec, level=f"{level+1}_to_{level}"
                        )
        
        node_offset += len(vertices)
    
    DG.graph["mesh_layout"] = "icosahedral_hierarchical"
    DG.graph["max_subdivisions"] = max_subdivisions
    DG.graph["radius"] = radius
    
    return DG


def connect_grid_to_mesh(
    grid_lat_lon: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    radius_factor: float = 0.6,
    add_edge_attributes: bool = True,
):
    """
    Grid to Mesh connections using radius-based neighbor lookup.
    
    Args:
        grid_lat_lon: (N_grid, 2) array of [lat, lon] in degrees
        mesh_vertices: (N_mesh, 3) cartesian coordinates
        mesh_faces: (M, 3) face indices
        radius_factor: multiplier for max edge distance
        add_edge_attributes: If True, add 'len' and 'vdiff' attributes
    
    Returns:
        edge_index: (2, E) array of [grid_node, mesh_node] connections
    """
    grid_cartesian = lat_lon_to_cartesian(grid_lat_lon[:, 0], grid_lat_lon[:, 1])
    
    max_edge_len = compute_max_edge_length(mesh_vertices, mesh_faces)
    query_radius = radius_factor * max_edge_len
    
    tree = KDTree(mesh_vertices)
    
    grid_indices, mesh_indices = [], []
    for i, point in enumerate(grid_cartesian):
        neighbors = tree.query_ball_point(point, query_radius)
        if neighbors:
            grid_indices.extend([i] * len(neighbors))
            mesh_indices.extend(neighbors)
    
    return np.array([grid_indices, mesh_indices])


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
        # This is a simplified approach - for production, use proper spherical barycentric
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