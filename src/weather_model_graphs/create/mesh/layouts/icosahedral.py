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
        pos_u = DG.nodes[u]["pos"]
        pos_v = DG.nodes[v]["pos"]
        
        # Calculate latitude difference (no wrap needed)
        dlat = pos_u[0] - pos_v[0]
        
        # Calculate longitude difference with proper wrapping
        dlon = pos_u[1] - pos_v[1]
        dlon = (dlon + 180) % 360 - 180  # wrap to [-180, 180]
        
        # Store the wrapped difference
        vec = np.array([dlat, dlon])

        dist = np.linalg.norm(DG.nodes[u]["pos3d"] - DG.nodes[v]["pos3d"])

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

    # Start with finest mesh as the base graph (level = max_subdivisions)
    DG = create_flat_icosahedral_mesh_graph(max_subdivisions, radius,
                                            add_edge_length, add_edge_vector)
    # Nodes 0..(N_finest-1) are already in DG at level=max_subdivisions (set by create_flat)
    # Relabel their level correctly
    for node in DG.nodes:
        DG.nodes[node]["level"] = max_subdivisions
    for u, v in DG.edges:
        DG.edges[u, v]["level"] = max_subdivisions

    node_offset = len(DG.nodes)  # = N_finest (e.g. 642)

    # Add coarser levels: mesh_list[0..max_subdivisions-1]
    for level, (vertices, faces) in enumerate(mesh_list[:-1]):
        # level=0 -> coarsest (12 nodes), level=max_subdivisions-1 -> second finest
        lat_lon = cartesian_to_lat_lon(vertices)
        level_nodes = []

        for i, (x, y, z) in enumerate(vertices):
            node_id = node_offset + i
            DG.add_node(
                node_id,
                pos=lat_lon[i],
                pos3d=np.array([x, y, z]),
                type="mesh",
                level=level,           # correct level label
            )
            level_nodes.append(node_id)

        # Intra-level edges
        for face in faces:
            for i in range(3):
                for j in range(i + 1, 3):
                    u, v = node_offset + face[i], node_offset + face[j]
                    if not DG.has_edge(u, v):
                        vec = DG.nodes[u]["pos"] - DG.nodes[v]["pos"]
                        dist = np.linalg.norm(DG.nodes[u]["pos3d"] - DG.nodes[v]["pos3d"])
                        DG.add_edge(u, v, len=dist, vdiff=vec, level=level)
                        DG.add_edge(v, u, len=dist, vdiff=-vec, level=level)

        # Inter-level edges: connect this coarse level to the next finer level
        # Next finer level = level+1, already in the graph
        # Its nodes start at: offset_of_finer_level
        if level == max_subdivisions - 1:
            # Next finer is the finest mesh, nodes 0..N_finest-1
            finer_start = 0
            finer_vertices = mesh_list[max_subdivisions][0]
        else:
            # Next finer is mesh_list[level+1], added after this iteration
            # We compute its future offset: node_offset + len(vertices) 
            finer_start = node_offset + len(vertices)
            finer_vertices = mesh_list[level + 1][0]

        tree = KDTree(finer_vertices)
        max_edge_len = compute_max_edge_length(finer_vertices, mesh_list[level + 1][1]
                                               if level < max_subdivisions - 1
                                               else mesh_list[max_subdivisions][1])
        radius_query = 1.1 * max_edge_len

        for i, coarse_node in enumerate(level_nodes):
            coarse_pos = vertices[i]
            fine_indices = tree.query_ball_point(coarse_pos, radius_query)
            for fine_idx in fine_indices:
                fine_node = finer_start + fine_idx
                coarse_lat_lon = DG.nodes[coarse_node]["pos"]
                # fine_node may not exist yet if finer_start > node_offset (future level)
                # so use coordinates directly
                fine_lat_lon = cartesian_to_lat_lon(finer_vertices[fine_idx:fine_idx+1])[0]
                vec3d = coarse_pos - finer_vertices[fine_idx]
                dist = np.linalg.norm(vec3d)
                vec2d = coarse_lat_lon - fine_lat_lon

                DG.add_edge(fine_node, coarse_node,
                            len=dist, vdiff=vec2d,
                            level=f"{level+1}_to_{level}")
                DG.add_edge(coarse_node, fine_node,
                            len=dist, vdiff=-vec2d,
                            level=f"{level}_to_{level+1}")

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
    grid_cartesian = lat_lon_to_cartesian(grid_lat_lon[:, 0], grid_lat_lon[:, 1])

    mesh_indices, grid_indices, weights = [], [], []

    for grid_idx, point in enumerate(grid_cartesian):
        face_idx, bary_weights = find_containing_triangle(point, mesh_vertices, mesh_faces)
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