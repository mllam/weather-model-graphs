"""Icosahedral mesh layout for global graphs."""

import warnings

import networkx as nx
import numpy as np
from scipy.spatial import KDTree


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
    return mesh_list


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
            level=None,
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
        vec = DG.nodes[u]["pos3d"] - DG.nodes[v]["pos3d"]
        dist = np.linalg.norm(vec)
        DG.add_edge(u, v, len=dist, vdiff=vec, level=None)
        DG.add_edge(v, u, len=dist, vdiff=-vec, level=None)

    DG.graph["mesh_layout"] = "icosahedral"
    DG.graph["subdivisions"] = subdivisions
    DG.graph["radius"] = radius
    DG.graph["vertices"] = vertices
    DG.graph["faces"] = faces
    DG.graph["is_hierarchical"] = False

    return DG


def create_hierarchical_icosahedral_mesh_graph(
    max_subdivisions: int = 3,
    radius: float = 1.0,
    add_edge_length: bool = True,
    add_edge_vector: bool = True,
):
    """
    Create a hierarchical icosahedral mesh graph with multiple refinement levels.
    """
    mesh_list = create_hierarchy_of_icosahedral_meshes(max_subdivisions, radius)

    level_offsets = {}
    current_offset = 0
    for level in range(max_subdivisions, -1, -1):
        level_offsets[level] = current_offset
        vertices = mesh_list[level][0]
        current_offset += len(vertices)

    DG = nx.DiGraph()
    vertices_by_level = []
    faces_by_level = []

    for level in range(max_subdivisions + 1):
        vertices, faces = mesh_list[level]
        vertices_by_level.append(vertices)
        faces_by_level.append(faces)
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

    for coarse_level in range(max_subdivisions):
        fine_level = coarse_level + 1
        coarse_vertices, _ = mesh_list[coarse_level]
        fine_vertices, fine_faces = mesh_list[fine_level]
        coarse_offset = level_offsets[coarse_level]
        fine_offset = level_offsets[fine_level]

        tree = KDTree(fine_vertices)
        max_edge_len = compute_max_edge_length(fine_vertices, fine_faces)
        radius_query = 1.1 * max_edge_len

        for i, coarse_pos in enumerate(coarse_vertices):
            coarse_node = coarse_offset + i
            fine_indices = tree.query_ball_point(coarse_pos, radius_query)
            for fine_idx in fine_indices:
                fine_node = fine_offset + fine_idx
                vec = coarse_pos - fine_vertices[fine_idx]
                dist = np.linalg.norm(vec)
                DG.add_edge(
                    fine_node,
                    coarse_node,
                    len=dist,
                    vdiff=vec,
                    level=f"{fine_level}_to_{coarse_level}",
                )
                DG.add_edge(
                    coarse_node,
                    fine_node,
                    len=dist,
                    vdiff=-vec,
                    level=f"{coarse_level}_to_{fine_level}",
                )

    DG.graph["mesh_layout"] = "icosahedral_hierarchical"
    DG.graph["max_subdivisions"] = max_subdivisions
    DG.graph["radius"] = radius
    DG.graph["level_offsets"] = level_offsets
    DG.graph["mesh_vertices_by_level"] = vertices_by_level
    DG.graph["mesh_faces_by_level"] = faces_by_level
    DG.graph["is_hierarchical"] = True

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
    if len(grid_lat_lon) == 0:
        return np.array([[], []], dtype=int)

    grid_cartesian = lat_lon_to_cartesian(grid_lat_lon[:, 0], grid_lat_lon[:, 1])
    max_edge_len = compute_max_edge_length(mesh_vertices, mesh_faces)
    query_radius = radius_factor * max_edge_len
    tree = KDTree(mesh_vertices)
    neighbor_lists = tree.query_ball_point(grid_cartesian, query_radius)

    total_connections = sum(len(neighbors) for neighbors in neighbor_lists)
    if total_connections == 0:
        return np.array([[], []], dtype=int)

    grid_indices = np.zeros(total_connections, dtype=int)
    mesh_indices = np.zeros(total_connections, dtype=int)
    start_idx = 0
    for i, neighbors in enumerate(neighbor_lists):
        if neighbors:
            n_neighbors = len(neighbors)
            end_idx = start_idx + n_neighbors
            grid_indices[start_idx:end_idx] = i
            mesh_indices[start_idx:end_idx] = neighbors
            start_idx = end_idx

    return np.array([grid_indices, mesh_indices])


def connect_mesh_to_grid(
    mesh_vertices, mesh_faces, grid_lat_lon, fallback_to_nearest=True
):
    """
    Mesh to Grid connections (m2g).
    For each grid point, find containing mesh triangle and return
    barycentric weights for interpolation. Falls back to nearest neighbour
    if triangle containment fails.

    Returns:
        edge_index: (2, E) array of [mesh_node, grid_node] connections
        weights: (E,) barycentric weights for each edge
    """
    grid_cartesian = lat_lon_to_cartesian(grid_lat_lon[:, 0], grid_lat_lon[:, 1])
    face_centroids = mesh_vertices[mesh_faces].mean(axis=1)
    centroid_tree = KDTree(face_centroids)

    if fallback_to_nearest:
        vertex_tree = KDTree(mesh_vertices)

    mesh_indices, grid_indices, weights = [], [], []
    failed_points = 0

    for grid_idx, point in enumerate(grid_cartesian):
        face_idx, bary_weights = find_containing_triangle(
            point,
            mesh_vertices,
            mesh_faces,
            face_centroids,
            centroid_tree,
            k_candidates=10,
        )

        # Clip negative weights (numerical noise near edges), renormalise to sum=1.
        # If all weights clip to zero (degenerate case), treat as a failed containment.
        contained = False
        if face_idx is not None:
            clipped = np.clip(bary_weights, 0.0, None)
            w_sum = clipped.sum()
            if w_sum > 1e-12:
                normalised = clipped / w_sum
                for mesh_idx, w in zip(mesh_faces[face_idx], normalised):
                    # Skip zero-weight vertices (on triangle edge) — contribute nothing
                    if w > 0.0:
                        mesh_indices.append(mesh_idx)
                        grid_indices.append(grid_idx)
                        weights.append(float(w))
                contained = True

        if not contained:
            if fallback_to_nearest:
                failed_points += 1
                dist, nearest_idx = vertex_tree.query(point)
                mesh_indices.append(nearest_idx)
                grid_indices.append(grid_idx)
                weights.append(1.0)
            else:
                failed_points += 1

    if failed_points > 0 and len(grid_cartesian) > 0:
        total_points = len(grid_cartesian)
        warnings.warn(
            f"Triangle containment failed for {failed_points}/{total_points} "
            f"({failed_points / total_points * 100:.1f}%) grid points. "
            f"{'Used nearest neighbour fallback.' if fallback_to_nearest else 'Points were skipped.'}",
            UserWarning,
        )

    if len(mesh_indices) == 0:
        return np.array([[], []], dtype=int), np.array([])

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
    lon = np.degrees(np.arctan2(y, x))

    if np.any(np.abs(z) > 1.0):
        n_clipped = np.sum(np.abs(z) > 1.0)
        warnings.warn(
            f"Clipped {n_clipped} values outside [-1, 1] in cartesian_to_lat_lon. "
            "This is likely due to floating point errors.",
            UserWarning,
            stacklevel=2,
        )

    z_clipped = np.clip(z, -1.0, 1.0)
    lat = np.degrees(np.arcsin(z_clipped))
    return np.column_stack([lat, lon])


def compute_max_edge_length(vertices, faces):
    """Compute longest edge in mesh."""
    edge_pairs = faces[:, [[0, 1], [1, 2], [2, 0]]]
    all_edges = edge_pairs.reshape(-1, 2)
    all_edges = np.unique(np.sort(all_edges, axis=1), axis=0)
    v1 = vertices[all_edges[:, 0]]
    v2 = vertices[all_edges[:, 1]]
    edge_lengths = np.linalg.norm(v1 - v2, axis=1)
    return np.max(edge_lengths)


def find_containing_triangle(
    point_cartesian: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    face_centroids: np.ndarray = None,
    centroid_tree: KDTree = None,
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
    if face_centroids is None or centroid_tree is None:
        face_centroids = mesh_vertices[mesh_faces].mean(axis=1)
        centroid_tree = KDTree(face_centroids)

    candidate_indices = centroid_tree.query(point_norm, k=k_candidates)[1]

    best_face = None
    best_weights = None
    best_sum = float("inf")

    for face_idx in candidate_indices:
        face = mesh_faces[face_idx]
        a, b, c = mesh_vertices[face]
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

    try:
        import trimesh
    except ImportError as e:
        raise ImportError(
            "trimesh is required for icosahedral mesh generation. "
            "Please install it with: pip install trimesh"
        ) from e

    try:
        mesh = trimesh.creation.icosphere(subdivisions=refinement_level, radius=radius)
    except ImportError as e:
        raise ImportError(
            "trimesh is required for icosahedral mesh generation. "
            "Please install it with: pip install trimesh"
        ) from e

    nodes = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    return nodes, faces


def refinement_level_from_grid_spacing(
    grid_spacing_deg: float, radius: float = 1.0
) -> int:
    """Determine the appropriate refinement level for a desired grid spacing.

    Selects the finest icosahedral refinement level whose mesh spacing is still
    >= grid_spacing_deg. This ensures the mesh is not finer than the input grid,
    avoiding unnecessary computation while maintaining adequate coverage.
    """
    # Approximate angular spacing in degrees for each refinement level
    level_spacing_deg = {
        0: 63.4,  # level0
        1: 31.7,  # level1
        2: 15.8,  # level2
        3: 7.9,  # level3
        4: 3.95,  # level4
        5: 1.98,  # level5
    }

    # Find all levels whose mesh spacing is >= requested grid spacing
    coarse_enough = {
        lvl: spacing
        for lvl, spacing in level_spacing_deg.items()
        if spacing >= grid_spacing_deg
    }

    if coarse_enough:
        # Pick the finest (highest level) among valid candidates
        chosen_level = max(coarse_enough.keys())
    else:
        # Requested spacing is finer than any available level → use finest
        chosen_level = max(level_spacing_deg.keys())
        warnings.warn(
            f"Requested grid spacing {grid_spacing_deg}° is finer than the finest "
            f"available mesh spacing {level_spacing_deg[chosen_level]:.2f}° at level {chosen_level}. "
            f"Using finest available level.",
            UserWarning,
        )

    return chosen_level
