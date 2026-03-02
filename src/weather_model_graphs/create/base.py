"""
Generic routines for creating the graph components used in the message-passing
graph, and for connecting nodes across these component graphs.

The graph components, grid-to-mesh (g2m), mesh-to-mesh (m2m) and mesh-to-grid (m2g), are
used to represent the encode-process-decode steps respectively. These are created with
`create_all_graph_components` which takes the following arguments. Internally, this
function uses `connect_nodes_across_graphs` to connect nodes across the component graphs.
"""


from typing import Iterable
import warnings
import networkx
import networkx as nx
import numpy as np
import pyproj
import scipy.spatial
from loguru import logger

from ..networkx_utils import (
    replace_node_labels_with_unique_ids,
    split_graph_by_edge_attribute,
    split_on_edge_attribute_existance,
)
from .grid import create_grid_graph_nodes
from .mesh.kinds.flat import (
    create_flat_multiscale_mesh_graph,
    create_flat_singlescale_mesh_graph,
)
from .mesh.kinds.hierarchical import create_hierarchical_multiscale_mesh_graph


def create_all_graph_components(
    coords: np.ndarray,
    m2m_connectivity: str,
    m2g_connectivity: str,
    g2m_connectivity: str,
    m2m_connectivity_kwargs={},
    m2g_connectivity_kwargs={},
    g2m_connectivity_kwargs={},
    coords_crs: pyproj.crs.CRS | None = None,
    graph_crs: pyproj.crs.CRS | None = None,
    decode_mask: Iterable[bool] | None = None,
    return_components: bool = False,
):
    """
    Create all graph components used in creating the message-passing graph,
        grid-to-mesh (g2m), mesh-to-mesh (m2m) and mesh-to-grid (m2g),
    representing the encode-process-decode respectively.

    For each graph component, the method for connecting nodes across graphs
    should be specified (with the `*_connectivity` arguments, e.g. `m2g_connectivity`).
    And the method-specific arguments should be passed as keyword arguments using
    the `*_connectivity_kwargs` arguments (e.g. `m2g_connectivity_kwargs`).

    The following methods are available for connecting nodes across graphs:

    g2m_connectivity:
    - "nearest_neighbour": Find the nearest neighbour in grid for each node in mesh
    - "nearest_neighbours": Find the `max_num_neighbours` nearest neighbours in grid for each node in mesh
    - "within_radius": Find all neighbours in grid within an absolute distance
        of `max_dist` or relative distance of `rel_max_dist` from each node in mesh

    m2m_connectivity:
    - "flat": Create a single-level 2D mesh graph with `mesh_node_distance`,
        similar to Keisler et al. (2022)
    - "flat_multiscale": Create a flat multiscale mesh graph with `max_num_levels`,
        `mesh_node_distance` and `level_refinement_factor`,
        similar to GraphCast, Lam et al. (2023)
    - "hierarchical": Create a hierarchical mesh graph with `max_num_levels`,
        `mesh_node_distance` and `level_refinement_factor`,
        similar to Oskarsson et al. (2023)

    m2g_connectivity:
    - "nearest_neighbour": Find the nearest neighbour in mesh for each node in grid
    - "nearest_neighbours": Find the `max_num_neighbours` nearest neighbours in mesh for each node in grid
    - "within_radius": Find all neighbours in mesh within an absolute distance
        of `max_dist` or relative distance of `rel_max_dist` from each node in grid
    - "containing_rectangle": For each grid node, find the rectangle with 4 mesh nodes as corners
        such that the grid node is contained within it. Connect these 4 (or less along edges)
        mesh nodes to the grid node.

    `coords_crs` and `graph_crs` should either be a pyproj.crs.CRS or None.
    Note that this includes a cartopy.crs.CRS. If both are given the coordinates
    will be transformed from their original Coordinate Reference System (`coords_crs`)
    to the CRS where the graph creation should take place (`graph_crs`).
    If any one of them is None the graph creation is carried out using the original coords.

    `decode_mask` should be an Iterable of booleans, masking which grid positions should be
    decoded to (included in the m2g subgraph), i.e. which positions should be output. It should have the same length as the number of
    grid position coordinates given in `coords`.  The mask being set to True means that corresponding
    grid nodes should be included in g2m. If `decode_mask=None` (default), all grid nodes are included.

    `return_components` is a boolean flag, if True the function returns a dict with
    m2g, m2m and g2m as separate graphs. If false returns one combined graph.
    """
    graph_components: dict[networkx.DiGraph] = {}

    assert (
        len(coords.shape) == 2 and coords.shape[1] == 2
    ), "Grid node coordinates should be given as an array of shape [num_grid_nodes, 2]."

    # Translate between coordinate crs and crs to use for graph creation
    if coords_crs is None and coords_crs is None:
        logger.debug(
            "No `coords_crs` given: Assuming `coords` contains in-projection Cartesian coordinates."
        )
        xy = coords
    elif (coords_crs is None) != (graph_crs is None):  # xor, only one is None
        logger.warning(
            "Only one of `coords_crs` and `graph_crs` given. Both are needed to "
            "transform coordinates to a different crs for constructing the graph: "
            "Assuming `coords` contains in-projection Cartesian coordinates."
        )
        xy = coords
    else:
        logger.debug(
            f"Projecting coords from CRS({coords_crs}) to CRS({graph_crs}) for graph creation."
        )
        # Convert from coords_crs to to graph_crs
        coord_transformer = pyproj.Transformer.from_crs(
            coords_crs, graph_crs, always_xy=True
        )
        xy_tuple = coord_transformer.transform(xx=coords[:, 0], yy=coords[:, 1])
        xy = np.stack(xy_tuple, axis=1)

    if m2m_connectivity == "flat":
        graph_components["m2m"] = create_flat_singlescale_mesh_graph(
            xy,
            **m2m_connectivity_kwargs,
        )
        grid_connect_graph = graph_components["m2m"]
    elif m2m_connectivity == "hierarchical":
        # hierarchical mesh graph have three sub-graphs:
        # `m2m` (mesh-to-mesh), `mesh_up` (up edge connections) and `mesh_down` (down edge connections)
        graph_components["m2m"] = create_hierarchical_multiscale_mesh_graph(
            xy=xy,
            **m2m_connectivity_kwargs,
        )
        # Only connect grid to bottom level of hierarchy
        grid_connect_graph = split_graph_by_edge_attribute(
            graph_components["m2m"], "level"
        )[0]
    elif m2m_connectivity == "flat_multiscale":
        graph_components["m2m"] = create_flat_multiscale_mesh_graph(
            xy=xy,
            **m2m_connectivity_kwargs,
        )
        grid_connect_graph = graph_components["m2m"]
    elif m2m_connectivity == "icosahedral":
        from weather_model_graphs.create.mesh.layouts.icosahedral import (
            create_flat_icosahedral_mesh_graph,
            generate_icosahedral_mesh,
            create_hierarchical_icosahedral_mesh_graph,
        )

        if graph_crs is not None and not graph_crs.is_geographic:
            warnings.warn(
                "Icosahedral mesh is designed for geographic coordinates. "
                "Using with non-geographic CRS may produce unexpected results.",
                UserWarning
            )

        if m2m_connectivity_kwargs.get("hierarchical", False):
            graph_components["m2m"] = create_hierarchical_icosahedral_mesh_graph(
                max_subdivisions=m2m_connectivity_kwargs.get("max_subdivisions", 3),
                radius=m2m_connectivity_kwargs.get("radius", 1.0),
            )

            # Connect grid to finest level only
            grid_connect_graph = split_graph_by_edge_attribute(
                graph_components["m2m"], "level"
            )[0]    
        else:
            graph_components["m2m"] = create_flat_icosahedral_mesh_graph(
                subdivisions=m2m_connectivity_kwargs.get("subdivisions", 3),
                radius=m2m_connectivity_kwargs.get("radius", 1.0),
            )
            grid_connect_graph = graph_components["m2m"]
    else:
        graph_components["m2m"] = create_flat_icosahedral_mesh_graph(
            subdivisions=m2m_connectivity_kwargs.get("subdivisions", 3),
            radius=m2m_connectivity_kwargs.get("radius", 1.0),
        )
        grid_connect_graph = graph_components["m2m"]
    # else:
    #     raise NotImplementedError(f"Kind {m2m_connectivity} not implemented")

    G_grid = create_grid_graph_nodes(xy=xy)

    G_g2m = connect_nodes_across_graphs(
        G_source=G_grid,
        G_target=grid_connect_graph,
        method=g2m_connectivity,
        **g2m_connectivity_kwargs,
    )
    graph_components["g2m"] = G_g2m

    if decode_mask is None:
        # decode to all grid nodes
        decode_grid = G_grid
    else:
        # Select subset of grid nodes to decode to, where m2g should connect
        filter_nodes = [
            n for n, include in zip(G_grid.nodes, decode_mask, strict=True) if include
        ]
        decode_grid = G_grid.subgraph(filter_nodes)

    G_m2g = connect_nodes_across_graphs(
        G_source=grid_connect_graph,
        G_target=decode_grid,
        method=m2g_connectivity,
        **m2g_connectivity_kwargs,
    )
    graph_components["m2g"] = G_m2g

    # add graph component identifier to each edge in each component graph
    for name, graph in graph_components.items():
        for edge in graph.edges:
            graph.edges[edge]["component"] = name

    if return_components:
        # Because merging to a single graph and then splitting again leads to changes in node indexing when converting to `pyg.Data` objects (this in part is due to the to `m2g` and `g2m` having a different set of grid nodes) the ability to return the graph components (`g2m`, `m2m` and `m2g`) has been added here. See https://github.com/mllam/weather-model-graphs/pull/34#issuecomment-2507980752 for details
        # Give each component unique ids
        graph_components = {
            comp_name: replace_node_labels_with_unique_ids(subgraph)
            for comp_name, subgraph in graph_components.items()
        }
        return graph_components

    # merge to single graph
    G_tot = networkx.compose_all(graph_components.values())
    # only keep graph attributes that are the same for all components
    for key in graph_components["m2m"].graph.keys():
        if not all(
            graph.graph.get(key, None) == graph_components["m2m"].graph[key]
            for graph in graph_components.values()
        ):
            # delete
            del G_tot.graph[key]

    G_tot = replace_node_labels_with_unique_ids(graph=G_tot)

    return G_tot


def connect_nodes_across_graphs(
    G_source,
    G_target,
    method="nearest_neighbour",
    max_dist=None,
    rel_max_dist=None,
    max_num_neighbours=None,
    mesh_vertices=None,
    mesh_faces=None,
):
    """
    Create a new graph containing the nodes in `G_source` and `G_target` and add
    directed edges from nodes in `G_source` to nodes in `G_target` based on the
    method specified.

    This can for example be used to create mesh-to-grid (m2g) connections where
    each grid node (the target) has connections to it from the 4 nearest mesh nodes (the source) by using
    the `nearest_neighbours` method with `max_num_neighbours=4`

    Parameters
    ----------
    G_source : networkx.Graph
        Source graph, edge connections are made from nodes in this graph (existing edges are ignored)
    G_target : networkx.Graph
        Target graph, edge connections are made to nodes in this graph (existing edges are ignored)
    method : str
        Method to use for finding neighbours in `G_source` for each node in `G_target`.
        Options are:
        - "nearest_neighbour": Find the nearest neighbour in `G_target` for each node in `G_source`
        - "nearest_neighbours": Find the `max_num_neighbours` nearest neighbours in `G_target` for each node in `G_source`
        - "within_radius": Find all neighbours in `G_target` within a distance of `max_dist` from each node in `G_source`
        - "containing_rectangle": For each node in `G_target`, find the rectangle in `G_source`
            with 4 nodes as corners such that the `G_target` node is contained within it.
            Connect these 4 (or less along edges) corner nodes to the `G_target` node.
            Requires that `G_source` has dx and dy properties, i.e. is a quadrilateral mesh graph.
        - "containing_triangle": For each node in `G_target`, find the spherical triangle in `G_source`
            that contains it and connect the 3 corner mesh nodes with barycentric weights as edge attributes.
            Requires mesh_vertices and mesh_faces to be passed.
    max_dist : float
        Maximum distance to search for neighbours in `G_target` for each node in `G_source`
    rel_max_dist : float
        Maximum distance to search for neighbours in `G_target` for each node in `G_source`,
        relative to longest edge in (bottom level of) `G_source` and `G_target`.
    max_num_neighbours : int
        Maximum number of neighbours to search for in `G_target` for each node in `G_source`
    mesh_vertices : np.ndarray, optional
        (N, 3) Cartesian coordinates of mesh vertices. Required for `containing_triangle` method.
    mesh_faces : np.ndarray, optional
        (M, 3) face indices of mesh triangles. Required for `containing_triangle` method.

    Returns
    -------
    networkx.DiGraph
        Graph containing the nodes in `G_source` and `G_target` and directed edges
        from nodes in `G_source` to nodes in `G_target`
    """
    source_nodes_list = sorted(G_source.nodes)
    target_nodes_list = sorted(G_target.nodes)

    sample_source = source_nodes_list[0]
    sample_target = target_nodes_list[0]

    source_has_3d = "pos3d" in G_source.nodes[sample_source]
    target_has_3d = "pos3d" in G_target.nodes[sample_target]

    # Prepare source points for KDTree
    if source_has_3d:
        # m2g: mesh is source, use its native 3D positions
        xy_source = np.array([G_source.nodes[n]["pos3d"] for n in source_nodes_list])
        use_3d = True
    elif target_has_3d:
        # g2m: grid is source, project lat/lon → 3D to match mesh edge-length units
        from weather_model_graphs.create.mesh.layouts.icosahedral import lat_lon_to_cartesian
        source_lats = np.array([G_source.nodes[n]["pos"][0] for n in source_nodes_list])
        source_lons = np.array([G_source.nodes[n]["pos"][1] for n in source_nodes_list])
        xy_source = lat_lon_to_cartesian(source_lats, source_lons)
        use_3d = True
    else:
        # plain 2D (rectilinear grids, no icosahedral mesh)
        # Need to handle longitude wrapping by duplicating points
        source_lat_lon = np.array([G_source.nodes[n]["pos"] for n in source_nodes_list])
        
        # Create three copies shifted by -360°, 0°, and +360° in longitude
        source_positions = []
        for offset in [-360, 0, 360]:
            shifted = source_lat_lon.copy()
            shifted[:, 1] += offset
            source_positions.append(shifted)
        
        xy_source = np.vstack(source_positions)
        # Store mapping from KDTree index to original node
        source_node_mapping = []
        for offset_idx, offset in enumerate([-360, 0, 360]):
            for orig_idx, node in enumerate(source_nodes_list):
                source_node_mapping.append((node, offset, orig_idx))
        use_3d = False

    kdt_s = scipy.spatial.KDTree(xy_source)

    if method == "containing_rectangle":
        if (
            max_dist is not None
            or rel_max_dist is not None
            or max_num_neighbours is not None
        ):
            raise Exception(
                "to use `containing_rectangle` you should not set `max_dist`, `rel_max_dist` or `max_num_neighbours`"
            )
        assert (
            "dx" in G_source.graph and "dy" in G_source.graph
        ), "Source graph must have dx and dy properties to connect nodes using method containing_rectangle"

        rad_graph = connect_nodes_across_graphs(
            G_source, G_target, method="within_radius", rel_max_dist=1.0
        )

        mesh_node_dx = G_source.graph["dx"]
        mesh_node_dy = G_source.graph["dy"]

        if isinstance(mesh_node_dx, dict):
            mesh_node_dx = mesh_node_dx[0]
            mesh_node_dy = mesh_node_dy[0]

        def _edge_filter(edge_prop):
            abs_diffs = np.abs(edge_prop["vdiff"])
            return abs_diffs[0] < mesh_node_dx and abs_diffs[1] < mesh_node_dy

        filtered_edges = [
            (u, v, edge_prop)
            for u, v, edge_prop in rad_graph.edges(data=True)
            if _edge_filter(edge_prop)
        ]

        filtered_graph = networkx.DiGraph()
        filtered_graph.add_nodes_from(rad_graph.nodes(data=True))
        filtered_graph.add_edges_from(filtered_edges)
        return filtered_graph

    elif method == "nearest_neighbour":
        if (
            max_dist is not None
            or rel_max_dist is not None
            or max_num_neighbours is not None
        ):
            raise Exception(
                "to use `nearest_neighbour` you should not set `max_dist`, `rel_max_dist` or `max_num_neighbours`"
            )

        def _find_neighbour_node_idxs_in_source_mesh(query_point):
            if use_3d:
                neigh_idx = kdt_s.query(query_point, 1)[1]
                return [neigh_idx]
            else:
                # For 2D, query all three copies and map back
                all_neigh_idxs = []
                for qp in query_point:
                    neigh_idx = kdt_s.query(qp, 1)[1]
                    all_neigh_idxs.append(neigh_idx)
                
                # Map back to original indices
                original_idxs = set()
                for idx in all_neigh_idxs:
                    original_idxs.add(source_node_mapping[idx][2])  # orig_idx
                return list(original_idxs)

    elif method == "nearest_neighbours":
        if max_num_neighbours is None:
            raise Exception(
                "to use `nearest_neighbours` you should set the max number with `max_num_neighbours`"
            )
        if max_dist is not None or rel_max_dist is not None:
            raise Exception(
                "to use `nearest_neighbours` you should not set `max_dist` or `rel_max_dist`"
            )

        def _find_neighbour_node_idxs_in_source_mesh(query_point):
            if use_3d:
                neigh_idxs = kdt_s.query(query_point, max_num_neighbours)[1]
                return neigh_idxs
            else:
                # For 2D, query all three copies
                all_neigh_idxs = []
                for qp in query_point:
                    neigh_idxs = kdt_s.query(qp, max_num_neighbours)[1]
                    all_neigh_idxs.extend(neigh_idxs)
                
                # Map back to original indices, removing duplicates
                original_idxs = set()
                for idx in all_neigh_idxs:
                    original_idxs.add(source_node_mapping[idx][2])
                return list(original_idxs)

    elif method == "within_radius":
        if max_num_neighbours is not None:
            raise Exception(
                "to use `within_radius` method you should not set `max_num_neighbours`"
            )
        
        # Determine query distance
        if max_dist is not None:
            query_dist = max_dist
        elif rel_max_dist is not None:
            # Calculate based on longest edge
            longest_edge = 0.0
            for edge_check_graph in (G_source, G_target):
                if len(edge_check_graph.edges) > 0:
                    # Get edges with 'len' attribute
                    edge_lengths = []
                    for _, _, data in edge_check_graph.edges(data=True):
                        if 'len' in data:
                            edge_lengths.append(data['len'])
                    
                    if edge_lengths:
                        longest_graph_edge = max(edge_lengths)
                        longest_edge = max(longest_edge, longest_graph_edge)
            
            if longest_edge == 0.0:
                # Fallback to a reasonable default
                longest_edge = 0.5  # Default radius in radians (~28 degrees)
                warnings.warn(
                    f"No edges with 'len' attribute found when computing rel_max_dist. "
                    f"Using default longest_edge={longest_edge}",
                    UserWarning
                )
            
            query_dist = longest_edge * rel_max_dist
            print(f"query_dist = {query_dist:.4f}  (longest_edge={longest_edge:.4f}, rel={rel_max_dist})")
        else:
            # No distance parameters provided - this should not happen if called correctly
            # But provide a fallback for backward compatibility
            query_dist = 0.5
            warnings.warn(
                f"No max_dist or rel_max_dist provided for within_radius method. "
                f"Using default query_dist={query_dist}",
                UserWarning
            )

        def _find_neighbour_node_idxs_in_source_mesh(query_point):
            if use_3d:
                neigh_idxs = kdt_s.query_ball_point(query_point, query_dist)
                return neigh_idxs
            else:
                # For 2D, query all three copies
                all_neigh_idxs = []
                for qp in query_point:
                    neigh_idxs = kdt_s.query_ball_point(qp, query_dist)
                    all_neigh_idxs.extend(neigh_idxs)
                
                # Map back to original indices, removing duplicates
                original_idxs = set()
                for idx in all_neigh_idxs:
                    if idx < len(source_node_mapping):  # Safety check
                        original_idxs.add(source_node_mapping[idx][2])
                return list(original_idxs)

    elif method == "containing_triangle":
        from weather_model_graphs.create.mesh.layouts.icosahedral import (
            lat_lon_to_cartesian,
            find_containing_triangle,
        )

        if mesh_vertices is None or mesh_faces is None:
            raise ValueError(
                "containing_triangle method requires mesh_vertices and mesh_faces "
                "to be passed to connect_nodes_across_graphs."
            )

        G_connect = networkx.DiGraph()
        G_connect.add_nodes_from(sorted(G_source.nodes(data=True)))
        G_connect.add_nodes_from(sorted(G_target.nodes(data=True)))
        target_sample = target_nodes_list[0]
        use_3d_target = "pos3d" in G_target.nodes[target_sample]
        if use_3d_target and not use_3d:
            # g2m case: source=grid (2D only), target=mesh (has pos3d)
            # Rebuild KDTree on target's 3D positions for querying
            xy_target_3d = np.array([G_target.nodes[n]["pos3d"] for n in source_nodes_list
                                    if n in G_target.nodes])
            # Actually we need source KDTree on source nodes but queried with 3D points.
            # The KDTree was built on source (grid, 2D). Instead rebuild on target mesh in 3D
            # and for each source (grid) node, find neighbours in target (mesh).
            # But connect_nodes_across_graphs goes source->target directionally.
            # The KDTree must be on SOURCE. So convert source grid lat/lon to 3D:
            from weather_model_graphs.create.mesh.layouts.icosahedral import lat_lon_to_cartesian
            xy_source_3d = np.array([
                lat_lon_to_cartesian(
                    np.array([G_source.nodes[n]["pos"][0]]),
                    np.array([G_source.nodes[n]["pos"][1]])
                )[0]
                for n in source_nodes_list
            ])
            kdt_s = scipy.spatial.KDTree(xy_source_3d)

        for target_node in target_nodes_list:
            target_pos_2d = G_target.nodes[target_node]["pos"]

            # Convert target query point to 3D if KDTree was built in 3D
            if use_3d or use_3d_target:
                from weather_model_graphs.create.mesh.layouts.icosahedral import lat_lon_to_cartesian
                query_point = lat_lon_to_cartesian(
                    np.array([target_pos_2d[0]]),
                    np.array([target_pos_2d[1]])
                )[0]
            else:
                query_point = target_pos_2d

            neigh_idxs = _find_neighbour_node_idxs_in_source_mesh(query_point)

            for i in neigh_idxs:
                source_node = source_nodes_list[i]
                source_pos_2d = G_connect.nodes[source_node]["pos"]

                # Distance: use 3D if either side is icosahedral
                if use_3d:
                    source_pos_3d = G_connect.nodes[source_node]["pos3d"]
                    d = np.sqrt(np.sum((source_pos_3d - query_point) ** 2))
                elif use_3d_target:
                    # source is grid, convert to 3D for distance
                    d = np.sqrt(np.sum((query_point - lat_lon_to_cartesian(  # query_point is target in 3D
                        np.array([source_pos_2d[0]]),
                        np.array([source_pos_2d[1]])
                    )[0]) ** 2))
                else:
                    # 2D distance with longitude wrapping
                    dlat = source_pos_2d[0] - target_pos_2d[0]
                    dlon = source_pos_2d[1] - target_pos_2d[1]
                    dlon = (dlon + 180) % 360 - 180
                    d = np.sqrt(dlat**2 + dlon**2)

                G_connect.add_edge(source_node, target_node)
                G_connect.edges[source_node, target_node]["len"] = d
                
                # Use 3D Cartesian vdiff for icosahedral graphs, 2D lat/lon for rectilinear
                if use_3d:
                    source_pos_3d = G_connect.nodes[source_node]["pos3d"]
                    vdiff = source_pos_3d - query_point
                elif use_3d_target:
                    source_pos_3d = lat_lon_to_cartesian(
                        np.array([source_pos_2d[0]]), np.array([source_pos_2d[1]])
                    )[0]
                    vdiff = source_pos_3d - query_point
                else:
                    dlat = source_pos_2d[0] - target_pos_2d[0]
                    dlon = (source_pos_2d[1] - target_pos_2d[1] + 180) % 360 - 180
                    vdiff = np.array([dlat, dlon])
                G_connect.edges[source_node, target_node]["vdiff"] = vdiff

        return G_connect  # early return, skips generic block below

    else:
        raise NotImplementedError(method)

    # Generic edge-building block for all non-early-return methods
    # (nearest_neighbour, nearest_neighbours, within_radius)
    G_connect = networkx.DiGraph()
    G_connect.add_nodes_from(sorted(G_source.nodes(data=True)))
    G_connect.add_nodes_from(sorted(G_target.nodes(data=True)))

    for target_node in target_nodes_list:
        target_pos_2d = G_target.nodes[target_node]["pos"]

        if use_3d:
            from weather_model_graphs.create.mesh.layouts.icosahedral import lat_lon_to_cartesian
            query_point = lat_lon_to_cartesian(
                np.array([target_pos_2d[0]]),
                np.array([target_pos_2d[1]])
            )[0]
            query_points_for_kdt = query_point
        else:
            # For 2D queries with wrapping, create three query points
            query_points_for_kdt = np.array([
                [target_pos_2d[0], target_pos_2d[1] - 360],
                [target_pos_2d[0], target_pos_2d[1]],
                [target_pos_2d[0], target_pos_2d[1] + 360]
            ])

        neigh_idxs = _find_neighbour_node_idxs_in_source_mesh(query_points_for_kdt)

        for i in neigh_idxs:
            source_node = source_nodes_list[i]
            source_pos_2d = G_connect.nodes[source_node]["pos"]

            # Compute distance in 3D if icosahedral, else 2D with wrapping
            if source_has_3d:
                # m2g: source is mesh, pos3d exists natively
                source_pos_3d = G_connect.nodes[source_node]["pos3d"]
                # Convert target to 3D for distance
                target_pos_3d = lat_lon_to_cartesian(
                    np.array([target_pos_2d[0]]),
                    np.array([target_pos_2d[1]])
                )[0]
                d = np.sqrt(np.sum((source_pos_3d - target_pos_3d) ** 2))
            elif target_has_3d:
                # g2m: source is grid, no pos3d stored — project on the fly
                from weather_model_graphs.create.mesh.layouts.icosahedral import lat_lon_to_cartesian
                source_pos_3d = lat_lon_to_cartesian(
                    np.array([source_pos_2d[0]]),
                    np.array([source_pos_2d[1]])
                )[0]
                target_pos_3d = G_connect.nodes[target_node]["pos3d"]
                d = np.sqrt(np.sum((source_pos_3d - target_pos_3d) ** 2))
            else:
                # 2D distance with longitude wrapping
                dlat = source_pos_2d[0] - target_pos_2d[0]
                dlon = source_pos_2d[1] - target_pos_2d[1]
                dlon = (dlon + 180) % 360 - 180
                d = np.sqrt(dlat**2 + dlon**2)

            G_connect.add_edge(source_node, target_node)
            G_connect.edges[source_node, target_node]["len"] = d
            
            # Use 3D Cartesian vdiff for icosahedral graphs, 2D lat/lon for rectilinear
            if source_has_3d:
                source_pos_3d = G_connect.nodes[source_node]["pos3d"]
                target_pos_3d = lat_lon_to_cartesian(
                    np.array([target_pos_2d[0]]), np.array([target_pos_2d[1]])
                )[0]
                vdiff = source_pos_3d - target_pos_3d
            elif target_has_3d:
                source_pos_3d = lat_lon_to_cartesian(
                    np.array([source_pos_2d[0]]), np.array([source_pos_2d[1]])
                )[0]
                target_pos_3d = G_connect.nodes[target_node]["pos3d"]
                vdiff = source_pos_3d - target_pos_3d
            else:
                dlat = source_pos_2d[0] - target_pos_2d[0]
                dlon = (source_pos_2d[1] - target_pos_2d[1] + 180) % 360 - 180
                vdiff = np.array([dlat, dlon])
            G_connect.edges[source_node, target_node]["vdiff"] = vdiff

    return G_connect