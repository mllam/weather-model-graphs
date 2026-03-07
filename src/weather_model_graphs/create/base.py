"""
Generic routines for creating the graph components used in the message-passing
graph, and for connecting nodes across these component graphs.

The graph components, grid-to-mesh (g2m), mesh-to-mesh (m2m) and mesh-to-grid (m2g), are
used to represent the encode-process-decode steps respectively. These are created with
`create_all_graph_components` which takes the following arguments. Internally, this
function uses `connect_nodes_across_graphs` to connect nodes across the component graphs.
"""

import warnings
from typing import Iterable

import networkx
import numpy as np
import pyproj
import scipy.spatial
from loguru import logger

from ..networkx_utils import (
    replace_node_labels_with_unique_ids,
    split_graph_by_edge_attribute,
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
    mesh_layout_kwargs: dict = {},
    m2m_connectivity_kwargs: dict = {},
    m2g_connectivity_kwargs: dict = {},
    g2m_connectivity_kwargs: dict = {},
    coords_crs: pyproj.crs.CRS | None = None,
    graph_crs: pyproj.crs.CRS | None = None,
    decode_mask: Iterable[bool] | None = None,
    return_components: bool = False,
    mesh_layout: str = "rectilinear",
):
    """
    Create all graph components used in creating the message-passing graph,
        grid-to-mesh (g2m), mesh-to-mesh (m2m) and mesh-to-grid (m2g),
    representing the encode-process-decode respectively.

    Parameters
    ----------
    coords : np.ndarray
        Grid point coordinates, shape [num_grid_nodes, 2]
    mesh_layout : str, default="rectilinear"
        Type of mesh to create. Options:
        - "rectilinear": Regular 2D grid mesh
        - "icosahedral": Spherical icosahedral mesh for global domains
    m2m_connectivity : str
        Method for mesh-to-mesh connections
    m2g_connectivity : str
        Method for mesh-to-grid connections
    g2m_connectivity : str
        Method for grid-to-mesh connections
    mesh_layout_kwargs : dict, optional
        Keyword arguments for mesh layout creation.
        For "icosahedral" layout:
        - subdivisions : int (default=3) - Refinement level for flat mesh
        - max_subdivisions : int (default=3) - Max refinement for hierarchical
        - hierarchical : bool (default=False) - Use hierarchical mesh
        - radius : float (default=1.0) - Sphere radius
        - grid_spacing : float, optional - Desired grid spacing in degrees
          (alternative to subdivisions/max_subdivisions)
    m2m_connectivity_kwargs : dict, optional
        Keyword arguments for mesh-to-mesh connectivity method
    m2g_connectivity_kwargs : dict, optional
        Keyword arguments for mesh-to-grid connectivity method
    g2m_connectivity_kwargs : dict, optional
        Keyword arguments for grid-to-mesh connectivity method
    coords_crs : pyproj.crs.CRS, optional
        Coordinate reference system of input coordinates
    graph_crs : pyproj.crs.CRS, optional
        Coordinate reference system for graph construction
    decode_mask : Iterable[bool], optional
        Mask for which grid nodes to include in m2g (decode step)
    return_components : bool, default=False
        If True, return dict of component graphs instead of merged graph

    Returns
    -------
    networkx.DiGraph or dict
        Either merged graph or dict of component graphs

    Notes
    -----
    Available connectivity methods:

    g2m_connectivity:
    - "nearest_neighbour": Find nearest neighbour in grid for each mesh node
    - "nearest_neighbours": Find k nearest neighbours in grid for each mesh node
    - "within_radius": Find all grid nodes within radius of each mesh node

    m2m_connectivity (for mesh_layout="rectilinear"):
    - "flat": Single-level 2D mesh graph
    - "flat_multiscale": Flat multiscale mesh with multiple levels
    - "hierarchical": Hierarchical mesh with up/down connections

    m2m_connectivity (for mesh_layout="icosahedral"):
    - "flat": Single-level icosahedral mesh (ignored, mesh determines connectivity)
    - "hierarchical": Multi-level icosahedral mesh with inter-level connections

    m2g_connectivity:
    - "nearest_neighbour": Find nearest mesh node for each grid node
    - "nearest_neighbours": Find k nearest mesh nodes for each grid node
    - "within_radius": Find all mesh nodes within radius of each grid node
    - "containing_rectangle": Find containing rectangle in rectilinear mesh
    - "containing_triangle": Find containing triangle in icosahedral mesh
      (requires mesh_layout="icosahedral")
    """
    graph_components: dict[networkx.DiGraph] = {}

    assert (
        len(coords.shape) == 2 and coords.shape[1] == 2
    ), "Grid node coordinates should be given as an array of shape [num_grid_nodes, 2]."

    if coords_crs is None and graph_crs is None:
        logger.debug(
            "No `coords_crs` given: Assuming `coords` contains in-projection Cartesian coordinates."
        )
        xy = coords
    elif (coords_crs is None) != (graph_crs is None):
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
        coord_transformer = pyproj.Transformer.from_crs(
            coords_crs, graph_crs, always_xy=True
        )
        xy_tuple = coord_transformer.transform(xx=coords[:, 0], yy=coords[:, 1])
        xy = np.stack(xy_tuple, axis=1)

    # Early validation: containing_triangle requires icosahedral layout
    if m2g_connectivity == "containing_triangle" and mesh_layout != "icosahedral":
        raise ValueError(
            f"containing_triangle method is only valid for mesh_layout='icosahedral'. "
            f"Got mesh_layout='{mesh_layout}'"
        )

    if mesh_layout == "rectilinear":
        if m2m_connectivity == "flat":
            graph_components["m2m"] = create_flat_singlescale_mesh_graph(
                xy,
                **m2m_connectivity_kwargs,
            )
            grid_connect_graph = graph_components["m2m"]
        elif m2m_connectivity == "hierarchical":
            graph_components["m2m"] = create_hierarchical_multiscale_mesh_graph(
                xy=xy,
                **m2m_connectivity_kwargs,
            )
            grid_connect_graph = split_graph_by_edge_attribute(
                graph_components["m2m"], "level"
            )[0]
        elif m2m_connectivity == "flat_multiscale":
            graph_components["m2m"] = create_flat_multiscale_mesh_graph(
                xy=xy,
                **m2m_connectivity_kwargs,
            )
            grid_connect_graph = graph_components["m2m"]
        else:
            raise ValueError(
                f"Unknown m2m_connectivity '{m2m_connectivity}' for mesh_layout='rectilinear'"
            )

    elif mesh_layout == "icosahedral":
        from weather_model_graphs.create.mesh.layouts.icosahedral import (
            create_flat_icosahedral_mesh_graph,
            create_hierarchical_icosahedral_mesh_graph,
            generate_icosahedral_mesh,
            refinement_level_from_grid_spacing,
        )

        def _is_geographic_crs(crs):
            """Check if a CRS is geographic. Returns True if geographic or unknown."""
            if crs is None:
                return True

            # Normalise to a pyproj CRS object so .is_geographic is a reliable scalar.
            try:
                parsed = pyproj.CRS.from_user_input(crs)
                val = parsed.is_geographic
                # Guard against unusual pyproj builds that return array-like values
                if hasattr(val, "__iter__"):
                    import numpy as _np

                    return bool(int(_np.asarray(val).flat[0]))
                return bool(int(val))
            except Exception:
                pass

            # String heuristic fallback
            try:
                crs_str = str(crs).upper()
                if any(
                    k in crs_str for k in ("4326", "WGS84", "GEOGRAPHIC", "LATLONG")
                ):
                    return True
                return False
            except Exception:
                return True  # Can't determine → assume geographic, suppress warning

        if graph_crs is not None:
            is_geographic = _is_geographic_crs(graph_crs)
            if not is_geographic:
                warnings.warn(
                    "Icosahedral mesh is designed for geographic coordinates. "
                    "Using with non-geographic CRS may produce unexpected results.",
                    UserWarning,
                    stacklevel=2,
                )

        radius = mesh_layout_kwargs.get("radius", 1.0)
        grid_spacing = mesh_layout_kwargs.get("grid_spacing")
        hierarchical = mesh_layout_kwargs.get("hierarchical", False)

        if grid_spacing is not None:
            if (
                "subdivisions" in mesh_layout_kwargs
                or "max_subdivisions" in mesh_layout_kwargs
            ):
                raise ValueError(
                    "Cannot specify both grid_spacing and subdivisions/max_subdivisions. "
                    "Choose one method."
                )
            refinement_level = refinement_level_from_grid_spacing(grid_spacing, radius)
            if hierarchical:
                mesh_layout_kwargs["max_subdivisions"] = refinement_level
                logger.debug(
                    f"grid_spacing={grid_spacing}° mapped to max_subdivisions={refinement_level}"
                )
            else:
                mesh_layout_kwargs["subdivisions"] = refinement_level
                logger.debug(
                    f"grid_spacing={grid_spacing}° mapped to subdivisions={refinement_level}"
                )

        if hierarchical:
            max_subdivisions = mesh_layout_kwargs.get("max_subdivisions", 3)
            graph_components["m2m"] = create_hierarchical_icosahedral_mesh_graph(
                max_subdivisions=max_subdivisions,
                radius=radius,
            )
            grid_connect_graph = split_graph_by_edge_attribute(
                graph_components["m2m"], "level"
            )[0]
            finest_vertices, finest_faces = generate_icosahedral_mesh(
                refinement_level=max_subdivisions,
                radius=radius,
            )
        else:
            subdivisions = mesh_layout_kwargs.get("subdivisions", 3)
            graph_components["m2m"] = create_flat_icosahedral_mesh_graph(
                subdivisions=subdivisions,
                radius=radius,
            )
            grid_connect_graph = graph_components["m2m"]
            finest_vertices, finest_faces = generate_icosahedral_mesh(
                refinement_level=subdivisions,
                radius=radius,
            )

        grid_connect_graph.graph["mesh_vertices"] = finest_vertices
        grid_connect_graph.graph["mesh_faces"] = finest_faces
        graph_components["m2m"].graph["mesh_vertices"] = finest_vertices
        graph_components["m2m"].graph["mesh_faces"] = finest_faces

        if graph_crs is not None:
            graph_components["m2m"].graph["crs"] = graph_crs
            grid_connect_graph.graph["crs"] = graph_crs

    else:
        raise ValueError(
            f"Unknown mesh_layout '{mesh_layout}'. "
            "Supported: 'rectilinear', 'icosahedral'"
        )

    G_grid = create_grid_graph_nodes(xy=xy)
    if graph_crs is not None:
        G_grid.graph["crs"] = graph_crs

    G_g2m = connect_nodes_across_graphs(
        G_source=G_grid,
        G_target=grid_connect_graph,
        method=g2m_connectivity,
        **g2m_connectivity_kwargs,
    )
    graph_components["g2m"] = G_g2m

    if decode_mask is None:
        decode_grid = G_grid
    else:
        filter_nodes = [
            n for n, include in zip(G_grid.nodes, decode_mask, strict=True) if include
        ]
        decode_grid = G_grid.subgraph(filter_nodes)

    if m2g_connectivity == "containing_triangle":
        if mesh_layout != "icosahedral":
            raise ValueError(
                f"containing_triangle method is only valid for mesh_layout='icosahedral'. "
                f"Got mesh_layout='{mesh_layout}'"
            )
        G_m2g = connect_nodes_across_graphs(
            G_source=grid_connect_graph,
            G_target=decode_grid,
            method=m2g_connectivity,
            mesh_vertices=grid_connect_graph.graph.get("mesh_vertices"),
            mesh_faces=grid_connect_graph.graph.get("mesh_faces"),
            **m2g_connectivity_kwargs,
        )
    elif m2g_connectivity == "containing_rectangle":
        if mesh_layout != "rectilinear":
            raise ValueError(
                f"containing_rectangle method is only valid for mesh_layout='rectilinear'. "
                f"Got mesh_layout='{mesh_layout}'"
            )
        G_m2g = connect_nodes_across_graphs(
            G_source=grid_connect_graph,
            G_target=decode_grid,
            method=m2g_connectivity,
            **m2g_connectivity_kwargs,
        )
    else:
        G_m2g = connect_nodes_across_graphs(
            G_source=grid_connect_graph,
            G_target=decode_grid,
            method=m2g_connectivity,
            **m2g_connectivity_kwargs,
        )
    graph_components["m2g"] = G_m2g

    for name, graph in graph_components.items():
        for edge in graph.edges:
            graph.edges[edge]["component"] = name

    if return_components:
        graph_components = {
            comp_name: replace_node_labels_with_unique_ids(subgraph)
            for comp_name, subgraph in graph_components.items()
        }
        return graph_components

    G_tot = networkx.compose_all(graph_components.values())

    def _graph_attr_equal(a, b):
        """Safe equality check for graph attributes that may be numpy arrays."""
        if a is b:
            return True
        if a is None or b is None:
            return a is b
        try:
            import numpy as _np

            if isinstance(a, _np.ndarray) or isinstance(b, _np.ndarray):
                a_arr, b_arr = _np.asarray(a), _np.asarray(b)
                return a_arr.shape == b_arr.shape and bool(
                    _np.array_equal(a_arr, b_arr)
                )
        except Exception:
            pass
        try:
            result = a == b
            # If result is array-like, use array_equal logic
            if hasattr(result, "__len__"):
                return False  # Different array values → not equal
            return bool(result)
        except Exception:
            return False

    for key in graph_components["m2m"].graph.keys():
        ref_val = graph_components["m2m"].graph[key]
        if not all(
            _graph_attr_equal(graph.graph.get(key, None), ref_val)
            for graph in graph_components.values()
        ):
            if key in G_tot.graph:
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
    **kwargs,
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

    if source_has_3d:
        xy_source = np.array([G_source.nodes[n]["pos3d"] for n in source_nodes_list])
        use_3d = True
    elif target_has_3d:
        from weather_model_graphs.create.mesh.layouts.icosahedral import (
            lat_lon_to_cartesian,
        )

        source_lats = np.array([G_source.nodes[n]["pos"][0] for n in source_nodes_list])
        source_lons = np.array([G_source.nodes[n]["pos"][1] for n in source_nodes_list])
        xy_source = lat_lon_to_cartesian(source_lats, source_lons)
        use_3d = True
    else:
        source_lat_lon = np.array([G_source.nodes[n]["pos"] for n in source_nodes_list])
        source_positions = []
        for offset in [-360, 0, 360]:
            shifted = source_lat_lon.copy()
            shifted[:, 1] += offset
            source_positions.append(shifted)
        xy_source = np.vstack(source_positions)
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
                all_neigh_idxs = []
                for qp in query_point:
                    neigh_idx = kdt_s.query(qp, 1)[1]
                    all_neigh_idxs.append(neigh_idx)
                original_idxs = set()
                for idx in all_neigh_idxs:
                    original_idxs.add(source_node_mapping[idx][2])
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
                all_neigh_idxs = []
                for qp in query_point:
                    neigh_idxs = kdt_s.query(qp, max_num_neighbours)[1]
                    all_neigh_idxs.extend(neigh_idxs)
                original_idxs = set()
                for idx in all_neigh_idxs:
                    original_idxs.add(source_node_mapping[idx][2])
                return list(original_idxs)

    elif method == "within_radius":
        if max_num_neighbours is not None:
            raise Exception(
                "to use `within_radius` method you should not set `max_num_neighbours`"
            )

        if max_dist is not None:
            query_dist = max_dist
        elif rel_max_dist is not None:
            longest_edge = 0.0
            for edge_check_graph in (G_source, G_target):
                if len(edge_check_graph.edges) > 0:
                    edge_lengths = []
                    for _, _, data in edge_check_graph.edges(data=True):
                        if "len" in data:
                            edge_lengths.append(data["len"])
                    if edge_lengths:
                        longest_graph_edge = max(edge_lengths)
                        longest_edge = max(longest_edge, longest_graph_edge)
            if longest_edge == 0.0:
                longest_edge = 0.5
                warnings.warn(
                    f"No edges with 'len' attribute found when computing rel_max_dist. "
                    f"Using default longest_edge={longest_edge}",
                    UserWarning,
                )
            query_dist = longest_edge * rel_max_dist
            print(
                f"query_dist = {query_dist:.4f}  (longest_edge={longest_edge:.4f}, rel={rel_max_dist})"
            )
        else:
            query_dist = 0.5
            warnings.warn(
                f"No max_dist or rel_max_dist provided for within_radius method. "
                f"Using default query_dist={query_dist}",
                UserWarning,
            )

        def _find_neighbour_node_idxs_in_source_mesh(query_point):
            if use_3d:
                neigh_idxs = kdt_s.query_ball_point(query_point, query_dist)
                return neigh_idxs
            else:
                all_neigh_idxs = []
                for qp in query_point:
                    neigh_idxs = kdt_s.query_ball_point(qp, query_dist)
                    all_neigh_idxs.extend(neigh_idxs)
                original_idxs = set()
                for idx in all_neigh_idxs:
                    if idx < len(source_node_mapping):
                        original_idxs.add(source_node_mapping[idx][2])
                return list(original_idxs)

    elif method == "containing_triangle":
        from weather_model_graphs.create.mesh.layouts.icosahedral import (
            connect_mesh_to_grid,
            lat_lon_to_cartesian,
        )

        if mesh_vertices is None or mesh_faces is None:
            raise ValueError(
                "containing_triangle method requires mesh_vertices and mesh_faces "
                "to be passed to connect_nodes_across_graphs."
            )

        grid_lat_lon = np.array([G_target.nodes[n]["pos"] for n in target_nodes_list])
        fallback_to_nearest = kwargs.get("fallback_to_nearest", True)

        edge_index, weights = connect_mesh_to_grid(
            mesh_vertices=mesh_vertices,
            mesh_faces=mesh_faces,
            grid_lat_lon=grid_lat_lon,
            fallback_to_nearest=fallback_to_nearest,
        )

        if edge_index.shape[1] == 0:
            warnings.warn(
                "No triangle containment connections found. Grid points may be outside mesh domain.",
                UserWarning,
            )
            G_connect = networkx.DiGraph()
            G_connect.add_nodes_from(sorted(G_source.nodes(data=True)))
            G_connect.add_nodes_from(sorted(G_target.nodes(data=True)))
            return G_connect

        G_connect = networkx.DiGraph()
        G_connect.add_nodes_from(sorted(G_source.nodes(data=True)))
        G_connect.add_nodes_from(sorted(G_target.nodes(data=True)))

        grid_points_with_fallback = set()

        for col in range(edge_index.shape[1]):
            mesh_idx = edge_index[0, col]
            grid_idx = edge_index[1, col]
            weight = weights[col]

            # Skip zero-weight edges: a triangle vertex with w=0 contributes nothing
            # to interpolation and would fail the barycentric_weight > 0 invariant.
            if weight <= 0.0:
                continue

            source_node = source_nodes_list[mesh_idx]
            target_node = target_nodes_list[grid_idx]

            if abs(weight - 1.0) < 1e-10:
                grid_points_with_fallback.add(grid_idx)

            source_pos_2d = G_connect.nodes[source_node]["pos"]
            target_pos_2d = G_connect.nodes[target_node]["pos"]

            if "pos3d" in G_connect.nodes[source_node]:
                source_pos_3d = G_connect.nodes[source_node]["pos3d"]
                target_pos_3d = lat_lon_to_cartesian(
                    np.array([target_pos_2d[0]]), np.array([target_pos_2d[1]])
                )[0]
                d = np.sqrt(np.sum((source_pos_3d - target_pos_3d) ** 2))
                vdiff = source_pos_3d - target_pos_3d
            elif "pos3d" in G_connect.nodes[target_node]:
                source_pos_3d = lat_lon_to_cartesian(
                    np.array([source_pos_2d[0]]), np.array([source_pos_2d[1]])
                )[0]
                target_pos_3d = G_connect.nodes[target_node]["pos3d"]
                d = np.sqrt(np.sum((source_pos_3d - target_pos_3d) ** 2))
                vdiff = source_pos_3d - target_pos_3d
            else:
                dlat = source_pos_2d[0] - target_pos_2d[0]
                dlon = source_pos_2d[1] - target_pos_2d[1]
                dlon = (dlon + 180) % 360 - 180
                d = np.sqrt(dlat**2 + dlon**2)
                vdiff = np.array([dlat, dlon])

            if G_connect.has_edge(source_node, target_node):
                # Duplicate edge (same mesh vertex connected to same grid point by
                # two different triangles) — accumulate barycentric weight.
                G_connect.edges[source_node, target_node][
                    "barycentric_weight"
                ] += weight
            else:
                G_connect.add_edge(source_node, target_node)
                G_connect.edges[source_node, target_node].update(
                    {
                        "len": d,
                        "vdiff": vdiff,
                        "barycentric_weight": weight,
                        "component": "m2g",
                    }
                )

        num_fallback_points = len(grid_points_with_fallback)
        if num_fallback_points > 0:
            total_grid_points = len(target_nodes_list)
            warnings.warn(
                f"Triangle containment failed for {num_fallback_points}/{total_grid_points} "
                f"({num_fallback_points/total_grid_points*100:.1f}%) grid points. "
                f"Used nearest neighbour fallback.",
                UserWarning,
            )

        G_connect.graph["mesh_vertices"] = mesh_vertices
        G_connect.graph["mesh_faces"] = mesh_faces
        return G_connect

    else:
        raise NotImplementedError(method)

    # Generic edge-building block
    G_connect = networkx.DiGraph()
    G_connect.add_nodes_from(sorted(G_source.nodes(data=True)))
    G_connect.add_nodes_from(sorted(G_target.nodes(data=True)))

    for target_node in target_nodes_list:
        target_pos_2d = G_target.nodes[target_node]["pos"]

        if use_3d:
            from weather_model_graphs.create.mesh.layouts.icosahedral import (
                lat_lon_to_cartesian,
            )

            query_point = lat_lon_to_cartesian(
                np.array([target_pos_2d[0]]), np.array([target_pos_2d[1]])
            )[0]
            query_points_for_kdt = query_point
        else:
            query_points_for_kdt = np.array(
                [
                    [target_pos_2d[0], target_pos_2d[1] - 360],
                    [target_pos_2d[0], target_pos_2d[1]],
                    [target_pos_2d[0], target_pos_2d[1] + 360],
                ]
            )

        neigh_idxs = _find_neighbour_node_idxs_in_source_mesh(query_points_for_kdt)

        for i in neigh_idxs:
            source_node = source_nodes_list[i]
            source_pos_2d = G_connect.nodes[source_node]["pos"]

            if source_has_3d:
                source_pos_3d = G_connect.nodes[source_node]["pos3d"]
                target_pos_3d = lat_lon_to_cartesian(
                    np.array([target_pos_2d[0]]), np.array([target_pos_2d[1]])
                )[0]
                d = np.sqrt(np.sum((source_pos_3d - target_pos_3d) ** 2))
            elif target_has_3d:
                from weather_model_graphs.create.mesh.layouts.icosahedral import (
                    lat_lon_to_cartesian,
                )

                source_pos_3d = lat_lon_to_cartesian(
                    np.array([source_pos_2d[0]]), np.array([source_pos_2d[1]])
                )[0]
                target_pos_3d = G_connect.nodes[target_node]["pos3d"]
                d = np.sqrt(np.sum((source_pos_3d - target_pos_3d) ** 2))
            else:
                dlat = source_pos_2d[0] - target_pos_2d[0]
                dlon = source_pos_2d[1] - target_pos_2d[1]
                dlon = (dlon + 180) % 360 - 180
                d = np.sqrt(dlat**2 + dlon**2)

            G_connect.add_edge(source_node, target_node)
            G_connect.edges[source_node, target_node]["len"] = d

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
