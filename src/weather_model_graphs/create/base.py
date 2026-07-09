"""
Generic routines for creating the graph components used in the message-passing
graph, and for connecting nodes across these component graphs.

The graph components, grid-to-mesh (g2m), mesh-to-mesh (m2m) and mesh-to-grid (m2g), are
used to represent the encode-process-decode steps respectively. These are created with
`create_all_graph_components` which takes the following arguments. Internally, this
function uses `connect_nodes_across_graphs` to connect nodes across the component graphs.
"""

import warnings
from typing import Dict, Iterable, List, Tuple, Union

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
from .mesh.connectivity.flat import (
    create_flat_multiscale_from_coordinates,
    create_flat_singlescale_from_coordinates,
)
from .mesh.connectivity.hierarchical import create_hierarchical_from_coordinates
from .mesh.coords import (
    create_multirange_2d_mesh_primitives,
    create_single_level_2d_mesh_primitive,
)


def _migrate_deprecated_kwargs(
    mesh_layout_kwargs, m2m_connectivity_kwargs
) -> Tuple[dict, dict]:
    """Migrate old-style kwargs to the new mesh_layout_kwargs structure.

    In the old API, ``mesh_node_distance``, ``level_refinement_factor``, and
    ``max_num_levels`` were passed via ``m2m_connectivity_kwargs``. In the new
    design these belong in ``mesh_layout_kwargs`` (as ``mesh_node_spacing``,
    ``refinement_factor``, and ``max_num_refinement_levels`` respectively).

    This helper emits deprecation warnings for each migrated key and moves
    the value into *mesh_layout_kwargs*. It is intended to be removed once the
    old API is no longer supported.

    Parameters
    ----------
    mesh_layout_kwargs : dict
        Mutable dict of mesh layout keyword arguments.
    m2m_connectivity_kwargs : dict
        Mutable dict of m2m connectivity keyword arguments.

    Returns
    -------
    tuple[dict, dict]
        Updated (mesh_layout_kwargs, m2m_connectivity_kwargs).
    """
    if (
        "mesh_node_distance" in m2m_connectivity_kwargs
        and "mesh_node_spacing" not in mesh_layout_kwargs
    ):
        logger.warning(
            "Passing 'mesh_node_distance' in m2m_connectivity_kwargs is deprecated. "
            "Use mesh_layout_kwargs=dict(mesh_node_spacing=...) instead."
        )
        mesh_layout_kwargs["mesh_node_spacing"] = m2m_connectivity_kwargs.pop(
            "mesh_node_distance"
        )
    if (
        "level_refinement_factor" in m2m_connectivity_kwargs
        and "refinement_factor" not in mesh_layout_kwargs
    ):
        logger.warning(
            "Passing 'level_refinement_factor' in m2m_connectivity_kwargs is deprecated. "
            "Use mesh_layout_kwargs=dict(refinement_factor=...) instead."
        )
        mesh_layout_kwargs["refinement_factor"] = m2m_connectivity_kwargs.pop(
            "level_refinement_factor"
        )
    if (
        "max_num_levels" in m2m_connectivity_kwargs
        and "max_num_refinement_levels" not in mesh_layout_kwargs
    ):
        logger.warning(
            "Passing 'max_num_levels' in m2m_connectivity_kwargs is deprecated. "
            "Use mesh_layout_kwargs=dict(max_num_refinement_levels=...) instead."
        )
        mesh_layout_kwargs["max_num_refinement_levels"] = m2m_connectivity_kwargs.pop(
            "max_num_levels"
        )
    return mesh_layout_kwargs, m2m_connectivity_kwargs


def create_all_graph_components(
    coords: np.ndarray,
    m2m_connectivity: str,
    m2g_connectivity: str,
    g2m_connectivity: str,
    mesh_layout: str,
    mesh_layout_kwargs: dict = None,
    m2m_connectivity_kwargs: dict = None,
    m2g_connectivity_kwargs: dict = None,
    g2m_connectivity_kwargs: dict = None,
    coords_crs: pyproj.crs.CRS | None = None,
    graph_crs: pyproj.crs.CRS | None = None,
    decode_mask: Iterable[bool] | None = None,
    return_components: bool = False,
) -> Union[networkx.DiGraph, Dict[str, networkx.DiGraph]]:
    """
    Create all graph components used in creating the message-passing graph,
        grid-to-mesh (g2m), mesh-to-mesh (m2m) and mesh-to-grid (m2g),
    representing the encode-process-decode respectively.

    The mesh graph creation follows a two-step process:
    1. **Coordinate creation** (controlled by `mesh_layout` + `mesh_layout_kwargs`):
       Creates an undirected graph (nx.Graph) with node positions and spatial
       adjacency edges annotated with adjacency types.
    2. **Connectivity creation** (controlled by `m2m_connectivity` + `m2m_connectivity_kwargs`):
       Converts the coordinate graph to directed connectivity (nx.DiGraph)
       based on the specified pattern and connectivity method.

    For each graph component, the method for connecting nodes across graphs
    should be specified (with the `*_connectivity` arguments, e.g. `m2g_connectivity`).
    And the method-specific arguments should be passed as keyword arguments using
    the `*_connectivity_kwargs` arguments (e.g. `m2g_connectivity_kwargs`).

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

    mesh_layout:
    - "rectilinear": Uniform regular grid with ``mesh_node_spacing`` resolution.
      Produces an undirected mesh primitive with 4-star (cardinal) and
      8-star (cardinal + diagonal) spatial adjacency edges.
    - "icosahedral": Spherical icosahedral mesh for global domains.
      mesh_layout_kwargs: subdivisions, max_subdivisions, hierarchical, radius,
      grid_spacing.

    mesh_layout_kwargs (for mesh_layout="rectilinear"):
    - mesh_node_spacing: float, distance between mesh nodes in coordinate units.
    - refinement_factor: int, refinement factor between levels
      (for multi-level and hierarchical mesh graphs, default: 3)
    - max_num_refinement_levels: int, maximum number of mesh levels
      (for multi-level and hierarchical mesh graphs)

    Wherever the ``pattern`` argument appears below it defines the spatial
    neighbourhood connectivity:
    - ``"4-star"``: only cardinal directions (horizontal and vertical neighbours)
    - ``"8-star"``: cardinal plus diagonal neighbours (all 8 surrounding nodes)

    m2m_connectivity:
    - "flat": Create a single-level directed mesh graph.
      m2m_connectivity_kwargs: pattern (default: "8-star")
    - "flat_multiscale": Create a flat multiscale mesh graph.
      m2m_connectivity_kwargs: pattern (default: "8-star")
    - "hierarchical": Create a hierarchical mesh graph with up/down connections.
      m2m_connectivity_kwargs: intra_level=dict(pattern=...), inter_level=dict(pattern=..., k=...)

    m2g_connectivity:
    - "nearest_neighbour": Find nearest mesh node for each grid node
    - "nearest_neighbours": Find k nearest mesh nodes for each grid node
    - "within_radius": Find all mesh nodes within radius of each grid node
    - "containing_rectangle": Find containing rectangle in rectilinear mesh
    - "containing_triangle": Find containing triangle in icosahedral mesh
      (requires mesh_layout="icosahedral")
    """
    graph_components: dict[networkx.DiGraph] = {}

    # Initialize mutable default arguments (and copy to avoid mutating caller's dicts)
    if mesh_layout_kwargs is None:
        mesh_layout_kwargs = {}
    else:
        mesh_layout_kwargs = dict(mesh_layout_kwargs)
    if m2m_connectivity_kwargs is None:
        m2m_connectivity_kwargs = {}
    else:
        m2m_connectivity_kwargs = dict(m2m_connectivity_kwargs)
    if m2g_connectivity_kwargs is None:
        m2g_connectivity_kwargs = {}
    else:
        m2g_connectivity_kwargs = dict(m2g_connectivity_kwargs)
    if g2m_connectivity_kwargs is None:
        g2m_connectivity_kwargs = {}
    else:
        g2m_connectivity_kwargs = dict(g2m_connectivity_kwargs)

    # Migrate deprecated kwargs (to be removed in a future version)
    mesh_layout_kwargs, m2m_connectivity_kwargs = _migrate_deprecated_kwargs(
        mesh_layout_kwargs, m2m_connectivity_kwargs
    )

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

    # containing_triangle is only defined for the icosahedral mesh; validate early
    # so the error is clear before any mesh construction is attempted
    if m2g_connectivity == "containing_triangle" and mesh_layout != "icosahedral":
        raise ValueError(
            f"containing_triangle method is only valid for mesh_layout='icosahedral'. "
            f"Got mesh_layout='{mesh_layout}'"
        )

    # Validate m2m_connectivity early so that we raise a clear NotImplementedError
    # before any coordinate creation is attempted
    _supported_m2m_connectivity = {"flat", "hierarchical", "flat_multiscale"}
    if m2m_connectivity not in _supported_m2m_connectivity:
        raise NotImplementedError(
            f"Kind {m2m_connectivity} not implemented. "
            f"Supported: {sorted(_supported_m2m_connectivity)}"
        )

    if mesh_layout == "rectilinear":
        # Step 1: coordinate creation — produces the mesh primitive graph(s)
        mesh_node_spacing = mesh_layout_kwargs.get(
            "mesh_node_spacing"
        ) or mesh_layout_kwargs.get("grid_spacing")
        if mesh_node_spacing is None:
            raise ValueError(
                "mesh_layout='rectilinear' requires 'mesh_node_spacing' in "
                "mesh_layout_kwargs (or 'mesh_node_distance' in "
                "m2m_connectivity_kwargs for backward compatibility)."
            )

        if m2m_connectivity == "flat":
            G_mesh_coords = create_single_level_2d_mesh_primitive(
                xy, mesh_node_spacing=mesh_node_spacing
            )
        else:
            primitives_kwargs = dict(xy=xy, mesh_node_spacing=mesh_node_spacing)
            if "refinement_factor" in mesh_layout_kwargs:
                primitives_kwargs["interlevel_refinement_factor"] = mesh_layout_kwargs[
                    "refinement_factor"
                ]
            if "max_num_refinement_levels" in mesh_layout_kwargs:
                primitives_kwargs["max_num_levels"] = mesh_layout_kwargs[
                    "max_num_refinement_levels"
                ]
            G_mesh_coords = create_multirange_2d_mesh_primitives(**primitives_kwargs)

        # Step 2: connectivity creation — mesh primitives to directed graph
        if m2m_connectivity == "flat":
            graph_components["m2m"] = create_flat_singlescale_from_coordinates(
                G_mesh_coords, **m2m_connectivity_kwargs
            )
            grid_connect_graph = graph_components["m2m"]
        elif m2m_connectivity == "hierarchical":
            graph_components["m2m"] = create_hierarchical_from_coordinates(
                G_mesh_coords, **m2m_connectivity_kwargs
            )
            # Only connect grid to bottom level of hierarchy
            grid_connect_graph = split_graph_by_edge_attribute(
                graph_components["m2m"], "level"
            )[0]
        elif m2m_connectivity == "flat_multiscale":
            graph_components["m2m"] = create_flat_multiscale_from_coordinates(
                G_mesh_coords, **m2m_connectivity_kwargs
            )
            grid_connect_graph = graph_components["m2m"]

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
        raise NotImplementedError(
            f"mesh_layout='{mesh_layout}' is not yet supported. "
            "Supported: 'rectilinear', 'icosahedral'."
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
) -> networkx.DiGraph:
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

    source_is_icosahedral = G_source.graph.get("mesh_layout", "").startswith(
        "icosahedral"
    )
    target_is_icosahedral = G_target.graph.get("mesh_layout", "").startswith(
        "icosahedral"
    )
    use_spherical = source_is_icosahedral or target_is_icosahedral

    if use_spherical:
        from weather_model_graphs.create.mesh.layouts.icosahedral import (
            lat_lon_to_cartesian,
            tangential_plane_vdiff,
        )

    if source_is_icosahedral:
        source_lats = np.array([G_source.nodes[n]["pos"][0] for n in source_nodes_list])
        source_lons = np.array([G_source.nodes[n]["pos"][1] for n in source_nodes_list])
        xy_source = lat_lon_to_cartesian(source_lats, source_lons)
        use_3d = True
    elif target_is_icosahedral:
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

            if source_is_icosahedral:
                source_pos_3d = lat_lon_to_cartesian(
                    np.array([source_pos_2d[0]]), np.array([source_pos_2d[1]])
                )[0]
                target_pos_3d = lat_lon_to_cartesian(
                    np.array([target_pos_2d[0]]), np.array([target_pos_2d[1]])
                )[0]
                d = np.sqrt(np.sum((source_pos_3d - target_pos_3d) ** 2))
                vdiff = tangential_plane_vdiff(source_pos_3d, target_pos_3d)
            elif target_is_icosahedral:
                source_pos_3d = lat_lon_to_cartesian(
                    np.array([source_pos_2d[0]]), np.array([source_pos_2d[1]])
                )[0]
                target_pos_3d = lat_lon_to_cartesian(
                    np.array([target_pos_2d[0]]), np.array([target_pos_2d[1]])
                )[0]
                d = np.sqrt(np.sum((source_pos_3d - target_pos_3d) ** 2))
                vdiff = tangential_plane_vdiff(source_pos_3d, target_pos_3d)
            else:
                dlat = source_pos_2d[0] - target_pos_2d[0]
                dlon = (source_pos_2d[1] - target_pos_2d[1] + 180) % 360 - 180
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
                f"({num_fallback_points / total_grid_points * 100:.1f}%) grid points. "
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

            if source_is_icosahedral:
                source_pos_3d = lat_lon_to_cartesian(
                    np.array([source_pos_2d[0]]), np.array([source_pos_2d[1]])
                )[0]
                target_pos_3d = lat_lon_to_cartesian(
                    np.array([target_pos_2d[0]]), np.array([target_pos_2d[1]])
                )[0]
                d = np.sqrt(np.sum((source_pos_3d - target_pos_3d) ** 2))
            elif target_is_icosahedral:
                source_pos_3d = lat_lon_to_cartesian(
                    np.array([source_pos_2d[0]]), np.array([source_pos_2d[1]])
                )[0]
                target_pos_3d = lat_lon_to_cartesian(
                    np.array([target_pos_2d[0]]), np.array([target_pos_2d[1]])
                )[0]
                d = np.sqrt(np.sum((source_pos_3d - target_pos_3d) ** 2))
            else:
                dlat = source_pos_2d[0] - target_pos_2d[0]
                dlon = source_pos_2d[1] - target_pos_2d[1]
                dlon = (dlon + 180) % 360 - 180
                d = np.sqrt(dlat**2 + dlon**2)

            G_connect.add_edge(source_node, target_node)
            G_connect.edges[source_node, target_node]["len"] = d

            if source_is_icosahedral:
                source_pos_3d = lat_lon_to_cartesian(
                    np.array([source_pos_2d[0]]), np.array([source_pos_2d[1]])
                )[0]
                target_pos_3d = lat_lon_to_cartesian(
                    np.array([target_pos_2d[0]]), np.array([target_pos_2d[1]])
                )[0]
                vdiff = tangential_plane_vdiff(source_pos_3d, target_pos_3d)
            elif target_is_icosahedral:
                source_pos_3d = lat_lon_to_cartesian(
                    np.array([source_pos_2d[0]]), np.array([source_pos_2d[1]])
                )[0]
                target_pos_3d = lat_lon_to_cartesian(
                    np.array([target_pos_2d[0]]), np.array([target_pos_2d[1]])
                )[0]
                vdiff = tangential_plane_vdiff(source_pos_3d, target_pos_3d)
            else:
                dlat = source_pos_2d[0] - target_pos_2d[0]
                dlon = (source_pos_2d[1] - target_pos_2d[1] + 180) % 360 - 180
                vdiff = np.array([dlat, dlon])
            G_connect.edges[source_node, target_node]["vdiff"] = vdiff

    return G_connect
