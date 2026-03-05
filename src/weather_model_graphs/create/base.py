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
    create_flat_multiscale_from_coordinates,
    create_flat_singlescale_from_coordinates,
)
from .mesh.kinds.hierarchical import (
    create_hierarchical_from_coordinates,
)
from .mesh.coords import (
    create_multirange_2d_mesh_primitives,
    create_single_level_2d_mesh_primitive,
)
from .mesh.kinds.triangular import (
    create_flat_multiscale_from_triangular_coordinates,
    create_multirange_2d_triangular_mesh_primitives,
    create_single_level_2d_triangular_mesh_primitive,
)


def _migrate_deprecated_kwargs(mesh_layout_kwargs, m2m_connectivity_kwargs):
    """Migrate old-style kwargs to the new mesh_layout_kwargs structure.

    In the old API, ``mesh_node_distance``, ``level_refinement_factor``, and
    ``max_num_levels`` were passed via ``m2m_connectivity_kwargs``.  In the new
    design these belong in ``mesh_layout_kwargs`` (as ``mesh_node_spacing``,
    ``refinement_factor``, and ``max_num_refinement_levels`` respectively).

    This helper emits ``DeprecationWarning`` for each migrated key and moves
    the value into *mesh_layout_kwargs*.  It is intended to be removed once the
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
    if "mesh_node_distance" in m2m_connectivity_kwargs and "mesh_node_spacing" not in mesh_layout_kwargs:
        warnings.warn(
            "Passing 'mesh_node_distance' in m2m_connectivity_kwargs is deprecated. "
            "Use mesh_layout_kwargs=dict(mesh_node_spacing=...) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        mesh_layout_kwargs["mesh_node_spacing"] = m2m_connectivity_kwargs.pop(
            "mesh_node_distance"
        )
    if "level_refinement_factor" in m2m_connectivity_kwargs and "refinement_factor" not in mesh_layout_kwargs:
        warnings.warn(
            "Passing 'level_refinement_factor' in m2m_connectivity_kwargs is deprecated. "
            "Use mesh_layout_kwargs=dict(refinement_factor=...) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        mesh_layout_kwargs["refinement_factor"] = (
            m2m_connectivity_kwargs.pop("level_refinement_factor")
        )
    if "max_num_levels" in m2m_connectivity_kwargs and "max_num_refinement_levels" not in mesh_layout_kwargs:
        warnings.warn(
            "Passing 'max_num_levels' in m2m_connectivity_kwargs is deprecated. "
            "Use mesh_layout_kwargs=dict(max_num_refinement_levels=...) instead.",
            DeprecationWarning,
            stacklevel=3,
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
):
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

    The following methods are available for connecting nodes across graphs:

    g2m_connectivity:
    - "nearest_neighbour": Find the nearest neighbour in grid for each node in mesh
    - "nearest_neighbours": Find the `max_num_neighbours` nearest neighbours in grid for each node in mesh
    - "within_radius": Find all neighbours in grid within an absolute distance
        of `max_dist` or relative distance of `rel_max_dist` from each node in mesh

    mesh_layout:
    - "rectilinear": Uniform regular grid with ``mesh_node_spacing`` resolution.
        Produces an undirected mesh primitive with 4-star (cardinal) and
        8-star (cardinal + diagonal) spatial adjacency edges.
    - "triangular": Regular triangular lattice with ``mesh_node_spacing``
        resolution.  Uses ``networkx.triangular_lattice_graph`` to produce
        equilateral triangles with 6-connectivity.  A CRS warning is emitted
        if ``graph_crs`` is geographic (lat/lon).

    mesh_layout_kwargs (for mesh_layout="rectilinear" or "triangular"):
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

    # CRS warning for triangular layout in geographic coordinates
    if (
        mesh_layout == "triangular"
        and graph_crs is not None
        and graph_crs.is_geographic
    ):
        warnings.warn(
            "mesh_layout='triangular' produces non-uniform physical spacing in "
            "geographic coordinates. Consider mesh_layout='icosahedral' for "
            "uniform coverage on a sphere.",
            UserWarning,
            stacklevel=2,
        )

    if m2m_connectivity == "flat":
        # --- Step 1: Coordinate creation based on mesh_layout ---
        if mesh_layout == "rectilinear":
            mesh_node_spacing = mesh_layout_kwargs.get("mesh_node_spacing")
            # Backward compat: also check for old name "grid_spacing"
            if mesh_node_spacing is None:
                mesh_node_spacing = mesh_layout_kwargs.get("grid_spacing")
            if mesh_node_spacing is None:
                raise ValueError(
                    "mesh_layout='rectilinear' requires 'mesh_node_spacing' in "
                    "mesh_layout_kwargs (or 'mesh_node_distance' in "
                    "m2m_connectivity_kwargs for backward compatibility)."
                )
            # Compute number of mesh nodes from mesh_node_spacing
            range_x, range_y = np.ptp(xy, axis=0)
            nx_mesh = int(range_x / mesh_node_spacing)
            ny_mesh = int(range_y / mesh_node_spacing)
            if nx_mesh == 0 or ny_mesh == 0:
                raise ValueError(
                    "The given `mesh_node_spacing` is too large for the provided "
                    f"coordinates. Got mesh_node_spacing={mesh_node_spacing}, but the "
                    f"x-range is {range_x} and y-range is {range_y}. Maybe you "
                    "want to decrease the `mesh_node_spacing` so that the mesh nodes "
                    "are spaced closer together?"
                )
            G_mesh_coords = create_single_level_2d_mesh_primitive(
                xy, nx_mesh, ny_mesh
            )

            # --- Step 2: Connectivity creation ---
            pattern = m2m_connectivity_kwargs.get("pattern", "8-star")
            graph_components["m2m"] = create_flat_singlescale_from_coordinates(
                G_mesh_coords, pattern=pattern
            )
            grid_connect_graph = graph_components["m2m"]

        elif mesh_layout == "triangular":
            # --- Step 1: Coordinate creation (triangular lattice) ---
            mesh_node_spacing = mesh_layout_kwargs.get("mesh_node_spacing")
            if mesh_node_spacing is None:
                mesh_node_spacing = mesh_layout_kwargs.get("grid_spacing")
            if mesh_node_spacing is None:
                raise ValueError(
                    "mesh_layout='triangular' requires 'mesh_node_spacing' in "
                    "mesh_layout_kwargs (or 'mesh_node_distance' in "
                    "m2m_connectivity_kwargs for backward compatibility)."
                )
            range_x, range_y = np.ptp(xy, axis=0)
            nx_mesh = int(range_x / mesh_node_spacing)
            ny_mesh = int(range_y / (mesh_node_spacing * np.sqrt(3) / 2))
            if nx_mesh == 0 or ny_mesh == 0:
                raise ValueError(
                    "The given `mesh_node_spacing` is too large for the provided "
                    f"coordinates. Got mesh_node_spacing={mesh_node_spacing}, but the "
                    f"x-range is {range_x} and y-range is {range_y}. Maybe you "
                    "want to decrease the `mesh_node_spacing` so that the mesh nodes "
                    "are spaced closer together?"
                )
            G_mesh_coords = create_single_level_2d_triangular_mesh_primitive(
                xy, nx_mesh, ny_mesh
            )

            # --- Step 2: Connectivity creation ---
            pattern = m2m_connectivity_kwargs.get("pattern", "4-star")
            graph_components["m2m"] = create_flat_singlescale_from_coordinates(
                G_mesh_coords, pattern=pattern
            )
            grid_connect_graph = graph_components["m2m"]

        else:
            raise NotImplementedError(
                f"mesh_layout='{mesh_layout}' is not yet supported. "
                "Currently supported: 'rectilinear', 'triangular'."
            )

    elif m2m_connectivity == "hierarchical":
        # --- Step 1: Coordinate creation based on mesh_layout ---
        if mesh_layout == "rectilinear":
            mesh_node_spacing = mesh_layout_kwargs.get("mesh_node_spacing")
            if mesh_node_spacing is None:
                mesh_node_spacing = mesh_layout_kwargs.get("grid_spacing")
            refinement_factor = mesh_layout_kwargs.get("refinement_factor", 3)
            max_num_refinement_levels = mesh_layout_kwargs.get(
                "max_num_refinement_levels"
            )
            if mesh_node_spacing is None:
                raise ValueError(
                    "mesh_layout='rectilinear' with m2m_connectivity='hierarchical' "
                    "requires 'mesh_node_spacing' in mesh_layout_kwargs."
                )
            G_coords_list = create_multirange_2d_mesh_primitives(
                max_num_levels=max_num_refinement_levels,
                xy=xy,
                mesh_node_spacing=mesh_node_spacing,
                interlevel_refinement_factor=refinement_factor,
            )

        elif mesh_layout == "triangular":
            mesh_node_spacing = mesh_layout_kwargs.get("mesh_node_spacing")
            if mesh_node_spacing is None:
                mesh_node_spacing = mesh_layout_kwargs.get("grid_spacing")
            refinement_factor = mesh_layout_kwargs.get("refinement_factor", 3)
            max_num_refinement_levels = mesh_layout_kwargs.get(
                "max_num_refinement_levels"
            )
            if mesh_node_spacing is None:
                raise ValueError(
                    "mesh_layout='triangular' with m2m_connectivity='hierarchical' "
                    "requires 'mesh_node_spacing' in mesh_layout_kwargs."
                )
            G_coords_list = create_multirange_2d_triangular_mesh_primitives(
                max_num_levels=max_num_refinement_levels,
                xy=xy,
                mesh_node_spacing=mesh_node_spacing,
                interlevel_refinement_factor=refinement_factor,
            )

        else:
            raise NotImplementedError(
                f"mesh_layout='{mesh_layout}' is not yet supported. "
                "Currently supported: 'rectilinear', 'triangular'."
            )

        # --- Step 2: Connectivity creation ---
        # hierarchical mesh graph have three sub-graphs:
        # `m2m` (mesh-to-mesh), `mesh_up` (up edge connections) and
        # `mesh_down` (down edge connections)
        intra_level = m2m_connectivity_kwargs.get("intra_level")
        if intra_level is None and mesh_layout == "triangular":
            intra_level = {"pattern": "4-star"}
        graph_components["m2m"] = create_hierarchical_from_coordinates(
            G_coords_list,
            intra_level=intra_level,
            inter_level=m2m_connectivity_kwargs.get("inter_level"),
        )
        # Only connect grid to bottom level of hierarchy
        grid_connect_graph = split_graph_by_edge_attribute(
            graph_components["m2m"], "level"
        )[0]

    elif m2m_connectivity == "flat_multiscale":
        # --- Step 1: Coordinate creation based on mesh_layout ---
        if mesh_layout == "rectilinear":
            mesh_node_spacing = mesh_layout_kwargs.get("mesh_node_spacing")
            if mesh_node_spacing is None:
                mesh_node_spacing = mesh_layout_kwargs.get("grid_spacing")
            refinement_factor = mesh_layout_kwargs.get("refinement_factor", 3)
            max_num_refinement_levels = mesh_layout_kwargs.get(
                "max_num_refinement_levels"
            )
            if mesh_node_spacing is None:
                raise ValueError(
                    "mesh_layout='rectilinear' with m2m_connectivity='flat_multiscale' "
                    "requires 'mesh_node_spacing' in mesh_layout_kwargs."
                )
            G_coords_list = create_multirange_2d_mesh_primitives(
                max_num_levels=max_num_refinement_levels,
                xy=xy,
                mesh_node_spacing=mesh_node_spacing,
                interlevel_refinement_factor=refinement_factor,
            )

        elif mesh_layout == "triangular":
            mesh_node_spacing = mesh_layout_kwargs.get("mesh_node_spacing")
            if mesh_node_spacing is None:
                mesh_node_spacing = mesh_layout_kwargs.get("grid_spacing")
            refinement_factor = mesh_layout_kwargs.get("refinement_factor", 3)
            max_num_refinement_levels = mesh_layout_kwargs.get(
                "max_num_refinement_levels"
            )
            if mesh_node_spacing is None:
                raise ValueError(
                    "mesh_layout='triangular' with m2m_connectivity='flat_multiscale' "
                    "requires 'mesh_node_spacing' in mesh_layout_kwargs."
                )
            G_coords_list = create_multirange_2d_triangular_mesh_primitives(
                max_num_levels=max_num_refinement_levels,
                xy=xy,
                mesh_node_spacing=mesh_node_spacing,
                interlevel_refinement_factor=refinement_factor,
            )

        else:
            raise NotImplementedError(
                f"mesh_layout='{mesh_layout}' is not yet supported. "
                "Currently supported: 'rectilinear', 'triangular'."
            )

        # --- Step 2: Connectivity creation ---
        pattern = m2m_connectivity_kwargs.get("pattern")
        if pattern is None:
            pattern = "4-star" if mesh_layout == "triangular" else "8-star"
        if mesh_layout == "triangular":
            graph_components["m2m"] = create_flat_multiscale_from_triangular_coordinates(
                G_coords_list,
                pattern=pattern,
            )
        else:
            graph_components["m2m"] = create_flat_multiscale_from_coordinates(
                G_coords_list,
                pattern=pattern,
            )
        grid_connect_graph = graph_components["m2m"]
    else:
        raise NotImplementedError(f"Kind {m2m_connectivity} not implemented")

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
    max_dist : float
        Maximum distance to search for neighbours in `G_target` for each node in `G_source`
    rel_max_dist : float
        Maximum distance to search for neighbours in `G_target` for each node in `G_source`,
        relative to longest edge in (bottom level of) `G_source` and `G_target`.
    max_num_neighbours : int
        Maximum number of neighbours to search for in `G_target` for each node in `G_source`

    Returns
    -------
    networkx.DiGraph
        Graph containing the nodes in `G_source` and `G_target` and directed edges
        from nodes in `G_source` to nodes in `G_target`
    """
    source_nodes_list = list(G_source.nodes)
    target_nodes_list = list(G_target.nodes)

    # build kd tree for source nodes (e.g. the mesh nodes when constructing m2g)
    xy_source = np.array([G_source.nodes[node]["pos"] for node in G_source.nodes])
    kdt_s = scipy.spatial.KDTree(xy_source)

    # Determine method and perform checks once
    # Conditionally define _find_neighbour_node_idxs_in_source_mesh for use in
    # loop later
    if method == "containing_rectangle":
        if (
            max_dist is not None
            or rel_max_dist is not None
            or max_num_neighbours is not None
        ):
            raise Exception(
                "to use `containing_rectangle` you should not set `max_dist`, `rel_max_dist`or `max_num_neighbours`"
            )
        assert (
            "dx" in G_source.graph and "dy" in G_source.graph
        ), "Source graph must have dx and dy properties to connect nodes using method containing_rectangle"

        # Connect to all nodes that could potentially be close enough,
        # which is at a relative distance of 1. This relative distance is equal
        # to the diagonal of one rectangle.
        rad_graph = connect_nodes_across_graphs(
            G_source, G_target, method="within_radius", rel_max_dist=1.0
        )

        # Filter edges to those that fit within a rectangle of measurements dx,dy
        mesh_node_dx = G_source.graph["dx"]
        mesh_node_dy = G_source.graph["dy"]

        if isinstance(mesh_node_dx, dict):
            # In hierarchical graph these properties are dicts, in that case use
            # values for bottom level.
            mesh_node_dx = mesh_node_dx[0]
            mesh_node_dy = mesh_node_dy[0]

        # This function is a filter that applies to edges, represented as vectors (vx, vy) in R^ 2.
        # The filter is True if |vx| < dx & |vy| < dy, where dx and dy are the distance between
        # rows and columns in source quadrilateral graph.
        def _edge_filter(edge_prop):
            abs_diffs = np.abs(edge_prop["vdiff"])
            return abs_diffs[0] < mesh_node_dx and abs_diffs[1] < mesh_node_dy

        filtered_edges = [
            (u, v, edge_prop)
            for u, v, edge_prop in rad_graph.edges(data=True)
            if _edge_filter(edge_prop)
        ]

        # Construct subgraph with only filtered edges, but all nodes
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
                "to use `nearest_neighbour` you should not set `max_dist`, `rel_max_dist`or `max_num_neighbours`"
            )

        def _find_neighbour_node_idxs_in_source_mesh(xy_target):
            neigh_idx = kdt_s.query(xy_target, 1)[1]
            return [neigh_idx]

    elif method == "nearest_neighbours":
        if max_num_neighbours is None:
            raise Exception(
                "to use `nearest_neighbours` you should set the max number with `max_num_neighbours`"
            )
        if max_dist is not None or rel_max_dist is not None:
            raise Exception(
                "to use `nearest_neighbours` you should not set `max_dist` or `rel_max_dist`"
            )

        def _find_neighbour_node_idxs_in_source_mesh(xy_target):
            neigh_idxs = kdt_s.query(xy_target, max_num_neighbours)[1]
            return neigh_idxs

    elif method == "within_radius":
        if max_num_neighbours is not None:
            raise Exception(
                "to use `within_radius` method you should not set `max_num_neighbours`"
            )
        # Determine actual query length to use
        if max_dist is not None:
            if rel_max_dist is not None:
                raise Exception(
                    "to use `witin_radius` method you should only set one of `max_dist` or `rel_max_dist"
                )
            query_dist = max_dist
        elif rel_max_dist is not None:
            if max_dist is not None:
                raise Exception(
                    "to use `witin_radius` method you should only set one of `max_dist` or `rel_max_dist"
                )
            # Figure out longest edge in (lowest level) mesh graph
            longest_edge = 0.0
            for edge_check_graph in (G_source, G_target):
                # Check if graph has edges
                if len(edge_check_graph.edges) > 0:
                    (
                        level_subgraph,
                        no_level_subgraph,
                    ) = split_on_edge_attribute_existance(edge_check_graph, "level")

                    # Check if graph has levels (hierarchical or multi-scale edges)
                    if nx.is_empty(level_subgraph):
                        # Consider edges in whole graph (whole graph is level 1)
                        first_level_graph = edge_check_graph  # == no_level_subgraph
                    else:
                        # Has levels, only consider edges in level 1 graph
                        first_level_graph = split_graph_by_edge_attribute(
                            level_subgraph, "level"
                        )[0]
                    longest_graph_edge = max(
                        first_level_graph.edges(data=True),
                        key=lambda x: x[2].get("len", 0),
                    )[2]["len"]
                    longest_edge = max(longest_edge, longest_graph_edge)
            query_dist = longest_edge * rel_max_dist
        else:
            raise Exception(
                "to use `witin_radius` method you shold set `max_dist` or `rel_max_dist"
            )

        def _find_neighbour_node_idxs_in_source_mesh(xy_target):
            neigh_idxs = kdt_s.query_ball_point(xy_target, query_dist)
            return neigh_idxs

    else:
        raise NotImplementedError(method)

    G_connect = networkx.DiGraph()
    G_connect.add_nodes_from(sorted(G_source.nodes(data=True)))
    G_connect.add_nodes_from(sorted(G_target.nodes(data=True)))

    # sort nodes by index
    source_nodes_list = sorted(G_source.nodes)

    # add edges
    for target_node in target_nodes_list:
        xy_target = G_target.nodes[target_node]["pos"]
        neigh_idxs = _find_neighbour_node_idxs_in_source_mesh(xy_target)
        for i in neigh_idxs:
            source_node = source_nodes_list[i]
            # add edge from source to target
            G_connect.add_edge(source_node, target_node)
            d = np.sqrt(
                np.sum(
                    (
                        G_connect.nodes[source_node]["pos"]
                        - G_connect.nodes[target_node]["pos"]
                    )
                    ** 2
                )
            )
            G_connect.edges[source_node, target_node]["len"] = d
            G_connect.edges[source_node, target_node]["vdiff"] = (
                G_connect.nodes[source_node]["pos"]
                - G_connect.nodes[target_node]["pos"]
            )

    return G_connect
