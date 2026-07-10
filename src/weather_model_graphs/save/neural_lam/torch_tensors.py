from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import networkx
import numpy as np
from loguru import logger

from ...networkx_utils import MissingEdgeAttributeError, split_graph_by_edge_attribute
from ..base import DEFAULT_EDGE_FEATURES, DEFAULT_NODE_FEATURES, HAS_PYG

if HAS_PYG:
    import torch

# Version of the neural-lam graph storage spec that the tensor-on-disk output
# conforms to. Written into metainfo.yaml by ``to_torch_tensors_on_disk``.
GRAPH_STORAGE_SPEC_VERSION = "0.1.0"


def _graph_to_edge_tensors(
    graph: networkx.DiGraph,
    sender_map: Dict[int, int],
    receiver_map: Dict[int, int],
    edge_features: Tuple[str, ...] = DEFAULT_EDGE_FEATURES,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Convert a networkx DiGraph to edge_index and edge_features tensors.

    Edge indices are expressed in the per-node-set zero-based index spaces
    defined by ``sender_map`` and ``receiver_map``, as required by the
    neural-lam graph storage spec (each node set is numbered independently
    from ``0`` to ``N-1``).

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph to convert, must have integer node labels.
    sender_map : dict
        Mapping from global node label to zero-based index within the
        sender node set.
    receiver_map : dict
        Mapping from global node label to zero-based index within the
        receiver node set.
    edge_features : tuple of str, optional
        Edge attribute names to include, concatenated (in order) into the
        feature columns. Default: ``("len", "vdiff")``.

    Returns
    -------
    edge_index : torch.Tensor
        Shape (2, num_edges), dtype int64. Row 0 is sender, row 1 receiver.
    features : torch.Tensor
        Shape (num_edges, num_feature_cols), dtype float32. With default
        features this is (num_edges, 3) for [len, vdiff_x, vdiff_y].

    Raises
    ------
    ValueError
        If the graph contains no edges. Every graph component is expected to
        have at least one edge.
    """
    if not HAS_PYG:
        raise RuntimeError(
            "install weather-model-graphs[pytorch] to enable writing to torch files"
        )

    # Sort edges by (sender index, receiver index) so the on-disk edge order
    # is deterministic and independent of networkx's internal iteration
    # order. Without this the same graph could serialise to different (but
    # equivalent) edge_index/feature orderings between runs or machines,
    # which breaks byte-for-byte reproducibility and makes diffing graphs
    # harder.
    edges = sorted(
        graph.edges(data=True),
        key=lambda e: (sender_map[e[0]], receiver_map[e[1]]),
    )

    # Every graph component is expected to contain at least one edge. An
    # empty edge set indicates a malformed graph (e.g. mesh nodes left
    # completely unconnected), so fail loudly instead of silently writing an
    # empty tensor.
    if len(edges) == 0:
        raise ValueError(
            "Cannot serialise an empty edge set: the (sub)graph passed to "
            "_graph_to_edge_tensors has no edges. Every graph component must "
            "contain at least one edge."
        )

    edge_index = torch.tensor(
        [
            [sender_map[u] for u, _, _ in edges],
            [receiver_map[v] for _, v, _ in edges],
        ],
        dtype=torch.int64,
    )

    rows = []
    for _, _, data in edges:
        # np.atleast_1d normalises scalar and vector attributes to a common
        # shape so they concatenate into one row: "len" is a scalar while
        # "vdiff" is a 2-/3-vector, and np.concatenate requires 1-D inputs.
        vals = [
            np.atleast_1d(np.asarray(data[f], dtype=np.float64)) for f in edge_features
        ]
        rows.append(np.concatenate(vals))
    features = torch.tensor(np.stack(rows), dtype=torch.float32)

    return edge_index, features


def _node_features_from_labels(
    graph: networkx.DiGraph,
    labels: List[int],
    node_features: Tuple[str, ...] = DEFAULT_NODE_FEATURES,
) -> "torch.Tensor":
    """Extract node feature tensor for the given node labels, in order.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph containing the nodes referenced by ``labels``.
    labels : list
        Node labels defining the row order of the output tensor. This MUST
        be the same ordering used to build the corresponding node index map
        so that edge indices and node features stay consistent.
    node_features : tuple of str, optional
        Node attribute names to include, concatenated (in order) into the
        feature columns. Default: ``("pos",)``.

    Returns
    -------
    torch.Tensor
        Shape (len(labels), num_feature_cols), dtype float32. With default
        features this is (num_nodes, 2) for [pos_x, pos_y].
    """
    if not HAS_PYG:
        raise RuntimeError(
            "install weather-model-graphs[pytorch] to enable writing to torch files"
        )

    rows = []
    for n in labels:
        data = graph.nodes[n]
        vals = [
            np.atleast_1d(np.asarray(data[f], dtype=np.float64)) for f in node_features
        ]
        rows.append(np.concatenate(vals))
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def _mesh_map_by_position(
    graph: networkx.DiGraph,
    pos_to_idx: Dict[Tuple[float, ...], int],
    component_name: str,
) -> Dict[int, int]:
    """Map a component's mesh node labels to mesh indices by node position.

    The g2m/m2g component graphs label their mesh nodes differently from
    the m2m component, so the only reliable cross-component key is the node
    position. Positions originate from the same coordinate arrays in
    ``create_all_graph_components`` so exact float equality holds.

    Parameters
    ----------
    graph : networkx.DiGraph
        Component graph (g2m or m2g) whose mesh nodes are mapped.
    pos_to_idx : dict
        Mapping from a mesh node position (tuple of coordinates) to its
        zero-based index in the bottom-level mesh node set.
    component_name : str
        Name of the component ("g2m"/"m2g"), used only in error messages.

    Returns
    -------
    dict
        Mapping from each mesh node label in ``graph`` to its bottom-level
        mesh index.

    Raises
    ------
    ValueError
        If a mesh node's position has no match in ``pos_to_idx`` (i.e. the
        component connects to a mesh node outside the bottom level).
    """
    mesh_map = {}
    for n, d in graph.nodes(data=True):
        if d.get("type") != "mesh":
            continue
        key = tuple(d["pos"])
        if key not in pos_to_idx:
            raise ValueError(
                f"{component_name} mesh node {n} at position {key} has no "
                f"matching node in the bottom-level mesh node set of the m2m "
                f"component. g2m/m2g must connect to the bottom mesh level "
                f"only."
            )
        mesh_map[n] = pos_to_idx[key]
    return mesh_map


def _interlevel_edge_tensors(
    graph: networkx.DiGraph,
    sorted_levels: List,
    level_maps: List[Dict[int, int]],
    direction: str,
) -> Tuple[List["torch.Tensor"], List["torch.Tensor"]]:
    """Build per-level-pair edge tensors for hierarchical up/down edges.

    Entry ``i`` connects mesh level ``i`` and level ``i+1``: for
    ``direction="up"`` the sender is level ``i`` and receiver level ``i+1``,
    for ``direction="down"`` the sender is level ``i+1`` and receiver level
    ``i``. Sender and receiver levels are determined from the ``"level"``
    node attribute of each edge's endpoints, so this does not depend on the
    format of the ``"levels"`` edge attribute.

    Parameters
    ----------
    graph : networkx.DiGraph
        Sub-graph containing only the inter-level edges of one direction.
    sorted_levels : list
        Mesh level identifiers in ascending order (bottom level first).
    level_maps : list of dict
        Per-level mapping from global node label to zero-based index within
        that level, indexed by level position.
    direction : str
        Either ``"up"`` or ``"down"``; selects the sender/receiver ordering.

    Returns
    -------
    edge_indices : list of torch.Tensor
        One ``(2, E_i)`` int64 tensor per level pair.
    features_list : list of torch.Tensor
        One ``(E_i, N_f)`` float32 tensor per level pair.
    """
    level_index = {lvl: i for i, lvl in enumerate(sorted_levels)}

    # Group edges by the index of the lower level of the pair they connect
    pair_edges = {}
    for u, v, data in graph.edges(data=True):
        u_level = level_index[graph.nodes[u]["level"]]
        v_level = level_index[graph.nodes[v]["level"]]
        lower = min(u_level, v_level)
        pair_edges.setdefault(lower, []).append((u, v, data))

    edge_indices = []
    features_list = []
    for lower in range(len(sorted_levels) - 1):
        edges = pair_edges.get(lower, [])
        if direction == "up":
            sender_map, receiver_map = level_maps[lower], level_maps[lower + 1]
        else:
            sender_map, receiver_map = level_maps[lower + 1], level_maps[lower]

        sub = networkx.DiGraph()
        for u, v, data in edges:
            sub.add_edge(u, v, **data)
        ei, ef = _graph_to_edge_tensors(
            sub, sender_map=sender_map, receiver_map=receiver_map
        )
        edge_indices.append(ei)
        features_list.append(ef)

    return edge_indices, features_list


def to_torch_tensors_on_disk(
    graph_components: Dict[str, networkx.DiGraph],
    output_directory: str,
    hierarchical: bool = False,
) -> None:
    """
    Save graph components to the neural-lam tensor-on-disk format.

    A "graph component" is one of the message-passing subgraphs that make up
    a neural-lam graph: ``g2m`` (grid-to-mesh, the encoder edges), ``m2m``
    (mesh-to-mesh, the processor edges) and ``m2g`` (mesh-to-grid, the
    decoder edges). Each is a ``networkx.DiGraph`` whose nodes carry
    ``"type"`` ("grid"/"mesh"), ``"pos"`` and (for hierarchical graphs)
    ``"level"`` attributes, exactly as returned by
    ``wmg.create.archetype.*(..., return_components=True)``. This function
    turns those graphs into the ``.pt`` files that
    ``neural_lam.utils.load_graph()`` expects, conforming to the neural-lam
    graph storage specification (version ``0.1.0``).

    Steps performed:

    1. Validate that ``graph_components`` contains the required ``g2m``,
       ``m2m`` and ``m2g`` keys.
    2. Build per-node-set zero-based index maps: the grid node set is
       numbered ``0..N_grid-1`` and each mesh level is numbered
       ``0..N_level-1`` independently (in sorted global-node-label order),
       as required by the spec's per-node-set index space.
    3. Match the mesh nodes referenced by ``g2m``/``m2g`` to the bottom mesh
       level by position, since those components label mesh nodes
       differently from ``m2m``.
    4. Convert ``g2m`` and ``m2g`` to single ``edge_index``/``features``
       tensors and save them.
    5. Convert ``m2m``: for hierarchical graphs, split by ``"direction"``
       into intra-level (``m2m``) plus inter-level (``mesh_up``/
       ``mesh_down``) tensors; for non-hierarchical graphs, write a single
       merged ``m2m`` level (``L == 1``).
    6. Write ``mesh_features.pt`` — the raw per-level mesh node positions.
    7. Write ``metainfo.yaml`` recording the graph storage spec version.

    Edge features and mesh node features are written **raw** (unnormalized);
    neural-lam applies normalization at load time.

    Parameters
    ----------
    graph_components : dict of networkx.DiGraph
        Dictionary with keys ``"g2m"``, ``"m2m"``, and ``"m2g"`` (see above).
    output_directory : str
        Directory where the ``.pt`` files will be saved.
    hierarchical : bool, optional
        If True, the m2m graph is expected to contain hierarchical edges
        with ``"direction"`` attribute (``"same"``, ``"up"``, ``"down"``).
        Additional mesh_up/mesh_down files are written. Default: False.

    Returns
    -------
    None

    Notes
    -----
    **Output files** (always produced):

    - ``g2m_edge_index.pt`` — ``torch.Tensor`` of shape ``(2, M_g2m)``
    - ``g2m_features.pt`` — ``torch.Tensor`` of shape ``(M_g2m, 3)``
    - ``m2g_edge_index.pt`` — ``torch.Tensor`` of shape ``(2, M_m2g)``
    - ``m2g_features.pt`` — ``torch.Tensor`` of shape ``(M_m2g, 3)``
    - ``m2m_edge_index.pt`` — ``List[torch.Tensor]``, each ``(2, M_l)``
    - ``m2m_features.pt`` — ``List[torch.Tensor]``, each ``(M_l, 3)``
    - ``mesh_features.pt`` — ``List[torch.Tensor]``, each ``(N_l, 2)``
    - ``metainfo.yaml`` — graph storage spec version marker

    **Additional files** (hierarchical only):

    - ``mesh_up_edge_index.pt`` — ``List[torch.Tensor]``, each ``(2, M_up_l)``
    - ``mesh_up_features.pt`` — ``List[torch.Tensor]``, each ``(M_up_l, 3)``
    - ``mesh_down_edge_index.pt`` — ``List[torch.Tensor]``, each ``(2, M_down_l)``
    - ``mesh_down_features.pt`` — ``List[torch.Tensor]``, each ``(M_down_l, 3)``

    Edge features have 3 columns: ``[len, vdiff_x, vdiff_y]``.
    Mesh node features have 2 columns: ``[pos_x, pos_y]`` (raw coordinates).

    Non-hierarchical graphs are always written with a single mesh level
    (``L == 1``), so for flat multiscale graphs (e.g. the graphcast
    archetype) all m2m edges are stored in one entry over the merged mesh
    node set, matching the graph storage spec requirement that
    non-hierarchical graphs have ``L == 1``.
    """
    if not HAS_PYG:
        raise RuntimeError(
            "install weather-model-graphs[pytorch] to enable writing to torch files"
        )

    required_keys = {"g2m", "m2m", "m2g"}
    missing = required_keys - set(graph_components.keys())
    if missing:
        raise ValueError(
            f"graph_components is missing required keys: {sorted(missing)}. "
            f"Expected keys: {sorted(required_keys)}"
        )

    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True, parents=True)

    g2m_graph = graph_components["g2m"]
    m2m_graph = graph_components["m2m"]
    m2g_graph = graph_components["m2g"]

    # --- Per-node-set zero-based index maps ---
    # Grid nodes: all nodes with type == "grid" across components
    grid_labels = sorted(
        {
            n
            for g in (g2m_graph, m2g_graph)
            for n, d in g.nodes(data=True)
            if d.get("type") == "grid"
        }
    )
    grid_map = {n: i for i, n in enumerate(grid_labels)}

    # Mesh nodes: from the m2m graph, grouped by level for hierarchical
    # graphs, or as one merged set for non-hierarchical graphs
    if hierarchical:
        mesh_labels_by_level = {}
        for n, d in m2m_graph.nodes(data=True):
            mesh_labels_by_level.setdefault(d["level"], set()).add(n)
        sorted_levels = sorted(mesh_labels_by_level.keys())
        level_labels = [sorted(mesh_labels_by_level[lvl]) for lvl in sorted_levels]
        level_maps = [{n: i for i, n in enumerate(labels)} for labels in level_labels]
    else:
        mesh_labels = sorted(
            n for n, d in m2m_graph.nodes(data=True) if d.get("type") == "mesh"
        )
        level_labels = [mesh_labels]
        level_maps = [{n: i for i, n in enumerate(mesh_labels)}]

    # The g2m/m2g components carry their own node labels for mesh nodes
    # (different from the m2m graph labels), so their mesh nodes are matched
    # to the bottom-level mesh index space by position. The lookup is built
    # from the bottom level only, because coarser-level nodes can coincide
    # in position with bottom-level nodes.
    bottom_pos_to_idx = {
        tuple(m2m_graph.nodes[n]["pos"]): i for i, n in enumerate(level_labels[0])
    }
    g2m_mesh_map = _mesh_map_by_position(g2m_graph, bottom_pos_to_idx, "g2m")
    m2g_mesh_map = _mesh_map_by_position(m2g_graph, bottom_pos_to_idx, "m2g")

    # --- g2m (grid-to-mesh): single tensor ---
    g2m_edge_index, g2m_features = _graph_to_edge_tensors(
        g2m_graph, sender_map=grid_map, receiver_map=g2m_mesh_map
    )
    torch.save(g2m_edge_index, output_dir / "g2m_edge_index.pt")
    torch.save(g2m_features, output_dir / "g2m_features.pt")
    logger.info(f"Saved g2m edges: {g2m_edge_index.shape[1]} edges")

    # --- m2g (mesh-to-grid): single tensor ---
    m2g_edge_index, m2g_features = _graph_to_edge_tensors(
        m2g_graph, sender_map=m2g_mesh_map, receiver_map=grid_map
    )
    torch.save(m2g_edge_index, output_dir / "m2g_edge_index.pt")
    torch.save(m2g_features, output_dir / "m2g_features.pt")
    logger.info(f"Saved m2g edges: {m2g_edge_index.shape[1]} edges")

    # --- m2m (mesh-to-mesh): list of tensors per level ---
    if hierarchical:
        # Split by direction: "same", "up", "down"
        direction_subgraphs = split_graph_by_edge_attribute(m2m_graph, attr="direction")

        # --- Intra-level (same-level) m2m edges ---
        same_graph = direction_subgraphs["same"]
        try:
            level_subgraphs = split_graph_by_edge_attribute(same_graph, attr="level")
        except MissingEdgeAttributeError:
            level_subgraphs = {sorted_levels[0]: same_graph}

        m2m_edge_indices = []
        m2m_features_list = []
        for i, lvl in enumerate(sorted_levels):
            sub = level_subgraphs[lvl]
            ei, ef = _graph_to_edge_tensors(
                sub, sender_map=level_maps[i], receiver_map=level_maps[i]
            )
            m2m_edge_indices.append(ei)
            m2m_features_list.append(ef)

        # --- Inter-level up edges (level i -> level i+1) ---
        up_graph = direction_subgraphs["up"]
        mesh_up_edge_indices, mesh_up_features_list = _interlevel_edge_tensors(
            up_graph, sorted_levels, level_maps, direction="up"
        )

        # --- Inter-level down edges (level i+1 -> level i) ---
        down_graph = direction_subgraphs["down"]
        mesh_down_edge_indices, mesh_down_features_list = _interlevel_edge_tensors(
            down_graph, sorted_levels, level_maps, direction="down"
        )

        # Save hierarchical-only files
        torch.save(mesh_up_edge_indices, output_dir / "mesh_up_edge_index.pt")
        torch.save(mesh_up_features_list, output_dir / "mesh_up_features.pt")
        torch.save(mesh_down_edge_indices, output_dir / "mesh_down_edge_index.pt")
        torch.save(mesh_down_features_list, output_dir / "mesh_down_features.pt")
        logger.info(
            f"Saved hierarchical mesh_up ({len(mesh_up_edge_indices)} levels) "
            f"and mesh_down ({len(mesh_down_edge_indices)} levels)"
        )

    else:
        # Non-hierarchical graphs always have L == 1 (per the graph storage
        # spec), so all m2m edges go into a single entry over the merged
        # mesh node set (this includes flat multiscale graphs)
        ei, ef = _graph_to_edge_tensors(
            m2m_graph, sender_map=level_maps[0], receiver_map=level_maps[0]
        )
        m2m_edge_indices = [ei]
        m2m_features_list = [ef]

    # Save m2m edge tensors (always as lists)
    torch.save(m2m_edge_indices, output_dir / "m2m_edge_index.pt")
    torch.save(m2m_features_list, output_dir / "m2m_features.pt")
    logger.info(f"Saved m2m edges: {len(m2m_edge_indices)} level(s)")

    # --- mesh_features.pt: raw (unnormalized) mesh node positions ---
    # The graph storage spec requires unnormalized coordinates; neural-lam
    # normalizes after loading
    mesh_node_features_list = [
        _node_features_from_labels(m2m_graph, labels) for labels in level_labels
    ]
    torch.save(mesh_node_features_list, output_dir / "mesh_features.pt")
    logger.info(
        f"Saved mesh_features: {len(mesh_node_features_list)} level(s), "
        f"raw (unnormalized) coordinates"
    )

    # --- metainfo.yaml: graph storage spec version marker ---
    (output_dir / "metainfo.yaml").write_text(
        f"spec_version: {GRAPH_STORAGE_SPEC_VERSION}\n"
    )
    logger.info(f"Saved metainfo.yaml (spec_version: {GRAPH_STORAGE_SPEC_VERSION})")
