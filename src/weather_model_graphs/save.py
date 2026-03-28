import pickle
from pathlib import Path
from typing import Dict, List

import networkx
from loguru import logger

from .networkx_utils import (
    MissingEdgeAttributeError,
    sort_nodes_in_graph,
    split_graph_by_edge_attribute,
)

try:
    import torch
    import torch_geometric as pyg
    import torch_geometric.utils.convert as pyg_convert

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def to_pyg(
    graph: networkx.DiGraph,
    output_directory: str,
    name: str,
    edge_features: List[str] | None = None,
    node_features: List[str] | None = None,
    list_from_attribute=None,
):
    """
    Save the networkx graph to PyTorch Geometric format that matches what the
    neural-lam model expects as input

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph to save.
    output_directory : str
        Directory to save the graph to.
    name : str
        Name of the graph, this is used to name the files. The edge index and features
        are saved to {output_directory}/{name}_edge_index.pt and
        {output_directory}/{name}_features.pt respectively.
    list_from_attribute : str, optional
        If provided, the graph is split by the attribute value of the edges. The
        stored edge index and features are then the concatenation of the split graphs,
        so that a separate pyg.Data object can be created for each subgraph
        (e.g. one for each level in a multi-level graph). Default is None.
    edge_features: List[str]
        list of edge attributes to include in `{name}_edge_features.pt` file
    node_features: List[str]
        list of node attributes to include in `{name}_node_features.pt` file

    Returns
    -------
    None
    """
    if name is None:
        raise ValueError("Name must be provided.")

    if not HAS_PYG:
        raise Exception(
            "install weather-mode-graphs[pytorch] to enable writing to torch files"
        )

    # Default values for arguments
    if edge_features is None:
        edge_features = ["len", "vdiff"]

    if node_features is None:
        node_features = ["pos"]

    # check that the node labels are integers and unique so that they can be used as indices
    if not all(isinstance(node, int) for node in graph.nodes):
        node_types = set([type(node) for node in graph.nodes])
        raise ValueError(
            f"Node labels must be integers. Instead they are of types {node_types}."
        )
    if len(set(graph.nodes)) != len(graph.nodes):
        raise ValueError("Node labels must be unique.")

    # remove all node attributes but the ones we want to keep
    for node in graph.nodes:
        for attr in list(graph.nodes[node].keys()):
            if attr not in node_features:
                del graph.nodes[node][attr]

    def _get_edge_indecies(pyg_g):
        return pyg_g.edge_index

    def _concat_pyg_features(
        pyg_g: "pyg.data.Data", features: List[str]
    ) -> torch.Tensor:
        """Convert features from pyg.Data object to torch.Tensor.
        Each feature should be column in the resulting 2D tensor (n_edges or n_nodes, n_features).
        Note, this function can handle node AND edge features.
        """
        v_concat = []
        for f in features:
            v = pyg_g[f]
            # Convert 1D features into 1xN tensor
            if v.ndim == 1:
                v = v.unsqueeze(1)
            v_concat.append(v)

        return torch.cat(v_concat, dim=1).to(torch.float32)

    if list_from_attribute is not None:
        # create a list of graph objects by splitting the graph by the list_from_attribute
        try:
            sub_graphs = [
                value
                for key, value in sorted(
                    split_graph_by_edge_attribute(
                        graph=graph, attr=list_from_attribute
                    ).items()
                )
            ]
        except MissingEdgeAttributeError:
            # neural-lam still expects a list of graphs, so if the attribute is missing
            # we just return the original graph as a list
            sub_graphs = [graph]
        # Nodes must be sorted if we want to preserve the ordering in node
        # labels when we convert to a pyg object. This conversion does not care
        # about node labels inherently.
        pyg_graphs = [
            pyg_convert.from_networkx(sort_nodes_in_graph(g)) for g in sub_graphs
        ]
    else:
        pyg_graphs = [pyg_convert.from_networkx(sort_nodes_in_graph(graph))]

    edge_features_values = [
        _concat_pyg_features(pyg_g, features=edge_features) for pyg_g in pyg_graphs
    ]
    edge_indecies = [_get_edge_indecies(pyg_g) for pyg_g in pyg_graphs]
    node_features_values = [
        _concat_pyg_features(pyg_g, features=node_features) for pyg_g in pyg_graphs
    ]

    if list_from_attribute is None:
        edge_features_values = edge_features_values[0]
        edge_indecies = edge_indecies[0]

    Path(output_directory).mkdir(exist_ok=True, parents=True)
    fp_edge_index = Path(output_directory) / f"{name}_edge_index.pt"
    fp_features = Path(output_directory) / f"{name}_features.pt"
    torch.save(edge_indecies, fp_edge_index)
    torch.save(edge_features_values, fp_features)
    logger.info(
        f"Saved edge index to {fp_edge_index} and features {edge_features} to {fp_features}."
    )

    # save node features
    fp_node_features = Path(output_directory) / f"{name}_node_features.pt"
    torch.save(node_features_values, fp_node_features)
    logger.info(f"Saved node features {node_features} to {fp_node_features}.")


def _graph_to_edge_tensors(graph, edge_features=None):
    """Convert a single networkx DiGraph to edge_index and edge_features tensors.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph to convert, must have integer node labels.
    edge_features : list of str, optional
        Edge attribute names to include. Default: ["len", "vdiff"].

    Returns
    -------
    edge_index : torch.Tensor
        Shape (2, num_edges).
    features : torch.Tensor
        Shape (num_edges, num_feature_cols). With default features
        this is (num_edges, 3) for [len, vdiff_x, vdiff_y].
    """
    if not HAS_PYG:
        raise RuntimeError(
            "install weather-model-graphs[pytorch] to enable writing to torch files"
        )

    if edge_features is None:
        edge_features = ["len", "vdiff"]

    # Strip node attributes to only "pos" and edge attributes to only the
    # requested features so that from_networkx does not fail on heterogeneous
    # attribute sets (e.g. g2m graphs with grid + mesh nodes).
    clean = networkx.DiGraph()
    for node, data in sorted(graph.nodes(data=True)):
        clean.add_node(node, pos=data["pos"])
    for u, v, data in graph.edges(data=True):
        edge_data = {k: data[k] for k in edge_features if k in data}
        clean.add_edge(u, v, **edge_data)

    sorted_graph = sort_nodes_in_graph(clean)
    pyg_graph = pyg_convert.from_networkx(sorted_graph)

    edge_index = pyg_graph.edge_index

    v_concat = []
    for f in edge_features:
        v = pyg_graph[f]
        if v.ndim == 1:
            v = v.unsqueeze(1)
        v_concat.append(v)
    features = torch.cat(v_concat, dim=1).to(torch.float32)

    return edge_index, features


def _graph_to_node_features(graph, node_features=None):
    """Extract node feature tensor from a networkx DiGraph.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph with integer node labels and node attributes.
    node_features : list of str, optional
        Node attribute names to include. Default: ["pos"].

    Returns
    -------
    torch.Tensor
        Shape (num_nodes, num_feature_cols). With default features
        this is (num_nodes, 2) for [pos_x, pos_y].
    """
    if not HAS_PYG:
        raise RuntimeError(
            "install weather-model-graphs[pytorch] to enable writing to torch files"
        )

    if node_features is None:
        node_features = ["pos"]

    # Strip to only requested node attributes for clean PyG conversion
    clean = networkx.DiGraph()
    for node, data in sorted(graph.nodes(data=True)):
        keep = {k: data[k] for k in node_features if k in data}
        clean.add_node(node, **keep)
    clean.add_edges_from(graph.edges())

    sorted_graph = sort_nodes_in_graph(clean)
    pyg_graph = pyg_convert.from_networkx(sorted_graph)

    v_concat = []
    for f in node_features:
        v = pyg_graph[f]
        if v.ndim == 1:
            v = v.unsqueeze(1)
        v_concat.append(v)

    return torch.cat(v_concat, dim=1).to(torch.float32)


def to_neural_lam(
    graph_components: Dict[str, networkx.DiGraph],
    output_directory: str,
    hierarchical: bool = False,
):
    """
    Save graph components to the neural-lam tensor-on-disk format.

    Takes graph components as returned by
    ``wmg.create.archetype.*(..., return_components=True)`` and writes
    ``.pt`` files matching the format expected by
    ``neural_lam.utils.load_graph()``.

    Edge features are written **raw** (unnormalized) — neural-lam normalizes
    at load time. Mesh node features (positions) are normalized by
    ``max(abs(pos))`` before saving, matching the existing neural-lam convention.

    Parameters
    ----------
    graph_components : dict of networkx.DiGraph
        Dictionary with keys ``"g2m"``, ``"m2m"``, and ``"m2g"``, each mapping
        to a directed graph. This is the output of
        ``wmg.create.archetype.*(..., return_components=True)``.
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

    **Additional files** (hierarchical only):

    - ``mesh_up_edge_index.pt`` — ``List[torch.Tensor]``, each ``(2, M_up_l)``
    - ``mesh_up_features.pt`` — ``List[torch.Tensor]``, each ``(M_up_l, 3)``
    - ``mesh_down_edge_index.pt`` — ``List[torch.Tensor]``, each ``(2, M_down_l)``
    - ``mesh_down_features.pt`` — ``List[torch.Tensor]``, each ``(M_down_l, 3)``

    Edge features have 3 columns: ``[len, vdiff_x, vdiff_y]``.
    Mesh node features have 2 columns: ``[pos_x, pos_y]`` (normalized).
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

    # --- g2m (grid-to-mesh): single tensor ---
    g2m_graph = graph_components["g2m"]
    g2m_edge_index, g2m_features = _graph_to_edge_tensors(g2m_graph)
    torch.save(g2m_edge_index, output_dir / "g2m_edge_index.pt")
    torch.save(g2m_features, output_dir / "g2m_features.pt")
    logger.info(f"Saved g2m edges: {g2m_edge_index.shape[1]} edges")

    # --- m2g (mesh-to-grid): single tensor ---
    m2g_graph = graph_components["m2g"]
    m2g_edge_index, m2g_features = _graph_to_edge_tensors(m2g_graph)
    torch.save(m2g_edge_index, output_dir / "m2g_edge_index.pt")
    torch.save(m2g_features, output_dir / "m2g_features.pt")
    logger.info(f"Saved m2g edges: {m2g_edge_index.shape[1]} edges")

    # --- m2m (mesh-to-mesh): list of tensors per level ---
    m2m_graph = graph_components["m2m"]

    if hierarchical:
        # Split by direction: "same", "up", "down"
        direction_subgraphs = split_graph_by_edge_attribute(
            m2m_graph, attr="direction"
        )

        # --- Intra-level (same-level) m2m edges ---
        same_graph = direction_subgraphs["same"]
        try:
            level_subgraphs = split_graph_by_edge_attribute(
                same_graph, attr="level"
            )
        except MissingEdgeAttributeError:
            level_subgraphs = {0: same_graph}
        sorted_levels = sorted(level_subgraphs.keys())

        m2m_edge_indices = []
        m2m_features_list = []
        mesh_node_features_list = []
        for level_key in sorted_levels:
            sub = level_subgraphs[level_key]
            ei, ef = _graph_to_edge_tensors(sub)
            nf = _graph_to_node_features(sub)
            m2m_edge_indices.append(ei)
            m2m_features_list.append(ef)
            mesh_node_features_list.append(nf)

        # --- Inter-level up edges ---
        up_graph = direction_subgraphs["up"]
        try:
            up_subgraphs = split_graph_by_edge_attribute(up_graph, attr="levels")
        except MissingEdgeAttributeError:
            up_subgraphs = {"0": up_graph}
        sorted_up_keys = sorted(up_subgraphs.keys())

        mesh_up_edge_indices = []
        mesh_up_features_list = []
        for key in sorted_up_keys:
            ei, ef = _graph_to_edge_tensors(up_subgraphs[key])
            mesh_up_edge_indices.append(ei)
            mesh_up_features_list.append(ef)

        # --- Inter-level down edges ---
        down_graph = direction_subgraphs["down"]
        try:
            down_subgraphs = split_graph_by_edge_attribute(
                down_graph, attr="levels"
            )
        except MissingEdgeAttributeError:
            down_subgraphs = {"0": down_graph}
        sorted_down_keys = sorted(down_subgraphs.keys())

        mesh_down_edge_indices = []
        mesh_down_features_list = []
        for key in sorted_down_keys:
            ei, ef = _graph_to_edge_tensors(down_subgraphs[key])
            mesh_down_edge_indices.append(ei)
            mesh_down_features_list.append(ef)

        # Save hierarchical-only files
        torch.save(
            mesh_up_edge_indices, output_dir / "mesh_up_edge_index.pt"
        )
        torch.save(
            mesh_up_features_list, output_dir / "mesh_up_features.pt"
        )
        torch.save(
            mesh_down_edge_indices, output_dir / "mesh_down_edge_index.pt"
        )
        torch.save(
            mesh_down_features_list, output_dir / "mesh_down_features.pt"
        )
        logger.info(
            f"Saved hierarchical mesh_up ({len(mesh_up_edge_indices)} levels) "
            f"and mesh_down ({len(mesh_down_edge_indices)} levels)"
        )

    else:
        # Non-hierarchical: split by "level" if available, otherwise single list
        try:
            level_subgraphs = split_graph_by_edge_attribute(
                m2m_graph, attr="level"
            )
        except MissingEdgeAttributeError:
            level_subgraphs = {0: m2m_graph}
        sorted_levels = sorted(level_subgraphs.keys())

        m2m_edge_indices = []
        m2m_features_list = []
        mesh_node_features_list = []
        for level_key in sorted_levels:
            sub = level_subgraphs[level_key]
            ei, ef = _graph_to_edge_tensors(sub)
            nf = _graph_to_node_features(sub)
            m2m_edge_indices.append(ei)
            m2m_features_list.append(ef)
            mesh_node_features_list.append(nf)

    # Save m2m edge tensors (always as lists)
    torch.save(m2m_edge_indices, output_dir / "m2m_edge_index.pt")
    torch.save(m2m_features_list, output_dir / "m2m_features.pt")
    logger.info(f"Saved m2m edges: {len(m2m_edge_indices)} level(s)")

    # --- mesh_features.pt: normalized mesh node positions ---
    pos_max = max(
        torch.max(torch.abs(nf)) for nf in mesh_node_features_list
    )
    mesh_features_normalized = [nf / pos_max for nf in mesh_node_features_list]
    torch.save(mesh_features_normalized, output_dir / "mesh_features.pt")
    logger.info(
        f"Saved mesh_features: {len(mesh_features_normalized)} level(s), "
        f"normalized by pos_max={pos_max:.4f}"
    )


def to_pickle(graph: networkx.DiGraph, output_directory: str, name: str):
    """
    Save the networkx graph to a pickle file.
    """
    fp = Path(output_directory) / f"{name}.pickle"
    with open(fp, "wb") as f:
        pickle.dump(graph, f)
    logger.info(f"Saved graph to {fp}.")
