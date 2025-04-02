import pickle
from pathlib import Path
from typing import List

import networkx
import networkx as nx
import numpy as np
import xarray as xr
from loguru import logger

from ..networkx_utils import MissingEdgeAttributeError, split_graph_by_edge_attribute

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
            sub_graphs = list(
                split_graph_by_edge_attribute(
                    graph=graph, attr=list_from_attribute
                ).values()
            )
        except MissingEdgeAttributeError:
            # neural-lam still expects a list of graphs, so if the attribute is missing
            # we just return the original graph as a list
            sub_graphs = [graph]
        pyg_graphs = [pyg_convert.from_networkx(g) for g in sub_graphs]
    else:
        pyg_graphs = [pyg_convert.from_networkx(graph)]

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


def to_pickle(graph: networkx.DiGraph, output_directory: str, name: str):
    """
    Save the networkx graph to a pickle file.
    """
    fp = Path(output_directory) / f"{name}.pickle"
    with open(fp, "wb") as f:
        pickle.dump(graph, f)
    logger.info(f"Saved graph to {fp}.")


def extract_edges_to_dataset(
    graph: nx.DiGraph, edge_attrs: List, edge_id_attr: str
) -> xr.Dataset:
    """
    Extract edge indices and features from a NetworkX DiGraph into an xarray.Dataset.

    This function assumes that all node labels can be uniquely and deterministically
    converted into integers. These integer-converted node labels will be used directly
    as edge indices (i.e., no remapping or reindexing is performed).

    Parameters
    ----------
    graph : nx.DiGraph
        A directed graph where:
        - Each node label must be convertible to an integer (e.g. int, str of int).
        - Edges may have feature attributes (optional, must be uniform).
    edge_attrs : list
        List of edge attributes to extract and include as features in the dataset.
    edge_id_attr : str
        The name of the edge attribute to use as the unique identifier for each
        edge. Using this ensures that we can use a globally unique id for each
        edge across the entire graph.

    Returns
    -------
    xr.Dataset
        A dataset with:
        - 'adjacency_list' : (edge_index, node_position) → integer node indices
        - 'edge_features'  : (edge_index, edge_feature) → optional edge attributes
        - Coordinates:
            * 'edge_index' : unique index per edge
            * 'node_position' : ['src_index', 'dst_index']
            * 'edge_feature' : (if edge features exist)

    Raises
    ------
    ValueError
        If any node label cannot be converted into an integer.
    """
    try:
        # Map node labels to integers (check all)
        nodes = list(graph.nodes)
        node_labels = [int(n) for n in nodes]  # This will raise ValueError if invalid
        node_to_idx = dict(zip(node_labels, nodes))
    except ValueError:
        raise ValueError(
            "All node labels must be convertible to integers (e.g. 0, '1', etc.)"
        )

    if len(nodes) == 0:
        raise ValueError("Graph must contain at least one node.")

    # Build adjacency list
    edges = list(graph.edges)
    adjacency_list = np.array(
        [[node_to_idx[u], node_to_idx[v]] for u, v in edges], dtype=np.int64
    )

    if len(edges) == 0:
        raise ValueError("Graph must contain at least one edge.")

    edge_feature_values = []
    edge_feature_labels = []
    for attr in edge_attrs:
        if attr not in graph.edges[edges[0]]:
            raise MissingEdgeAttributeError(
                f"Missing edge attribute '{attr}' in the graph."
            )
        vals = np.array([graph.edges[e][attr] for e in edges])
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
            edge_feature_labels.append(attr)
        elif vals.ndim == 2:
            # If the attribute is a list, we need to stack them
            edge_feature_labels.extend([f"{attr}_{i}" for i in range(vals.shape[1])])
        else:
            raise ValueError(
                f"Edge attribute '{attr}' has unsupported shape {vals.shape}."
            )
        edge_feature_values.append(vals)

    edge_features = np.concatenate(edge_feature_values, axis=1)

    edge_indexes = np.array(
        [graph.edges[e][edge_id_attr] for e in edges], dtype=np.int64
    )

    ds = xr.Dataset(
        data_vars={
            "adjacency_list": (["edge_index", "node"], adjacency_list),
        },
        coords={
            "edge_index": edge_indexes,
            "node": ["src_index", "dst_index"],
        },
    )

    if edge_features is not None and edge_feature_labels:
        ds["edge_features"] = (["edge_index", "edge_feature"], edge_features)
        ds = ds.assign_coords(edge_feature=edge_feature_labels)

    return ds
