from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import networkx
from loguru import logger

from ...networkx_utils import (
    MissingEdgeAttributeError,
    sort_nodes_in_graph,
    split_graph_by_edge_attribute,
)
from ..base import DEFAULT_EDGE_FEATURES, DEFAULT_NODE_FEATURES, HAS_PYG

if HAS_PYG:
    import torch
    import torch_geometric as pyg
    import torch_geometric.utils.convert as pyg_convert


def to_pyg(
    graph: networkx.DiGraph,
    output_directory: str,
    name: str,
    edge_features: Tuple[str, ...] = DEFAULT_EDGE_FEATURES,
    node_features: Tuple[str, ...] = DEFAULT_NODE_FEATURES,
    list_from_attribute: Optional[str] = None,
) -> None:
    """Save the networkx graph to a PyTorch Geometric format on disk.

    .. deprecated::
        This function is deprecated and will no longer be maintained. Use
        :func:`weather_model_graphs.save.to_torch_tensors_on_disk` instead,
        which writes graphs in the neural-lam graph storage format.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph to save.
    output_directory : str
        Directory to save the graph to.
    name : str
        Name of the graph, used to name the files. The edge index and
        features are saved to ``{output_directory}/{name}_edge_index.pt`` and
        ``{output_directory}/{name}_features.pt`` respectively.
    edge_features : tuple of str, optional
        Edge attributes to include in the ``{name}_features.pt`` file.
        Default: ``DEFAULT_EDGE_FEATURES``.
    node_features : tuple of str, optional
        Node attributes to include in the ``{name}_node_features.pt`` file.
        Default: ``DEFAULT_NODE_FEATURES``.
    list_from_attribute : str, optional
        If provided, the graph is split by this edge attribute value. The
        stored edge index and features are then the concatenation of the
        split graphs, so a separate pyg.Data object can be created for each
        subgraph (e.g. one per level in a multi-level graph). Default: None.

    Returns
    -------
    None
    """
    warnings.warn(
        "weather_model_graphs.save.to_pyg is deprecated and will no longer be "
        "maintained. Use weather_model_graphs.save.to_torch_tensors_on_disk "
        "instead, which writes graphs in the neural-lam graph storage format.",
        DeprecationWarning,
        stacklevel=2,
    )

    if name is None:
        raise ValueError("Name must be provided.")

    if not HAS_PYG:
        raise Exception(
            "install weather-mode-graphs[pytorch] to enable writing to torch files"
        )

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

    def _get_edge_indecies(pyg_g: "pyg.data.Data") -> "torch.Tensor":
        """Return the edge-index tensor of a pyg.Data object.

        Parameters
        ----------
        pyg_g : torch_geometric.data.Data
            Graph to read the edge index from.

        Returns
        -------
        torch.Tensor
            The ``edge_index`` tensor of shape ``(2, num_edges)``.
        """
        return pyg_g.edge_index

    def _concat_pyg_features(
        pyg_g: "pyg.data.Data", features: List[str]
    ) -> "torch.Tensor":
        """Concatenate named features from a pyg.Data object into a 2D tensor.

        Handles both node and edge features. Each named feature becomes one
        or more columns of the resulting 2D tensor of shape
        ``(n_edges_or_nodes, n_feature_columns)``.

        Parameters
        ----------
        pyg_g : torch_geometric.data.Data
            Graph to read the features from.
        features : list of str
            Names of the attributes to concatenate, in order.

        Returns
        -------
        torch.Tensor
            The concatenated features as a float32 tensor.
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
