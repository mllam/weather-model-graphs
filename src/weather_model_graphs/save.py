import pickle
from pathlib import Path

import networkx

try:
    import torch

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
from loguru import logger

from .networkx_utils import MissingEdgeAttributeError, split_graph_by_edge_attribute

try:
    import torch_geometric.utils.convert as pyg_convert

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def to_pyg(
    graph: networkx.DiGraph,
    output_directory: str,
    name: str,
    edge_features=["vdiff"],
    node_features=["pos"],
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

    if not HAS_PYTORCH or not HAS_PYG:
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

    def _get_edge_indecies(pyg_g):
        return pyg_g.edge_index

    def _get_edge_features(pyg_g):
        if edge_features != ["vdiff"]:
            raise NotImplementedError(edge_features_values)
        # TODO: handle features of different types more generally, i.e. both single ("len") values and tuples (like "vdiff")
        return torch.cat((pyg_g.len.unsqueeze(1), pyg_g.vdiff), dim=1).to(torch.float32)

    def _get_node_features(pyg_g):
        if node_features != ["pos"]:
            raise NotImplementedError(node_features_values)
        return pyg_g.pos.to(torch.float32)

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

    edge_features_values = [_get_edge_features(pyg_g) for pyg_g in pyg_graphs]
    edge_indecies = [_get_edge_indecies(pyg_g) for pyg_g in pyg_graphs]
    node_features_values = [_get_node_features(pyg_g) for pyg_g in pyg_graphs]

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
