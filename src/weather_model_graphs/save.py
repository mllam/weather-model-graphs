import torch
from pathlib import Path
import networkx
import pickle
from loguru import logger

from .networkx_utils import split_graph_by_edge_attribute

try:
    import torch_geometric.utils.convert as pyg_convert
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def save_edges(graph, name, base_path):
    torch.save(
        graph.edge_index, os.path.join(base_path, f"{name}_edge_index.pt")
    )
    edge_features = torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(
        torch.float32
    )  # Save as float32
    torch.save(edge_features, os.path.join(base_path, f"{name}_features.pt"))


def save_edges_list(graphs, name, base_path):
    torch.save(
        [graph.edge_index for graph in graphs],
        os.path.join(base_path, f"{name}_edge_index.pt"),
    )
    edge_features = [
        torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(
            torch.float32
        )
        for graph in graphs
    ]  # Save as float32
    torch.save(edge_features, os.path.join(base_path, f"{name}_features.pt"))

def to_pyg(graph: networkx.DiGraph, output_directory: str, name: str, list_from_attribute=None):
    """
    Save the networkx graph to PyTorch Geometric format.
    """
    if name is None:
        raise ValueError("Name must be provided.")
        
    # check that the node labels are integers and unique so that they can be used as indices
    if not all(isinstance(node, int) for node in graph.nodes):
        node_types = set([type(node) for node in graph.nodes])
        raise ValueError(
            f"Node labels must be integers. Instead they are of types {node_types}."
        )
    if len(set(graph.nodes)) != len(graph.nodes):
        raise ValueError("Node labels must be unique.")
        
    def _get_edge_indecies(pyg_g):
        return pyg_g.edge_index
    
    def _get_edge_features(pyg_g):
        return torch.cat((pyg_g.len.unsqueeze(1), pyg_g.vdiff), dim=1).to(torch.float32)
        
    if list_from_attribute is not None:
        # create a list of graph objects by splitting the graph by the list_from_attribute
        sub_graphs = split_graph_by_edge_attribute(graph=graph, attribute=list_from_attribute)
        pyg_graphs = [pyg_convert.from_networkx(g) for g in sub_graphs]
    else:
        pyg_graphs = [pyg_convert.from_networkx(graph)]
        
    edge_features = [_get_edge_features(pyg_g) for pyg_g in pyg_graphs]
    edge_indecies = [_get_edge_indecies(pyg_g) for pyg_g in pyg_graphs]
    
    if len(pyg_graphs) == 1:
        edge_features = edge_features[0]
        edge_indecies = edge_indecies[0]

    fp_edge_index = Path(output_directory) / f"{name}_edge_index.pt"
    fp_features = Path(output_directory) / f"{name}_features.pt"
    torch.save(edge_indecies, fp_edge_index)
    torch.save(edge_features, fp_features)
    logger.info(f"Saved edge index and features to {fp_edge_index} and {fp_features}.")
    

def to_pickle(graph: networkx.DiGraph, output_directory: str, name: str):
    """
    Save the networkx graph to a pickle file.
    """
    fp = Path(output_directory) / f"{name}.pickle"
    with open(fp, "wb") as f:
        pickle.dump(graph, f)
    logger.info(f"Saved graph to {fp}.")