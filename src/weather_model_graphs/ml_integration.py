"""
Integration with ML frameworks (PyTorch, TensorFlow).

Provides utilities for creating data loaders and batching graphs
for training machine learning models.
"""

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from loguru import logger

try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Dummy classes when torch is not available
    class Dataset:
        pass
    DataLoader = None

try:
    import torch_geometric as pyg
    from torch_geometric.data import Data, DataLoader as PyGDataLoader

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    Data = None
    PyGDataLoader = None


class GraphDataset(Dataset):
    """PyTorch Dataset for graph data."""

    def __init__(
        self,
        graphs: List[nx.DiGraph],
        labels: Optional[np.ndarray] = None,
        node_features: Optional[List[str]] = None,
        edge_features: Optional[List[str]] = None,
    ):
        """
        Initialize graph dataset.

        Parameters
        ----------
        graphs : List[nx.DiGraph]
            List of NetworkX graphs
        labels : np.ndarray, optional
            Target labels
        node_features : List[str], optional
            Node feature attribute names
        edge_features : List[str], optional
            Edge feature attribute names
        """
        self.graphs = graphs
        self.labels = labels
        self.node_features = node_features or ["pos"]
        self.edge_features = edge_features or ["len"]

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get single sample.

        Returns
        -------
        sample : tuple
            (graph, label) if labels provided, else (graph,)
        """
        graph = self.graphs[idx]

        if self.labels is not None:
            return graph, self.labels[idx]
        return (graph,)


class PyGGraphDataset(Dataset):
    """PyTorch Geometric Dataset for graph data."""

    def __init__(
        self,
        graphs: List[nx.DiGraph],
        node_features: Optional[List[str]] = None,
        edge_features: Optional[List[str]] = None,
        labels: Optional[np.ndarray] = None,
    ):
        """Initialize PyG dataset."""
        self.graphs = graphs
        self.node_features = node_features or ["pos"]
        self.edge_features = edge_features or ["len"]
        self.labels = labels

        self.data_list = [self._graph_to_pyg(g) for g in graphs]

    def _graph_to_pyg(self, graph: nx.DiGraph) -> Data:
        """Convert NetworkX graph to PyG Data object."""
        from .backend import NetworkXBackend

        backend = NetworkXBackend(graph)
        data = backend.to_pyg()

        return data

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        """Get single sample."""
        return self.data_list[idx]


def create_dataloader(
    graphs: List[nx.DiGraph],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    backend: str = "networkx",
    labels: Optional[np.ndarray] = None,
    **kwargs
) -> DataLoader:
    """
    Create PyTorch DataLoader for graphs.

    Parameters
    ----------
    graphs : List[nx.DiGraph]
        List of graphs
    batch_size : int
        Batch size
    shuffle : bool
        Shuffle data
    num_workers : int
        Number of data loading workers
    backend : str
        "networkx" or "pyg"
    labels : np.ndarray, optional
        Target labels
    **kwargs
        Additional arguments to DataLoader

    Returns
    -------
    dataloader : DataLoader
        PyTorch DataLoader
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not installed. Install with: pip install torch")

    if backend == "networkx":
        dataset = GraphDataset(graphs, labels=labels)
    elif backend == "pyg":
        if not HAS_PYG:
            raise ImportError(
                "PyTorch Geometric not installed. "
                "Install with: pip install torch-geometric"
            )
        dataset = PyGGraphDataset(graphs, labels=labels)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )


def create_pyg_dataloader(
    graphs: List[nx.DiGraph],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> "PyGDataLoader":
    """
    Create PyTorch Geometric DataLoader.

    Parameters
    ----------
    graphs : List[nx.DiGraph]
        List of graphs
    batch_size : int
        Batch size
    shuffle : bool
        Shuffle data
    num_workers : int
        Number of workers
    **kwargs
        Additional arguments

    Returns
    -------
    dataloader : torch_geometric.data.DataLoader
        PyG DataLoader
    """
    if not HAS_PYG:
        raise ImportError("PyTorch Geometric not installed")

    dataset = PyGGraphDataset(graphs)
    return PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )


def batch_graphs(
    graphs: List[nx.DiGraph],
    node_feature_keys: Optional[List[str]] = None,
    edge_feature_keys: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Batch multiple graphs into tensor format.

    Parameters
    ----------
    graphs : List[nx.DiGraph]
        List of graphs
    node_feature_keys : List[str], optional
        Node feature keys to extract
    edge_feature_keys : List[str], optional
        Edge feature keys to extract

    Returns
    -------
    node_features : np.ndarray
        (total_nodes, n_features)
    edge_indices : List[np.ndarray]
        List of edge index arrays with graph offsets
    edge_features : List[np.ndarray]
        List of edge feature arrays
    """
    if node_feature_keys is None:
        node_feature_keys = ["pos"]
    if edge_feature_keys is None:
        edge_feature_keys = ["len"]

    all_node_features = []
    all_edge_indices = []
    all_edge_features = []

    node_offset = 0

    for graph in graphs:
        # Extract node features
        nodes = sorted(graph.nodes())
        node_feats = []

        for node in nodes:
            feats = []
            for key in node_feature_keys:
                if key in graph.nodes[node]:
                    val = graph.nodes[node][key]
                    if isinstance(val, np.ndarray):
                        feats.extend(val.flatten().tolist())
                    elif isinstance(val, (list, tuple)):
                        feats.extend(val)
                    else:
                        feats.append(val)
            if feats:
                node_feats.append(feats)

        if node_feats:
            all_node_features.extend(node_feats)

        # Extract edges with offset
        edges = list(graph.edges(data=True))
        if edges:
            edge_indices = np.array([[u + node_offset, v + node_offset] for u, v, _ in edges]).T
            all_edge_indices.append(edge_indices)

            # Extract edge features
            edge_feats = []
            for _, _, data in edges:
                feats = []
                for key in edge_feature_keys:
                    if key in data:
                        val = data[key]
                        if isinstance(val, (int, float, np.number)):
                            feats.append(val)
                        elif isinstance(val, (list, tuple)):
                            feats.extend(val)
                if feats:
                    edge_feats.append(feats)
            if edge_feats:
                all_edge_features.append(np.array(edge_feats))

        node_offset += len(nodes)

    result_node_features = (
        np.array(all_node_features) if all_node_features else np.array([])
    )
    return result_node_features, all_edge_indices, all_edge_features


def create_model_input(
    graph: nx.DiGraph,
    backend: str = "networkx",
    node_features: Optional[List[str]] = None,
    edge_features: Optional[List[str]] = None,
):
    """
    Create model input from graph.

    Parameters
    ----------
    graph : nx.DiGraph
        Input graph
    backend : str
        "networkx", "pyg", or "dgl"
    node_features : List[str], optional
        Node feature keys
    edge_features : List[str], optional
        Edge feature keys

    Returns
    -------
    model_input
        Format depends on backend
    """
    from .backend import get_backend

    graph_backend = get_backend(graph)

    if backend == "networkx":
        return graph

    elif backend == "pyg":
        return graph_backend.to_pyg()

    elif backend == "dgl":
        return graph_backend.to_dgl()

    else:
        raise ValueError(f"Unknown backend: {backend}")


def split_graph_for_training(
    graph: nx.DiGraph,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split nodes into train/val/test sets.

    Parameters
    ----------
    graph : nx.DiGraph
        Input graph
    train_size : float
        Training set fraction
    val_size : float
        Validation set fraction

    Returns
    -------
    train_nodes : List[int]
        Training node indices
    val_nodes : List[int]
        Validation node indices
    test_nodes : List[int]
        Test node indices
    """
    nodes = list(graph.nodes())
    n_nodes = len(nodes)

    n_train = int(train_size * n_nodes)
    n_val = int(val_size * n_nodes)

    # Random split
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_nodes)

    train_nodes = [nodes[i] for i in indices[:n_train]]
    val_nodes = [nodes[i] for i in indices[n_train : n_train + n_val]]
    test_nodes = [nodes[i] for i in indices[n_train + n_val :]]

    return train_nodes, val_nodes, test_nodes
