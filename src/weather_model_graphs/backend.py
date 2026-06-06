"""
Backend abstraction layer for flexible graph format support.

Allows seamless conversion between NetworkX, PyTorch Geometric, and DGL formats
for scalable weather model graph processing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
from loguru import logger

try:
    import torch
    import torch_geometric as pyg
    from torch_geometric.data import Data

    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    import dgl
    import dgl.function as fn

    HAS_DGL = True
except ImportError:
    HAS_DGL = False


class GraphBackend(ABC):
    """Abstract base class for graph backends."""

    @abstractmethod
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph."""
        pass

    @abstractmethod
    def to_pyg(self) -> Optional["Data"]:
        """Convert to PyTorch Geometric Data object."""
        pass

    @abstractmethod
    def to_dgl(self) -> Optional["dgl.DGLGraph"]:
        """Convert to DGL graph."""
        pass

    @abstractmethod
    def get_edge_index(self) -> np.ndarray:
        """Get edge indices as (2, num_edges) array."""
        pass

    @abstractmethod
    def get_node_features(self) -> Optional[np.ndarray]:
        """Get node features if available."""
        pass

    @abstractmethod
    def get_edge_features(self) -> Optional[np.ndarray]:
        """Get edge features if available."""
        pass


class NetworkXBackend(GraphBackend):
    """Backend for NetworkX graphs."""

    def __init__(self, graph: nx.DiGraph):
        """
        Initialize NetworkX backend.

        Parameters
        ----------
        graph : nx.DiGraph
            NetworkX directed graph
        """
        self.graph = graph
        self._validate_graph()

    def _validate_graph(self):
        """Validate that the graph has necessary attributes."""
        if not isinstance(self.graph, nx.DiGraph):
            raise ValueError("Graph must be a NetworkX DiGraph")

    def to_networkx(self) -> nx.DiGraph:
        """Return the NetworkX graph."""
        return self.graph

    def to_pyg(self) -> Optional["Data"]:
        """Convert to PyTorch Geometric Data object."""
        if not HAS_PYG:
            logger.warning(
                "PyTorch Geometric not installed. Install with:"
                " pip install weather-model-graphs[pytorch]"
            )
            return None

        # Get edge indices
        edge_index = self.get_edge_index()
        edge_index_tensor = torch.from_numpy(edge_index).long()

        # Get node features
        node_features = self.get_node_features()
        x = None
        if node_features is not None:
            x = torch.from_numpy(node_features).float()

        # Get edge features
        edge_features = self.get_edge_features()
        edge_attr = None
        if edge_features is not None:
            edge_attr = torch.from_numpy(edge_features).float()

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr,
            num_nodes=self.graph.number_of_nodes(),
        )

        return data

    def to_dgl(self) -> Optional["dgl.DGLGraph"]:
        """Convert to DGL graph."""
        if not HAS_DGL:
            logger.warning("DGL not installed. Install with: pip install dgl")
            return None

        # Convert using DGL's from_networkx
        dgl_graph = dgl.from_networkx(self.graph)

        # Add node features if available
        node_features = self.get_node_features()
        if node_features is not None:
            dgl_graph.ndata["x"] = torch.from_numpy(node_features).float()

        # Add edge features if available
        edge_features = self.get_edge_features()
        if edge_features is not None:
            dgl_graph.edata["edge_attr"] = torch.from_numpy(edge_features).float()

        return dgl_graph

    def get_edge_index(self) -> np.ndarray:
        """Get edge indices as (2, num_edges) array."""
        edges = list(self.graph.edges())
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)

        edge_array = np.array(edges, dtype=np.int64)
        return edge_array.T

    def get_node_features(self) -> Optional[np.ndarray]:
        """Extract node features from node attributes."""
        if self.graph.number_of_nodes() == 0:
            return None

        # Try common node feature names
        feature_keys = ["pos", "x", "features"]
        for key in feature_keys:
            try:
                features_dict = nx.get_node_attributes(self.graph, key)
                if features_dict:
                    nodes = sorted(self.graph.nodes())
                    features = np.array([features_dict[node] for node in nodes])
                    if len(features.shape) == 1:
                        features = features.reshape(-1, 1)
                    return features.astype(np.float32)
            except Exception:
                continue

        return None

    def get_edge_features(self) -> Optional[np.ndarray]:
        """Extract edge features from edge attributes."""
        if self.graph.number_of_edges() == 0:
            return None

        # Try common edge feature names
        feature_keys = ["len", "vdiff", "weight", "features"]
        for key in feature_keys:
            try:
                features_dict = nx.get_edge_attributes(self.graph, key)
                if features_dict:
                    edges = sorted(self.graph.edges())
                    features = np.array([features_dict[edge] for edge in edges])
                    if len(features.shape) == 1:
                        features = features.reshape(-1, 1)
                    return features.astype(np.float32)
            except Exception:
                continue

        return None


class PyGBackend(GraphBackend):
    """Backend for PyTorch Geometric graphs."""

    def __init__(self, data: "Data"):
        """
        Initialize PyTorch Geometric backend.

        Parameters
        ----------
        data : torch_geometric.data.Data
            PyTorch Geometric Data object
        """
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric not installed")
        self.data = data

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph."""
        graph = nx.DiGraph()

        # Add nodes
        num_nodes = self.data.num_nodes
        graph.add_nodes_from(range(num_nodes))

        # Add node attributes
        if self.data.x is not None:
            for i in range(num_nodes):
                graph.nodes[i]["x"] = self.data.x[i].cpu().numpy()

        # Add edges
        edges = self.data.edge_index.cpu().numpy()
        for i in range(edges.shape[1]):
            src, dst = edges[0, i], edges[1, i]
            graph.add_edge(int(src), int(dst))

            # Add edge attributes
            if self.data.edge_attr is not None:
                graph[int(src)][int(dst)]["weight"] = self.data.edge_attr[i].cpu().numpy()

        return graph

    def to_pyg(self) -> "Data":
        """Return the PyG Data object."""
        return self.data

    def to_dgl(self) -> Optional["dgl.DGLGraph"]:
        """Convert to DGL graph."""
        if not HAS_DGL:
            logger.warning("DGL not installed. Install with: pip install dgl")
            return None

        # Convert via NetworkX
        nx_graph = self.to_networkx()
        dgl_graph = dgl.from_networkx(nx_graph)

        # Add node features
        if self.data.x is not None:
            dgl_graph.ndata["x"] = self.data.x

        # Add edge features
        if self.data.edge_attr is not None:
            dgl_graph.edata["edge_attr"] = self.data.edge_attr

        return dgl_graph

    def get_edge_index(self) -> np.ndarray:
        """Get edge indices."""
        return self.data.edge_index.cpu().numpy()

    def get_node_features(self) -> Optional[np.ndarray]:
        """Get node features."""
        if self.data.x is not None:
            return self.data.x.cpu().numpy().astype(np.float32)
        return None

    def get_edge_features(self) -> Optional[np.ndarray]:
        """Get edge features."""
        if self.data.edge_attr is not None:
            return self.data.edge_attr.cpu().numpy().astype(np.float32)
        return None


class DGLBackend(GraphBackend):
    """Backend for DGL graphs."""

    def __init__(self, graph: "dgl.DGLGraph"):
        """
        Initialize DGL backend.

        Parameters
        ----------
        graph : dgl.DGLGraph
            DGL graph
        """
        if not HAS_DGL:
            raise ImportError("DGL not installed")
        self.graph = graph

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph."""
        return self.graph.to_networkx().to_directed()

    def to_pyg(self) -> Optional["Data"]:
        """Convert to PyTorch Geometric Data object."""
        if not HAS_PYG:
            logger.warning("PyTorch Geometric not installed")
            return None

        # Convert via NetworkX
        nx_graph = self.to_networkx()
        backend = NetworkXBackend(nx_graph)
        return backend.to_pyg()

    def to_dgl(self) -> "dgl.DGLGraph":
        """Return the DGL graph."""
        return self.graph

    def get_edge_index(self) -> np.ndarray:
        """Get edge indices."""
        src, dst = self.graph.edges()
        edge_index = torch.stack([src, dst])
        return edge_index.cpu().numpy()

    def get_node_features(self) -> Optional[np.ndarray]:
        """Get node features."""
        if "x" in self.graph.ndata:
            return self.graph.ndata["x"].cpu().numpy().astype(np.float32)
        return None

    def get_edge_features(self) -> Optional[np.ndarray]:
        """Get edge features."""
        if "edge_attr" in self.graph.edata:
            return self.graph.edata["edge_attr"].cpu().numpy().astype(np.float32)
        return None


def get_backend(graph: Any) -> GraphBackend:
    """
    Auto-detect graph format and return appropriate backend.

    Parameters
    ----------
    graph : Any
        Graph in NetworkX, PyTorch Geometric, or DGL format

    Returns
    -------
    GraphBackend
        Appropriate backend instance
    """
    if isinstance(graph, nx.DiGraph):
        return NetworkXBackend(graph)
    elif HAS_PYG and isinstance(graph, Data):
        return PyGBackend(graph)
    elif HAS_DGL and isinstance(graph, dgl.DGLGraph):
        return DGLBackend(graph)
    else:
        raise ValueError(
            f"Unsupported graph format: {type(graph)}. "
            "Supported formats: nx.DiGraph, torch_geometric.Data, dgl.DGLGraph"
        )
