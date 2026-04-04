"""
Backend abstraction layer for multi-backend graph processing.

This module provides a unified interface for working with graphs across different
backends: NetworkX (debugging), PyTorch Geometric (training), and DGL (high-performance).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import networkx as nx
import numpy as np


class GraphBackend(ABC):
    """Abstract base class for graph backends."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of the backend."""
        pass

    @abstractmethod
    def create_graph(self, directed: bool = True) -> Any:
        """Create an empty graph."""
        pass

    @abstractmethod
    def add_nodes(self, graph: Any, nodes: List[Any], node_attrs: Optional[Dict[str, Any]] = None) -> None:
        """Add nodes to the graph."""
        pass

    @abstractmethod
    def add_edges(self, graph: Any, edges: List[Tuple[Any, Any]], edge_attrs: Optional[Dict[str, Any]] = None) -> None:
        """Add edges to the graph."""
        pass

    @abstractmethod
    def get_nodes(self, graph: Any) -> List[Any]:
        """Get list of nodes."""
        pass

    @abstractmethod
    def get_edges(self, graph: Any) -> List[Tuple[Any, Any]]:
        """Get list of edges."""
        pass

    @abstractmethod
    def get_node_attrs(self, graph: Any, node: Any) -> Dict[str, Any]:
        """Get attributes of a node."""
        pass

    @abstractmethod
    def get_edge_attrs(self, graph: Any, edge: Tuple[Any, Any]) -> Dict[str, Any]:
        """Get attributes of an edge."""
        pass

    @abstractmethod
    def set_node_attrs(self, graph: Any, node: Any, attrs: Dict[str, Any]) -> None:
        """Set attributes of a node."""
        pass

    @abstractmethod
    def set_edge_attrs(self, graph: Any, edge: Tuple[Any, Any], attrs: Dict[str, Any]) -> None:
        """Set attributes of an edge."""
        pass

    @abstractmethod
    def num_nodes(self, graph: Any) -> int:
        """Get number of nodes."""
        pass

    @abstractmethod
    def num_edges(self, graph: Any) -> int:
        """Get number of edges."""
        pass

    @abstractmethod
    def to_networkx(self, graph: Any) -> nx.Graph:
        """Convert to NetworkX graph."""
        pass

    @abstractmethod
    def from_networkx(self, nx_graph: nx.Graph) -> Any:
        """Convert from NetworkX graph."""
        pass


class NetworkXBackend(GraphBackend):
    """NetworkX backend for graph operations."""

    @property
    def backend_name(self) -> str:
        return "networkx"

    def create_graph(self, directed: bool = True) -> nx.DiGraph:
        if directed:
            return nx.DiGraph()
        else:
            return nx.Graph()

    def add_nodes(self, graph: nx.Graph, nodes: List[Any], node_attrs: Optional[Dict[str, Any]] = None) -> None:
        if node_attrs:
            # Add nodes with attributes
            for node in nodes:
                graph.add_node(node, **node_attrs)
        else:
            graph.add_nodes_from(nodes)

    def add_edges(self, graph: nx.Graph, edges: List[Tuple[Any, Any]], edge_attrs: Optional[Dict[str, Any]] = None) -> None:
        if edge_attrs:
            # Add edges with attributes
            for edge in edges:
                graph.add_edge(edge[0], edge[1], **edge_attrs)
        else:
            graph.add_edges_from(edges)

    def get_nodes(self, graph: nx.Graph) -> List[Any]:
        return list(graph.nodes())

    def get_edges(self, graph: nx.Graph) -> List[Tuple[Any, Any]]:
        return list(graph.edges())

    def get_node_attrs(self, graph: nx.Graph, node: Any) -> Dict[str, Any]:
        return dict(graph.nodes[node])

    def get_edge_attrs(self, graph: nx.Graph, edge: Tuple[Any, Any]) -> Dict[str, Any]:
        return dict(graph.edges[edge])

    def set_node_attrs(self, graph: nx.Graph, node: Any, attrs: Dict[str, Any]) -> None:
        for key, value in attrs.items():
            graph.nodes[node][key] = value

    def set_edge_attrs(self, graph: nx.Graph, edge: Tuple[Any, Any], attrs: Dict[str, Any]) -> None:
        for key, value in attrs.items():
            graph.edges[edge][key] = value

    def num_nodes(self, graph: nx.Graph) -> int:
        return graph.number_of_nodes()

    def num_edges(self, graph: nx.Graph) -> int:
        return graph.number_of_edges()

    def to_networkx(self, graph: nx.Graph) -> nx.Graph:
        return graph.copy()

    def from_networkx(self, nx_graph: nx.Graph) -> nx.Graph:
        return nx_graph.copy()


class PyGBackend(GraphBackend):
    """PyTorch Geometric backend for graph operations."""

    def __init__(self):
        try:
            import torch
            import torch_geometric
            self.torch = torch
            self.torch_geometric = torch_geometric
        except ImportError:
            raise ImportError("PyTorch Geometric backend requires 'torch' and 'torch-geometric' packages. Install with: pip install torch torch-geometric")

    @property
    def backend_name(self) -> str:
        return "pytorch_geometric"

    def create_graph(self, directed: bool = True) -> Any:
        # PyG Data object
        return self.torch_geometric.data.Data()

    def add_nodes(self, graph: Any, nodes: List[Any], node_attrs: Optional[Dict[str, Any]] = None) -> None:
        # In PyG, nodes are typically indexed by integers
        # We'll store node identifiers in node_attrs if provided
        if node_attrs:
            for key, values in node_attrs.items():
                if hasattr(graph, key):
                    # Extend existing attribute
                    existing = getattr(graph, key)
                    if existing is None:
                        setattr(graph, key, self.torch.tensor(values))
                    else:
                        setattr(graph, key, self.torch.cat([existing, self.torch.tensor(values)]))
                else:
                    setattr(graph, key, self.torch.tensor(values))

        # Update num_nodes
        current_nodes = getattr(graph, 'num_nodes', 0)
        graph.num_nodes = current_nodes + len(nodes)

    def add_edges(self, graph: Any, edges: List[Tuple[Any, Any]], edge_attrs: Optional[Dict[str, Any]] = None) -> None:
        # Convert edges to tensor format
        edge_index = self.torch.tensor([[e[0], e[1]] for e in edges], dtype=self.torch.long).t()

        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            # Concatenate with existing edges
            graph.edge_index = self.torch.cat([graph.edge_index, edge_index], dim=1)
        else:
            graph.edge_index = edge_index

        # Handle edge attributes
        if edge_attrs:
            for key, values in edge_attrs.items():
                edge_attr_tensor = self.torch.tensor(values)
                if hasattr(graph, key):
                    existing = getattr(graph, key)
                    if existing is not None:
                        setattr(graph, key, self.torch.cat([existing, edge_attr_tensor]))
                    else:
                        setattr(graph, key, edge_attr_tensor)
                else:
                    setattr(graph, key, edge_attr_tensor)

    def get_nodes(self, graph: Any) -> List[Any]:
        num_nodes = getattr(graph, 'num_nodes', 0)
        return list(range(num_nodes))

    def get_edges(self, graph: Any) -> List[Tuple[Any, Any]]:
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            return [(int(e[0]), int(e[1])) for e in graph.edge_index.t().tolist()]
        return []

    def get_node_attrs(self, graph: Any, node: Any) -> Dict[str, Any]:
        attrs = {}
        for key in dir(graph):
            if not key.startswith('_') and key not in ['edge_index', 'num_nodes']:
                value = getattr(graph, key)
                if isinstance(value, self.torch.Tensor) and len(value.shape) > 0:
                    # Assume node attributes are stored as tensors
                    if value.shape[0] == graph.num_nodes:
                        attrs[key] = value[node].item() if value[node].dim() == 0 else value[node].tolist()
        return attrs

    def get_edge_attrs(self, graph: Any, edge: Tuple[Any, Any]) -> Dict[str, Any]:
        # Find edge index
        edge_index = graph.edge_index
        if edge_index is None:
            return {}

        # Find the position of this edge
        edge_tensor = self.torch.tensor([edge[0], edge[1]], dtype=self.torch.long)
        matches = (edge_index == edge_tensor.unsqueeze(1)).all(0)
        edge_pos = matches.nonzero(as_tuple=True)[0]

        if len(edge_pos) == 0:
            return {}

        edge_pos = edge_pos[0].item()
        attrs = {}

        for key in dir(graph):
            if not key.startswith('_') and key not in ['edge_index', 'num_nodes']:
                value = getattr(graph, key)
                if isinstance(value, self.torch.Tensor) and len(value.shape) > 0:
                    # Check if this is an edge attribute
                    if value.shape[0] == graph.num_edges:
                        attrs[key] = value[edge_pos].item() if value[edge_pos].dim() == 0 else value[edge_pos].tolist()

        return attrs

    def set_node_attrs(self, graph: Any, node: Any, attrs: Dict[str, Any]) -> None:
        for key, value in attrs.items():
            if hasattr(graph, key):
                tensor = getattr(graph, key)
                if isinstance(tensor, self.torch.Tensor):
                    tensor[node] = value if isinstance(value, (int, float)) else self.torch.tensor(value)
                else:
                    setattr(graph, key, self.torch.tensor([value]))
            else:
                # Create new attribute tensor
                attr_tensor = self.torch.zeros(graph.num_nodes, dtype=self.torch.float)
                attr_tensor[node] = value
                setattr(graph, key, attr_tensor)

    def set_edge_attrs(self, graph: Any, edge: Tuple[Any, Any], attrs: Dict[str, Any]) -> None:
        # Find edge position
        edge_index = graph.edge_index
        edge_tensor = self.torch.tensor([edge[0], edge[1]], dtype=self.torch.long)
        matches = (edge_index == edge_tensor.unsqueeze(1)).all(0)
        edge_pos = matches.nonzero(as_tuple=True)[0]

        if len(edge_pos) == 0:
            return

        edge_pos = edge_pos[0].item()

        for key, value in attrs.items():
            if hasattr(graph, key):
                tensor = getattr(graph, key)
                if isinstance(tensor, self.torch.Tensor) and tensor.shape[0] == graph.num_edges:
                    tensor[edge_pos] = value if isinstance(value, (int, float)) else self.torch.tensor(value)
            else:
                # Create new edge attribute tensor
                attr_tensor = self.torch.zeros(graph.num_edges, dtype=self.torch.float)
                attr_tensor[edge_pos] = value
                setattr(graph, key, attr_tensor)

    def num_nodes(self, graph: Any) -> int:
        return getattr(graph, 'num_nodes', 0)

    def num_edges(self, graph: Any) -> int:
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            return graph.edge_index.shape[1]
        return 0

    def to_networkx(self, graph: Any) -> nx.Graph:
        return self.torch_geometric.utils.to_networkx(graph)

    def from_networkx(self, nx_graph: nx.Graph) -> Any:
        # PyG's from_networkx expects all nodes to have the same attributes
        # Let's ensure all nodes have all attributes, filling with defaults
        all_node_attrs = set()
        for node in nx_graph.nodes():
            all_node_attrs.update(nx_graph.nodes[node].keys())
        
        all_edge_attrs = set()
        for edge in nx_graph.edges():
            all_edge_attrs.update(nx_graph.edges[edge].keys())
        
        # Fill missing attributes with None
        for node in nx_graph.nodes():
            for attr in all_node_attrs:
                if attr not in nx_graph.nodes[node]:
                    nx_graph.nodes[node][attr] = None
        
        for edge in nx_graph.edges():
            for attr in all_edge_attrs:
                if attr not in nx_graph.edges[edge]:
                    nx_graph.edges[edge][attr] = None
        
        return self.torch_geometric.utils.from_networkx(nx_graph)


class DGLBackend(GraphBackend):
    """DGL backend for graph operations."""

    def __init__(self):
        try:
            import dgl
            # Ensure DGL uses PyTorch backend
            dgl.set_default_backend('pytorch')
            self.dgl = dgl
        except ImportError:
            raise ImportError("DGL backend requires 'dgl' package. Install with: pip install dgl")

    @property
    def backend_name(self) -> str:
        return "dgl"

    def create_graph(self, directed: bool = True) -> Any:
        return self.dgl.graph([], directed=directed)

    def add_nodes(self, graph: Any, nodes: List[Any], node_attrs: Optional[Dict[str, Any]] = None) -> None:
        # DGL graphs automatically grow when adding edges
        # For now, we'll just store the node count
        current_nodes = graph.num_nodes()
        num_new_nodes = len(nodes)
        graph.add_nodes(num_new_nodes)

        if node_attrs:
            for key, values in node_attrs.items():
                if isinstance(values, (list, np.ndarray)):
                    # Assume values correspond to nodes in order
                    node_data = {key: values}
                    graph.ndata.update(node_data)
                else:
                    # Single value for all nodes
                    node_data = {key: [values] * num_new_nodes}
                    graph.ndata.update(node_data)

    def add_edges(self, graph: Any, edges: List[Tuple[Any, Any]], edge_attrs: Optional[Dict[str, Any]] = None) -> None:
        if not edges:
            return

        src_nodes = [e[0] for e in edges]
        dst_nodes = [e[1] for e in edges]

        graph.add_edges(src_nodes, dst_nodes)

        if edge_attrs:
            for key, values in edge_attrs.items():
                if isinstance(values, (list, np.ndarray)):
                    edge_data = {key: values}
                    graph.edata.update(edge_data)
                else:
                    # Single value for all edges
                    edge_data = {key: [values] * len(edges)}
                    graph.edata.update(edge_data)

    def get_nodes(self, graph: Any) -> List[Any]:
        return list(range(graph.num_nodes()))

    def get_edges(self, graph: Any) -> List[Tuple[Any, Any]]:
        src, dst = graph.edges()
        return list(zip(src.tolist(), dst.tolist()))

    def get_node_attrs(self, graph: Any, node: Any) -> Dict[str, Any]:
        attrs = {}
        for key in graph.ndata:
            value = graph.ndata[key][node]
            if hasattr(value, 'item'):  # Tensor
                attrs[key] = value.item()
            else:
                attrs[key] = value
        return attrs

    def get_edge_attrs(self, graph: Any, edge: Tuple[Any, Any]) -> Dict[str, Any]:
        # Find edge ID
        src, dst = graph.edges()
        edge_ids = ((src == edge[0]) & (dst == edge[1])).nonzero(as_tuple=True)[0]

        if len(edge_ids) == 0:
            return {}

        edge_id = edge_ids[0].item()
        attrs = {}

        for key in graph.edata:
            value = graph.edata[key][edge_id]
            if hasattr(value, 'item'):  # Tensor
                attrs[key] = value.item()
            else:
                attrs[key] = value

        return attrs

    def set_node_attrs(self, graph: Any, node: Any, attrs: Dict[str, Any]) -> None:
        for key, value in attrs.items():
            if key in graph.ndata:
                graph.ndata[key][node] = value
            else:
                # Create new node data
                import torch
                node_data = torch.zeros(graph.num_nodes(), dtype=torch.float)
                node_data[node] = value
                graph.ndata[key] = node_data

    def set_edge_attrs(self, graph: Any, edge: Tuple[Any, Any], attrs: Dict[str, Any]) -> None:
        # Find edge ID
        src, dst = graph.edges()
        edge_ids = ((src == edge[0]) & (dst == edge[1])).nonzero(as_tuple=True)[0]

        if len(edge_ids) == 0:
            return

        edge_id = edge_ids[0].item()

        for key, value in attrs.items():
            if key in graph.edata:
                graph.edata[key][edge_id] = value
            else:
                # Create new edge data
                import torch
                edge_data = torch.zeros(graph.num_edges(), dtype=torch.float)
                edge_data[edge_id] = value
                graph.edata[key] = edge_data

    def num_nodes(self, graph: Any) -> int:
        return graph.num_nodes()

    def num_edges(self, graph: Any) -> int:
        return graph.num_edges()

    def to_networkx(self, graph: Any) -> nx.Graph:
        return graph.to_networkx()

    def from_networkx(self, nx_graph: nx.Graph) -> Any:
        return self.dgl.from_networkx(nx_graph)


# Global backend instances
_NETWORKX_BACKEND = NetworkXBackend()
_PYG_BACKEND = None
_DGL_BACKEND = None

def get_backend(backend_name: Optional[str] = None, graph: Optional[Any] = None) -> GraphBackend:
    """
    Get a graph backend instance.

    Parameters
    ----------
    backend_name : str, optional
        Name of the backend ('networkx', 'pytorch_geometric', 'dgl').
        If None, attempts automatic detection from the graph object.
    graph : Any, optional
        Graph object to detect backend from. Only used if backend_name is None.

    Returns
    -------
    GraphBackend
        Backend instance
    """
    global _PYG_BACKEND, _DGL_BACKEND

    if backend_name is None and graph is not None:
        # Automatic detection
        backend_name = _detect_backend(graph)

    if backend_name == 'networkx' or backend_name is None:
        return _NETWORKX_BACKEND
    elif backend_name == 'pytorch_geometric':
        if _PYG_BACKEND is None:
            _PYG_BACKEND = PyGBackend()
        return _PYG_BACKEND
    elif backend_name == 'dgl':
        if _DGL_BACKEND is None:
            _DGL_BACKEND = DGLBackend()
        return _DGL_BACKEND
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def _detect_backend(graph: Any) -> str:
    """Detect backend from graph object."""
    if hasattr(graph, 'nodes') and hasattr(graph, 'edges') and hasattr(graph, 'add_node'):
        # NetworkX graph
        return 'networkx'
    elif hasattr(graph, 'edge_index') and hasattr(graph, 'num_nodes'):
        # PyTorch Geometric Data
        return 'pytorch_geometric'
    elif hasattr(graph, 'num_nodes') and hasattr(graph, 'num_edges') and hasattr(graph, 'ndata'):
        # DGL graph
        return 'dgl'
    else:
        warnings.warn(f"Could not detect backend for graph type {type(graph)}, defaulting to networkx")
        return 'networkx'


def convert_graph(graph: Any, target_backend: str) -> Any:
    """
    Convert a graph between backends.

    Parameters
    ----------
    graph : Any
        Input graph
    target_backend : str
        Target backend name

    Returns
    -------
    Any
        Graph in target backend format
    """
    source_backend = get_backend(graph=graph)
    if source_backend.backend_name == target_backend:
        return graph

    # Convert to NetworkX first
    nx_graph = source_backend.to_networkx(graph)

    # Then convert to target backend
    target_backend_instance = get_backend(target_backend)
    return target_backend_instance.from_networkx(nx_graph)