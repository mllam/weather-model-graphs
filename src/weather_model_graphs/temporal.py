"""
Temporal graph support for time-series weather model predictions.

Enables dynamic graphs with temporal edges connecting graph states
across timesteps, enabling recurrent neural network architectures
and autoregressive weather prediction models.
"""

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from loguru import logger


class TemporalGraph:
    """
    Representation of a temporal graph with static and dynamic edges.

    Supports creating dynamic graphs where nodes can have temporal connections
    to represent weather evolution over time.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        timesteps: int = 1,
        temporal_window: int = 1,
    ):
        """
        Initialize temporal graph.

        Parameters
        ----------
        graph : nx.DiGraph
            Single timestep graph structure
        timesteps : int
            Number of timesteps to unroll
        temporal_window : int
            Number of past timesteps to connect to (temporal edges)
        """
        self.base_graph = graph
        self.timesteps = timesteps
        self.temporal_window = temporal_window
        self.graphs: Dict[int, nx.DiGraph] = {}

        if timesteps > 0:
            self._create_temporal_unroll()

    def _create_temporal_unroll(self):
        """Create unrolled temporal graphs with time edges."""
        logger.debug(f"Creating temporal unroll for {self.timesteps} timesteps")

        # Create a graph for each timestep
        for t in range(self.timesteps):
            G_t = nx.DiGraph()

            # Add nodes for this timestep with temporal index
            for node in self.base_graph.nodes():
                node_id = self._get_temporal_node_id(node, t)
                G_t.add_node(node_id, **self.base_graph.nodes[node])
                G_t.nodes[node_id]["timestep"] = t

            # Add spatial edges (same as base graph)
            for u, v, data in self.base_graph.edges(data=True):
                u_id = self._get_temporal_node_id(u, t)
                v_id = self._get_temporal_node_id(v, t)
                G_t.add_edge(u_id, v_id, **data)
                G_t[u_id][v_id]["edge_type"] = "spatial"

            # Add temporal edges to previous timesteps
            if t > 0:
                for node in self.base_graph.nodes():
                    for tau in range(1, min(self.temporal_window + 1, t + 1)):
                        src_id = self._get_temporal_node_id(node, t - tau)
                        dst_id = self._get_temporal_node_id(node, t)
                        G_t.add_edge(src_id, dst_id, lag=tau, edge_type="temporal")

            self.graphs[t] = G_t

    def _get_temporal_node_id(self, node_id: int, timestep: int) -> int:
        """Create unique ID for node at specific timestep."""
        num_nodes = len(self.base_graph.nodes())
        return node_id + timestep * num_nodes

    def get_graph(self, timestep: int) -> nx.DiGraph:
        """Get graph for specific timestep."""
        if timestep not in self.graphs:
            raise ValueError(f"Timestep {timestep} not in graph")
        return self.graphs[timestep]

    def get_combined_graph(self) -> nx.DiGraph:
        """Get all timesteps combined into single graph."""
        combined = nx.DiGraph()

        # Merge all timestep graphs
        for t in range(self.timesteps):
            combined = nx.compose(combined, self.graphs[t])

        return combined

    def get_edges_by_type(self, edge_type: str = "spatial") -> List[Tuple]:
        """
        Get edges of specific type across all timesteps.

        Parameters
        ----------
        edge_type : str
            "spatial" or "temporal"

        Returns
        -------
        edges : List[Tuple]
            List of (source, target) edge pairs
        """
        edges = []
        combined = self.get_combined_graph()

        for u, v, data in combined.edges(data=True):
            if data.get("edge_type") == edge_type:
                edges.append((u, v))

        return edges

    def get_statistics(self) -> Dict:
        """Get statistics about temporal graph."""
        combined = self.get_combined_graph()
        spatial_edges = len(self.get_edges_by_type("spatial"))
        temporal_edges = len(self.get_edges_by_type("temporal"))

        return {
            "timesteps": self.timesteps,
            "temporal_window": self.temporal_window,
            "nodes_per_step": len(self.base_graph.nodes()),
            "total_nodes": combined.number_of_nodes(),
            "spatial_edges": spatial_edges,
            "temporal_edges": temporal_edges,
            "total_edges": combined.number_of_edges(),
        }


def create_temporal_graph(
    coords: np.ndarray,
    timesteps: int = 10,
    temporal_window: int = 2,
    connectivity: str = "nearest_neighbour",
    connectivity_kwargs: Optional[Dict] = None,
) -> TemporalGraph:
    """
    Create a temporal graph from coordinates.

    Parameters
    ----------
    coords : np.ndarray
        (N, 2) or (N, 3) coordinate array
    timesteps : int
        Number of timesteps to unroll
    temporal_window : int
        Number of past timesteps to include in temporal edges
    connectivity : str
        Type of spatial connectivity ("nearest_neighbour", etc.)
    connectivity_kwargs : dict, optional
        Keyword arguments for connectivity method

    Returns
    -------
    TemporalGraph
        Temporal graph object
    """
    # Create base spatial graph
    from .create.base import create_all_graph_components

    if connectivity_kwargs is None:
        connectivity_kwargs = {}

    # Create a simple base graph (grid or mesh)
    base_graph = _create_spatial_base_graph(coords, connectivity, connectivity_kwargs)

    # Wrap in TemporalGraph
    temporal_graph = TemporalGraph(
        base_graph, timesteps=timesteps, temporal_window=temporal_window
    )

    logger.info(
        f"Created temporal graph with {timesteps} timesteps, "
        f"window size {temporal_window}, {base_graph.number_of_nodes()} nodes"
    )

    return temporal_graph


def _create_spatial_base_graph(coords: np.ndarray, connectivity: str, kwargs: Dict):
    """Create base spatial graph for temporal graph."""
    # Simple spatial graph creation (can be extended)
    G = nx.DiGraph()

    # Add nodes
    for i, coord in enumerate(coords):
        G.add_node(i, pos=coord)

    # Add edges based on connectivity
    if connectivity == "nearest_neighbour":
        max_neighbors = kwargs.get("max_neighbors", 4)
        from .spatial_index import find_neighbors_vectorized

        neighbors = find_neighbors_vectorized(
            coords, coords, max_neighbors=max_neighbors
        )
        for i, neighbor_list in enumerate(neighbors):
            for j in neighbor_list:
                if i != j:
                    dist = np.linalg.norm(coords[i] - coords[j])
                    G.add_edge(i, j, len=dist)

    return G


def add_temporal_edges_to_graph(
    graph: nx.DiGraph,
    num_nodes_per_step: int,
    num_timesteps: int,
    temporal_window: int = 1,
) -> nx.DiGraph:
    """
    Add temporal edges to an existing unrolled spatial graph.

    Parameters
    ----------
    graph : nx.DiGraph
        Graph with time-unrolled nodes
    num_nodes_per_step : int
        Number of nodes per timestep
    num_timesteps : int
        Total number of timesteps
    temporal_window : int
        How many previous timesteps to connect

    Returns
    -------
    graph : nx.DiGraph
        Graph with temporal edges added
    """
    for t in range(1, num_timesteps):
        for node_idx in range(num_nodes_per_step):
            for lag in range(1, min(temporal_window + 1, t + 1)):
                src_node = node_idx + (t - lag) * num_nodes_per_step
                dst_node = node_idx + t * num_nodes_per_step

                if src_node in graph.nodes() and dst_node in graph.nodes():
                    graph.add_edge(src_node, dst_node, lag=lag, edge_type="temporal")

    return graph


def unfold_temporal_predictions(
    predictions: np.ndarray,
    num_timesteps: int,
    num_nodes: int,
) -> Dict[int, np.ndarray]:
    """
    Unfold temporal predictions for visualization/analysis.

    Parameters
    ----------
    predictions : np.ndarray
        (num_timesteps * num_nodes, features) predictions
    num_timesteps : int
        Number of timesteps
    num_nodes : int
        Number of nodes per timestep

    Returns
    -------
    unfolded : Dict[int, np.ndarray]
        Timestep -> node predictions
    """
    unfolded = {}
    for t in range(num_timesteps):
        start_idx = t * num_nodes
        end_idx = (t + 1) * num_nodes
        unfolded[t] = predictions[start_idx:end_idx]

    return unfolded
