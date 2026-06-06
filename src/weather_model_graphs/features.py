"""
Feature engineering utilities for weather model graphs.

Provides tools for adding computed features like wind velocity,
pressure gradients, and temporal encodings to make graphs
ML-ready for neural network training.
"""

from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np
from loguru import logger


class FeatureExtractor:
    """Extract and engineer features for graph nodes and edges."""

    @staticmethod
    def add_wind_velocity(
        graph: nx.DiGraph, u_attr: str = "u", v_attr: str = "v"
    ) -> nx.DiGraph:
        """
        Add wind velocity magnitude as node feature.

        Parameters
        ----------
        graph : nx.DiGraph
            Input graph
        u_attr : str
            Name of u-component attribute
        v_attr : str
            Name of v-component attribute

        Returns
        -------
        graph : nx.DiGraph
            Graph with wind_velocity feature added
        """
        for node in graph.nodes():
            node_data = graph.nodes[node]
            if u_attr in node_data and v_attr in node_data:
                u = node_data[u_attr]
                v = node_data[v_attr]
                velocity = np.sqrt(u**2 + v**2)
                graph.nodes[node]["wind_velocity"] = velocity

        logger.info("Added wind_velocity feature to nodes")
        return graph

    @staticmethod
    def add_wind_direction(
        graph: nx.DiGraph, u_attr: str = "u", v_attr: str = "v"
    ) -> nx.DiGraph:
        """
        Add wind direction (angle) as node feature.

        Parameters
        ----------
        graph : nx.DiGraph
            Input graph
        u_attr : str
            Name of u-component attribute
        v_attr : str
            Name of v-component attribute

        Returns
        -------
        graph : nx.DiGraph
            Graph with wind_direction feature added
        """
        for node in graph.nodes():
            node_data = graph.nodes[node]
            if u_attr in node_data and v_attr in node_data:
                u = node_data[u_attr]
                v = node_data[v_attr]
                direction = np.arctan2(v, u)
                graph.nodes[node]["wind_direction"] = direction

        logger.info("Added wind_direction feature to nodes")
        return graph

    @staticmethod
    def add_pressure_gradient(
        graph: nx.DiGraph, pressure_attr: str = "pressure"
    ) -> nx.DiGraph:
        """
        Add pressure gradient magnitude as edge feature.

        Parameters
        ----------
        graph : nx.DiGraph
            Input graph
        pressure_attr : str
            Name of pressure attribute

        Returns
        -------
        graph : nx.DiGraph
            Graph with pressure_gradient feature
        """
        for u, v, data in graph.edges(data=True):
            if pressure_attr in graph.nodes[u] and pressure_attr in graph.nodes[v]:
                p_u = graph.nodes[u][pressure_attr]
                p_v = graph.nodes[v][pressure_attr]
                gradient = abs(p_v - p_u)
                if "len" in data:
                    gradient /= data["len"]
                graph[u][v]["pressure_gradient"] = gradient

        logger.info("Added pressure_gradient feature to edges")
        return graph

    @staticmethod
    def add_temporal_encoding(
        graph: nx.DiGraph,
        max_period: int = 24,
        num_frequencies: int = 8,
    ) -> nx.DiGraph:
        """
        Add temporal positional encoding (sinusoidal).

        Parameters
        ----------
        graph : nx.DiGraph
            Input graph
        max_period : int
            Maximum period for encoding (hours)
        num_frequencies : int
            Number of frequency components

        Returns
        -------
        graph : nx.DiGraph
            Graph with temporal_encoding features
        """
        # Create frequency bands
        frequencies = np.linspace(0, 1, num_frequencies)
        periods = max_period * (2.0 ** frequencies)
        timestep = 0  # Can be parameterized

        temporal_encoding = []
        for period in periods:
            temporal_encoding.append(np.sin(2 * np.pi * timestep / period))
            temporal_encoding.append(np.cos(2 * np.pi * timestep / period))

        encoding = np.array(temporal_encoding)

        # Add to each node (shared across all nodes at same timestep)
        for node in graph.nodes():
            graph.nodes[node]["temporal_encoding"] = encoding

        logger.info(f"Added temporal_encoding ({len(encoding)} dims) to nodes")
        return graph

    @staticmethod
    def add_spatial_encoding(
        graph: nx.DiGraph,
        num_frequencies: int = 8,
        scale: float = 1.0,
    ) -> nx.DiGraph:
        """
        Add spatial positional encoding based on coordinates.

        Parameters
        ----------
        graph : nx.DiGraph
            Input graph
        num_frequencies : int
            Number of frequency components
        scale : float
            Scaling factor for coordinates

        Returns
        -------
        graph : nx.DiGraph
            Graph with spatial_encoding features
        """
        # Get coordinate bounds
        pos_data = []
        for node in graph.nodes():
            if "pos" in graph.nodes[node]:
                pos_data.append(graph.nodes[node]["pos"])

        if not pos_data:
            logger.warning("No position data found in graph nodes")
            return graph

        pos_array = np.array(pos_data)
        pos_min = pos_array.min(axis=0)
        pos_max = pos_array.max(axis=0)
        pos_range = pos_max - pos_min + 1e-8

        # Create frequency bands
        frequencies = np.logspace(-1, 1, num_frequencies)

        for node in graph.nodes():
            if "pos" in graph.nodes[node]:
                pos = (graph.nodes[node]["pos"] - pos_min) / pos_range * scale
                encoding = []

                for dim in range(len(pos)):
                    for freq in frequencies:
                        encoding.append(np.sin(2 * np.pi * freq * pos[dim]))
                        encoding.append(np.cos(2 * np.pi * freq * pos[dim]))

                graph.nodes[node]["spatial_encoding"] = np.array(encoding)

        logger.info(f"Added spatial_encoding to {graph.number_of_nodes()} nodes")
        return graph

    @staticmethod
    def add_node_degree_features(graph: nx.DiGraph) -> nx.DiGraph:
        """
        Add in-degree and out-degree as node features.

        Parameters
        ----------
        graph : nx.DiGraph
            Input graph

        Returns
        -------
        graph : nx.DiGraph
            Graph with degree features
        """
        for node in graph.nodes():
            graph.nodes[node]["in_degree"] = graph.in_degree(node)
            graph.nodes[node]["out_degree"] = graph.out_degree(node)

        logger.info("Added degree features to nodes")
        return graph

    @staticmethod
    def normalize_features(
        graph: nx.DiGraph,
        feature_keys: Optional[List[str]] = None,
        method: str = "minmax",
    ) -> nx.DiGraph:
        """
        Normalize node features.

        Parameters
        ----------
        graph : nx.DiGraph
            Input graph
        feature_keys : List[str], optional
            Features to normalize. If None, normalize all numeric features.
        method : str
            "minmax" (0-1) or "zscore" (mean=0, std=1)

        Returns
        -------
        graph : nx.DiGraph
            Graph with normalized features
        """
        if feature_keys is None:
            # Auto-detect numeric features
            feature_keys = set()
            for node in graph.nodes():
                for key, val in graph.nodes[node].items():
                    if isinstance(val, (int, float, np.number)):
                        feature_keys.add(key)

        for feature_key in feature_keys:
            # Collect all values
            values = []
            for node in graph.nodes():
                if feature_key in graph.nodes[node]:
                    val = graph.nodes[node][feature_key]
                    if isinstance(val, (int, float, np.number)):
                        values.append(val)

            if not values:
                continue

            values = np.array(values)

            if method == "minmax":
                v_min, v_max = values.min(), values.max()
                if v_max > v_min:
                    normalized = (values - v_min) / (v_max - v_min)
                else:
                    normalized = np.zeros_like(values)
            elif method == "zscore":
                mean, std = values.mean(), values.std()
                if std > 0:
                    normalized = (values - mean) / std
                else:
                    normalized = np.zeros_like(values)
            else:
                raise ValueError(f"Unknown normalization method: {method}")

            # Update graph
            value_dict = {node: val for node, val in zip(
                [n for n in graph.nodes() if feature_key in graph.nodes[n]],
                normalized
            )}
            for node, val in value_dict.items():
                graph.nodes[node][feature_key] = val

        logger.info(f"Normalized {len(feature_keys)} features using {method}")
        return graph


# Convenience functions
def add_wind_velocity(graph: nx.DiGraph, **kwargs) -> nx.DiGraph:
    """Add wind velocity feature."""
    return FeatureExtractor.add_wind_velocity(graph, **kwargs)


def add_wind_direction(graph: nx.DiGraph, **kwargs) -> nx.DiGraph:
    """Add wind direction feature."""
    return FeatureExtractor.add_wind_direction(graph, **kwargs)


def add_pressure_gradient(graph: nx.DiGraph, **kwargs) -> nx.DiGraph:
    """Add pressure gradient feature."""
    return FeatureExtractor.add_pressure_gradient(graph, **kwargs)


def add_temporal_encoding(graph: nx.DiGraph, **kwargs) -> nx.DiGraph:
    """Add temporal encoding."""
    return FeatureExtractor.add_temporal_encoding(graph, **kwargs)


def add_spatial_encoding(graph: nx.DiGraph, **kwargs) -> nx.DiGraph:
    """Add spatial encoding."""
    return FeatureExtractor.add_spatial_encoding(graph, **kwargs)


def add_node_degree_features(graph: nx.DiGraph) -> nx.DiGraph:
    """Add degree features."""
    return FeatureExtractor.add_node_degree_features(graph)


def normalize_features(graph: nx.DiGraph, **kwargs) -> nx.DiGraph:
    """Normalize features."""
    return FeatureExtractor.normalize_features(graph, **kwargs)
