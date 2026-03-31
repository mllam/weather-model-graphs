"""
Configuration-driven graph creation pipeline.

Enables declarative YAML-based graph definition for reproducible,
shareable weather model graph architectures.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
from loguru import logger

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class GraphConfig:
    """Configuration for graph creation."""

    graph_type: str
    grid_size: int
    mesh_distance: float
    temporal_steps: int = 1
    temporal_window: int = 1
    features: List[str] = field(default_factory=list)
    connectivity: Dict[str, Any] = field(default_factory=dict)
    encoding: Optional[Dict[str, Any]] = None
    processing: Optional[Dict[str, Any]] = None
    decoding: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GraphConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GraphConfig":
        """Load config from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "graph_type": self.graph_type,
            "grid_size": self.grid_size,
            "mesh_distance": self.mesh_distance,
            "temporal_steps": self.temporal_steps,
            "temporal_window": self.temporal_window,
            "features": self.features,
            "connectivity": self.connectivity,
            "encoding": self.encoding,
            "processing": self.processing,
            "decoding": self.decoding,
            "metadata": self.metadata,
        }

    def to_yaml(self, output_path: str):
        """Save config to YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML not installed")

        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class PipelineBuilder:
    """
    Build graphs from configuration.

    Supports creating graphs from YAML/dict configurations
    with predefined architectures.
    """

    def __init__(self):
        """Initialize pipeline builder."""
        self._architectures = self._register_architectures()

    @staticmethod
    def _register_architectures() -> Dict[str, callable]:
        """Register available graph architectures."""
        return {
            "keisler": lambda cfg: _create_keisler_architecture(cfg),
            "graphcast": lambda cfg: _create_graphcast_architecture(cfg),
            "meshgraphnet": lambda cfg: _create_meshgraphnet_architecture(cfg),
        }

    def build_from_config(self, config: GraphConfig) -> nx.DiGraph:
        """
        Build graph from configuration.

        Parameters
        ----------
        config : GraphConfig
            Graph configuration

        Returns
        -------
        graph : nx.DiGraph
            Created graph
        """
        if config.graph_type not in self._architectures:
            raise ValueError(
                f"Unknown graph type: {config.graph_type}. "
                f"Available: {list(self._architectures.keys())}"
            )

        logger.info(f"Building {config.graph_type} architecture")
        builder = self._architectures[config.graph_type]
        graph = builder(config)

        # Add metadata
        graph.graph.update(config.metadata)
        graph.graph["config"] = config.to_dict()

        return graph

    def build_from_yaml(self, yaml_path: str) -> nx.DiGraph:
        """Build graph from YAML configuration file."""
        config = GraphConfig.from_yaml(yaml_path)
        return self.build_from_config(config)

    def build_from_dict(self, config_dict: Dict[str, Any]) -> nx.DiGraph:
        """Build graph from dictionary configuration."""
        config = GraphConfig.from_dict(config_dict)
        return self.build_from_config(config)


def _create_keisler_architecture(config: GraphConfig) -> nx.DiGraph:
    """Create Keisler (2021) style graph architecture."""
    from .create.archetype import create_keisler_graph

    # Create grid coordinates
    size = config.grid_size
    xs, ys = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    coords = np.stack([xs.flatten(), ys.flatten()], axis=-1)

    # Create graph
    graph = create_keisler_graph(
        coords=coords,
        mesh_node_distance=config.mesh_distance,
    )

    return graph


def _create_graphcast_architecture(config: GraphConfig) -> nx.DiGraph:
    """Create GraphCast style multiscale graph architecture."""
    from .create.base import create_all_graph_components

    # Create grid coordinates
    size = config.grid_size
    xs, ys = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    coords = np.stack([xs.flatten(), ys.flatten()], axis=-1)

    # Create multiscale mesh graph
    connectivity_kwargs = config.connectivity.get("m2m_connectivity_kwargs", {})
    connectivity_kwargs.setdefault("max_num_levels", 2)
    connectivity_kwargs.setdefault("mesh_node_distance", config.mesh_distance)
    connectivity_kwargs.setdefault("level_refinement_factor", 3)  # Must be odd

    graph = create_all_graph_components(
        coords=coords,
        m2m_connectivity="flat_multiscale",
        m2g_connectivity="nearest_neighbour",
        g2m_connectivity="nearest_neighbour",
        m2m_connectivity_kwargs=connectivity_kwargs,
        m2g_connectivity_kwargs={},
        g2m_connectivity_kwargs={},
    )

    return graph


def _create_meshgraphnet_architecture(config: GraphConfig) -> nx.DiGraph:
    """Create MeshGraphNet style hierarchical architecture."""
    from .create.base import create_all_graph_components

    # Create grid coordinates
    size = config.grid_size
    xs, ys = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    coords = np.stack([xs.flatten(), ys.flatten()], axis=-1)

    # Create hierarchical mesh graph
    connectivity_kwargs = config.connectivity.get("m2m_connectivity_kwargs", {})
    connectivity_kwargs.setdefault("max_num_levels", 3)
    connectivity_kwargs.setdefault("mesh_node_distance", config.mesh_distance)
    connectivity_kwargs.setdefault("level_refinement_factor", 3)  # Must be odd

    graph = create_all_graph_components(
        coords=coords,
        m2m_connectivity="hierarchical",
        m2g_connectivity="containing_rectangle",
        g2m_connectivity="nearest_neighbour",
        m2m_connectivity_kwargs=connectivity_kwargs,
        m2g_connectivity_kwargs={},
        g2m_connectivity_kwargs={},
    )

    return graph


def create_graph_from_config(config_path_or_dict) -> nx.DiGraph:
    """
    Create graph from configuration file or dictionary.

    Parameters
    ----------
    config_path_or_dict : str or dict
        Path to YAML config file or configuration dictionary

    Returns
    -------
    graph : nx.DiGraph
        Created graph
    """
    builder = PipelineBuilder()

    if isinstance(config_path_or_dict, str):
        logger.info(f"Loading config from {config_path_or_dict}")
        return builder.build_from_yaml(config_path_or_dict)
    elif isinstance(config_path_or_dict, dict):
        logger.info("Loading config from dictionary")
        return builder.build_from_dict(config_path_or_dict)
    else:
        raise ValueError("Config must be YAML path or dictionary")
