"""
Prebuilt graph architectures for common weather models.

Provides ready-to-use graph definitions for:
- Keisler (2021) - Simple flat graph
- GraphCast (Lam et al., 2023) - Multiscale mesh
- MeshGraphNet (Pfaff et al., 2021) - Hierarchical mesh
"""

from typing import Optional

import networkx as nx
import numpy as np
from loguru import logger

from .create.archetype import create_keisler_graph
from .create.base import create_all_graph_components


class GraphArchetype:
    """Base class for prebuilt graph architectures."""

    def __init__(self, name: str, description: str):
        """Initialize archetype."""
        self.name = name
        self.description = description

    def create(self, **kwargs) -> nx.DiGraph:
        """Create graph with given parameters."""
        raise NotImplementedError


class KeislerArchetype(GraphArchetype):
    """
    Keisler (2021) flat graph architecture.

    Single-scale mesh with grid-to-mesh and mesh-to-grid connectivity.
    """

    def __init__(self):
        """Initialize Keisler archetype."""
        super().__init__(
            "keisler",
            "Single-scale mesh graph (Keisler, 2021)",
        )

    def create(
        self,
        grid_size: int = 32,
        mesh_node_distance: float = 0.0625,
        **kwargs
    ) -> nx.DiGraph:
        """
        Create Keisler graph.

        Parameters
        ----------
        grid_size : int
            Grid resolution (grid_size x grid_size)
        mesh_node_distance : float
            Distance between mesh nodes
        **kwargs
            Additional arguments (ignored)

        Returns
        -------
        graph : nx.DiGraph
            Keisler architecture graph
        """
        logger.info(f"Creating Keisler graph (grid_size={grid_size})")

        # Create regular grid coordinates
        xs, ys = np.meshgrid(
            np.linspace(0, 1, grid_size),
            np.linspace(0, 1, grid_size)
        )
        coords = np.stack([xs.flatten(), ys.flatten()], axis=-1)

        graph = create_keisler_graph(
            coords=coords,
            mesh_node_distance=mesh_node_distance,
        )

        return graph

        return graph


class GraphCastArchetype(GraphArchetype):
    """
    GraphCast (Lam et al., 2023) multiscale architecture.

    Flat multiscale mesh with multiple refinement levels.
    """

    def __init__(self):
        """Initialize GraphCast archetype."""
        super().__init__(
            "graphcast",
            "Multiscale mesh graph with flat hierarchy (GraphCast, Lam et al., 2023)",
        )

    def create(
        self,
        grid_size: int = 32,
        mesh_node_distance: float = 0.0625,
        max_levels: int = 2,
        level_refinement_factor: int = 2,
        **kwargs
    ) -> nx.DiGraph:
        """
        Create GraphCast architecture graph.

        Parameters
        ----------
        grid_size : int
            Grid resolution
        mesh_node_distance : float
            Initial mesh node distance
        max_levels : int
            Number of mesh refinement levels
        level_refinement_factor : int
            Refinement factor between levels
        **kwargs
            Additional arguments (ignored)

        Returns
        -------
        graph : nx.DiGraph
            GraphCast architecture graph
        """
        logger.info(
            f"Creating GraphCast graph (grid_size={grid_size}, "
            f"levels={max_levels})"
        )

        # Create regular grid coordinates
        xs, ys = np.meshgrid(
            np.linspace(0, 1, grid_size),
            np.linspace(0, 1, grid_size)
        )
        coords = np.stack([xs.flatten(), ys.flatten()], axis=-1)

        # Ensure level_refinement_factor is odd (required by implementation)
        if level_refinement_factor % 2 == 0:
            level_refinement_factor = max(3, level_refinement_factor + 1)

        graph = create_all_graph_components(
            coords=coords,
            m2m_connectivity="flat_multiscale",
            m2g_connectivity="nearest_neighbour",
            g2m_connectivity="nearest_neighbour",
            m2m_connectivity_kwargs={
                "max_num_levels": max_levels,
                "mesh_node_distance": mesh_node_distance,
                "level_refinement_factor": level_refinement_factor,
            },
            m2g_connectivity_kwargs={},
            g2m_connectivity_kwargs={},
            return_components=False,
        )

        return graph


class MeshGraphNetArchetype(GraphArchetype):
    """
    MeshGraphNet (Pfaff et al., 2021) hierarchical architecture.

    Hierarchical multiscale mesh for structured atmosphere.
    """

    def __init__(self):
        """Initialize MeshGraphNet archetype."""
        super().__init__(
            "meshgraphnet",
            "Hierarchical multiscale mesh (MeshGraphNet, Pfaff et al., 2021)",
        )

    def create(
        self,
        grid_size: int = 32,
        mesh_node_distance: float = 0.0625,
        max_levels: int = 3,
        level_refinement_factor: int = 2,
        **kwargs
    ) -> nx.DiGraph:
        """
        Create MeshGraphNet architecture graph.

        Parameters
        ----------
        grid_size : int
            Grid resolution
        mesh_node_distance : float
            Initial mesh node distance
        max_levels : int
            Number of hierarchical levels
        level_refinement_factor : int
            Refinement factor between levels
        **kwargs
            Additional arguments (ignored)

        Returns
        -------
        graph : nx.DiGraph
            MeshGraphNet architecture graph
        """
        logger.info(
            f"Creating MeshGraphNet graph (grid_size={grid_size}, "
            f"levels={max_levels})"
        )

        # Create regular grid coordinates
        xs, ys = np.meshgrid(
            np.linspace(0, 1, grid_size),
            np.linspace(0, 1, grid_size)
        )
        coords = np.stack([xs.flatten(), ys.flatten()], axis=-1)

        # Ensure level_refinement_factor is odd (required by implementation)
        if level_refinement_factor % 2 == 0:
            level_refinement_factor = max(3, level_refinement_factor + 1)

        graph = create_all_graph_components(
            coords=coords,
            m2m_connectivity="hierarchical",
            m2g_connectivity="containing_rectangle",
            g2m_connectivity="nearest_neighbour",
            m2m_connectivity_kwargs={
                "max_num_levels": max_levels,
                "mesh_node_distance": mesh_node_distance,
                "level_refinement_factor": level_refinement_factor,
            },
            m2g_connectivity_kwargs={},
            g2m_connectivity_kwargs={},
            return_components=False,
        )

        return graph


class GraphArchetypeRegistry:
    """Registry of available graph architectures."""

    def __init__(self):
        """Initialize registry with default archetypes."""
        self._archetypes = {
            "keisler": KeislerArchetype(),
            "graphcast": GraphCastArchetype(),
            "meshgraphnet": MeshGraphNetArchetype(),
        }

    def register(self, name: str, archetype: GraphArchetype):
        """Register new archetype."""
        self._archetypes[name] = archetype
        logger.info(f"Registered archetype: {name}")

    def get(self, name: str) -> GraphArchetype:
        """Get archetype by name."""
        if name not in self._archetypes:
            raise ValueError(
                f"Unknown archetype: {name}. "
                f"Available: {list(self._archetypes.keys())}"
            )
        return self._archetypes[name]

    def list(self) -> dict:
        """List all available archetypes."""
        return {
            name: archetype.description
            for name, archetype in self._archetypes.items()
        }

    def create(self, name: str, **kwargs) -> nx.DiGraph:
        """Create graph from archetype."""
        archetype = self.get(name)
        return archetype.create(**kwargs)


# Global registry
_registry = GraphArchetypeRegistry()


def load_prebuilt(
    name: str,
    grid_size: int = 32,
    mesh_node_distance: float = 0.0625,
    **kwargs
) -> nx.DiGraph:
    """
    Load prebuilt graph architecture.

    Parameters
    ----------
    name : str
        Architecture name ("keisler", "graphcast", "meshgraphnet")
    grid_size : int
        Grid resolution
    mesh_node_distance : float
        Mesh node spacing
    **kwargs
        Additional architecture-specific parameters

    Returns
    -------
    graph : nx.DiGraph
        Prebuilt graph

    Examples
    --------
    >>> import weather_model_graphs as wmg
    >>> graph = wmg.load.prebuilt("graphcast", grid_size=64)
    """
    return _registry.create(
        name,
        grid_size=grid_size,
        mesh_node_distance=mesh_node_distance,
        **kwargs
    )


def list_prebuilt() -> dict:
    """List available prebuilt architectures."""
    return _registry.list()


def register_archetype(name: str, archetype: GraphArchetype):
    """Register custom graph archetype."""
    _registry.register(name, archetype)
