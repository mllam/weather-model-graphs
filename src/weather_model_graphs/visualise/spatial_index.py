"""
Spatial indexing utilities for efficient neighbor queries in weather model graphs.

This module provides optimized spatial indexing using KD-Trees and Ball Trees
for fast neighbor searches, reducing complexity from O(N²) to O(N log N).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
from scipy.spatial import KDTree, cKDTree

# Optional BallTree
try:
    from sklearn.neighbors import BallTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SpatialIndex(ABC):
    """Abstract base class for spatial indexing."""

    def __init__(self, points: np.ndarray):
        """
        Initialize spatial index with points.

        Parameters
        ----------
        points : np.ndarray
            Array of shape (N, D) where N is number of points, D is dimension.
        """
        self.points = points
        self._build_index()

    @abstractmethod
    def _build_index(self):
        """Build the spatial index."""
        pass

    @abstractmethod
    def query(self, query_points: np.ndarray, k: int = 1, radius: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query the index for nearest neighbors.

        Parameters
        ----------
        query_points : np.ndarray
            Points to query, shape (M, D)
        k : int
            Number of nearest neighbors
        radius : float
            Search radius (for radius queries)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Distances and indices of neighbors
        """
        pass


class KDTreeIndex(SpatialIndex):
    """KD-Tree based spatial index using scipy."""

    def _build_index(self):
        """Build KD-Tree index."""
        self.index = cKDTree(self.points)  # Use cKDTree for better performance

    def query(self, query_points: np.ndarray, k: int = 1, radius: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query KD-Tree for nearest neighbors.

        Parameters
        ----------
        query_points : np.ndarray
            Points to query, shape (M, D)
        k : int
            Number of nearest neighbors
        radius : float
            Search radius (ignored for k-nearest, used for radius queries)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Distances and indices of neighbors
        """
        if radius is not None:
            # Radius query
            indices = self.index.query_ball_point(query_points, radius)
            # For consistency, return distances and indices
            distances = []
            flat_indices = []
            for i, idx_list in enumerate(indices):
                if idx_list:
                    dists = np.linalg.norm(self.points[idx_list] - query_points[i], axis=1)
                    distances.extend(dists)
                    flat_indices.extend(idx_list)
                else:
                    distances.append(np.inf)
                    flat_indices.append(-1)
            return np.array(distances), np.array(flat_indices)
        else:
            # k-nearest neighbors
            distances, indices = self.index.query(query_points, k=k)
            # Handle scalar case when k=1
            if k == 1:
                distances = distances.reshape(-1, 1)
                indices = indices.reshape(-1, 1)
            return distances, indices


class BallTreeIndex(SpatialIndex):
    """Ball Tree based spatial index using sklearn."""

    def __init__(self, points: np.ndarray):
        if not HAS_SKLEARN:
            raise ImportError("BallTree requires scikit-learn. Install with: pip install scikit-learn")
        super().__init__(points)

    def _build_index(self):
        """Build Ball Tree index."""
        self.index = BallTree(self.points)

    def query(self, query_points: np.ndarray, k: int = 1, radius: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query Ball Tree for nearest neighbors.

        Parameters
        ----------
        query_points : np.ndarray
            Points to query, shape (M, D)
        k : int
            Number of nearest neighbors
        radius : float
            Search radius (ignored for k-nearest, used for radius queries)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Distances and indices of neighbors
        """
        if radius is not None:
            # Radius query
            indices = self.index.query_radius(query_points, radius)
            distances = []
            flat_indices = []
            for i, idx_list in enumerate(indices):
                if len(idx_list) > 0:
                    dists = np.linalg.norm(self.points[idx_list] - query_points[i], axis=1)
                    distances.extend(dists)
                    flat_indices.extend(idx_list)
                else:
                    distances.append(np.inf)
                    flat_indices.append(-1)
            return np.array(distances), np.array(flat_indices)
        else:
            # k-nearest neighbors
            distances, indices = self.index.query(query_points, k=k)
            return distances, indices


def create_spatial_index(points: np.ndarray, method: str = "kdtree") -> SpatialIndex:
    """
    Create a spatial index for efficient neighbor queries.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, D) where N is number of points, D is dimension
    method : str
        Indexing method: "kdtree" or "balltree"

    Returns
    -------
    SpatialIndex
        Configured spatial index
    """
    if method.lower() == "kdtree":
        return KDTreeIndex(points)
    elif method.lower() == "balltree":
        return BallTreeIndex(points)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kdtree' or 'balltree'")


def find_neighbors_vectorized(
    query_points: np.ndarray,
    index: SpatialIndex,
    k: int = 1,
    radius: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized neighbor search using spatial index.

    Parameters
    ----------
    query_points : np.ndarray
        Points to query, shape (M, D)
    index : SpatialIndex
        Pre-built spatial index
    k : int
        Number of nearest neighbors
    radius : float
        Search radius

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Distances and indices of neighbors
    """
    return index.query(query_points, k=k, radius=radius)