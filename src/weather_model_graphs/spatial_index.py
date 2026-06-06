"""
Spatial indexing utilities for efficient graph connectivity.

Uses KD-Tree and Ball Tree for fast neighbor search, enabling
O(log N) lookups instead of O(N²) for large weather models.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree, distance_matrix
from loguru import logger


class SpatialIndex:
    """Base class for spatial indexing structures."""

    def __init__(self, coords: np.ndarray):
        """
        Initialize spatial index.

        Parameters
        ----------
        coords : np.ndarray
            (N, 2) or (N, 3) array of coordinate positions
        """
        self.coords = coords
        self.n_points = len(coords)

    def query_radius(
        self, center: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find all points within radius of center.

        Parameters
        ----------
        center : np.ndarray
            Query point (2,) or (3,) array
        radius : float
            Search radius

        Returns
        -------
        indices : np.ndarray
            Indices of points within radius
        distances : np.ndarray
            Distances to those points
        """
        raise NotImplementedError

    def query_knn(
        self, center: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors.

        Parameters
        ----------
        center : np.ndarray
            Query point
        k : int
            Number of neighbors to return

        Returns
        -------
        indices : np.ndarray
            Indices of k nearest neighbors
        distances : np.ndarray
            Distances to those points
        """
        raise NotImplementedError


class KDTreeIndex(SpatialIndex):
    """KD-Tree based spatial indexing for fast neighbor queries."""

    def __init__(self, coords: np.ndarray):
        """
        Initialize KD-Tree spatial index.

        Parameters
        ----------
        coords : np.ndarray
            (N, 2) or (N, 3) array of coordinate positions
        """
        super().__init__(coords)
        logger.debug(f"Building KD-Tree for {self.n_points} points")
        self.tree = cKDTree(coords)

    def query_radius(
        self, center: np.ndarray, radius: float, return_sorted: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find all points within radius using KD-Tree.

        Parameters
        ----------
        center : np.ndarray
            Query point
        radius : float
            Search radius
        return_sorted : bool
            If True, return sorted by distance

        Returns
        -------
        indices : np.ndarray
            Indices of points within radius
        distances : np.ndarray
            Distances to those points
        """
        # Use sparse matrix output for efficiency
        results = self.tree.query_ball_point(center, r=radius, return_sorted=return_sorted)
        if isinstance(results, list):
            indices = np.array(results)
        else:
            indices = results

        # Calculate distances
        if len(indices) > 0:
            distances = distance_matrix([center], self.coords[indices])[0]
        else:
            distances = np.array([])

        return indices, distances

    def query_knn(self, center: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors using KD-Tree.

        Parameters
        ----------
        center : np.ndarray
            Query point
        k : int
            Number of neighbors

        Returns
        -------
        indices : np.ndarray
            Indices of k nearest neighbors
        distances : np.ndarray
            Distances to those points
        """
        # Ensure k doesn't exceed total points
        k = min(k, self.n_points)
        distances, indices = self.tree.query(center, k=k)

        # Handle scalar case (single NN)
        if np.isscalar(indices):
            indices = np.array([indices])
            distances = np.array([distances])
        else:
            indices = np.array(indices)
            distances = np.array(distances)

        return indices, distances


class BallTreeIndex(SpatialIndex):
    """Ball Tree based spatial indexing."""

    def __init__(self, coords: np.ndarray, leaf_size: int = 40):
        """
        Initialize Ball Tree spatial index.

        Parameters
        ----------
        coords : np.ndarray
            (N, 2) or (N, 3) array of coordinate positions
        leaf_size : int
            Leaf size for tree construction
        """
        try:
            from sklearn.neighbors import BallTree
        except ImportError:
            logger.warning(
                "scikit-learn not available. Falling back to KDTree. "
                "Install scikit-learn for better performance on high-dimensional data."
            )
            self.__dict__ = KDTreeIndex(coords).__dict__
            return

        super().__init__(coords)
        logger.debug(f"Building Ball Tree for {self.n_points} points")
        self.tree = BallTree(coords, leaf_size=leaf_size)

    def query_radius(
        self, center: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find all points within radius using Ball Tree."""
        center = center.reshape(1, -1)
        indices_list = self.tree.query_radius(center, r=radius)[0]
        indices = np.array(list(indices_list))

        if len(indices) > 0:
            distances = distance_matrix(center, self.coords[indices])[0]
        else:
            distances = np.array([])

        return indices, distances

    def query_knn(self, center: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors using Ball Tree."""
        k = min(k, self.n_points)
        center = center.reshape(1, -1)
        distances, indices = self.tree.query(center, k=k)
        return indices[0], distances[0]


def create_spatial_index(
    coords: np.ndarray, method: str = "kdtree"
) -> SpatialIndex:
    """
    Create spatial index for efficient neighbor queries.

    Parameters
    ----------
    coords : np.ndarray
        (N, 2) or (N, 3) coordinate array
    method : str
        Index method: "kdtree" or "balltree"

    Returns
    -------
    SpatialIndex
        Spatial index object with query methods
    """
    if method == "kdtree":
        return KDTreeIndex(coords)
    elif method == "balltree":
        return BallTreeIndex(coords)
    else:
        raise ValueError(f"Unknown spatial index method: {method}")


def find_neighbors_vectorized(
    query_coords: np.ndarray,
    target_coords: np.ndarray,
    radius: Optional[float] = None,
    max_neighbors: Optional[int] = None,
    method: str = "kdtree",
) -> List[np.ndarray]:
    """
    Vectorized neighbor search using spatial indexing.

    Parameters
    ----------
    query_coords : np.ndarray
        (N, 2) or (N, 3) query coordinates
    target_coords : np.ndarray
        (M, 2) or (M, 3) target coordinates
    radius : float, optional
        Search radius (mutually exclusive with max_neighbors)
    max_neighbors : int, optional
        Maximum number of neighbors to return
    method : str
        Indexing method ("kdtree" or "balltree")

    Returns
    -------
    neighbors : List[np.ndarray]
        List of neighbor indices for each query point
    """
    index = create_spatial_index(target_coords, method=method)

    neighbors = []
    for coord in query_coords:
        if radius is not None:
            indices, _ = index.query_radius(coord, radius)
        elif max_neighbors is not None:
            indices, _ = index.query_knn(coord, max_neighbors)
        else:
            indices, _ = index.query_knn(coord, 1)

        neighbors.append(indices)

    return neighbors
