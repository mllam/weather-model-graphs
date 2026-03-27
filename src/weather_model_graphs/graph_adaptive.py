"""
Adaptive graph construction utilities for non-uniform input grids.

Supports:
- k-NN graph construction (for sparse data)
- Delaunay triangulation (for irregular grids)
- Degree diagnostics for graph analysis
"""

from typing import List, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay


def build_knn_graph(coords: np.ndarray, k: int = 5) -> List[Tuple[int, int]]:
    if coords.shape[0] < k:
        raise ValueError("Number of points must be >= k")

    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(coords)
    _, indices = nbrs.kneighbors(coords)

    edges = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:
                edges.append((i, j))

    return edges


def build_delaunay_graph(coords: np.ndarray) -> List[Tuple[int, int]]:
    if coords.shape[0] < 3:
        raise ValueError("Need at least 3 points")

    tri = Delaunay(coords)
    edges = set()

    for simplex in tri.simplices:
        for i in range(3):
            u = simplex[i]
            v = simplex[(i + 1) % 3]
            edges.add((u, v))
            edges.add((v, u))

    return list(edges)


def compute_degree(edges: List[Tuple[int, int]], num_nodes: int) -> List[int]:
    degree = [0] * num_nodes

    for u, _ in edges:
        degree[u] += 1

    return degree


def degree_statistics(degree: List[int]) -> dict:
    return {
        "min": min(degree),
        "max": max(degree),
        "mean": sum(degree) / len(degree),
    }