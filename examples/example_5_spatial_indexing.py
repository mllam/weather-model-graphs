"""
Example 5: Efficient Spatial Indexing
=======================================

This example demonstrates KD-Tree based spatial indexing for
fast neighbor queries on large point sets, critical for large weather grids.
"""

import weather_model_graphs as wmg
import numpy as np
import time

print("=" * 60)
print("Example 5: Efficient Spatial Indexing")
print("=" * 60)

# Create a large coordinate set
print("\nGenerating large coordinate set...")
np.random.seed(42)
n_points = 1_000_000
coords = np.random.rand(n_points, 2)
print(f"Generated {n_points:,} random points")

# Build KD-Tree index
print("\nBuilding KD-Tree spatial index...")
start = time.time()
index = wmg.create_spatial_index(coords, method="kdtree")
build_time = time.time() - start
print(f"KD-Tree built in {build_time:.3f} seconds")

# Query: Find k nearest neighbors
print("\nFinding 10 nearest neighbors for random point...")
query_point = coords[0]
start = time.time()
neighbors, distances = index.query_knn(query_point, k=10)
query_time = time.time() - start
print(f"Query completed in {query_time*1000:.3f} ms")
print(f"Neighbors found: {neighbors}")
print(f"Distances: {distances}")

# Query: Find all points within radius
print("\nFinding all points within radius 0.05...")
start = time.time()
radius_neighbors, _ = index.query_radius(query_point, radius=0.05)
radius_time = time.time() - start
print(f"Query completed in {radius_time*1000:.3f} ms")
print(f"Points found: {len(radius_neighbors)} out of {n_points:,}")

# Performance comparison: O(log N) vs O(N)
print("\n" + "=" * 60)
print("Performance Analysis:")
print("=" * 60)
print(f"\nNaive O(N) approach would calculate {n_points:,} distances")
print(f"KD-Tree O(log N) approach: ~log({n_points:,}) = {np.log2(n_points):.1f} operations")
print(f"Estimated speedup: ~{n_points / np.log2(n_points):.0f}x")

# Vectorized batch queries
print("\nVectorized neighbor search for multiple points...")
query_points = coords[:100]
start = time.time()
batch_neighbors = wmg.find_neighbors_vectorized(
    query_points,
    coords,
    max_neighbors=4,
    method="kdtree"
)
batch_time = time.time() - start
print(f"Batch query for 100 points completed in {batch_time*1000:.3f} ms")
print(f"Average per-point query: {batch_time*10:.3f} ms")

print("\n✓ Example 5 complete!\n")
