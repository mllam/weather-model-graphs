"""
Example 3: Temporal Graphs for Time Series
=============================================

This example shows how to create temporal graphs that unroll
graph structures over time, enabling autoregressive weather prediction.
"""

import weather_model_graphs as wmg
import numpy as np

print("=" * 60)
print("Example 3: Temporal Graphs for Time Series")
print("=" * 60)

# Create temporal graph with 10 timesteps
print("\nCreating temporal graph (10 timesteps, 2-step history)...")

coords = np.random.rand(100, 2)
temporal_graph = wmg.create_temporal_graph(
    coords=coords,
    timesteps=10,
    temporal_window=2,  # Connect to 2 previous timesteps
    connectivity="nearest_neighbour",
    connectivity_kwargs={"max_neighbors": 4},
)

# Get statistics
stats = temporal_graph.get_statistics()
print(f"\nTemporal Graph Statistics:")
for key, val in stats.items():
    print(f"  - {key}: {val}")

# Get combined graph
full_graph = temporal_graph.get_combined_graph()
print(f"\nCombined temporal graph:")
print(f"  - Total nodes: {full_graph.number_of_nodes()}")
print(f"  - Total edges: {full_graph.number_of_edges()}")

# Get edges by type
spatial_edges = temporal_graph.get_edges_by_type("spatial")
temporal_edges = temporal_graph.get_edges_by_type("temporal")

print(f"\nEdge breakdown:")
print(f"  - Spatial edges: {len(spatial_edges)}")
print(f"  - Temporal edges: {len(temporal_edges)}")
print(f"  - Temporal / Spatial ratio: {len(temporal_edges) / (len(spatial_edges) + 1):.2f}x")

# Get graph at specific timestep
print(f"\nGraph at timestep 5:")
graph_t5 = temporal_graph.get_graph(5)
print(f"  - Nodes: {graph_t5.number_of_nodes()}")
print(f"  - Edges: {graph_t5.number_of_edges()}")

# Unfold predictions (example)
predictions = np.random.rand(10 * 100, 5)  # 10 timesteps, 100 nodes, 5 features
unfolded = wmg.unfold_temporal_predictions(predictions, 10, 100)
print(f"\nUnfolded predictions:")
print(f"  - Timesteps: {len(unfolded)}")
print(f"  - Prediction shape at t=0: {unfolded[0].shape}")

print("\n✓ Example 3 complete!\n")
