"""
Example 4: Feature Engineering
================================

This example demonstrates how to add ML-ready features to graphs,
including wind velocity, pressure gradients, temporal/spatial encodings.
"""

import weather_model_graphs as wmg
import numpy as np

print("=" * 60)
print("Example 4: Feature Engineering")
print("=" * 60)

# Load base graph
print("\nLoading base graph...")
graph = wmg.load_prebuilt("keisler", grid_size=16)

# Add sample data to nodes for feature engineering
print("Adding sample weather data...")
for node in graph.nodes():
    pos = graph.nodes[node].get("pos", np.array([0.5, 0.5]))
    # Simulate some weather variables
    graph.nodes[node]["u_wind"] = 5 + 2 * np.sin(pos[0] * 2 * np.pi)
    graph.nodes[node]["v_wind"] = 3 + 2 * np.cos(pos[1] * 2 * np.pi)
    graph.nodes[node]["pressure"] = 1000 + 50 * np.sin(pos[0] * np.pi)
    graph.nodes[node]["temperature"] = 15 + 10 * np.cos(pos[1] * np.pi)

# Add features
print("\nAdding engineered features...")

# Wind features
graph = wmg.add_wind_velocity(graph, u_attr="u_wind", v_attr="v_wind")
graph = wmg.add_wind_direction(graph, u_attr="u_wind", v_attr="v_wind")

# Pressure features
graph = wmg.add_pressure_gradient(graph, pressure_attr="pressure")

# Temporal and spatial encodings
graph = wmg.add_temporal_encoding(graph, max_period=24, num_frequencies=4)
graph = wmg.add_spatial_encoding(graph, num_frequencies=4)

# Topological features
graph = wmg.add_node_degree_features(graph)

print("Features added to graph")

# Show what features are in the graph
print("\nNode features:")
sample_node = list(graph.nodes())[0]
for key, val in graph.nodes[sample_node].items():
    if isinstance(val, (int, float, np.number)):
        print(f"  - {key}: {val:.4f}")
    elif isinstance(val, np.ndarray):
        if len(val.shape) == 1:
            print(f"  - {key}: array({len(val)} dims)")
        else:
            print(f"  - {key}: array{val.shape}")

# Normalize features
print("\nNormalizing features...")
graph = wmg.normalize_features(
    graph,
    feature_keys=["wind_velocity", "pressure"],
    method="minmax"
)

print(f"\nAfter normalization:")
sample_node = list(graph.nodes())[0]
wind_vel = graph.nodes[sample_node].get("wind_velocity")
pressure = graph.nodes[sample_node].get("pressure")
print(f"  - wind_velocity: {wind_vel:.4f} (normalized to [0, 1])")
print(f"  - pressure: {pressure:.4f} (normalized to [0, 1])")

print("\n✓ Example 4 complete!\n")
