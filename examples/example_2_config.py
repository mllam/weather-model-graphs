"""
Example 2: Configuration-Driven Graph Creation
================================================

This example demonstrates creating graphs from YAML or dictionary configurations
for reproducible, shareable graph definitions.
"""

import weather_model_graphs as wmg

print("=" * 60)
print("Example 2: Configuration-Driven Graph Creation")
print("=" * 60)

# Method 1: From dictionary
print("\nMethod 1: Creating graph from dictionary configuration...")
config_dict = {
    "graph_type": "graphcast",
    "grid_size": 32,
    "mesh_distance": 0.0625,
    "temporal_steps": 1,
    "features": ["temperature", "humidity", "pressure"],
    "metadata": {
        "name": "test-config",
        "resolution": "medium",
    }
}

graph = wmg.create_graph_from_config(config_dict)
print(f"Graph created with {graph.number_of_nodes()} nodes")

# Method 2: From YAML file
try:
    print("\nMethod 2: Creating graph from YAML configuration...")
    graph_yaml = wmg.create_graph_from_config("examples/config_graphcast.yaml")
    print(f"Graph created from YAML with {graph_yaml.number_of_nodes()} nodes")
except FileNotFoundError:
    print("YAML file not found (this is okay for this example)")

# Save configuration to YAML
print("\nSaving configuration to YAML...")
config = wmg.GraphConfig.from_dict(config_dict)
config.to_yaml("examples/my_config.yaml")
print("Configuration saved to: examples/my_config.yaml")

# Access configuration later
print(f"\nMetadata: {config.metadata}")
print(f"Graph type: {config.graph_type}")
print(f"Features: {config.features}")

print("\n✓ Example 2 complete!\n")
