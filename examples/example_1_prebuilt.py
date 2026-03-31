"""
Example 1: Quick Start with Prebuilt Graphs
============================================

This example shows how to quickly load prebuilt graph architectures
without needing to understand the details of graph creation.
"""

import weather_model_graphs as wmg
import numpy as np

print("=" * 60)
print("Example 1: Prebuilt Graph Architectures")
print("=" * 60)

# List available architectures
print("\nAvailable prebuilt architectures:")
for name, description in wmg.list_prebuilt().items():
    print(f"  - {name}: {description}")

# Load GraphCast architecture (multiscale mesh)
print("\nLoading GraphCast architecture (64x64 grid)...")
graph = wmg.load_prebuilt("graphcast", grid_size=64, mesh_node_distance=0.0625)

print(f"Graph created:")
print(f"  - Nodes: {graph.number_of_nodes()}")
print(f"  - Edges: {graph.number_of_edges()}")
print(f"  - Graph density: {graph.number_of_edges() / (graph.number_of_nodes() ** 2) * 100:.3f}%")

# Get backend and convert to different formats
print("\nConverting between backends...")
backend = wmg.get_backend(graph)

# To PyTorch Geometric
pyg_data = backend.to_pyg()
if pyg_data is not None:
    print(f"PyG Data: x={pyg_data.x.shape if pyg_data.x is not None else None}, "
          f"edge_index={pyg_data.edge_index.shape}")
else:
    print("PyG Data: Not available (PyTorch Geometric not installed)")

# To DGL (if available)
try:
    dgl_graph = backend.to_dgl()
    if dgl_graph is not None:
        print(f"DGL Graph: {dgl_graph.num_nodes()} nodes, {dgl_graph.num_edges()} edges")
    else:
        print("DGL Graph: Not available (DGL not installed)")
except Exception as e:
    print(f"DGL conversion skipped: {e}")

print("\n✓ Example 1 complete!\n")
