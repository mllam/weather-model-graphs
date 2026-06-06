"""
Example 6: ML Pipeline Integration
====================================

This example shows how to prepare graphs for PyTorch training,
including creating DataLoaders and batching graphs.
"""

import weather_model_graphs as wmg
import numpy as np

print("=" * 60)
print("Example 6: ML Pipeline Integration")
print("=" * 60)

# Create multiple graphs for a dataset
print("\nCreating synthetic dataset of 5 graphs...")
graphs = []
for i in range(5):
    graph = wmg.load_prebuilt("keisler", grid_size=16)
    # Add some node features
    for node in graph.nodes():
        graph.nodes[node]["features"] = np.random.rand(4)
    graphs.append(graph)
print(f"Created dataset with {len(graphs)} graphs")

# Method 1: Create PyTorch DataLoader
print("\nMethod 1: PyTorch DataLoader (NetworkX backend)...")
try:
    dataloader = wmg.create_dataloader(
        graphs,
        batch_size=2,
        shuffle=True,
        backend="networkx",
        num_workers=0,  # 0 for example; use >0 in production
    )
    print(f"Created ​DataLoader with {len(dataloader)} batches")
except Exception as e:
    print(f"DataLoader creation skipped: {e}")

# Method 2: Create PyG DataLoader
print("\nMethod 2: PyTorch Geometric DataLoader...")
try:
    pyg_dataloader = wmg.create_dataloader(
        graphs,
        batch_size=2,
        shuffle=True,
        backend="pyg",
        num_workers=0,
    )
    print(f"Created PyG DataLoader with {len(pyg_dataloader)} batches")
    
    # Iterate through batches
    print("\nIterating through PyG batches:")
    for batch_idx, batch in enumerate(pyg_dataloader):
        print(f"  Batch {batch_idx}:")
        print(f"    - Nodes: {batch.num_nodes}")
        print(f"    - Edges: {batch.num_edges}")
        if batch.x is not None:
            print(f"    - Node features: {batch.x.shape}")
        if batch_idx == 0:  # Just show first batch
            break
except Exception as e:
    print(f"PyG DataLoader creation skipped: {e}")

# Method 3: Train/Val/Test split
print("\nMethod 3: Train/Val/Test node split...")
graph = graphs[0]
train_nodes, val_nodes, test_nodes = wmg.split_graph_for_training(
    graph,
    train_size=0.7,
    val_size=0.15,
)
print(f"Train nodes: {len(train_nodes)} ({len(train_nodes)/graph.number_of_nodes()*100:.1f}%)")
print(f"Val nodes: {len(val_nodes)} ({len(val_nodes)/graph.number_of_nodes()*100:.1f}%)")
print(f"Test nodes: {len(test_nodes)} ({len(test_nodes)/graph.number_of_nodes()*100:.1f}%)")

# Method 4: Batch graphs
print("\nMethod 4: Batch multiple graphs into tensors...")
node_features, edge_indices, edge_features = wmg.batch_graphs(
    graphs[:3],
    node_feature_keys=["pos"],
    edge_feature_keys=["len"],
)
print(f"Batched node features: {node_features.shape}")
print(f"Number of edge index arrays: {len(edge_indices)}")
print(f"Number of edge feature arrays: {len(edge_features)}")

# Method 5: Create model inputs
print("\nMethod 5: Create model inputs in different formats...")
graph = graphs[0]

# NetworkX format
nx_input = wmg.create_model_input(graph, backend="networkx")
print(f"NetworkX input: {type(nx_input).__name__} with {nx_input.number_of_nodes()} nodes")

# PyG format
try:
    pyg_input = wmg.create_model_input(graph, backend="pyg")
    print(f"PyG input: {type(pyg_input).__name__} with {pyg_input.num_nodes} nodes")
except Exception as e:
    print(f"PyG input creation skipped: {e}")

# DGL format
try:
    dgl_input = wmg.create_model_input(graph, backend="dgl")
    print(f"DGL input: {type(dgl_input).__name__} with {dgl_input.num_nodes()} nodes")
except Exception as e:
    print(f"DGL input creation skipped: {e}")

print("\n✓ Example 6 complete!\n")
