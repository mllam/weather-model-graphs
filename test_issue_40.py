import numpy as np
from weather_model_graphs.create.mesh.mesh import create_single_level_2d_mesh_graph

# 1. Create a "circular" cluster of data points
theta = np.linspace(0, 2*np.pi, 100)
r = 10.0
xy = np.column_stack((r*np.cos(theta), r*np.sin(theta)))

# 2. Create a standard (uncropped) mesh
# This will build a massive 20x20 square grid covering the circle
uncropped_graph = create_single_level_2d_mesh_graph(xy, nx=20, ny=20, crop_to_convex_hull=False)
uncropped_nodes = len(uncropped_graph.nodes)

# 3. Create our cropped mesh
cropped_graph = create_single_level_2d_mesh_graph(xy, nx=20, ny=20, crop_to_convex_hull=True)
cropped_nodes = len(cropped_graph.nodes)

print(f"Nodes in standard rectangular mesh: {uncropped_nodes}")
print(f"Nodes in Convex Hull cropped mesh: {cropped_nodes}")
print(f"Nodes deleted from empty space: {uncropped_nodes - cropped_nodes}")

if cropped_nodes < uncropped_nodes:
    print("\n✅ SUCCESS! The Convex Hull math correctly chopped off the empty corners!")
else:
    print("\n❌ FAILED. No nodes were removed.")