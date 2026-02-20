import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from weather_model_graphs.create.mesh.mesh import create_single_level_2d_mesh_graph

# 1. Create irregular data (a heart shape or 'blob')
t = np.linspace(0, 2*np.pi, 100)
x = 16 * np.sin(t)**3
y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
xy_data = np.column_stack((x, y))

# 2. Generate Meshes
# Standard rectangular mesh (Before)
g_before = create_single_level_2d_mesh_graph(xy_data, nx=25, ny=25, crop_to_convex_hull=False)
pos_before = np.array([data["pos"] for _, data in g_before.nodes(data=True)])

# Cropped mesh (After)
g_after = create_single_level_2d_mesh_graph(xy_data, nx=25, ny=25, crop_to_convex_hull=True)
pos_after = np.array([data["pos"] for _, data in g_after.nodes(data=True)])

# 3. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot Before
ax1.scatter(pos_before[:, 0], pos_before[:, 1], c='blue', s=10, alpha=0.3, label='Mesh Nodes')
ax1.plot(xy_data[:, 0], xy_data[:, 1], c='red', linewidth=2, label='Data Boundary')
ax1.set_title(f"Before: Standard Bounding Box\n({len(pos_before)} nodes)")
ax1.legend()

# Plot After
ax2.scatter(pos_after[:, 0], pos_after[:, 1], c='green', s=10, alpha=0.6, label='Pruned Mesh')
ax2.plot(xy_data[:, 0], xy_data[:, 1], c='red', linewidth=2, label='Data Boundary')
ax2.set_title(f"After: Convex Hull Cropped\n({len(pos_after)} nodes)")
ax2.legend()

plt.tight_layout()
plt.savefig('convex_hull_verification.png')
print("✅ Visualization saved as 'convex_hull_verification.png'")