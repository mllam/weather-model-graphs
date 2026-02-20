import numpy as np
from weather_model_graphs.create.mesh.kinds.flat import create_flat_singlescale_mesh_graph

# 1. Create a tiny coordinate grid (A box from 0 to 1)
# This means the maximum extent (range) is just 1.0 in both x and y.
xy = np.array([
    [0.0, 0.0],
    [1.0, 1.0]
])

# 2. Set a massive mesh distance (100.0)
# Since 100 is way bigger than our grid size of 1, this WILL trigger our new error.
print("Triggering the error...")

try:
    create_flat_singlescale_mesh_graph(xy, mesh_node_distance=100.0)
    print("Wait, it didn't crash? Something is wrong.")
except ValueError as e:
    print("\n✅ SUCCESS! Caught the expected error. Here is your new message:\n")
    print(f"--> {e}")