import numpy as np
import pytest
from loguru import logger
import networkx as nx

import weather_model_graphs as wmg


def test_isolated_nodes_warning():
    # Setup log capture
    captured_warnings = []
    handler_id = logger.add(
        lambda msg: captured_warnings.append(msg.record["message"]),
        level="WARNING",
    )

    try:
        # Create a small grid setup where max_dist is too small to reach the outer node
        # Mesh nodes: a dense cluster
        xy_mesh = np.array([
            [0.1, 0.1], [0.1, 0.2], [0.2, 0.1], [0.2, 0.2]
        ])
        # Grid nodes: one close to the mesh, one very far away (sparse)
        xy_grid = np.array([
            [0.15, 0.15], # Close, will connect
            [1.50, 1.50], # Far, will be isolated
        ])

        G_source = wmg.create.mesh.create_single_level_2d_mesh_graph(xy=xy_mesh, nx=2, ny=2)
        G_target = wmg.create.grid.create_grid_graph_nodes(xy=xy_grid)

        # Connect with a restrictive max_dist
        G_connect = wmg.create.base.connect_nodes_across_graphs(
            G_source=G_source,
            G_target=G_target,
            method="within_radius",
            max_dist=0.5,
        )

        assert any("isolated target nodes" in w for w in captured_warnings), \
            f"Expected warning about isolated nodes but got {captured_warnings}"

    finally:
        logger.remove(handler_id)
