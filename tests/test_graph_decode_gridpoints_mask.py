import tempfile
from pathlib import Path

import numpy as np
import torch

import weather_model_graphs as wmg
from tests import utils as test_utils


def load_adjecency_matrix(graph_name, output_directory="."):
    fp = Path(output_directory) / f"{graph_name}_edge_index.pt"
    edge_index = torch.load(fp)
    return edge_index.numpy()


def test_graph_decode_gridpoints_mask():
    """
    Test to ensure that when applying a mask to select which grid nodes to
    decode to that the resulting adjecency matrix contains the grid-indexes
    of the retained nodes.
    """

    xy = test_utils.create_rectangular_fake_xy(Nx=5, Ny=5)
    mesh_node_distance = 2.5
    fn = wmg.create.archetype.create_keisler_graph

    unfiltered_graph = fn(coords=xy, mesh_node_distance=mesh_node_distance)

    # mask every 3rd gridpoint
    decode_mask = (np.arange(xy.shape[0]) % 3 == 0).astype(int)
    filtered_graph = fn(
        coords=xy, mesh_node_distance=mesh_node_distance, decode_mask=decode_mask
    )

    # store the graphs to disk and load the adjecency matrices for each
    with tempfile.TemporaryDirectory() as tmpdirname:
        name_filtered = "example_keisler_graph_filtered"
        name_unfiltered = "example_keisler_graph"

        wmg.save.to_pyg(
            graph=unfiltered_graph, output_directory=tmpdirname, name=name_unfiltered
        )
        wmg.save.to_pyg(
            graph=filtered_graph, output_directory=tmpdirname, name=name_filtered
        )

        adj_filtered = load_adjecency_matrix(name_filtered, output_directory=tmpdirname)
        adj_unfiltered = load_adjecency_matrix(
            name_unfiltered, output_directory=tmpdirname
        )

    # manually filter the edge connections from
    grid_indexes_to_remove = np.arange(0, xy.shape[0])[decode_mask == 0]
    adj_pairs = []
    for i in range(adj_unfiltered.shape[1]):
        m_idx, g_idx = adj_unfiltered[:, i]
        if g_idx in grid_indexes_to_remove:
            continue
        adj_pairs.append((m_idx, g_idx))

    adj_unfiltered_masked = np.array(adj_pairs).T

    np.testing.assert_equal(adj_filtered, adj_unfiltered_masked)
