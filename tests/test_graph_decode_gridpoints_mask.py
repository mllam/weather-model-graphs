import tempfile
from pathlib import Path

import numpy as np
import torch

import weather_model_graphs as wmg
from tests import utils as test_utils


def test_graph_decode_gridpoints_mask():
    """
    Test to ensure that when applying a mask to select which grid nodes to
    decode to that the resulting adjacency list contains the grid-indexes
    of the retained nodes.
    """

    xy = test_utils.create_rectangular_fake_xy(Nx=5, Ny=5)
    mesh_node_distance = 2.5
    fn = wmg.create.archetype.create_keisler_graph

    unfiltered_graph = fn(
        coords=xy, mesh_node_distance=mesh_node_distance, return_components=True
    )["m2g"]

    # keep only every 3rd gridpoint
    decode_mask = np.arange(xy.shape[0]) % 3 == 0
    filtered_graph = fn(
        coords=xy,
        mesh_node_distance=mesh_node_distance,
        decode_mask=decode_mask,
        return_components=True,
    )["m2g"]

    def load_edge_index(graph_name, output_directory="."):
        file_path = Path(output_directory) / f"{graph_name}_edge_index.pt"
        edge_index = torch.load(file_path)
        return edge_index.numpy()

    # store the graphs to disk and load the adjecency matrices for each
    with tempfile.TemporaryDirectory() as tmpdirname:
        name_filtered = "m2g_filtered"
        name_unfiltered = "m2g"

        wmg.save.to_pyg(
            graph=unfiltered_graph, output_directory=tmpdirname, name=name_unfiltered
        )
        wmg.save.to_pyg(
            graph=filtered_graph, output_directory=tmpdirname, name=name_filtered
        )

        adj_filtered = load_edge_index(name_filtered, output_directory=tmpdirname)
        adj_unfiltered = load_edge_index(name_unfiltered, output_directory=tmpdirname)

    # Use the decode mask to find out which edges to retain by taking the
    # grid-index values from the m2g adjacency list and indexing into the
    # decode mask
    unfiltered_edge_mask = decode_mask[adj_unfiltered[1]]
    # Filter edges from full set
    adj_unfiltered_masked = adj_unfiltered[:, unfiltered_edge_mask]

    np.testing.assert_equal(adj_filtered.shape, adj_unfiltered_masked.shape)

    # Re-index nodes from unfiltered m2g to match
    # New index is number of kept nodes before in decode_mask
    reindex_map = decode_mask.cumsum() - 1
    adj_grid_reindexed = reindex_map[adj_unfiltered_masked[1]]

    # Reindex mesh nodes to start at 0 and sort edge index tensor
    # before comparing. Checks that these represent the same graph.
    reindexed_wmg_filtered = np.sort(
        np.concatenate(
            (
                adj_filtered[0] - adj_filtered[0].min(),
                adj_filtered[1],
            ),
            axis=0,
        )
    )
    reindexed_post_filtered = np.sort(
        np.concatenate(
            (
                adj_unfiltered_masked[0] - adj_unfiltered_masked[0].min(),
                adj_grid_reindexed,
            ),
            axis=0,
        )
    )

    np.testing.assert_equal(reindexed_wmg_filtered, reindexed_post_filtered)
