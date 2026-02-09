import pytest
import xarray as xr

import weather_model_graphs as wmg


def test_graph_from_icon_grid_remote():
    url = "https://swift.dkrz.de/v1/dkrz_948e7d4bbfbb445fbff5315fc433e36a/grids/EUREC4A_PR1250m_DOM01.zarr"

    try:
        ds = xr.open_zarr(url, consolidated=True)
    except Exception as exc:
        pytest.skip(f"Could not open remote zarr dataset: {exc}")

    clat = ds["clat"]
    if clat.ndim != 1:
        pytest.skip("Unexpected clat dimensions in ICON grid dataset.")

    cell_dim = clat.dims[0]
    ds_small = ds.isel({cell_dim: slice(0, 1000)})

    graph = wmg.load.graph_from_icon_grid(ds_small)

    n_cells = ds_small["clat"][cell_dim].size
    assert graph.number_of_nodes() == n_cells
    assert graph.number_of_edges() > 0

    node0 = next(iter(graph.nodes))
    assert "clat" in graph.nodes[node0]
    assert "clon" in graph.nodes[node0]
    assert "pos" in graph.nodes[node0]
