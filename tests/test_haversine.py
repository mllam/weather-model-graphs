import numpy as np

import weather_model_graphs as wmg


def test_1d_equator_line():
    # create N equally spaced points along the equator and return lat/lon
    N = 40
    lats = np.zeros(N)
    lons = np.linspace(0, 360, N, endpoint=False)
    coords = np.stack([lats, lons], axis=1)

    dl = 20.0

    # TODO add test that ensure that
    wmg.create.archetype.create_keisler_graph(
        coords=coords, mesh_node_distance=dl, distance_metric="haversine"
    )
