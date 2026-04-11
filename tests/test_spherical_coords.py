import numpy as np
import pytest
import weather_model_graphs as wmg
from weather_model_graphs.create.base import create_all_graph_components

def test_flat_coords():
    coords=np.array([[0,0],[0,1],[1,0],[1,1]])
    graph=wmg.create.archetype.create_keisler_graph(
        coords=coords,
        mesh_node_distance=0.5
    )
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    print(f"Success flat graph has {len(graph.nodes)}nodes and {len(graph.edges)}edges")


def test_spherical_coords():
    coords=np.array([
        [-10.0,0.0],
        [10.0,0.0],
        [0.0,10.0],
        [0.0,-10.0],
    ])
    graph=wmg.create.archetype.create_keisler_graph(
        coords=coords,
        coords_crs="EPSG:4326",
        graph_crs="EPSG:3857",
        mesh_node_distance=500_000
    )
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    for _, _, data in graph.edges(data=True):
        if 'len' in data:
            print(f"Edge length in metres:{data['len']:.1f}m")
    print(f"Success! Spherical graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")



if __name__ == "__main__":
    print("Running Test 1 --Flat Cartesian...")
    test_flat_coords()

    print("\nRunning Test 2 --Spherical....")
    test_spherical_coords()

    print("\nAll Tests Passed!")
    
