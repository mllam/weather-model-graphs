import numpy as np
import pytest
import networkx

from weather_model_graphs.create.mesh.kinds.hierarchical import (
    create_hierarchical_multiscale_mesh_graph,
)
from weather_model_graphs.create.mesh.mesh import (
    create_single_level_2d_mesh_graph,
)

# A simple valid coordinate grid
xy_data = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0]
])

def test_flat_mesh_supports_tuple_positions(monkeypatch):
    """
    Test flat mesh creation logic handles tuple positions (e.g., for cuGraph backend).
    """
    
    original_grid = networkx.grid_2d_graph
    
    def mock_grid_2d_graph(*args, **kwargs):
        g = original_grid(*args, **kwargs)
        return g
        
    monkeypatch.setattr(networkx, "grid_2d_graph", mock_grid_2d_graph)

    original_digraph = networkx.DiGraph
    
    def mock_digraph(g, *args, **kwargs):
        dg = original_digraph(g, *args, **kwargs)
        for node in dg.nodes:
            if "pos" in dg.nodes[node]:
                dg.nodes[node]["pos"] = tuple(dg.nodes[node]["pos"])
        return dg
        
    monkeypatch.setattr(networkx, "DiGraph", mock_digraph)
    
    dg = create_single_level_2d_mesh_graph(xy_data, nx=2, ny=2)

    for u, v, data in dg.edges(data=True):
        assert "len" in data
        assert "vdiff" in data
        assert isinstance(data["len"], (float, np.floating))
        assert isinstance(data["vdiff"], (np.ndarray, list, tuple))

def test_flat_mesh_supports_list_positions(monkeypatch):
    """Test flat mesh creation logic handles list positions."""
    original_digraph = networkx.DiGraph
    
    def mock_digraph(g, *args, **kwargs):
        dg = original_digraph(g, *args, **kwargs)
        for node in dg.nodes:
            if "pos" in dg.nodes[node]:
                dg.nodes[node]["pos"] = list(dg.nodes[node]["pos"])
        return dg
        
    monkeypatch.setattr(networkx, "DiGraph", mock_digraph)
    
    dg = create_single_level_2d_mesh_graph(xy_data, nx=2, ny=2)

def test_flat_mesh_supports_numpy_positions():
    """Test standard numpy array positions."""
    dg = create_single_level_2d_mesh_graph(xy_data, nx=2, ny=2)

    
def test_hierarchical_mesh_supports_tuple_positions(monkeypatch):
    """Test hierarchical mesh logic handles tuple positions (e.g., for cuGraph)."""
    
    original_mk_2d_graph = networkx.grid_2d_graph
    
    def mock_grid_2d_graph(nx, ny, *args, **kwargs):
        g = original_mk_2d_graph(nx, ny, *args, **kwargs)
        return g

    original_digraph = networkx.DiGraph
    
    def mock_digraph(g=None, *args, **kwargs):
        dg = original_digraph(g, *args, **kwargs)
        if g is not None:
            for node in dg.nodes:
                if "pos" in dg.nodes[node]:
                    dg.nodes[node]["pos"] = tuple(dg.nodes[node]["pos"])
        return dg
        
    monkeypatch.setattr(networkx, "DiGraph", mock_digraph)

    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    xx, yy = np.meshgrid(x, y)
    xy_large = np.column_stack([xx.ravel(), yy.ravel()])
    
    components = create_hierarchical_multiscale_mesh_graph(
        xy_large,
        mesh_node_distance=10.0,
        level_refinement_factor=3.0,
        max_num_levels=3
    )

