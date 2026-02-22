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
    Test that flat mesh creation logic properly handles nodes with tuple positions.
    When networkx uses the cugraph backend (NX_CUGRAPH_AUTOCONFIG=True),
    positions can be returned as tuples instead of numpy arrays.
    """
    
    # We patch networkx.grid_2d_graph to yield tuples when requested
    original_grid = networkx.grid_2d_graph
    
    def mock_grid_2d_graph(*args, **kwargs):
        g = original_grid(*args, **kwargs)
        # Instead of wrapping the dict assignment, we'll patch the graph directly
        # But we must do it AFTER create_single_level_2d_mesh_graph sets the 'pos'
        # So we patch networkx.DiGraph where the math actually happens
        return g
        
    monkeypatch.setattr(networkx, "grid_2d_graph", mock_grid_2d_graph)

    # Actually, we can just patch `networkx.DiGraph` to intercept the graph creation
    # right before the distance math:
    original_digraph = networkx.DiGraph
    
    def mock_digraph(g, *args, **kwargs):
        dg = original_digraph(g, *args, **kwargs)
        # Convert all 'pos' attributes to tuples to simulate cugraph
        for node in dg.nodes:
            if "pos" in dg.nodes[node]:
                dg.nodes[node]["pos"] = tuple(dg.nodes[node]["pos"])
        return dg
        
    monkeypatch.setattr(networkx, "DiGraph", mock_digraph)
    
    # Run the function. If this raises a TypeError (like unsupported operand type(s) for -: 'tuple' and 'tuple')
    # then our test failed and the PR fix was not applied or was deficient.
    # Run the function. Wait, let it bubble up exceptions.
    dg = create_single_level_2d_mesh_graph(xy_data, nx=2, ny=2)

    # Verify attributes were calculated reasonably
    for u, v, data in dg.edges(data=True):
        assert "len" in data
        assert "vdiff" in data
        assert isinstance(data["len"], (float, np.floating))
        assert isinstance(data["vdiff"], (np.ndarray, list, tuple))

def test_flat_mesh_supports_list_positions(monkeypatch):
    """
    Test that flat mesh creation logic properly handles nodes with list positions.
    """
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
    """
    Test that flat mesh creation logic properly handles nodes with standard numpy arrays.
    Provides a baseline that nothing broke for standard environments.
    """
    dg = create_single_level_2d_mesh_graph(xy_data, nx=2, ny=2)

    
def test_hierarchical_mesh_supports_tuple_positions(monkeypatch):
    """
    Test that hierarchical mesh creation logic properly handles nodes with tuple positions.
    When networkx uses the cugraph backend (NX_CUGRAPH_AUTOCONFIG=True),
    positions can be returned as tuples instead of numpy arrays.
    """
    # For hierarchical, the math happens directly on G_down.
    # We can patch networkx.relabel_nodes which is called just before the up/down links are built.
    
    # Actually, G_to and G_from are passed around, and up/down edges are built.
    # What's an easy way to intercept? We can patch networkx.grid_2d_graph
    # which is what everything starts with (mk_2d_graph). Wait, no, the bug happens
    # during inter-level mesh edges creation where the graphs G_to and G_from are used.
    # If we just replace 'pos' with tuples right after the base graphs are made, it will cascade.
    
    original_mk_2d_graph = networkx.grid_2d_graph
    
    def mock_grid_2d_graph(nx, ny, *args, **kwargs):
        g = original_mk_2d_graph(nx, ny, *args, **kwargs)
        # However, POS is assigned AFTER grid_2d_graph returns.
        return g

    original_digraph = networkx.DiGraph
    
    def mock_digraph(g=None, *args, **kwargs):
        dg = original_digraph(g, *args, **kwargs)
        # Convert all 'pos' attributes to tuples to simulate cugraph
        if g is not None:
            for node in dg.nodes:
                if "pos" in dg.nodes[node]:
                    dg.nodes[node]["pos"] = tuple(dg.nodes[node]["pos"])
        return dg
        
    monkeypatch.setattr(networkx, "DiGraph", mock_digraph)

    # A 100x100 km grid - large enough to allow at least 2 hierarchical levels
    # with mesh_node_distance=10.0 and level_refinement_factor=3.0
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    xx, yy = np.meshgrid(x, y)
    xy_large = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Generate enough for multiple levels so the up/down logic is hit
    components = create_hierarchical_multiscale_mesh_graph(
        xy_large,
        mesh_node_distance=10.0,
        level_refinement_factor=3.0,
        max_num_levels=3
    )

