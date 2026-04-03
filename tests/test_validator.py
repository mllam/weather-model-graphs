import torch
import pytest
from weather_model_graphs.validator import validate_geometric_consistency, validate_connectivity_health

def test_spherical_consistency_valid():
    """Test that points perfectly on the unit sphere pass validation."""
    # Create points perfectly on the sphere (Norm = 1)
    # Example: Cardinal points [1,0,0], [0,1,0], [0,0,1]
    valid_pos = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    assert validate_geometric_consistency([valid_pos]) is True

def test_spherical_consistency_invalid():
    """Test that points not on the unit sphere fail validation."""
    # One point is offset (Norm != 1)
    invalid_pos = torch.tensor([
        [1.0, 0.0, 0.0],
        [2.0, 2.0, 2.0]  
    ], dtype=torch.float32)

    # We expect the validator to return False
    assert validate_geometric_consistency([invalid_pos]) is False

def test_connectivity_health_valid():
    """Test that fully connected graphs pass connectivity validation."""
    # Create a fully connected graph with 3 nodes
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1]
    ])  # All pairs connected
    graph_tensors = {
        'm2m_edge_index': [edge_index]
    }

    assert validate_connectivity_health(graph_tensors) is True

def test_connectivity_health_invalid():
    """Test that graphs with isolated nodes fail connectivity validation."""
    # Create edges that reference node 3
    edge_index = torch.tensor([[3, 3], [4, 5]])  
    graph_tensors = {
        'm2m_edge_index': [edge_index]
    }

    # The validator should detect that nodes 0,1,2 are missing
    assert validate_connectivity_health(graph_tensors) is False