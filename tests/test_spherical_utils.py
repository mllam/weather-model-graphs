import torch
# We import from the new file you just created
from weather_model_graphs.spherical_utils import lat_lon_to_cartesian

def test_north_pole_singularity():
    """Test that different longitudes at the North Pole converge to the same 3D point."""
    lat = torch.tensor([90.0, 90.0])
    lon = torch.tensor([0.0, 180.0])
    
    coords = lat_lon_to_cartesian(lat, lon)
    distance = torch.norm(coords[0] - coords[1])
    
    # The physical distance between them in 3D space should be exactly 0
    assert torch.isclose(distance, torch.tensor(0.0), atol=1e-6), "Distance at the pole must be zero."

def test_anti_meridian_crossing():
    """Test that points across the Date Line calculate physical distance correctly."""
    lat = torch.tensor([0.0, 0.0])
    lon = torch.tensor([179.0, -179.0])
    
    coords = lat_lon_to_cartesian(lat, lon)
    distance = torch.norm(coords[0] - coords[1])
    
    # 2 degrees apart on a unit sphere (chord length)
    expected_distance = 2 * torch.sin(torch.tensor(torch.pi / 180.0))
    assert torch.isclose(distance, expected_distance, atol=1e-5), "Anti-meridian distance calculation failed."