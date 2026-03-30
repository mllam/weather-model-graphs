import torch

def lat_lon_to_cartesian(lat: torch.Tensor, lon: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """
    Vectorized conversion from Latitude/Longitude (degrees) to 3D Cartesian coordinates.
    Assumes lat is in [-90, 90] and lon is in [-180, 180].
    """
    # Convert degrees to radians
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)

    # Calculate x, y, z components
    x = radius * torch.cos(lat_rad) * torch.cos(lon_rad)
    y = radius * torch.cos(lat_rad) * torch.sin(lon_rad)
    z = radius * torch.sin(lat_rad)

    # Stack into a single tensor of shape (..., 3)
    return torch.stack([x, y, z], dim=-1)