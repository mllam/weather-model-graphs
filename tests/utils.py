import numpy as np


def create_fake_xy(N=10):
    return create_rectangular_fake_xy(N, N)


def create_rectangular_fake_xy(Nx=10, Ny=20):
    x = np.linspace(0, Nx, Nx)
    y = np.linspace(0, Ny, Ny)
    xy_mesh = np.meshgrid(x, y)
    xy = np.stack([mg_coord.flatten() for mg_coord in xy_mesh], axis=1)
    return xy


def create_fake_irregular_coords(num_grid_points=100):
    """
    Create fake grid points on random coordinates
    """
    rng = np.random.default_rng(seed=42)  # Fixed seed
    # All coordinates in [0,1]^2
    return rng.random((num_grid_points, 2))
