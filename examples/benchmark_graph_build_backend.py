"""
benchmark_graph_build_backend.py
---------------------------------
A simple benchmark script to measure graph creation speed with and without
the NetworkX cuGraph backend (NX_CUGRAPH_AUTOCONFIG=True).

Usage
-----
  Standard backend:
    python examples/benchmark_graph_build_backend.py

  cuGraph backend (requires nx-cugraph and a CUDA-capable GPU):
    NX_CUGRAPH_AUTOCONFIG=True python examples/benchmark_graph_build_backend.py

References
----------
  - nx-cugraph: https://github.com/rapidsai/nx-cugraph
  - neural-lam issue #164: https://github.com/mllam/neural-lam/issues/164
"""
import os
import time

import numpy as np

try:
    import nx_cugraph  # noqa: F401

    HAS_CUGRAPH = True
except ImportError:
    HAS_CUGRAPH = False

from weather_model_graphs.create.mesh.mesh import create_single_level_2d_mesh_graph


def _make_xy(nx: int, ny: int, extent: float = 100.0) -> np.ndarray:
    """Create a regular (nx*ny, 2) grid spanning [0, extent]^2."""
    x = np.linspace(0, extent, nx)
    y = np.linspace(0, extent, ny)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def run_benchmark(nx: int, ny: int, label: str) -> float:
    xy = _make_xy(nx, ny)
    print(f"\n  Graph size : {nx} x {ny}  ({nx*ny:,} data points)")
    t0 = time.perf_counter()
    g = create_single_level_2d_mesh_graph(xy, nx=nx, ny=ny)
    elapsed = time.perf_counter() - t0
    print(f"  Nodes      : {len(g.nodes):,}")
    print(f"  Edges      : {len(g.edges):,}")
    print(f"  Time       : {elapsed:.4f} s  [{label}]")
    return elapsed


if __name__ == "__main__":
    cugraph_active = os.environ.get("NX_CUGRAPH_AUTOCONFIG", "").lower() in (
        "1",
        "true",
        "yes",
    )
    label = "cuGraph" if cugraph_active else "standard NetworkX"

    print("=" * 60)
    print("  weather-model-graphs  –  Graph Build Backend Benchmark")
    print("=" * 60)
    print(f"\nBackend : {label}")

    if cugraph_active and not HAS_CUGRAPH:
        print(
            "\n[WARNING] NX_CUGRAPH_AUTOCONFIG is set but 'nx-cugraph' is not "
            "installed.\nResults will use the standard NetworkX backend.\n"
            "Install via:  pip install nx-cugraph-cu12\n"
        )

    for grid in [(50, 50), (100, 100)]:
        run_benchmark(*grid, label=label)

    print("\nDone.")
