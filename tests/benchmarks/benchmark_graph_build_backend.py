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
import networkx

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


def _cuda_hardware_summary() -> str:
    """Best-effort CUDA device summary. Never raises; returns a human string."""
    # Prefer CuPy if available (common in RAPIDS CUDA stacks)
    try:
        import cupy as cp  # type: ignore

        if cp.cuda.runtime.getDeviceCount() <= 0:
            return "CUDA       : not detected"
        dev_id = int(cp.cuda.runtime.getDevice())
        props = cp.cuda.runtime.getDeviceProperties(dev_id)
        raw_name = props.get("name", b"")
        name = (
            raw_name.decode("utf-8", errors="ignore")
            if isinstance(raw_name, (bytes, bytearray))
            else str(raw_name)
        )
        major = props.get("major", "?")
        minor = props.get("minor", "?")
        mem_gb = float(props.get("totalGlobalMem", 0)) / (1024**3)
        return f"CUDA       : device {dev_id} - {name} (cc {major}.{minor}, {mem_gb:.1f} GiB)"
    except Exception:
        pass

    # Fallback to PyTorch if available
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return "CUDA       : not detected"
        dev_id = int(torch.cuda.current_device())
        name = torch.cuda.get_device_name(dev_id)
        cap = torch.cuda.get_device_capability(dev_id)
        props = torch.cuda.get_device_properties(dev_id)
        mem_gb = float(getattr(props, "total_memory", 0)) / (1024**3)
        return f"CUDA       : device {dev_id} - {name} (cc {cap[0]}.{cap[1]}, {mem_gb:.1f} GiB)"
    except Exception:
        return "CUDA       : unknown (no cupy/torch or query failed)"


def run_benchmark(nx: int, ny: int, label: str) -> float:
    xy = _make_xy(nx, ny)
    print(f"\n  Graph size : {nx} x {ny}  ({nx*ny:,} data points)")
    
    # 1. Benchmark Graph Construction
    t0 = time.perf_counter()
    g = create_single_level_2d_mesh_graph(xy, nx=nx, ny=ny)
    build_time = time.perf_counter() - t0
    
    # 2. Benchmark heavy algorithm (PageRank) to show cuGraph acceleration
    t_algo_0 = time.perf_counter()
    _ = networkx.pagerank(g)
    algo_time = time.perf_counter() - t_algo_0

    print(f"  Nodes      : {len(g.nodes):,}")
    print(f"  Edges      : {len(g.edges):,}")
    print(f"  Build Time : {build_time:.4f} s  [{label}]")
    print(f"  Algo Time  : {algo_time:.4f} s  (PageRank)")
    
    return build_time


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
    print(_cuda_hardware_summary())

    if cugraph_active and not HAS_CUGRAPH:
        print(
            "\n[WARNING] NX_CUGRAPH_AUTOCONFIG is set but 'nx-cugraph' is not "
            "installed.\nResults will use the standard NetworkX backend.\n"
            "Install via:  pip install nx-cugraph-cu12\n"
        )

    for grid in [(50, 50), (100, 100), (250, 400)]:
        run_benchmark(*grid, label=label)

    print("\nDone.")