import argparse
import json
import time
import tracemalloc
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

import tests.utils as test_utils
import weather_model_graphs as wmg


def run_benchmark(
    min_N: int,
    max_N: int,
    num_steps: int,
    archetype: str,
    track_memory: bool = False,
) -> List[Dict[str, float]]:
    """
    Run the graph creation benchmark over a range of grid sizes.

    Returns a list of dicts with keys:
        "grid_points" (int), "runtime_s" (float), "peak_memory_mb" (float, optional).
    """
    Ns = np.linspace(min_N, max_N, num_steps, dtype=int)
    fn_name = f"create_{archetype}_graph"
    create_fn = getattr(wmg.create.archetype, fn_name)

    results = []

    for n in Ns:
        num_nodes = int(n * n)
        print(f"Testing N={n:4d} ({num_nodes:7d} nodes)...", end="", flush=True)

        xy = test_utils.create_fake_xy(N=n)

        if track_memory:
            tracemalloc.start()

        t0 = time.time()
        graph = create_fn(coords=xy)
        t1 = time.time()
        duration = t1 - t0

        peak_mb = None
        if track_memory:
            _, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)
            tracemalloc.stop()

        print(f" {duration:.3f} seconds.", end="")
        if peak_mb is not None:
            print(f" Peak memory: {peak_mb:.1f} MB")
        else:
            print()

        results.append(
            {
                "grid_points": num_nodes,
                "runtime_s": duration,
                "peak_memory_mb": peak_mb,
            }
        )

    return results


def plot_runtime_scaling(
    results: List[Dict[str, float]], archetype: str, output_path: str
):
    """Create a scaling plot for runtime vs number of grid points."""
    grid_points = [r["grid_points"] for r in results]
    times = [r["runtime_s"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(grid_points, times, marker="o", linestyle="-", linewidth=2)

    # Add O(N) reference line fitted to the first point
    ref_linear = [times[0] * (gp / grid_points[0]) for gp in grid_points]
    plt.plot(
        grid_points, ref_linear, linestyle="--", color="gray", label="O(N) Reference"
    )

    plt.title(f"Graph Creation Runtime Scaling: {archetype}")
    plt.xlabel("Number of Input Grid Nodes")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Runtime scaling plot saved to {output_path}")


def plot_memory_scaling(
    results: List[Dict[str, float]], archetype: str, output_path: str
):
    """Create a scaling plot for peak memory vs number of grid points."""
    # Filter out results without memory data (should not happen if track_memory=True)
    memory_results = [r for r in results if r["peak_memory_mb"] is not None]
    if not memory_results:
        raise ValueError(
            "No memory data available. Run with --track-memory to collect memory profiles."
        )

    grid_points = [r["grid_points"] for r in memory_results]
    memory = [r["peak_memory_mb"] for r in memory_results]

    plt.figure(figsize=(10, 6))
    plt.plot(grid_points, memory, marker="s", linestyle="-", linewidth=2, color="green")

    plt.title(f"Graph Creation Memory Scaling: {archetype}")
    plt.xlabel("Number of Input Grid Nodes")
    plt.ylabel("Peak Memory Usage (MB)")
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Memory scaling plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark graph creation scaling.")
    parser.add_argument(
        "--min-N", type=int, default=50, help="Minimum grid size N (NxN nodes)"
    )
    parser.add_argument(
        "--max-N", type=int, default=400, help="Maximum grid size N (NxN nodes)"
    )
    parser.add_argument(
        "--num-steps", type=int, default=8, help="Number of intermediate steps"
    )
    parser.add_argument(
        "--archetype",
        choices=["keisler", "oskarsson_hierarchical", "graphcast"],
        default="keisler",
        help="Graph archetype to create",
    )
    parser.add_argument(
        "--output-plot-runtime",
        type=str,
        default="runtime_scaling.png",
        help="Output file for runtime plot",
    )
    parser.add_argument(
        "--output-plot-memory",
        type=str,
        help="Output file for memory scaling plot (requires --track-memory)",
    )
    parser.add_argument("--output-json", type=str, help="Save raw results to JSON file")
    parser.add_argument(
        "--track-memory", action="store_true", help="Profile peak memory usage"
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")

    args = parser.parse_args()

    if args.output_plot_memory and not args.track_memory:
        parser.error("--output-plot-memory requires --track-memory")

    results = run_benchmark(
        min_N=args.min_N,
        max_N=args.max_N,
        num_steps=args.num_steps,
        archetype=args.archetype,
        track_memory=args.track_memory,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Raw results saved to {args.output_json}")

    # Always plot runtime (if we have results)
    if results:
        plot_runtime_scaling(results, args.archetype, args.output_plot_runtime)

    if args.output_plot_memory:
        plot_memory_scaling(results, args.archetype, args.output_plot_memory)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
