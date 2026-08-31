import argparse
import json
import statistics
import time
import tracemalloc
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

import tests.utils as test_utils
import weather_model_graphs as wmg


def _measure_runtimes(create_fn, xy, repetitions: int) -> List[float]:
    """Time ``repetitions`` graph creations, returning each duration in seconds.

    ``tracemalloc`` is deliberately *not* active while timing: it slows graph
    creation down by roughly 5x, which would both inflate the reported
    runtimes and blow the CI time budget once repetitions are involved.
    """
    durations = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        create_fn(coords=xy)
        durations.append(time.perf_counter() - t0)
    return durations


def _measure_peak_memory(create_fn, xy) -> float:
    """Measure peak memory of a single graph creation, in MB.

    Measured once rather than once per repetition: for a fixed input the peak
    allocation is essentially deterministic (measured run-to-run spread well
    under 0.01%), while ``tracemalloc`` costs ~5x in runtime -- so repeating it
    would dominate the benchmark without making the number any better.
    """
    tracemalloc.start()
    try:
        create_fn(coords=xy)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return float(peak) / (1024 * 1024)


def run_benchmark(
    min_N: int,
    max_N: int,
    num_steps: int,
    archetype: str,
    track_memory: bool = False,
    repetitions: int = 1,
) -> List[Dict[str, Any]]:
    """
    Run the graph creation benchmark over a range of grid sizes.

    Each grid size is timed ``repetitions`` times and the *median* is reported,
    so that a single unusually slow (or fast) run doesn't move the result. This
    keeps the PR-vs-base comparison in CI stable enough to act on (see #144).

    Returns a list of dicts with keys:
        "grid_points" (int), "runtime_s" (float, the median),
        "peak_memory_mb" (float, optional), "repetitions" (int),
        "runtime_samples" (list of float, the individual timings).
    """
    if repetitions < 1:
        raise ValueError(f"repetitions must be >= 1, got {repetitions}")

    Ns = np.linspace(min_N, max_N, num_steps, dtype=int)
    fn_name = f"create_{archetype}_graph"
    create_fn = getattr(wmg.create.archetype, fn_name)

    results = []

    for n in Ns:
        num_nodes = int(n * n)  # convert to Python int
        logger.info(
            f"Testing N={n:4d} ({num_nodes:7d} nodes), "
            f"{repetitions} repetition(s)..."
        )

        xy = test_utils.create_fake_xy(N=n)

        durations = _measure_runtimes(create_fn, xy, repetitions)
        duration = statistics.median(durations)

        peak_mb = _measure_peak_memory(create_fn, xy) if track_memory else None

        if repetitions > 1:
            logger.info(
                f" {duration:.3f} seconds (median of {repetitions}: "
                f"min {min(durations):.3f}, max {max(durations):.3f})."
            )
        else:
            logger.info(f" {duration:.3f} seconds.")
        if peak_mb is not None:
            logger.info(f" Peak memory: {peak_mb:.1f} MB")

        results.append(
            {
                "grid_points": num_nodes,
                "runtime_s": duration,
                "peak_memory_mb": peak_mb,
                "repetitions": repetitions,
                "runtime_samples": durations,
            }
        )

    return results


def plot_runtime_scaling(
    results: List[Dict[str, Any]], archetype: str, output_path: str
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

    plt.savefig(output_path)
    logger.info(f"Runtime scaling plot saved to {output_path}")


def plot_memory_scaling(
    results: List[Dict[str, Any]], archetype: str, output_path: str
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
    logger.info(f"Memory scaling plot saved to {output_path}")


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
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of times to time each grid size. The median is reported, "
        "which damps run-to-run noise; peak memory is still measured once, "
        "since it is deterministic for a given input (default: 1)",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")

    args = parser.parse_args()

    if args.output_plot_memory and not args.track_memory:
        parser.error("--output-plot-memory requires --track-memory")

    if args.repetitions < 1:
        parser.error("--repetitions must be >= 1")

    results = run_benchmark(
        min_N=args.min_N,
        max_N=args.max_N,
        num_steps=args.num_steps,
        archetype=args.archetype,
        track_memory=args.track_memory,
        repetitions=args.repetitions,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Raw results saved to {args.output_json}")

    # Always plot runtime (if we have results)
    if results:
        plot_runtime_scaling(results, args.archetype, args.output_plot_runtime)

    if args.output_plot_memory:
        plot_memory_scaling(results, args.archetype, args.output_plot_memory)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
