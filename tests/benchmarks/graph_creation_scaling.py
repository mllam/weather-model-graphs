import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

import tests.utils as test_utils
import weather_model_graphs as wmg


def main():
    parser = argparse.ArgumentParser(description="Benchmark graph creation scaling.")
    parser.add_argument(
        "--min-N", type=int, default=50, help="Minimum grid size N (NxN nodes)."
    )
    parser.add_argument(
        "--max-N", type=int, default=400, help="Maximum grid size N (NxN nodes)."
    )
    parser.add_argument(
        "--num-steps", type=int, default=8, help="Number of intermediate steps."
    )
    parser.add_argument(
        "--archetype",
        type=str,
        default="keisler",
        choices=["keisler", "oskarsson_hierarchical", "graphcast"],
        help="Graph archetype to create.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scaling_plot.png",
        help="Path to save the output plot.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show the plot interactively."
    )

    args = parser.parse_args()

    # Generate an array of N values
    Ns = np.linspace(args.min_N, args.max_N, args.num_steps, dtype=int)

    fn_name = f"create_{args.archetype}_graph"
    create_fn = getattr(wmg.create.archetype, fn_name)

    num_nodes_list = []
    times = []

    print(f"Benchmarking scaling for {fn_name}...")
    for n in Ns:
        num_nodes = n * n
        print(f"Testing N={n:4d} ({num_nodes:7d} nodes)...", end="", flush=True)
        xy = test_utils.create_fake_xy(N=n)

        t0 = time.time()
        _ = create_fn(coords=xy)
        t1 = time.time()

        duration = t1 - t0
        print(f" {duration:.3f} seconds.")

        num_nodes_list.append(num_nodes)
        times.append(duration)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(num_nodes_list, times, marker="o", linestyle="-", linewidth=2)

    # Add a reference line for linear scaling (O(N)) fitted to the first point
    ref_linear = [times[0] * (nodes / num_nodes_list[0]) for nodes in num_nodes_list]
    plt.plot(
        num_nodes_list, ref_linear, linestyle="--", color="gray", label="O(N) Reference"
    )

    plt.title(f"Graph Creation Scaling: {args.archetype}")
    plt.xlabel("Number of Input Grid Nodes (N²)")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plt.savefig(args.output)
    print(f"\nPlot saved to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
