# Graph Creation Benchmarks

This directory contains benchmarking scripts to profile the execution time and performance bottlenecks during graph creation.

## Requirements

The benchmarks rely on `pyinstrument` and `matplotlib` to generate call-stack flamegraphs and scaling plots.
Make sure you have installed the development dependencies:

```bash
uv sync --all-extras --dev
# or specifically
uv add --dev pyinstrument
```

## 1. Call Stack Flamegraphs (`graph_creation_flamegraph.py`)

You can run the script from the root of the project to profile graph creation for a specific archetype and grid size. Because the script uses the `tests` utility module, run it via the Python module syntax. By default, it will open an interactive HTML flamegraph in your browser!

```bash
uv run python -m tests.benchmarks.graph_creation_flamegraph
```

### Options

- `--N <int>`: Set the size of the input grid ($N \times N$). Default is `425` which produces ~180k points (a roughly 10s baseline for the `keisler` graph).
- `--archetype <name>`: The archetype graph to create. Options are `keisler`, `oskarsson_hierarchical`, and `graphcast`.
- `--console`: Print the profiling hierarchy to the console instead of opening the HTML flamegraph in the browser.
- `--save-flamegraph [FILENAME]`: Saves the interactive HTML flamegraph to disk and exits. If no filename is provided, defaults to `pyinstrument_profile.html`.

**Examples:**

Profile the hierarchical archetype with $200 \times 200$ points (opens in browser):
```bash
uv run python -m tests.benchmarks.graph_creation_flamegraph --N 200 --archetype oskarsson_hierarchical
```

Save the flamegraph to a custom file without opening a server:
```bash
uv run python -m tests.benchmarks.graph_creation_flamegraph --N 425 --save-flamegraph my_profile.html
```

## 2. Runtime Scaling Plot (`graph_creation_scaling.py`)

This script runs the graph creation process across a range of different grid sizes and plots the execution time versus the number of input nodes. This helps visualize how the algorithm's runtime scales as the coordinate size increases.

```bash
uv run python -m tests.benchmarks.graph_creation_scaling
```

### Options

- `--min-N <int>`: The minimum grid size N ($N \times N$ nodes). Default: 50
- `--max-N <int>`: The maximum grid size N ($N \times N$ nodes). Default: 400
- `--num-steps <int>`: Number of intermediate grid sizes to test between min and max. Default: 8
- `--archetype <name>`: The archetype graph to create. Options are `keisler`, `oskarsson_hierarchical`, and `graphcast`.
- `--output <path>`: The file path to save the generated plot. Default: `scaling_plot.png`
- `--show`: Opens a matplotlib interactive window to display the plot after benchmarking.

**Examples:**

Test scaling from $100 \times 100$ to $500 \times 500$ and open the plot interactively:
```bash
uv run python -m tests.benchmarks.graph_creation_scaling --min-N 100 --max-N 500 --num-steps 10 --show
```
