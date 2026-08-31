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
- `--repetitions <int>`: Time each grid size this many times and report the **median**. Default: 1
- `--track-memory`: Also record peak memory usage (via `tracemalloc`) for each grid size.
- `--output-plot-runtime <path>`: File path for the runtime plot. Default: `runtime_scaling.png`
- `--output-plot-memory <path>`: File path for the memory plot (requires `--track-memory`).
- `--output-json <path>`: Save the raw results as JSON (consumed by `compare.py`, below).
- `--show`: Opens a matplotlib interactive window to display the plot after benchmarking.

**Examples:**

Test scaling from $100 \times 100$ to $500 \times 500$ and open the plot interactively:
```bash
uv run python -m tests.benchmarks.graph_creation_scaling --min-N 100 --max-N 500 --num-steps 10 --show
```

#### A note on `--repetitions`

A single timing is surprisingly noisy — on a shared machine the same code
timed twice can easily differ by 10% or more, which is enough to look like a
performance regression when it isn't. Passing `--repetitions 5` times each
grid size five times and reports the median, which discards those one-off
outliers; the individual timings are kept in the JSON output (as
`runtime_samples`) if you want to inspect the spread.

Only the *timed* runs are repeated. Peak memory is measured once per grid
size, because for a fixed input it is deterministic, and measuring it is
comparatively expensive (`tracemalloc` slows graph creation down by roughly
5x, so it is also kept out of the timed runs entirely).

## 3. CI Regression Comparison (`compare.py`)

`compare.py` compares two `--output-json` files produced by
`graph_creation_scaling.py` — typically one from `main` and one from a pull
request — and renders a Markdown table of the relative runtime and peak-memory
change per grid size. This is what powers the automated benchmark regression
check in CI (see
[\#144](https://github.com/mllam/weather-model-graphs/issues/144)).

```bash
uv run python -m tests.benchmarks.compare main.json pr.json --threshold-pct 0.1
```

### Options

- `--threshold-pct <float>`: Flag a grid size whose runtime or peak memory grows by more than this percentage. Default: 0.1
- `--baseline-label <str>` / `--contender-label <str>`: Column headings for the two runs. Defaults: `main` / `PR`
- `--output <path>`: Also write the Markdown report to a file (used for the PR comment).
- `--fail-on-regression`: Exit non-zero if anything regressed. Off by default, so the check stays informational.
