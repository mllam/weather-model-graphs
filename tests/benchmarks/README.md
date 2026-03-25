# Graph Creation Benchmarks

This directory contains benchmarking scripts to profile the execution time and performance bottlenecks during graph creation.

## Requirements

The benchmarks rely on `pyinstrument` to generate call-stack flamegraphs and timing hierarchies.
Make sure you have installed the development dependencies:

```bash
uv sync --all-extras --dev
# or specifically
uv add --dev pyinstrument
```

## Running the Benchmark

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

Print a fast profile stack to the console instead of the browser:
```bash
uv run python -m tests.benchmarks.graph_creation_flamegraph --N 100 --console
```

Save the flamegraph to a custom file without opening a server:
```bash
uv run python -m tests.benchmarks.graph_creation_flamegraph --N 425 --save-flamegraph my_profile.html
```
