# Graph Construction Benchmarks

This directory contains scripts to benchmark the performance of graph generation across different NetworkX backends (e.g., standard CPU `networkx` vs. GPU-accelerated `nx-cugraph`).

## Purpose
Currently, swapping to the `nx-cugraph` backend yields significant speedups for downstream analytical algorithms (like PageRank). However, the initial graph *construction* time remains largely identical between backends because the current generation logic relies on CPU-bound `scipy.spatial` distance calculations and iterative loops.

## Future Evolution
These benchmarks serve as the baseline for future optimization work. As we evolve `weather-model-graphs`, we will use these metrics to identify exactly where to implement natively vectorized operations or GPU-accelerated nearest-neighbor searches to eliminate the initial graph construction bottlenecks.