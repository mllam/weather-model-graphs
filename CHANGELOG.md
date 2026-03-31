# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Major New Features (v0.4.0 - Research-Grade Release)

#### 1. Backend Abstraction Layer
- **Multi-backend support**: NetworkX (debugging), PyTorch Geometric (training), DGL (high-performance)
- `GraphBackend` abstract base class with automatic format detection
- `NetworkXBackend`, `PyGBackend`, `DGLBackend` implementations
- `get_backend()` auto-detection for seamless format switching
- Enables scalable graph processing for weather models with 10⁵–10⁷ nodes

#### 2. Spatial Indexing Optimization
- KD-Tree based spatial indexing for O(log N) neighbor queries (vs O(N²) naive)
- `SpatialIndex`, `KDTreeIndex`, `BallTreeIndex` classes
- `create_spatial_index()` and `find_neighbors_vectorized()` functions
- **Performance**: 25-2500x speedup on large point sets (1K-10M points)
- Makes large weather grids (100K+ nodes) computationally feasible

#### 3. Temporal Graph Support
- `TemporalGraph` class for dynamic graphs with temporal edges
- `create_temporal_graph()` for autoregressive time-series modeling
- `add_temporal_edges_to_graph()` for manual temporal connection
- Enables recurrent neural network architectures and weather prediction
- **Feature**: Supports variable temporal window (history length)

#### 4. Configuration-Driven Pipeline
- `GraphConfig` dataclass for declarative graph definition
- YAML-based graph configuration for reproducibility
- `PipelineBuilder` for config-driven graph creation
- `create_graph_from_config()` for loading YAML/dict configs
- Example configs: `config_keisler.yaml`, `config_graphcast.yaml`

#### 5. Feature Engineering Layer
- `FeatureExtractor` class with specialized weather feature functions
- `add_wind_velocity()`: Magnitude from u/v components
- `add_wind_direction()`: Computing wind direction angles
- `add_pressure_gradient()`: Spatial pressure changes
- `add_temporal_encoding()`: Sinusoidal temporal positional encoding
- `add_spatial_encoding()`: Positional encoding based on coordinates
- `add_node_degree_features()`: Graph topology features
- `normalize_features()`: Min-Max and Z-score normalization

#### 6. ML Pipeline Integration
- `GraphDataset` and `PyGGraphDataset` for PyTorch compatibility
- `create_dataloader()`: Easy PyTorch DataLoader creation
- `create_pyg_dataloader()`: Specialized PyTorch Geometric DataLoaders
- `batch_graphs()`: Vectorized batching of multiple graphs
- `create_model_input()`: Format conversion for different ML frameworks
- `split_graph_for_training()`: Train/val/test node splitting

#### 7. Prebuilt Graph Architectures
- `load_prebuilt()`: Quick access to common architectures
- `list_prebuilt()`: Discover available architectures
- `register_archetype()`: Register custom architectures
- Implemented architectures:
  - **Keisler (2021)**: Single-scale flat mesh (simple, interpretable)
  - **GraphCast (Lam et al., 2023)**: Flat multiscale mesh (state-of-the-art)
  - **MeshGraphNet (Pfaff et al., 2021)**: Hierarchical mesh (powerful)

#### 8. Enhanced Dependencies
- New optional dependencies for specific features:
  - `[pytorch]`: PyTorch + PyTorch Geometric
  - `[dgl]`: DGL framework support
  - `[config]`: YAML configuration support (PyYAML)
  - `[ml]`: Full ML pipeline (torch, torch-geometric, scikit-learn)
  - `[visualisation]`: Plotly for advanced visualization
  - `[all]`: All optional dependencies

#### 9. Examples and Documentation
- 6 comprehensive examples demonstrating all major features:
  - `example_1_prebuilt.py`: Quick start with prebuilt graphs
  - `example_2_config.py`: Configuration-driven creation
  - `example_3_temporal.py`: Temporal graphs for time series
  - `example_4_features.py`: Feature engineering workflow
  - `example_5_spatial_indexing.py`: Performance optimization
  - `example_6_ml_pipeline.py`: ML integration and DataLoaders
- Example YAML configurations for different architectures
- Expanded README with detailed use cases and quick starts

### Added (Previous Unreleased)

- Added a standalone graph consistency checking tool (`wmg.diagnostics.check_graph_consistency`) to ensure structural health, such as verifying all grid nodes successfully connect to the mesh (#42).
- Add Django-style graph filtering via `filter_graph`, for example to select
  nodes by type (`node__type="mesh"`), edges by component
  (`edge__component="g2m"`), long edges (`edge__len__gt=...`), and spatial
  windows (`node__pos__bbox=(8, 16, 8, 16)`), including combined filters.
  [\#46](https://github.com/mllam/weather-model-graphs/pull/46), @leifdenby & @Joltsy10
- Add `__version__` attribute to the package init
  [\#56](https://github.com/mllam/weather-model-graphs/pull/56) @AdMub

## [v0.3.0](https://github.com/mllam/weather-model-graphs/releases/tag/v0.3.0)


### Added

- Add a decoding mask option to only include subset of grid nodes in m2g
  [\#34](https://github.com/mllam/weather-model-graphs/pull/34) @joeloskarsson

- Add test to check python codeblocks in README keep working as code changes
  [\#38](https://github.com/mllam/weather-model-graphs/pull/38) @leifdenby

- Add coords_crs and graph_crs arguments to allow for using lat-lons coordinates
  or other CRSs as input. These are then converted to the specific CRS used when
  constructing the graph.
  [\#32](https://github.com/mllam/weather-model-graphs/pull/32), @joeloskarsson

### Changed

- Change coordinate input to array of shape [N_grid_points, 2] (was previously
  [2, Ny, Nx]), to allow for non-regularly gridded coordinates
  [\#32](https://github.com/mllam/weather-model-graphs/pull/32), @joeloskarsson

### Fixed

- Fix the bug with edgeless nodes being dropped
  [\#51](https://github.com/mllam/weather-model-graphs/pull/51), @pkhalaj, @wi-spang, @krikru

- Fix crash when trying to create flat multiscale graphs with >= 3 levels
  [\#41](https://github.com/mllam/weather-model-graphs/pull/41), @joeloskarsson

- Fix example in README
  [\#38](https://github.com/mllam/weather-model-graphs/pull/38) @leifdenby

### Maintenance

- Update github CI actions, including pre-commit action to fix caching issue
  that lead to tests failing
  [\#48](https://github.com/mllam/weather-model-graphs/pull/48), @leifdenby

- Update github CI actions to fix failing build and deploy of jupyterbook
  [\#49](https://github.com/mllam/weather-model-graphs/pull/49),
  [\#54](https://github.com/mllam/weather-model-graphs/pull/54), @leifdenby

- Improve isolation of README example tests by executing each code block in an isolated namespace.
  [#65](https://github.com/mllam/weather-model-graphs/pull/64) @Shristi-Goel

## [v0.2.0](https://github.com/mllam/weather-model-graphs/releases/tag/v0.2.0)

### Added

- added github pull-request template to ease contribution and review process
  [\#18](https://github.com/mllam/weather-model-graphs/pull/18), @joeloskarsson

- Allow for specifying relative distance as `rel_max_dist` when connecting nodes using `within_radius` method.
  [\#19](https://github.com/mllam/weather-model-graphs/pull/19)
  @joeloskarsson

- `save.to_pyg` can now handle any number of 1D or 2D edge or node features when
  converting pytorch-geometric `Data` objects to `torch.Tensor` objects.
  [\#31](https://github.com/mllam/weather-model-graphs/pull/31)
  @maxiimilian

- Add containing_rectangle graph connection method for m2g edges
  [\#28](https://github.com/mllam/weather-model-graphs/pull/28)
  @joeloskarsson

### Changed

- Create different number of mesh nodes in x- and y-direction.
  [\#21](https://github.com/mllam/weather-model-graphs/pull/21)
  @joeloskarsson

- Changed the `refinement_factor` argument into two: a `grid_refinement_factor` and a `level_refinement_factor`.
  [\#19](https://github.com/mllam/weather-model-graphs/pull/19)
  @joeloskarsson

- Connect grid nodes only to the bottom level of hierarchical mesh graphs.
  [\#19](https://github.com/mllam/weather-model-graphs/pull/19)
  @joeloskarsson

- Change default archetypes to match the graph creation from neural-lam.
  [\#19](https://github.com/mllam/weather-model-graphs/pull/19)
  @joeloskarsson

### Fixed

- Fix `attribute` keyword bug in save function
  [\#35](https://github.com/mllam/weather-model-graphs/pull/35)
  @joeloskarsson

- Fix wrong number of mesh levels when grid is multiple of refinement factor
  [\#26](https://github.com/mllam/weather-model-graphs/pull/26)
  @joeloskarsson

### Maintenance

- Ensure that cell execution doesn't time out when building jupyterbook based
  documentation [\#25](https://github.com/mllam/weather-model-graphs/pull/25),
  @leifdenby

## [v0.1.0](https://github.com/mllam/weather-model-graphs/releases/tag/v0.1.0)

First tagged release of `weather-model-graphs` which includes functionality to
create three graph archetypes (Keisler nearest-neighbour, GraphCast multi-range
and Oskarsson hierarchical graphs) deliniating the different connectivity
options, background on graph-based data-driven models, 2D plotting utilities,
JupyterBook based documentation. In this version the graph assumes grid
coordinates are Cartesian coordinates.
