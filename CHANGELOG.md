# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased](https://github.com/mllam/weather-model-graphs/compare/v0.1.0...HEAD)

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

- Fix wrong number of mesh levels when grid is multiple of refinement factor
  [\#26](https://github.com/mllam/weather-model-graphs/pull/26)
  @joeloskarsson

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
