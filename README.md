# weather-model-graphs

[![tests](https://github.com/mllam/weather-model-graphs/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/mllam/weather-model-graphs/actions/workflows/ci-tests.yml) [![linting](https://github.com/mllam/weather-model-graphs/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/mllam/weather-model-graphs/actions/workflows/pre-commit.yml) [![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://mllam.github.io/weather-model-graphs)

`weather-model-graphs` is a package for creating, visualising and storing graphs used in message-passing graph-based data-driven weather models.

The package is designed to use `networkx.DiGraph` objects as the primary data structure for the graph representation right until the graph is to be stored on disk into a specific format.
This makes the graph generation process modular (every step outputs a `networkx.DiGraph`), easy to debug (visualise the graph at any step) and allows output to different file-formats and file-structures to be easily implemented. More details are given in the [background](https://mllam.github.io/weather-model-graphs/background.html) and [design](https://mllam.github.io/weather-model-graphs/design.html) section of the online [documentation](https://mllam.github.io/weather-model-graphs/).


## Installation

If you simply want to install and use `weather-model-graphs` as-is you can install the most recent release directly from pypi with pip

```bash
python -m pip install weather-model-graphs
```

If you want to be able to save to pytorch-geometric data-structure used in
[neural-lam](https://github.com/mllam/neural-lam) then you will need to install
pytorch and pytorch-geometric too. This can be done by with the `pytorch`
optional extra in `weather-model-graphs`:

```bash
python -m pip install weather-model-graphs[pytorch]
```

This will install the CPU version of pytorch by default. If you want to install
a GPU variant you should [install that
first](https://pytorch.org/get-started/locally/) before installing
`weather-model-graphs`.


## Developing `weather-model-graphs`

The easiest way to work on developing `weather-model-graphs` is to fork the [main repo](https://github.com/mllam/weather-model-graphs) under your github account, clone this repo locally, install [pdm](https://pdm-project.org/en/latest/), create a venv with pdm and then install `weather-model-graphs` (and all development dependencies):

```
git clone https://github.com/<your-github-username>/weather-model-graphs
cd weather-model-graphs
pdm venv create
pdm use --venv in-project
pdm install --dev
```

All linting is handeled with [pre-commit](https://pre-commit.com/) which you can ensure automatically executes on all commits by installing the git hook:

```bash
pdm run pre-commit install
```

Then branch, commit, push and create a pull-request!


### pytorch support

cpu only:

```bash
PIP_INDEX_URL=https://download.pytorch.org/whl/cpu pdm install --group pytorch
```

gpu support (see https://pytorch.org/get-started/locally/#linux-pip for older versions of CUDA):


```bash
pdm install --group pytorch
```

# Usage

The best way to understand how to use `weather-model-graphs` is to look at the [documentation](https://mllam.github.io/weather-model-graphs) (which are executable Jupyter notebooks!), to have look at the tests in [tests/](tests/) or simply to read through the source code.

## Example, Keisler 2021 flat graph architecture

```python
import numpy as np
import weather_model_graphs as wmg

# define your (x,y) grid coodinates
xy_grid = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32))
xy_grid = np.stack(xy_grid, axis=0)

# create the full graph
graph = wmg.create.archetype.create_keisler_graph(xy_grid=xy_grid)

# split the graph by component
graph_components = wmg.split_graph_by_edge_attribute(graph=graph, attr='component')

# save the graph components to disk in pytorch-geometric format
for component, graph in graph_components.items():
    wmg.save.to_pyg(graph=graph, name=component, output_directory=".")
```

# Documentation

The documentation is built using [Jupyter Book](https://jupyterbook.org/intro.html) and can be found at [https://mllam.github.io/weather-model-graphs](https://mllam.github.io/weather-model-graphs). This includes background on graph-based weather models, the design principles of `weather-model-graphs` and how to use it to create your own graph architectures.
