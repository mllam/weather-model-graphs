# weather-model-graphs

## Installation

```
pdm install
```

### pytorch support

cpu only:

```bash
PIP_INDEX_URL=https://download.pytorch.org/whl/cpu pdm install --group pytorch
```

gpu support (see https://pytorch.org/get-started/locally/#linux-pip for older versions of CUDA):


```bash
pdm install --group pytorch
```
