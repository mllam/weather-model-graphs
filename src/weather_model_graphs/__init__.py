import importlib.metadata

try:
    __version__ = importlib.metadata.version("weather-model-graphs")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from . import create, save, visualise
from .backends import (
    GraphBackend,
    NetworkXBackend,
    PyGBackend,
    DGLBackend,
    get_backend,
    convert_graph,
)
from .filtering import filter_graph
from .networkx_utils import (
    replace_node_labels_with_unique_ids,
    split_graph_by_edge_attribute,
)
