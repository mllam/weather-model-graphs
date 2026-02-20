from . import create, save, visualise
from .networkx_utils import (
    replace_node_labels_with_unique_ids,
    split_graph_by_edge_attribute,
)
import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"