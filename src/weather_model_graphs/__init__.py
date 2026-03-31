import importlib.metadata

try:
    __version__ = importlib.metadata.version("weather-model-graphs")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from . import create, save, visualise, backend, spatial_index, temporal, config, features, ml_integration, prebuilt
from .backend import GraphBackend, NetworkXBackend, PyGBackend, DGLBackend, get_backend
from .spatial_index import (
    SpatialIndex,
    KDTreeIndex,
    BallTreeIndex,
    create_spatial_index,
    find_neighbors_vectorized,
)
from .temporal import (
    TemporalGraph,
    create_temporal_graph,
    add_temporal_edges_to_graph,
    unfold_temporal_predictions,
)
from .config import GraphConfig, PipelineBuilder, create_graph_from_config
from .features import (
    FeatureExtractor,
    add_wind_velocity,
    add_wind_direction,
    add_pressure_gradient,
    add_temporal_encoding,
    add_spatial_encoding,
    add_node_degree_features,
    normalize_features,
)
from .ml_integration import (
    GraphDataset,
    PyGGraphDataset,
    create_dataloader,
    create_pyg_dataloader,
    batch_graphs,
    create_model_input,
    split_graph_for_training,
)
from .prebuilt import load_prebuilt, list_prebuilt, register_archetype
from .filtering import filter_graph
from .networkx_utils import (
    replace_node_labels_with_unique_ids,
    split_graph_by_edge_attribute,
)
