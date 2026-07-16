from __future__ import annotations

import pickle
from pathlib import Path

import networkx
from loguru import logger

try:
    import torch  # noqa: F401
    import torch_geometric  # noqa: F401

    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# Default edge/node attributes serialised for each graph component. Kept as
# tuples (immutable) so they are safe to use directly as function-argument
# defaults, and shared by both the pyg (deprecated) and torch-tensor output
# methods.
DEFAULT_EDGE_FEATURES = ("len", "vdiff")
DEFAULT_NODE_FEATURES = ("pos",)


def to_pickle(graph: networkx.DiGraph, output_directory: str, name: str) -> None:
    """Save a networkx graph to disk as a pickle file.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph to serialise.
    output_directory : str
        Directory to write the file into.
    name : str
        Basename (without extension); the graph is written to
        ``{output_directory}/{name}.pickle``.

    Returns
    -------
    None
    """
    fp = Path(output_directory) / f"{name}.pickle"
    with open(fp, "wb") as f:
        pickle.dump(graph, f)
    logger.info(f"Saved graph to {fp}.")
