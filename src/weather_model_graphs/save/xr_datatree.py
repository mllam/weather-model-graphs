from typing import Any, Dict, List, Union

import networkx as nx
import numpy as np
import xarray as xr

from ..networkx_utils import MissingEdgeAttributeError, split_graph_by_edge_attribute


def extract_edges_to_dataset(
    graph: nx.DiGraph, edge_attrs: List, edge_id_attr: str
) -> xr.Dataset:
    """
    Extract edge indices and features from a NetworkX DiGraph into an xarray.Dataset.

    This function assumes that all node labels can be uniquely and deterministically
    converted into integers. These integer-converted node labels will be used directly
    as edge indices (i.e., no remapping or reindexing is performed).

    Parameters
    ----------
    graph : nx.DiGraph
        A directed graph where:
        - Each node label must be convertible to an integer (e.g. int, str of int).
        - Edges may have feature attributes (optional, must be uniform).
    edge_attrs : list
        List of edge attributes to extract and include as features in the dataset.
    edge_id_attr : str
        The name of the edge attribute to use as the unique identifier for each
        edge. Using this ensures that we can use a globally unique id for each
        edge across the entire graph.

    Returns
    -------
    xr.Dataset
        A dataset with:
        - 'adjacency_list' : (edge_index, node) → integer node indices
        - 'edge_features'  : (edge_index, edge_feature) → optional edge attributes
        - Coordinates:
            * 'edge_index' : unique index per edge
            * 'node' : ['src_index', 'dst_index']
            * 'edge_feature' : (if edge features exist)
        - 'edge_id' : unique edge identifier (if edge features exist)

    Raises
    ------
    ValueError
        If any node label cannot be converted into an integer.
    """
    try:
        # Map node labels to integers (check all)
        nodes = list(graph.nodes)
        node_labels = [int(n) for n in nodes]  # This will raise ValueError if invalid
        node_to_idx = dict(zip(node_labels, nodes))
    except ValueError:
        raise ValueError(
            "All node labels must be convertible to integers (e.g. 0, '1', etc.)"
        )

    if len(nodes) == 0:
        raise ValueError("Graph must contain at least one node.")

    # Build adjacency list
    edges = list(graph.edges)
    adjacency_list = np.array(
        [[node_to_idx[u], node_to_idx[v]] for u, v in edges], dtype=np.int64
    )

    if len(edges) == 0:
        raise ValueError("Graph must contain at least one edge.")

    edge_feature_values = []
    edge_feature_labels = []
    for attr in edge_attrs:
        if attr not in graph.edges[edges[0]]:
            raise MissingEdgeAttributeError(
                f"Missing edge attribute '{attr}' in the graph."
            )
        vals = np.array([graph.edges[e][attr] for e in edges])
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
            edge_feature_labels.append(attr)
        elif vals.ndim == 2:
            # If the attribute is a list, we need to stack them
            edge_feature_labels.extend([f"{attr}_{i}" for i in range(vals.shape[1])])
        else:
            raise ValueError(
                f"Edge attribute '{attr}' has unsupported shape {vals.shape}."
            )
        edge_feature_values.append(vals)

    edge_features = np.concatenate(edge_feature_values, axis=1)

    edge_indexes = np.array(
        [graph.edges[e][edge_id_attr] for e in edges], dtype=np.int64
    )

    ds = xr.Dataset(
        data_vars={
            "adjacency_list": (["edge_index", "node"], adjacency_list),
        },
        coords={
            "edge_index": edge_indexes,
            "node": ["src_index", "dst_index"],
        },
    )

    if edge_features is not None and edge_feature_labels:
        ds["edge_features"] = (["edge_index", "edge_feature"], edge_features)
        ds = ds.assign_coords(edge_feature=edge_feature_labels)

    return ds


DEFAULT_SPLIT_RULES = {
    "component": {
        "m2m": {"direction": {"same": "level", "up": "levels", "down": "levels"}},
    }
}


def graph_to_datatree(
    graph: nx.DiGraph,
    split_by: Union[str, Dict[str, Any]],
    edge_feature_attrs: List = ["len", "vdiff"],
    edge_id_attr="edge_id",
) -> xr.DataTree:
    """
    Recursively split a NetworkX DiGraph based on node attributes and
    store edge-related data in a hierarchical xarray.DataTree.

    Uses `extract_edges_to_dataset()` to extract edge indices and features
    at the leaf level.

    Parameters
    ----------
    graph : nx.DiGraph
        A directed acyclic graph (DAG) with integer-convertible node labels.

    split_by : Union[str, Dict[str, Any]]
        A string or a nested dictionary defining how to split the graph.

        If a string, it is treated as a single edge attribute to split by. I.e.
        a common choice will be to split the global graph by the `component`
        edge attribute (which is used for denoting the `g2m`, `m2m`, `m2g`
        components).

        If a dictionary, then the graph will be split recursively. Here
        alternate keys in the hierarchy of the dictionary are the edge
        attributes to split by, the next level keys select the subgraphs (by
        value of the attribute) to split further. For example, the structure
        below will first split by the `component` edge attribute, it will then
        further split the subgraph which has `component == m2m` by the
        `direction` edge attribute, and finally split the subgraphs with
        `direction == same` by the `level` edge attribute, and the subgraphs
        with `direction == up` and `direction == down` by the `levels` edge
        attribute.

        {
            "component": {
                "m2m": {
                    "direction": {
                        "same": "level",
                        "up": "levels",
                        "down": "levels"
                    }
                },
            }
        }

    Returns
    -------
    xr.DataTree
        A hierarchical DataTree where each leaf node contains only edge-related data.
    """

    def _extract_within_subgraph(graph, rules, path):
        if isinstance(rules, str):
            rules = {rules: {}}
        for attr, subrules_by_values in rules.items():
            for attr_val, subgraph in split_graph_by_edge_attribute(
                graph=graph, attr=attr
            ).items():
                subrule = subrules_by_values.get(attr_val, None)
                subpath = f"{path}/{attr_val}"
                if subrule is None:
                    ds_subgraph = extract_edges_to_dataset(
                        graph=subgraph,
                        edge_attrs=edge_feature_attrs,
                        edge_id_attr=edge_id_attr,
                    )
                    yield subpath, ds_subgraph
                else:
                    # Recursively populate child tree
                    for subgraph_path, subgraph_ds in _extract_within_subgraph(
                        graph=subgraph, rules=subrule, path=subpath
                    ):
                        yield subgraph_path, subgraph_ds

    subgraph_datasets = {}
    for subgraph_path, subgraph_ds in _extract_within_subgraph(
        graph=graph, rules=split_by, path=""
    ):
        subgraph_datasets[subgraph_path] = subgraph_ds

    # make `node` and `edge_feature` coordinates global by traversing the whole
    # tree to check that each dataset has the same values for these
    # coordinates, then drop the coordinates and add them to the root. This
    # just makes the resulting datatree a bit cleaner

    common_coords = dict()

    def _check_and_remove_common_coords(ds):
        for c in ["node", "edge_feature"]:
            if c in ds.coords:
                if c in common_coords:
                    if not np.array_equal(common_coords[c], ds.coords[c].values):
                        raise ValueError(
                            f"Coordinate '{c}' is not the same across all datasets."
                        )
                else:
                    common_coords[c] = ds.coords[c].values
                ds = ds.drop_vars(c)
        return ds

    for path, ds in subgraph_datasets.items():
        subgraph_datasets[path] = _check_and_remove_common_coords(ds)

    tree = xr.DataTree(name="root", dataset=xr.Dataset(coords=common_coords))
    for path, subgraph_ds in subgraph_datasets.items():
        tree[path] = subgraph_ds

    return tree
