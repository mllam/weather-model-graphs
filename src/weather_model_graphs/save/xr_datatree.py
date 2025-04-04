import copy
from typing import Any, Dict, List, Union

import networkx as nx
import numpy as np
import xarray as xr

from ..split import MissingEdgeAttributeError, split_graph_by_edge_attribute
from .defaults import DEFAULT_EDGE_FEATURES, DEFAULT_NODE_FEATURES

VECTOR_FEATURE_NAME_FORMAT = "{attr}:{i}"


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
            edge_feature_labels.extend(
                [
                    VECTOR_FEATURE_NAME_FORMAT.format(attr=attr, i=i)
                    for i in range(vals.shape[1])
                ]
            )
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


def _extract_node_features_to_data_array(
    graph: nx.DiGraph, node_feature_attrs: List
) -> xr.DataArray:
    """
    From the graph, extract the node features and return them as a dataset.
    This function uses the node labels as indexes for the nodes and assumes
    these node labels can be cast as integers.

    Parameters
    ----------
    graph : nx.DiGraph
        A directed graph where:
        - Each node label must be convertible to an integer (e.g. int, str of int).
        - Nodes must have the feature attributes specified in `node_feature_attrs`.
    node_feature_attrs : list
        List of node attributes to extract and include as features in the dataset.

    Returns
    -------
    xr.DataArray
        A DataArray with:
        - 'node_features' : (node_index, node_feature) → node attributes
        - Coordinates:
            * 'node_index' : unique index per node (integer-converted node labels)
            * 'node_feature' : node feature attributes
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

    node_feature_values = []
    node_feature_labels = []
    for attr in node_feature_attrs:
        if attr not in graph.nodes[nodes[0]]:
            raise MissingEdgeAttributeError(
                f"Missing node attribute '{attr}' in the graph."
            )
        vals = np.array([graph.nodes[n][attr] for n in nodes])
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
            node_feature_labels.append(attr)
        elif vals.ndim == 2:
            # If the attribute is a list, we need to stack them
            node_feature_labels.extend(
                [
                    VECTOR_FEATURE_NAME_FORMAT.format(attr=attr, i=i)
                    for i in range(vals.shape[1])
                ]
            )
        else:
            raise ValueError(
                f"Node attribute '{attr}' has unsupported shape {vals.shape}."
            )
        node_feature_values.append(vals)

    node_features = np.concatenate(node_feature_values, axis=1)

    da = xr.DataArray(
        data=node_features,
        dims=["node_index", "node_feature"],
        coords={
            "node_index": list(node_to_idx.keys()),
            "node_feature": node_feature_labels,
        },
    )

    return da


def _move_common_coordinate_to_root(dt: xr.DataTree, coord_name: str) -> xr.DataTree:
    """
    Traverse the DataTree and move the specified coordinate to the root if it is identical
    across all datasets. This is useful for cleaning up the DataTree structure.

    Parameters
    ----------
    dt : xr.DataTree
        The DataTree to traverse and modify.
    coord_name : str
        The name of the coordinate to check and potentially move.

    Returns
    -------
    xr.DataTree
        The modified DataTree with the specified coordinate moved to the root if identical.
    """

    def _collect_all_coord_data_arrays(dt, coord_name):
        for child_dt in dt.children.values():
            if not child_dt.is_hollow or child_dt.is_leaf:
                ds = child_dt.dataset
                if coord_name in ds.coords:
                    yield ds.coords[coord_name]

            for da_coord in _collect_all_coord_data_arrays(child_dt, coord_name):
                yield da_coord

    das_coord_values = list(_collect_all_coord_data_arrays(dt, coord_name))
    if len(das_coord_values) == 0:
        return dt

    def _drop_vars_in_tree(dt, var_names):
        if var_names in dt.dataset.data_vars or var_names in dt.dataset.coords:
            ds = dt.to_dataset()
            ds = ds.drop_vars(var_names)
            dt.dataset = ds
            return dt

        new_children = {}
        for child_name, child_dt in dt.children.items():
            new_children[child_name] = _drop_vars_in_tree(child_dt, var_names)
        dt.children = new_children

        return dt

    # Check if all values are identical, if so move the coordinate to the root
    da_first_coord = das_coord_values[0]

    if all(np.array_equal(da_first_coord, da) for da in das_coord_values):
        dt = _drop_vars_in_tree(dt, coord_name)

        ds = dt.to_dataset()
        ds[coord_name] = da_first_coord
        dt.dataset = ds

    return dt


def graph_to_datatree(
    graph: nx.DiGraph,
    split_by: Union[str, Dict[str, Any]],
    node_feature_attrs: List = DEFAULT_NODE_FEATURES,
    edge_feature_attrs: List = DEFAULT_EDGE_FEATURES,
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

    def _extract_within_subgraph(graph, rules, split_path_attrs={}):
        if isinstance(rules, str):
            rules = {rules: {}}
        for attr, subrules_by_values in rules.items():
            for attr_val, subgraph in split_graph_by_edge_attribute(
                graph=graph, attr=attr
            ).items():
                subrule = subrules_by_values.get(attr_val, None)
                if subrule is None:
                    ds_subgraph = extract_edges_to_dataset(
                        graph=subgraph,
                        edge_attrs=edge_feature_attrs,
                        edge_id_attr=edge_id_attr,
                    )
                    # set the edge features (and their values) that have been
                    # split on as attributes on the dataset (so we can
                    # reference these later)
                    ds_subgraph.attrs.update(split_path_attrs)
                    ds_subgraph.attrs[attr] = attr_val
                    children = {}
                else:
                    subgraph_attrs = copy.deepcopy(split_path_attrs)
                    subgraph_attrs[attr] = attr_val
                    ds_subgraph = xr.Dataset()
                    # Recursively populate child tree
                    children = {}
                    for subgraph_path, subgraph_ds in _extract_within_subgraph(
                        graph=subgraph, rules=subrule, split_path_attrs=subgraph_attrs
                    ):
                        children[subgraph_path] = subgraph_ds

                dt = xr.DataTree(dataset=ds_subgraph)
                dt.children = children
                yield str(attr_val), dt

    dt = xr.DataTree(name="root")
    subgraph_datasets = {}
    for subgraph_identifier, subgraph_ds in _extract_within_subgraph(
        graph=graph, rules=split_by
    ):
        subgraph_datasets[subgraph_identifier] = subgraph_ds
    dt.children = subgraph_datasets

    # make `node` and `edge_feature` coordinates global by traversing the whole
    # tree to check that each dataset has the same values for these
    # coordinates, then drop the coordinates and add them to the root. This
    # just makes the resulting datatree a bit cleaner

    for c in ["node", "edge_feature"]:
        dt = _move_common_coordinate_to_root(dt=dt, coord_name=c)

    # dt["node_features"] = _extract_node_features_to_data_array(
    #     graph=graph, node_feature_attrs=node_feature_attrs
    # )

    return dt
