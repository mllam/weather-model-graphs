from collections import defaultdict

import networkx as nx
import xarray as xr

VECTOR_FEATURE_NAME_FORMAT = "{attr}:{i}"


def collect_datasets(tree):
    datasets = []
    if tree.ds is not None:
        datasets.append(tree.ds)
    for child in tree.children.values():
        datasets.extend(collect_datasets(child))
    return datasets


def datatree_to_graph(dt: xr.DataTree):
    """
    Create a graph from an xarray datatree. The datatree should consist of
    datasets that contain the following variables:
    - adjacency_list: The adjacency list of the graph.
    - edge_features: The edge features of the graph.

    """

    ds = xr.merge(collect_datasets(dt))
    graph = nx.DiGraph()
    adjacency_da = ds.adjacency_list  # DataArray for adjacency list

    for edge_index in ds.edge_index.values:
        edge_coords = adjacency_da.sel(edge_index=edge_index)
        src_index = edge_coords.sel(node="src_index").item()
        dst_index = edge_coords.sel(node="dst_index").item()

        if src_index == dst_index:
            raise ValueError("Self-loops are not allowed in the graph.")

        graph.add_edge(src_index, dst_index)

        # Add edge features
        if "edge_features" in ds:
            da_feats = ds.edge_features.sel(edge_index=edge_index)
            edge_attrs = dict(zip(da_feats.edge_feature.values, da_feats.values))

            # Combine features with ":" in their names
            combined_features = defaultdict(list)
            for key, value in edge_attrs.items():
                if ":" in key:
                    base_key, index = key.split(":")
                    combined_features[base_key].append((int(index), value))
                else:
                    graph.edges[src_index, dst_index][key] = value

            for base_key, values in combined_features.items():
                # Sort by index and combine into a list
                sorted_values = [v for _, v in sorted(values)]
                graph.edges[src_index, dst_index][base_key] = sorted_values

    # Add node features
    if "node_features" in ds:
        node_features = ds.node_features
        for node_index in node_features.node_index.values:
            node_attrs = dict(
                zip(
                    node_features.node_feature.values,
                    node_features.sel(node_index=node_index).values,
                )
            )

            # combine features with ":" in their names
            combined_features = defaultdict(list)
            for key, value in node_attrs.items():
                if ":" in key:
                    base_key, index = key.split(":")
                    combined_features[base_key].append((int(index), value))
                else:
                    graph.nodes[node_index][key] = value

            for base_key, values in combined_features.items():
                # Sort by index and combine into a list
                sorted_values = [v for _, v in sorted(values)]
                graph.nodes[node_index][base_key] = sorted_values

    return graph
