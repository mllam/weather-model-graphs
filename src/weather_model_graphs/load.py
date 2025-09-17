from collections import defaultdict

import networkx as nx
import xarray as xr

VECTOR_FEATURE_NAME_FORMAT = "{attr}:{i}"


def collect_datasets(tree):
    datasets = []
    ds = tree.to_dataset()

    if "edge_index" in ds:
        # the attributes set on the subgraph datasets are the attributes (and
        # their values) that were used for splitting. To set these attributes
        # on each edge in the graph when we convert back to at networkx.DiGraph
        # we here broadcast the attribute values for each edge_index.
        das_extra_attrs = []
        attrs = dict(ds.attrs)
        for key, value in attrs.items():
            da_extra_attr = xr.DataArray(value).expand_dims(
                {"edge_index": ds.edge_index, "edge_feature": 1}
            )
            da_extra_attr.coords["edge_feature"] = [key]
            das_extra_attrs.append(da_extra_attr)

            del ds.attrs[key]

        # have to remove `edge_features` variable and `edge_feature` dimension
        # from dataset otherwise we can increase the number of edge features
        # (the coordinate values stays the same)
        da_edge_features = ds.edge_features
        ds = ds.drop_vars(["edge_features", "edge_feature"])
        ds["edge_features"] = xr.concat(
            das_extra_attrs + [da_edge_features], dim="edge_feature"
        )

    if "node_index" in ds:
        # the attributes set on the subgraph datasets are the attributes (and
        # their values) that were used for splitting. To set these attributes
        # on each node in the graph when we convert back to at networkx.DiGraph
        # we here broadcast the attribute values for each node_index.
        das_extra_attrs = []
        attrs = dict(ds.attrs)
        for key, value in attrs.items():
            da_extra_attr = xr.DataArray(value).expand_dims(
                {"node_index": ds.node_index, "node_feature": 1}
            )
            da_extra_attr.coords["node_feature"] = [key]
            das_extra_attrs.append(da_extra_attr)

            del ds.attrs[key]

        # have to remove `node_features` variable and `node_feature` dimension
        # from dataset otherwise we can increase the number of node features
        # (the coordinate values stays the same)
        da_node_features = ds.node_features
        ds = ds.drop_vars(["node_features", "node_feature"])
        ds["node_features"] = xr.concat(
            das_extra_attrs + [da_node_features], dim="node_feature"
        )

    datasets.append(ds)

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

    subgraph_datasets = collect_datasets(dt)
    ds = xr.merge(subgraph_datasets)
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
