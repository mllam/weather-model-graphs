import networkx


def prepend_node_index(graph, new_index):
    """
    Prepend node index to node tuple in graph, i.e. (i, j) -> (new_index, i, j)

    Parameters
    ----------
    graph : networkx.Graph
        Graph to relabel
    new_index : int
        New index to prepend to node tuple

    Returns
    -------
    networkx.Graph
        Graph with relabeled nodes
    """
    ijk = [tuple((new_index,) + x) for x in graph.nodes]
    to_mapping = dict(zip(graph.nodes, ijk))
    return networkx.relabel_nodes(graph, to_mapping, copy=True)


def sort_nodes_internally(nx_graph, node_attribute=None, edge_attribute=None):
    # For some reason the networkx .nodes() return list can not be sorted,
    # but this is the ordering used by pyg when converting.
    # This function fixes this.
    H = networkx.DiGraph()
    if node_attribute is not None:
        H.add_nodes_from(
            sorted(nx_graph.nodes(data=True), key=lambda x: x[1][node_attribute])
        )
    else:
        H.add_nodes_from(sorted(nx_graph.nodes(data=True)))

    if edge_attribute is not None:
        H.add_edges_from(
            sorted(nx_graph.edges(data=True), key=lambda x: x[2][edge_attribute])
        )
    else:
        H.add_edges_from(nx_graph.edges(data=True))
    return H


class MissingEdgeAttributeError(Exception):
    pass


def split_graph_by_edge_attribute(graph, attribute):
    """
    Split a graph into subgraphs based on an edge attribute, returning
    a dictionary of subgraphs keyed by the edge attribute value.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to split
    attribute : str
        Edge attribute to split the graph by

    Returns
    -------
    dict
        Dictionary of subgraphs keyed by edge attribute value
    """

    # check if any node has the attribute
    if not any(attribute in graph.edges[edge] for edge in graph.edges):
        raise MissingEdgeAttributeError(
            f"Edge attribute '{attribute}' not found in graph. Check the attribute."
        )

    # Get unique edge attribute values
    edge_values = set(networkx.get_edge_attributes(graph, attribute).values())

    # Create a dictionary of subgraphs keyed by edge attribute value
    subgraphs = {}
    for edge_value in edge_values:
        subgraphs[edge_value] = graph.copy().edge_subgraph(
            [edge for edge in graph.edges if graph.edges[edge][attribute] == edge_value]
        )

    # copy node attributes
    for subgraph in subgraphs.values():
        for node in subgraph.nodes:
            subgraph.nodes[node].update(graph.nodes[node])

    # check that at least one subgraph was created
    if len(subgraphs) == 0:
        raise ValueError(
            f"No subgraphs were created. Check the edge attribute '{attribute}'."
        )

    return subgraphs


def replace_node_labels_with_unique_ids(graph):
    """
    Rename node labels with unique id.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to rename node labels

    Returns
    -------
    networkx.Graph
        Graph with node labels renamed
    """
    return networkx.relabel_nodes(
        graph, {node: i for i, node in enumerate(graph.nodes)}, copy=True
    )
