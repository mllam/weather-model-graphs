import networkx


class MissingEdgeAttributeError(Exception):
    pass


def split_graph_by_edge_attribute(graph, attr):
    """
    Split a graph into subgraphs based on an edge attribute, returning
    a dictionary of subgraphs keyed by the edge attribute value.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to split
    attr : str
        Edge attribute to split the graph by

    Returns
    -------
    dict
        Dictionary of subgraphs keyed by edge attribute value
    """

    # check if any node has the attribute
    if not any(attr in graph.edges[edge] for edge in graph.edges):
        raise MissingEdgeAttributeError(
            f"Edge attribute '{attr}' not found in graph. Check the attribute."
        )

    # Get unique edge attribute values
    edge_values = set(networkx.get_edge_attributes(graph, attr).values())

    # Create a dictionary of subgraphs keyed by edge attribute value
    subgraphs = {}
    for edge_value in edge_values:
        subgraphs[edge_value] = graph.copy().edge_subgraph(
            # Subgraphs only contain edges that actually has attribute,
            # and where it is correct value
            [
                edge
                for edge in graph.edges
                if attr in graph.edges[edge] and graph.edges[edge][attr] == edge_value
            ]
        )

    # copy node attributes
    for subgraph in subgraphs.values():
        for node in subgraph.nodes:
            subgraph.nodes[node].update(graph.nodes[node])

    # check that at least one subgraph was created
    if len(subgraphs) == 0:
        raise ValueError(
            f"No subgraphs were created. Check the edge attribute '{attr}'."
        )

    # copy node attributes
    for subgraph in subgraphs.values():
        for node in subgraph.nodes:
            subgraph.nodes[node].update(graph.nodes[node])

    # check that at least one subgraph was created
    if len(subgraphs) == 0:
        raise ValueError(
            f"No subgraphs were created. Check the edge attribute '{attr}'."
        )

    # copy node attributes
    for subgraph in subgraphs.values():
        for node in subgraph.nodes:
            subgraph.nodes[node].update(graph.nodes[node])

    # check that at least one subgraph was created
    if len(subgraphs) == 0:
        raise ValueError(
            f"No subgraphs were created. Check the edge attribute '{attr}'."
        )

    # copy node attributes
    for subgraph in subgraphs.values():
        for node in subgraph.nodes:
            subgraph.nodes[node].update(graph.nodes[node])

    # check that at least one subgraph was created
    if len(subgraphs) == 0:
        raise ValueError(
            f"No subgraphs were created. Check the edge attribute '{attr}'."
        )

    return subgraphs


def split_on_edge_attribute_existance(graph, attr):
    """
    Split up graph based on if edges have specific attribute.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to check for levels
    attr : str
        Attribute to consider, split the graph depending on if this attribute
        exists or not

    Returns
    -------
    graph_with_attr : networkx.Graph
        Subgraph with edges with attribute
    graph_without_attr : networkx.Graph
        Subgraph with edges without the attribute
    """
    edges = list(graph.edges(data=True))
    edges_with_attr = [e[:2] for e in edges if attr in e[2]]
    edges_without_attr = [e[:2] for e in edges if attr not in e[2]]

    graph_with_attr = graph.edge_subgraph(edges_with_attr)
    graph_without_attr = graph.edge_subgraph(edges_without_attr)

    return graph_with_attr, graph_without_attr
