"""
Diagnostics and safety checks for graph connectivity during mesh generation.
"""
import networkx as nx

def check_graph_consistency(
    graph: nx.DiGraph | dict[str, nx.DiGraph], 
    allow_unconnected_grid_nodes: bool = False
) -> None:
    """
    Runs a suite of topological diagnostic checks on the generated graph.
    
    Args:
        graph: The generated networkx.DiGraph (or dict of component subgraphs).
        allow_unconnected_grid_nodes: If True, bypasses the grid-to-mesh connection check.
        
    Raises:
        ValueError: If graph inconsistencies are found (e.g., isolated grid nodes).
    """
    if allow_unconnected_grid_nodes:
        return

    disconnected_count = 0

    if isinstance(graph, dict):
        if "g2m" not in graph:
            return
        g2m_graph = graph["g2m"]
        
        # Identify grid nodes (they act as sources in the g2m graph)
        grid_nodes = [n for n, d in g2m_graph.nodes(data=True) if d.get("type") == "grid"]
        if not grid_nodes:
            # Fallback if 'type' is missing: assume nodes with no incoming edges are grid nodes
            grid_nodes = [n for n in g2m_graph.nodes if g2m_graph.in_degree(n) == 0]

        disconnected_count = sum(1 for node in grid_nodes if g2m_graph.out_degree(node) == 0)

    else:
        # Merged graph approach
        grid_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "grid"]
        
        for node in grid_nodes:
            # Check if this grid node has any outgoing edges belonging to the 'g2m' component
            has_g2m_edges = any(
                edge_data.get("component") == "g2m" 
                for _, _, edge_data in graph.out_edges(node, data=True)
            )
            if not has_g2m_edges:
                disconnected_count += 1

    if disconnected_count > 0:
        raise ValueError(
            f"{disconnected_count} grid node(s) are not connected to any mesh nodes in the g2m graph. "
            "This usually happens if the connection radius is too small or the mesh resolution is too sparse."
        )