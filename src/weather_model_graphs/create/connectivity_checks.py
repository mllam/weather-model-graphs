"""
Diagnostics and safety checks for graph connectivity during mesh generation.
"""
import networkx as nx


def check_for_unconnected_grid_nodes(
    g2m_graph: nx.DiGraph, num_grid_nodes: int
) -> None:
    """
    Asserts that all grid nodes in the Grid-to-Mesh (g2m) graph are connected
    to at least one mesh node.

    Args:
        g2m_graph: The directed bipartite graph connecting grid nodes to mesh nodes.
        num_grid_nodes: The total number of grid nodes (assumed to be indexed 0 to N-1).

    Raises:
        ValueError: If any grid node has an out-degree of 0.
    """
    # Grid nodes are the source nodes in g2m, so we check their out-degree
    disconnected_count = sum(
        1 for i in range(num_grid_nodes) if g2m_graph.out_degree(i) == 0
    )

    if disconnected_count > 0:
        raise ValueError(
            f"{disconnected_count} grid node(s) are not connected to any mesh nodes in the g2m graph. "
            "This usually happens if the connection radius is too small or the mesh resolution is too sparse."
        )
