"""
Diagnostics and safety checks for graph connectivity during mesh generation.
"""
import networkx as nx


def check_for_unconnected_grid_nodes(g2m_graph: nx.DiGraph, grid_nodes: list) -> None:
    """
    Asserts that all grid nodes in the Grid-to-Mesh (g2m) graph are connected
    to at least one mesh node.

    Args:
        g2m_graph: The directed bipartite graph connecting grid nodes to mesh nodes.
        grid_nodes: A list of the grid node identifiers.

    Raises:
        ValueError: If any grid node has an out-degree of 0.
    """
    # Grid nodes are the source nodes in g2m, so we check their out-degree
    disconnected_count = sum(
        1 for node in grid_nodes if g2m_graph.out_degree(node) == 0
    )

    if disconnected_count > 0:
        raise ValueError(
            f"{disconnected_count} grid node(s) are not connected to any mesh nodes in the g2m graph. "
            "This usually happens if the connection radius is too small or the mesh resolution is too sparse."
        )
