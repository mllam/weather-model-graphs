import networkx as nx
import pytest

from weather_model_graphs.create.connectivity_checks import (
    check_for_unconnected_grid_nodes,
)


def test_g2m_connectivity_raises_value_error_on_isolated_nodes():
    """Test that isolated grid nodes trigger a ValueError."""
    # Create a dummy graph with 3 grid nodes (0, 1, 2)
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2])

    # Connect only nodes 0 and 1 to a dummy mesh node (index 3)
    G.add_edge(0, 3)
    G.add_edge(1, 3)
    # Node 2 is left isolated (out-degree = 0)

    with pytest.raises(ValueError, match=r"1 grid node\(s\) are not connected"):
        check_for_unconnected_grid_nodes(G, grid_nodes=[0, 1, 2])
