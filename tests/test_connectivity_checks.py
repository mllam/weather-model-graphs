import networkx as nx
import pytest

from weather_model_graphs.diagnostics import check_graph_consistency


def test_g2m_connectivity_raises_value_error_on_isolated_nodes():
    """Test that isolated grid nodes trigger a ValueError."""
    G = nx.DiGraph()

    # Add 3 grid nodes. The new diagnostic looks for the type="grid" attribute.
    G.add_node(0, type="grid")
    G.add_node(1, type="grid")
    G.add_node(2, type="grid")

    # Connect only nodes 0 and 1 to a dummy mesh node (index 3).
    # The new diagnostic looks for the component="g2m" attribute on the edges.
    G.add_edge(0, 3, component="g2m")
    G.add_edge(1, 3, component="g2m")
    # Node 2 is left isolated (no outgoing g2m edges)

    # We expect this to fail because allow_unconnected_grid_nodes is False by default
    with pytest.raises(ValueError, match=r"1 grid node\(s\) are not connected"):
        check_graph_consistency(G)
