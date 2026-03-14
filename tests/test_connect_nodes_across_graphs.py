import numpy as np
import networkx as nx

from weather_model_graphs.create.base import connect_nodes_across_graphs


def test_kdtree_node_mapping_order():
    G_source = nx.Graph()

    # intentionally unsorted insertion order
    G_source.add_node("z", pos=np.array([0.0, 0.0]))
    G_source.add_node("a", pos=np.array([10.0, 0.0]))
    G_source.add_node("m", pos=np.array([20.0, 0.0]))

    G_target = nx.Graph()

    G_target.add_node("t0", pos=np.array([0.2, 0.0]))
    G_target.add_node("t1", pos=np.array([9.9, 0.0]))
    G_target.add_node("t2", pos=np.array([19.7, 0.0]))

    G = connect_nodes_across_graphs(
        G_source,
        G_target,
        method="nearest_neighbour"
    )

    edges = sorted(G.edges())

    expected = [
        ("z", "t0"),
        ("a", "t1"),
        ("m", "t2")
    ]

    assert edges == expected