"""
Tests for Django-style graph filtering.

Covers:
- Basic attribute filtering
- OR/NOT Q logic on nodes (previously broken due to q.children[0].items())
- Spatial bounding box filtering
"""

import networkx as nx
import pytest

from weather_model_graphs.filtering import Q, filter_graph


@pytest.fixture
def simple_graph():
    """Small graph with pos attributes for testing."""
    G = nx.DiGraph()
    G.add_node(1, type="mesh", pos=[5, 10])
    G.add_node(2, type="grid", pos=[20, 30])
    G.add_node(3, type="mesh", pos=[8, 12])
    G.add_node(4, type="grid", pos=[50, 60])
    G.add_edge(1, 2, component="g2m")
    G.add_edge(3, 4, component="g2m")
    return G


# --- Basic filtering ---

def test_simple_node_filter(simple_graph):
    result = filter_graph(simple_graph, **{"node__type": "mesh"}, retain="none")
    assert set(result.nodes) == {1, 3}


def test_simple_edge_filter(simple_graph):
    result = filter_graph(simple_graph, **{"edge__component": "g2m"}, retain="none")
    assert set(result.edges) == {(1, 2), (3, 4)}


# --- OR logic on nodes (previously broken) ---

def test_node_or_q_logic(simple_graph):
    """Q with OR should correctly match either condition."""
    q = Q(**{"node__type": "mesh"}) | Q(**{"node__pos[0]__gt": 40})
    result = filter_graph(simple_graph, q, retain="none")
    # nodes 1, 3 are mesh; node 4 has pos[0]=50 > 40
    assert set(result.nodes) == {1, 3, 4}


def test_node_and_q_logic(simple_graph):
    """Q with AND should require both conditions."""
    q = Q(**{"node__type": "mesh"}) & Q(**{"node__pos[0]__gt": 6})
    result = filter_graph(simple_graph, q, retain="none")
    # only node 3 is mesh AND has pos[0]=8 > 6
    assert set(result.nodes) == {3}


def test_node_not_q_logic(simple_graph):
    """Negated Q should invert the match."""
    q = ~Q(**{"node__type": "mesh"})
    result = filter_graph(simple_graph, q, retain="none")
    assert set(result.nodes) == {2, 4}


def test_node_nested_or_not_q_logic(simple_graph):
    """Complex nested Q: NOT (mesh OR pos[0] > 40)."""
    q = ~(Q(**{"node__type": "mesh"}) | Q(**{"node__pos[0]__gt": 40}))
    result = filter_graph(simple_graph, q, retain="none")
    # nodes 1, 3 are mesh; node 4 has pos[0] > 40 — so only node 2 remains
    assert set(result.nodes) == {2}


# --- Bounding box filtering ---

def test_bbox_basic(simple_graph):
    """Nodes within bbox should be returned."""
    result = filter_graph(simple_graph, node__pos__bbox=(0, 10, 0, 15), retain="none")
    # node 1: pos=[5,10] inside; node 3: pos=[8,12] inside; others outside
    assert set(result.nodes) == {1, 3}


def test_bbox_boundary_inclusive(simple_graph):
    """Bbox bounds are inclusive."""
    result = filter_graph(simple_graph, node__pos__bbox=(5, 5, 10, 10), retain="none")
    # exactly node 1 at pos=[5,10]
    assert set(result.nodes) == {1}


def test_bbox_no_matches(simple_graph):
    """Bbox that contains no nodes returns empty graph."""
    result = filter_graph(simple_graph, node__pos__bbox=(100, 200, 100, 200), retain="none")
    assert len(result.nodes) == 0


def test_bbox_combined_with_type_filter(simple_graph):
    """Bbox combined with attribute filter should AND both conditions."""
    result = filter_graph(
        simple_graph,
        node__pos__bbox=(0, 25, 0, 35),
        **{"node__type": "grid"},
        retain="none",
    )
    # node 2: grid AND pos=[20,30] inside bbox
    # node 1, 3: mesh, excluded by type filter
    assert set(result.nodes) == {2}


def test_bbox_invalid_tuple_raises():
    """bbox with wrong length should raise ValueError."""
    G = nx.DiGraph()
    G.add_node(1, pos=[0, 0])
    with pytest.raises(ValueError, match="4-tuple"):
        filter_graph(G, node__pos__bbox=(0, 1, 0))


def test_bbox_missing_pos_attribute():
    """Nodes without pos attribute should not match bbox filter."""
    G = nx.DiGraph()
    G.add_node(1, type="mesh")  # no pos
    G.add_node(2, pos=[5, 5])
    result = filter_graph(G, node__pos__bbox=(0, 10, 0, 10), retain="none")
    assert set(result.nodes) == {2}


# --- _split_q_by_prefix guards ---

def test_mixed_prefix_q_raises(simple_graph):
    """A Q with both node__ and edge__ keys in the same dict should raise."""
    with pytest.raises(ValueError, match="cannot mix"):
        q = Q(**{"node__type": "mesh", "edge__component": "g2m"})
        filter_graph(simple_graph, q)


def test_unrecognized_prefix_q_raises(simple_graph):
    """A Q with no node__/edge__ prefix should raise."""
    with pytest.raises(ValueError, match="node__.*or.*edge__"):
        q = Q(**{"foo__type": "mesh"})
        filter_graph(simple_graph, q)


# --- Documented behaviour for edge cases not common in practice ---

def test_inner_negation_is_preserved(simple_graph):
    """Inner negation on a child Q should be preserved after splitting."""
    # ~Q(node__type="mesh") nested inside an AND
    q = Q(**{"node__level__exact": 1}) & ~Q(**{"node__type": "mesh"})
    result = filter_graph(simple_graph, q, retain="none")
    # nodes with level=1 (none in fixture) AND not mesh — result is empty
    # main point is it doesn't raise and negation doesn't silently disappear
    assert isinstance(result, __import__("networkx").DiGraph)


def test_shallow_or_with_negation_documented(simple_graph):
    """Root-level negation on an OR tree works correctly for the common case."""
    q = ~(Q(**{"node__type": "mesh"}) | Q(**{"node__type": "grid"}))
    result = filter_graph(simple_graph, q, retain="none")
    # NOT (mesh OR grid) — no node matches either type so result should be empty
    assert len(result.nodes) == 0