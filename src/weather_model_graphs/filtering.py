"""
Django-style filtering for NetworkX DiGraph objects.

This module provides an expressive query interface for filtering nodes and edges
of a NetworkX `DiGraph` using Django ORM-style syntax. It supports flexible
attribute lookups, logical composition with `Q()` objects, nested indexing (e.g. `attr[0]__gt`),
and returns a new filtered graph.

Example
-------
>>> import networkx as nx
>>> from filter_graph import filter_graph, Q

>>> G = nx.DiGraph()
>>> G.add_node(1, name="StationA", label="Person", pos=[5, 15])
>>> G.add_node(2, name="StationB", label="Person", pos=[20, 30])
>>> G.add_node(3, name="StationC", label="Company", pos=[8, 9])
>>> G.add_edge(1, 3, relation="works_at", since=2015)
>>> G.add_edge(2, 3, relation="works_at", since=2021)

# Filter nodes with pos[0] > 10 using keyword arguments
>>> filtered = filter_graph(G, **{"node__pos[0]__gt": 10})
>>> list(filtered.nodes)
[2]

# Filter nodes named 'StationC' or with pos[0] < 10 using Q objects
>>> q = Q(node__name="StationC") | Q(node__pos[0]__lt=10)
>>> filtered_q = filter_graph(G, q)
>>> list(filtered_q.nodes)
[1, 3]

Features
--------
- Django-style field lookups: exact, lt, lte, gt, gte, contains, in, startswith, endswith, isnull
- Attribute indexing support: e.g., node__pos[0]__gt=5
- Logical composition with Q objects: AND (&), OR (|), NOT (~)
- Returns a new DiGraph with matched nodes and edges
"""


import operator
import re
from collections import defaultdict
from typing import Any, Dict, Union

import networkx as nx

LOOKUPS = {
    "exact": operator.eq,
    "lt": operator.lt,
    "lte": operator.le,
    "gt": operator.gt,
    "gte": operator.ge,
    "contains": lambda a, b: b in a,
    "in": lambda a, b: a in b,
    "startswith": lambda a, b: str(a).startswith(b),
    "endswith": lambda a, b: str(a).endswith(b),
    "isnull": lambda a, b: (a is None) if b else (a is not None),
}


INDEX_RE = re.compile(r"([^\[]+)(\[[^\]]+\])*")


def _parse_lookup(key: str) -> tuple[str, str]:
    """
    Arguments
    ---------
    key : str
        A Django-style filter key like 'age__gt' or 'pos[0]__lt'.

    Returns
    -------
    tuple[str, str]
        A tuple (field, lookup) where lookup defaults to 'exact'.
    """
    parts = key.split("__")
    if len(parts) == 1:
        return parts[0], "exact"
    return "__".join(parts[:-1]), parts[-1]


def _get_nested_attr_value(attrs: Dict[str, Any], key: str) -> Any:
    """
    Arguments
    ---------
    attrs : dict
        The attribute dictionary of a node or edge.
    key : str
        The attribute path with optional indexing, e.g. 'pos[0][1]'.

    Returns
    -------
    Any
        The resolved attribute value or None if not found.
    """
    match = INDEX_RE.match(key)
    if not match:
        return attrs.get(key)

    attr_name = match.group(1)
    indices = re.findall(r"\[([0-9]+)\]", key)

    val = attrs.get(attr_name)
    for idx in indices:
        if val is None:
            return None
        try:
            val = val[int(idx)]
        except (IndexError, TypeError):
            return None
    return val


def _match(attrs: Dict[str, Any], key: str, value: Any, node_id: Any = None) -> bool:
    """
    Arguments
    ---------
    attrs : dict
        The attribute dictionary of a node or edge.
    key : str
        The attribute path and lookup.
    value : Any
        The value to compare against.
    node_id : Any, optional
        The node ID for special case handling (e.g., label).

    Returns
    -------
    bool
        Whether the attribute satisfies the condition.
    """
    attr, lookup = _parse_lookup(key)
    if lookup not in LOOKUPS:
        raise ValueError(f"Unsupported lookup: {lookup}")

    # Special case: node label
    if attr == "label" and node_id is not None:
        attr_val = node_id
    else:
        attr_val = _get_nested_attr_value(attrs, attr)

    return LOOKUPS[lookup](attr_val, value)


class Q:
    """
    Arguments
    -------
    **kwargs : Django-style field lookups

    Supports
    -------
    - AND (&), OR (|), NOT (~) combinations
    """

    def __init__(self, **kwargs):
        self.children = [kwargs]
        self.connector = "AND"
        self.negated = False

    def __or__(self, other: "Q") -> "Q":
        return self._combine(other, "OR")

    def __and__(self, other: "Q") -> "Q":
        return self._combine(other, "AND")

    def __invert__(self) -> "Q":
        q = Q()
        q.children = self.children
        q.connector = self.connector
        q.negated = not self.negated
        return q

    def _combine(self, other: "Q", connector: str) -> "Q":
        q = Q()
        q.children = [self, other]
        q.connector = connector
        return q


def _evaluate_q_object(q: Union[Q, dict], attrs: dict) -> bool:
    """
    Arguments
    ---------
    q : Q or dict
        The query object or raw lookup dictionary.
    attrs : dict
        The node or edge attributes to test.

    Returns
    -------
    bool
        Whether the attributes match the query.
    """
    if isinstance(q, dict):
        return all(_match(attrs, k, v) for k, v in q.items())

    results = []
    for child in q.children:
        if isinstance(child, Q):
            result = _evaluate_q_object(child, attrs)
        else:
            result = all(_match(attrs, k, v) for k, v in child.items())
        results.append(result)

    combined = all(results) if q.connector == "AND" else any(results)
    return not combined if q.negated else combined


def filter_graph(
    graph: nx.DiGraph,
    *args,
    node_limit: int = None,
    edge_limit: int = None,
    node_offset: int = 0,
    edge_offset: int = 0,
    retain: str = "connected",
    **kwargs,
) -> nx.DiGraph:
    """
    Filter a NetworkX DiGraph using Django-style lookups and return a new filtered graph.

    Arguments
    ---------
    graph : nx.DiGraph
        The graph to filter.
    *args : Q
        Optional positional Q objects for complex logic.
    node_limit : int, optional
        Limit the number of matched nodes returned (applied before retention).
    edge_limit : int, optional
        Limit the number of matched edges returned (applied before retention).
    node_offset : int, optional
        Number of matching nodes to skip before collecting results (default 0).
    edge_offset : int, optional
        Number of matching edges to skip before collecting results (default 0).
    retain : str, optional
        Retention policy for nodes/edges. One of:
        - "connected": (default) include anything connected to matches
        - "strict": only include nodes and edges that both match filters
        - "none": include only matching nodes or edges, with no extras
        - "filter_connected": include edges that match edge filters and connect to at least one node matching node filters
    **kwargs : dict
        Django-style field lookups prefixed with 'node__' or 'edge__'.

    Returns
    -------
    nx.DiGraph
        A new DiGraph containing the filtered results.

    Note
    ----
    When filtering on 'node__label', the filter will match against the node's ID (label),
    not an attribute called 'label'. This allows filtering like node__label__startswith="mesh".
    """
    filters = defaultdict(list)

    for k, v in kwargs.items():
        if "__" not in k:
            raise ValueError(f"Query key must start with 'node__' or 'edge__': {k}")
        prefix, rest = k.split("__", 1)
        if prefix not in ("node", "edge"):
            raise ValueError(f"Unknown filter target: {prefix}")
        filters[prefix].append(Q(**{rest: v}))

    for q in args:
        if not isinstance(q, Q):
            raise ValueError("Positional args must be Q objects")

        def classify_q(q):
            if isinstance(q, Q):
                new_q = Q()
                new_q.connector = q.connector
                new_q.negated = q.negated
                new_q.children = [classify_q(c) for c in q.children]
                return new_q
            elif isinstance(q, dict):
                classified = defaultdict(dict)
                for k, v in q.items():
                    prefix, rest = k.split("__", 1)
                    classified[prefix][rest] = v
                return {k: Q(**v) for k, v in classified.items()}

        split_q = classify_q(q)
        for prefix in ["node", "edge"]:
            if prefix in split_q:
                filters[prefix].append(split_q[prefix])

    node_match = set()
    edge_match = set()

    if "node" in filters:
        for n, attrs in graph.nodes(data=True):
            if all(
                _match(attrs, k, v, node_id=n)
                for q in filters["node"]
                for k, v in q.children[0].items()
            ):
                node_match.add(n)

    if "edge" in filters:
        edge_match = {
            (u, v)
            for u, v, attrs in graph.edges(data=True)
            if all(_evaluate_q_object(q, attrs) for q in filters["edge"])
        }

    if node_match:
        node_match = list(sorted(node_match))
        node_match = node_match[node_offset:]
        if node_limit is not None:
            node_match = node_match[:node_limit]
        node_match = set(node_match)

    if edge_match:
        edge_match = list(sorted(edge_match))
        edge_match = edge_match[edge_offset:]
        if edge_limit is not None:
            edge_match = edge_match[:edge_limit]
        edge_match = set(edge_match)

    H = nx.DiGraph()

    if node_match and edge_match:
        if retain == "strict":
            for u, v in edge_match:
                if u in node_match and v in node_match:
                    H.add_node(u, **graph.nodes[u])
                    H.add_node(v, **graph.nodes[v])
                    H.add_edge(u, v, **graph.edges[u, v])
        elif retain == "filter_connected":
            for u, v in edge_match:
                if u in node_match or v in node_match:
                    H.add_node(u, **graph.nodes[u])
                    H.add_node(v, **graph.nodes[v])
                    H.add_edge(u, v, **graph.edges[u, v])
        elif retain == "connected":
            for u, v in graph.edges:
                if u in node_match or v in node_match:
                    H.add_node(u, **graph.nodes[u])
                    H.add_node(v, **graph.nodes[v])
                    H.add_edge(u, v, **graph.edges[u, v])
            for u, v in edge_match:
                H.add_node(u, **graph.nodes[u])
                H.add_node(v, **graph.nodes[v])
                H.add_edge(u, v, **graph.edges[u, v])
        elif retain == "none":
            H.add_nodes_from((n, graph.nodes[n]) for n in node_match)
            H.add_edges_from(((u, v, graph.edges[u, v]) for u, v in edge_match))

    elif node_match:
        H.add_nodes_from((n, graph.nodes[n]) for n in node_match)
        if retain == "connected":
            for u, v, data in graph.edges(data=True):
                if u in node_match or v in node_match:
                    H.add_node(u, **graph.nodes[u])
                    H.add_node(v, **graph.nodes[v])
                    H.add_edge(u, v, **data)

    elif edge_match:
        H.add_edges_from((u, v, graph.edges[u, v]) for u, v in edge_match)
        if retain != "none":
            nodes_in_edges = set(u for u, _ in edge_match) | set(
                v for _, v in edge_match
            )
            H.add_nodes_from((n, graph.nodes[n]) for n in nodes_in_edges)

    return H
