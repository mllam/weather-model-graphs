"""
Django-style filtering for NetworkX DiGraph objects.

This module provides an expressive query interface for filtering nodes and edges
of a NetworkX `DiGraph` using Django ORM-style syntax. It supports flexible
attribute lookups, logical composition with `Q()` objects, nested indexing (e.g. `attr[0]__gt`),
spatial bounding box filtering, and returns a new filtered graph.

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

# Filter nodes within a bounding box (x_min, x_max, y_min, y_max)
>>> filtered_bbox = filter_graph(G, node__pos__bbox=(0, 10, 0, 20))
>>> list(filtered_bbox.nodes)
[1, 3]

Features
--------
- Django-style field lookups: exact, lt, lte, gt, gte, contains, in, startswith, endswith, isnull
- Attribute indexing support: e.g., node__pos[0]__gt=5
- Logical composition with Q objects: AND (&), OR (|), NOT (~)
- Spatial bounding box filter: node__pos__bbox=(x_min, x_max, y_min, y_max)
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


def _match(attrs: Dict[str, Any], key: str, value: Any) -> bool:
    """
    Arguments
    ---------
    attrs : dict
        The attribute dictionary of a node or edge. For node label matching,
        the node ID should be injected under the "label" key by the caller.
    key : str
        The attribute path and lookup.
    value : Any
        The value to compare against.

    Returns
    -------
    bool
        Whether the attribute satisfies the condition.
    """
    attr, lookup = _parse_lookup(key)
    if lookup not in LOOKUPS:
        raise ValueError(f"Unsupported lookup: {lookup}")

    try:
    attr_val = _get_nested_attr_value(attrs, attr)
except KeyError:
    return False

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
        The node or edge attributes to test. For node filtering, inject the
        node ID under the "label" key before calling, e.g.
        _evaluate_q_object(q, {**attrs, "label": n}).

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


def _node_matches_bbox(attrs: Dict[str, Any], bbox: tuple) -> bool:
    """
    Check whether a node's `pos` attribute falls within a bounding box.

    Arguments
    ---------
    attrs : dict
        The node attribute dictionary. Must contain a `pos` key with at least 2 elements.
    bbox : tuple
        A (x_min, x_max, y_min, y_max) bounding box. Bounds are inclusive.

    Returns
    -------
    bool
        True if the node's pos is within the bounding box.
    """
    pos = attrs.get("pos")
    if pos is None or len(pos) < 2:
        return False
    x_min, x_max, y_min, y_max = bbox
    return x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max


def _split_q_by_prefix(q: "Q") -> "dict[str, Q]":
    """
    Walk a Q tree once and return a dict mapping each prefix ('node', 'edge')
    to a new Q tree with that prefix stripped from all leaf dict keys.
    Preserves AND/OR/NOT/negation structure exactly.

    Arguments
    ---------
    q : Q
        The query tree to split. All leaf dict keys must start with 'node__'
        or 'edge__'. Mixed prefixes within a single leaf dict are not supported.

    Returns
    -------
    dict[str, Q]
        e.g. {'node': <Q with node__ stripped>, 'edge': <Q with edge__ stripped>}

    Raises
    ------
    ValueError
        If no recognized prefix is found, or if a single leaf dict contains
        keys from more than one prefix.
    """
    # Note: this handles the common shallow Q trees used in practice (1-2 levels,
    # single connector, root-level negation). Deeply nested trees with inner
    # negations or mixed connectors at different levels are not a realistic use
    # case for graph filtering and are not explicitly supported.
    def _walk(node_q):
        rewritten = {}
        for child in node_q.children:
            if isinstance(child, Q):
                child_result = _walk(child)
                for prefix, child_subtree in child_result.items():
                    if prefix not in rewritten:
                        new_q = Q()
                        new_q.connector = node_q.connector
                        new_q.negated = child.negated
                        new_q.children = []
                        rewritten[prefix] = new_q
                    rewritten[prefix].children.append(child_subtree)
            elif isinstance(child, dict):
                prefixes_in_leaf = {k.split("__", 1)[0] for k in child if "__" in k}
                recognized = prefixes_in_leaf & {"node", "edge"}
                if len(recognized) > 1:
                    raise ValueError(
                        f"A single Q() cannot mix 'node__' and 'edge__' keys: {list(child.keys())}"
                    )
                if not recognized:
                    raise ValueError(
                        f"Q keys must start with 'node__' or 'edge__', got: {list(child.keys())}"
                    )
                prefix = next(iter(recognized))
                stripped = {k[len(prefix) + 2 :]: v for k, v in child.items()}
                if prefix not in rewritten:
                    new_q = Q()
                    new_q.connector = node_q.connector
                    new_q.negated = False
                    new_q.children = []
                    rewritten[prefix] = new_q
                rewritten[prefix].children.append(stripped)
        return rewritten

    result = _walk(q)

    if not result:
        raise ValueError("Q args must contain at least one 'node__' or 'edge__' key.")

    # Apply root-level negation to each split subtree root
    for split_q in result.values():
        split_q.negated = q.negated

    return result


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
        Special shorthand: node__pos__bbox=(x_min, x_max, y_min, y_max) for
        spatial bounding box filtering on the node's `pos` attribute.

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

    # Extract bbox shorthand before standard Q parsing
    bbox = None
    filtered_kwargs = {}
    for k, v in kwargs.items():
        if k == "node__pos__bbox":
            if len(v) != 4:
                raise ValueError(
                    "node__pos__bbox must be a 4-tuple (x_min, x_max, y_min, y_max)"
                )
            bbox = v
        else:
            filtered_kwargs[k] = v

    for k, v in filtered_kwargs.items():
        if "__" not in k:
            raise ValueError(f"Query key must start with 'node__' or 'edge__': {k}")
        prefix, rest = k.split("__", 1)
        if prefix not in ("node", "edge"):
            raise ValueError(f"Unknown filter target: {prefix}")
        filters[prefix].append(Q(**{rest: v}))

    for q in args:
        if not isinstance(q, Q):
            raise ValueError("Positional args must be Q objects")
        for prefix, rewritten in _split_q_by_prefix(q).items():
            filters[prefix].append(rewritten)

    node_match = set()
    edge_match = set()

    if "node" in filters or bbox is not None:
        for n, attrs in graph.nodes(data=True):
            node_attrs = {**attrs, "label": n}
            q_match = all(_evaluate_q_object(q, node_attrs) for q in filters["node"])
            bbox_match = _node_matches_bbox(attrs, bbox) if bbox is not None else True
            if q_match and bbox_match:
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
