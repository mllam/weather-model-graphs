"""
3D interactive graph visualisation using Plotly.
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np

try:
    import plotly.graph_objects as _go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Constants defined at module level, not imported from within the module
DEFAULT_COMPONENT_COLORS: dict[str, str] = {
    "g2m": "blue",
    "m2m": "green",
    "m2g": "red",
    "unknown": "#9E9E9E",
}

# marker symbol used for grid nodes vs mesh nodes.
_GRID_MARKER = "circle"
_MESH_MARKER = "diamond"

_GRID_Z = -1


def _get_node_positions(
    graph: nx.DiGraph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Return (x, y, z, node_ids) arrays for all nodes in graph.

    z is derived from the level node attribute (integer).  Nodes
    without a level attribute (grid nodes) are placed at z = _GRID_Z.

    Parameters
    ----------
    graph:
        Any networkx DiGraph produced by weather_model_graphs.

    Returns
    -------
    x, y, z : np.ndarray of shape (N,)
    node_ids : list of length N, the node identifiers in the same order.
    """
    node_ids = list(graph.nodes())
    xs, ys, raw_levels = [], [], []

    for node in node_ids:
        attrs = graph.nodes[node]
        pos = attrs["pos"]
        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
        raw_levels.append(attrs.get("level"))  # None for grid nodes

    # Find minimum level to normalize mesh node z values to start at 0
    mesh_levels = [lvl for lvl in raw_levels if lvl is not None]
    level_offset = min(mesh_levels) if mesh_levels else 0

    zs = []
    for level in raw_levels:
        if level is not None:
            zs.append(float(level - level_offset))
        else:
            zs.append(float(_GRID_Z))

    return np.array(xs), np.array(ys), np.array(zs), node_ids


def _node_id_to_index(node_ids: list) -> dict:
    """Build a reverse mapping node_id -> position_in_list."""
    return {nid: i for i, nid in enumerate(node_ids)}


def _build_edge_traces(
    graph: nx.DiGraph,
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    z_nodes: np.ndarray,
    node_index: dict,
    component_colors: dict[str, str],
    edge_width: float,
) -> list:
    """Return one Scatter3d trace per graph component.

    All edges belonging to the same component are batched into a single trace
    using None separators so that Plotly draws them as disconnected
    segments without linking unrelated endpoints. This keeps file sizes small.

    Parameters
    ----------
    graph:
        Source graph. Edges are expected to have a component attribute
        (g2m, m2m, m2g). Edges without this attribute are
        grouped under unknown.
    x_nodes, y_nodes, z_nodes:
        Node position arrays (same order as node_index).
    node_index:
        Mapping from node id to index into the position arrays.
    component_colors:
        Mapping from component name to hex/CSS colour string.
    edge_width:
        Line width for all edge traces.

    Returns
    -------
    list of plotly.graph_objects.Scatter3d
    """
    # Group edges by component
    component_edges: dict[str, list[tuple]] = {}
    for u, v, attrs in graph.edges(data=True):
        comp = attrs.get("component", "unknown")
        component_edges.setdefault(comp, []).append((u, v))

    traces = []
    for comp, edges in sorted(component_edges.items()):
        # Build batched coordinate arrays with None separators
        ex, ey, ez = [], [], []
        for u, v in edges:
            ui, vi = node_index[u], node_index[v]
            ex.extend([x_nodes[ui], x_nodes[vi], None])
            ey.extend([y_nodes[ui], y_nodes[vi], None])
            ez.extend([z_nodes[ui], z_nodes[vi], None])

        color = component_colors.get(comp, component_colors.get("unknown", "#9E9E9E"))
        trace = _go.Scatter3d(
            x=ex,
            y=ey,
            z=ez,
            mode="lines",
            name=comp,
            line=dict(color=color, width=edge_width),
            hoverinfo="none",
        )
        traces.append(trace)
    return traces


def _build_node_traces(
    graph: nx.DiGraph,
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    z_nodes: np.ndarray,
    node_ids: list,
    node_size: float,
) -> list:
    """Return Scatter3d node traces, one per node kind(grid / mesh level).

    Grid nodes (z == _GRID_Z) are drawn with a circle marker; mesh nodes
    at each level are drawn with a diamond marker and labelled by level.

    Parameters
    ----------
    graph:
        Source graph used only for fetching hover text.
    x_nodes, y_nodes, z_nodes:
        Node position arrays.
    node_ids:
        Node identifiers in the same order as the position arrays.
    node_size:
        Marker size in plotly units.

    Returns
    -------
    list of plotly.graph_objects.Scatter3d
    """
    # Split nodes into grid nodes and per-level mesh node groups
    groups: dict[str, dict] = {}  # key -> {"x":[], "y":[], "z":[], "text":[]}

    for i, node in enumerate(node_ids):
        z = z_nodes[i]
        if z == _GRID_Z:
            key = "grid"
            symbol = _GRID_MARKER
            label = "grid"
        else:
            level = int(z)
            key = f"mesh_level_{level}"
            symbol = _MESH_MARKER
            label = f"mesh level {level}"

        if key not in groups:
            groups[key] = {
                "x": [],
                "y": [],
                "z": [],
                "text": [],
                "symbol": symbol,
                "label": label,
            }

        pos = graph.nodes[node]["pos"]
        hover = f"node: {node}<br>pos: ({pos[0]:.3f}, {pos[1]:.3f})<br>{label}"
        groups[key]["x"].append(float(x_nodes[i]))
        groups[key]["y"].append(float(y_nodes[i]))
        groups[key]["z"].append(float(z))
        groups[key]["text"].append(hover)

    # Assign colours: grid nodes grey, mesh nodes use a qualitative palette
    mesh_palette = [
        "#E91E63",
        "#9C27B0",
        "#673AB7",
        "#3F51B5",
        "#00BCD4",
        "#009688",
        "#8BC34A",
        "#FF9800",
    ]
    mesh_level_keys = sorted(k for k in groups if k.startswith("mesh_level_"))
    level_colors = {
        k: mesh_palette[i % len(mesh_palette)] for i, k in enumerate(mesh_level_keys)
    }

    traces = []
    # Draw grid nodes first (they are at the bottom z-plane)
    if "grid" in groups:
        g = groups["grid"]
        traces.append(
            _go.Scatter3d(
                x=g["x"],
                y=g["y"],
                z=g["z"],
                mode="markers",
                name="grid",
                marker=dict(
                    size=node_size * 0.7,  # slightly smaller than mesh nodes
                    color="#78909C",  # blue-grey
                    symbol=g["symbol"],
                    opacity=0.6,
                ),
                text=g["text"],
                hoverinfo="text",
            )
        )

    # Draw mesh nodes, ordered by level
    for key in mesh_level_keys:
        g = groups[key]
        traces.append(
            _go.Scatter3d(
                x=g["x"],
                y=g["y"],
                z=g["z"],
                mode="markers",
                name=g["label"],
                marker=dict(
                    size=node_size,
                    color=level_colors[key],
                    symbol=g["symbol"],
                    opacity=0.9,
                    line=dict(width=0.5, color="white"),
                ),
                text=g["text"],
                hoverinfo="text",
            )
        )

    return traces


def _build_layout(title: str | None) -> "_go.Layout":
    """Return a clean go.Layout for the 3D scene.

    The axes are labelled x, y, and level (z-axis).  The
    background is kept white so the graph is the visual focus.

    Parameters
    ----------
    title:
        Optional figure title.

    Returns
    -------
    plotly.graph_objects.Layout
    """
    axis_style = dict(
        backgroundcolor="white",
        gridcolor="#E0E0E0",
        showbackground=True,
        zerolinecolor="#BDBDBD",
        tickfont=dict(size=10),
    )
    return _go.Layout(
        title=dict(text=title or "Weather Model Graph (3D)", x=0.5),
        scene=dict(
            xaxis=dict(title="x", **axis_style),
            yaxis=dict(title="y", **axis_style),
            zaxis=dict(title="level (z)", **axis_style),
            aspectmode="data",
        ),
        legend=dict(
            title=dict(text="component / node kind"),
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor="white",
    )


def render_with_plotly(
    graph: nx.DiGraph,
    *,
    show: bool = True,
    title: str | None = None,
    node_size: float = 4.0,
    edge_width: float = 1.5,
    component_colors: dict[str, str] | None = None,
) -> "_go.Figure":
    """Render a weather-model-graphs networkx.DiGraph in 3D using Plotly.

    Node positions are taken from the pos node attribute (x, y).  The
    vertical axis is determined by the level node attribute (integer), with
    grid nodes which carry no level placed at z = -1.

    Edges are batched per component (g2m, m2m, m2g) into single
    Scatter3d traces separated by None values, keeping the HTML output
    small even for large global graphs.

    Parameters
    ----------
    graph:
        A networkx.DiGraph as returned by any wmg.create.* function or
        wmg.create.create_all_graph_components().  The graph must have
        pos as a node attribute on every node.
    show:
        If True (default), call fig.show() before returning.  Set to
        False when using the figure programmatically (e.g. in tests or
        Jupyter notebooks that render the return value directly).
    title:
        Optional title shown at the top of the figure.  Defaults to
        "Weather Model Graph (3D)".
    node_size:
        Marker size for mesh nodes in Plotly units.  Grid nodes are drawn at
        0.7 * node_size.  Default is 4.0.
    edge_width:
        Line width for all edge traces.  Default is 1.5.
    component_colors:
        Mapping from component name to CSS/hex colour string.  Any component
        not present in the mapping falls back to DEFAULT_COMPONENT_COLORS.
        Pass an empty dict to use all defaults.

    Returns
    -------
    plotly.graph_objects.Figure
        The fully configured interactive figure.
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly is required for 3D visualisation but is not installed.\n"
            "Install it with:  pip install plotly\n"
            "or:               pip install weather-model-graphs[visualisation]"
        )

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot render an empty graph (no nodes).")

    # Validate that every node carries a pos attribute
    missing_pos = [n for n in graph.nodes() if "pos" not in graph.nodes[n]]
    if missing_pos:
        raise ValueError(
            f"{len(missing_pos)} node(s) are missing the required 'pos' attribute. "
            f"First offending node: {missing_pos[0]!r}"
        )

    # Merge caller-supplied colours with defaults
    colors = {**DEFAULT_COMPONENT_COLORS, **(component_colors or {})}

    # extract positions
    x_nodes, y_nodes, z_nodes, node_ids = _get_node_positions(graph)
    node_index = _node_id_to_index(node_ids)

    # build traces
    traces: list = []

    # Edge traces first so they render beneath nodes
    if graph.number_of_edges() > 0:
        edge_traces = _build_edge_traces(
            graph=graph,
            x_nodes=x_nodes,
            y_nodes=y_nodes,
            z_nodes=z_nodes,
            node_index=node_index,
            component_colors=colors,
            edge_width=edge_width,
        )
        traces.extend(edge_traces)
    else:
        warnings.warn(
            "Graph has no edges; rendering node positions only.",
            stacklevel=2,
        )

    node_traces = _build_node_traces(
        graph=graph,
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        z_nodes=z_nodes,
        node_ids=node_ids,
        node_size=node_size,
    )
    traces.extend(node_traces)

    # assemble figure
    fig = _go.Figure(data=traces, layout=_build_layout(title))

    if show:
        fig.show()

    return fig
