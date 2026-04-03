"""
3D interactive graph visualisation using Plotly, with support for
flat and concentric spherical layouts, and an optional coastline layer.
"""

from __future__ import annotations

import warnings
from typing import Literal

import networkx as nx
import numpy as np

try:
    import plotly.graph_objects as _go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Import coastline data from the module contributed by @Prince637-bo (scraped out as a suggestion by @Mohit-Lakra)
import cartopy.feature as cfeature

DEFAULT_COMPONENT_COLORS: dict[str, str] = {
    "g2m": "blue",
    "m2m": "green",
    "m2g": "red",
    "unknown": "#9E9E9E",
}

_GRID_MARKER = "circle"
_MESH_MARKER = "diamond"
_GRID_Z_FLAT = -1  # z‑coordinate for grid nodes in flat mode

# Spherical layout defaults
_DEFAULT_SPHERE_RADIUS_BASE = 1.0
_DEFAULT_SPHERE_RADIUS_STEP = 0.5  # additional radius per level


def _lat_lon_to_cartesian(
    lat: float, lon: float, radius: float
) -> tuple[float, float, float]:
    """
    Convert latitude/longitude (degrees) to 3D Cartesian coordinates on a sphere of given radius.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z


def _build_coastline_trace(radius: float) -> "_go.Scatter3d":
    """
    Build a Plotly trace for coastlines at a given sphere radius
    using Cartopy coastline geometries.
    """
    xs, ys, zs = [], [], []

    coastlines = cfeature.COASTLINE.geometries()

    for geom in coastlines:
        try:
            # Handle MultiLineString or LineString
            if hasattr(geom, "geoms"):
                lines = geom.geoms
            else:
                lines = [geom]

            for line in lines:
                coords = list(line.coords)
                for lon, lat in coords:
                    x, y, z = _lat_lon_to_cartesian(lat, lon, radius)
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)

                # Separator between segments
                xs.append(None)
                ys.append(None)
                zs.append(None)

        except Exception:
            continue  # skip problematic geometries safely

    return _go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name="coastlines",
        line=dict(color="#424242", width=1),
        opacity=0.4,
        hoverinfo="none",
    )


def _get_positions_flat(
    graph: nx.DiGraph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Compute flat layout positions (x=lon, y=lat, z=level)."""
    node_ids = list(graph.nodes())
    xs, ys, raw_levels = [], [], []

    for node in node_ids:
        attrs = graph.nodes[node]
        pos = attrs["pos"]  # (lat, lon)
        xs.append(float(pos[1]))
        ys.append(float(pos[0]))
        raw_levels.append(attrs.get("level"))

    mesh_levels = [lvl for lvl in raw_levels if lvl is not None]
    level_offset = min(mesh_levels) if mesh_levels else 0

    zs = []
    for level in raw_levels:
        if level is not None:
            zs.append(float(level - level_offset))
        else:
            zs.append(float(_GRID_Z_FLAT))

    return np.array(xs), np.array(ys), np.array(zs), node_ids


def _get_positions_concentric(
    graph: nx.DiGraph,
    base_radius: float = _DEFAULT_SPHERE_RADIUS_BASE,
    radius_step: float = _DEFAULT_SPHERE_RADIUS_STEP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Compute concentric spherical layout positions."""
    node_ids = list(graph.nodes())

    lats, lons, levels = [], [], []
    for node in node_ids:
        attrs = graph.nodes[node]
        pos = attrs["pos"]
        lats.append(float(pos[0]))
        lons.append(float(pos[1]))
        levels.append(attrs.get("level"))

    mesh_levels = [lvl for lvl in levels if lvl is not None]
    if not mesh_levels:
        warnings.warn(
            "No mesh nodes found. Falling back to flat layout.",
            UserWarning,
            stacklevel=2,
        )
        return _get_positions_flat(graph)

    min_level = min(mesh_levels)
    radius_map = {
        lvl: base_radius + (lvl - min_level) * radius_step for lvl in mesh_levels
    }

    xs, ys, zs = [], [], []
    for lat, lon, lvl in zip(lats, lons, levels):
        if lvl is None:
            # Place grid nodes slightly inside the innermost mesh sphere
            r = base_radius - radius_step * 0.5
            if r <= 0:
                r = 0.5
            x, y, z = _lat_lon_to_cartesian(lat, lon, r)
        else:
            r = radius_map[lvl]
            x, y, z = _lat_lon_to_cartesian(lat, lon, r)
        xs.append(x)
        ys.append(y)
        zs.append(z)

    return np.array(xs), np.array(ys), np.array(zs), node_ids


def _node_id_to_index(node_ids: list) -> dict:
    """Map node identifier to its index in the list."""
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
    """Create one Scatter3d trace per edge component (g2m, m2m, m2g)."""
    component_edges: dict[str, list[tuple]] = {}
    for u, v, attrs in graph.edges(data=True):
        comp = attrs.get("component", "unknown")
        component_edges.setdefault(comp, []).append((u, v))

    traces = []
    for comp, edges in sorted(component_edges.items()):
        ex, ey, ez = [], [], []
        for u, v in edges:
            ui, vi = node_index[u], node_index[v]
            ex.extend([x_nodes[ui], x_nodes[vi], None])
            ey.extend([y_nodes[ui], y_nodes[vi], None])
            ez.extend([z_nodes[ui], z_nodes[vi], None])

        color = component_colors.get(comp, component_colors.get("unknown", "#9E9E9E"))
        traces.append(
            _go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                name=comp,
                line=dict(color=color, width=edge_width),
                hoverinfo="none",
            )
        )
    return traces


def _build_node_traces(
    graph: nx.DiGraph,
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    z_nodes: np.ndarray,
    node_ids: list,
    node_size: float,
) -> list:
    """Create separate traces for grid nodes and each mesh level."""
    groups: dict[str, dict] = {}
    for i, node in enumerate(node_ids):
        attrs = graph.nodes[node]
        pos = attrs["pos"]

        if "level" not in attrs or attrs["level"] is None:
            key = "grid"
            symbol = _GRID_MARKER
            label = "grid"
        else:
            level = int(attrs["level"])
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

        hover = f"node: {node}<br>pos: ({pos[0]:.3f}, {pos[1]:.3f})<br>{label}"
        groups[key]["x"].append(x_nodes[i])
        groups[key]["y"].append(y_nodes[i])
        groups[key]["z"].append(z_nodes[i])
        groups[key]["text"].append(hover)

    # Colour palette for mesh levels
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
    # Grid nodes first
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
                    size=node_size * 0.7,
                    color="#78909C",
                    symbol=g["symbol"],
                    opacity=0.6,
                ),
                text=g["text"],
                hoverinfo="text",
            )
        )

    # Mesh nodes, ordered by level
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


def _build_layout(
    title: str | None,
    layout_type: Literal["flat", "concentric"],
) -> "_go.Layout":
    """Create a clean Plotly layout with appropriate axis labels."""
    axis_style = dict(
        backgroundcolor="white",
        gridcolor="#E0E0E0",
        showbackground=True,
        zerolinecolor="#BDBDBD",
        tickfont=dict(size=10),
    )
    if layout_type == "flat":
        scene = dict(
            xaxis=dict(title="x (longitude)", **axis_style),
            yaxis=dict(title="y (latitude)", **axis_style),
            zaxis=dict(title="level", **axis_style),
            aspectmode="data",
        )
    else:  # concentric
        scene = dict(
            xaxis=dict(title="x", **axis_style),
            yaxis=dict(title="y", **axis_style),
            zaxis=dict(title="z", **axis_style),
            aspectmode="data",
        )
    return _go.Layout(
        title=dict(text=title or "Weather Model Graph (3D)", x=0.5),
        scene=scene,
        legend=dict(title=dict(text="component / node kind"), itemsizing="constant"),
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
    layout: Literal["flat", "concentric"] = "flat",
    add_coastlines: bool = False,
    sphere_base_radius: float = _DEFAULT_SPHERE_RADIUS_BASE,
    sphere_radius_step: float = _DEFAULT_SPHERE_RADIUS_STEP,
) -> "_go.Figure":
    """
    Render a weather-model-graphs networkx.DiGraph in 3D using Plotly.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph with 'pos' attribute on all nodes; optionally 'level' on mesh nodes.
    show : bool, default=True
        If True, call fig.show() before returning.
    title : str or None, optional
        Figure title.
    node_size : float, default=4.0
        Marker size for mesh nodes. Grid nodes are slightly smaller.
    edge_width : float, default=1.5
        Line width for edges.
    component_colors : dict or None, optional
        Mapping from edge component name to CSS color.
    layout : {"flat", "concentric"}, default="flat"
        Layout style.
    add_coastlines : bool, default=False
        If True, add a coastline layer (only effective in concentric layout).
    sphere_base_radius : float, default=1.0
        Radius for the innermost mesh level (concentric layout).
    sphere_radius_step : float, default=0.5
        Additional radius per level (concentric layout).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly is required for 3D visualisation but is not installed.\n"
            "Install it with:  pip install plotly\n"
            "or:               pip install weather-model-graphs[visualisation]"
        )

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot render an empty graph (no nodes).")

    missing_pos = [n for n in graph.nodes() if "pos" not in graph.nodes[n]]
    if missing_pos:
        raise ValueError(
            f"{len(missing_pos)} node(s) are missing the required 'pos' attribute. "
            f"First offending node: {missing_pos[0]!r}"
        )

    colors = {**DEFAULT_COMPONENT_COLORS, **(component_colors or {})}

    if layout == "concentric":
        x_nodes, y_nodes, z_nodes, node_ids = _get_positions_concentric(
            graph, base_radius=sphere_base_radius, radius_step=sphere_radius_step
        )
    else:
        x_nodes, y_nodes, z_nodes, node_ids = _get_positions_flat(graph)

    node_index = _node_id_to_index(node_ids)

    traces = []

    # Add coastline if requested (only meaningful in concentric layout)
    if add_coastlines:
        if layout == "concentric":
            # Place coastline slightly below the innermost mesh sphere
            coast_radius = sphere_base_radius - sphere_radius_step * 0.5
            if coast_radius <= 0:
                coast_radius = 0.5
            traces.append(_build_coastline_trace(radius=coast_radius))
        else:
            warnings.warn(
                "Coastlines are only supported in concentric layout. "
                "Ignoring add_coastlines=True.",
                UserWarning,
                stacklevel=2,
            )

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
            "Graph has no edges; rendering node positions only.", stacklevel=2
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

    fig = _go.Figure(data=traces, layout=_build_layout(title, layout))

    if show:
        fig.show()
    return fig
