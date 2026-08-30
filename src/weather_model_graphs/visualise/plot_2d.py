import logging

import matplotlib.pyplot as plt
import networkx
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

from .. import networkx_utils as nx_utils

logger = logging.getLogger(__name__)

_NODE_ARRAY_KWARGS = {"node_color", "node_size"}

_EDGE_ARRAY_KWARGS = {"edge_color"}


def _filter_nan_positions(g, pos, **kwargs):
    """
    Remove nodes whose positions contain NaN values and return a filtered
    graph, position dict, and kwargs arrays.

    NaN positions arise when a cartopy projection (e.g. Orthographic) is
    used with ``coords_crs`` and some nodes fall on the far side of the
    globe.  These NaN coordinates cannot be handled by matplotlib's
    ``FancyArrowPatch`` (used by networkx when ``arrows=True``), causing
    a ``StopIteration`` crash.  Dropping the invisible nodes and their
    incident edges avoids the problem.
    """
    nodes = list(g.nodes())
    valid_mask = np.array([not np.any(np.isnan(pos[n])) for n in nodes])

    if np.all(valid_mask):
        return g, pos, kwargs

    num_filtered = len(nodes) - np.sum(valid_mask)
    logger.warning(
        "%d node(s) are outside the visible map extent and will not be drawn. "
        "Edges connecting to these nodes will also be omitted.",
        num_filtered,
    )

    valid_nodes = [n for i, n in enumerate(nodes) if valid_mask[i]]
    valid_set = set(valid_nodes)

    if not valid_set:
        raise ValueError(
            "All node positions are NaN after CRS transformation — "
            "cannot draw the graph. Check that the axes projection "
            "is compatible with the data CRS."
        )

    for key in _NODE_ARRAY_KWARGS:
        if key in kwargs and isinstance(kwargs[key], (np.ndarray, list)):
            kwargs[key] = np.array(kwargs[key])[valid_mask]

    orig_edges = list(g.edges())
    edge_valid_mask = np.array(
        [u in valid_set and v in valid_set for (u, v) in orig_edges]
    )
    for key in _EDGE_ARRAY_KWARGS:
        if key in kwargs and isinstance(kwargs[key], (np.ndarray, list)):
            kwargs[key] = np.array(kwargs[key])[edge_valid_mask]

    g = g.subgraph(valid_nodes).copy()
    pos = {n: pos[n] for n in valid_nodes}

    return g, pos, kwargs


def nx_draw_with_pos(g, with_labels=False, coords_crs=None, **kwargs):
    """Draw a networkx graph using the ``pos`` attribute on each node.

    Parameters
    ----------
    g : networkx.Graph
        The graph to draw. Each node must have a ``pos`` attribute containing
        a 2-element array-like of (x, y) coordinates.
    with_labels : bool, optional
        Whether to draw node labels, by default False.
    coords_crs : cartopy.crs.CRS or pyproj.crs.CRS, optional
        The coordinate reference system of the node positions. Use this when
        drawing on a cartopy GeoAxes with a map projection, since
        ``networkx.draw_networkx`` does not support a ``transform`` argument
        for CRS reprojection. When set and the axes has a ``.projection``
        attribute (cartopy GeoAxes), positions are transformed from
        *coords_crs* to the axes projection before drawing.
    **kwargs : dict
        Additional keyword arguments passed to ``networkx.draw_networkx``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    pos = {node: g.nodes[node]["pos"] for node in g.nodes()}
    ax = kwargs.pop("ax", None)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    if coords_crs is not None and hasattr(ax, "projection"):
        nodes = list(g.nodes())
        xs = np.array([pos[n][0] for n in nodes])
        ys = np.array([pos[n][1] for n in nodes])
        transformed = ax.projection.transform_points(coords_crs, xs, ys)
        pos = {n: transformed[i, :2] for i, n in enumerate(nodes)}
        g, pos, kwargs = _filter_nan_positions(g, pos, **kwargs)

    networkx.draw_networkx(
        ax=ax, G=g, pos=pos, hide_ticks=False, with_labels=with_labels, **kwargs
    )

    return ax


def _get_graph_attr_values(g, attr_name, component="edges"):
    if component == "edges":
        features = list(g.edges(data=True))[0][2].keys()
    elif component == "nodes":
        features = list(g.nodes(data=True))[0][1].keys()
    else:
        raise ValueError(
            f"`component` should be either 'edges' or 'nodes', but got '{component}'"
        )

    if attr_name not in features:
        raise ValueError(f"feature {attr_name} not in {component} features {features}")

    if component == "edges":
        attr_vals = np.array([g.edges[edge][attr_name] for edge in g.edges()])
    elif component == "nodes":
        attr_vals = np.array([g.nodes[node][attr_name] for node in g.nodes()])

    attr_vals_for_plot = dict()

    if len(attr_vals.shape) > 1:
        raise NotImplementedError("Can't use multi-dimensional features for colors")
    elif np.issubdtype(attr_vals.dtype, np.str_):
        unique_strings = np.unique(attr_vals)
        val_str_map = {s: i for (i, s) in enumerate(unique_strings)}
        plot_values = np.array([val_str_map[s] for s in attr_vals])
        attr_vals_for_plot["values"] = plot_values
        attr_vals_for_plot["discrete_labels"] = val_str_map
    elif np.issubdtype(attr_vals.dtype, np.integer):
        unique_ints = np.unique(attr_vals)
        val_int_map = {val: i for (i, val) in enumerate(unique_ints)}
        plot_values = np.array([val_int_map[s] for s in attr_vals])
        attr_vals_for_plot["values"] = plot_values
        attr_vals_for_plot["discrete_labels"] = val_int_map
    elif np.issubdtype(attr_vals.dtype, np.floating):
        attr_vals_for_plot["values"] = attr_vals
    else:
        raise NotImplementedError(
            f"Array feature values of type {type(attr_vals[0])} not supported"
        )

    return attr_vals_for_plot


def _create_graph_attr_legend(
    ax, discrete_labels, cmap, attr_kind, attr_name, loc, norm
):
    if attr_kind == "edge":
        kwargs = dict(marker="")
        colouring = "color"
    elif attr_kind == "node":
        kwargs = dict(marker="o", color="w")
        colouring = "markerfacecolor"

    legend_handles = [
        plt.Line2D([0], [0], label=label, **kwargs, **{colouring: cmap(norm(val))})
        for (label, val) in discrete_labels.items()
    ]
    legend = ax.legend(
        handles=legend_handles, title=f"{attr_kind} {attr_name}", loc=loc
    )
    return legend


def _create_graph_attr_colorbar(ax, cmap, norm, attr_name, attr_kind, loc):
    if loc == "upper left":
        ax_inset = ax.inset_axes([0.05, 0.94, 0.1, 0.02])
    elif loc == "upper right":
        ax_inset = ax.inset_axes([0.87, 0.94, 0.1, 0.02])
    else:
        raise ValueError(f"loc {loc} not in ['upper left', 'upper right']")

    cbar = ColorbarBase(ax=ax_inset, cmap=cmap, norm=norm, orientation="horizontal")

    ax_inset.set_title(f"{attr_kind} {attr_name}", fontsize=10)
    return cbar


def nx_draw_with_pos_and_attr(
    graph,
    ax=None,
    edge_color_attr=None,
    node_color_attr=None,
    node_zorder_attr=None,
    node_size=100,
    connectionstyle="arc3, rad=0.1",
    with_labels=False,
    coords_crs=None,
    **kwargs,
):
    """Draw a networkx graph with edges and/or nodes coloured by attributes.

    Both continuous and discrete attributes are supported, with a colorbar
    legend and a discrete legend respectively.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to draw. Each node must have a ``pos`` attribute containing
        a 2-element array-like of (x, y) coordinates.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.
    edge_color_attr : str, optional
        Attribute name used to colour edges. Values are mapped through the
        colormap (``edge_cmap``) or used as discrete categories.
    node_color_attr : str, optional
        Attribute name used to colour nodes. Values are mapped through the
        colormap (``cmap``) or used as discrete categories.
    node_zorder_attr : str, optional
        Attribute name used to determine the z-order of nodes. Nodes are
        drawn in ascending attribute order via
        :func:`~weather_model_graphs.networkx_utils.sort_nodes_internally`.
    node_size : int, optional
        Size of the nodes, by default 100.
    connectionstyle : str, optional
        Style of edge connections passed to ``networkx.draw_networkx_edges``,
        by default ``"arc3, rad=0.1"`` (giving curved edges).
    with_labels : bool, optional
        Whether to draw node labels, by default False.
    coords_crs : cartopy.crs.CRS or pyproj.crs.CRS, optional
        The coordinate reference system of the node positions. Use this when
        drawing on a cartopy GeoAxes with a map projection, since
        ``networkx.draw_networkx`` does not support a ``transform`` argument
        for CRS reprojection. When set and the axes has a ``.projection``
        attribute (cartopy GeoAxes), positions are transformed from
        *coords_crs* to the axes projection before drawing.
    **kwargs : dict
        Additional keyword arguments passed to ``networkx.draw_networkx``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if node_zorder_attr is not None:
        graph = nx_utils.sort_nodes_internally(graph, node_attr=node_zorder_attr)

    if edge_color_attr is not None:
        edge_attr_vals = _get_graph_attr_values(
            graph, edge_color_attr, component="edges"
        )

        if "cmap" not in kwargs:
            if "discrete_labels" in edge_attr_vals:
                kwargs["edge_cmap"] = plt.get_cmap("tab20")
            else:
                kwargs["edge_cmap"] = plt.get_cmap("viridis")
        kwargs["edge_color"] = edge_attr_vals["values"]
        kwargs["edge_vmin"] = min(edge_attr_vals["values"])
        kwargs["edge_vmax"] = max(edge_attr_vals["values"])

    if node_color_attr is not None:
        node_attr_vals = _get_graph_attr_values(
            graph, node_color_attr, component="nodes"
        )
        if "cmap" not in kwargs:
            if "discrete_labels" in node_attr_vals:
                kwargs["cmap"] = plt.get_cmap("tab20")
            else:
                kwargs["cmap"] = plt.get_cmap("viridis")
        kwargs["node_color"] = node_attr_vals["values"]
        kwargs["vmin"] = min(node_attr_vals["values"])
        kwargs["vmax"] = max(node_attr_vals["values"])

    ax = nx_draw_with_pos(
        graph,
        ax=ax,
        arrows=True,
        with_labels=with_labels,
        node_size=node_size,
        connectionstyle=connectionstyle,
        coords_crs=coords_crs,
        **kwargs,
    )

    legends = []

    if node_color_attr is not None:
        norm = Normalize(vmin=kwargs["vmin"], vmax=kwargs["vmax"])
        if "discrete_labels" in node_attr_vals:
            legend = _create_graph_attr_legend(
                ax=ax,
                discrete_labels=node_attr_vals["discrete_labels"],
                cmap=kwargs["cmap"],
                attr_kind="node",
                attr_name=node_color_attr,
                loc="upper left",
                norm=norm,
            )
            legends.append(legend)
        else:
            _create_graph_attr_colorbar(
                ax=ax,
                cmap=kwargs["cmap"],
                norm=norm,
                attr_name=node_color_attr,
                loc="upper left",
                attr_kind="node",
            )

    if edge_color_attr is not None:
        norm = Normalize(vmin=kwargs["edge_vmin"], vmax=kwargs["edge_vmax"])
        if "discrete_labels" in edge_attr_vals:
            legend = _create_graph_attr_legend(
                ax=ax,
                discrete_labels=edge_attr_vals["discrete_labels"],
                cmap=kwargs["edge_cmap"],
                attr_kind="edge",
                attr_name=edge_color_attr,
                loc="upper right",
                norm=norm,
            )
            legends.append(legend)
        else:
            _create_graph_attr_colorbar(
                ax=ax,
                cmap=kwargs["edge_cmap"],
                norm=norm,
                attr_name=edge_color_attr,
                loc="upper right",
                attr_kind="edge",
            )

    for legend in legends:
        ax.add_artist(legend)

    return ax
