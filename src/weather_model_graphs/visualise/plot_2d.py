import matplotlib.pyplot as plt
import networkx
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

from .. import networkx_utils as nx_utils


def nx_draw_with_pos(g, with_labels=False, **kwargs):
    pos = {node: g.nodes[node]["pos"] for node in g.nodes()}
    ax = kwargs.pop("ax", None)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
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
    **kwargs,
):
    """
    Create a networkx plot where edges and nodes can be coloured by attributes (
    both continuous and discrete attributes are supported, with a colorbar legend
    and a discrete legend respectively).

    Parameters
    ----------
    graph : networkx.Graph
        The graph to plot
    ax : matplotlib.axes.Axes, optional
        The axes to plot on, by default None (and a new figure is created)
    edge_color_attr : str, optional
        The attribute to use for edge coloring, by default None
    node_color_attr : str, optional
        The attribute to use for node coloring, by default None
    node_zorder_attr : str, optional
        The attribute to use for sorting nodes, by default None
    node_size : int, optional
        The size of the nodes, by default 100
    connectionstyle : str, optional
        The style of the edge connections, by default "arc3, rad=0.1"
        (giving curved edges)
    **kwargs : dict
        Additional keyword arguments to passed down to networkx.draw_networkx

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot
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
        with_labels=False,
        node_size=node_size,
        connectionstyle=connectionstyle,
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
