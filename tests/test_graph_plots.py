import tempfile

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pytest

import tests.utils as test_utils
import weather_model_graphs as wmg


def test_plot():
    xy = test_utils.create_fake_xy(10)

    graph = wmg.create.create_all_graph_components(
        m2m_connectivity="flat_multiscale",
        coords=xy,
        mesh_layout="rectilinear",
        m2m_connectivity_kwargs=dict(
            max_num_levels=3,
            mesh_node_distance=2,
            level_refinement_factor=3,
        ),
        g2m_connectivity="nearest_neighbour",
        m2g_connectivity="nearest_neighbour",
    )

    graph_components = wmg.split_graph_by_edge_attribute(graph=graph, attr="component")

    def _is_ndarray(val):
        return isinstance(val, np.ndarray)

    def _is_valid_color_attr(val):
        return isinstance(val, (int, float, str))

    fig, ax = plt.subplots()
    for graph in graph_components.values():
        node_attrs = list(list(graph.nodes(data=True))[0][1].keys())
        edge_attrs = list(list(graph.edges(data=True))[0][2].keys())

        for edge_attr in edge_attrs + []:
            for node_attr in node_attrs + []:
                should_raise = None
                if not _is_valid_color_attr(
                    list(graph.edges(data=True))[0][2][edge_attr]
                ):
                    should_raise = NotImplementedError
                elif not _is_valid_color_attr(
                    list(graph.nodes(data=True))[0][1][node_attr]
                ):
                    should_raise = NotImplementedError

                def fn():
                    wmg.visualise.nx_draw_with_pos_and_attr(
                        graph,
                        ax=ax,
                        edge_color_attr=edge_attr,
                        node_color_attr=node_attr,
                    )

                if should_raise is not None:
                    with pytest.raises(should_raise):
                        fn()
                else:
                    fn()

    with tempfile.NamedTemporaryFile(suffix=".png") as fh:
        fig.savefig(fh)


def test_plot_with_projection():
    longitudes = np.linspace(-90, 90, 10)
    latitudes = np.linspace(-45, 45, 10)
    meshgridded_lat_lons = np.meshgrid(longitudes, latitudes)
    coords = np.stack([mg_coord.flatten() for mg_coord in meshgridded_lat_lons], axis=1)

    graph = wmg.create.archetype.create_keisler_graph(
        coords,
        mesh_node_distance=30,
        coords_crs=ccrs.PlateCarree(),
        graph_crs=ccrs.PlateCarree(),
    )

    fig, ax = plt.subplots(
        subplot_kw={
            "projection": ccrs.Orthographic(central_longitude=0, central_latitude=0)
        }
    )

    wmg.visualise.nx_draw_with_pos(
        graph,
        ax=ax,
        node_size=30,
        edge_color="gray",
        node_color="blue",
        with_labels=False,
        coords_crs=ccrs.PlateCarree(),
    )
    ax.coastlines()

    with tempfile.NamedTemporaryFile(suffix=".png") as fh:
        fig.savefig(fh)


def test_plot_with_projection_and_attr():
    longitudes = np.linspace(-90, 90, 10)
    latitudes = np.linspace(-45, 45, 10)
    meshgridded_lat_lons = np.meshgrid(longitudes, latitudes)
    coords = np.stack([mg_coord.flatten() for mg_coord in meshgridded_lat_lons], axis=1)

    graph = wmg.create.archetype.create_keisler_graph(
        coords,
        mesh_node_distance=30,
        coords_crs=ccrs.PlateCarree(),
        graph_crs=ccrs.PlateCarree(),
    )

    fig, ax = plt.subplots(
        subplot_kw={
            "projection": ccrs.Orthographic(central_longitude=0, central_latitude=0)
        }
    )

    wmg.visualise.nx_draw_with_pos_and_attr(
        graph,
        ax=ax,
        node_size=30,
        coords_crs=ccrs.PlateCarree(),
    )
    ax.coastlines()

    with tempfile.NamedTemporaryFile(suffix=".png") as fh:
        fig.savefig(fh)


def test_plot_with_projection_full_globe():
    """Reproduces the notebook scenario: full-globe PlateCarree graph on
    Orthographic projection.  Nodes on the far side become NaN after
    coordinate transformation — the helper must filter them out."""
    longitudes = np.linspace(-180, 180, 20)
    latitudes = np.linspace(-90, 90, 20)
    meshgridded_lat_lons = np.meshgrid(longitudes, latitudes)
    coords = np.stack([mg_coord.flatten() for mg_coord in meshgridded_lat_lons], axis=1)

    graph = wmg.create.archetype.create_keisler_graph(
        coords,
        mesh_node_distance=30,
        coords_crs=ccrs.PlateCarree(),
        graph_crs=ccrs.PlateCarree(),
    )

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Orthographic()})

    wmg.visualise.nx_draw_with_pos_and_attr(
        graph,
        ax=ax,
        node_size=30,
        edge_color_attr="component",
        node_color_attr="type",
        coords_crs=ccrs.PlateCarree(),
    )
    ax.coastlines()

    with tempfile.NamedTemporaryFile(suffix=".png") as fh:
        fig.savefig(fh)
