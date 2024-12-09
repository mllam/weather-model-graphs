import tempfile

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

import tests.utils as test_utils
import weather_model_graphs as wmg


def test_create_single_level_mesh_graph():
    xy = test_utils.create_fake_xy(N=4)
    mesh_graph = wmg.create.mesh.create_single_level_2d_mesh_graph(xy=xy, nx=5, ny=5)

    pos = {node: mesh_graph.nodes[node]["pos"] for node in mesh_graph.nodes()}
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    nx.draw_networkx(ax=ax, G=mesh_graph, pos=pos, with_labels=True, hide_ticks=False)

    ax.scatter(xy[0, ...], xy[1, ...], color="r")
    ax.axison = True

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        fig.savefig(f.name)


@pytest.mark.parametrize("kind", ["graphcast", "keisler", "oskarsson_hierarchical"])
def test_create_graph_archetype(kind):
    xy = test_utils.create_fake_xy(N=64)
    fn_name = f"create_{kind}_graph"
    fn = getattr(wmg.create.archetype, fn_name)

    fn(coords=xy)


# list the connectivity options for g2m and m2g and the kwargs to test
G2M_CONNECTIVITY_OPTIONS = dict(
    nearest_neighbour=[],
    nearest_neighbours=[dict(max_num_neighbours=4), dict(max_num_neighbours=8)],
    within_radius=[
        dict(max_dist=3.2),
        dict(max_dist=6.4),
        dict(rel_max_dist=0.51),
        dict(rel_max_dist=1.0),
    ],
)
# containing_rectangle option should only be used for m2g
M2G_CONNECTIVITY_OPTIONS = G2M_CONNECTIVITY_OPTIONS.copy()
M2G_CONNECTIVITY_OPTIONS["containing_rectangle"] = [dict()]

# list the connectivity options for m2m and the kwargs to test
M2M_CONNECTIVITY_OPTIONS = dict(
    flat=[],
    flat_multiscale=[
        dict(max_num_levels=3, mesh_node_distance=3, level_refinement_factor=3),
        dict(max_num_levels=1, mesh_node_distance=5, level_refinement_factor=5),
    ],
    hierarchical=[
        dict(max_num_levels=3, mesh_node_distance=3, level_refinement_factor=3),
        dict(max_num_levels=None, mesh_node_distance=3, level_refinement_factor=3),
    ],
)


@pytest.mark.parametrize("g2m_connectivity", G2M_CONNECTIVITY_OPTIONS.keys())
@pytest.mark.parametrize("m2g_connectivity", M2G_CONNECTIVITY_OPTIONS.keys())
@pytest.mark.parametrize("m2m_connectivity", M2M_CONNECTIVITY_OPTIONS.keys())
def test_create_graph_generic(m2g_connectivity, g2m_connectivity, m2m_connectivity):
    xy = test_utils.create_fake_xy(N=32)

    for g2m_kwargs in G2M_CONNECTIVITY_OPTIONS[g2m_connectivity]:
        for m2g_kwargs in M2G_CONNECTIVITY_OPTIONS[m2g_connectivity]:
            for m2m_kwargs in M2M_CONNECTIVITY_OPTIONS[m2m_connectivity]:
                graph = wmg.create.create_all_graph_components(
                    coords=xy,
                    m2m_connectivity=m2m_connectivity,
                    m2m_connectivity_kwargs=m2m_kwargs,
                    g2m_connectivity=g2m_connectivity,
                    g2m_connectivity_kwargs=g2m_kwargs,
                    m2g_connectivity=m2g_connectivity,
                    m2g_connectivity_kwargs=m2g_kwargs,
                )

                assert isinstance(graph, nx.DiGraph)

                graph_components = wmg.split_graph_by_edge_attribute(
                    graph=graph, attr="component"
                )
                assert all(
                    isinstance(graph, nx.DiGraph) for graph in graph_components.values()
                )
                assert set(graph_components.keys()) == {"m2m", "m2g", "g2m"}


@pytest.mark.parametrize("kind", ["graphcast", "keisler", "oskarsson_hierarchical"])
def test_create_rectangular_graph(kind):
    """
    Tests that graphs can be created for non-square areas, both thin and wide
    """
    # Test thin
    xy = test_utils.create_rectangular_fake_xy(Nx=20, Ny=64)
    fn_name = f"create_{kind}_graph"
    fn = getattr(wmg.create.archetype, fn_name)
    fn(coords=xy, mesh_node_distance=2)

    # Test wide
    xy = test_utils.create_rectangular_fake_xy(Nx=64, Ny=20)
    fn_name = f"create_{kind}_graph"
    fn = getattr(wmg.create.archetype, fn_name)
    fn(coords=xy, mesh_node_distance=2)


@pytest.mark.parametrize("mesh_node_distance", (2, 3))
@pytest.mark.parametrize("level_refinement_factor", (2, 3, 5))
def test_create_exact_refinement(mesh_node_distance, level_refinement_factor):
    """
    This test is to check that it is possible to create graph hierarchies when
    the refinement factors are an exact multiple of the number of nodes. In these
    situations it should be possible to create multi-level graphs, but it was not
    earlier due to numerical issues.
    """
    N = mesh_node_distance * (level_refinement_factor**2)
    xy = test_utils.create_fake_xy(N)

    # Build hierarchical graph, should have 2 levels and not give error
    wmg.create.archetype.create_oskarsson_hierarchical_graph(
        xy,
        mesh_node_distance=mesh_node_distance,
        level_refinement_factor=level_refinement_factor,
    )


@pytest.mark.parametrize(
    "kind_and_num_mesh",
    [
        ("keisler", 20**2),  # 20 mesh nodes in bottom layer in each direction
        ("graphcast", 9**2),  # Can only fit 9 x 9 with level_refinement_factor=3
        (
            "oskarsson_hierarchical",
            9**2 + 3**2,
        ),  # As above, with additional 3 x 3 layer
    ],
)
def test_create_irregular_grid(kind_and_num_mesh):
    """
    Tests that graphs can be created for irregular layouts of grid points
    """
    kind, num_mesh = kind_and_num_mesh
    num_grid = 100
    xy = test_utils.create_fake_irregular_coords(num_grid - 4)

    # Need to include corners if we  want to know actual size of covered area
    xy = np.concatenate(
        (
            xy,
            np.array(
                [[0.0, 0.0], [0.0, 1.0], [1.0, 0], [1.0, 1.0]]
            ),  # Remaining 4 nodes
        ),
        axis=0,
    )

    fn_name = f"create_{kind}_graph"
    fn = getattr(wmg.create.archetype, fn_name)

    graph = fn(coords=xy, mesh_node_distance=0.05)

    assert len(graph.nodes) == num_grid + num_mesh


@pytest.mark.parametrize("kind", ["graphcast", "keisler", "oskarsson_hierarchical"])
def test_create_lat_lon(kind):
    """
    Tests that graphs can be created from lat-lon coordinates + projection spec.
    """
    lon_coords = np.linspace(10, 30, 10)
    lat_coords = np.linspace(35, 65, 10)
    coords_crs = ccrs.PlateCarree()
    graph_crs = ccrs.LambertConformal()
    mesh_node_distance = 0.2 * 10**6

    meshgridded = np.meshgrid(lon_coords, lat_coords)
    coords = np.stack([mg_coord.flatten() for mg_coord in meshgridded], axis=1)

    fn_name = f"create_{kind}_graph"
    fn = getattr(wmg.create.archetype, fn_name)

    fn(
        coords=coords,
        mesh_node_distance=mesh_node_distance,
        coords_crs=coords_crs,
        graph_crs=graph_crs,
    )


@pytest.mark.parametrize("kind", ["graphcast", "keisler", "oskarsson_hierarchical"])
def test_create_decode_mask(kind):
    """
    Tests that the decode mask for m2g works, resulting in less edges than
    no filtering.
    """
    xy = test_utils.create_fake_irregular_coords(100)
    fn_name = f"create_{kind}_graph"
    fn = getattr(wmg.create.archetype, fn_name)
    # ~= 20 mesh nodes in bottom layer in each direction
    mesh_node_distance = 0.05

    unfiltered_graph = fn(coords=xy, mesh_node_distance=mesh_node_distance)

    # Filter to only 20 / 100 grid nodes
    decode_mask = np.concatenate((np.ones(20), np.zeros(80))).astype(bool)
    filtered_graph = fn(
        coords=xy, mesh_node_distance=mesh_node_distance, decode_mask=decode_mask
    )

    # Check that some filtering has been performed
    assert len(filtered_graph.edges) < len(unfiltered_graph.edges)
