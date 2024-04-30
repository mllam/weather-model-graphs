import weather_model_graphs as wmg

from loguru import logger
import numpy as np
from torch_geometric.utils.convert import from_networkx as pyg_from_networkx
import pytest
import networkx as nx
import matplotlib.pyplot as plt
import tempfile


def _create_fake_xy(N=10):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xy = np.meshgrid(x, y)
    xy = np.stack(xy, axis=0)
    return xy


def test_create_single_level_mesh_graph():
    xy = _create_fake_xy(N=4)
    mesh_graph = wmg.mesh.create_single_level_2d_mesh_graph(xy=xy, nx=5, ny=5)

    pos = {node: mesh_graph.nodes[node]["pos"] for node in mesh_graph.nodes()}
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    nx.draw_networkx(
        ax=ax, G=mesh_graph, pos=pos, with_labels=True, hide_ticks=False
    )

    ax.scatter(xy[0, ...], xy[1, ...], color="r")
    ax.axison = True

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        fig.savefig(f.name)

    lev = 0
    s = 8.0
    Nx, Ny = xy.shape[1:]
    r = Nx / Ny
    fig, ax = plt.subplots(figsize=(s * r + 2.0, s), dpi=200)
    ax.scatter(xy[0], xy[1], color="r", marker=".", s=0.5, alpha=0.5)
    g = mesh_graph
    logger.warning("Need to implemented 2D plot in test")
    # wmg.plot_2d.plot_graph(pyg_from_networkx(g), ax=ax, title=f"Mesh graph, level {lev}")
    # fig.savefig(f"mesh_{lev}.png")


@pytest.mark.parametrize("kind", ["graphcast", "keissler", "oscarsson_hierarchical"])
@pytest.mark.parametrize("merge_components", [True, False])
def test_create_graph_architype(merge_components, kind):
    xy = _create_fake_xy(N=64)
    fn_name = f"create_{kind}_graph"
    fn = getattr(wmg.create.architypes, fn_name)

    fn(xy_grid=xy, merge_components=merge_components)

# list the connectivity options for g2m and m2g and the kwargs to test
G2M_M2G_CONNECTIVITY_OPTIONS = dict(
    nearest_neighbour=[],
    nearest_neighbours=[dict(max_num_neighbours=4), dict(max_num_neighbours=8)],
    within_radius=[dict(max_dist=0.1), dict(max_dist=0.2)],
)

# list the connectivity options for m2m and the kwargs to test
M2M_CONNECTIVITY_OPTIONS = dict(
    # flat=[],
    # flat_multiscale=[dict(max_num_levels=3, refinement_factor=3), dict(max_num_levels=1, refinement_factor=5)],
    hierarchical=[dict(max_num_levels=3, refinement_factor=3), dict(max_num_levels=None, refinement_factor=3)],
)

@pytest.mark.parametrize("g2m_connectivity", G2M_M2G_CONNECTIVITY_OPTIONS.keys())
@pytest.mark.parametrize("m2g_connectivity", G2M_M2G_CONNECTIVITY_OPTIONS.keys())
@pytest.mark.parametrize("m2m_connectivity", M2M_CONNECTIVITY_OPTIONS.keys())
@pytest.mark.parametrize("merge_components", [True, False])
def test_create_graph_generic(m2g_connectivity, g2m_connectivity, m2m_connectivity, merge_components):
    xy = _create_fake_xy(N=32)
    
    for g2m_kwargs in G2M_M2G_CONNECTIVITY_OPTIONS[g2m_connectivity]:
        for m2g_kwargs in G2M_M2G_CONNECTIVITY_OPTIONS[m2g_connectivity]:
            for m2m_kwargs in M2M_CONNECTIVITY_OPTIONS[m2m_connectivity]:
                result = wmg.create.create_all_graph_components(
                    xy=xy,
                    merge_components=merge_components,
                    m2m_connectivity=m2m_connectivity,
                    m2m_connectivity_kwargs=m2m_kwargs,
                    g2m_connectivity=g2m_connectivity,
                    g2m_connectivity_kwargs=g2m_kwargs,
                    m2g_connectivity=m2g_connectivity,
                    m2g_connectivity_kwargs=m2g_kwargs,
                )
            
                if merge_components:
                    graph = result
                    assert isinstance(graph, nx.DiGraph)
                else:
                    graph_components = result
                    assert all(isinstance(graph, nx.DiGraph) for graph in graph_components.values())
                    assert set(graph_components.keys()) == {"m2m", "m2g", "g2m"}
