import numpy as np
import weather_model_graphs as wmg
import tempfile


def _create_fake_xy(N=10):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xy = np.meshgrid(x, y)
    xy = np.stack(xy, axis=0)
    return xy


def test_save_to_pyg():
    xy = _create_fake_xy(N=64)
    graph = wmg.create.architypes.create_oscarsson_hierarchical_graph(xy_grid=xy)
    
    graph_components = wmg.split_graph_by_edge_attribute(graph=graph, attribute="component")

    # split the m2m graph into the different parts that create the up, in-level and down connections respectively
    # this is how the graphs is stored in the neural-lam codebase
    m2m_graph = graph_components.pop("m2m")
    m2m_graph_components = wmg.split_graph_by_edge_attribute(graph=m2m_graph, attribute="direction")
    m2m_graph_components = {
        f"m2m_{name}": graph for name, graph in m2m_graph_components.items()
    }
    graph_components.update(m2m_graph_components)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, graph in graph_components.items():
            wmg.save.to_pyg(graph=graph, output_directory=tmpdir, name=name)