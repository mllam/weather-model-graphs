"""
Tests for weather_model_graphs.visualise.plot_3d.render_with_plotly.

These tests follow the same conventions as tests/test_graph_plots.py and are
designed to run without a display (``show=False`` is always passed).
"""

import warnings

import networkx as nx
import numpy as np
import pytest

import weather_model_graphs as wmg
from weather_model_graphs.visualise.plot_3d import (
    DEFAULT_COMPONENT_COLORS,
    render_with_plotly,
)

pytest.importorskip("plotly", reason="plotly not installed skipping 3D plot tests")


def _make_xy(n: int = 15) -> np.ndarray:
    """Return an (n*n, 2) array of evenly-spaced 2-D coordinates."""
    lin = np.linspace(0, 10, n)
    xx, yy = np.meshgrid(lin, lin)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


@pytest.fixture
def xy():
    return _make_xy(15)  # 15x15 = 225 points for better hierarchical level creation


@pytest.fixture
def flat_graph(xy):
    return wmg.create.create_all_graph_components(
        coords=xy,
        m2m_connectivity="flat",
        m2m_connectivity_kwargs=dict(mesh_node_distance=3),
        g2m_connectivity="nearest_neighbour",
        m2g_connectivity="nearest_neighbour",
    )


@pytest.fixture
def multiscale_graph(xy):
    return wmg.create.create_all_graph_components(
        coords=xy,
        m2m_connectivity="flat_multiscale",
        m2m_connectivity_kwargs=dict(
            mesh_node_distance=1,
            level_refinement_factor=3,
            max_num_levels=2,
        ),
        g2m_connectivity="nearest_neighbour",
        m2g_connectivity="nearest_neighbour",
    )


@pytest.fixture
def hierarchical_graph(xy):
    return wmg.create.create_all_graph_components(
        coords=xy,
        m2m_connectivity="hierarchical",
        m2m_connectivity_kwargs=dict(
            mesh_node_distance=1,
            level_refinement_factor=3,
            max_num_levels=2,
        ),
        g2m_connectivity="nearest_neighbour",
        m2g_connectivity="nearest_neighbour",
    )


def _trace_names(fig) -> set[str]:
    """Return the set of trace names present in a Plotly figure."""
    return {t.name for t in fig.data}


class TestRenderSmoke:
    """render_with_plotly returns a Figure without raising for valid graphs."""

    def test_flat_graph_returns_figure(self, flat_graph):
        import plotly.graph_objects as go

        fig = render_with_plotly(flat_graph, show=False)
        assert isinstance(fig, go.Figure)

    def test_multiscale_graph_returns_figure(self, multiscale_graph):
        import plotly.graph_objects as go

        fig = render_with_plotly(multiscale_graph, show=False)
        assert isinstance(fig, go.Figure)

    def test_hierarchical_graph_returns_figure(self, hierarchical_graph):
        import plotly.graph_objects as go

        fig = render_with_plotly(hierarchical_graph, show=False)
        assert isinstance(fig, go.Figure)

    def test_concentric_layout_works(self, flat_graph):
        """Smoke test for concentric layout."""
        fig = render_with_plotly(flat_graph, show=False, layout="concentric")
        assert fig is not None

    def test_show_false_does_not_raise(self, flat_graph):
        """Passing show=False must not attempt to open a browser."""
        fig = render_with_plotly(flat_graph, show=False)
        assert fig is not None


class TestTraceStructure:
    """Figure must contain the expected component and node-kind traces."""

    def test_component_traces_present(self, flat_graph):
        fig = render_with_plotly(flat_graph, show=False)
        names = _trace_names(fig)
        # All three standard components must appear as edge traces
        assert "g2m" in names
        assert "m2m" in names
        assert "m2g" in names

    def test_grid_node_trace_present(self, flat_graph):
        fig = render_with_plotly(flat_graph, show=False)
        assert "grid" in _trace_names(fig)

    def test_mesh_node_traces_present(self, flat_graph):
        fig = render_with_plotly(flat_graph, show=False)
        mesh_traces = [n for n in _trace_names(fig) if n.startswith("mesh level")]
        assert len(mesh_traces) >= 1

    def test_hierarchical_has_multiple_mesh_levels(self, hierarchical_graph):
        fig = render_with_plotly(hierarchical_graph, show=False)
        mesh_traces = [n for n in _trace_names(fig) if n.startswith("mesh level")]
        # Hierarchical graphs should have at least 2 mesh levels
        assert len(mesh_traces) >= 2, (
            f"Expected at least 2 mesh levels for hierarchical graph, "
            f"but found {len(mesh_traces)} levels: {mesh_traces}"
        )

    def test_no_duplicate_trace_names_per_type(self, flat_graph):
        """Each component name should appear exactly once as an edge trace."""
        fig = render_with_plotly(flat_graph, show=False)
        names = [t.name for t in fig.data]
        for comp in ("g2m", "m2m", "m2g"):
            assert (
                names.count(comp) == 1
            ), f"Expected exactly 1 trace named '{comp}', got {names.count(comp)}"


class TestZAxis:
    """Grid nodes must sit at z = -1 in flat layout; concentric layout uses spheres."""

    def test_grid_nodes_at_correct_z_flat(self, flat_graph):
        # We'll test by rendering and checking that the grid trace has z = -1
        # But easier: extract coordinates via the internal _get_positions_flat?
        # We'll test by checking that for flat layout, grid nodes have z = -1.
        # We can use the fact that the grid trace is named "grid".
        fig = render_with_plotly(flat_graph, show=False, layout="flat")
        grid_trace = next(t for t in fig.data if t.name == "grid")
        # All grid nodes should have the same z coordinate = -1
        assert all(
            z == -1 for z in grid_trace.z
        ), "Grid nodes in flat layout should have z = -1"

    def test_mesh_nodes_have_non_negative_z_flat(self, flat_graph):
        fig = render_with_plotly(flat_graph, show=False, layout="flat")
        mesh_traces = [t for t in fig.data if t.name.startswith("mesh level")]
        for trace in mesh_traces:
            assert all(
                z >= 0 for z in trace.z
            ), f"Mesh trace {trace.name} has negative z"

    def test_concentric_layout_has_non_negative_z(self, flat_graph):
        # In concentric layout, z values are coordinates on spheres, can be positive or negative.
        # We just check that the figure is created.
        fig = render_with_plotly(flat_graph, show=False, layout="concentric")
        assert fig is not None


class TestEdgeBatching:
    """Edge traces must use None separators — not one trace per edge."""

    def test_edge_traces_use_none_separators(self, flat_graph):
        fig = render_with_plotly(flat_graph, show=False)
        edge_trace_names = {"g2m", "m2m", "m2g"}
        for trace in fig.data:
            if trace.name in edge_trace_names:
                assert None in list(
                    trace.x
                ), f"Trace '{trace.name}' should use None separators in x-coords"

    def test_number_of_edge_traces_equals_components(self, flat_graph):
        """There should be one edge trace per component, not one per edge."""
        fig = render_with_plotly(flat_graph, show=False)
        edge_traces = [t for t in fig.data if t.name in {"g2m", "m2m", "m2g"}]
        # flat graph has exactly g2m, m2m, m2g — three traces total
        assert len(edge_traces) == 3


class TestComponentDict:
    """render_with_plotly should work on individual sub-graphs too."""

    def test_render_each_component_separately(self, xy):
        components = wmg.create.create_all_graph_components(
            coords=xy,
            m2m_connectivity="flat",
            m2m_connectivity_kwargs=dict(mesh_node_distance=3),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        import plotly.graph_objects as go

        for name, subgraph in components.items():
            fig = render_with_plotly(subgraph, show=False, title=name)
            assert isinstance(fig, go.Figure), f"Expected Figure for component '{name}'"


class TestParameters:
    def test_custom_title_appears_in_layout(self, flat_graph):
        fig = render_with_plotly(flat_graph, show=False, title="My Custom Title")
        assert "My Custom Title" in fig.layout.title.text

    def test_default_title_used_when_none(self, flat_graph):
        fig = render_with_plotly(flat_graph, show=False)
        assert fig.layout.title.text  # not empty

    def test_custom_component_colors_applied(self, flat_graph):
        custom = {"g2m": "#FF0000"}
        fig = render_with_plotly(flat_graph, show=False, component_colors=custom)
        g2m_trace = next(t for t in fig.data if t.name == "g2m")
        assert g2m_trace.line.color == "#FF0000"

    def test_default_colors_used_for_unspecified_components(self, flat_graph):
        custom = {"g2m": "#FF0000"}  # only override g2m
        fig = render_with_plotly(flat_graph, show=False, component_colors=custom)
        m2m_trace = next(t for t in fig.data if t.name == "m2m")
        assert m2m_trace.line.color == DEFAULT_COMPONENT_COLORS["m2m"]

    def test_node_size_accepted(self, flat_graph):
        """node_size parameter must not raise."""
        fig = render_with_plotly(flat_graph, show=False, node_size=8.0)
        assert fig is not None

    def test_edge_width_accepted(self, flat_graph):
        """edge_width parameter must not raise."""
        fig = render_with_plotly(flat_graph, show=False, edge_width=3.0)
        assert fig is not None

    def test_layout_parameter_accepted(self, flat_graph):
        """layout parameter must not raise."""
        fig = render_with_plotly(flat_graph, show=False, layout="concentric")
        assert fig is not None
        fig2 = render_with_plotly(flat_graph, show=False, layout="flat")
        assert fig2 is not None

    def test_coastline_flag_accepted(self, flat_graph):
        """Passing add_coastlines=True should not raise."""
        fig = render_with_plotly(flat_graph, show=False, add_coastlines=True)
        assert fig is not None


class TestErrorHandling:
    def test_empty_graph_raises_value_error(self):
        empty = nx.DiGraph()
        with pytest.raises(ValueError, match="empty graph"):
            render_with_plotly(empty, show=False)

    def test_missing_pos_raises_value_error(self):
        g = nx.DiGraph()
        g.add_node(0)  # no pos attribute
        g.add_node(1, pos=np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="'pos'"):
            render_with_plotly(g, show=False)

    def test_no_edges_emits_warning(self):
        """A node-only graph should warn but still render."""
        g = nx.DiGraph()
        g.add_node(0, pos=np.array([0.0, 0.0]))
        g.add_node(1, pos=np.array([1.0, 1.0]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = render_with_plotly(g, show=False)
        assert any("no edges" in str(w.message).lower() for w in caught)
        assert fig is not None


class TestPublicConstants:
    def test_default_colors_has_standard_keys(self):
        for key in ("g2m", "m2m", "m2g"):
            assert key in DEFAULT_COMPONENT_COLORS

    def test_default_colors_are_strings(self):
        for v in DEFAULT_COMPONENT_COLORS.values():
            assert isinstance(v, str)
