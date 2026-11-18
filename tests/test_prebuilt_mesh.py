"""
Tests for mesh_layout="prebuilt" support (Issue #79).

Tests verify:
1. Input validation (nodes-only contract: pos/type/level attributes,
   duplicate positions, edges rejected)
2. Single- and multi-level primitive creation (edge-less node clouds,
   tuple relabelling, dx/dy spacing estimate, level splitting)
3. Directed mesh graph construction from node clouds
   (method="delaunay": Delaunay edges, bidirectional len/vdiff,
   small/degenerate node clouds, pattern-vs-method argument validation)
4. Integration through create_all_graph_components for flat and
   hierarchical connectivity (including the np.ndarray convenience input)
5. Generated-layout behaviour is unchanged (pattern default equivalence)
"""

import networkx as nx
import numpy as np
import pytest

import tests.utils as test_utils
import weather_model_graphs as wmg
from weather_model_graphs.create.mesh.connectivity.general import (
    create_directed_mesh_graph,
)
from weather_model_graphs.create.mesh.connectivity.hierarchical import (
    create_hierarchical_from_coordinates,
)
from weather_model_graphs.create.mesh.layout.prebuilt import (
    create_multi_level_prebuilt_mesh_primitives,
    create_single_level_prebuilt_mesh_primitive,
    validate_prebuilt_mesh_nodes,
)

# ===========================
# Fixtures
# ===========================


@pytest.fixture
def xy_grid():
    """Grid point coordinates covering [0, 10]^2."""
    return test_utils.create_fake_xy(N=10) * 10 / 10


@pytest.fixture
def mesh_xy():
    """Irregular mesh node positions inside the grid domain."""
    rng = np.random.default_rng(seed=7)
    return rng.random((25, 2)) * 10


def _nodes_only_graph(positions, level=None, label=lambda i: i):
    """Build a nodes-only mesh graph from an [N, 2] position array."""
    g = nx.Graph()
    for i, pos in enumerate(positions):
        attrs = dict(pos=np.asarray(pos, dtype=float), type="mesh")
        if level is not None:
            attrs["level"] = level
        g.add_node(label(i), **attrs)
    return g


@pytest.fixture
def mesh_graph(mesh_xy):
    """Nodes-only mesh graph with string labels."""
    return _nodes_only_graph(mesh_xy, label=lambda i: f"station_{i}")


@pytest.fixture
def mesh_graph_two_levels(mesh_xy):
    """Nodes-only mesh graph with two levels (1 = fine, 2 = coarse)."""
    rng = np.random.default_rng(seed=11)
    g = _nodes_only_graph(mesh_xy, level=1, label=lambda i: ("f", i))
    for i, pos in enumerate(rng.random((6, 2)) * 10):
        g.add_node(("c", i), pos=pos, type="mesh", level=2)
    return g


# ===========================
# 1. Input validation
# ===========================


class TestValidatePrebuiltMeshNodes:
    def test_valid_nodes_pass(self, mesh_graph):
        validate_prebuilt_mesh_nodes(mesh_graph)

    def test_empty_graph_raises(self):
        with pytest.raises(ValueError, match="at least one node"):
            validate_prebuilt_mesh_nodes(nx.Graph())

    def test_missing_pos_raises(self):
        g = nx.Graph()
        g.add_node(0, type="mesh")
        with pytest.raises(ValueError, match="missing the required 'pos'"):
            validate_prebuilt_mesh_nodes(g)

    def test_wrong_pos_shape_raises(self):
        g = nx.Graph()
        g.add_node(0, pos=np.array([1.0, 2.0, 3.0]), type="mesh")
        with pytest.raises(ValueError, match="shape"):
            validate_prebuilt_mesh_nodes(g)

    def test_non_finite_pos_raises(self):
        g = nx.Graph()
        g.add_node(0, pos=np.array([np.nan, 0.0]), type="mesh")
        with pytest.raises(ValueError, match="non-finite"):
            validate_prebuilt_mesh_nodes(g)

    def test_missing_type_raises(self):
        g = nx.Graph()
        g.add_node(0, pos=np.array([0.0, 0.0]))
        with pytest.raises(ValueError, match="type"):
            validate_prebuilt_mesh_nodes(g)

    def test_wrong_type_value_raises(self):
        g = nx.Graph()
        g.add_node(0, pos=np.array([0.0, 0.0]), type="grid")
        with pytest.raises(ValueError, match="expected 'mesh'"):
            validate_prebuilt_mesh_nodes(g)

    def test_duplicate_positions_raise(self):
        g = _nodes_only_graph([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="duplicate node positions"):
            validate_prebuilt_mesh_nodes(g)

    def test_edges_not_yet_supported(self, mesh_graph):
        mesh_graph.add_edge("station_0", "station_1")
        with pytest.raises(NotImplementedError, match="nodes-only"):
            validate_prebuilt_mesh_nodes(mesh_graph)

    def test_mixed_level_presence_raises(self, mesh_xy):
        g = _nodes_only_graph(mesh_xy)
        g.nodes[0]["level"] = 1
        with pytest.raises(ValueError, match="'level'"):
            validate_prebuilt_mesh_nodes(g)

    def test_non_integer_level_raises(self, mesh_xy):
        g = _nodes_only_graph(mesh_xy, level=1)
        g.nodes[0]["level"] = "fine"
        with pytest.raises(ValueError, match="non-integer 'level'"):
            validate_prebuilt_mesh_nodes(g)

    def test_require_levels_missing_raises(self, mesh_graph):
        with pytest.raises(ValueError, match="no node has one"):
            validate_prebuilt_mesh_nodes(mesh_graph, require_levels=True)

    def test_require_levels_single_level_raises(self, mesh_xy):
        g = _nodes_only_graph(mesh_xy, level=1)
        with pytest.raises(ValueError, match="two distinct mesh levels"):
            validate_prebuilt_mesh_nodes(g, require_levels=True)


# ===========================
# 2. Primitive creation (coordinate creation step)
# ===========================


class TestSingleLevelPrimitive:
    def test_is_edge_less_node_cloud(self, mesh_graph, mesh_xy):
        g = create_single_level_prebuilt_mesh_primitive(mesh_graph)
        assert g.number_of_nodes() == len(mesh_xy)
        assert g.number_of_edges() == 0

    def test_nodes_relabelled_to_tuples(self, mesh_graph):
        g = create_single_level_prebuilt_mesh_primitive(mesh_graph)
        assert all(isinstance(n, tuple) for n in g.nodes)
        # tuple labels must sort against the (level_id, i) grid node labels
        assert sorted(g.nodes) == list(g.nodes)

    def test_positions_preserved(self, mesh_graph, mesh_xy):
        g = create_single_level_prebuilt_mesh_primitive(mesh_graph)
        positions = np.stack([g.nodes[n]["pos"] for n in g.nodes])
        assert np.allclose(np.sort(positions, axis=0), np.sort(mesh_xy, axis=0))

    def test_spacing_estimate_set(self, mesh_graph):
        g = create_single_level_prebuilt_mesh_primitive(mesh_graph)
        assert g.graph["dx"] > 0
        assert g.graph["dx"] == g.graph["dy"]

    def test_ndarray_input(self, mesh_xy):
        g = create_single_level_prebuilt_mesh_primitive(mesh_xy)
        assert g.number_of_nodes() == len(mesh_xy)
        assert g.number_of_edges() == 0

    def test_bad_ndarray_shape_raises(self):
        with pytest.raises(ValueError, match=r"\[N_mesh_nodes, 2\]"):
            create_single_level_prebuilt_mesh_primitive(np.zeros((3, 4)))

    def test_bad_input_type_raises(self):
        with pytest.raises(TypeError, match="mesh_graph must be"):
            create_single_level_prebuilt_mesh_primitive([[0, 0], [1, 1]])

    def test_edge_less_digraph_accepted(self, mesh_xy):
        dg = nx.DiGraph()
        for i, pos in enumerate(mesh_xy):
            dg.add_node(i, pos=pos, type="mesh")
        g = create_single_level_prebuilt_mesh_primitive(dg)
        assert g.number_of_nodes() == len(mesh_xy)


class TestMultiLevelPrimitives:
    def test_split_by_level_finest_first(self, mesh_graph_two_levels, mesh_xy):
        primitives = create_multi_level_prebuilt_mesh_primitives(mesh_graph_two_levels)
        assert len(primitives) == 2
        # level 1 (finest, 25 nodes) must come first as level index 0
        assert primitives[0].number_of_nodes() == len(mesh_xy)
        assert primitives[0].graph["level"] == 0
        assert primitives[1].number_of_nodes() == 6
        assert primitives[1].graph["level"] == 1

    def test_primitives_are_edge_less(self, mesh_graph_two_levels):
        primitives = create_multi_level_prebuilt_mesh_primitives(mesh_graph_two_levels)
        assert all(g.number_of_edges() == 0 for g in primitives)

    def test_level_values_need_not_be_contiguous(self, mesh_xy):
        g = _nodes_only_graph(mesh_xy[:10], level=3)
        for i, pos in enumerate(mesh_xy[10:15]):
            g.add_node(("coarse", i), pos=pos, type="mesh", level=7)
        primitives = create_multi_level_prebuilt_mesh_primitives(g)
        assert [p.graph["level"] for p in primitives] == [0, 1]
        assert primitives[0].number_of_nodes() == 10

    def test_per_level_spacing_estimates(self, mesh_graph_two_levels):
        primitives = create_multi_level_prebuilt_mesh_primitives(mesh_graph_two_levels)
        assert all(g.graph["dx"] > 0 for g in primitives)


# ===========================
# 3. Directed mesh graph from node clouds (connectivity step)
# ===========================


class TestNodeCloudDirectedGraph:
    def test_delaunay_bidirectional_len_vdiff(self, mesh_graph):
        g_prim = create_single_level_prebuilt_mesh_primitive(mesh_graph)
        dg = create_directed_mesh_graph(g_prim)
        assert isinstance(dg, nx.DiGraph)
        assert dg.number_of_edges() > 0
        for u, v, d in dg.edges(data=True):
            assert dg.has_edge(v, u)
            assert d["len"] > 0
            assert np.allclose(d["vdiff"], -dg.edges[v, u]["vdiff"])
            assert np.isclose(d["len"], np.linalg.norm(d["vdiff"]))

    def test_delaunay_edges_match_scipy(self, mesh_xy):
        import scipy.spatial

        g_prim = create_single_level_prebuilt_mesh_primitive(mesh_xy)
        dg = create_directed_mesh_graph(g_prim)
        tri = scipy.spatial.Delaunay(mesh_xy)
        expected_pairs = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = sorted((simplex[i], simplex[(i + 1) % 3]))
                expected_pairs.add((a, b))
        assert dg.number_of_edges() == 2 * len(expected_pairs)

    def test_single_node_no_edges(self):
        g_prim = create_single_level_prebuilt_mesh_primitive(np.array([[1.0, 2.0]]))
        dg = create_directed_mesh_graph(g_prim)
        assert dg.number_of_nodes() == 1
        assert dg.number_of_edges() == 0

    def test_two_nodes_bidirectional_pair(self):
        g_prim = create_single_level_prebuilt_mesh_primitive(
            np.array([[0.0, 0.0], [3.0, 4.0]])
        )
        dg = create_directed_mesh_graph(g_prim)
        assert dg.number_of_edges() == 2
        (d,) = [d for _, _, d in dg.edges(data=True) if d["vdiff"][0] < 0]
        assert np.isclose(d["len"], 5.0)

    def test_three_nodes_triangle(self):
        g_prim = create_single_level_prebuilt_mesh_primitive(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )
        dg = create_directed_mesh_graph(g_prim)
        assert dg.number_of_edges() == 6

    def test_collinear_nodes_raise(self):
        g_prim = create_single_level_prebuilt_mesh_primitive(
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        )
        with pytest.raises(ValueError, match="collinear"):
            create_directed_mesh_graph(g_prim)

    def test_pattern_on_node_cloud_raises(self, mesh_graph):
        g_prim = create_single_level_prebuilt_mesh_primitive(mesh_graph)
        with pytest.raises(ValueError, match="method='delaunay'"):
            create_directed_mesh_graph(g_prim, pattern="8-star")

    def test_unknown_method_raises(self, mesh_graph):
        g_prim = create_single_level_prebuilt_mesh_primitive(mesh_graph)
        with pytest.raises(NotImplementedError, match="'delaunay'"):
            create_directed_mesh_graph(g_prim, method="knn")

    def test_method_on_lattice_primitive_raises(self, xy_grid):
        from weather_model_graphs.create.mesh.layout.rectilinear import (
            create_single_level_2d_mesh_primitive,
        )

        g_prim = create_single_level_2d_mesh_primitive(xy_grid, nx=4, ny=4)
        with pytest.raises(ValueError, match="already"):
            create_directed_mesh_graph(g_prim, method="delaunay")


# ===========================
# 4. Integration through create_all_graph_components
# ===========================


class TestFlatEndToEnd:
    def test_flat_components(self, xy_grid, mesh_graph, mesh_xy):
        components = wmg.create.create_all_graph_components(
            coords=xy_grid,
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh_graph),
            m2m_connectivity="flat",
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        assert set(components.keys()) == {"m2m", "g2m", "m2g"}
        m2m = components["m2m"]
        mesh_nodes = [n for n, d in m2m.nodes(data=True) if d.get("type") == "mesh"]
        assert len(mesh_nodes) == len(mesh_xy)
        assert m2m.number_of_edges() > 0
        assert components["g2m"].number_of_edges() == len(mesh_xy)

    def test_explicit_delaunay_method_matches_default(self, xy_grid, mesh_graph):
        kwargs = dict(
            coords=xy_grid,
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh_graph),
            m2m_connectivity="flat",
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        default = wmg.create.create_all_graph_components(**kwargs)
        explicit = wmg.create.create_all_graph_components(
            m2m_connectivity_kwargs=dict(method="delaunay"), **kwargs
        )
        assert default["m2m"].number_of_edges() == explicit["m2m"].number_of_edges()

    def test_merged_single_graph(self, xy_grid, mesh_xy):
        graph = wmg.create.create_all_graph_components(
            coords=xy_grid,
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh_xy),
            m2m_connectivity="flat",
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
        )
        assert graph.number_of_nodes() == len(xy_grid) + len(mesh_xy)

    def test_missing_mesh_graph_raises(self, xy_grid):
        with pytest.raises(ValueError, match="mesh_graph"):
            wmg.create.create_all_graph_components(
                coords=xy_grid,
                mesh_layout="prebuilt",
                m2m_connectivity="flat",
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )

    def test_flat_multiscale_not_supported(self, xy_grid, mesh_graph):
        with pytest.raises(NotImplementedError, match="flat_multiscale"):
            wmg.create.create_all_graph_components(
                coords=xy_grid,
                mesh_layout="prebuilt",
                mesh_layout_kwargs=dict(mesh_graph=mesh_graph),
                m2m_connectivity="flat_multiscale",
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )


class TestHierarchicalEndToEnd:
    def test_hierarchical_components(self, xy_grid, mesh_graph_two_levels, mesh_xy):
        components = wmg.create.create_all_graph_components(
            coords=xy_grid,
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh_graph_two_levels),
            m2m_connectivity="hierarchical",
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        m2m = components["m2m"]
        directions = {
            d["direction"] for _, _, d in m2m.edges(data=True) if "direction" in d
        }
        assert directions == {"same", "up", "down"}
        n_up = sum(1 for _, _, d in m2m.edges(data=True) if d.get("direction") == "up")
        # nearest with k=1: one up edge per fine node
        assert n_up == len(mesh_xy)
        # the grid connects only to the finest level
        assert components["g2m"].number_of_edges() == len(mesh_xy)

    def test_intra_level_method_kwarg(self, xy_grid, mesh_graph_two_levels):
        components = wmg.create.create_all_graph_components(
            coords=xy_grid,
            mesh_layout="prebuilt",
            mesh_layout_kwargs=dict(mesh_graph=mesh_graph_two_levels),
            m2m_connectivity="hierarchical",
            m2m_connectivity_kwargs=dict(
                intra_level=dict(method="delaunay"),
                inter_level=dict(pattern="nearest", k=2),
            ),
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        n_up = sum(
            1
            for _, _, d in components["m2m"].edges(data=True)
            if d.get("direction") == "up"
        )
        # k=2: two up edges per fine node
        assert n_up == 2 * mesh_graph_two_levels.number_of_nodes() - 2 * 6

    def test_hierarchical_without_levels_raises(self, xy_grid, mesh_graph):
        with pytest.raises(ValueError, match="level"):
            wmg.create.create_all_graph_components(
                coords=xy_grid,
                mesh_layout="prebuilt",
                mesh_layout_kwargs=dict(mesh_graph=mesh_graph),
                m2m_connectivity="hierarchical",
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
            )

    def test_direct_hierarchical_from_prebuilt_primitives(self, mesh_graph_two_levels):
        primitives = create_multi_level_prebuilt_mesh_primitives(mesh_graph_two_levels)
        m2m = create_hierarchical_from_coordinates(primitives)
        assert m2m.number_of_edges() > 0
        assert set(m2m.graph["dx"].keys()) == {0, 1}


# ===========================
# 5. Generated layouts unchanged (pattern default equivalence)
# ===========================


class TestGeneratedLayoutsUnchanged:
    @pytest.mark.parametrize("mesh_layout", ["rectilinear", "triangular"])
    def test_no_pattern_equals_8_star(self, xy_grid, mesh_layout):
        kwargs = dict(
            coords=xy_grid,
            mesh_layout=mesh_layout,
            mesh_layout_kwargs=dict(mesh_node_spacing=2),
            m2m_connectivity="flat",
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            return_components=True,
        )
        default = wmg.create.create_all_graph_components(**kwargs)
        explicit = wmg.create.create_all_graph_components(
            m2m_connectivity_kwargs=dict(pattern="8-star"), **kwargs
        )
        assert default["m2m"].number_of_edges() == explicit["m2m"].number_of_edges()
