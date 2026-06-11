import tempfile
from pathlib import Path

import pytest
import torch
from loguru import logger

import tests.utils as test_utils
import weather_model_graphs as wmg
from weather_model_graphs.save import HAS_PYG


@pytest.mark.parametrize("list_from_attribute", [None, "level"])
def test_save_to_pyg(list_from_attribute):
    if not HAS_PYG:
        logger.warning(
            "Skipping test_save_to_pyg because weather-model-graphs[pytorch] is not installed."
        )
        return

    xy = test_utils.create_fake_xy(N=64)
    graph = wmg.create.archetype.create_oskarsson_hierarchical_graph(coords=xy)

    graph_components = wmg.split_graph_by_edge_attribute(graph=graph, attr="component")

    # split the m2m graph into the different parts that create the up, in-level and down connections respectively
    # this is how the graphs is stored in the neural-lam codebase
    m2m_graph = graph_components.pop("m2m")
    m2m_graph_components = wmg.split_graph_by_edge_attribute(
        graph=m2m_graph, attr="direction"
    )
    m2m_graph_components = {
        f"m2m_{name}": graph for name, graph in m2m_graph_components.items()
    }
    graph_components.update(m2m_graph_components)

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, graph in graph_components.items():
            wmg.save.to_pyg(
                graph=graph,
                output_directory=tmpdir,
                name=name,
                list_from_attribute=list_from_attribute,
            )


# ─── to_torch_tensors_on_disk tests ───

# Files expected for all graph types
CORE_FILES = [
    "g2m_edge_index.pt",
    "g2m_features.pt",
    "m2g_edge_index.pt",
    "m2g_features.pt",
    "m2m_edge_index.pt",
    "m2m_features.pt",
    "mesh_features.pt",
    "metainfo.yaml",
]

# Additional files for hierarchical graphs
HIERARCHICAL_FILES = [
    "mesh_up_edge_index.pt",
    "mesh_up_features.pt",
    "mesh_down_edge_index.pt",
    "mesh_down_features.pt",
]


def _skip_if_no_pyg():
    if not HAS_PYG:
        pytest.skip("weather-model-graphs[pytorch] not installed")


def _create_and_save(archetype, hierarchical, N=64):
    """Helper: create graph components and save to temp dir, return path."""
    xy = test_utils.create_fake_xy(N=N)

    if archetype == "keisler":
        components = wmg.create.archetype.create_keisler_graph(
            coords=xy, return_components=True
        )
    elif archetype == "graphcast":
        components = wmg.create.archetype.create_graphcast_graph(
            coords=xy, return_components=True
        )
    elif archetype == "hierarchical":
        components = wmg.create.archetype.create_oskarsson_hierarchical_graph(
            coords=xy, return_components=True
        )
    else:
        raise ValueError(f"Unknown archetype: {archetype}")

    tmpdir = tempfile.mkdtemp()
    wmg.save.to_torch_tensors_on_disk(
        graph_components=components,
        output_directory=tmpdir,
        hierarchical=hierarchical,
    )
    return tmpdir, components


class TestToTorchTensorsKeisler:
    """Tests for to_torch_tensors_on_disk with keisler (flat single-scale) archetype."""

    def test_core_files_created(self):
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("keisler", hierarchical=False)
        for fname in CORE_FILES:
            assert (Path(tmpdir) / fname).exists(), f"Missing: {fname}"
        # Hierarchical files should NOT exist
        for fname in HIERARCHICAL_FILES:
            assert not (Path(tmpdir) / fname).exists(), f"Unexpected: {fname}"

    def test_g2m_m2g_are_single_tensors(self):
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("keisler", hierarchical=False)
        g2m_ei = torch.load(Path(tmpdir) / "g2m_edge_index.pt", weights_only=True)
        g2m_f = torch.load(Path(tmpdir) / "g2m_features.pt", weights_only=True)
        m2g_ei = torch.load(Path(tmpdir) / "m2g_edge_index.pt", weights_only=True)
        m2g_f = torch.load(Path(tmpdir) / "m2g_features.pt", weights_only=True)

        # Must be plain tensors, not lists
        assert isinstance(g2m_ei, torch.Tensor)
        assert isinstance(g2m_f, torch.Tensor)
        assert isinstance(m2g_ei, torch.Tensor)
        assert isinstance(m2g_f, torch.Tensor)

        # Shape checks
        assert g2m_ei.shape[0] == 2
        assert g2m_f.shape[1] == 3
        assert m2g_ei.shape[0] == 2
        assert m2g_f.shape[1] == 3

        # Edge count consistency
        assert g2m_ei.shape[1] == g2m_f.shape[0]
        assert m2g_ei.shape[1] == m2g_f.shape[0]

    def test_m2m_is_list_of_one(self):
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("keisler", hierarchical=False)
        m2m_ei = torch.load(Path(tmpdir) / "m2m_edge_index.pt", weights_only=True)
        m2m_f = torch.load(Path(tmpdir) / "m2m_features.pt", weights_only=True)
        mesh_f = torch.load(Path(tmpdir) / "mesh_features.pt", weights_only=True)

        # Keisler is single-level, so lists of length 1
        assert isinstance(m2m_ei, list)
        assert len(m2m_ei) == 1
        assert isinstance(m2m_f, list)
        assert len(m2m_f) == 1
        assert isinstance(mesh_f, list)
        assert len(mesh_f) == 1

        # Shape checks for tensors inside lists
        assert m2m_ei[0].shape[0] == 2
        assert m2m_f[0].shape[1] == 3
        assert mesh_f[0].shape[1] == 2

        # Edge count consistency
        assert m2m_ei[0].shape[1] == m2m_f[0].shape[0]

    def test_mesh_features_are_raw_coordinates(self):
        """Mesh node features must be raw (unnormalized) positions, matching
        the mesh node positions in the m2m component (graph storage spec:
        normalization happens inside neural-lam at load time)."""
        _skip_if_no_pyg()
        tmpdir, components = _create_and_save("keisler", hierarchical=False)
        mesh_f = torch.load(Path(tmpdir) / "mesh_features.pt", weights_only=True)

        m2m_graph = components["m2m"]
        mesh_labels = sorted(
            n for n, d in m2m_graph.nodes(data=True) if d.get("type") == "mesh"
        )
        expected = torch.tensor(
            [m2m_graph.nodes[n]["pos"] for n in mesh_labels], dtype=torch.float32
        )
        assert len(mesh_f) == 1
        assert torch.allclose(mesh_f[0], expected)

    def test_metainfo_yaml_has_spec_version(self):
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("keisler", hierarchical=False)
        content = (Path(tmpdir) / "metainfo.yaml").read_text()
        assert "spec_version" in content
        assert "0.1.0" in content

    def test_edge_indices_are_zero_based_per_node_set(self):
        """Edge indices must use per-node-set zero-based numbering (graph
        storage spec 3.1.1): g2m receivers and m2g senders are indices into
        the mesh node set, not offset by the number of grid nodes."""
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("keisler", hierarchical=False)
        g2m_ei = torch.load(Path(tmpdir) / "g2m_edge_index.pt", weights_only=True)
        m2g_ei = torch.load(Path(tmpdir) / "m2g_edge_index.pt", weights_only=True)
        m2m_ei = torch.load(Path(tmpdir) / "m2m_edge_index.pt", weights_only=True)
        mesh_f = torch.load(Path(tmpdir) / "mesh_features.pt", weights_only=True)

        n_mesh = mesh_f[0].shape[0]
        n_grid = int(g2m_ei[0].max()) + 1  # senders cover the grid node set

        # g2m: senders are grid indices, receivers are mesh indices
        assert int(g2m_ei[1].max()) < n_mesh
        assert int(g2m_ei[1].min()) >= 0
        # m2g: senders are mesh indices, receivers are grid indices
        assert int(m2g_ei[0].max()) < n_mesh
        assert int(m2g_ei[0].min()) >= 0
        assert int(m2g_ei[1].max()) < n_grid
        # m2m: both within mesh node set
        assert int(m2m_ei[0].max()) < n_mesh

    def test_edge_features_are_raw(self):
        """Edge features should NOT be normalized (neural-lam normalizes at load time)."""
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("keisler", hierarchical=False)
        m2m_f = torch.load(Path(tmpdir) / "m2m_features.pt", weights_only=True)
        # Column 0 is edge length — should be > 0 for all edges
        for level_f in m2m_f:
            assert torch.all(level_f[:, 0] > 0)

    def test_has_nonzero_edges(self):
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("keisler", hierarchical=False)
        g2m_ei = torch.load(Path(tmpdir) / "g2m_edge_index.pt", weights_only=True)
        m2g_ei = torch.load(Path(tmpdir) / "m2g_edge_index.pt", weights_only=True)
        m2m_ei = torch.load(Path(tmpdir) / "m2m_edge_index.pt", weights_only=True)
        assert g2m_ei.shape[1] > 0
        assert m2g_ei.shape[1] > 0
        assert m2m_ei[0].shape[1] > 0


class TestToTorchTensorsGraphcast:
    """Tests for to_torch_tensors_on_disk with graphcast (flat multiscale) archetype."""

    def test_core_files_created(self):
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("graphcast", hierarchical=False)
        for fname in CORE_FILES:
            assert (Path(tmpdir) / fname).exists(), f"Missing: {fname}"
        for fname in HIERARCHICAL_FILES:
            assert not (Path(tmpdir) / fname).exists(), f"Unexpected: {fname}"

    def test_m2m_is_single_level(self):
        """Non-hierarchical graphs always have L == 1 per the graph storage
        spec, so flat multiscale (graphcast) m2m edges are merged into a
        single entry over the merged mesh node set."""
        _skip_if_no_pyg()
        tmpdir, components = _create_and_save("graphcast", hierarchical=False, N=64)
        m2m_ei = torch.load(Path(tmpdir) / "m2m_edge_index.pt", weights_only=True)
        m2m_f = torch.load(Path(tmpdir) / "m2m_features.pt", weights_only=True)
        mesh_f = torch.load(Path(tmpdir) / "mesh_features.pt", weights_only=True)

        assert isinstance(m2m_ei, list)
        assert isinstance(m2m_f, list)
        assert isinstance(mesh_f, list)

        # Non-hierarchical: single level
        assert len(m2m_ei) == len(m2m_f) == len(mesh_f) == 1

        # The single entry contains ALL m2m edges over the merged node set
        assert m2m_ei[0].shape[1] == components["m2m"].number_of_edges()
        assert mesh_f[0].shape[0] == components["m2m"].number_of_nodes()

        # Shape checks
        assert m2m_ei[0].shape[0] == 2
        assert m2m_f[0].shape[1] == 3
        assert mesh_f[0].shape[1] == 2
        assert m2m_ei[0].shape[1] == m2m_f[0].shape[0]

        # All indices within the merged mesh node set
        assert int(m2m_ei[0].max()) < mesh_f[0].shape[0]

    def test_g2m_m2g_single_tensors(self):
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("graphcast", hierarchical=False)
        g2m_ei = torch.load(Path(tmpdir) / "g2m_edge_index.pt", weights_only=True)
        m2g_ei = torch.load(Path(tmpdir) / "m2g_edge_index.pt", weights_only=True)
        assert isinstance(g2m_ei, torch.Tensor)
        assert isinstance(m2g_ei, torch.Tensor)
        assert g2m_ei.shape[0] == 2
        assert m2g_ei.shape[0] == 2


class TestToTorchTensorsHierarchical:
    """Tests for to_torch_tensors_on_disk with oskarsson hierarchical archetype."""

    def test_all_files_created(self):
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("hierarchical", hierarchical=True)
        for fname in CORE_FILES + HIERARCHICAL_FILES:
            assert (Path(tmpdir) / fname).exists(), f"Missing: {fname}"

    def test_m2m_shapes(self):
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("hierarchical", hierarchical=True)
        m2m_ei = torch.load(Path(tmpdir) / "m2m_edge_index.pt", weights_only=True)
        m2m_f = torch.load(Path(tmpdir) / "m2m_features.pt", weights_only=True)
        mesh_f = torch.load(Path(tmpdir) / "mesh_features.pt", weights_only=True)

        n_levels = len(m2m_ei)
        assert n_levels > 1, "Hierarchical graph should have multiple levels"
        assert len(m2m_f) == n_levels
        assert len(mesh_f) == n_levels

        for i in range(n_levels):
            assert m2m_ei[i].shape[0] == 2
            assert m2m_f[i].shape[1] == 3
            assert mesh_f[i].shape[1] == 2
            assert m2m_ei[i].shape[1] == m2m_f[i].shape[0]

    def test_up_down_shapes(self):
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("hierarchical", hierarchical=True)
        m2m_ei = torch.load(Path(tmpdir) / "m2m_edge_index.pt", weights_only=True)
        up_ei = torch.load(Path(tmpdir) / "mesh_up_edge_index.pt", weights_only=True)
        up_f = torch.load(Path(tmpdir) / "mesh_up_features.pt", weights_only=True)
        down_ei = torch.load(
            Path(tmpdir) / "mesh_down_edge_index.pt", weights_only=True
        )
        down_f = torch.load(Path(tmpdir) / "mesh_down_features.pt", weights_only=True)

        n_levels = len(m2m_ei)
        # Up/down should have n_levels - 1 entries
        assert len(up_ei) == n_levels - 1
        assert len(up_f) == n_levels - 1
        assert len(down_ei) == n_levels - 1
        assert len(down_f) == n_levels - 1

        for i in range(n_levels - 1):
            assert up_ei[i].shape[0] == 2
            assert up_f[i].shape[1] == 3
            assert down_ei[i].shape[0] == 2
            assert down_f[i].shape[1] == 3
            assert up_ei[i].shape[1] == up_f[i].shape[0]
            assert down_ei[i].shape[1] == down_f[i].shape[0]

    def test_up_down_have_nonzero_edges(self):
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("hierarchical", hierarchical=True)
        up_ei = torch.load(Path(tmpdir) / "mesh_up_edge_index.pt", weights_only=True)
        down_ei = torch.load(
            Path(tmpdir) / "mesh_down_edge_index.pt", weights_only=True
        )
        for i, (u, d) in enumerate(zip(up_ei, down_ei)):
            assert u.shape[1] > 0, f"up level {i} has no edges"
            assert d.shape[1] > 0, f"down level {i} has no edges"

    def test_mesh_features_are_raw_coordinates(self):
        """Mesh node features must be raw (unnormalized) positions per level,
        matching the m2m component node positions (graph storage spec)."""
        _skip_if_no_pyg()
        tmpdir, components = _create_and_save("hierarchical", hierarchical=True)
        mesh_f = torch.load(Path(tmpdir) / "mesh_features.pt", weights_only=True)

        m2m_graph = components["m2m"]
        labels_by_level = {}
        for n, d in m2m_graph.nodes(data=True):
            labels_by_level.setdefault(d["level"], []).append(n)

        assert len(mesh_f) == len(labels_by_level)
        for i, lvl in enumerate(sorted(labels_by_level.keys())):
            expected = torch.tensor(
                [m2m_graph.nodes[n]["pos"] for n in sorted(labels_by_level[lvl])],
                dtype=torch.float32,
            )
            assert torch.allclose(mesh_f[i], expected)

    def test_interlevel_indices_are_zero_based_per_level(self):
        """mesh_up entry i goes level i -> i+1, mesh_down entry i goes
        level i+1 -> i, with indices zero-based within each level's node
        set (graph storage spec 3.1.1 / 3.2.1)."""
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("hierarchical", hierarchical=True)
        mesh_f = torch.load(Path(tmpdir) / "mesh_features.pt", weights_only=True)
        up_ei = torch.load(Path(tmpdir) / "mesh_up_edge_index.pt", weights_only=True)
        down_ei = torch.load(
            Path(tmpdir) / "mesh_down_edge_index.pt", weights_only=True
        )
        g2m_ei = torch.load(Path(tmpdir) / "g2m_edge_index.pt", weights_only=True)
        m2g_ei = torch.load(Path(tmpdir) / "m2g_edge_index.pt", weights_only=True)

        n_per_level = [f.shape[0] for f in mesh_f]

        # g2m receivers / m2g senders index into the bottom mesh level
        assert int(g2m_ei[1].max()) < n_per_level[0]
        assert int(m2g_ei[0].max()) < n_per_level[0]

        for i in range(len(n_per_level) - 1):
            # up: sender level i, receiver level i+1
            assert int(up_ei[i][0].max()) < n_per_level[i]
            assert int(up_ei[i][1].max()) < n_per_level[i + 1]
            # down: sender level i+1, receiver level i
            assert int(down_ei[i][0].max()) < n_per_level[i + 1]
            assert int(down_ei[i][1].max()) < n_per_level[i]


class TestToTorchTensorsEdgeCases:
    """Edge case and validation tests."""

    def test_missing_component_raises(self):
        _skip_if_no_pyg()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="missing required keys"):
                wmg.save.to_torch_tensors_on_disk(
                    graph_components={"g2m": None, "m2m": None},
                    output_directory=tmpdir,
                )

    def test_empty_components_dict_raises(self):
        _skip_if_no_pyg()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="missing required keys"):
                wmg.save.to_torch_tensors_on_disk(
                    graph_components={},
                    output_directory=tmpdir,
                )

    def test_output_dir_created_if_missing(self):
        _skip_if_no_pyg()
        xy = test_utils.create_fake_xy(N=64)
        components = wmg.create.archetype.create_keisler_graph(
            coords=xy, return_components=True
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "deeply" / "nested" / "dir"
            assert not nested_dir.exists()
            wmg.save.to_torch_tensors_on_disk(
                graph_components=components,
                output_directory=str(nested_dir),
                hierarchical=False,
            )
            assert nested_dir.exists()
            for fname in CORE_FILES:
                assert (nested_dir / fname).exists()

    def test_edge_features_have_positive_lengths(self):
        """Column 0 of edge features should be edge length > 0 for
        intra-component edges. Inter-level (up/down) edges may have
        zero-length when coarser nodes coincide with finer nodes."""
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("hierarchical", hierarchical=True)

        # Intra-component edges: strictly positive lengths
        for fname in [
            "g2m_features.pt",
            "m2g_features.pt",
            "m2m_features.pt",
        ]:
            data = torch.load(Path(tmpdir) / fname, weights_only=True)
            if isinstance(data, list):
                for level_f in data:
                    assert torch.all(
                        level_f[:, 0] > 0
                    ), f"{fname} has non-positive edge lengths"
            else:
                assert torch.all(
                    data[:, 0] > 0
                ), f"{fname} has non-positive edge lengths"

        # Inter-level edges: allow zero-length (coincident nodes)
        for fname in [
            "mesh_up_features.pt",
            "mesh_down_features.pt",
        ]:
            data = torch.load(Path(tmpdir) / fname, weights_only=True)
            for level_f in data:
                assert torch.all(
                    level_f[:, 0] >= 0
                ), f"{fname} has negative edge lengths"

    def test_all_tensors_are_float32(self):
        """All feature tensors should be float32."""
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("keisler", hierarchical=False)
        for fname in [
            "g2m_features.pt",
            "m2g_features.pt",
            "m2m_features.pt",
            "mesh_features.pt",
        ]:
            data = torch.load(Path(tmpdir) / fname, weights_only=True)
            if isinstance(data, list):
                for t in data:
                    assert t.dtype == torch.float32, f"{fname} not float32"
            else:
                assert data.dtype == torch.float32, f"{fname} not float32"

    def test_edge_index_dtype_is_int64(self):
        """Edge index tensors should be int64 (standard PyG format)."""
        _skip_if_no_pyg()
        tmpdir, _ = _create_and_save("keisler", hierarchical=False)
        for fname in ["g2m_edge_index.pt", "m2g_edge_index.pt", "m2m_edge_index.pt"]:
            data = torch.load(Path(tmpdir) / fname, weights_only=True)
            if isinstance(data, list):
                for t in data:
                    assert t.dtype == torch.int64, f"{fname} not int64"
            else:
                assert data.dtype == torch.int64, f"{fname} not int64"

    def test_rectangular_grid(self):
        """Test with non-square grid coordinates."""
        _skip_if_no_pyg()
        xy = test_utils.create_rectangular_fake_xy(Nx=40, Ny=80)
        components = wmg.create.archetype.create_keisler_graph(
            coords=xy, return_components=True
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            wmg.save.to_torch_tensors_on_disk(
                graph_components=components,
                output_directory=tmpdir,
                hierarchical=False,
            )
            for fname in CORE_FILES:
                assert (Path(tmpdir) / fname).exists(), f"Missing: {fname}"

    def test_overwrite_existing_files(self):
        """Saving twice to same dir should overwrite without error."""
        _skip_if_no_pyg()
        xy = test_utils.create_fake_xy(N=64)
        components = wmg.create.archetype.create_keisler_graph(
            coords=xy, return_components=True
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            wmg.save.to_torch_tensors_on_disk(
                graph_components=components,
                output_directory=tmpdir,
                hierarchical=False,
            )
            # Save again — should not raise
            wmg.save.to_torch_tensors_on_disk(
                graph_components=components,
                output_directory=tmpdir,
                hierarchical=False,
            )
            for fname in CORE_FILES:
                assert (Path(tmpdir) / fname).exists()
