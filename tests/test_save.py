import importlib.util
import sys
import tempfile
import urllib.request
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


# ─── to_torch_tensors_on_disk: validated against neural-lam's own validator ───
#
# The neural-lam graph storage spec is defined and enforced by a single
# standalone validator script (``docs/validate_graph.py``) that lives in
# neural-lam. Rather than re-implement those spec checks here (where they
# could silently drift from the real contract), these tests download that
# exact script and run it against the output of ``to_torch_tensors_on_disk``.
#
# The graph storage spec + validator were merged into neural-lam via
# mllam/neural-lam#323, so we pull the validator from ``main``.
VALIDATOR_URL = (
    "https://raw.githubusercontent.com/mllam/neural-lam/main/docs/validate_graph.py"
)

# Files expected for all graph types (used by the function-behaviour tests
# below, which the validator cannot exercise — see their docstrings).
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


@pytest.fixture(scope="session")
def graph_validator():
    """Download neural-lam's ``validate_graph.py`` and load it as a module.

    The validator is the single source of truth for the on-disk graph format,
    so we run WMG's output through the *actual* neural-lam script (loaded
    inline via ``importlib``, the same pattern as neural-lam's
    ``tests/test_validate_graph_script.py``) instead of re-implementing the
    spec checks here.

    A download failure is a hard error rather than a skip: if the validator
    cannot be fetched we must not let the suite go green, otherwise a missing
    validator could be mistaken for a passing test.
    """
    dest = Path(tempfile.mkdtemp(prefix="nl-validator-")) / "validate_graph.py"
    try:
        with urllib.request.urlopen(VALIDATOR_URL, timeout=60) as response:
            dest.write_bytes(response.read())
    except Exception as exc:  # noqa: BLE001 — any failure must fail the test
        raise RuntimeError(
            f"Could not download the neural-lam graph validator from "
            f"{VALIDATOR_URL!r}: {exc}. The graph-format validation tests "
            f"cannot run without it."
        ) from exc

    spec = importlib.util.spec_from_file_location("validate_graph_script", dest)
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_graph_script"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def _fail_details(report):
    """Collect the detail strings of any failing checks for assert messages."""
    return "\n".join(r.detail for r in report.results if r.status == "FAIL")


# ─── Validator-backed spec conformance ───
# These replace all the hand-rolled spec assertions (file presence, shapes,
# int64/float32 dtypes, per-node-set zero-based indexing, index ranges, finite
# values, metainfo/spec_version): every one of those is checked by the
# neural-lam validator, so they cannot diverge from the real contract.


@pytest.mark.parametrize(
    "archetype, hierarchical",
    [
        ("keisler", False),  # flat single-scale
        ("graphcast", False),  # flat multiscale
        ("hierarchical", True),  # multi-level hierarchical
    ],
)
def test_output_passes_neural_lam_validator(graph_validator, archetype, hierarchical):
    """A saved graph must pass neural-lam's own on-disk format validator.

    Covers keisler (flat), graphcast (flat multiscale) and oskarsson
    (hierarchical) archetypes — the validator checks required files, tensor
    shapes, int64 edge-index / float32 feature dtypes, per-node-set index
    ranges, finite values and the metainfo spec version.
    """
    _skip_if_no_pyg()
    tmpdir, _ = _create_and_save(archetype, hierarchical=hierarchical)
    report, _spec, _props = graph_validator.validate_graph_directory(tmpdir)
    assert not report.has_fails(), _fail_details(report)


def test_rectangular_grid_passes_validator(graph_validator):
    """Non-square grids must also produce a spec-conformant graph.

    Guards against assumptions that the grid is square in the index/shape
    bookkeeping; validated end-to-end with the neural-lam validator.
    """
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
        report, _spec, _props = graph_validator.validate_graph_directory(tmpdir)
        assert not report.has_fails(), _fail_details(report)


# ─── Tests the validator cannot cover ───
# Kept deliberately because the neural-lam validator only inspects an
# already-written directory and has no reference to the source graph. The
# docstrings record *why* each one is not (and cannot be) covered there.


def test_missing_component_raises():
    """The validator cannot test input rejection.

    ``to_torch_tensors_on_disk`` must refuse incomplete ``graph_components``
    up front, before anything is written. There is no output directory for
    the validator to inspect in this case, so this contract is WMG-side only.
    """
    _skip_if_no_pyg()
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="missing required keys"):
            wmg.save.to_torch_tensors_on_disk(
                graph_components={"g2m": None, "m2m": None},
                output_directory=tmpdir,
            )


def test_empty_components_dict_raises():
    """The validator cannot test input rejection.

    An empty ``graph_components`` mapping must raise rather than write a
    partial/empty directory. Like ``test_missing_component_raises``, there is
    no artifact for the validator to look at, so this is WMG-side only.
    """
    _skip_if_no_pyg()
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="missing required keys"):
            wmg.save.to_torch_tensors_on_disk(
                graph_components={},
                output_directory=tmpdir,
            )


def test_output_dir_created_if_missing():
    """The validator cannot test filesystem side-effects.

    ``to_torch_tensors_on_disk`` must create a missing (nested) output
    directory itself. The validator only reads an existing directory, so it
    can never verify that the writer created one — this is WMG-side only.
    """
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


def test_overwrite_existing_files():
    """The validator cannot test write behaviour across repeated calls.

    Saving twice into the same directory must overwrite cleanly without
    error. The validator only inspects the final state of a directory, so it
    cannot exercise idempotent re-saving — this is WMG-side only.
    """
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


def test_mesh_features_are_raw_coordinates_keisler():
    """The validator cannot check feature *values* against the source graph.

    It confirms ``mesh_features`` are float32/finite/2-column, but has no
    reference to the originating graph, so it cannot tell whether the written
    values are the correct raw (unnormalized) node positions or were
    accidentally normalized. That semantic correctness — the bug this format
    work fixed — is verifiable only WMG-side, where the source graph exists.
    """
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


def test_mesh_features_are_raw_coordinates_hierarchical():
    """The validator cannot check per-level feature *values* against the source.

    As with the keisler case, the validator confirms shape/dtype but cannot
    verify that each level's ``mesh_features`` are the correct raw node
    positions (it has no source graph to compare against). This checks every
    hierarchical level, so it is WMG-side only.
    """
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
