import urllib.request
import tempfile
from pathlib import Path
import importlib.util

import pytest

import weather_model_graphs as wmg
import tests.utils as test_utils


VALIDATOR_URL = (
    "https://raw.githubusercontent.com/mllam/neural-lam/"
    "feat/graph-on-disk-spec-and-validator/docs/validate_graph.py"
)


def _download_validator(tmpdir):
    """Download the neural-lam validator script."""
    validator_path = Path(tmpdir) / "validate_graph.py"

    try:
        urllib.request.urlretrieve(VALIDATOR_URL, validator_path)
    except Exception:
        pytest.skip("Could not download neural-lam validator script")

    return validator_path


def _load_validator_module(script_path):
    """Dynamically load the validator module."""
    spec = importlib.util.spec_from_file_location("validator", script_path)
    module = importlib.util.module_from_spec(spec)

    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module


def test_saved_graph_passes_neural_lam_validator():
    """Ensure graphs saved with save.to_pyg() follow neural-lam disk specification."""

    xy = test_utils.create_fake_xy(N=64)

    graph = wmg.create.archetype.create_oskarsson_hierarchical_graph(
        coords=xy
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_dir = Path(tmpdir) / "graph"
        graph_dir.mkdir()

        # Split graph into components
        graph_components = wmg.split_graph_by_edge_attribute(graph, attr="component")
        graph = list(graph_components.values())[0]

        # Find common edge attributes
        edge_attrs_sets = [set(d.keys()) for _, _, d in graph.edges(data=True)]
        common_attrs = set.intersection(*edge_attrs_sets) if edge_attrs_sets else set()
        edge_features = [f for f in common_attrs if f != "component"]

        # Save graph
        wmg.save.to_pyg(
            graph=graph,
            output_directory=str(graph_dir),
            name="test_graph",
            edge_features=edge_features,
        )

        # Ensure files were created
        written_files = list(graph_dir.rglob("*"))
        assert written_files, "save.to_pyg() did not create any files"

        # Download validator
        validator_script = _download_validator(tmpdir)

        # Load validator module
        validator = _load_validator_module(validator_script)

        # Run validation
        report = validator.validate_graph_directory(graph_dir)

        assert report.ok, f"Graph validation failed: {report.errors}"