"""Unit tests for the benchmark comparison script (``tests/benchmarks/compare.py``).

These are pure-stdlib tests (no ``weather_model_graphs`` import needed) so they
run in the ordinary pytest job and give us confidence in the regression logic
independently of an actual benchmark run.
"""

import json

import pytest

from tests.benchmarks import compare


def _write(tmp_path, name, records):
    path = tmp_path / name
    path.write_text(json.dumps(records))
    return str(path)


def _rec(grid_points, runtime_s, peak_memory_mb=None):
    return {
        "grid_points": grid_points,
        "runtime_s": runtime_s,
        "peak_memory_mb": peak_memory_mb,
    }


def test_pct_change_basic():
    assert compare._pct_change(1.0, 1.5) == pytest.approx(50.0)
    assert compare._pct_change(2.0, 1.0) == pytest.approx(-50.0)


def test_pct_change_zero_baseline_is_none():
    assert compare._pct_change(0.0, 1.0) is None
    assert compare._pct_change(-1.0, 1.0) is None


def test_compare_flags_regression_above_threshold():
    baseline = {1024: _rec(1024, 1.00)}
    contender = {1024: _rec(1024, 1.30)}  # +30%
    rows = compare.compare(baseline, contender, threshold_pct=25.0)
    assert len(rows) == 1
    assert rows[0].delta_pct == pytest.approx(30.0)
    assert rows[0].is_regression is True


def test_compare_no_flag_below_threshold():
    baseline = {1024: _rec(1024, 1.00)}
    contender = {1024: _rec(1024, 1.02)}  # +2%
    rows = compare.compare(baseline, contender, threshold_pct=25.0)
    assert rows[0].is_regression is False


def test_threshold_boundary_is_strict_greater_than():
    # Exactly at threshold must NOT flag (we only flag strictly above it).
    # Use an exact 0% delta against a 0% threshold to avoid float-rounding fuzz.
    baseline = {1024: _rec(1024, 1.00)}
    contender = {1024: _rec(1024, 1.00)}  # +0.0%
    rows = compare.compare(baseline, contender, threshold_pct=0.0)
    assert rows[0].delta_pct == pytest.approx(0.0)
    assert rows[0].is_regression is False  # 0.0 is not > 0.0

    # And a hair above the threshold must flag.
    rows_above = compare.compare(
        {1024: _rec(1024, 1.00)}, {1024: _rec(1024, 1.05)}, threshold_pct=1.0
    )
    assert rows_above[0].is_regression is True


def test_improvement_is_never_a_regression():
    baseline = {1024: _rec(1024, 2.00)}
    contender = {1024: _rec(1024, 1.00)}  # -50%
    rows = compare.compare(baseline, contender, threshold_pct=0.1)
    assert rows[0].delta_pct == pytest.approx(-50.0)
    assert rows[0].is_regression is False


def test_only_common_grid_points_compared_and_sorted():
    baseline = {4096: _rec(4096, 2.0), 1024: _rec(1024, 1.0), 256: _rec(256, 0.5)}
    contender = {1024: _rec(1024, 1.0), 4096: _rec(4096, 2.0), 9001: _rec(9001, 9.0)}
    rows = compare.compare(baseline, contender, threshold_pct=0.1)
    assert [r.grid_points for r in rows] == [1024, 4096]  # sorted, intersection only


def test_zero_baseline_row_is_not_a_regression():
    baseline = {1024: _rec(1024, 0.0)}
    contender = {1024: _rec(1024, 1.0)}
    rows = compare.compare(baseline, contender, threshold_pct=0.1)
    assert rows[0].delta_pct is None
    assert rows[0].is_regression is False


def test_load_results_rejects_empty(tmp_path):
    path = _write(tmp_path, "empty.json", [])
    with pytest.raises(ValueError, match="non-empty list"):
        compare.load_results(path)


def test_load_results_rejects_missing_keys(tmp_path):
    path = _write(tmp_path, "bad.json", [{"grid_points": 10}])
    with pytest.raises(ValueError, match="missing required keys"):
        compare.load_results(path)


def test_load_results_indexes_by_grid_points(tmp_path):
    path = _write(tmp_path, "ok.json", [_rec(1024, 1.0), _rec(4096, 2.0)])
    indexed = compare.load_results(path)
    assert set(indexed) == {1024, 4096}
    assert indexed[4096]["runtime_s"] == 2.0


def test_render_markdown_contains_marker_and_table():
    rows = compare.compare(
        {1024: _rec(1024, 1.0)}, {1024: _rec(1024, 1.3)}, threshold_pct=0.1
    )
    md = compare.render_markdown(rows, 0.1, "main", "PR")
    assert compare.STICKY_MARKER in md
    assert "| grid points | main | PR | Δ runtime |" in md
    assert "+30.0% ⚠️" in md
    assert "1,024" in md  # thousands separator


def test_render_markdown_clean_run_reports_pass():
    rows = compare.compare(
        {1024: _rec(1024, 1.0)}, {1024: _rec(1024, 1.0)}, threshold_pct=0.1
    )
    md = compare.render_markdown(rows, 0.1, "main", "PR")
    assert "No runtime regression" in md
    assert "⚠️" not in md


def test_render_markdown_no_overlap_message():
    md = compare.render_markdown([], 0.1, "main", "PR", unmatched=[1024])
    assert "No overlapping grid sizes" in md


def test_build_report_end_to_end(tmp_path):
    base = _write(tmp_path, "main.json", [_rec(1024, 1.0), _rec(4096, 2.0)])
    cont = _write(tmp_path, "pr.json", [_rec(1024, 1.0), _rec(4096, 3.0)])  # +50%
    md, rows = compare.build_report(base, cont, 0.1, "main", "PR")
    assert len(rows) == 2
    assert any(r.is_regression for r in rows)
    assert "+50.0% ⚠️" in md


def test_main_fail_on_regression_exit_code(tmp_path, capsys):
    base = _write(tmp_path, "main.json", [_rec(1024, 1.0)])
    cont = _write(tmp_path, "pr.json", [_rec(1024, 2.0)])
    rc = compare.main([base, cont, "--threshold-pct", "0.1", "--fail-on-regression"])
    assert rc == 1
    # Without the flag it must stay informational (exit 0).
    rc2 = compare.main([base, cont, "--threshold-pct", "0.1"])
    assert rc2 == 0


def test_main_writes_output_file(tmp_path):
    base = _write(tmp_path, "main.json", [_rec(1024, 1.0)])
    cont = _write(tmp_path, "pr.json", [_rec(1024, 1.0)])
    out = tmp_path / "comment.md"
    rc = compare.main([base, cont, "--output", str(out)])
    assert rc == 0
    assert compare.STICKY_MARKER in out.read_text(encoding="utf-8")
