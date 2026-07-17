"""Compare two scaling-benchmark JSON outputs and render a regression report.

This consumes the JSON produced by ``graph_creation_scaling.py --output-json``
(added in #140), i.e. a list of records of the form::

    [{"grid_points": int, "runtime_s": float, "peak_memory_mb": float | null}, ...]

Given a *baseline* run (typically ``main``) and a *contender* run (the PR), it
matches records by ``grid_points``, computes the relative change in runtime, and
renders a Markdown table suitable for posting as a sticky pull-request comment.

The design follows the discussion in
https://github.com/mllam/weather-model-graphs/issues/144: we compare relative
(%) change rather than absolute seconds, because the two runs are executed
back-to-back on the *same* CI runner and only the library under test differs, so
the per-runner noise largely cancels.

Usage::

    python -m tests.benchmarks.compare main.json pr.json
    python -m tests.benchmarks.compare main.json pr.json \\
        --threshold-pct 0.1 --output comment.md --fail-on-regression

The script only depends on the standard library so it can run in a minimal CI
step without installing the package.
"""

import argparse
import json
import sys
from typing import Dict, List, Optional

# Hidden marker used by the CI workflow to find-and-update a single sticky
# comment instead of posting a new comment on every run.
STICKY_MARKER = "<!-- benchmark-regression-check -->"


def load_results(path: str) -> Dict[int, dict]:
    """Load a benchmark JSON file and index the records by ``grid_points``.

    Raises ``ValueError`` with an actionable message if the file is empty or
    does not match the expected schema, so CI failures are easy to diagnose.
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(
            f"{path!r} does not contain a non-empty list of benchmark records"
        )

    indexed: Dict[int, dict] = {}
    for record in data:
        if "grid_points" not in record or "runtime_s" not in record:
            raise ValueError(
                f"{path!r} contains a record missing required keys "
                f"'grid_points'/'runtime_s': {record!r}"
            )
        indexed[int(record["grid_points"])] = record
    return indexed


def _pct_change(baseline: float, contender: float) -> Optional[float]:
    """Return the percentage change from ``baseline`` to ``contender``.

    Returns ``None`` when the baseline is zero (or negative), since a relative
    change is undefined there and we would rather skip the row than divide by
    zero.
    """
    if baseline <= 0:
        return None
    return (contender - baseline) / baseline * 100.0


class Row:
    """A single grid-size comparison line in the report."""

    def __init__(
        self,
        grid_points: int,
        baseline_s: float,
        contender_s: float,
        delta_pct: Optional[float],
        is_regression: bool,
    ):
        self.grid_points = grid_points
        self.baseline_s = baseline_s
        self.contender_s = contender_s
        self.delta_pct = delta_pct
        self.is_regression = is_regression


def compare(
    baseline: Dict[int, dict],
    contender: Dict[int, dict],
    threshold_pct: float,
) -> List[Row]:
    """Build the per-grid-size comparison rows for the runtime metric.

    Only ``grid_points`` present in *both* runs are compared; unmatched sizes
    are skipped (they show up in the PR diff of the benchmark itself, so there is
    nothing to compare against). Rows are returned sorted by ``grid_points``.
    """
    common = sorted(set(baseline) & set(contender))
    rows: List[Row] = []
    for gp in common:
        b = float(baseline[gp]["runtime_s"])
        c = float(contender[gp]["runtime_s"])
        delta = _pct_change(b, c)
        is_regression = delta is not None and delta > threshold_pct
        rows.append(Row(gp, b, c, delta, is_regression))
    return rows


def _fmt_seconds(value: float) -> str:
    """Format a runtime in seconds with millisecond-level readability."""
    if value < 1.0:
        return f"{value * 1000:.0f}ms"
    return f"{value:.3f}s"


def _fmt_delta(delta: Optional[float], is_regression: bool) -> str:
    if delta is None:
        return "n/a"
    icon = "⚠️" if is_regression else "✅"
    return f"{delta:+.1f}% {icon}"


def render_markdown(
    rows: List[Row],
    threshold_pct: float,
    baseline_label: str,
    contender_label: str,
    unmatched: Optional[List[int]] = None,
) -> str:
    """Render the comparison as a Markdown block, prefixed with the sticky marker."""
    lines = [STICKY_MARKER, "## ⏱️ Graph-creation benchmark: regression check", ""]

    if not rows:
        lines.append(
            "No overlapping grid sizes to compare between "
            f"`{baseline_label}` and `{contender_label}`."
        )
        return "\n".join(lines) + "\n"

    regressions = [r for r in rows if r.is_regression]
    if regressions:
        lines.append(
            f"⚠️ **{len(regressions)} of {len(rows)}** grid sizes exceed the "
            f"**+{threshold_pct:g}%** runtime threshold."
        )
    else:
        lines.append(
            f"✅ No runtime regression above **+{threshold_pct:g}%** "
            f"across {len(rows)} grid sizes."
        )
    lines.append("")

    lines.append(f"| grid points | {baseline_label} | {contender_label} | Δ runtime |")
    lines.append("|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.grid_points:,} | {_fmt_seconds(r.baseline_s)} | "
            f"{_fmt_seconds(r.contender_s)} | {_fmt_delta(r.delta_pct, r.is_regression)} |"
        )

    if unmatched:
        pretty = ", ".join(f"{gp:,}" for gp in unmatched)
        lines.append("")
        lines.append(
            f"_Note: {len(unmatched)} grid size(s) not present in both runs were "
            f"skipped: {pretty}._"
        )

    lines.append("")
    lines.append(
        "_Runs execute back-to-back on the same runner; only the library under "
        "test differs, so relative change is compared rather than absolute time._"
    )
    return "\n".join(lines) + "\n"


def build_report(
    baseline_path: str,
    contender_path: str,
    threshold_pct: float,
    baseline_label: str,
    contender_label: str,
):
    """Load both files, compute rows, and return ``(markdown, rows)``."""
    baseline = load_results(baseline_path)
    contender = load_results(contender_path)
    rows = compare(baseline, contender, threshold_pct)
    unmatched = sorted(set(baseline) ^ set(contender))
    markdown = render_markdown(
        rows, threshold_pct, baseline_label, contender_label, unmatched
    )
    return markdown, rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two scaling-benchmark JSON outputs (baseline vs PR)."
    )
    parser.add_argument("baseline", help="Baseline JSON (e.g. main.json).")
    parser.add_argument("contender", help="Contender JSON (e.g. pr.json).")
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=0.1,
        help="Flag a grid size when its runtime increases by more than this "
        "percentage. Start low and raise it once the runner's noise floor is "
        "known (default: 0.1).",
    )
    parser.add_argument(
        "--baseline-label", default="main", help="Column label for the baseline."
    )
    parser.add_argument(
        "--contender-label", default="PR", help="Column label for the contender."
    )
    parser.add_argument(
        "--output",
        help="Also write the Markdown report to this file (for the CI comment).",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any grid size regresses (off by default so the "
        "check is informational).",
    )
    return parser.parse_args(argv)


def _print_utf8(text: str) -> None:
    """Print ``text`` as UTF-8 regardless of the console's default encoding.

    The report contains emoji (✅/⚠️); on a Windows console (cp1252) a plain
    ``print`` raises ``UnicodeEncodeError``. CI runners are UTF-8, but we keep
    the tool robust for local use.
    """
    stream = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass
    try:
        print(text)
    except UnicodeEncodeError:
        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            buffer.write(text.encode("utf-8") + b"\n")
        else:  # pragma: no cover - extremely unusual stdout replacement
            print(text.encode("utf-8", "backslashreplace").decode("ascii"))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    markdown, rows = build_report(
        args.baseline,
        args.contender,
        args.threshold_pct,
        args.baseline_label,
        args.contender_label,
    )

    _print_utf8(markdown)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(markdown)

    if args.fail_on_regression and any(r.is_regression for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
