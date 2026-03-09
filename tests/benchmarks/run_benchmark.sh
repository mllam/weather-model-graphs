run_quiet_or_die() {
  local out rc
  echo "+ $*" >&2
  out="$("$@" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '%s\n' "$out"
    exit "$rc"
  fi
}

# CPU-backed benchmark, no cu-graph
echo "=== Running CPU-backed benchmark (standard NetworkX backend) ===" >&2
run_quiet_or_die rm -rf .venv
run_quiet_or_die uv sync --all-extras
uv run python examples/benchmark_graph_build_backend.py

# GPU-backed benchmark, with cu-graph
echo "=== Running GPU-backed benchmark (NX_CUGRAPH_AUTOCONFIG=True) ===" >&2
run_quiet_or_die rm -rf .venv/
uv venv --no-project
run_quiet_or_die uv pip install torch --index-url https://download.pytorch.org/whl/cu130
run_quiet_or_die uv sync --all-extras
run_quiet_or_die uv pip install "nx-cugraph-cu12>=26.2.0"
NX_CUGRAPH_AUTOCONFIG=True uv run python examples/benchmark_graph_build_backend.py
