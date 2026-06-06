import json
import sys

def check_regression(json_file, runtime_threshold_s, memory_threshold_mb):
    with open(json_file) as f:
        data = json.load(f)
    # Assume we only have one data point (fixed grid size)
    if not data:
        print("No data in JSON file")
        return False
    result = data[0]
    runtime_ok = result["runtime_s"] <= runtime_threshold_s
    memory_ok = result["peak_memory_mb"] <= memory_threshold_mb
    if not runtime_ok:
        print(f"Runtime regression: {result['runtime_s']:.2f}s > {runtime_threshold_s}s")
    if not memory_ok:
        print(f"Memory regression: {result['peak_memory_mb']:.1f}MB > {memory_threshold_mb}MB")
    return runtime_ok and memory_ok

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: check_regression.py <json_file> <runtime_threshold_s> <memory_threshold_mb>")
        sys.exit(1)
    ok = check_regression(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
    sys.exit(0 if ok else 1)