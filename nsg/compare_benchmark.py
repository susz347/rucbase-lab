"""
A/B Comparison: Baseline vs Optimized NSG Search
Compares QPS, Recall, and Speedup.

Usage: python compare_benchmark.py
Place this script in the nsg/ directory and run after building the project.
"""
import os, sys, struct, subprocess, json, time, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(PROJECT_ROOT, "data")
BIN = os.path.join(SCRIPT_DIR, "build_msvc")


def read_ivecs(filename):
    data = []
    with open(filename, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("<i", dim_bytes)[0]
            vec = np.frombuffer(f.read(dim * 4), dtype="<i4")
            data.append(vec)
    return np.array(data)


def compute_recall(gt_path, result_path, k=100):
    gt = read_ivecs(gt_path)
    result = read_ivecs(result_path)
    n = gt.shape[0]
    recall = sum(len(set(gt[i, :k]) & set(result[i, :k])) / k for i in range(n)) / n
    return recall


def run_search(exe, data_path, query_path, nsg_path, search_L, search_K, result_path, n_runs=3):
    cmd = [exe, data_path, query_path, nsg_path, str(search_L), str(search_K), result_path]
    best_time = float('inf')
    for _ in range(n_runs):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        for line in output.split("\n"):
            if "search time" in line:
                t = float(line.split(":")[-1].strip())
                if t < best_time:
                    best_time = t
    return best_time


def main():
    data_path = os.path.join(DATA, "sift_base.fvecs")
    query_path = os.path.join(DATA, "sift_query.fvecs")
    nsg_path = os.path.join(DATA, "sift.nsg")
    gt_path = os.path.join(DATA, "sift_groundtruth.ivecs")

    if not os.path.exists(gt_path):
        print("WARNING: No groundtruth file. Recall computation disabled.")
        gt_path = None

    with open(query_path, "rb") as f:
        dim_bytes = f.read(4)
        dim = struct.unpack("<i", dim_bytes)[0]
        f.seek(0, os.SEEK_END)
        n_query = os.fstat(f.fileno()).st_size // (dim + 1) // 4
    print(f"Dataset: {n_query} queries, dim={dim}")

    baseline_exe = os.path.join(BIN, "test_nsg_optimized_search.exe")
    optimized_exe = os.path.join(BIN, "test_nsg_optimized_v2.exe")

    search_L_values = [100, 120, 150, 200]
    search_K = 100

    print(f"\n{'='*70}")
    print(f"{'A/B Comparison: Baseline vs Optimized NSG Search':^70}")
    print(f"{'='*70}")
    print(f"{'L':<8} {'Base QPS':<16} {'Opt QPS':<16} {'Speedup':<10} {'Recall@100':<12}")
    print(f"{'-'*62}")

    results = []
    for search_L in search_L_values:
        result_base = os.path.join(DATA, f"cmp_base_L{search_L}.ivecs")
        result_opt = os.path.join(DATA, f"cmp_opt_L{search_L}.ivecs")

        t_base = run_search(baseline_exe, data_path, query_path, nsg_path,
                            search_L, search_K, result_base)
        qps_base = n_query / t_base

        t_opt = run_search(optimized_exe, data_path, query_path, nsg_path,
                           search_L, search_K, result_opt)
        qps_opt = n_query / t_opt

        recall = compute_recall(gt_path, result_opt, search_K) if gt_path and os.path.exists(gt_path) else 0
        speedup = qps_opt / qps_base

        results.append({"L": search_L, "qps_base": qps_base, "qps_opt": qps_opt,
                        "speedup": speedup, "recall": recall})
        print(f"{search_L:<8} {qps_base:<16.0f} {qps_opt:<16.0f} {speedup:<10.2f}x {recall*100:<12.2f}%")

    avg_speedup = np.mean([r["speedup"] for r in results])
    max_speedup = max([r["speedup"] for r in results])
    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    print(f"Maximum Speedup: {max_speedup:.2f}x")

    results_file = os.path.join(DATA, "comparison_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
