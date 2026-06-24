"""
一键复现脚本 — NSG 查询性能优化实验

用法：
    python reproduce.py                    # 完整复现（生成数据→编译→构建索引→A/B对比）
    python reproduce.py --data-only        # 仅生成数据
    python reproduce.py --build-only       # 仅编译
    python reproduce.py --benchmark        # 仅运行基准测试（需已有数据和编译产物）

环境要求：
    - Windows: MSVC 2022+ 或 Visual Studio Build Tools
    - CPU 支持 AVX2 + FMA
    - Python 3.10+ with numpy
"""

import os
import sys
import argparse
import subprocess
import struct
import json
import time
import numpy as np

# ---------------------------------------------------------------------------
# 路径配置（可修改 MSVC_ROOT 指向你的 MSVC 安装目录）
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NSG_DIR = os.path.join(PROJECT_ROOT, "nsg")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BUILD_DIR = os.path.join(NSG_DIR, "build_msvc")

# MSVC 路径 — 按优先级自动检测
MSVC_CANDIDATES = [
    r"E:\visutal studio c++\VC\Tools\MSVC\14.50.35717",
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807",
    r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.40.33807",
    r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\14.40.33807",
]

WIN_SDK_CANDIDATES = [
    r"C:\Program Files (x86)\Windows Kits\10",
]


def find_msvc():
    """Auto-detect MSVC toolchain. Returns (cl_exe, include_path, lib_path) or None."""
    import glob as g

    # Try candidates
    for base in MSVC_CANDIDATES:
        pattern = os.path.join(base, "bin", "Hostx64", "x64", "cl.exe")
        matches = g.glob(pattern)
        if matches:
            cl_exe = matches[0]
            include_dir = os.path.join(base, "include")
            lib_dir = os.path.join(base, "lib", "x64")
            return cl_exe, include_dir, lib_dir

    # Try to auto-find via vswhere
    try:
        result = subprocess.run(
            ['where', 'cl'], capture_output=True, text=True, shell=True, timeout=10)
        if result.returncode == 0:
            cl = result.stdout.strip().split('\n')[0]
            return cl, None, None
    except Exception:
        pass

    # Fallback: assume cl is in PATH
    return "cl", None, None


def find_windows_sdk():
    """Find Windows SDK include/lib paths."""
    import glob as g
    for base in WIN_SDK_CANDIDATES:
        inc_pattern = os.path.join(base, "Include", "10.*")
        inc_matches = sorted(g.glob(inc_pattern), reverse=True)
        lib_pattern = os.path.join(base, "Lib", "10.*")
        lib_matches = sorted(g.glob(lib_pattern), reverse=True)
        if inc_matches and lib_matches:
            return inc_matches[0], lib_matches[0]
    return None, None


def get_msvc_env():
    """Build environment dict for MSVC compilation."""
    cl, inc, lib = find_msvc()
    sdk_inc, sdk_lib = find_windows_sdk()

    env = os.environ.copy()

    include_dirs = []
    if inc:
        include_dirs.append(inc)
    if sdk_inc:
        include_dirs.extend([
            os.path.join(sdk_inc, "ucrt"),
            os.path.join(sdk_inc, "shared"),
            os.path.join(sdk_inc, "um"),
        ])
    if include_dirs:
        env["INCLUDE"] = ";".join(include_dirs)

    lib_dirs = []
    if lib:
        lib_dirs.append(lib)
    if sdk_lib:
        lib_dirs.extend([
            os.path.join(sdk_lib, "ucrt", "x64"),
            os.path.join(sdk_lib, "um", "x64"),
        ])
    if lib_dirs:
        env["LIB"] = ";".join(lib_dirs)

    return cl, env


# ---------------------------------------------------------------------------
# 数据生成
# ---------------------------------------------------------------------------
def generate_data(n_base=200000, n_query=200, dim=128):
    """Generate synthetic data with SIFT-like distribution."""
    print(f"\n{'='*60}")
    print("Step 1: Generating synthetic data")
    print(f"{'='*60}")
    print(f"  Base vectors: {n_base}")
    print(f"  Query vectors: {n_query}")
    print(f"  Dimension: {dim}")

    os.makedirs(DATA_DIR, exist_ok=True)
    np.random.seed(42)

    n_clusters = 500
    centers = np.random.randn(n_clusters, dim).astype(np.float32) * 0.5

    base = np.zeros((n_base, dim), dtype=np.float32)
    chunk = 50000
    for i in range(0, n_base, chunk):
        end = min(i + chunk, n_base)
        c_ids = np.random.randint(0, n_clusters, end - i)
        base[i:end] = centers[c_ids] + np.random.randn(end - i, dim).astype(np.float32) * 0.2

    query = np.zeros((n_query, dim), dtype=np.float32)
    q_ids = np.random.randint(0, n_clusters, n_query)
    query = centers[q_ids] + np.random.randn(n_query, dim).astype(np.float32) * 0.1

    # Normalize
    base_norm = np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-8)
    query_norm = np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-8)
    base = (base / base_norm).astype(np.float32)
    query = (query / query_norm).astype(np.float32)

    # Write fvecs
    def write_fvecs(path, vecs):
        n, d = vecs.shape
        with open(path, "wb") as f:
            for i in range(n):
                f.write(struct.pack("<i", d))
                f.write(vecs[i].tobytes())

    base_path = os.path.join(DATA_DIR, "sift_base.fvecs")
    query_path = os.path.join(DATA_DIR, "sift_query.fvecs")
    write_fvecs(base_path, base)
    write_fvecs(query_path, query)
    print(f"  Saved: {base_path} ({os.path.getsize(base_path)/1024/1024:.1f} MB)")
    print(f"  Saved: {query_path} ({os.path.getsize(query_path)/1024:.1f} KB)")

    # Build kNN graph (brute-force for synthetic data)
    print("\n  Building kNN graph (K=200)...")
    k = 200
    graph = np.zeros((n_base, k), dtype=np.int32)
    graph_chunk = 1000
    for i in range(0, n_base, graph_chunk):
        end = min(i + graph_chunk, n_base)
        dists = -np.dot(base[i:end], base.T)
        top_k = np.argpartition(-dists, k, axis=1)[:, :k]
        for j in range(end - i):
            idx = top_k[j]
            graph[i + j] = idx[np.argsort(-dists[j, idx])]
        if (end) % 20000 == 0:
            print(f"    {end}/{n_base}")

    kng_path = os.path.join(DATA_DIR, "sift_200nn.graph")
    with open(kng_path, "wb") as f:
        f.write(struct.pack("<i", k))
        for i in range(n_base):
            f.write(struct.pack("<i", k))
            f.write(graph[i].tobytes())
    print(f"  Saved: {kng_path} ({os.path.getsize(kng_path)/1024/1024:.1f} MB)")

    # Compute groundtruth
    print("\n  Computing groundtruth (K=100)...")
    gt_k = 100
    gt = np.zeros((n_query, gt_k), dtype=np.int32)
    for i in range(0, n_query, 50):
        end = min(i + 50, n_query)
        dists = -np.dot(query[i:end], base.T)
        top_k = np.argpartition(-dists, gt_k, axis=1)[:, :gt_k]
        for j in range(end - i):
            idx = top_k[j]
            gt[i + j] = idx[np.argsort(-dists[j, idx])]
        if (end) % 100 == 0:
            print(f"    {end}/{n_query}")

    def write_ivecs(path, ids):
        n, k = ids.shape
        with open(path, "wb") as f:
            for i in range(n):
                f.write(struct.pack("<i", k))
                f.write(ids[i].astype("<i4").tobytes())

    gt_path = os.path.join(DATA_DIR, "sift_groundtruth.ivecs")
    write_ivecs(gt_path, gt)
    print(f"  Saved: {gt_path}")

    print("  Data generation complete.")
    return base_path, query_path, kng_path, gt_path


# ---------------------------------------------------------------------------
# 编译
# ---------------------------------------------------------------------------
def build_project():
    """Compile the NSG project using MSVC."""
    print(f"\n{'='*60}")
    print("Step 2: Building NSG project")
    print(f"{'='*60}")

    cl, env = get_msvc_env()
    print(f"  MSVC compiler: {cl}")

    build_script = os.path.join(NSG_DIR, "build_msvc.py")

    # Instead of calling build_msvc.py (which has hardcoded paths), do a custom build
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_msvc", build_script)
    bm = importlib.util.module_from_spec(spec)

    # Monkey-patch find_cl to use our detected compiler
    original_dir = os.getcwd()
    os.chdir(NSG_DIR)
    try:
        spec.loader.exec_module(bm)
        # Replace find_cl
        bm.find_cl = lambda: cl
        bm.build()
    finally:
        os.chdir(original_dir)

    # Verify outputs
    expected = ["test_nsg_index.exe", "test_nsg_optimized_search.exe", "test_nsg_optimized_v2.exe"]
    for exe in expected:
        path = os.path.join(BUILD_DIR, exe)
        if os.path.exists(path):
            print(f"  [OK] {exe}")
        else:
            print(f"  [MISSING] {exe}")

    print("  Build complete.")


# ---------------------------------------------------------------------------
# 索引构建
# ---------------------------------------------------------------------------
def build_nsg_index(data_path, kng_path, L=40, R=50, C=500):
    """Build NSG index."""
    print(f"\n{'='*60}")
    print("Step 3: Building NSG index")
    print(f"{'='*60}")

    nsg_path = os.path.join(DATA_DIR, "sift.nsg")
    if os.path.exists(nsg_path):
        print(f"  Using existing index: {nsg_path}")
        return nsg_path

    exe = os.path.join(BUILD_DIR, "test_nsg_index.exe")
    cmd = [exe, data_path, kng_path, str(L), str(R), str(C), nsg_path]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr
    for line in output.split('\n'):
        if any(kw in line for kw in ['Degree', 'indexing', 'time']):
            print(f"  {line.strip()}")
    print(f"  Index saved: {nsg_path}")
    return nsg_path


# ---------------------------------------------------------------------------
# 基准测试
# ---------------------------------------------------------------------------
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
    recall = 0.0
    for i in range(n):
        gt_set = set(gt[i, :k])
        res_set = set(result[i, :k])
        recall += len(gt_set & res_set) / k
    return recall / n


def run_search(exe, data_path, query_path, nsg_path, search_L, search_K, result_path, n_runs=3):
    """Run search, return best elapsed time in seconds."""
    cmd = [exe, data_path, query_path, nsg_path, str(search_L), str(search_K), result_path]
    best = float('inf')
    for _ in range(n_runs):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        for line in output.split('\n'):
            if "search time" in line:
                t = float(line.split(":")[-1].strip())
                if t < best:
                    best = t
    return best


def run_benchmark():
    """A/B comparison: Baseline vs Optimized."""
    print(f"\n{'='*60}")
    print("Step 4: Running A/B benchmark")
    print(f"{'='*60}")

    data_path = os.path.join(DATA_DIR, "sift_base.fvecs")
    query_path = os.path.join(DATA_DIR, "sift_query.fvecs")
    nsg_path = os.path.join(DATA_DIR, "sift.nsg")
    gt_path = os.path.join(DATA_DIR, "sift_groundtruth.ivecs")

    baseline_exe = os.path.join(BUILD_DIR, "test_nsg_optimized_search.exe")
    optimized_exe = os.path.join(BUILD_DIR, "test_nsg_optimized_v2.exe")

    # Count queries
    with open(query_path, "rb") as f:
        dim_bytes = f.read(4)
        dim = struct.unpack("<i", dim_bytes)[0]
        f.seek(0, os.SEEK_END)
        n_query = os.fstat(f.fileno()).st_size // (dim + 1) // 4

    print(f"  {n_query} queries, dim={dim}")

    search_L_values = [100, 120, 150, 200]
    search_K = 100

    print(f"\n  {'L':<8} {'Base QPS':<14} {'Opt QPS':<14} {'Speedup':<10} {'Recall@100':<12}")
    print(f"  {'-'*58}")

    results = []
    for sl in search_L_values:
        r_base = os.path.join(DATA_DIR, f"result_base_L{sl}.ivecs")
        r_opt = os.path.join(DATA_DIR, f"result_opt_L{sl}.ivecs")

        t_base = run_search(baseline_exe, data_path, query_path, nsg_path, sl, search_K, r_base)
        t_opt = run_search(optimized_exe, data_path, query_path, nsg_path, sl, search_K, r_opt)

        qps_base = n_query / t_base
        qps_opt = n_query / t_opt
        recall = compute_recall(gt_path, r_opt, search_K)
        speedup = qps_opt / qps_base

        results.append({"L": sl, "qps_base": qps_base, "qps_opt": qps_opt,
                        "speedup": speedup, "recall": recall})
        print(f"  {sl:<8} {qps_base:<14.0f} {qps_opt:<14.0f} {speedup:<10.2f}x {recall*100:<12.2f}%")

    # Summary
    avg_speedup = np.mean([r["speedup"] for r in results])
    max_speedup = max([r["speedup"] for r in results])
    print(f"\n  Average speedup: {avg_speedup:.2f}x")
    print(f"  Maximum speedup: {max_speedup:.2f}x")

    # Save results
    results_file = os.path.join(DATA_DIR, "comparison_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_file}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NSG 查询性能优化实验 — 一键复现")
    parser.add_argument("--data-only", action="store_true", help="仅生成数据")
    parser.add_argument("--build-only", action="store_true", help="仅编译")
    parser.add_argument("--benchmark", action="store_true", help="仅运行基准测试")
    args = parser.parse_args()

    do_all = not (args.data_only or args.build_only or args.benchmark)

    t_start = time.time()

    if do_all or args.data_only:
        data_path, query_path, kng_path, gt_path = generate_data()

    if do_all or args.build_only:
        build_project()

    if do_all:
        nsg_path = build_nsg_index(
            os.path.join(DATA_DIR, "sift_base.fvecs"),
            os.path.join(DATA_DIR, "sift_200nn.graph"),
        )

    if do_all or args.benchmark:
        results = run_benchmark()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Total time: {elapsed:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
