"""
NSG Baseline Benchmark Script
Generates Recall-QPS data by varying search_L.

Usage: python benchmark.py
Place this script in the nsg/ directory and run after building the project.
"""
import os, sys, struct, time, subprocess, json, numpy as np

# Paths relative to this script (nsg/ directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_msvc")
TEST_EXE = os.path.join(BUILD_DIR, "test_nsg_optimized_search.exe")


def write_fvecs(filename, vectors):
    n, dim = vectors.shape
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(struct.pack("<i", dim))
            f.write(vectors[i].astype("<f4").tobytes())


def write_ivecs(filename, ids):
    n, k = ids.shape
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(struct.pack("<i", k))
            f.write(ids[i].astype("<i4").tobytes())


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


def generate_data(n_base=200000, n_query=1000, dim=128):
    print(f"Generating data: {n_base} base, {n_query} query, dim={dim}")
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

    base_norm = np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-8)
    query_norm = np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-8)
    base = (base / base_norm).astype(np.float32)
    query = (query / query_norm).astype(np.float32)

    return base, query


def compute_groundtruth_approx(base, query, k=100):
    print(f"Computing groundtruth (K={k})...")
    n_query = query.shape[0]
    gt = np.zeros((n_query, k), dtype=np.int32)
    chunk = 200
    for i in range(0, n_query, chunk):
        end = min(i + chunk, n_query)
        dists = -np.dot(query[i:end], base.T)
        top_k = np.argpartition(-dists, k, axis=1)[:, :k]
        for j in range(end - i):
            idx = top_k[j]
            gt[i + j] = idx[np.argsort(-dists[j, idx])]
        if (end) % 200 == 0:
            print(f"  {end}/{n_query}")
    return gt


def build_knn_graph(base, k=200, kng_path=None):
    n_base = base.shape[0]
    print(f"Building kNN graph (K={k})...")
    graph = np.zeros((n_base, k), dtype=np.int32)
    chunk = 1000
    for i in range(0, n_base, chunk):
        end = min(i + chunk, n_base)
        dists = -np.dot(base[i:end], base.T)
        top_k = np.argpartition(-dists, k, axis=1)[:, :k]
        for j in range(end - i):
            idx = top_k[j]
            graph[i + j] = idx[np.argsort(-dists[j, idx])]
        if (end) % 20000 == 0:
            print(f"  {end}/{n_base}")
    if kng_path:
        with open(kng_path, "wb") as f:
            f.write(struct.pack("<i", k))
            for i in range(n_base):
                f.write(struct.pack("<i", k))
                f.write(graph[i].tobytes())
    return graph


def build_nsg(data_path, kng_path, nsg_path, L=40, R=50, C=500):
    exe = os.path.join(BUILD_DIR, "test_nsg_index.exe")
    cmd = [exe, data_path, kng_path, str(L), str(R), str(C), nsg_path]
    print(f"Building NSG index (L={L}, R={R}, C={C})...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr
    for line in output.split("\n"):
        if "indexing time" in line or "Degree" in line:
            print(f"  {line.strip()}")
    return True


def run_search(data_path, query_path, nsg_path, search_L, search_K, result_path):
    cmd = [TEST_EXE, data_path, query_path, nsg_path, str(search_L), str(search_K), result_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr
    for line in output.split("\n"):
        if "search time" in line:
            return float(line.split(":")[-1].strip())
    return None


def compute_recall(gt_path, result_path, k=100):
    gt = read_ivecs(gt_path)
    result = read_ivecs(result_path)
    n = gt.shape[0]
    recall = sum(len(set(gt[i, :k]) & set(result[i, :k])) / k for i in range(n)) / n
    return recall


def run_benchmark(name, data_path, query_path, nsg_path, gt_path,
                  search_L_values, search_K=100, n_query=None):
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"{'='*60}")

    with open(query_path, "rb") as f:
        dim_bytes = f.read(4)
        dim = struct.unpack("<i", dim_bytes)[0]
        f.seek(0, os.SEEK_END)
        query_num = os.fstat(f.fileno()).st_size // (dim + 1) // 4
    if n_query:
        query_num = min(query_num, n_query)

    results = []
    for search_L in search_L_values:
        result_path = os.path.join(DATA_DIR, f"result_L{search_L}.ivecs")
        print(f"\n  search_L={search_L}:")
        run_search(data_path, query_path, nsg_path, search_L, search_K, result_path)  # warm-up
        elapsed = run_search(data_path, query_path, nsg_path, search_L, search_K, result_path)
        if elapsed is None:
            print(f"    ERROR: Search failed")
            continue
        qps = query_num / elapsed
        recall = compute_recall(gt_path, result_path, search_K)
        results.append((search_L, recall, qps, elapsed))
        print(f"    Recall@{search_K}: {recall*100:.2f}%")
        print(f"    QPS: {qps:.0f}")
        print(f"    Time: {elapsed*1000:.1f} ms ({query_num} queries)")

    # Summary
    print(f"\n  Summary ({name}):")
    print(f"  {'L':<10} {'Recall':<14} {'QPS':<14} {'Time(ms)':<14}")
    print(f"  {'-'*52}")
    for sl, rec, qps, t in results:
        print(f"  {sl:<10} {rec*100:<14.2f}% {qps:<14.0f} {t*1000:<14.1f}")

    return results


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    base, query = generate_data(n_base=200000, n_query=1000, dim=128)

    data_path = os.path.join(DATA_DIR, "sift_base.fvecs")
    query_path = os.path.join(DATA_DIR, "sift_query.fvecs")
    kng_path = os.path.join(DATA_DIR, "sift_200nn.graph")
    nsg_path = os.path.join(DATA_DIR, "sift.nsg")
    gt_path = os.path.join(DATA_DIR, "sift_groundtruth.ivecs")

    write_fvecs(data_path, base)
    write_fvecs(query_path, query)

    if not os.path.exists(kng_path):
        build_knn_graph(base, k=200, kng_path=kng_path)
    if not os.path.exists(nsg_path):
        build_nsg(data_path, kng_path, nsg_path, L=40, R=50, C=500)
    if not os.path.exists(gt_path):
        gt = compute_groundtruth_approx(base, query, k=100)
        write_ivecs(gt_path, gt)

    results = run_benchmark(
        "Baseline", data_path, query_path, nsg_path, gt_path,
        [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200],
        search_K=100, n_query=500)

    results_file = os.path.join(DATA_DIR, "baseline_results.json")
    with open(results_file, "w") as f:
        json.dump([(sl, float(r), float(q), float(t)) for sl, r, q, t in results], f)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
