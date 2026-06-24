"""
Generate synthetic test data in SIFT1M format (.fvecs/.ivecs) for testing.
Also generates groundtruth for Recall calculation.
"""
import numpy as np
import struct
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def write_fvecs(filename, vectors):
    """Write vectors in .fvecs format (4-byte dim + 4*dim bytes per vector)."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    n, dim = vectors.shape
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('<i', dim))
            f.write(vectors[i].astype('<f4').tobytes())
    print(f"Wrote {filename}: {n} x {dim} vectors ({os.path.getsize(filename)/1024/1024:.1f} MB)")


def write_ivecs(filename, ids):
    """Write ids in .ivecs format."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    n, k = ids.shape
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('<i', k))
            f.write(ids[i].astype('<i4').tobytes())
    print(f"Wrote {filename}: {n} x {k} ids ({os.path.getsize(filename)/1024/1024:.1f} MB)")


def read_fvecs(filename):
    """Read .fvecs file and return numpy array."""
    data = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('<i', dim_bytes)[0]
            vec = np.frombuffer(f.read(dim * 4), dtype='<f4')
            data.append(vec)
    return np.array(data)


def compute_groundtruth(base, query, k=100):
    """Compute exact k-NN groundtruth by brute force."""
    n_base = base.shape[0]
    n_query = query.shape[0]
    gt = np.zeros((n_query, k), dtype=np.int32)

    chunk_size = 1000
    for i in range(0, n_query, chunk_size):
        end = min(i + chunk_size, n_query)
        chunk = query[i:end]
        # Compute all distances for this chunk
        dists = np.linalg.norm(chunk[:, np.newaxis, :] - base[np.newaxis, :, :], axis=2)
        # Get top-k nearest
        top_k = np.argpartition(dists, k, axis=1)[:, :k]
        # Sort within the top-k by distance
        for j in range(end - i):
            idx = top_k[j]
            sorted_idx = idx[np.argsort(dists[j, idx])]
            gt[i + j] = sorted_idx
        if (i + chunk_size) % 5000 == 0:
            print(f"  groundtruth progress: {min(end, n_query)}/{n_query}")

    return gt


def compute_recall(gt, result, k=100):
    """Compute Recall@k between groundtruth and search result."""
    n = gt.shape[0]
    recall_sum = 0
    for i in range(n):
        gt_set = set(gt[i, :k])
        res_set = set(result[i, :k])
        recall_sum += len(gt_set & res_set) / k
    return recall_sum / n


def generate_synthetic_sift(n_base=1000000, n_query=10000, dim=128, k=100):
    """Generate synthetic SIFT-like data."""
    print(f"Generating synthetic SIFT-like data: {n_base} base, {n_query} query, dim={dim}")

    # Use SIFT-like distribution (clustered data, not completely random)
    # Generate random cluster centers
    n_clusters = 1000
    np.random.seed(42)
    centers = np.random.randn(n_clusters, dim).astype(np.float32) * 0.5

    # Generate base vectors around clusters
    print("  Generating base vectors...")
    base = np.zeros((n_base, dim), dtype=np.float32)
    chunk = 100000
    for i in range(0, n_base, chunk):
        end = min(i + chunk, n_base)
        cluster_ids = np.random.randint(0, n_clusters, end - i)
        base[i:end] = centers[cluster_ids] + np.random.randn(end - i, dim).astype(np.float32) * 0.2
        if (i + chunk) % 500000 == 0:
            print(f"    base: {min(end, n_base)}/{n_base}")

    # Generate query vectors (near cluster centers but with offset)
    print("  Generating query vectors...")
    query = np.zeros((n_query, dim), dtype=np.float32)
    query_cluster_ids = np.random.randint(0, n_clusters, n_query)
    query = centers[query_cluster_ids] + np.random.randn(n_query, dim).astype(np.float32) * 0.1

    # Normalize for numerical stability
    base_norms = np.linalg.norm(base, axis=1, keepdims=True)
    base = base / np.maximum(base_norms, 1e-8)
    query_norms = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / np.maximum(query_norms, 1e-8)

    return base, query


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if real SIFT data exists, otherwise generate synthetic
    base_path = os.path.join(DATA_DIR, "sift_base.fvecs")
    query_path = os.path.join(DATA_DIR, "sift_query.fvecs")
    gt_path = os.path.join(DATA_DIR, "sift_groundtruth.ivecs")

    if os.path.exists(base_path) and os.path.exists(query_path):
        print("Loading existing SIFT data...")
        base = read_fvecs(base_path)
        query = read_fvecs(query_path)
    else:
        print("Generating synthetic data (format-compatible with SIFT1M)...")
        base, query = generate_synthetic_sift(
            n_base=100000,  # Smaller for development, use 1M for final
            n_query=1000,   # Smaller for development, use 10K for final
            dim=128,
            k=100
        )
        write_fvecs(base_path, base)
        write_fvecs(query_path, query)

    print(f"Base: {base.shape}, Query: {query.shape}")

    # Compute groundtruth
    if not os.path.exists(gt_path):
        print("Computing groundtruth...")
        gt = compute_groundtruth(base, query, k=100)
        write_ivecs(gt_path, gt)
    else:
        print("Groundtruth already exists.")
        gt = None

    print("Done!")


def generate_full_sift():
    """Generate full SIFT1M-scale synthetic data."""
    base, query = generate_synthetic_sift(
        n_base=1000000,
        n_query=10000,
        dim=128,
        k=100
    )
    base_path = os.path.join(DATA_DIR, "sift_base.fvecs")
    query_path = os.path.join(DATA_DIR, "sift_query.fvecs")
    write_fvecs(base_path, base)
    write_fvecs(query_path, query)

    print("Computing groundtruth for 1M dataset (this will take a while)...")
    gt = compute_groundtruth(base, query, k=100)
    gt_path = os.path.join(DATA_DIR, "sift_groundtruth.ivecs")
    write_ivecs(gt_path, gt)
    print("Full SIFT generation complete!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        generate_full_sift()
    else:
        main()
