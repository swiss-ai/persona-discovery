"""Geometry of the trait probe directions.

Cosine similarity between the probe directions of all traits (per layer, then averaged over
layers), PCA + k-means/hierarchical clustering on that similarity matrix, and a silhouette +
permutation test of whether the expert trait categories (data/traits/trait_categories.json)
are more cohesive than chance.

Reads the probe weights written by src/3_train_probes.py
($SCRATCH/apertus-pulse/data/probes/<checkpoint>/...) and writes
data/plots/probe_geometry/<checkpoint>/geometry_l1_<l1>.json.

    python -m src.4a_analyze_trait_geometry --checkpoint apertus_v1.5_8B --l1 0.0
"""
import os, json
import numpy as np
import torch
import fire
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from src.utils.paths import repo_path, data_path


def _load_w(pt_path):
    """Weight vector of a LinearProbe state dict, as a flat numpy array."""
    return torch.load(pt_path, map_location="cpu")["w"].detach().cpu().numpy().reshape(-1)


def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def main(
    checkpoint: str = "apertus_v1.5_8B",
    submodule: str = "residual.post_mlp",
    pooling: str = "mean_pooling",
    l1: float = 0.0,
    set_name: str = "cal",
    n_traits: int = 30,
    probe_dir: str = None,
    categories_path: str = None,
    all_traits_path: str = None,
    out_dir: str = None,
    n_perm: int = 20000,
    seed: int = 0,
):
    probe_dir = probe_dir or data_path("probes")
    categories_path = categories_path or repo_path("data", "traits", "trait_categories.json")
    all_traits_path = all_traits_path or repo_path("data", "traits", "all_traits.json")
    # One output folder per checkpoint.
    out_dir = out_dir or repo_path("data", "plots", "probe_geometry", checkpoint)
    probe_root = os.path.join(probe_dir, checkpoint)
    trait_ids = list(range(1, n_traits + 1))

    # Discover layers shared by every trait for this (submodule, pooling, l1) config.
    layer_sets = []
    for tid in trait_ids:
        base = os.path.join(probe_root, f"trait_{tid}", set_name, "log_probes", submodule, pooling, f"l1_{l1}")
        layers = sorted(int(f.split("_")[1].split(".")[0]) for f in os.listdir(base) if f.startswith("layer_"))
        layer_sets.append(set(layers))
    shared_layers = sorted(set.intersection(*layer_sets))
    print(f"[INFO] {len(shared_layers)} shared layers for l1={l1}: {shared_layers[:3]}...{shared_layers[-3:]}")

    vecs = {}
    for tid in trait_ids:
        base = os.path.join(probe_root, f"trait_{tid}", set_name, "log_probes", submodule, pooling, f"l1_{l1}")
        for L in shared_layers:
            vecs[(tid, L)] = _load_w(os.path.join(base, f"layer_{L}.pt"))
    print(f"[INFO] loaded probes for {len(trait_ids)} traits")

    # Per-layer cosine-similarity matrices, then mean across layers.
    mat_stack = []
    for L in shared_layers:
        M = np.zeros((n_traits, n_traits))
        for i, ti in enumerate(trait_ids):
            for j, tj in enumerate(trait_ids):
                M[i, j] = 1.0 if ti == tj else _cosine(vecs[(ti, L)], vecs[(tj, L)])
        mat_stack.append(M)
    mean_sim = np.nanmean(np.stack(mat_stack, axis=0), axis=0)
    off_diag = mean_sim[~np.eye(n_traits, dtype=bool)]
    print(f"[INFO] off-diagonal similarity: mean={off_diag.mean():.4f} min={off_diag.min():.4f} max={off_diag.max():.4f}")

    # PCA on the similarity matrix rows + k-means/hierarchical clustering.
    # The self-similarity entry is 1 by construction and says nothing about how a trait relates to
    # the others. Because it sits in a different column in every row, it acts as a trait-identity
    # feature: a full-rank identity term that raises every eigenvalue equally, flattens the spectrum
    # and inflates the apparent dimensionality. Replacing it with the off-diagonal mean makes the
    # entry neutral; zero would not be neutral, as it is a real cosine value (orthogonal to itself).
    # This is PCA preprocessing only: `mean_sim`, and the saved `mean_similarity_matrix`, keep the
    # unit diagonal, so the heatmap, the off-diagonal statistics and the distance matrix below are
    # unaffected.
    pca_input = mean_sim.copy()
    np.fill_diagonal(pca_input, off_diag.mean())
    pca = PCA(n_components=min(10, n_traits))
    pca_coords = pca.fit_transform(pca_input)
    explained = pca.explained_variance_ratio_
    cum = np.cumsum(explained)
    n_comp_cluster = max(2, min(int(np.searchsorted(cum, 0.90)) + 1, pca_coords.shape[1]))
    cluster_features = pca_coords[:, :n_comp_cluster]
    print(f"[INFO] PCA explained variance (first 3): {explained[:3]}; clustering on {n_comp_cluster} components")

    # k-means can end in different local optima on these features; many initialisations (n_init=100)
    # make the best-inertia solution much more likely to be found, without guaranteeing the global
    # optimum. Ward clustering is deterministic.
    kmeans = KMeans(n_clusters=4, n_init=100, random_state=seed).fit(cluster_features)
    agglo = AgglomerativeClustering(n_clusters=4).fit(cluster_features)

    # Ground-truth categories.
    cat_data = json.load(open(categories_path))
    cat_map = cat_data["trait_categories"]
    categories_ordered = [cat_map[str(tid)] for tid in trait_ids]
    cat_names = sorted(set(categories_ordered))
    cat_to_idx = {c: i for i, c in enumerate(cat_names)}
    cat_labels = np.array([cat_to_idx[c] for c in categories_ordered])

    ari_kmeans = adjusted_rand_score(cat_labels, kmeans.labels_)
    ari_agglo = adjusted_rand_score(cat_labels, agglo.labels_)
    print(f"[INFO] ARI vs categories: kmeans={ari_kmeans:.4f} hierarchical={ari_agglo:.4f}")

    # Silhouette of the expert categories on cosine distance, tested by permutation.
    dist = 1.0 - mean_sim
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    observed_sil = silhouette_score(dist, cat_labels, metric="precomputed")
    per_trait_sil = silhouette_samples(dist, cat_labels, metric="precomputed")
    per_cat_mean = {c: float(np.mean(per_trait_sil[[i for i in range(n_traits) if categories_ordered[i] == c]])) for c in cat_names}

    rng = np.random.default_rng(seed)
    null_scores = np.empty(n_perm)
    for i in range(n_perm):
        perm_labels = rng.permutation(cat_labels)
        null_scores[i] = silhouette_score(dist, perm_labels, metric="precomputed")
    # Finite-permutation p-value, (b+1)/(m+1) rather than the naive proportion. The naive form
    # understates p by ~1/n_perm and can return exactly zero, which is not a possible p-value for a
    # randomly drawn permutation set (Phipson & Smyth 2010). One-sided, because the hypothesis is
    # directional: traits of the same category are more cohesive than chance.
    p_value = float((np.sum(null_scores >= observed_sil) + 1) / (n_perm + 1))
    print(f"[INFO] silhouette (true categories): {observed_sil:.4f}; null mean={null_scores.mean():.4f} "
          f"std={null_scores.std():.4f}; p={p_value:.4f}")

    all_traits = json.load(open(all_traits_path))["therapist_traits"]
    name_map = {t["trait_id"]: t["trait_name_pos"] for t in all_traits}
    trait_names = [name_map[tid] for tid in trait_ids]

    result = {
        "checkpoint": checkpoint,
        "config": {"submodule": submodule, "pooling": pooling, "l1": l1, "set_name": set_name, "n_layers": len(shared_layers)},
        "trait_ids": trait_ids,
        "trait_names": trait_names,
        "categories": categories_ordered,
        "category_names": cat_names,
        "mean_similarity_matrix": mean_sim.tolist(),
        "pca_explained_variance": explained.tolist(),
        "pca_coords_2d": pca_coords[:, :2].tolist(),
        "kmeans_labels": kmeans.labels_.tolist(),
        "agglo_labels": agglo.labels_.tolist(),
        "ari_kmeans": ari_kmeans,
        "ari_agglo": ari_agglo,
        "n_comp_used_for_clustering": n_comp_cluster,
        "silhouette": {
            "observed": observed_sil,
            "null_mean": float(null_scores.mean()),
            "null_std": float(null_scores.std()),
            "p_value": p_value,
            "n_permutations": n_perm,
            "per_category_mean": per_cat_mean,
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"geometry_l1_{l1}.json")
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"[INFO] wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
