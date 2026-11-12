"""All 18 unsupervised fit-only runs and permutation/stability evaluations."""
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from common import save_csv, write_json


def entropy_mi(groups, labels, k, n_labels=12):
    counts = np.zeros((k, n_labels), dtype=float)
    np.add.at(counts, (groups, labels), 1)
    joint = counts / counts.sum()
    pg = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    mi = np.sum(joint[nonzero] * np.log(joint[nonzero] / (pg @ py)[nonzero]))
    positive = py[py > 0]
    return float(-np.sum(positive * np.log(positive))), float(mi)


def fit_groups(root, features, fit, y, config):
    z = PCA(n_components=20, svd_solver="full", random_state=42).fit(features)
    space = z.transform(features)
    joblib.dump(z, root / "artifacts/pca.joblib")
    assignments, quality, shuffle_rows, fitted = [], [], [], {}
    for method in ["kmeans", "gmm"]:
        for k in [6, 12, 18]:
            results = {}
            for seed in [0, 1, 2]:
                if method == "kmeans":
                    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(space)
                    groups = model.labels_
                    representative_score = -np.linalg.norm(space - model.cluster_centers_[groups], axis=1)
                else:
                    model = GaussianMixture(n_components=k, covariance_type="diag", n_init=2,
                                            reg_covar=config["gmm_reg_covar"], random_state=seed,
                                            max_iter=300).fit(space)
                    membership = model.predict_proba(space)
                    groups = membership.argmax(axis=1)
                    representative_score = membership.max(axis=1)
                results[seed] = groups
                fitted[f"{method}_{k}_{seed}"] = model
                run_id = f"{method}_k{k}_seed{seed}"
                for sid, label, group, score in zip(fit.source_id, fit.category, groups, representative_score):
                    assignments.append({"source_id": sid, "category": label, "split": "fit", "method": method,
                                        "k": k, "seed": seed, "cluster_id": int(group),
                                        "representative_score": float(score), "run_id": run_id})
            aris = {f"ari_{a}_{b}": float(adjusted_rand_score(results[a], results[b]))
                    for a, b in combinations([0, 1, 2], 2)}
            entropy, mi = entropy_mi(results[0], y, k)
            rng = np.random.default_rng(42)
            shuffled = [entropy_mi(results[0], rng.permutation(y), k)[1] for _ in range(100)]
            for i, value in enumerate(shuffled):
                shuffle_rows.append({"method": method, "k": k, "iteration": i, "mutual_information": value,
                                     "run_id": f"{method}_k{k}_seed0", "shuffle_seed": 42})
            quality.append({"method": method, "k": k, **aris, "mean_ari": np.mean(list(aris.values())),
                            "entropy_y": entropy, "mutual_information": mi,
                            "shuffled_mean": np.mean(shuffled), "shuffled_low": np.percentile(shuffled, 2.5),
                            "shuffled_high": np.percentile(shuffled, 97.5), "shuffle_count": 100,
                            "shuffle_seed": 42, "run_id": f"{method}_k{k}_seed0"})
            print(f"Grouping {method} k={k}: MI={mi:.3f}, ARI={np.mean(list(aris.values())):.3f}", flush=True)
    save_csv(root / "results/clusters.csv", assignments)
    save_csv(root / "results/clustering_results.csv", quality)
    save_csv(root / "results/clustering_shuffles.csv", shuffle_rows)
    joblib.dump(fitted, root / "artifacts/clustering.joblib")
    # Fixed fit-only criterion: stability dominates; association is scaled by H(Y).
    best = max(quality, key=lambda q: q["mean_ari"] + .15 * q["mutual_information"] / q["entropy_y"]
               - .005 * abs(q["k"] - 12))
    default = {"method": best["method"], "k": best["k"], "seed": 0,
               "criterion": "mean_ari + 0.15*MI/H(Y) - 0.005*abs(k-12); fit data only"}
    write_json(root / "results/grouping_default.json", default)
    return default
