#!/usr/bin/env python3
"""Probe hidden substructure inside the two primary acoustic clusters."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def remap_labels(labels: np.ndarray) -> np.ndarray:
    unique = sorted(set(int(x) for x in labels if int(x) != -1))
    mapping = {u: i for i, u in enumerate(unique)}
    return np.array([mapping.get(int(x), -1) for x in labels], dtype=int)


def safe_k(labels: np.ndarray) -> int:
    return len({int(x) for x in labels if int(x) != -1})


def pairwise_dist(X: np.ndarray) -> np.ndarray:
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))


def permutation_silhouette(X: np.ndarray, labels: np.ndarray, n_perm: int = 1000, seed: int = 42) -> dict[str, float]:
    if safe_k(labels) < 2:
        return {
            "observed_silhouette": np.nan,
            "null_mean": np.nan,
            "null_sd": np.nan,
            "p_value": np.nan,
        }
    rng = np.random.default_rng(seed)
    D = pairwise_dist(X)
    observed = float(silhouette_score(D, labels, metric="precomputed"))
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(len(X))
        Dp = D[np.ix_(perm, perm)]
        null[i] = silhouette_score(Dp, labels, metric="precomputed")
    return {
        "observed_silhouette": observed,
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null, ddof=1)),
        "p_value": float((np.sum(null >= observed) + 1) / (n_perm + 1)),
    }


def evaluate_cluster(X: np.ndarray, k_values: list[int], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    gmm_rows = []
    hier_rows = []
    for k in k_values:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=seed,
            n_init=10,
        ).fit(X)
        gmm_labels = remap_labels(gmm.predict(X))
        gmm_rows.append(
            {
                "k": k,
                "bic": float(gmm.bic(X)),
                "aic": float(gmm.aic(X)),
                "silhouette": float(silhouette_score(X, gmm_labels)) if k > 1 else np.nan,
            }
        )
        hier = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
        hier_labels = remap_labels(hier.labels_)
        hier_rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(X, hier_labels)) if k > 1 else np.nan,
            }
        )
    return pd.DataFrame(gmm_rows), pd.DataFrame(hier_rows)


def subcluster_report(base_dir: Path) -> dict:
    sns.set_theme(style="whitegrid")
    out_dir = base_dir / "subcluster_analysis"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(base_dir / "features" / "call_features_cleaned_standardized.csv")
    assignments = pd.read_csv(base_dir / "stats" / "cluster_assignments.csv")
    pca = pd.read_csv(base_dir / "features" / "pca_scores.csv")

    df = features.merge(assignments, on=["file_id", "call_id"]).merge(pca, on=["file_id", "call_id"])
    feature_cols = [c for c in features.columns if c not in {"file_id", "call_id", "segmented_call_wav"}]
    df["day"] = df["file_id"].str.extract(r"(\d{4}-\d{2}-\d{2})")

    overall = {
        "n_calls_total": int(len(df)),
        "primary_cluster_sizes": df["cluster_primary_conservative"].value_counts().sort_index().to_dict(),
        "subcluster_findings": [],
    }

    for primary_cluster, sub in df.groupby("cluster_primary_conservative"):
        X = sub[feature_cols].to_numpy(dtype=float)
        n = len(sub)
        k_values = list(range(2, min(5, n - 1) + 1))
        finding = {
            "primary_cluster": int(primary_cluster),
            "n_calls": int(n),
            "days": sorted(set(sub["day"])),
            "files": int(sub["file_id"].nunique()),
        }
        if n < 8 or not k_values:
            finding["status"] = "too_small_for_subclustering"
            overall["subcluster_findings"].append(finding)
            continue

        gmm_df, hier_df = evaluate_cluster(X, k_values, seed=42)
        gmm_df.to_csv(out_dir / f"cluster_{primary_cluster}_gmm_candidates.csv", index=False)
        hier_df.to_csv(out_dir / f"cluster_{primary_cluster}_hier_candidates.csv", index=False)

        best_gmm_k = int(gmm_df.sort_values(["bic", "silhouette"], ascending=[True, False]).iloc[0]["k"])
        best_hier_k = int(hier_df.sort_values("silhouette", ascending=False).iloc[0]["k"])
        finding["best_gmm_k"] = best_gmm_k
        finding["best_hier_k"] = best_hier_k

        gmm = GaussianMixture(
            n_components=best_gmm_k,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=42,
            n_init=10,
        ).fit(X)
        gmm_labels = remap_labels(gmm.predict(X))
        hier_labels = remap_labels(AgglomerativeClustering(n_clusters=best_hier_k, linkage="ward").fit_predict(X))
        hdb = HDBSCAN(min_cluster_size=max(2, min(4, n // 3)), min_samples=1, allow_single_cluster=True).fit(X)
        hdb_labels = remap_labels(hdb.labels_)

        finding["hdbscan_k"] = safe_k(hdb_labels)
        finding["gmm_silhouette"] = float(silhouette_score(X, gmm_labels)) if safe_k(gmm_labels) > 1 else np.nan
        finding["hier_silhouette"] = float(silhouette_score(X, hier_labels)) if safe_k(hier_labels) > 1 else np.nan

        # Choose a non-conservative "best hidden structure" only if both methods split.
        if best_gmm_k >= 2 and best_hier_k >= 2:
            chosen_labels = gmm_labels if finding["gmm_silhouette"] >= finding["hier_silhouette"] else hier_labels
            chosen_method = "gmm" if finding["gmm_silhouette"] >= finding["hier_silhouette"] else "hierarchical"
            perm = permutation_silhouette(X, chosen_labels, n_perm=1000, seed=42)
            finding["chosen_subcluster_method"] = chosen_method
            finding["chosen_subcluster_k"] = int(safe_k(chosen_labels))
            finding["chosen_subcluster_silhouette"] = perm["observed_silhouette"]
            finding["chosen_subcluster_silhouette_p"] = perm["p_value"]

            sub_assign = sub[["file_id", "call_id", "day", "PC1", "PC2", "PC3"]].copy()
            sub_assign["subcluster"] = chosen_labels
            sub_assign.to_csv(out_dir / f"cluster_{primary_cluster}_subcluster_assignments.csv", index=False)

            # Day composition
            day_tab = pd.crosstab(sub_assign["subcluster"], sub_assign["day"])
            day_tab.to_csv(out_dir / f"cluster_{primary_cluster}_subcluster_by_day.csv")
            day_prop = day_tab.div(day_tab.sum(axis=1), axis=0).fillna(0.0)
            max_day_dom = float(day_prop.max(axis=1).max()) if len(day_prop) else np.nan
            finding["max_day_dominance"] = max_day_dom

            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            sns.heatmap(day_prop, annot=True, fmt=".2f", cmap="viridis", ax=ax)
            ax.set_title(f"Primary cluster {primary_cluster}: subcluster-by-day")
            fig.savefig(fig_dir / f"cluster_{primary_cluster}_subcluster_by_day.png", dpi=220)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
            sns.scatterplot(data=sub_assign, x="PC1", y="PC2", hue="subcluster", style="day", s=70, ax=ax)
            ax.set_title(f"Primary cluster {primary_cluster}: PCA substructure")
            fig.savefig(fig_dir / f"cluster_{primary_cluster}_pca_subclusters.png", dpi=220)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
            ax.plot(gmm_df["k"], gmm_df["bic"], marker="o", label="GMM BIC")
            ax2 = ax.twinx()
            ax2.plot(hier_df["k"], hier_df["silhouette"], marker="s", color="tab:orange", label="Hier silhouette")
            ax.set_xlabel("Candidate subcluster number")
            ax.set_title(f"Primary cluster {primary_cluster}: hidden structure search")
            ax.set_ylabel("BIC")
            ax2.set_ylabel("Silhouette")
            fig.savefig(fig_dir / f"cluster_{primary_cluster}_model_selection.png", dpi=220)
            plt.close(fig)

            # Dendrogram
            link = hierarchy.linkage(X, method="ward")
            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            hierarchy.dendrogram(link, labels=sub["call_id"].tolist(), leaf_rotation=90, leaf_font_size=6, ax=ax)
            ax.set_title(f"Primary cluster {primary_cluster}: dendrogram")
            fig.savefig(fig_dir / f"cluster_{primary_cluster}_dendrogram.png", dpi=220)
            plt.close(fig)
        else:
            finding["chosen_subcluster_method"] = "none"
            finding["chosen_subcluster_k"] = 1
            finding["chosen_subcluster_silhouette"] = np.nan
            finding["chosen_subcluster_silhouette_p"] = np.nan
            finding["max_day_dominance"] = np.nan

        overall["subcluster_findings"].append(finding)

    # Estimate a softer upper bound by summing chosen hidden structure inside primary clusters.
    upper_bound = 0
    for finding in overall["subcluster_findings"]:
        upper_bound += int(max(1, finding.get("chosen_subcluster_k", 1)))
    overall["nonconservative_upper_bound_from_subclusters"] = int(upper_bound)

    summary_lines = [
        "Within-primary-cluster substructure analysis",
        "",
        f"- Total calls analyzed: {overall['n_calls_total']}",
        f"- Primary cluster sizes: {overall['primary_cluster_sizes']}",
        f"- Non-conservative upper bound from hidden substructure search: {overall['nonconservative_upper_bound_from_subclusters']}",
        "",
    ]
    for finding in overall["subcluster_findings"]:
        summary_lines.append(
            f"- Primary cluster {finding['primary_cluster']}: n={finding['n_calls']}, "
            f"best GMM k={finding.get('best_gmm_k', 'NA')}, best hierarchical k={finding.get('best_hier_k', 'NA')}, "
            f"chosen hidden subclusters={finding.get('chosen_subcluster_k', 'NA')}, "
            f"p={finding.get('chosen_subcluster_silhouette_p', np.nan)}"
        )
    (out_dir / "subcluster_summary.txt").write_text("\n".join(summary_lines))
    with open(out_dir / "subcluster_summary.json", "w") as f:
        json.dump(overall, f, indent=2)
    return overall


if __name__ == "__main__":
    base_dir = Path("/Users/omer/Documents/Playground/bat_alarm_signature_pipeline_output")
    result = subcluster_report(base_dir)
    print(json.dumps(result, indent=2))
