from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE = Path('/Users/omer/Documents/Playground/reduced_48_feature_reanalysis')
RAW = BASE / 'tables' / 'reduced_48_features_raw.csv'
ASSIGN = BASE / 'tables' / 'reduced_48_assignments.csv'
SUMMARY = BASE / 'tables' / 'reduced_48_summary.json'
OUT = BASE / 'subcluster_analysis'
OUT.mkdir(exist_ok=True)
(OUT / 'tables').mkdir(exist_ok=True)

RNG = np.random.default_rng(42)


def silhouette_perm(X, labels, n_perm=1000):
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return {'observed': np.nan, 'null_mean': np.nan, 'null_sd': np.nan, 'p': np.nan}
    D = pairwise_distances(X)
    obs = float(silhouette_score(D, labels, metric='precomputed'))
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = RNG.permutation(len(X))
        Dp = D[np.ix_(perm, perm)]
        null[i] = silhouette_score(Dp, labels, metric='precomputed')
    return {
        'observed': obs,
        'null_mean': float(np.mean(null)),
        'null_sd': float(np.std(null, ddof=1)),
        'p': float((np.sum(null >= obs) + 1) / (n_perm + 1)),
    }


def main():
    raw = pd.read_csv(RAW)
    assign = pd.read_csv(ASSIGN)
    summary = json.loads(Path(SUMMARY).read_text())
    feature_cols = [c for c in raw.columns if c not in ['file_id', 'call_id', 'segmented_call_wav']]
    # Use only the available requested feature columns from the previous reanalysis.
    missing = set(summary.get('missing_requested_features', []))
    feature_cols = [c for c in feature_cols if c not in missing and raw[c].dtype.kind in 'biufc']
    X_all = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(raw[feature_cols]))

    meta = raw[['file_id', 'call_id']].merge(assign[['file_id','call_id','hier_best_k']], on=['file_id','call_id'], how='left')
    findings=[]
    all_rows=[]
    for cluster_id in sorted(meta['hier_best_k'].dropna().unique()):
        idx = np.where(meta['hier_best_k'].to_numpy() == cluster_id)[0]
        X = X_all[idx]
        sub_meta = meta.iloc[idx].copy().reset_index(drop=True)
        n = len(sub_meta)
        max_k = min(6, n-1)
        rows=[]
        labels_by_k={}
        for k in range(2, max_k+1):
            lab = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)
            sil = float(silhouette_score(X, lab))
            rows.append({'parent_cluster': int(cluster_id), 'method': 'hierarchical', 'k': k, 'silhouette': sil})
            labels_by_k[k]=lab
        hier_df = pd.DataFrame(rows)
        if len(hier_df):
            best_k = int(hier_df.sort_values('silhouette', ascending=False).iloc[0]['k'])
            best_lab = labels_by_k[best_k]
            perm = silhouette_perm(X, best_lab, n_perm=1000)
        else:
            best_k = 1
            best_lab = np.zeros(n, dtype=int)
            perm = {'observed': np.nan, 'null_mean': np.nan, 'null_sd': np.nan, 'p': np.nan}

        # GMM as additional check.
        gmm_rows=[]
        for k in range(1, max_k+1):
            gm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=20, reg_covar=1e-5)
            gm.fit(X)
            gmm_rows.append({'parent_cluster': int(cluster_id), 'k': k, 'bic': float(gm.bic(X)), 'aic': float(gm.aic(X))})
        gmm_df = pd.DataFrame(gmm_rows)
        gmm_best_k = int(gmm_df.sort_values('bic').iloc[0]['k']) if len(gmm_df) else 1

        out_assign = sub_meta.copy()
        out_assign['parent_cluster'] = int(cluster_id)
        out_assign['subcluster'] = best_lab
        out_assign['subcluster_label'] = [f'R48_C{int(cluster_id)+1}_S{x}' for x in best_lab]
        out_assign.to_csv(OUT / 'tables' / f'reduced48_parent_{int(cluster_id)}_subclusters.csv', index=False)
        hier_df.to_csv(OUT / 'tables' / f'reduced48_parent_{int(cluster_id)}_hier_candidates.csv', index=False)
        gmm_df.to_csv(OUT / 'tables' / f'reduced48_parent_{int(cluster_id)}_gmm_candidates.csv', index=False)
        all_rows.append(out_assign)
        findings.append({
            'parent_cluster': int(cluster_id),
            'n_calls': int(n),
            'best_hierarchical_k': int(best_k),
            'best_hierarchical_silhouette': float(perm['observed']),
            'best_hierarchical_null_mean': float(perm['null_mean']),
            'best_hierarchical_null_sd': float(perm['null_sd']),
            'best_hierarchical_perm_p': float(perm['p']),
            'gmm_best_k_by_bic': int(gmm_best_k),
        })
    all_assign = pd.concat(all_rows, ignore_index=True)
    all_assign.to_csv(OUT / 'tables' / 'reduced48_all_subcluster_assignments.csv', index=False)
    (OUT / 'tables' / 'reduced48_subcluster_summary.json').write_text(json.dumps({'findings': findings}, indent=2))
    print(json.dumps({'findings': findings}, indent=2))
    print('Saved to', OUT)

if __name__ == '__main__':
    main()
