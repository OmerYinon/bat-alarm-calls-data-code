from __future__ import annotations

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal, stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import pairwise_distances

BASE = Path('/Users/omer/Documents/Playground/bat_alarm_signature_pipeline_output')
OUT = Path('/Users/omer/Documents/Playground/reduced_48_feature_reanalysis')
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'tables').mkdir(exist_ok=True)
(OUT / 'figures').mkdir(exist_ok=True)

FEATURE_XLSX = Path('/Users/omer/Desktop/Book1.xlsx')
SEGMENTS_CSV = BASE / 'stats' / 'call_segments_final.csv'
PRIMARY_CSV = BASE / 'stats' / 'cluster_assignments.csv'
SUB0_CSV = BASE / 'subcluster_analysis' / 'cluster_0_subcluster_assignments.csv'
SUB1_CSV = BASE / 'subcluster_analysis' / 'cluster_1_subcluster_assignments.csv'

FMIN = 335.0
FMAX = 10095.0
RNG = np.random.default_rng(42)


def feature_list() -> list[str]:
    x = pd.read_excel(FEATURE_XLSX)
    # The first feature (Duration) is the column header in the user-supplied file.
    feats = [str(x.columns[0]).strip()]
    feats.extend(x.iloc[:, 0].dropna().astype(str).str.strip().tolist())
    return feats


def read_wav(path: str) -> tuple[int, np.ndarray]:
    sr, y = wavfile.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / max(1.0, np.iinfo(y.dtype).max)
    else:
        y = y.astype(np.float32)
    return int(sr), y


def frame_spectrum_features(freqs: np.ndarray, spec: np.ndarray, suffix: str) -> dict[str, float]:
    spec = np.asarray(spec, dtype=float)
    spec = np.maximum(spec, 0.0)
    if spec.size == 0 or not np.isfinite(spec).any() or float(np.sum(spec)) <= 0:
        base = {
            f'Peak freq({suffix})': np.nan,
            f'Fundamental({suffix})': np.nan,
            f'Min freq({suffix})': np.nan,
            f'Max freq({suffix})': np.nan,
            f'Bandw({suffix})': np.nan,
            f'Quart25({suffix})': np.nan,
            f'Quart50({suffix})': np.nan,
            f'Quart75({suffix})': np.nan,
            f'Entropy({suffix})': np.nan,
            f'HNR({suffix})': np.nan,
            f'Peaks({suffix})': np.nan,
        }
        for i in range(1, 6):
            base[f'Freq {i}({suffix})'] = np.nan
        return base

    total = float(np.sum(spec))
    peak_idx = int(np.argmax(spec))
    peak_freq = float(freqs[peak_idx])
    peak_val = float(spec[peak_idx])

    active = spec >= (0.10 * peak_val)
    if np.any(active):
        min_freq = float(freqs[np.where(active)[0][0]])
        max_freq = float(freqs[np.where(active)[0][-1]])
    else:
        min_freq = peak_freq
        max_freq = peak_freq

    csum = np.cumsum(spec) / total
    q25 = float(freqs[np.searchsorted(csum, 0.25, side='left')])
    q50 = float(freqs[np.searchsorted(csum, 0.50, side='left')])
    q75 = float(freqs[np.searchsorted(csum, 0.75, side='left')])

    p = spec / total
    entropy = float(-(p * np.log2(p + 1e-12)).sum() / math.log2(len(p)))

    peaks, props = signal.find_peaks(spec, height=0.15 * peak_val, distance=2)
    if peaks.size == 0:
        peaks = np.array([peak_idx])
    heights = spec[peaks]
    order = np.argsort(heights)[::-1]
    peaks_by_amp = peaks[order]
    top_energy = float(np.sum(spec[peaks_by_amp[: min(5, len(peaks_by_amp))]]))
    noise_energy = max(total - top_energy, 1e-12)
    hnr = float(10.0 * np.log10((top_energy + 1e-12) / noise_energy))

    peak_freqs_by_amp = [float(freqs[i]) for i in peaks_by_amp[:5]]
    fundamental = float(min(peak_freqs_by_amp)) if peak_freqs_by_amp else np.nan

    out = {
        f'Peak freq({suffix})': peak_freq,
        f'Fundamental({suffix})': fundamental,
        f'Min freq({suffix})': min_freq,
        f'Max freq({suffix})': max_freq,
        f'Bandw({suffix})': max_freq - min_freq,
        f'Quart25({suffix})': q25,
        f'Quart50({suffix})': q50,
        f'Quart75({suffix})': q75,
        f'Entropy({suffix})': entropy,
        f'HNR({suffix})': hnr,
        f'Peaks({suffix})': float(len(peaks)),
    }
    for i in range(1, 6):
        out[f'Freq {i}({suffix})'] = peak_freqs_by_amp[i - 1] if len(peak_freqs_by_amp) >= i else np.nan
    return out


def extract_for_segment(row: pd.Series) -> dict[str, float | str]:
    sr, y = read_wav(row['segmented_call_wav'])
    duration = len(y) / sr if sr else np.nan
    out = {
        'file_id': row['file_id'],
        'call_id': row['call_id'],
        'Duration': float(duration),
        'Start time': float(row['start_time']),
        'End time': float(row['end_time']),
    }
    if len(y) < 64:
        return out

    nperseg = min(512, max(128, int(2 ** np.floor(np.log2(max(64, len(y) // 2))))))
    noverlap = int(nperseg * 0.75)
    freqs, times, S = signal.spectrogram(
        y,
        fs=sr,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling='spectrum',
        mode='magnitude',
    )
    mask = (freqs >= FMIN) & (freqs <= min(FMAX, sr / 2 - 1))
    freqs = freqs[mask]
    S = S[mask]
    if S.size == 0 or S.shape[1] == 0:
        return out

    frame_energy = np.sum(S, axis=0)
    start_idx = 0
    end_idx = S.shape[1] - 1
    max_idx = int(np.argmax(frame_energy))

    out.update(frame_spectrum_features(freqs, S[:, start_idx], 'start'))
    out.update(frame_spectrum_features(freqs, S[:, end_idx], 'end'))
    out.update(frame_spectrum_features(freqs, S[:, max_idx], 'max'))
    return out


def silhouette_permutation(X: np.ndarray, labels: np.ndarray, n_perm: int = 1000) -> dict[str, float]:
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return {'observed': np.nan, 'null_mean': np.nan, 'null_sd': np.nan, 'p': np.nan}
    D = pairwise_distances(X)
    observed = float(silhouette_score(D, labels, metric='precomputed'))
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = RNG.permutation(len(X))
        Dp = D[np.ix_(perm, perm)]
        null[i] = silhouette_score(Dp, labels, metric='precomputed')
    return {
        'observed': observed,
        'null_mean': float(np.mean(null)),
        'null_sd': float(np.std(null, ddof=1)),
        'p': float((np.sum(null >= observed) + 1) / (n_perm + 1)),
    }


def main() -> None:
    requested = feature_list()
    print('Requested features:', len(requested))
    seg = pd.read_csv(SEGMENTS_CSV)
    rows = []
    for _, row in seg.iterrows():
        try:
            rows.append(extract_for_segment(row))
        except Exception as exc:
            warnings.warn(f'Failed {row.get("call_id")}: {exc}')
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / 'tables' / 'reduced_48_features_raw.csv', index=False)

    available = [f for f in requested if f in raw.columns]
    missing = [f for f in requested if f not in raw.columns]
    print('Available requested features:', len(available))
    print('Missing requested features:', missing)

    X_raw = raw[available].copy()
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(X_raw))

    pca = PCA(n_components=min(10, X.shape[0], X.shape[1]), random_state=42)
    pcs = pca.fit_transform(X)
    pca_df = raw[['file_id', 'call_id']].copy()
    for i in range(min(3, pcs.shape[1])):
        pca_df[f'PC{i+1}'] = pcs[:, i]
    pca_df.to_csv(OUT / 'tables' / 'reduced_48_pca_scores.csv', index=False)

    # Candidate hierarchical solutions.
    cand = []
    labels_by_k = {}
    for k in range(2, min(9, len(raw) - 1)):
        lab = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)
        sil = float(silhouette_score(X, lab))
        cand.append({'k': k, 'silhouette': sil})
        labels_by_k[k] = lab
    cand_df = pd.DataFrame(cand)
    best_k = int(cand_df.sort_values('silhouette', ascending=False).iloc[0]['k'])
    best_lab = labels_by_k[best_k]

    # GMM BIC/AIC.
    gmm_rows = []
    for k in range(1, min(9, len(raw) - 1)):
        gm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=20, reg_covar=1e-5)
        gm.fit(X)
        gmm_rows.append({'k': k, 'bic': float(gm.bic(X)), 'aic': float(gm.aic(X))})
    gmm_df = pd.DataFrame(gmm_rows)
    gmm_best_k = int(gmm_df.sort_values('bic').iloc[0]['k'])

    # Previous labels comparison.
    primary = pd.read_csv(PRIMARY_CSV)
    merged_primary = raw[['file_id','call_id']].merge(primary, on=['file_id','call_id'], how='left')
    prev_primary = merged_primary['cluster_primary_conservative'].to_numpy()

    sub0 = pd.read_csv(SUB0_CSV)
    sub1 = pd.read_csv(SUB1_CSV)
    sub0['prev_subcluster_5'] = 'P0_S' + sub0['subcluster'].astype(str)
    sub1['prev_subcluster_5'] = 'P1_S' + sub1['subcluster'].astype(str)
    prev_sub = pd.concat([sub0[['file_id','call_id','prev_subcluster_5']], sub1[['file_id','call_id','prev_subcluster_5']]], ignore_index=True)
    merged_sub = raw[['file_id','call_id']].merge(prev_sub, on=['file_id','call_id'], how='left')
    prev5 = pd.factorize(merged_sub['prev_subcluster_5'])[0]

    prev_primary_sil = silhouette_permutation(X, prev_primary, n_perm=1000)
    prev5_sil = silhouette_permutation(X, prev5, n_perm=1000)
    best_sil = silhouette_permutation(X, best_lab, n_perm=1000)

    ari_primary_vs_best = float(adjusted_rand_score(prev_primary, best_lab))
    ari_prev5_vs_best = float(adjusted_rand_score(prev5, best_lab))

    assignments = raw[['file_id', 'call_id']].copy()
    assignments['hier_best_k'] = best_lab
    assignments['prev_primary_2'] = prev_primary
    assignments['prev_subcluster_5'] = merged_sub['prev_subcluster_5']
    assignments.to_csv(OUT / 'tables' / 'reduced_48_assignments.csv', index=False)
    cand_df.to_csv(OUT / 'tables' / 'reduced_48_hierarchical_candidates.csv', index=False)
    gmm_df.to_csv(OUT / 'tables' / 'reduced_48_gmm_candidates.csv', index=False)

    summary = {
        'requested_features': len(requested),
        'available_requested_features': len(available),
        'missing_requested_features': missing,
        'n_calls': int(len(raw)),
        'pca_variance_pc1': float(pca.explained_variance_ratio_[0]),
        'pca_variance_pc2': float(pca.explained_variance_ratio_[1]),
        'pca_variance_pc3': float(pca.explained_variance_ratio_[2]),
        'hierarchical_best_k_by_silhouette': best_k,
        'hierarchical_best_silhouette': float(cand_df.sort_values('silhouette', ascending=False).iloc[0]['silhouette']),
        'gmm_best_k_by_bic': gmm_best_k,
        'previous_primary_2_silhouette_test': prev_primary_sil,
        'previous_5_subclusters_silhouette_test': prev5_sil,
        'new_best_hierarchical_silhouette_test': best_sil,
        'ari_previous_primary2_vs_new_best': ari_primary_vs_best,
        'ari_previous_subcluster5_vs_new_best': ari_prev5_vs_best,
    }
    (OUT / 'tables' / 'reduced_48_summary.json').write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print('Saved to', OUT)

if __name__ == '__main__':
    main()
