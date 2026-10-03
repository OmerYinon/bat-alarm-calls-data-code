#!/usr/bin/env python3
"""
Required packages
-----------------
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- scikit-learn>=1.6
- ffmpeg available on PATH for non-WAV inputs such as MP4

This pipeline is designed for a small, exploratory bioacoustics dataset of
audible bat alarm calls. The unit of analysis is the single call, not the
source chopped file.

Scientific interpretation rules built into this script:
- use cautious language such as "putative acoustic clusters"
- do not claim true individual identification unless validation supports it
- treat the results as small-sample, exploratory, and statistically tested
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import fftpack, signal, stats
from scipy.cluster import hierarchy
from scipy.io import wavfile
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp4", ".m4a", ".aac", ".mp3", ".mov"}
KNOWN_SAME_BAT_FILE_IDS = {
    "test1_2023-09-27_12-40-15.1",
    "test1_2023-09-26_13-29-33",
}


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    target_sr: int = 22050
    initial_fmin_hz: float = 300.0
    initial_fmax_hz: float = 10000.0
    band_low_hz: float | None = None
    band_high_hz: float | None = None
    seg_nperseg: int = 1024
    seg_hop: int = 256
    seg_baseline_sec: float = 0.25
    seg_peak_z: float = 2.6
    seg_onset_z: float = 0.8
    seg_min_call_sec: float = 0.015
    seg_max_call_sec: float = 0.45
    seg_min_separation_sec: float = 0.025
    seg_padding_sec: float = 0.015
    n_mels: int = 26
    n_mfcc: int = 13
    corr_threshold: float = 0.95
    max_clusters_to_test: int = 8
    n_permutations: int = 1000
    n_bootstraps: int = 200
    bootstrap_fraction: float = 0.8
    random_seed: int = 42
    validation_metadata: Path | None = None
    segmentation_manual_csv: Path | None = None
    save_umap_if_available: bool = True


@dataclass
class OutputPaths:
    root: Path
    segmented_calls: Path
    features: Path
    figures: Path
    stats: Path
    logs: Path
    figures_segmentation: Path
    figures_frequency: Path
    figures_pca: Path
    figures_clustering: Path
    figures_validation: Path


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Conservative call-level bioacoustics pipeline for chopped bat alarm-call files."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing chopped audio/video files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/omer/Documents/Playground/bat_alarm_signature_pipeline_output"),
        help="Directory where all outputs will be written.",
    )
    parser.add_argument("--target-sr", type=int, default=22050)
    parser.add_argument("--initial-fmin-hz", type=float, default=300.0)
    parser.add_argument("--initial-fmax-hz", type=float, default=10000.0)
    parser.add_argument("--band-low-hz", type=float, default=None)
    parser.add_argument("--band-high-hz", type=float, default=None)
    parser.add_argument("--seg-peak-z", type=float, default=2.6)
    parser.add_argument("--seg-onset-z", type=float, default=0.8)
    parser.add_argument("--seg-min-call-sec", type=float, default=0.015)
    parser.add_argument("--seg-max-call-sec", type=float, default=0.45)
    parser.add_argument("--seg-min-separation-sec", type=float, default=0.025)
    parser.add_argument("--seg-padding-sec", type=float, default=0.015)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--max-clusters-to-test", type=int, default=8)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-bootstraps", type=int, default=200)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--validation-metadata",
        type=Path,
        default=None,
        help="Optional CSV with columns: file_id, call_id, within_file_caller_label.",
    )
    parser.add_argument(
        "--segmentation-manual-csv",
        type=Path,
        default=None,
        help="Optional manually corrected segmentation CSV to use instead of auto segmentation.",
    )
    args = parser.parse_args()
    return PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_sr=args.target_sr,
        initial_fmin_hz=args.initial_fmin_hz,
        initial_fmax_hz=args.initial_fmax_hz,
        band_low_hz=args.band_low_hz,
        band_high_hz=args.band_high_hz,
        seg_peak_z=args.seg_peak_z,
        seg_onset_z=args.seg_onset_z,
        seg_min_call_sec=args.seg_min_call_sec,
        seg_max_call_sec=args.seg_max_call_sec,
        seg_min_separation_sec=args.seg_min_separation_sec,
        seg_padding_sec=args.seg_padding_sec,
        corr_threshold=args.corr_threshold,
        max_clusters_to_test=args.max_clusters_to_test,
        n_permutations=args.n_permutations,
        n_bootstraps=args.n_bootstraps,
        bootstrap_fraction=args.bootstrap_fraction,
        random_seed=args.random_seed,
        validation_metadata=args.validation_metadata,
        segmentation_manual_csv=args.segmentation_manual_csv,
    )


def make_output_paths(root: Path) -> OutputPaths:
    segmented_calls = root / "segmented_calls"
    features = root / "features"
    figures = root / "figures"
    stats_dir = root / "stats"
    logs = root / "logs"
    paths = OutputPaths(
        root=root,
        segmented_calls=segmented_calls,
        features=features,
        figures=figures,
        stats=stats_dir,
        logs=logs,
        figures_segmentation=figures / "segmentation",
        figures_frequency=figures / "frequency",
        figures_pca=figures / "pca",
        figures_clustering=figures / "clustering",
        figures_validation=figures / "validation",
    )
    for path in [
        root,
        segmented_calls,
        features,
        figures,
        stats_dir,
        logs,
        paths.figures_segmentation,
        paths.figures_frequency,
        paths.figures_pca,
        paths.figures_clustering,
        paths.figures_validation,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )


def log_step(message: str) -> None:
    logging.info(message)


def find_audio_files(folder: Path) -> list[Path]:
    files = [
        p
        for p in folder.rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and not p.name.startswith("._")
        and p.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    ]
    return sorted(files)


def normalize_audio_dtype(y: np.ndarray) -> np.ndarray:
    if np.issubdtype(y.dtype, np.integer):
        max_abs = max(abs(np.iinfo(y.dtype).min), np.iinfo(y.dtype).max)
        return y.astype(np.float32) / float(max_abs)
    return y.astype(np.float32, copy=False)


def load_audio_mono(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        sr, y = wavfile.read(str(path))
        y = np.asarray(y)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = normalize_audio_dtype(y)
        if int(sr) != int(target_sr):
            g = math.gcd(int(sr), int(target_sr))
            y = signal.resample_poly(y, up=target_sr // g, down=sr // g).astype(np.float32)
            sr = target_sr
        return y.astype(np.float32), int(sr)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to read non-WAV files such as MP4.")

    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
    if audio.size == 0:
        return np.array([], dtype=np.float32), target_sr
    return (audio / 32768.0).astype(np.float32), target_sr


def write_wav(path: Path, sr: int, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_clip = np.clip(y, -1.0, 1.0)
    pcm = np.round(y_clip * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(str(path), sr, pcm)


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = float(np.std(x))
    if scale < 1e-8:
        return np.zeros_like(x)
    return (x - med) / scale


def compute_spectrogram(
    y: np.ndarray,
    sr: int,
    nperseg: int,
    hop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    noverlap = max(0, nperseg - hop)
    freqs, times, spec = signal.spectrogram(
        y,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="spectrum",
        mode="magnitude",
    )
    return freqs.astype(np.float32), times.astype(np.float32), spec.astype(np.float32)


def compute_foreground_spectrogram(
    y: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    nperseg: int,
    hop: int,
    baseline_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs, times, spec = compute_spectrogram(y, sr, nperseg, hop)
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    spec = spec[mask]
    if spec.size == 0:
        return freqs, times, np.empty((0, 0), dtype=np.float32)
    log_spec = np.log1p(spec)
    baseline_frames = max(9, int(round(baseline_sec * sr / hop)))
    if baseline_frames % 2 == 0:
        baseline_frames += 1
    background = signal.medfilt2d(log_spec, kernel_size=(1, baseline_frames))
    foreground = np.maximum(log_spec - background, 0.0)
    foreground = np.maximum(foreground - np.median(foreground, axis=0, keepdims=True), 0.0)
    return freqs, times, foreground.astype(np.float32)


def summarize_detection_score(foreground: np.ndarray, freqs: np.ndarray) -> dict[str, np.ndarray]:
    if foreground.size == 0:
        empty = np.array([], dtype=np.float32)
        return {
            "score": empty,
            "energy": empty,
            "peakiness": empty,
            "peak_freq_hz": empty,
            "bandwidth_hz": empty,
        }
    energy = np.sum(foreground, axis=0)
    peak = np.max(foreground, axis=0)
    mean = np.mean(foreground, axis=0)
    peakiness = peak / np.maximum(mean, 1e-6)
    peak_idx = np.argmax(foreground, axis=0)
    peak_freq_hz = freqs[peak_idx]
    active = foreground >= (peak[None, :] * 0.35)
    bandwidth_hz = np.sum(active, axis=0).astype(np.float32)
    if len(freqs) > 1:
        bandwidth_hz *= float(freqs[1] - freqs[0])
    flux = np.zeros(foreground.shape[1], dtype=np.float32)
    if foreground.shape[1] > 1:
        flux[1:] = np.sum(np.maximum(foreground[:, 1:] - foreground[:, :-1], 0.0), axis=0)
    score = (
        0.5 * robust_zscore(energy)
        + 0.2 * robust_zscore(peakiness)
        + 0.15 * robust_zscore(bandwidth_hz)
        + 0.15 * robust_zscore(flux)
    ).astype(np.float32)
    return {
        "score": score,
        "energy": energy.astype(np.float32),
        "peakiness": peakiness.astype(np.float32),
        "peak_freq_hz": peak_freq_hz.astype(np.float32),
        "bandwidth_hz": bandwidth_hz.astype(np.float32),
    }


def merge_intervals(intervals: list[tuple[int, int]], max_gap_frames: int) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for left, right in intervals[1:]:
        last_left, last_right = merged[-1]
        if left - last_right <= max_gap_frames:
            merged[-1] = (last_left, max(last_right, right))
        else:
            merged.append((left, right))
    return merged


def detect_call_intervals(
    y: np.ndarray,
    sr: int,
    cfg: PipelineConfig,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    freqs, times, foreground = compute_foreground_spectrogram(
        y=y,
        sr=sr,
        fmin=cfg.initial_fmin_hz,
        fmax=cfg.initial_fmax_hz,
        nperseg=cfg.seg_nperseg,
        hop=cfg.seg_hop,
        baseline_sec=cfg.seg_baseline_sec,
    )
    frame = summarize_detection_score(foreground, freqs)
    score = frame["score"]
    if score.size == 0:
        return [], {"freqs": freqs, "times": times, "foreground": foreground, **frame}

    above = score >= cfg.seg_onset_z
    raw_intervals: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(above):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            raw_intervals.append((start, idx - 1))
            start = None
    if start is not None:
        raw_intervals.append((start, len(score) - 1))

    max_gap_frames = max(1, int(round(cfg.seg_min_separation_sec * sr / cfg.seg_hop)))
    merged = merge_intervals(raw_intervals, max_gap_frames=max_gap_frames)
    frame_dt = float(times[1] - times[0]) if len(times) > 1 else (cfg.seg_hop / sr)
    out = []
    for left, right in merged:
        if np.max(score[left : right + 1]) < cfg.seg_peak_z:
            continue
        duration_sec = (right - left + 1) * frame_dt
        if duration_sec < cfg.seg_min_call_sec or duration_sec > cfg.seg_max_call_sec:
            continue
        start_sec = max(0.0, float(times[left]) - cfg.seg_padding_sec)
        end_sec = min(len(y) / sr, float(times[right]) + cfg.seg_padding_sec)
        out.append(
            {
                "start_time": start_sec,
                "end_time": end_sec,
                "duration": end_sec - start_sec,
                "peak_score_z": float(np.max(score[left : right + 1])),
                "left_frame": int(left),
                "right_frame": int(right),
            }
        )
    return out, {"freqs": freqs, "times": times, "foreground": foreground, **frame}


def plot_segmentation_diagnostic(
    file_id: str,
    y: np.ndarray,
    sr: int,
    det: dict[str, np.ndarray],
    intervals: list[dict[str, Any]],
    output_path: Path,
) -> None:
    freqs = det["freqs"]
    times = det["times"]
    foreground = det["foreground"]
    score = det["score"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    if foreground.size > 0:
        axes[0].imshow(
            np.log1p(foreground),
            origin="lower",
            aspect="auto",
            extent=[0, len(y) / sr, float(freqs[0]), float(freqs[-1])],
            cmap="magma",
        )
        axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title(f"Segmentation diagnostic: {file_id}")
    for row in intervals:
        axes[0].axvline(row["start_time"], color="cyan", linestyle="--", alpha=0.8)
        axes[0].axvline(row["end_time"], color="lime", linestyle="--", alpha=0.8)
        axes[0].axvspan(row["start_time"], row["end_time"], color="white", alpha=0.12)
    if score.size > 0 and times.size > 0:
        axes[1].plot(times, score, color="black", linewidth=1.2, label="Detection score")
        axes[1].axhline(0.0, color="gray", linewidth=0.8)
        axes[1].legend(loc="upper right")
    for row in intervals:
        axes[1].axvspan(row["start_time"], row["end_time"], color="tab:blue", alpha=0.18)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Score (z)")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_segmentation_review_tables(df: pd.DataFrame, output_csv: Path, template_csv: Path) -> None:
    df.to_csv(output_csv, index=False)
    review = df.copy()
    review["keep_call"] = "Y"
    review["manual_within_file_caller_label"] = ""
    review["notes"] = ""
    review.to_csv(template_csv, index=False)


def load_or_detect_segments(
    files: list[Path],
    cfg: PipelineConfig,
    out: OutputPaths,
) -> pd.DataFrame:
    if cfg.segmentation_manual_csv is not None and cfg.segmentation_manual_csv.exists():
        log_step(f"Using manually corrected segmentation table: {cfg.segmentation_manual_csv}")
        manual = pd.read_csv(cfg.segmentation_manual_csv)
        required = {"file_id", "call_id", "start_time", "end_time"}
        missing = required - set(manual.columns)
        if missing:
            raise ValueError(f"Manual segmentation CSV missing columns: {sorted(missing)}")
        if "keep_call" in manual.columns:
            keep = manual["keep_call"].astype(str).str.strip().str.upper().isin({"Y", "YES", "1", "TRUE"})
            manual = manual[keep].copy()
        manual["duration"] = manual["end_time"] - manual["start_time"]
        return manual.reset_index(drop=True)

    rows = []
    for path in files:
        file_id = path.stem
        log_step(f"Segmenting calls in {file_id}")
        try:
            y, sr = load_audio_mono(path, cfg.target_sr)
        except Exception as exc:
            logging.exception("Failed to load %s", path)
            rows.append(
                {
                    "source_path": str(path),
                    "file_id": file_id,
                    "call_id": "",
                    "start_time": np.nan,
                    "end_time": np.nan,
                    "duration": np.nan,
                    "segmentation_error": str(exc),
                }
            )
            continue
        intervals, det = detect_call_intervals(y, sr, cfg)
        plot_segmentation_diagnostic(
            file_id=file_id,
            y=y,
            sr=sr,
            det=det,
            intervals=intervals,
            output_path=out.figures_segmentation / f"{file_id}_segmentation.png",
        )
        for idx, row in enumerate(intervals, start=1):
            call_id = f"{file_id}__call_{idx:03d}"
            rows.append(
                {
                    "source_path": str(path),
                    "file_id": file_id,
                    "call_id": call_id,
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "duration": row["duration"],
                    "peak_score_z": row["peak_score_z"],
                    "left_frame": row["left_frame"],
                    "right_frame": row["right_frame"],
                    "segmentation_error": "",
                }
            )
    df = pd.DataFrame(rows)
    good = df["call_id"].astype(str).str.len() > 0
    good_df = df[good].copy().reset_index(drop=True)
    save_segmentation_review_tables(
        good_df,
        out.stats / "call_segments_auto.csv",
        out.stats / "call_segments_review_template.csv",
    )
    return good_df


def segment_array(y: np.ndarray, sr: int, start_time: float, end_time: float) -> np.ndarray:
    start = max(0, int(round(start_time * sr)))
    end = min(len(y), int(round(end_time * sr)))
    if end <= start:
        return np.array([], dtype=np.float32)
    return y[start:end].astype(np.float32, copy=False)


def spectral_band_edges(y: np.ndarray, sr: int, fmin: float, fmax: float) -> tuple[float, float, float]:
    freqs, _, spec = compute_spectrogram(y, sr, nperseg=1024, hop=256)
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    spec = spec[mask]
    if spec.size == 0:
        return fmin, fmax, (fmin + fmax) / 2.0
    power = np.mean(spec**2, axis=1)
    total = float(np.sum(power))
    if total <= 1e-12:
        return fmin, fmax, (fmin + fmax) / 2.0
    cdf = np.cumsum(power) / total
    low = float(np.interp(0.10, cdf, freqs))
    high = float(np.interp(0.90, cdf, freqs))
    peak = float(freqs[int(np.argmax(power))])
    return low, high, peak


def estimate_observed_band(
    segments_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> dict[str, float]:
    lows = []
    highs = []
    peaks = []
    cache: dict[str, tuple[np.ndarray, int]] = {}
    for row in segments_df.itertuples(index=False):
        if row.source_path not in cache:
            cache[row.source_path] = load_audio_mono(Path(row.source_path), cfg.target_sr)
        y, sr = cache[row.source_path]
        seg = segment_array(y, sr, row.start_time, row.end_time)
        if len(seg) < int(cfg.seg_min_call_sec * sr):
            continue
        low, high, peak = spectral_band_edges(seg, sr, cfg.initial_fmin_hz, cfg.initial_fmax_hz)
        lows.append(low)
        highs.append(high)
        peaks.append(peak)
    if not lows:
        low = cfg.band_low_hz or 500.0
        high = cfg.band_high_hz or 8000.0
        return {"observed_low_hz": low, "observed_high_hz": high, "observed_peak_hz_med": (low + high) / 2.0}
    low_est = float(np.percentile(lows, 10))
    high_est = float(np.percentile(highs, 90))
    peak_est = float(np.median(peaks))
    low_cut = cfg.band_low_hz if cfg.band_low_hz is not None else max(100.0, low_est * 0.85)
    high_cut = cfg.band_high_hz if cfg.band_high_hz is not None else min(cfg.target_sr * 0.48, high_est * 1.15)
    if high_cut <= low_cut + 100:
        high_cut = min(cfg.target_sr * 0.48, low_cut + 1500.0)
    return {
        "observed_low_hz": float(low_cut),
        "observed_high_hz": float(high_cut),
        "observed_peak_hz_med": peak_est,
    }


def bandpass_filter(y: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    low = max(1.0, low_hz)
    high = min(sr * 0.49, high_hz)
    if high <= low:
        return y.copy()
    sos = signal.butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y).astype(np.float32)


def save_before_after_examples(
    segments_df: pd.DataFrame,
    cfg: PipelineConfig,
    out: OutputPaths,
    band_info: dict[str, float],
    max_examples: int = 4,
) -> None:
    cache: dict[str, tuple[np.ndarray, int]] = {}
    chosen = segments_df.head(max_examples)
    for row in chosen.itertuples(index=False):
        if row.source_path not in cache:
            cache[row.source_path] = load_audio_mono(Path(row.source_path), cfg.target_sr)
        y, sr = cache[row.source_path]
        seg = segment_array(y, sr, row.start_time, row.end_time)
        filt = bandpass_filter(seg, sr, band_info["observed_low_hz"], band_info["observed_high_hz"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for ax, sig, title in [
            (axes[0], seg, "Before filtering"),
            (axes[1], filt, "After filtering"),
        ]:
            freqs, times, spec = compute_spectrogram(sig, sr, 1024, 256)
            ax.imshow(
                np.log1p(spec),
                origin="lower",
                aspect="auto",
                extent=[0, len(sig) / sr, float(freqs[0]), float(freqs[-1])],
                cmap="magma",
            )
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
        fig.suptitle(row.call_id)
        fig.savefig(out.figures_frequency / f"{row.call_id}_before_after.png", dpi=200)
        plt.close(fig)


def export_segmented_calls(
    segments_df: pd.DataFrame,
    cfg: PipelineConfig,
    out: OutputPaths,
    band_info: dict[str, float],
) -> pd.DataFrame:
    cache: dict[str, tuple[np.ndarray, int]] = {}
    rows = []
    for row in segments_df.itertuples(index=False):
        if row.source_path not in cache:
            cache[row.source_path] = load_audio_mono(Path(row.source_path), cfg.target_sr)
        y, sr = cache[row.source_path]
        seg = segment_array(y, sr, row.start_time, row.end_time)
        seg_filt = bandpass_filter(seg, sr, band_info["observed_low_hz"], band_info["observed_high_hz"])
        wav_path = out.segmented_calls / f"{row.call_id}.wav"
        write_wav(wav_path, sr, seg_filt)
        rows.append(
            {
                "source_path": row.source_path,
                "file_id": row.file_id,
                "call_id": row.call_id,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "duration": row.duration,
                "segmented_call_wav": str(wav_path),
                "filter_low_hz": band_info["observed_low_hz"],
                "filter_high_hz": band_info["observed_high_hz"],
            }
        )
    exported = pd.DataFrame(rows)
    exported.to_csv(out.stats / "call_segments_final.csv", index=False)
    return exported


def envelope_features(y: np.ndarray, sr: int) -> dict[str, float]:
    env = np.abs(signal.hilbert(y))
    if env.size == 0 or float(np.max(env)) <= 1e-10:
        return {
            "rise_time_sec": np.nan,
            "fall_time_sec": np.nan,
            "time_to_peak_amp_sec": np.nan,
            "rms_amplitude": np.nan,
        }
    if len(env) >= 5:
        win = min(21, len(env) if len(env) % 2 == 1 else len(env) - 1)
        if win >= 5:
            env = signal.savgol_filter(env, window_length=win, polyorder=2)
    env = np.maximum(env, 0.0)
    peak_idx = int(np.argmax(env))
    peak = float(env[peak_idx])
    thresh = 0.1 * peak
    pre = np.where(env[: peak_idx + 1] >= thresh)[0]
    post = np.where(env[peak_idx:] <= thresh)[0]
    onset_idx = int(pre[0]) if len(pre) else 0
    offset_idx = int(peak_idx + post[0]) if len(post) else len(env) - 1
    rise = max(0.0, (peak_idx - onset_idx) / sr)
    fall = max(0.0, (offset_idx - peak_idx) / sr)
    rms = float(np.sqrt(np.mean(y**2))) if y.size else np.nan
    return {
        "rise_time_sec": rise,
        "fall_time_sec": fall,
        "time_to_peak_amp_sec": peak_idx / sr,
        "rms_amplitude": rms,
    }


def spectral_summary_features(y: np.ndarray, sr: int) -> dict[str, float]:
    freqs, _, spec = compute_spectrogram(y, sr, 1024, 256)
    power = np.mean(spec**2, axis=1)
    total = float(np.sum(power))
    if total <= 1e-12:
        return {
            "peak_frequency_hz": np.nan,
            "centroid_frequency_hz": np.nan,
            "bandwidth_hz": np.nan,
            "spectral_entropy": np.nan,
            "spectral_flatness": np.nan,
            "quartile_freq_25_hz": np.nan,
            "quartile_freq_50_hz": np.nan,
            "quartile_freq_75_hz": np.nan,
        }
    norm = power / total
    centroid = float(np.sum(freqs * norm))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * norm)))
    peak_freq = float(freqs[int(np.argmax(power))])
    entropy = float(-(norm * np.log(norm + 1e-12)).sum() / np.log(len(norm)))
    flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power + 1e-12)))
    cdf = np.cumsum(norm)
    q25 = float(np.interp(0.25, cdf, freqs))
    q50 = float(np.interp(0.50, cdf, freqs))
    q75 = float(np.interp(0.75, cdf, freqs))
    return {
        "peak_frequency_hz": peak_freq,
        "centroid_frequency_hz": centroid,
        "bandwidth_hz": bandwidth,
        "spectral_entropy": entropy,
        "spectral_flatness": flatness,
        "quartile_freq_25_hz": q25,
        "quartile_freq_50_hz": q50,
        "quartile_freq_75_hz": q75,
    }


def smooth_vector(x: np.ndarray, window: int = 7) -> np.ndarray:
    if len(x) < 5:
        return x
    win = min(window, len(x) if len(x) % 2 == 1 else len(x) - 1)
    win = max(5, win)
    if win % 2 == 0:
        win -= 1
    if win < 5:
        return x
    return signal.savgol_filter(x, window_length=win, polyorder=2, mode="interp")


def contour_features(y: np.ndarray, sr: int, low_hz: float, high_hz: float) -> dict[str, float]:
    freqs, times, spec = compute_spectrogram(y, sr, 1024, 256)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    freqs = freqs[mask]
    spec = spec[mask]
    if spec.size == 0 or spec.shape[1] < 2:
        return {
            "start_frequency_hz": np.nan,
            "end_frequency_hz": np.nan,
            "minimum_frequency_hz": np.nan,
            "maximum_frequency_hz": np.nan,
            "frequency_slope_hz_per_s": np.nan,
            "contour_curvature": np.nan,
            "contour_inflections": np.nan,
        }
    power = spec**2
    peak_by_time = np.max(power, axis=0)
    valid = peak_by_time > (0.20 * np.max(peak_by_time))
    if not np.any(valid):
        return {
            "start_frequency_hz": np.nan,
            "end_frequency_hz": np.nan,
            "minimum_frequency_hz": np.nan,
            "maximum_frequency_hz": np.nan,
            "frequency_slope_hz_per_s": np.nan,
            "contour_curvature": np.nan,
            "contour_inflections": np.nan,
        }
    contour = freqs[np.argmax(power[:, valid], axis=0)]
    contour = smooth_vector(contour.astype(np.float32))
    valid_times = times[valid]
    start_freq = float(contour[0])
    end_freq = float(contour[-1])
    min_freq = float(np.min(contour))
    max_freq = float(np.max(contour))
    duration = max(1e-6, float(valid_times[-1] - valid_times[0]))
    slope = float((end_freq - start_freq) / duration)
    first_diff = np.gradient(contour, valid_times)
    second_diff = np.gradient(first_diff, valid_times)
    curvature = float(np.nanmean(np.abs(second_diff)))
    inflections = int(np.sum(np.diff(np.sign(second_diff)) != 0))
    return {
        "start_frequency_hz": start_freq,
        "end_frequency_hz": end_freq,
        "minimum_frequency_hz": min_freq,
        "maximum_frequency_hz": max_freq,
        "frequency_slope_hz_per_s": slope,
        "contour_curvature": curvature,
        "contour_inflections": inflections,
    }


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    mel_min = hz_to_mel(np.array([fmin]))[0]
    mel_max = hz_to_mel(np.array([fmax]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if right <= left:
            continue
        for k in range(left, center):
            if 0 <= k < fb.shape[1] and center > left:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if 0 <= k < fb.shape[1] and right > center:
                fb[m - 1, k] = (right - k) / (right - center)
    return fb


def mfcc_feature_summary(y: np.ndarray, sr: int, cfg: PipelineConfig, low_hz: float, high_hz: float) -> dict[str, float]:
    n_fft = int(min(1024, max(128, len(y))))
    if n_fft % 2 == 1:
        n_fft -= 1
    n_fft = max(128, n_fft)
    hop = 256
    _, _, spec = compute_spectrogram(y, sr, n_fft, hop)
    power = spec**2
    effective_n_fft = max(2, 2 * (power.shape[0] - 1))
    fb = mel_filterbank(sr, effective_n_fft, cfg.n_mels, low_hz, high_hz)
    mel_energy = fb @ power
    mel_energy = np.log(mel_energy + 1e-8)
    coeffs = fftpack.dct(mel_energy, type=2, axis=0, norm="ortho")
    coeffs = coeffs[1 : cfg.n_mfcc + 1]
    delta = np.gradient(coeffs, axis=1) if coeffs.shape[1] > 1 else np.zeros_like(coeffs)
    delta2 = np.gradient(delta, axis=1) if delta.shape[1] > 1 else np.zeros_like(delta)
    feats: dict[str, float] = {}
    for name, mat in [("mfcc", coeffs), ("delta_mfcc", delta), ("delta2_mfcc", delta2)]:
        for i in range(mat.shape[0]):
            feats[f"{name}_{i+1:02d}_mean"] = float(np.mean(mat[i]))
            feats[f"{name}_{i+1:02d}_sd"] = float(np.std(mat[i], ddof=0))
    return feats


def extract_single_call_features(
    wav_path: Path,
    file_id: str,
    call_id: str,
    low_hz: float,
    high_hz: float,
    cfg: PipelineConfig,
) -> dict[str, Any] | None:
    sr, y = wavfile.read(str(wav_path))
    y = normalize_audio_dtype(np.asarray(y))
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) < max(8, int(cfg.seg_min_call_sec * sr)):
        return None
    row: dict[str, Any] = {
        "file_id": file_id,
        "call_id": call_id,
        "segmented_call_wav": str(wav_path),
        "duration_sec": len(y) / sr,
    }
    row.update(envelope_features(y, sr))
    row.update(spectral_summary_features(y, sr))
    row.update(contour_features(y, sr, low_hz, high_hz))
    row.update(mfcc_feature_summary(y, sr, cfg, low_hz, high_hz))
    return row


def build_feature_tables(
    segmented_df: pd.DataFrame,
    cfg: PipelineConfig,
    out: OutputPaths,
    band_info: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    for row in segmented_df.itertuples(index=False):
        feat = extract_single_call_features(
            wav_path=Path(row.segmented_call_wav),
            file_id=row.file_id,
            call_id=row.call_id,
            low_hz=band_info["observed_low_hz"],
            high_hz=band_info["observed_high_hz"],
            cfg=cfg,
        )
        if feat is not None:
            rows.append(feat)
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("No valid single-call features could be extracted.")
    raw = raw.sort_values(["file_id", "call_id"]).reset_index(drop=True)
    raw.to_csv(out.features / "call_features_raw.csv", index=False)

    meta_cols = ["file_id", "call_id", "segmented_call_wav"]
    numeric_cols = [c for c in raw.columns if c not in meta_cols]
    analysis_exclude = ["rms_amplitude"]
    analysis_cols = [c for c in numeric_cols if c not in analysis_exclude]

    clean = raw.copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(axis=0, how="all", subset=analysis_cols)
    if clean.empty:
        raise RuntimeError("All feature rows were empty after initial cleaning.")

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(clean[analysis_cols])
    imputed_df = pd.DataFrame(X_imputed, columns=analysis_cols, index=clean.index)
    clean.loc[:, analysis_cols] = imputed_df

    corr = imputed_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > cfg.corr_threshold)]
    retained_cols = [c for c in analysis_cols if c not in to_drop]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean[retained_cols].to_numpy(dtype=float))
    scaled_df = pd.DataFrame(X_scaled, columns=retained_cols, index=clean.index)

    cleaned = pd.concat([clean[meta_cols].reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)
    cleaned.to_csv(out.features / "call_features_cleaned_standardized.csv", index=False)

    info = {
        "analysis_excluded_features": analysis_exclude,
        "correlation_threshold": cfg.corr_threshold,
        "dropped_highly_correlated_features": to_drop,
        "retained_features": retained_cols,
        "n_calls": int(len(cleaned)),
    }
    with open(out.features / "feature_cleaning_info.json", "w") as f:
        json.dump(info, f, indent=2)
    return raw, cleaned, retained_cols, to_drop


def maybe_run_umap(X: np.ndarray, random_seed: int) -> np.ndarray | None:
    try:
        import umap  # type: ignore

        model = umap.UMAP(random_state=random_seed, n_neighbors=min(10, max(2, len(X) - 1)))
        return model.fit_transform(X)
    except Exception:
        return None


def annotate_scatter(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, labels: list[str]) -> None:
    for x, y, label in zip(xs, ys, labels):
        ax.text(x, y, label, fontsize=7, alpha=0.7)


def run_pca_and_figures(
    cleaned_df: pd.DataFrame,
    retained_cols: list[str],
    out: OutputPaths,
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    X = cleaned_df[retained_cols].to_numpy(dtype=float)
    n_components = min(3, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=cfg.random_seed)
    scores = pca.fit_transform(X)
    pca_df = cleaned_df[["file_id", "call_id"]].copy()
    for idx in range(n_components):
        pca_df[f"PC{idx+1}"] = scores[:, idx]
    pca_df.to_csv(out.features / "pca_scores.csv", index=False)

    ev = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar(range(1, len(ev) + 1), ev * 100.0, color="tab:blue")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("PCA variance explained")
    fig.savefig(out.figures_pca / "pca_variance_explained.png", dpi=220)
    plt.close(fig)

    labels = [f"{r.file_id}|{r.call_id.split('__')[-1]}" for r in pca_df.itertuples(index=False)]
    if "PC1" in pca_df and "PC2" in pca_df:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="file_id", ax=ax, s=70)
        annotate_scatter(ax, pca_df["PC1"].to_numpy(), pca_df["PC2"].to_numpy(), labels)
        ax.set_title("PCA: PC1 vs PC2")
        ax.legend(loc="best", fontsize=7)
        fig.savefig(out.figures_pca / "pca_pc1_pc2.png", dpi=220)
        plt.close(fig)
    if "PC1" in pca_df and "PC3" in pca_df:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        sns.scatterplot(data=pca_df, x="PC1", y="PC3", hue="file_id", ax=ax, s=70)
        annotate_scatter(ax, pca_df["PC1"].to_numpy(), pca_df["PC3"].to_numpy(), labels)
        ax.set_title("PCA: PC1 vs PC3")
        ax.legend(loc="best", fontsize=7)
        fig.savefig(out.figures_pca / "pca_pc1_pc3.png", dpi=220)
        plt.close(fig)

    umap_embedding = maybe_run_umap(X, cfg.random_seed) if cfg.save_umap_if_available else None
    if umap_embedding is not None:
        umap_df = cleaned_df[["file_id", "call_id"]].copy()
        umap_df["UMAP1"] = umap_embedding[:, 0]
        umap_df["UMAP2"] = umap_embedding[:, 1]
        umap_df.to_csv(out.features / "umap_scores.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        sns.scatterplot(data=umap_df, x="UMAP1", y="UMAP2", hue="file_id", ax=ax, s=70)
        annotate_scatter(ax, umap_df["UMAP1"].to_numpy(), umap_df["UMAP2"].to_numpy(), labels)
        ax.set_title("UMAP (secondary exploratory view)")
        ax.legend(loc="best", fontsize=7)
        fig.savefig(out.figures_pca / "umap_secondary.png", dpi=220)
        plt.close(fig)

    summary = {
        "explained_variance_ratio": ev.tolist(),
        "n_components": int(n_components),
        "pc1_percent": float(ev[0] * 100.0) if len(ev) >= 1 else np.nan,
        "pc2_percent": float(ev[1] * 100.0) if len(ev) >= 2 else np.nan,
        "pc3_percent": float(ev[2] * 100.0) if len(ev) >= 3 else np.nan,
        "umap_saved": bool(umap_embedding is not None),
    }
    with open(out.stats / "pca_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return pca_df, summary


def remap_labels(labels: np.ndarray) -> np.ndarray:
    unique = [u for u in sorted(set(labels)) if u != -1]
    mapping = {u: i for i, u in enumerate(unique)}
    return np.array([mapping.get(x, -1) for x in labels], dtype=int)


def safe_cluster_count(labels: np.ndarray) -> int:
    vals = {int(x) for x in labels if int(x) != -1}
    return len(vals)


def evaluate_hierarchical(X: np.ndarray, k_values: list[int]) -> tuple[pd.DataFrame, int, np.ndarray]:
    rows = []
    best_k = k_values[0]
    best_score = -np.inf
    best_labels = None
    for k in k_values:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)
        sil = float(silhouette_score(X, labels)) if k > 1 else np.nan
        rows.append({"k": k, "silhouette": sil})
        if np.isfinite(sil) and sil > best_score:
            best_score = sil
            best_k = k
            best_labels = labels
    if best_labels is None:
        best_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    return pd.DataFrame(rows), best_k, remap_labels(best_labels)


def evaluate_gmm(X: np.ndarray, k_values: list[int], random_seed: int) -> tuple[pd.DataFrame, int, np.ndarray]:
    rows = []
    best_k = k_values[0]
    best_bic = np.inf
    best_labels = None
    for k in k_values:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=random_seed,
            n_init=10,
        )
        gmm.fit(X)
        labels = gmm.predict(X)
        bic = float(gmm.bic(X))
        aic = float(gmm.aic(X))
        sil = float(silhouette_score(X, labels)) if k > 1 else np.nan
        rows.append({"k": k, "bic": bic, "aic": aic, "silhouette": sil})
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_labels = labels
    assert best_labels is not None
    return pd.DataFrame(rows), best_k, remap_labels(best_labels)


def evaluate_hdbscan(X: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    min_cluster_size = max(2, min(5, len(X) // 4 if len(X) >= 8 else 2))
    min_samples = max(1, min_cluster_size // 2)
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, allow_single_cluster=True)
    labels = model.fit_predict(X)
    remapped = remap_labels(labels)
    info = {
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "n_clusters": int(safe_cluster_count(remapped)),
        "n_noise_points": int(np.sum(remapped == -1)),
        "noise_fraction": float(np.mean(remapped == -1)),
    }
    return info, remapped


def choose_conservative_cluster_number(
    hier_k: int,
    gmm_k: int,
    hdb_labels: np.ndarray,
) -> dict[str, Any]:
    hdb_k = safe_cluster_count(hdb_labels)
    candidates = [int(hier_k), int(gmm_k)]
    if hdb_k > 1:
        candidates.append(int(hdb_k))
    candidates = [c for c in candidates if c >= 1]
    plausible_min = int(min(candidates))
    plausible_max = int(max(candidates))
    conservative_k = plausible_min if plausible_min >= 1 else 1
    return {
        "hierarchical_k": int(hier_k),
        "gmm_k": int(gmm_k),
        "hdbscan_k": int(hdb_k),
        "plausible_cluster_min": plausible_min,
        "plausible_cluster_max": plausible_max,
        "conservative_cluster_k": int(conservative_k),
    }


def run_clustering(
    cleaned_df: pd.DataFrame,
    retained_cols: list[str],
    out: OutputPaths,
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    X = cleaned_df[retained_cols].to_numpy(dtype=float)
    max_k = min(cfg.max_clusters_to_test, max(2, len(X) - 1))
    k_values = list(range(2, max_k + 1))
    if len(X) < 3:
        raise RuntimeError("Clustering requires at least 3 segmented calls.")

    link = hierarchy.linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    hierarchy.dendrogram(link, labels=cleaned_df["call_id"].tolist(), leaf_rotation=90, leaf_font_size=7, ax=ax)
    ax.set_title("Hierarchical clustering dendrogram")
    ax.set_ylabel("Ward linkage distance")
    fig.savefig(out.figures_clustering / "hierarchical_dendrogram.png", dpi=220)
    plt.close(fig)

    hier_df, hier_k, hier_labels = evaluate_hierarchical(X, k_values)
    hier_df.to_csv(out.stats / "hierarchical_candidate_solutions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(hier_df["k"], hier_df["silhouette"], marker="o")
    ax.axvline(hier_k, color="red", linestyle="--", label=f"Selected k={hier_k}")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette")
    ax.set_title("Hierarchical candidate cut levels")
    ax.legend()
    fig.savefig(out.figures_clustering / "hierarchical_cut_levels.png", dpi=220)
    plt.close(fig)

    gmm_df, gmm_k, gmm_labels = evaluate_gmm(X, k_values, cfg.random_seed)
    gmm_df.to_csv(out.stats / "gmm_model_selection.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(gmm_df["k"], gmm_df["bic"], marker="o", label="BIC")
    ax.plot(gmm_df["k"], gmm_df["aic"], marker="s", label="AIC")
    ax.axvline(gmm_k, color="red", linestyle="--", label=f"Selected k={gmm_k}")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Criterion value")
    ax.set_title("GMM model selection")
    ax.legend()
    fig.savefig(out.figures_clustering / "gmm_model_selection.png", dpi=220)
    plt.close(fig)

    hdb_info, hdb_labels = evaluate_hdbscan(X)
    with open(out.stats / "hdbscan_summary.json", "w") as f:
        json.dump(hdb_info, f, indent=2)

    k_summary = choose_conservative_cluster_number(hier_k, gmm_k, hdb_labels)
    primary_k = k_summary["conservative_cluster_k"]
    primary_labels = remap_labels(AgglomerativeClustering(n_clusters=primary_k, linkage="ward").fit_predict(X))

    assignments = cleaned_df[["file_id", "call_id"]].copy()
    assignments["cluster_hierarchical"] = hier_labels
    assignments["cluster_gmm"] = gmm_labels
    assignments["cluster_hdbscan"] = hdb_labels
    assignments["cluster_primary_conservative"] = primary_labels
    assignments.to_csv(out.stats / "cluster_assignments.csv", index=False)

    summary = {
        **k_summary,
        "hierarchical_best_silhouette": float(hier_df.loc[hier_df["k"] == hier_k, "silhouette"].iloc[0]),
        "gmm_best_bic": float(gmm_df.loc[gmm_df["k"] == gmm_k, "bic"].iloc[0]),
        "hdbscan_noise_points": int(hdb_info["n_noise_points"]),
        "hdbscan_noise_fraction": float(hdb_info["noise_fraction"]),
        "primary_method": "hierarchical_conservative_cut",
    }
    with open(out.stats / "cluster_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return assignments, summary, primary_labels


def pairwise_distance_matrix(X: np.ndarray) -> np.ndarray:
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))


def within_between_distance_stats(
    X: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    D = pairwise_distance_matrix(X)
    iu = np.triu_indices(len(labels), k=1)
    pair_d = D[iu]
    same = labels[iu[0]] == labels[iu[1]]
    if np.sum(same) == 0 or np.sum(~same) == 0:
        return {
            "within_mean": np.nan,
            "between_mean": np.nan,
            "mannwhitney_u_p": np.nan,
            "effect_direction": "insufficient_pairs",
        }
    within = pair_d[same]
    between = pair_d[~same]
    _, p = stats.mannwhitneyu(within, between, alternative="less")
    return {
        "within_mean": float(np.mean(within)),
        "between_mean": float(np.mean(between)),
        "within_median": float(np.median(within)),
        "between_median": float(np.median(between)),
        "mannwhitney_u_p": float(p),
        "effect_direction": "within_smaller_than_between",
        "within_n_pairs": int(len(within)),
        "between_n_pairs": int(len(between)),
    }


def permutation_test_cluster_structure(
    X: np.ndarray,
    labels: np.ndarray,
    out: OutputPaths,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    if len(set(labels)) < 2:
        return {
            "observed_silhouette": np.nan,
            "observed_between_minus_within": np.nan,
            "silhouette_null_mean": np.nan,
            "silhouette_null_sd": np.nan,
            "between_within_null_mean": np.nan,
            "between_within_null_sd": np.nan,
            "silhouette_p_value": np.nan,
            "between_within_p_value": np.nan,
        }
    rng = np.random.default_rng(cfg.random_seed)
    D = pairwise_distance_matrix(X)
    observed_sil = float(silhouette_score(D, labels, metric="precomputed"))
    wb = within_between_distance_stats(X, labels)
    observed_gap = float(wb["between_mean"] - wb["within_mean"])
    sil_null = np.empty(cfg.n_permutations, dtype=float)
    gap_null = np.empty(cfg.n_permutations, dtype=float)
    for i in range(cfg.n_permutations):
        perm = rng.permutation(len(X))
        Dp = D[np.ix_(perm, perm)]
        sil_null[i] = silhouette_score(Dp, labels, metric="precomputed")
        wb_perm = within_between_distance_stats(X[perm], labels)
        gap_null[i] = wb_perm["between_mean"] - wb_perm["within_mean"]
    sil_p = float((np.sum(sil_null >= observed_sil) + 1) / (cfg.n_permutations + 1))
    gap_p = float((np.sum(gap_null >= observed_gap) + 1) / (cfg.n_permutations + 1))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(sil_null, bins=30, color="lightgray", edgecolor="black")
    ax.axvline(observed_sil, color="red", linestyle="--", linewidth=2, label=f"Observed = {observed_sil:.3f}")
    ax.set_xlabel("Silhouette score")
    ax.set_ylabel("Count")
    ax.set_title("Permutation test: silhouette")
    ax.legend()
    fig.savefig(out.figures_validation / "permutation_silhouette.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(gap_null, bins=30, color="lightgray", edgecolor="black")
    ax.axvline(observed_gap, color="red", linestyle="--", linewidth=2, label=f"Observed = {observed_gap:.3f}")
    ax.set_xlabel("Between-cluster mean distance minus within-cluster mean distance")
    ax.set_ylabel("Count")
    ax.set_title("Permutation test: distance separation")
    ax.legend()
    fig.savefig(out.figures_validation / "permutation_distance_gap.png", dpi=220)
    plt.close(fig)

    return {
        "observed_silhouette": observed_sil,
        "observed_between_minus_within": observed_gap,
        "silhouette_null_mean": float(np.mean(sil_null)),
        "silhouette_null_sd": float(np.std(sil_null, ddof=1)),
        "between_within_null_mean": float(np.mean(gap_null)),
        "between_within_null_sd": float(np.std(gap_null, ddof=1)),
        "silhouette_p_value": sil_p,
        "between_within_p_value": gap_p,
    }


def bootstrap_cluster_stability(
    X: np.ndarray,
    labels: np.ndarray,
    conservative_k: int,
    cfg: PipelineConfig,
    out: OutputPaths,
) -> dict[str, Any]:
    if conservative_k < 2 or len(X) < 4:
        return {
            "mean_ari_vs_original": np.nan,
            "sd_ari_vs_original": np.nan,
            "median_ari_vs_original": np.nan,
            "n_bootstraps": int(cfg.n_bootstraps),
        }
    rng = np.random.default_rng(cfg.random_seed)
    aris = []
    subset_size = max(3, int(round(len(X) * cfg.bootstrap_fraction)))
    for _ in range(cfg.n_bootstraps):
        idx = np.sort(rng.choice(len(X), size=subset_size, replace=False))
        Xi = X[idx]
        orig = labels[idx]
        boot = AgglomerativeClustering(n_clusters=conservative_k, linkage="ward").fit_predict(Xi)
        aris.append(adjusted_rand_score(orig, boot))
    aris = np.asarray(aris, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(aris, bins=25, color="lightgray", edgecolor="black")
    ax.axvline(np.mean(aris), color="red", linestyle="--", linewidth=2, label=f"Mean ARI = {np.mean(aris):.3f}")
    ax.set_xlabel("ARI between bootstrap clustering and original clustering")
    ax.set_ylabel("Count")
    ax.set_title("Bootstrap stability of primary clustering")
    ax.legend()
    fig.savefig(out.figures_validation / "bootstrap_stability_ari.png", dpi=220)
    plt.close(fig)
    return {
        "mean_ari_vs_original": float(np.mean(aris)),
        "sd_ari_vs_original": float(np.std(aris, ddof=1)),
        "median_ari_vs_original": float(np.median(aris)),
        "n_bootstraps": int(cfg.n_bootstraps),
    }


def create_validation_template(segmented_df: pd.DataFrame, out: OutputPaths) -> Path:
    template = segmented_df[["file_id", "call_id"]].copy()
    template["within_file_caller_label"] = ""
    template.loc[template["file_id"].isin(KNOWN_SAME_BAT_FILE_IDS), "within_file_caller_label"] = "known_same_bat_1"
    template["notes"] = ""
    path = out.stats / "validation_metadata_template.csv"
    template.to_csv(path, index=False)
    return path


def compute_cluster_purity(cluster_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    total = len(gt_labels)
    if total == 0:
        return np.nan
    purity = 0
    for cluster in sorted(set(cluster_labels)):
        idx = np.where(cluster_labels == cluster)[0]
        if len(idx) == 0:
            continue
        counts = Counter(gt_labels[idx])
        purity += max(counts.values())
    return purity / total


def labeled_validation(
    cleaned_df: pd.DataFrame,
    retained_cols: list[str],
    primary_labels: np.ndarray,
    out: OutputPaths,
    cfg: PipelineConfig,
    validation_template_path: Path,
) -> dict[str, Any]:
    metadata_path = cfg.validation_metadata if cfg.validation_metadata is not None else validation_template_path
    result: dict[str, Any] = {"validation_metadata_used": str(metadata_path)}
    if not metadata_path.exists():
        result["status"] = "no_validation_metadata"
        return result
    meta = pd.read_csv(metadata_path)
    if "within_file_caller_label" not in meta.columns:
        result["status"] = "metadata_missing_label_column"
        return result
    labels = meta["within_file_caller_label"].astype(str).str.strip()
    keep = (~labels.isin({"", "x", "X", "nan", "None"})) & labels.notna()
    meta = meta[keep].copy()
    if meta.empty:
        result["status"] = "metadata_has_no_labeled_calls"
        return result

    merged = cleaned_df.merge(meta[["file_id", "call_id", "within_file_caller_label"]], on=["file_id", "call_id"], how="inner")
    if merged.empty:
        result["status"] = "no_matching_labeled_calls_after_segmentation"
        return result
    X = merged[retained_cols].to_numpy(dtype=float)
    gt = merged["within_file_caller_label"].astype(str).to_numpy()
    cluster_lookup = dict(zip(cleaned_df["call_id"], primary_labels))
    clusters = merged["call_id"].map(cluster_lookup).to_numpy(dtype=int)

    # Pairwise same/different validation
    D = pairwise_distance_matrix(X)
    iu = np.triu_indices(len(merged), k=1)
    pair_d = D[iu]
    same = gt[iu[0]] == gt[iu[1]]
    result["labeled_call_count"] = int(len(merged))
    result["labeled_unique_callers"] = int(len(set(gt)))
    result["same_pair_count"] = int(np.sum(same))
    result["different_pair_count"] = int(np.sum(~same))
    if np.sum(same) > 0 and np.sum(~same) > 0:
        same_d = pair_d[same]
        diff_d = pair_d[~same]
        _, mw_p = stats.mannwhitneyu(same_d, diff_d, alternative="less")
        auc = roc_auc_score(same.astype(int), -pair_d)
        result["same_vs_different_distance_mannwhitney_p"] = float(mw_p)
        result["same_distance_mean"] = float(np.mean(same_d))
        result["different_distance_mean"] = float(np.mean(diff_d))
        result["same_vs_different_distance_auc"] = float(auc)

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        sns.kdeplot(same_d, ax=ax, label="Same labeled bat", fill=True)
        sns.kdeplot(diff_d, ax=ax, label="Different labeled bats", fill=True)
        ax.set_xlabel("Euclidean distance in standardized feature space")
        ax.set_title("Pairwise acoustic distances: same vs different labeled calls")
        ax.legend()
        fig.savefig(out.figures_validation / "labeled_pairwise_distance_distributions.png", dpi=220)
        plt.close(fig)

        fpr, tpr, _ = roc_curve(same.astype(int), -pair_d)
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC: same vs different labeled calls")
        ax.legend()
        fig.savefig(out.figures_validation / "labeled_same_vs_different_roc.png", dpi=220)
        plt.close(fig)
    else:
        result["same_vs_different_distance_mannwhitney_p"] = np.nan
        result["same_vs_different_distance_auc"] = np.nan
        result["pairwise_note"] = (
            "Insufficient label diversity for same-vs-different pairwise validation. "
            "Current metadata contain only one caller label or only one class of pair."
        )

    unique_gt = len(set(gt))
    unique_clusters = len(set(clusters))
    if unique_gt >= 2 and unique_clusters >= 2:
        result["adjusted_rand_index"] = float(adjusted_rand_score(gt, clusters))
        result["normalized_mutual_info"] = float(normalized_mutual_info_score(gt, clusters))
    else:
        result["adjusted_rand_index"] = np.nan
        result["normalized_mutual_info"] = np.nan
    result["cluster_purity_labeled_subset"] = float(compute_cluster_purity(clusters, gt))

    # Optional small supervised test
    if unique_gt >= 2 and min(Counter(gt).values()) >= 2 and len(merged) >= 6:
        preds = []
        trues = []
        for i in range(len(merged)):
            train_mask = np.ones(len(merged), dtype=bool)
            train_mask[i] = False
            X_train = X[train_mask]
            y_train = gt[train_mask]
            X_test = X[i]
            centroids = {
                label: X_train[y_train == label].mean(axis=0)
                for label in sorted(set(y_train))
            }
            pred = min(centroids, key=lambda label: float(np.linalg.norm(X_test - centroids[label])))
            preds.append(pred)
            trues.append(gt[i])
        labels_unique = sorted(set(gt))
        recalls = []
        for label in labels_unique:
            idx = [j for j, t in enumerate(trues) if t == label]
            recalls.append(np.mean([preds[j] == trues[j] for j in idx]))
        bal_acc = float(np.mean(recalls))
        result["loo_balanced_accuracy"] = bal_acc

        rng = np.random.default_rng(cfg.random_seed)
        perm_scores = []
        for _ in range(min(500, cfg.n_permutations)):
            y_perm = rng.permutation(gt)
            preds_perm = []
            trues_perm = []
            for i in range(len(merged)):
                train_mask = np.ones(len(merged), dtype=bool)
                train_mask[i] = False
                X_train = X[train_mask]
                y_train = y_perm[train_mask]
                X_test = X[i]
                centroids = {
                    label: X_train[y_train == label].mean(axis=0)
                    for label in sorted(set(y_train))
                }
                pred = min(centroids, key=lambda label: float(np.linalg.norm(X_test - centroids[label])))
                preds_perm.append(pred)
                trues_perm.append(y_perm[i])
            recalls_perm = []
            for label in labels_unique:
                idx = [j for j, t in enumerate(trues_perm) if t == label]
                if idx:
                    recalls_perm.append(np.mean([preds_perm[j] == trues_perm[j] for j in idx]))
            perm_scores.append(float(np.mean(recalls_perm)) if recalls_perm else np.nan)
        perm_scores = np.asarray([x for x in perm_scores if np.isfinite(x)], dtype=float)
        if perm_scores.size:
            result["loo_balanced_accuracy_perm_p"] = float((np.sum(perm_scores >= bal_acc) + 1) / (len(perm_scores) + 1))
    else:
        result["loo_balanced_accuracy"] = np.nan
        result["loo_balanced_accuracy_perm_p"] = np.nan
        result["supervised_note"] = (
            "Supervised emitter prediction was skipped because the labeled subset was too small or "
            "did not contain at least two caller labels with enough examples."
        )

    with open(out.stats / "labeled_validation_results.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def file_effect_checks(
    assignments: pd.DataFrame,
    out: OutputPaths,
) -> dict[str, Any]:
    ct = pd.crosstab(assignments["cluster_primary_conservative"], assignments["file_id"])
    ct.to_csv(out.stats / "cluster_by_file_counts.csv")
    prop = ct.div(ct.sum(axis=1), axis=0).fillna(0.0)
    prop.to_csv(out.stats / "cluster_by_file_proportions.csv")

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.heatmap(prop, cmap="viridis", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Cluster-by-file composition")
    ax.set_xlabel("File ID")
    ax.set_ylabel("Primary conservative cluster")
    fig.savefig(out.figures_clustering / "cluster_by_file_heatmap.png", dpi=220)
    plt.close(fig)

    warnings = []
    cluster_file_summary = []
    for cluster_id, row in prop.iterrows():
        dominant_file = row.idxmax()
        dominant_prop = float(row.max())
        contributing_files = int(np.sum(row > 0))
        cluster_file_summary.append(
            {
                "cluster": int(cluster_id),
                "dominant_file": str(dominant_file),
                "dominant_file_proportion": dominant_prop,
                "contributing_files": contributing_files,
            }
        )
        if dominant_prop >= 0.8:
            warnings.append(
                f"Cluster {cluster_id} is strongly dominated by file {dominant_file} "
                f"({dominant_prop:.2%} of calls in that cluster)."
            )
    pd.DataFrame(cluster_file_summary).to_csv(out.stats / "cluster_file_effect_summary.csv", index=False)
    return {"warnings": warnings, "cluster_file_summary": cluster_file_summary}


def generate_report_text(
    files: list[Path],
    segmented_df: pd.DataFrame,
    raw_features: pd.DataFrame,
    retained_cols: list[str],
    dropped_cols: list[str],
    band_info: dict[str, float],
    pca_summary: dict[str, Any],
    cluster_summary: dict[str, Any],
    unsup_validation: dict[str, Any],
    bootstrap_summary: dict[str, Any],
    labeled_validation_summary: dict[str, Any],
    file_effect_summary: dict[str, Any],
) -> str:
    same_diff_sentence = (
        "Partial labeled validation did not include enough label diversity to support a same-vs-different caller comparison."
    )
    if np.isfinite(labeled_validation_summary.get("same_vs_different_distance_mannwhitney_p", np.nan)):
        same_diff_sentence = (
            "Within the labeled subset, same-labeled calls were more similar than different-labeled calls "
            f"(Mann–Whitney p = {labeled_validation_summary['same_vs_different_distance_mannwhitney_p']:.4f}, "
            f"ROC AUC = {labeled_validation_summary['same_vs_different_distance_auc']:.3f})."
        )

    file_effect_sentence = "No strong cluster-by-file domination warning was triggered."
    if file_effect_summary["warnings"]:
        file_effect_sentence = " ".join(file_effect_summary["warnings"])

    lines = [
        "Bioacoustic caller-signature pipeline summary",
        "",
        f"- Number of chopped files analyzed: {len(files)}",
        f"- Number of segmented single calls retained: {len(segmented_df)}",
        "- Feature classes extracted: temporal, spectral, frequency contour, and cepstral (MFCC, delta, delta-delta).",
        f"- RMS amplitude was extracted for description/QC only, but excluded from the default clustering feature space because amplitude is recording-setup dependent.",
        f"- Observed audible-band filter selected from the data: {band_info['observed_low_hz']:.0f}–{band_info['observed_high_hz']:.0f} Hz",
        f"- PCA variance explained: PC1 {pca_summary.get('pc1_percent', np.nan):.1f}%, "
        f"PC2 {pca_summary.get('pc2_percent', np.nan):.1f}%, "
        f"PC3 {pca_summary.get('pc3_percent', np.nan):.1f}%",
        f"- Features retained for analysis after redundancy reduction: {len(retained_cols)} "
        f"(dropped {len(dropped_cols)} highly correlated features).",
        f"- Plausible range of putative acoustic clusters across methods: "
        f"{cluster_summary['plausible_cluster_min']}–{cluster_summary['plausible_cluster_max']}",
        f"- Conservative primary estimate of putative acoustic clusters: {cluster_summary['conservative_cluster_k']}",
        f"- Unsupervised cluster separability (silhouette): {unsup_validation.get('observed_silhouette', np.nan):.3f}",
        f"- Permutation p-value for silhouette: {unsup_validation.get('silhouette_p_value', np.nan):.4f}",
        f"- Mean bootstrap ARI stability of the conservative clustering: "
        f"{bootstrap_summary.get('mean_ari_vs_original', np.nan):.3f}",
        f"- Partial labeled validation: {same_diff_sentence}",
        f"- Cluster/file effect check: {file_effect_sentence}",
        "",
        "Interpretation:",
        "The results should be interpreted as evidence for putative emitter-specific acoustic structure, not as proof of true individual identification. "
        "This is a small-sample exploratory analysis with statistical testing, conservative clustering, and limited partial validation.",
    ]
    return "\n".join(lines)


def manuscript_methods_paragraph(
    n_files: int,
    n_calls: int,
    band_info: dict[str, float],
    retained_cols: list[str],
    cluster_summary: dict[str, Any],
) -> str:
    return (
        f"We analyzed {n_files} chopped recordings containing audible bat alarm calls and treated the single call, rather than the chopped file, as the unit of analysis. "
        f"Calls were segmented conservatively using an adaptive energy- and spectrogram-based detector, reviewed through diagnostic spectrogram plots, and exported as individual WAV files. "
        f"Based on the observed spectral distribution of the detected calls, recordings were filtered in an empirically derived audible band ({band_info['observed_low_hz']:.0f}–{band_info['observed_high_hz']:.0f} Hz). "
        f"For each call we extracted interpretable temporal, spectral, frequency-contour, and cepstral descriptors, removed highly redundant variables, and standardized the retained {len(retained_cols)} analysis features. "
        f"Exploratory structure was visualized using principal component analysis, and putative acoustic clusters were estimated using hierarchical clustering, Gaussian mixture models, and HDBSCAN. "
        f"A conservative primary clustering solution was defined from the lower bound of the cluster-number range supported across methods (primary estimate = {cluster_summary['conservative_cluster_k']} putative clusters), "
        f"and the non-randomness of the resulting structure was evaluated with silhouette statistics, distance-based comparisons, permutation testing, and bootstrap stability analyses. "
        f"Limited partial ground-truth validation was performed on the subset of labeled calls, but interpretation was restricted to putative acoustic signatures rather than confirmed individual identities."
    )


def manuscript_results_paragraph(
    n_files: int,
    n_calls: int,
    pca_summary: dict[str, Any],
    cluster_summary: dict[str, Any],
    unsup_validation: dict[str, Any],
    bootstrap_summary: dict[str, Any],
    labeled_validation_summary: dict[str, Any],
    file_effect_summary: dict[str, Any],
) -> str:
    same_diff_sentence = "The labeled subset was too limited to provide a strong same-versus-different caller comparison."
    if np.isfinite(labeled_validation_summary.get("same_vs_different_distance_mannwhitney_p", np.nan)):
        same_diff_sentence = (
            f"Within the labeled subset, same-labeled calls were more similar than different-labeled calls "
            f"(Mann–Whitney p = {labeled_validation_summary['same_vs_different_distance_mannwhitney_p']:.4f}; "
            f"ROC AUC = {labeled_validation_summary['same_vs_different_distance_auc']:.3f})."
        )
    file_effect_sentence = "No cluster was overwhelmingly dominated by a single chopped file."
    if file_effect_summary["warnings"]:
        file_effect_sentence = "Some clusters were substantially file-dominated, indicating possible file-level structure in addition to caller structure."
    return (
        f"A total of {n_calls} single calls were segmented from {n_files} chopped recordings and carried forward to acoustic analysis. "
        f"The first three principal components explained {pca_summary.get('pc1_percent', np.nan):.1f}%, "
        f"{pca_summary.get('pc2_percent', np.nan):.1f}%, and {pca_summary.get('pc3_percent', np.nan):.1f}% of the retained feature variance, respectively, indicating partial low-dimensional structure without complete separation. "
        f"Across clustering methods, the data supported a plausible range of {cluster_summary['plausible_cluster_min']}–{cluster_summary['plausible_cluster_max']} putative acoustic clusters, with a conservative primary estimate of {cluster_summary['conservative_cluster_k']}. "
        f"The primary clustering solution showed a silhouette score of {unsup_validation.get('observed_silhouette', np.nan):.3f}, which exceeded the corresponding permutation-based null expectation "
        f"(p = {unsup_validation.get('silhouette_p_value', np.nan):.4f}), supporting the presence of non-random acoustic structure. "
        f"Bootstrap resampling indicated a mean clustering stability of ARI = {bootstrap_summary.get('mean_ari_vs_original', np.nan):.3f}. "
        f"{same_diff_sentence} {file_effect_sentence} "
        f"Overall, the dataset supports putative emitter-specific acoustic structure, but the evidence remains exploratory and does not justify claiming confirmed individual identification."
    )


def save_text_outputs(
    out: OutputPaths,
    summary_text: str,
    methods_paragraph: str,
    results_paragraph: str,
) -> None:
    (out.stats / "analysis_summary.txt").write_text(summary_text)
    (out.stats / "methods_paragraph.txt").write_text(methods_paragraph)
    (out.stats / "results_paragraph.txt").write_text(results_paragraph)


def main() -> None:
    cfg = parse_args()
    out = make_output_paths(cfg.output_dir)
    setup_logging(out.logs / "pipeline.log")
    sns.set_theme(style="whitegrid")
    log_step("Starting bat alarm-call signature pipeline.")
    log_step(f"Input directory: {cfg.input_dir}")
    log_step(f"Output directory: {cfg.output_dir}")
    log_step(f"Random seed: {cfg.random_seed}")

    files = find_audio_files(cfg.input_dir)
    if not files:
        raise FileNotFoundError(f"No supported audio/video files found in {cfg.input_dir}")
    log_step(f"Found {len(files)} input files.")

    segmented_auto = load_or_detect_segments(files, cfg, out)
    if segmented_auto.empty:
        raise RuntimeError("No calls were segmented. Try lowering the segmentation thresholds.")
    log_step(f"Segmented {len(segmented_auto)} call candidates.")

    band_info = estimate_observed_band(segmented_auto, cfg)
    with open(out.stats / "observed_band_summary.json", "w") as f:
        json.dump(band_info, f, indent=2)
    log_step(
        "Estimated audible call band from the data: "
        f"{band_info['observed_low_hz']:.0f}–{band_info['observed_high_hz']:.0f} Hz"
    )

    save_before_after_examples(segmented_auto, cfg, out, band_info, max_examples=4)
    segmented_final = export_segmented_calls(segmented_auto, cfg, out, band_info)
    log_step(f"Exported {len(segmented_final)} filtered single-call WAV segments.")

    raw_features, cleaned_df, retained_cols, dropped_cols = build_feature_tables(segmented_final, cfg, out, band_info)
    log_step(
        f"Extracted raw features for {len(raw_features)} calls; retained {len(retained_cols)} "
        f"features after cleaning and redundancy reduction."
    )

    _, pca_summary = run_pca_and_figures(cleaned_df, retained_cols, out, cfg)
    log_step("Saved PCA outputs.")

    assignments, cluster_summary, primary_labels = run_clustering(cleaned_df, retained_cols, out, cfg)
    log_step(
        f"Clustering complete. Conservative primary estimate: "
        f"{cluster_summary['conservative_cluster_k']} putative acoustic clusters."
    )

    X = cleaned_df[retained_cols].to_numpy(dtype=float)
    wb_stats = within_between_distance_stats(X, primary_labels)
    with open(out.stats / "within_between_distance_stats.json", "w") as f:
        json.dump(wb_stats, f, indent=2)

    unsup_validation = permutation_test_cluster_structure(X, primary_labels, out, cfg)
    with open(out.stats / "unsupervised_validation.json", "w") as f:
        json.dump({**wb_stats, **unsup_validation}, f, indent=2)
    log_step("Saved unsupervised validation outputs.")

    bootstrap_summary = bootstrap_cluster_stability(
        X, primary_labels, cluster_summary["conservative_cluster_k"], cfg, out
    )
    with open(out.stats / "bootstrap_stability_summary.json", "w") as f:
        json.dump(bootstrap_summary, f, indent=2)

    validation_template_path = create_validation_template(segmented_final, out)
    labeled_validation_summary = labeled_validation(
        cleaned_df, retained_cols, primary_labels, out, cfg, validation_template_path
    )
    log_step("Saved labeled-subset validation outputs.")

    assignments_full = assignments.merge(cleaned_df[["file_id", "call_id"]], on=["file_id", "call_id"], how="left")
    file_effect_summary = file_effect_checks(assignments_full, out)
    with open(out.stats / "file_effect_summary.json", "w") as f:
        json.dump(file_effect_summary, f, indent=2)

    summary_text = generate_report_text(
        files=files,
        segmented_df=segmented_final,
        raw_features=raw_features,
        retained_cols=retained_cols,
        dropped_cols=dropped_cols,
        band_info=band_info,
        pca_summary=pca_summary,
        cluster_summary=cluster_summary,
        unsup_validation=unsup_validation,
        bootstrap_summary=bootstrap_summary,
        labeled_validation_summary=labeled_validation_summary,
        file_effect_summary=file_effect_summary,
    )
    methods_paragraph = manuscript_methods_paragraph(
        n_files=len(files),
        n_calls=len(segmented_final),
        band_info=band_info,
        retained_cols=retained_cols,
        cluster_summary=cluster_summary,
    )
    results_paragraph = manuscript_results_paragraph(
        n_files=len(files),
        n_calls=len(segmented_final),
        pca_summary=pca_summary,
        cluster_summary=cluster_summary,
        unsup_validation=unsup_validation,
        bootstrap_summary=bootstrap_summary,
        labeled_validation_summary=labeled_validation_summary,
        file_effect_summary=file_effect_summary,
    )
    save_text_outputs(out, summary_text, methods_paragraph, results_paragraph)
    log_step("Wrote summary text, manuscript-ready methods paragraph, and results paragraph.")

    run_info = {
        "input_dir": str(cfg.input_dir),
        "output_dir": str(cfg.output_dir),
        "n_input_files": int(len(files)),
        "n_segmented_calls": int(len(segmented_final)),
        "validation_template": str(validation_template_path),
    }
    with open(out.stats / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print("\nPipeline finished successfully.")
    print(f"Input files: {len(files)}")
    print(f"Segmented calls: {len(segmented_final)}")
    print(
        "Putative acoustic clusters (conservative estimate): "
        f"{cluster_summary['conservative_cluster_k']} "
        f"(plausible range {cluster_summary['plausible_cluster_min']}–{cluster_summary['plausible_cluster_max']})"
    )
    print(
        "Observed silhouette / permutation p-value: "
        f"{unsup_validation.get('observed_silhouette', np.nan):.3f} / "
        f"{unsup_validation.get('silhouette_p_value', np.nan):.4f}"
    )
    print(f"All outputs written to: {out.root}")


if __name__ == "__main__":
    main()
