"""
03_ttest_callrate_and_acoustic_features.py
==========================================
Welch's t-tests for call-rate comparisons and trial-level acoustic feature contrasts.
Produces Figures 2, 3A, 3B, and Supplementary Figure 1 of the manuscript.

Call-rate analyses (Methods §5.2)
----------------------------------
Three comparisons, all using Welch's two-sided t-tests (unequal variances):
  (i)  2023 Day vs Night call rates (pooling Net and No-Net) → Fig. 3B
  (ii) 2022 daytime Net vs No-Net call rates               → Fig. 3A
  (iii)2023 daytime Net vs No-Net call rates               → Supplementary Fig. 1

Acoustic feature contrasts (Methods §5.3)
------------------------------------------
Trial-level medians of three SHAP-identified features (min freq at call start,
bandwidth at call end, maximum fundamental frequency) compared between alarm (2023)
and social calls using Welch's t-tests. Pearson and Spearman correlations computed
between min freq(start) and fundamental(max) at the trial level. → Fig. 2

INPUT
-----
all-vocalizations.xlsx
    Excel workbook with sheets: day-net(23), day-no-net(23), night-net(23),
    night-no-net(23), day-net(22), day-no-net(22), food, territorial,
    reproductive, unknown.

OUTPUT
------
outputs/ttests/
    Fig2_acoustic_features_alarm23_vs_social.png
    Fig3A_net_vs_nonet_2022_day.png/.svg
    Fig3B_day_vs_night_2023.png/.svg
    SuppFig1_net_vs_nonet_2023_day.png/.svg
    Fig2_trial_medians_used_for_ttests.csv          ← audit trail
    Fig2_trial_level_ttests_alarm23_vs_social.csv
    Fig2_feature_correlation_minfreq_vs_fundamental.csv
    callrate_pvalues_summary.csv

DEPENDENCIES
------------
Python 3.9+
numpy, pandas, matplotlib, scipy, seaborn, openpyxl
Install: pip install numpy pandas matplotlib scipy seaborn openpyxl

AUTHORS
-------
Omer Yinon, Yossi Yovel Lab, Tel Aviv University
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr, spearmanr

# =======================
# CONFIG
# =======================
XLSX_PATH = Path(__file__).resolve().parent / "all-vocalizations.xlsx"   # master input file

ALARM_SHEETS = [
    "day-net(23)", "day-no-net(23)", "night-net(23)",
    "night-no-net(23)", "day-no-net(22)", "day-net(22)",
]
SOCIAL_SHEETS = ["food", "territorial", "reproductive", "unknown"]

# Three SHAP-identified focal features (Methods §5.3)
SELECTED_FEATURES = ["min freq(start)", "bandw(end)", "fundamental(max)"]

# Output directory — relative to this script
OUTDIR = Path(__file__).parent / "outputs" / "ttests"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Fixed random seed for jitter reproducibility
JITTER_SEED = 42

# =======================
# PLOT STYLE
# =======================
SCI_STYLE = {
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.edgecolor":     "black",
    "axes.linewidth":     0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.color":        "black",
    "ytick.color":        "black",
    "axes.labelcolor":    "black",
    "text.color":         "black",
    "grid.color":         "0.8",
    "grid.linestyle":     "--",
    "grid.linewidth":     0.4,
    "grid.alpha":         0.3,
    "font.size":          16,
    "axes.labelsize":     16,
    "axes.titlesize":     16,
    "xtick.labelsize":    16,
    "ytick.labelsize":    16,
    "legend.fontsize":    16,
}

plt.rcParams["svg.fonttype"] = "none"


# =======================
# HELPERS — STATS
# =======================
def welch_df(x, y):
    """Welch–Satterthwaite degrees of freedom."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    num = (vx / nx + vy / ny) ** 2
    den = (vx**2) / (nx**2 * (nx - 1)) + (vy**2) / (ny**2 * (ny - 1))
    return num / den


def cohen_d(x, y):
    """Cohen's d (pooled SD)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / sp if sp > 0 else np.nan


def summarize(x):
    x = np.asarray(x, float)
    return dict(n=len(x), mean=np.mean(x), sd=np.std(x, ddof=1), median=np.median(x))


# =======================
# HELPERS — PLOTTING
# =======================
def p_to_stars(p: float) -> str:
    if p is None or not np.isfinite(p): return ""
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return ""


def format_p_label(p: float) -> str:
    if p is None or not np.isfinite(p): return "p = NA"
    p_str  = "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"
    stars  = p_to_stars(p)
    return f"{p_str} {stars}".rstrip()


def add_panel_label(ax, label: str, x=0.02, y=0.98, fontsize=18):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=fontsize, fontweight="bold")


def mono_boxplot(ax, data, positions, width=0.33, whis=1.5):
    return ax.boxplot(
        data, positions=positions, widths=width, whis=whis,
        patch_artist=True, showfliers=False,
        boxprops=dict(facecolor="none", edgecolor="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.6),
    )


def jitter_scatter(ax, values, center, seed, jitter=0.12, s=18, alpha=0.7):
    """Overlay individual points with reproducible jitter."""
    rng = np.random.default_rng(seed)
    x = center + (rng.random(len(values)) - 0.5) * 2 * jitter
    ax.scatter(x, values, s=s, facecolors="none", edgecolors="black",
               linewidths=0.7, alpha=alpha, zorder=3)


def add_sig_bracket(ax, x0, x1, y_sig, label, fontsize=15):
    ax.plot([x0, x0, x1, x1], [y_sig, y_sig + 2, y_sig + 2, y_sig], c="black")
    ax.text((x0 + x1) / 2, y_sig + 4, label, ha="center", va="bottom", fontsize=fontsize)


def save_fig(fig, path: Path, dpi=600):
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    print(f"Saved: {path.with_suffix('.png')}")


# =======================
# HELPERS — DATA LOADING
# =======================
def drop_xxxwav_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.apply(lambda row: row.astype(str).str.contains("xxx.wav", na=False), axis=1).any(axis=1)
    return df.loc[~mask].copy()


def load_sheet(xls: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    return drop_xxxwav_rows(xls.parse(sheet))


def count_calls_per_trial(df: pd.DataFrame, year_prefix: str) -> list[int]:
    """
    Count calls per 30-s trial. Trial boundaries are rows whose first column
    starts with `year_prefix` (e.g. '2023' or '2022').
    """
    trial_counts, call_count = [], 0
    for val in df.iloc[:, 0].astype(str):
        if val.startswith(year_prefix):
            trial_counts.append(call_count)
            call_count = 0
        else:
            call_count += 1
    if call_count > 0:
        trial_counts.append(call_count)
    return trial_counts


def _is_trial_marker_row(row: pd.Series) -> bool:
    """Detect trial-boundary rows: first cell looks like a filename, rest are NaN."""
    first = str(row.iloc[0]).strip().lower()
    rest_empty = row.iloc[1:].isna().all()
    looks_like_recording = (".wav" in first) or first.startswith("202")
    return rest_empty and looks_like_recording


def add_trial_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a trial_id to each call row; drop marker rows."""
    trial_id, trial_ids = 0, []
    for _, row in df.iterrows():
        if _is_trial_marker_row(row):
            trial_ids.append(np.nan)
            trial_id += 1
        else:
            trial_ids.append(trial_id)
    out = df.copy()
    out["trial_id"] = trial_ids
    out = out.dropna(subset=["trial_id"]).copy()
    out["trial_id"] = out["trial_id"].astype(int)
    return out


# =======================
# FIGURE 3B — Day vs Night (2023, pooled Net + No-Net)
# =======================
def fig3B_day_vs_night_2023(xls: pd.ExcelFile) -> float:
    day_trials, night_trials = [], []
    for sheet in ["day-net(23)", "day-no-net(23)", "night-net(23)", "night-no-net(23)"]:
        df     = load_sheet(xls, sheet)
        trials = count_calls_per_trial(df, "2023")
        (day_trials if "day" in sheet else night_trials).extend(trials)

    _, p_val = ttest_ind(day_trials, night_trials, equal_var=False)

    y_max = max(max(day_trials or [0]), max(night_trials or [0]))
    y_sig = y_max + 5

    with plt.rc_context(SCI_STYLE):
        fig, ax = plt.subplots(figsize=(6.2, 7.0))
        add_panel_label(ax, "B")
        mono_boxplot(ax, [day_trials, night_trials], positions=[0, 1])
        jitter_scatter(ax, day_trials,   0, seed=JITTER_SEED)
        jitter_scatter(ax, night_trials, 1, seed=JITTER_SEED + 1)
        add_sig_bracket(ax, 0, 1, y_sig, format_p_label(p_val))
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Day (n={len(day_trials)})", f"Night (n={len(night_trials)})"])
        ax.set_ylabel("Number of calls per 30 s")
        ax.minorticks_off()
        ax.grid(True, axis="y")
        plt.tight_layout()
        save_fig(fig, OUTDIR / "Fig3B_day_vs_night_2023")
        plt.show()

    return p_val


# =======================
# FIGURE 3A — Net vs No-Net (2022 day)
# =======================
def fig3A_net_vs_nonet_2022_day(xls: pd.ExcelFile, y_sig: float, y_top: float) -> tuple:
    net_22   = count_calls_per_trial(load_sheet(xls, "day-net(22)"),    "2022")
    nonet_22 = count_calls_per_trial(load_sheet(xls, "day-no-net(22)"), "2022")
    _, p_val = ttest_ind(net_22, nonet_22, equal_var=False)

    with plt.rc_context(SCI_STYLE):
        fig, ax = plt.subplots(figsize=(6.2, 7.0))
        add_panel_label(ax, "A")
        mono_boxplot(ax, [nonet_22, net_22], positions=[0, 1])
        jitter_scatter(ax, nonet_22, 0, seed=JITTER_SEED)
        jitter_scatter(ax, net_22,   1, seed=JITTER_SEED + 1)
        add_sig_bracket(ax, 0, 1, y_sig, format_p_label(p_val))
        ax.set_ylim(0, y_top)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"No-Net (n={len(nonet_22)})", f"Net (n={len(net_22)})"])
        ax.set_ylabel("Number of calls per 30 s")
        ax.minorticks_off()
        ax.grid(True, axis="y")
        plt.tight_layout()
        save_fig(fig, OUTDIR / "Fig3A_net_vs_nonet_2022_day")
        plt.show()

    return p_val, net_22, nonet_22


# =======================
# SUPPLEMENTARY FIG 1 — Net vs No-Net (2023 day, not significant)
# =======================
def suppfig1_net_vs_nonet_2023_day(xls: pd.ExcelFile, y_sig: float, y_top: float) -> tuple:
    net_23   = count_calls_per_trial(load_sheet(xls, "day-net(23)"),    "2023")
    nonet_23 = count_calls_per_trial(load_sheet(xls, "day-no-net(23)"), "2023")
    _, p_val = ttest_ind(net_23, nonet_23, equal_var=False)

    with plt.rc_context(SCI_STYLE):
        fig, ax = plt.subplots(figsize=(6.2, 7.0))
        add_panel_label(ax, "A")
        mono_boxplot(ax, [nonet_23, net_23], positions=[0, 1])
        jitter_scatter(ax, nonet_23, 0, seed=JITTER_SEED)
        jitter_scatter(ax, net_23,   1, seed=JITTER_SEED + 1)
        add_sig_bracket(ax, 0, 1, y_sig, format_p_label(p_val))
        ax.set_ylim(0, y_top)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"No-Net (n={len(nonet_23)})", f"Net (n={len(net_23)})"])
        ax.set_ylabel("Number of calls per 30 s")
        ax.minorticks_off()
        ax.grid(True, axis="y")
        plt.tight_layout()
        save_fig(fig, OUTDIR / "SuppFig1_net_vs_nonet_2023_day")
        plt.show()

    return p_val, net_23, nonet_23


# =======================
# FIGURE 2 — Trial-level acoustic features: Alarm (2023) vs Social
# =======================
def fig2_acoustic_features_alarm23_vs_social(xls: pd.ExcelFile):
    """
    For each of the three SHAP features, compute trial-level medians then run
    Welch's t-test (alarm 2023 vs social). Also exports Pearson + Spearman
    correlations between min freq(start) and fundamental(max). (Methods §5.3)
    """
    alarm_23_sheets = [s for s in ALARM_SHEETS if "(23)" in s]
    dfs = []

    for sheet in alarm_23_sheets:
        df = add_trial_ids(load_sheet(xls, sheet))
        df["group"], df["sheet"] = "Alarm", sheet
        dfs.append(df)
    for sheet in SOCIAL_SHEETS:
        df = add_trial_ids(load_sheet(xls, sheet))
        df["group"], df["sheet"] = "Social", sheet
        dfs.append(df)

    df_combined = pd.concat(dfs, ignore_index=True)
    feat_cols   = [f for f in SELECTED_FEATURES if f in df_combined.columns]
    keep        = ["group", "sheet", "trial_id"] + feat_cols
    df_combined = df_combined[keep].copy()

    # Trial-level medians — unit of analysis for all t-tests (Methods §5.3)
    trial_med = (
        df_combined
        .groupby(["group", "sheet", "trial_id"], as_index=False)[feat_cols]
        .median(numeric_only=True)
    )
    trial_med["trial_uid"] = trial_med["sheet"].astype(str) + "|" + trial_med["trial_id"].astype(str)
    trial_med.to_csv(OUTDIR / "Fig2_trial_medians_used_for_ttests.csv", index=False)

    # ── Welch t-tests ──
    stats_rows = []
    for feature in feat_cols:
        a = trial_med.loc[trial_med["group"] == "Alarm",  feature].dropna().astype(float).values
        s = trial_med.loc[trial_med["group"] == "Social", feature].dropna().astype(float).values
        if len(a) < 3 or len(s) < 3:
            continue
        t_stat, p_val = ttest_ind(a, s, equal_var=False)
        sa, ss = summarize(a), summarize(s)
        stats_rows.append({
            "feature":             feature,
            "comparison":          "Alarm(23) vs Social (trial medians)",
            "t":                   t_stat,
            "df_welch":            welch_df(a, s),
            "p":                   p_val,
            "cohen_d":             cohen_d(a, s),
            "Alarm_n":    sa["n"], "Alarm_mean":   sa["mean"],
            "Alarm_sd":   sa["sd"], "Alarm_median": sa["median"],
            "Social_n":   ss["n"], "Social_mean":  ss["mean"],
            "Social_sd":  ss["sd"], "Social_median": ss["median"],
        })
    pd.DataFrame(stats_rows).to_csv(
        OUTDIR / "Fig2_trial_level_ttests_alarm23_vs_social.csv", index=False
    )

    # ── Pearson + Spearman correlations (min freq start vs fundamental max) ──
    xcol, ycol = "min freq(start)", "fundamental(max)"
    corr_rows = []
    if xcol in trial_med.columns and ycol in trial_med.columns:
        for grp in ["All", "Alarm", "Social"]:
            sub = trial_med if grp == "All" else trial_med[trial_med["group"] == grp]
            sub = sub[[xcol, ycol]].dropna()
            if len(sub) < 3:
                continue
            x, y = sub[xcol].astype(float).values, sub[ycol].astype(float).values
            r_p, p_p     = pearsonr(x, y)
            rho_s, p_s   = spearmanr(x, y)
            corr_rows.append({
                "group": grp, "x": xcol, "y": ycol, "N_trials": len(x),
                "pearson_r": r_p, "pearson_R2": r_p**2, "pearson_p": p_p,
                "spearman_rho": rho_s, "spearman_p": p_s,
            })
    pd.DataFrame(corr_rows).to_csv(
        OUTDIR / "Fig2_feature_correlation_minfreq_vs_fundamental.csv", index=False
    )

    # ── Plot ──
    palette = {"Alarm": "#EF9A9A", "Social": "#90CAF9"}
    order   = ["Alarm", "Social"]
    centers = {"Alarm": 0, "Social": 1}

    with plt.rc_context(SCI_STYLE):
        fig, axes = plt.subplots(1, len(feat_cols), figsize=(18, 6), sharey=True)
        if len(feat_cols) == 1:
            axes = [axes]

        for ax, feature in zip(axes, feat_cols):
            data = trial_med[["group", feature]].dropna().copy()
            data[feature] = data[feature].astype(float)

            sns.boxplot(
                data=data, x="group", y=feature, order=order,
                width=0.33, palette=palette, showcaps=True,
                boxprops={"edgecolor": "black", "linewidth": 1.2},
                whiskerprops={"color": "black", "linewidth": 1.0},
                capprops={"color": "black", "linewidth": 1.0},
                medianprops={"color": "black", "linewidth": 1.6},
                flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none",
                            "markeredgecolor": "black", "linestyle": "none"},
                ax=ax,
            )

            # Reproducible jitter overlay
            rng = np.random.default_rng(JITTER_SEED)
            for g, xc in centers.items():
                yy = data.loc[data["group"] == g, feature].astype(float).values
                xx = xc + (rng.random(len(yy)) - 0.5) * 2 * 0.10
                ax.scatter(xx, yy, s=18, facecolors="none", edgecolors="black",
                           linewidths=0.7, alpha=0.55, zorder=3)

            ax.set_ylim(0, 8000)
            ax.set_ylabel("")
            ax.set_xlabel(feature, fontsize=18, labelpad=10)
            ax.set_xticklabels(["Alarm", "Social"], fontsize=14)
            ax.tick_params(axis="y", labelsize=13)
            ax.grid(True, axis="y", alpha=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        try:
            fig.supylabel("Frequency (Hz)", fontsize=16, x=0.02)
        except (TypeError, AttributeError):
            fig.text(0.02, 0.5, "Frequency (Hz)", rotation="vertical", va="center", fontsize=16)

        fig.subplots_adjust(left=0.10, bottom=0.18, wspace=0.25)
        save_fig(fig, OUTDIR / "Fig2_acoustic_features_alarm23_vs_social", dpi=600)
        plt.show()

    print("Stats saved:", OUTDIR / "Fig2_trial_level_ttests_alarm23_vs_social.csv")
    print("Correlation saved:", OUTDIR / "Fig2_feature_correlation_minfreq_vs_fundamental.csv")


# =======================
# MAIN
# =======================
def main():
    print(f"Output directory: {OUTDIR}\n")

    if not Path(XLSX_PATH).exists():
        raise FileNotFoundError(f"Data file not found: {Path(XLSX_PATH).resolve()}")

    xls = pd.ExcelFile(XLSX_PATH)

    # Fig. 3B — Day vs Night 2023
    p_day_night = fig3B_day_vs_night_2023(xls)

    # Compute shared y-axis limits across both call-rate figures (2022 + 2023)
    net22    = count_calls_per_trial(load_sheet(xls, "day-net(22)"),    "2022")
    nonet22  = count_calls_per_trial(load_sheet(xls, "day-no-net(22)"), "2022")
    net23    = count_calls_per_trial(load_sheet(xls, "day-net(23)"),    "2023")
    nonet23  = count_calls_per_trial(load_sheet(xls, "day-no-net(23)"), "2023")
    global_max = max(max(net22 or [0]), max(nonet22 or [0]),
                     max(net23 or [0]), max(nonet23 or [0]))
    y_sig = global_max + 5
    y_top = y_sig + 12

    # Fig. 3A — Net vs No-Net 2022
    p_net_nonet_22, _, _ = fig3A_net_vs_nonet_2022_day(xls, y_sig=y_sig, y_top=y_top)

    # Supplementary Fig. 1 — Net vs No-Net 2023 (not significant)
    p_net_nonet_23, _, _ = suppfig1_net_vs_nonet_2023_day(xls, y_sig=y_sig, y_top=y_top)

    # Fig. 2 — Acoustic feature t-tests + correlations
    fig2_acoustic_features_alarm23_vs_social(xls)

    # Save p-value summary
    pd.DataFrame([
        {"figure": "Fig. 3B", "comparison": "Day vs Night (2023, pooled)",    "p_value": p_day_night},
        {"figure": "Fig. 3A", "comparison": "Net vs No-Net (2022 day)",        "p_value": p_net_nonet_22},
        {"figure": "Supp. Fig. 1", "comparison": "Net vs No-Net (2023 day)",  "p_value": p_net_nonet_23},
    ]).to_csv(OUTDIR / "callrate_pvalues_summary.csv", index=False)

    print("\nDone. All outputs saved to:", OUTDIR)


if __name__ == "__main__":
    main()
