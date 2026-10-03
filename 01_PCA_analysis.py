"""
01_PCA_analysis.py
==================
Principal Component Analysis (PCA) of acoustic features from alarm and social calls
of the Mauritian flying fox (Pteropus niger). Produces Figure 1B of the manuscript.

Axis-aligned ellipses represent +/-0.5 SD along PC1 and PC2 (see Methods 5.1).

INPUT
-----
all-vocalizations.xlsx  (place next to this script)

OUTPUT
------
outputs/pca/fig1B_pca_alarm_vs_social.png  (350 dpi)
outputs/pca/fig1B_pca_alarm_vs_social.svg

DEPENDENCIES
------------
pip install numpy pandas matplotlib scikit-learn openpyxl

AUTHORS
-------
Omer Yinon, Yossi Yovel Lab, Tel Aviv University
"""

import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
XLSX_PATH     = Path(__file__).resolve().parent / "all-vocalizations.xlsx"
ELLIPSE_SCALE = 0.5

OUTDIR = Path(__file__).parent / "outputs" / "pca"
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["svg.fonttype"] = "none"

SOCIAL_BASE = ["food", "territorial", "reproductive", "unknown"]

_AMP_OR_LEVEL_RE = re.compile(r"(amp|ampl|ampli|amplit|amplitude|ample)", re.IGNORECASE)


def save_png_and_svg(fig, path, dpi=350, extra_artists=None):
    path = str(path)
    base, _ = os.path.splitext(path)
    png = base + ".png"
    svg = base + ".svg"
    extra_artists = [] if extra_artists is None else list(extra_artists)
    fig.savefig(png, dpi=dpi, bbox_inches="tight", bbox_extra_artists=extra_artists)
    fig.savefig(svg, format="svg", bbox_inches="tight", bbox_extra_artists=extra_artists)
    return png, svg


def is_amp_feature(name):
    return _AMP_OR_LEVEL_RE.search(str(name)) is not None


def drop_amp_features(df):
    num  = df.select_dtypes(include="number")
    keep = [c for c in num.columns if not is_amp_feature(c)]
    return num[keep].copy()


def infer_year(sheet):
    return "22" if "(22)" in sheet else "23"


def is_social(sheet):
    return sheet in SOCIAL_BASE or any(sheet == f"{b}(22)" for b in SOCIAL_BASE)


def is_alarm(sheet):
    return "net" in sheet.lower()


def alarm_label(sheet):
    year = infer_year(sheet)
    return f"no-net({year})" if "no-net" in sheet.lower() else f"net({year})"


def drop_trial_separator_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove trial-separator rows.

    In all-vocalizations.xlsx each trial/interaction ends with a separator row that
    holds only the recording filename (first column); all acoustic feature columns
    are empty. These rows are not calls. If retained, mean imputation turns them into
    artificial 'average' calls. Here, the first column is excluded and only rows with
    at least one non-missing feature value are kept.
    """
    feature_cols = [c for c in df.columns[1:] if c not in ("group", "sheet", "season",
                                                              "net_group", "plot_label", "year")]
    return df.dropna(subset=feature_cols, how="all").copy()


def base_social(sheet):
    for b in SOCIAL_BASE:
        if sheet == b or sheet == f"{b}(22)":
            return b
    return sheet


def percent_inside_axis_aligned_ellipse(pts, scale):
    if pts.shape[0] < 2:
        return float("nan")
    x  = pts["PC1"].to_numpy(dtype=float)
    y  = pts["PC2"].to_numpy(dtype=float)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0 or sy == 0:
        return float("nan")
    val = ((x - mx) / sx) ** 2 + ((y - my) / sy) ** 2
    return float(np.mean(val <= (scale ** 2)))


def load_all_data(xlsx):
    xls  = pd.ExcelFile(xlsx)
    rows = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        df = df[~df.apply(lambda r: r.astype(str).str.contains("xxx.wav"), axis=1).any(axis=1)]
        df = drop_trial_separator_rows(df)   # FIX: separator rows are not calls
        if is_social(sheet):
            group, label = "social", sheet
        elif is_alarm(sheet):
            group, label = "alarm", alarm_label(sheet)
        else:
            continue
        df["plot_label"] = label
        df["group"]      = group
        df["year"]       = infer_year(sheet)
        rows.append(df)
    if not rows:
        raise ValueError("No valid social/alarm sheets were found.")
    out = pd.concat(rows, ignore_index=True)
    print("Calls in PCA:", len(out), out["group"].value_counts().to_dict())
    return out


def run_pca(df, out_path, ellipse_scale=ELLIPSE_SCALE):
    X  = drop_amp_features(df)
    X  = X.dropna(axis=1, how="all").fillna(X.mean())
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xs)

    pca_df          = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    pca_df["label"] = df.loc[X.index, "plot_label"].values
    centroids       = pca_df.groupby("label")[["PC1", "PC2"]].mean()

    social_colors = {
        "food":         "#48D1CC",
        "territorial":  "#1CA9C9",
        "reproductive": "#2986CC",
        "unknown":      "#0047AB",
    }
    alarm_colors = {"net": "#FFA07A", "no-net": "#FF6347"}

    def label_color(lbl):
        if lbl.startswith("net("):     return alarm_colors["net"]
        if lbl.startswith("no-net("): return alarm_colors["no-net"]
        return social_colors.get(base_social(lbl), "#0047AB")

    def label_marker(lbl):
        return "^" if "(22)" in lbl else "o"

    fig, ax = plt.subplots(figsize=(10, 7))

    for lbl, row in centroids.iterrows():
        ax.scatter(row.PC1, row.PC2, s=220, marker=label_marker(lbl),
                   color=label_color(lbl), edgecolor="black", linewidth=1.5)

        pts = pca_df[pca_df["label"] == lbl][["PC1", "PC2"]]
        if len(pts) >= 5:
            mx, my = pts["PC1"].mean(), pts["PC2"].mean()
            sx, sy = pts["PC1"].std(ddof=1), pts["PC2"].std(ddof=1)
            ax.add_patch(Ellipse(
                xy=(mx, my),
                width=2 * sx * ellipse_scale,
                height=2 * sy * ellipse_scale,
                angle=0.0,
                edgecolor=label_color(lbl),
                facecolor="none",
                linewidth=1.2,
            ))
            frac = percent_inside_axis_aligned_ellipse(pts, ellipse_scale)
            if np.isfinite(frac):
                print(f"Ellipse ({ellipse_scale}) — {lbl}: {frac*100:.2f}% inside")

    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)", fontsize=14)
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.subplots_adjust(bottom=0.24)

    color_handles = [
        Line2D([], [], marker="o", linestyle="None", markerfacecolor=alarm_colors["net"],
               markeredgecolor="black", markersize=11, label="Alarm: Net"),
        Line2D([], [], marker="o", linestyle="None", markerfacecolor=alarm_colors["no-net"],
               markeredgecolor="black", markersize=11, label="Alarm: No-net"),
        Line2D([], [], marker="o", linestyle="None", markerfacecolor=social_colors["food"],
               markeredgecolor="black", markersize=11, label="Social: Food"),
        Line2D([], [], marker="o", linestyle="None", markerfacecolor=social_colors["territorial"],
               markeredgecolor="black", markersize=11, label="Social: Territorial"),
        Line2D([], [], marker="o", linestyle="None", markerfacecolor=social_colors["reproductive"],
               markeredgecolor="black", markersize=11, label="Social: Reproductive"),
        Line2D([], [], marker="o", linestyle="None", markerfacecolor=social_colors["unknown"],
               markeredgecolor="black", markersize=11, label="Social: Unknown"),
    ]
    shape_handles = [
        Line2D([], [], marker="^", linestyle="None", markerfacecolor="white",
               markeredgecolor="black", markersize=11, label="2022"),
        Line2D([], [], marker="o", linestyle="None", markerfacecolor="white",
               markeredgecolor="black", markersize=11, label="2023"),
    ]

    leg1 = fig.legend(handles=color_handles, loc="lower center",
                      bbox_to_anchor=(0.38, 0.03), ncol=2, frameon=False,
                      title="Color", fontsize=10)
    leg2 = fig.legend(handles=shape_handles, loc="lower center",
                      bbox_to_anchor=(0.82, 0.03), ncol=1, frameon=False,
                      title="Shape", fontsize=10)

    fig.canvas.draw()
    png, svg = save_png_and_svg(fig, out_path, dpi=350, extra_artists=[leg1, leg2])
    print("Saved:", png)
    print("Saved:", svg)
    plt.show()


if __name__ == "__main__":
    df = load_all_data(XLSX_PATH)
    run_pca(df, OUTDIR / "fig1B_pca_alarm_vs_social")
