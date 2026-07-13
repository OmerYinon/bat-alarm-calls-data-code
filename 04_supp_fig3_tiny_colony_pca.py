"""
04_supp_fig3_tiny_colony_pca.py
================================
PCA of alarm calls from the reduced-social (tiny-colony) experiment.
Produces Supplementary Figure 3 of the manuscript.

Each point represents one segmented alarm call (n = 54). Colors encode
nine chip-constrained potential-caller partitions; marker shapes encode
recording day. Filled circles mark group centroids. Axis-aligned ellipses
(±1 SD) show visual spread only — they do NOT confirm individual identity
(see Methods §5.4 and Supplementary Results).

INPUT
-----
tables/reduced48_potential9_assignments.csv
    Required columns: PC1, PC2, day, potential9, file_id, call_id

OUTPUT
------
outputs/supp_fig3/supplementary_fig3_tiny_colony_pca_potential9.png  (600 dpi)
outputs/supp_fig3/supplementary_fig3_tiny_colony_pca_potential9.svg

DEPENDENCIES
------------
Python 3.9+
pandas, numpy, matplotlib
Install: pip install pandas numpy matplotlib

AUTHORS
-------
Omer Yinon, Yossi Yovel Lab, Tel Aviv University
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


# ---------------- CONFIG ----------------
# Input CSV lives in a tables/ folder next to this script
TABLE_PATH = Path(__file__).parent / "tables" / "reduced48_potential9_assignments.csv"

# Output directory — relative to this script's location
OUTDIR = Path(__file__).parent / "outputs" / "supp_fig3"
OUTDIR.mkdir(parents=True, exist_ok=True)

# PCA variance explained (hard-coded from the fitted model; see Methods §5.4)
PC1_VARIANCE = 28.7
PC2_VARIANCE = 12.7

# ±1 SD axis-aligned ellipses (visual spread; not statistical confidence regions)
ELLIPSE_SD = 1

plt.rcParams["svg.fonttype"] = "none"


# ---------------- HELPERS ----------------
def save_png_and_svg(fig: plt.Figure, path: Path, dpi: int = 600) -> None:
    fig.savefig(str(path) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(path) + ".svg", format="svg", bbox_inches="tight")
    print("Saved:", str(path) + ".png")
    print("Saved:", str(path) + ".svg")


def axis_aligned_ellipse_size(
    points: pd.DataFrame, n_sd: float = ELLIPSE_SD
) -> tuple[float, float]:
    """Return (width, height) for an axis-aligned ±n_sd SD ellipse."""
    if len(points) <= 1:
        return 0.75, 0.75
    sx = float(points["PC1"].std(ddof=1))
    sy = float(points["PC2"].std(ddof=1))
    return max(2 * n_sd * sx, 0.75), max(2 * n_sd * sy, 0.75)


# ---------------- LOAD DATA ----------------
def load_plot_data() -> pd.DataFrame:
    """Load PCA scores and format labels for the figure."""
    if not TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing table: {TABLE_PATH}")

    df = pd.read_csv(TABLE_PATH)

    required = {"PC1", "PC2", "day", "potential9", "file_id", "call_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {TABLE_PATH.name}: {sorted(missing)}"
        )

    df = df.copy()
    # Rename P1..P9 → PC1..PC9 for the legend
    df["potential_caller"] = (
        df["potential9"].astype(str).str.replace("P", "PC", regex=False)
    )
    df["day"] = df["day"].astype(str)
    return df


# ---------------- PLOT ----------------
def draw_pca_figure(df: pd.DataFrame) -> None:
    """Draw and save the manuscript-style Supp. Fig. 3 PCA."""
    groups = sorted(df["potential_caller"].dropna().unique())
    days   = sorted(df["day"].dropna().unique())

    cmap   = plt.get_cmap("tab10")
    colors = {group: cmap(i % 10) for i, group in enumerate(groups)}

    # Marker shapes per recording day
    day_markers: dict[str, str] = {
        "2023-09-26": "o",
        "2023-09-27": "^",
        "2023-09-30": "s",
        "2023-10-03": "D",
        "2023-10-05": "P",
        "2023-10-06": "X",
    }
    fallback = ["v", "<", ">", "*", "h"]
    for i, day in enumerate(days):
        day_markers.setdefault(day, fallback[i % len(fallback)])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9,
            "legend.title_fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.75, color="#d9d9d9", alpha=0.85)
    ax.set_axisbelow(True)

    # --- Ellipses (drawn first, behind points) ---
    for group in groups:
        grp = df[df["potential_caller"] == group]
        cx  = float(grp["PC1"].mean())
        cy  = float(grp["PC2"].mean())
        w, h = axis_aligned_ellipse_size(grp)
        ax.add_patch(
            Ellipse(
                (cx, cy),
                width=w, height=h, angle=0,
                facecolor="none",
                edgecolor=colors[group],
                linewidth=1.6, alpha=0.85, zorder=1,
            )
        )

    # --- Scatter points (per day for shape mapping) ---
    for day in days:
        day_df = df[df["day"] == day]
        ax.scatter(
            day_df["PC1"], day_df["PC2"],
            c=[colors[g] for g in day_df["potential_caller"]],
            marker=day_markers[day],
            s=80,
            edgecolors="black", linewidths=0.65,
            alpha=0.9, zorder=3,
        )

    # --- Centroids ---
    centroids = df.groupby("potential_caller", as_index=False)[["PC1", "PC2"]].mean()
    ax.scatter(
        centroids["PC1"], centroids["PC2"],
        c=[colors[g] for g in centroids["potential_caller"]],
        marker="o", s=210,
        edgecolors="black", linewidths=1.25, zorder=4,
    )

    ax.set_xlabel(f"PC1 ({PC1_VARIANCE:.1f}%)")
    ax.set_ylabel(f"PC2 ({PC2_VARIANCE:.1f}%)")
    for spine in ax.spines.values():
        spine.set_linewidth(0.95)
        spine.set_color("black")

    # --- Legend: color ---
    color_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=colors[g], markeredgecolor="black",
               markersize=8, label=g)
        for g in groups
    ]
    centroid_handle = Line2D(
        [0], [0], marker="o", color="black",
        markerfacecolor="#bdbdbd", markeredgecolor="black",
        linestyle="none", markersize=8, label="Centroid",
    )
    leg1 = ax.legend(
        handles=color_handles + [centroid_handle],
        title="Color",
        loc="upper center", bbox_to_anchor=(0.35, -0.13),
        frameon=False, ncol=2,
        columnspacing=1.5, handletextpad=0.7,
    )
    ax.add_artist(leg1)

    # --- Legend: shape ---
    day_handles = [
        Line2D([0], [0], marker=day_markers[d], color="black",
               markerfacecolor="white", markeredgecolor="black",
               linestyle="none", markersize=8, label=d)
        for d in days
    ]
    ax.legend(
        handles=day_handles,
        title="Shape (recording day)",
        loc="upper center", bbox_to_anchor=(0.78, -0.13),
        frameon=False, ncol=1, handletextpad=0.7,
    )

    fig.tight_layout()
    save_png_and_svg(fig, OUTDIR / "supplementary_fig3_tiny_colony_pca_potential9")
    plt.show()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    df = load_plot_data()
    draw_pca_figure(df)
