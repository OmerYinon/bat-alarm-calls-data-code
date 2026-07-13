"""
02_RF_2022_3features.py
=======================
Random Forest classifier — 2022 cross-year generalization test (Figure 1C, Methods §5.3).

Tests whether the three SHAP-identified acoustic features (minimum frequency at call
start, bandwidth at call end, maximum fundamental frequency) — identified on the 2023
dataset — generalize to the independent 2022 colony.

Two analyses are run:
  1. Full imbalanced dataset (alarm n=1647, social n=473) with permutation test (n=1000).
  2. 100× balanced downsampling loop (alarm downsampled to match social n) to confirm
     results are not driven by class imbalance.

INPUT
-----
all-vocalizations.xlsx
    Master Excel workbook (place next to this script). Only the following
    sheets are used:
      Alarm  : day-net(22), day-no-net(22)
      Social : food(22), territorial(22), unknown(22)

OUTPUT
------
outputs/rf_2022/
    rf_2022_social_vs_alarm_3features_full_confusion_matrix.png/.svg
    rf_2022_social_vs_alarm_3feat_full_metrics.csv
    rf_2022_social_vs_alarm_3feat_full_permutation_null_ba.csv
    rf_2022_social_vs_alarm_3feat_full_permutation_summary.csv
    rf_2022_social_vs_alarm_3feat_downsampled_100runs.csv
    rf_2022_social_vs_alarm_3feat_downsampled_100runs_summary.csv
    rf_2022_social_vs_alarm_3features_full_shap_importance.csv

DEPENDENCIES
------------
Python 3.9+
numpy, pandas, matplotlib, scikit-learn, shap, openpyxl
Install: pip install numpy pandas matplotlib scikit-learn shap openpyxl

AUTHORS
-------
Omer Yinon, Yossi Yovel Lab, Tel Aviv University
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
)

import shap

# =======================
# CONFIG
# =======================
XLSX_PATH = Path(__file__).resolve().parent / "all-vocalizations.xlsx"

# 2022 alarm sheets
ALARM_22_SHEETS = ["day-net(22)", "day-no-net(22)"]

# 2022 social sheets — territorial(22) + unknown(22) (food merged into unknown)
SOCIAL_22_CANDIDATES = ["food(22)", "territorial(22)", "unknown(22)"]

# Amplitude/level-proxy features excluded (same filter as PCA script)
PREFIXES_AMP = ("amp", "ampli", "ample", "amplit", "amplitude")

# Restrict to the 3 SHAP-identified features from the 2023 model (Methods §5.3)
FORCE_3_FEATURES = True

# Column-matching rules (case-insensitive): must contain ALL listed substrings
FEATURE_SUBSTRINGS = {
    "min_freq_start":   ["min", "freq", "start"],
    "bandw_end":        ["bandw", "end"],
    "fundamental_max":  ["fundamental", "max"],
}

# Balanced downsampling settings (Methods §5.3)
DO_DOWNSAMPLE_LOOP   = True
N_DOWNSAMPLE_REPEATS = 100

# Output directory — relative to this script
OUTDIR = Path(__file__).parent / "outputs" / "rf_2022"
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["svg.fonttype"] = "none"


# =======================
# HELPERS
# =======================
def is_amp_feature(name: str) -> bool:
    return str(name).lower().strip().startswith(PREFIXES_AMP)


def drop_xxxwav_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.apply(lambda row: row.astype(str).str.contains("xxx.wav", na=False)).any(axis=1)
    return df.loc[~mask].copy()


def save_figure(fig: plt.Figure, out_base: Path, *, dpi: int = 300) -> dict:
    png = out_base.with_suffix(".png")
    svg = out_base.with_suffix(".svg")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    return {"png": str(png), "svg": str(svg)}


def _pick_column_by_substrings(columns, required_substrings: list) -> str | None:
    req = [s.lower() for s in required_substrings]
    for c in columns:
        if all(r in str(c).lower() for r in req):
            return c
    return None


def downsample_to_equal_classes(
    df: pd.DataFrame,
    label_col: str = "group",
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    vc = df[label_col].value_counts()
    if vc.shape[0] != 2:
        raise ValueError(f"Expected 2 classes in {label_col}, got {vc.to_dict()}")
    n_min = int(vc.min())
    return pd.concat(
        [sub.sample(n=n_min, replace=False, random_state=random_state)
         for _, sub in df.groupby(label_col)],
        ignore_index=True,
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.select_dtypes(include="number").copy()
    X = X[[c for c in X.columns if not is_amp_feature(c)]]
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.mean())

    if FORCE_3_FEATURES:
        col_min = _pick_column_by_substrings(X.columns, FEATURE_SUBSTRINGS["min_freq_start"])
        col_bw  = _pick_column_by_substrings(X.columns, FEATURE_SUBSTRINGS["bandw_end"])
        col_f0  = _pick_column_by_substrings(X.columns, FEATURE_SUBSTRINGS["fundamental_max"])

        missing = [name for name, col in [
            ("min freq (start)", col_min),
            ("bandw (end)", col_bw),
            ("fundamental (max)", col_f0),
        ] if col is None]

        if missing:
            raise ValueError(
                f"Could not find required feature columns: {missing}\n"
                f"Available columns: {list(X.columns)}"
            )

        X = X[[col_min, col_bw, col_f0]].copy()
        X.columns = ["min_freq_start", "bandw_end", "fundamental_max"]
        print("Using 3 SHAP-identified features:", list(X.columns))

    return X


def permutation_test_rf_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    n_perm: int = 1000,
    n_estimators: int = 100,
) -> dict:
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    idx = np.arange(len(y))
    train_idx, test_idx, y_train, y_test = train_test_split(
        idx, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    rf_obs = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf_obs.fit(X_train, y_train)
    ba_obs = balanced_accuracy_score(y_test, rf_obs.predict(X_test))

    rng = np.random.default_rng(random_state)
    y_values = y.values.copy()
    null_ba = np.empty(n_perm, dtype=float)

    for i in range(n_perm):
        rng.shuffle(y_values)
        y_perm = pd.Series(y_values)
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        rf.fit(X_train, y_perm.iloc[train_idx])
        null_ba[i] = balanced_accuracy_score(y_perm.iloc[test_idx], rf.predict(X_test))

    p_val = (np.sum(null_ba >= ba_obs) + 1) / (n_perm + 1)

    return {
        "ba_observed": float(ba_obs),
        "null_ba":     null_ba,
        "p_value":     float(p_val),
        "null_mean":   float(np.mean(null_ba)),
        "null_95":     float(np.quantile(null_ba, 0.95)),
        "null_99":     float(np.quantile(null_ba, 0.99)),
        "n_perm":      int(n_perm),
    }


def _fit_eval_one_run(
    df_in: pd.DataFrame,
    *,
    run_name: str,
    save_confusion: bool = False,
) -> dict:
    X = build_features(df_in)
    y = df_in["group"].map({"social": 0, "alarm": 1})
    assert X.shape[1] == 3, f"Expected 3 features, got {X.shape[1]}"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    ba  = balanced_accuracy_score(y_test, y_pred)

    out = {
        "run":               run_name,
        "balanced_accuracy": float(ba),
        "accuracy":          float(acc),
        "n_total":           int(len(y)),
        "n_features":        int(X.shape[1]),
        "features_used":     "|".join(X.columns.tolist()),
        "n_train":           int(len(y_train)),
        "n_test":            int(len(y_test)),
    }

    if save_confusion:
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_estimator(
            rf, X_test, y_test,
            display_labels=["social", "alarm"], cmap="Blues", ax=ax,
        )
        ax.set_title(f"RF Confusion Matrix — {run_name}\nBalanced Accuracy = {ba:.2%}")
        fig.tight_layout()
        paths = save_figure(fig, OUTDIR / f"{run_name}_confusion_matrix")
        plt.show()
        out.update(paths)

        # SHAP feature importance — robust to old and new SHAP versions
        try:
            explainer   = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_test)
            # Handle both old SHAP (list of arrays) and new SHAP (3D array)
            if isinstance(shap_values, list):
                shap_arr = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
            else:
                shap_arr = np.asarray(shap_values)
                if shap_arr.ndim == 3:
                    shap_arr = shap_arr[:, :, 1]
            importance = np.abs(shap_arr).mean(axis=0)

            shap_df = (
                pd.DataFrame({"feature": X_test.columns.tolist(), "mean_abs_shap": importance})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            shap_csv = OUTDIR / f"{run_name}_shap_importance.csv"
            shap_df.to_csv(shap_csv, index=False)
            print(f"\nSHAP feature importance:\n{shap_df.to_string(index=False)}")
            print(f"Saved: {shap_csv}")

        except Exception as e:
            print(f"[Warning] SHAP skipped: {e}")

    return out


# =======================
# MAIN
# =======================
def main():
    print(f"Output directory: {OUTDIR}\n")

    if not XLSX_PATH.exists():
        raise FileNotFoundError(f"Excel file not found: {XLSX_PATH.resolve()}")

    xls    = pd.ExcelFile(XLSX_PATH)
    sheets = list(xls.sheet_names)

    # Validate required sheets
    missing_alarm = [s for s in ALARM_22_SHEETS if s not in sheets]
    if missing_alarm:
        raise ValueError(f"Missing alarm sheets: {missing_alarm}\nAvailable: {sheets}")

    social_22 = [s for s in SOCIAL_22_CANDIDATES if s in sheets]
    if not social_22:
        raise ValueError(f"No social sheets found among {SOCIAL_22_CANDIDATES}\nAvailable: {sheets}")

    print("Alarm sheets :", ALARM_22_SHEETS)
    print("Social sheets:", social_22)

    # Load data
    rows = []
    for sh in ALARM_22_SHEETS:
        df = drop_xxxwav_rows(xls.parse(sh))
        df["group"] = "alarm"
        rows.append(df)
    for sh in social_22:
        df = drop_xxxwav_rows(xls.parse(sh))
        df["group"] = "social"
        rows.append(df)

    df_all = pd.concat(rows, ignore_index=True)
    print("\nClass counts:", df_all["group"].value_counts().to_dict())

    # ── 1. Single run on full (imbalanced) dataset ──
    single = _fit_eval_one_run(
        df_all,
        run_name="rf_2022_social_vs_alarm_3features_full",
        save_confusion=True,
    )
    print(f"\n=== Full dataset (imbalanced) ===")
    print(f"Balanced Accuracy: {single['balanced_accuracy']:.3%} | Accuracy: {single['accuracy']:.3%}")

    pd.DataFrame([{
        "analysis":            "2022_social_vs_alarm_3features_full",
        "alarm_sheets":        "|".join(ALARM_22_SHEETS),
        "social_sheets":       "|".join(social_22),
        "test_size":           0.25,
        "random_state":        42,
        **single,
    }]).to_csv(OUTDIR / "rf_2022_social_vs_alarm_3feat_full_metrics.csv", index=False)

    # ── Permutation test (full dataset) ──
    X_full = build_features(df_all)
    y_full = df_all["group"].map({"social": 0, "alarm": 1})

    perm = permutation_test_rf_holdout(
        X=X_full, y=y_full, test_size=0.25, random_state=42, n_perm=1000, n_estimators=100
    )
    pd.DataFrame({"balanced_accuracy": perm["null_ba"]}).to_csv(
        OUTDIR / "rf_2022_social_vs_alarm_3feat_full_permutation_null_ba.csv", index=False
    )
    pd.DataFrame([{
        "analysis":                   "2022_social_vs_alarm_3features_full",
        "balanced_accuracy_observed": single["balanced_accuracy"],
        "perm_null_mean":             perm["null_mean"],
        "perm_null_95":               perm["null_95"],
        "perm_null_99":               perm["null_99"],
        "perm_p_value":               perm["p_value"],
        "perm_n":                     perm["n_perm"],
        "features_used":              single["features_used"],
    }]).to_csv(OUTDIR / "rf_2022_social_vs_alarm_3feat_full_permutation_summary.csv", index=False)

    print(
        f"Permutation test: BA_obs={single['balanced_accuracy']:.3%} | "
        f"null_95={perm['null_95']:.3%} | p={perm['p_value']:.4g}"
    )

    # ── 2. 100× balanced downsampling loop ──
    if DO_DOWNSAMPLE_LOOP:
        print(f"\nRunning {N_DOWNSAMPLE_REPEATS}× balanced downsampling...")
        results = []
        for i in range(N_DOWNSAMPLE_REPEATS):
            seed_i = 1000 + i
            df_ds  = downsample_to_equal_classes(df_all, label_col="group", random_state=seed_i)
            out_i  = _fit_eval_one_run(
                df_ds,
                run_name="rf_2022_social_vs_alarm_3features_downsampled",
                save_confusion=False,
            )
            out_i["iter"]     = i + 1
            out_i["n_social"] = int((df_ds["group"] == "social").sum())
            out_i["n_alarm"]  = int((df_ds["group"] == "alarm").sum())
            results.append(out_i)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{N_DOWNSAMPLE_REPEATS} runs complete...")

        res_df  = pd.DataFrame(results)
        mean_ba = res_df["balanced_accuracy"].mean()
        sd_ba   = res_df["balanced_accuracy"].std(ddof=1)
        q025    = res_df["balanced_accuracy"].quantile(0.025)
        q975    = res_df["balanced_accuracy"].quantile(0.975)

        res_df.to_csv(OUTDIR / "rf_2022_social_vs_alarm_3feat_downsampled_100runs.csv", index=False)
        pd.DataFrame([{
            "n_runs":                    N_DOWNSAMPLE_REPEATS,
            "balanced_accuracy_mean":    mean_ba,
            "balanced_accuracy_sd":      sd_ba,
            "balanced_accuracy_q025":    q025,
            "balanced_accuracy_q975":    q975,
            "accuracy_mean":             res_df["accuracy"].mean(),
            "accuracy_sd":               res_df["accuracy"].std(ddof=1),
            "features_used":             single["features_used"],
        }]).to_csv(
            OUTDIR / "rf_2022_social_vs_alarm_3feat_downsampled_100runs_summary.csv", index=False
        )

        print(f"\n=== 100× downsampling (balanced classes) ===")
        print(f"Balanced Accuracy: mean={mean_ba:.3%}, SD={sd_ba:.3%}, 95% CI=[{q025:.3%}, {q975:.3%}]")


if __name__ == "__main__":
    main()
