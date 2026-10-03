"""
05_RF_2023_alarm_vs_social.py
==============================
Random Forest classifiers for acoustic separation of alarm and social calls
of the Mauritian flying fox (Pteropus niger). Produces Figure 1C and the
alarm-condition text results of the manuscript.

RF #1 — Alarm (2023) vs Social
    Trains on all 2023 alarm contexts (day/night x Net/No-Net) vs social calls.
    Produces the confusion matrix (Figure 1C) and SHAP feature importance.
    Balanced accuracy: 85.3% (test set).

RF #2 — Net vs No-Net, Season 2022
    Distinguishes higher- vs lower-threat alarm calls within 2022.
    Result cited in text (balanced accuracy: 67.9%); no dedicated figure.

RF #3 — Net vs No-Net, Season 2023
    Same contrast within 2023.
    Result cited in text (balanced accuracy: 68.25%); no dedicated figure.

RF #4 — Net vs No-Net, Pooled 2022 + 2023
    Cross-season pooled contrast; SHAP exported for Methods reporting.

RF #5 — Season 2022 Net vs All Other Alarm
    Cited in text; no dedicated figure.

All classifiers use stratified 75/25 train/test splits (random_state=42) and
empirical chance estimation via label permutation (n=1000, +1 p-value correction).

INPUT
-----
all-vocalizations.xlsx
    Excel workbook with one sheet per vocal category (same file used by
    scripts 01-03). Required alarm sheets: day-net(23), day-no-net(23),
    night-net(23), night-no-net(23), day-no-net(22), day-net(22).
    Required social sheets: food, territorial, reproductive, unknown.

OUTPUT
------
outputs/rf_2023/fig1c_alarm23_vs_social_confusion_matrix.png  (300 dpi)
outputs/rf_2023/fig1c_alarm23_vs_social_confusion_matrix.svg
outputs/rf_2023/fig1c_shap_feature_importance.csv
outputs/rf_2023/fig1c_rf_metrics.csv
outputs/rf_2023/rf_net_nonet_22_*
outputs/rf_2023/rf_net_nonet_23_*
outputs/rf_2023/rf_net_nonet_22_23_pooled_*
outputs/rf_2023/rf_net22_vs_other_alarm_*

DEPENDENCIES
------------
Python 3.9+
numpy, pandas, matplotlib, scikit-learn, shap
Install: pip install numpy pandas matplotlib scikit-learn shap openpyxl

AUTHORS
-------
Omer Yinon, Yossi Yovel Lab, Tel Aviv University
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
)


# ---------------- CONFIG ----------------
BASE_DIR  = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "all-vocalizations.xlsx"

OUTDIR = BASE_DIR / "outputs" / "rf_2023"
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["svg.fonttype"] = "none"


# ---------------- SHEETS ----------------
ALARM_SHEETS = [
    "day-net(23)", "day-no-net(23)", "night-net(23)",
    "night-no-net(23)", "day-no-net(22)", "day-net(22)",
]
SOCIAL_SHEETS = ["food", "territorial", "reproductive", "unknown"]


# ---------------- AMPLITUDE FEATURE FILTER ----------------
_AMP_TOKEN_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:amp|ampl|ampli|amplit|amplitude|ample)(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)


def is_amp_feature(name: str) -> bool:
    return _AMP_TOKEN_RE.search(str(name).strip().lower()) is not None


# ---------------- ROW FILTER ----------------
def drop_xxxwav_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.apply(
        lambda row: row.astype(str).str.contains("xxx.wav", na=False)
    ).any(axis=1)
    return df.loc[~mask].copy()


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


# ---------------- FIGURE EXPORT ----------------
def save_figure(fig: plt.Figure, out_base: Path, *, dpi: int = 300) -> dict:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png = out_base.with_suffix(".png")
    svg = out_base.with_suffix(".svg")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    print("Saved:", png)
    print("Saved:", svg)
    return {"png": str(png), "svg": str(svg)}


# ---------------- SHAP HELPER ----------------
def extract_shap_array(shap_values, X_test):
    """Return a 2D (n_samples x n_features) SHAP array for the positive class,
    compatible with both old SHAP (list) and new SHAP (3D ndarray)."""
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            arr = arr[:, :, 1]
    if arr.shape != (X_test.shape[0], X_test.shape[1]):
        raise ValueError(f"SHAP shape {arr.shape} != X_test {X_test.shape}")
    return arr


# ---------------- PERMUTATION TEST ----------------
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

    if y.nunique() != 2:
        raise ValueError(f"Permutation test expects binary labels; got {y.nunique()} classes.")

    rng = np.random.default_rng(random_state)
    idx = np.arange(len(y))
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        idx, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train = X.iloc[X_train_idx]
    X_test  = X.iloc[X_test_idx]

    rf_obs = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    rf_obs.fit(X_train, y_train)
    ba_obs = balanced_accuracy_score(y_test, rf_obs.predict(X_test))

    null_ba = np.empty(n_perm, dtype=float)
    y_values = y.values.copy()
    for i in range(n_perm):
        rng.shuffle(y_values)
        y_perm = pd.Series(y_values)
        rf_p = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )
        rf_p.fit(X_train, y_perm.iloc[X_train_idx])
        null_ba[i] = balanced_accuracy_score(
            y_perm.iloc[X_test_idx], rf_p.predict(X_test)
        )

    p_val = (np.sum(null_ba >= ba_obs) + 1) / (n_perm + 1)
    return {
        "ba_observed": float(ba_obs),
        "null_ba": null_ba,
        "p_value": float(p_val),
        "null_mean": float(np.mean(null_ba)),
        "null_95": float(np.quantile(null_ba, 0.95)),
        "null_99": float(np.quantile(null_ba, 0.99)),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "n_perm": int(n_perm),
        "n_estimators": int(n_estimators),
    }


# ---------------- RF #1: ALARM (2023) vs SOCIAL -> Fig. 1C ----------------
def rf_alarm23_vs_social(xls: pd.ExcelFile) -> dict:
    sheets_alarm_23 = [s for s in ALARM_SHEETS if "(23)" in s]
    selected_sheets = sheets_alarm_23 + SOCIAL_SHEETS

    df_list = []
    for sheet in selected_sheets:
        df = xls.parse(sheet)
        df = drop_xxxwav_rows(df)
        df = drop_trial_separator_rows(df)   # FIX: separator rows are not calls
        df["group"] = "alarm" if sheet in sheets_alarm_23 else "social"
        df_list.append(df)
    df_rf = pd.concat(df_list, ignore_index=True)
    print("Class counts (calls):", df_rf["group"].value_counts().to_dict())

    features = [
        c for c in df_rf.select_dtypes(include="number").columns
        if not is_amp_feature(c)
    ]
    X_rf = df_rf[features].copy()
    X_rf = X_rf.dropna(axis=1, how="all").fillna(X_rf.mean())

    y_rf = df_rf["group"].map({"alarm": 1, "social": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X_rf, y_rf, test_size=0.25, random_state=42, stratify=y_rf
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred       = rf.predict(X_test)
    accuracy     = accuracy_score(y_test, y_pred)
    bal_accuracy = balanced_accuracy_score(y_test, y_pred)

    print("Classification Report:\n" +
          classification_report(y_test, y_pred, target_names=["social", "alarm"]))
    print(f"Balanced Accuracy: {bal_accuracy:.3%} | Accuracy: {accuracy:.3%}")

    # Confusion matrix -> Fig. 1C
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(
        rf, X_test, y_test,
        display_labels=["social", "alarm"],
        cmap="Blues", ax=ax,
    )
    ax.set_title(
        f"Random Forest Confusion Matrix\nBalanced Accuracy = {bal_accuracy:.2%}"
    )
    fig.tight_layout()
    cm_paths = save_figure(fig, OUTDIR / "fig1c_alarm23_vs_social_confusion_matrix", dpi=300)
    plt.show()

    # Metrics CSV
    pd.DataFrame([{
        "model": "RandomForestClassifier",
        "n_estimators": rf.n_estimators,
        "balanced_accuracy": bal_accuracy,
        "accuracy": accuracy,
    }]).to_csv(OUTDIR / "fig1c_rf_metrics.csv", index=False)

    # SHAP feature importance
    shap_df = None
    try:
        explainer  = shap.TreeExplainer(rf)
        shap_vals  = explainer.shap_values(X_test)
        shap_arr   = extract_shap_array(shap_vals, X_test)
        feat_names = X_test.columns.tolist()
        imp        = np.abs(shap_arr).mean(axis=0)
        shap_df = (
            pd.DataFrame({"feature": feat_names, "importance": imp})
            .sort_values("importance", ascending=False)
        )
        shap_df.to_csv(OUTDIR / "fig1c_shap_feature_importance.csv", index=False)
        print(f"SHAP top feature: {shap_df.iloc[0]['feature']} "
              f"(mean|SHAP|={shap_df.iloc[0]['importance']:.6g})")
    except Exception as e:
        print(f"[Warning] SHAP skipped: {e}")

    return {
        "rf": rf,
        "balanced_accuracy": bal_accuracy,
        "accuracy": accuracy,
        "cm_paths": cm_paths,
        "shap_df": shap_df,
    }


# ---------------- SHARED HELPERS FOR CONDITION CLASSIFIERS ----------------
def _label_net_group(sheet: str):
    if "no-net" in sheet:
        return "no_net"
    if "net" in sheet:
        return "net"
    return None


def _season(sheet: str):
    if "(22)" in sheet:
        return "22"
    if "(23)" in sheet:
        return "23"
    return None


def _load_alarm_rows(xls: pd.ExcelFile, sheets: list) -> pd.DataFrame:
    rows = []
    for sh in sheets:
        df = xls.parse(sh)
        df = drop_xxxwav_rows(df)
        df = drop_trial_separator_rows(df)   # FIX: separator rows are not calls
        df["sheet"]     = sh
        df["season"]    = _season(sh)
        df["net_group"] = _label_net_group(sh)
        if df["net_group"] is not None:
            rows.append(df)
    if not rows:
        raise ValueError("No alarm rows found.")
    return pd.concat(rows, ignore_index=True)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.select_dtypes(include="number").copy()
    X = X[[c for c in X.columns if not is_amp_feature(c)]]
    X = X.dropna(axis=1, how="all").fillna(X.mean())
    return X


def _safe_split(X, y, test_size=0.25, random_state=42):
    if y.value_counts().min() < 2:
        raise ValueError(f"Insufficient class samples: {y.value_counts().to_dict()}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if y_tr.nunique() < 2 or y_te.nunique() < 2:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.5, random_state=random_state, stratify=y
        )
    return X_tr, X_te, y_tr, y_te


def _train_eval_plot_rf(
    X, y, labels_order, file_prefix, title_suffix,
    *, test_size=0.25, random_state=42,
    run_permutation_test=True, n_perm=1000,
) -> dict:
    X_train, X_test, y_train, y_test = _safe_split(
        X, y, test_size=test_size, random_state=random_state
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred  = rf.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    perm_res = None
    if run_permutation_test:
        perm_res = permutation_test_rf_holdout(
            X=X, y=y,
            test_size=test_size, random_state=random_state,
            n_perm=n_perm, n_estimators=100,
        )
        pd.DataFrame({"balanced_accuracy": perm_res["null_ba"]}).to_csv(
            OUTDIR / f"{file_prefix}_permutation_null_ba.csv", index=False
        )
        pd.DataFrame([{
            "title": title_suffix,
            "ba_observed": float(bal_acc),
            "perm_null_mean": perm_res["null_mean"],
            "perm_null_95":   perm_res["null_95"],
            "perm_null_99":   perm_res["null_99"],
            "perm_p_value":   perm_res["p_value"],
            "perm_n":         perm_res["n_perm"],
        }]).to_csv(OUTDIR / f"{file_prefix}_permutation_summary.csv", index=False)
        print(
            f"Permutation ({n_perm}x): "
            f"BA={bal_acc:.3%} | null_mean={perm_res['null_mean']:.3%} | "
            f"null_95={perm_res['null_95']:.3%} | p={perm_res['p_value']:.4g}"
        )

    # Confusion matrix
    true_counts = {0: int((y_test == 0).sum()), 1: int((y_test == 1).sum())}
    pred_counts = {0: int((y_pred == 0).sum()), 1: int((y_pred == 1).sum())}

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(
        rf, X_test, y_test,
        display_labels=labels_order, cmap="Blues", ax=ax,
    )
    ax.set_yticklabels([
        f"{labels_order[0]} (n={true_counts[0]})",
        f"{labels_order[1]} (n={true_counts[1]})",
    ])
    ax.set_xticklabels([
        f"{labels_order[0]} (n={pred_counts[0]})",
        f"{labels_order[1]} (n={pred_counts[1]})",
    ])
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.tick_params(labelsize=10)

    p_txt = f", p={perm_res['p_value']:.3g}" if perm_res else ""
    fig.suptitle(
        f"Random Forest - {title_suffix}\nBalanced Accuracy = {bal_acc:.2%}{p_txt}",
        fontsize=14, y=0.98,
    )
    fig.subplots_adjust(top=0.88)
    fig.tight_layout()
    cm_paths = save_figure(fig, OUTDIR / f"{file_prefix}_confusion_matrix", dpi=300)
    plt.show()

    print(classification_report(y_test, y_pred, target_names=labels_order, zero_division=0))
    print(f"Balanced Accuracy: {bal_acc:.3%} | Accuracy: {acc:.3%}")

    # SHAP
    try:
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X_test)
        shap_arr  = extract_shap_array(shap_vals, X_test)
        feat_names = X_test.columns.tolist()
        imp = np.abs(shap_arr).mean(axis=0)
        shap_df = (
            pd.DataFrame({"feature": feat_names, "importance": imp})
            .sort_values("importance", ascending=False)
        )
        shap_df.to_csv(
            OUTDIR / f"{file_prefix}_shap_feature_importance.csv", index=False
        )
        print(f"SHAP top: {shap_df.iloc[0]['feature']} "
              f"(mean|SHAP|={shap_df.iloc[0]['importance']:.6g})")
    except Exception as e:
        print(f"[Warning] SHAP skipped: {e}")

    # Metrics CSV
    metrics = {
        "title": title_suffix,
        "balanced_accuracy": float(bal_acc),
        "accuracy": float(acc),
        "n_features": int(X.shape[1]),
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    if perm_res:
        metrics.update({
            "perm_n":         int(n_perm),
            "perm_null_mean": perm_res["null_mean"],
            "perm_null_95":   perm_res["null_95"],
            "perm_p_value":   perm_res["p_value"],
        })
    pd.DataFrame([metrics]).to_csv(
        OUTDIR / f"{file_prefix}_metrics.csv", index=False
    )

    return {"rf": rf, "balanced_accuracy": bal_acc, "cm_paths": cm_paths, "perm_res": perm_res}


# ---------------- RF #2-#5: ALARM CONDITION CLASSIFIERS ----------------
def rf_condition_classifiers(xls: pd.ExcelFile) -> None:
    df_alarm = _load_alarm_rows(xls, ALARM_SHEETS)

    # RF #2 - Net vs No-Net, Season 2022
    df_22 = df_alarm[df_alarm["season"] == "22"].copy()
    X_22  = _build_features(df_22)
    y_22  = df_22["net_group"].map({"no_net": 0, "net": 1})
    _train_eval_plot_rf(
        X_22, y_22, ["no_net", "net"],
        file_prefix="rf_net_nonet_22",
        title_suffix="Net vs No-Net (Season 2022)",
    )

    # RF #3 - Net vs No-Net, Season 2023
    df_23 = df_alarm[df_alarm["season"] == "23"].copy()
    X_23  = _build_features(df_23)
    y_23  = df_23["net_group"].map({"no_net": 0, "net": 1})
    _train_eval_plot_rf(
        X_23, y_23, ["no_net", "net"],
        file_prefix="rf_net_nonet_23",
        title_suffix="Net vs No-Net (Season 2023)",
    )

    # RF #4 - Net vs No-Net, Pooled 2022 + 2023
    X_pool = _build_features(df_alarm)
    y_pool = df_alarm["net_group"].map({"no_net": 0, "net": 1})
    _train_eval_plot_rf(
        X_pool, y_pool, ["no_net", "net"],
        file_prefix="rf_net_nonet_22_23_pooled",
        title_suffix="Net vs No-Net (Pooled 2022+2023)",
    )

    # RF #5 - Season 2022 Net vs All Other Alarm
    mask_pos = (df_alarm["season"] == "22") & (df_alarm["net_group"] == "net")
    df_22net = df_alarm.copy()
    df_22net["label"] = np.where(mask_pos, "net22", "other_alarm")
    X_22n = _build_features(df_22net)
    y_22n = df_22net["label"].map({"other_alarm": 0, "net22": 1})
    _train_eval_plot_rf(
        X_22n, y_22n, ["other_alarm", "net22"],
        file_prefix="rf_net22_vs_other_alarm",
        title_suffix="Season 2022 Net vs All Other Alarm",
    )


# ---------------- MAIN ----------------
def main() -> None:
    print(f"Outputs -> {OUTDIR}\n")

    if not XLSX_PATH.exists():
        raise FileNotFoundError(
            f"Excel file not found: {XLSX_PATH}\n"
            "Place 'all-vocalizations.xlsx' next to this script."
        )

    xls = pd.ExcelFile(XLSX_PATH)

    print("=" * 60)
    print("RF #1 - Alarm (2023) vs Social  [-> Fig. 1C]")
    print("=" * 60)
    rf_alarm23_vs_social(xls)

    print("\n" + "=" * 60)
    print("RF #2-#5 - Alarm Condition Classifiers  [text results]")
    print("=" * 60)
    rf_condition_classifiers(xls)


if __name__ == "__main__":
    main()
