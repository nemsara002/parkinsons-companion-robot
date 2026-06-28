"""
Fall Detection - Evaluation Plots for Report
Run this in the same folder as windowed_features.csv and fall_detector_rf.pkl.

Generates:
  - confusion_matrix.png       (counts, at deployment threshold)
  - roc_curve.png              (ROC curve + AUC)
  - precision_recall_curve.png (PR curve, more informative than ROC for
                                 imbalanced classes - shows the actual
                                 precision/recall tradeoff at your threshold)
  - feature_importance.png     (top 15 features, horizontal bar chart)
  - threshold_sweep.png        (recall/precision/F1 vs threshold, with
                                 your chosen threshold marked)

All figures are saved at 300 DPI, suitable for direct inclusion in a
Word/PDF report.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save files
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay
)

INPUT_CSV = "windowed_features.csv"
MODEL_PATH = "fall_detector_rf.pkl"
OUTPUT_DIR = "."  # change if you want plots in a subfolder

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    threshold = bundle["threshold"]
    feature_cols = bundle["feature_cols"]
    fall_idx = list(model.classes_).index("FALL")

    df = pd.read_csv(INPUT_CSV)

    # ---- Recreate the EXACT same train/test split used during training ----
    # Must match train_fall_model.py's split (same test_size, random_state,
    # and stratify-by-clip-label logic) or these plots won't reflect the
    # model's actual held-out performance.
    clip_labels = df.groupby("clip_id")["label"].first()
    train_clips, test_clips = train_test_split(
        clip_labels.index, test_size=0.2,
        stratify=clip_labels.values, random_state=42,
    )
    test_df = df[df["clip_id"].isin(test_clips)]

    X_test = test_df[feature_cols]
    y_test = test_df["label"]
    y_true_binary = (y_test == "FALL").astype(int)

    probs = model.predict_proba(X_test)[:, fall_idx]
    y_pred = np.where(probs >= threshold, "FALL", "ADL")

    print(f"Test set: {len(test_df)} windows from {len(test_clips)} clips")
    print(f"Deployment threshold: {threshold}")
    print(f"Recall (FALL):    {recall_score(y_test, y_pred, pos_label='FALL'):.3f}")
    print(f"Precision (FALL): {precision_score(y_test, y_pred, pos_label='FALL'):.3f}")
    print(f"F1 (FALL):        {f1_score(y_test, y_pred, pos_label='FALL'):.3f}")

    # ==================================================================
    # 1. Confusion Matrix
    # ==================================================================
    cm = confusion_matrix(y_test, y_pred, labels=["ADL", "FALL"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ADL", "FALL"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix (threshold = {threshold})")
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
    plt.close(fig)
    print("\nSaved confusion_matrix.png")

    # ==================================================================
    # 2. ROC Curve
    # ==================================================================
    fpr, tpr, roc_thresholds = roc_curve(y_true_binary, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
    # mark the operating point at the chosen deployment threshold
    op_idx = np.argmin(np.abs(roc_thresholds - threshold))
    ax.scatter([fpr[op_idx]], [tpr[op_idx]], color="red", zorder=5, s=60,
               label=f"Operating point (thr={threshold})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve - Fall Detection")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
    plt.close(fig)
    print("Saved roc_curve.png")

    # ==================================================================
    # 3. Precision-Recall Curve (more informative given class imbalance)
    # ==================================================================
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true_binary, probs)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(recalls, precisions, color="#16a34a", lw=2, label="Precision-Recall curve")
    op_idx_pr = np.argmin(np.abs(pr_thresholds - threshold)) if len(pr_thresholds) > 0 else 0
    ax.scatter([recalls[op_idx_pr]], [precisions[op_idx_pr]], color="red", zorder=5,
               s=60, label=f"Operating point (thr={threshold})")
    ax.axhline(y_true_binary.mean(), color="gray", lw=1, linestyle="--",
               label=f"Baseline (FALL prevalence = {y_true_binary.mean():.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Fall Detection")
    ax.legend(loc="lower left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.savefig(f"{OUTPUT_DIR}/precision_recall_curve.png")
    plt.close(fig)
    print("Saved precision_recall_curve.png")

    # ==================================================================
    # 4. Feature Importance (top 15)
    # ==================================================================
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_features = importances.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#f97316" if "_raw" in name else "#2563eb" for name in top_features.index]
    ax.barh(top_features.index, top_features.values, color=colors)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 15 Feature Importances\n(orange = raw/absolute-scale, blue = normalized)")
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
    plt.close(fig)
    print("Saved feature_importance.png")

    # ==================================================================
    # 5. Threshold Sweep
    # ==================================================================
    sweep_thresholds = np.linspace(0.05, 0.95, 37)
    sweep_recall, sweep_precision, sweep_f1 = [], [], []
    for t in sweep_thresholds:
        pred_t = np.where(probs >= t, "FALL", "ADL")
        sweep_recall.append(recall_score(y_test, pred_t, pos_label="FALL", zero_division=0))
        sweep_precision.append(precision_score(y_test, pred_t, pos_label="FALL", zero_division=0))
        sweep_f1.append(f1_score(y_test, pred_t, pos_label="FALL", zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(sweep_thresholds, sweep_recall, label="Recall", color="#dc2626", lw=2)
    ax.plot(sweep_thresholds, sweep_precision, label="Precision", color="#2563eb", lw=2)
    ax.plot(sweep_thresholds, sweep_f1, label="F1", color="#16a34a", lw=2)
    ax.axvline(threshold, color="gray", linestyle="--", lw=1.5,
               label=f"Deployment threshold = {threshold}")
    ax.set_xlabel("Decision Threshold (P(FALL) cutoff)")
    ax.set_ylabel("Score")
    ax.set_title("Recall / Precision / F1 vs Decision Threshold")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_ylim(-0.02, 1.02)
    plt.savefig(f"{OUTPUT_DIR}/threshold_sweep.png")
    plt.close(fig)
    print("Saved threshold_sweep.png")

    print("\nAll plots saved. Ready to insert into report.")


if __name__ == "__main__":
    main()
