"""
Diagnostic: which FALL clips are hardest to detect, and why?
Run in the same folder as windowed_features.csv, AFTER train_fall_model.py
has been run at least once (so fall_detector_rf.pkl exists).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

INPUT_CSV = "windowed_features.csv"
MODEL_PATH = "fall_detector_rf.pkl"

RAW_FEATURES = [
    "torso_angle",
    "knee_angle",
    "head_hip_dist",
    "head_hip_dist_raw",
    "shoulder_hip_y_diff",
    "hip_velocity",
    "angular_velocity",
    "bbox_h",
    "bbox_w",
    "bbox_area",
    "bbox_h_raw",
    "bbox_w_raw",
    "bbox_area_raw",
    "aspect_ratio",
    "bbox_h_velocity",
    "bbox_area_velocity",
    "bbox_h_velocity_raw",
    "bbox_area_velocity_raw",
    "uprightness_score",
    "transition_speed",
]
FEATURE_COLS = (
    [f"{f}_mean" for f in RAW_FEATURES]
    + [f"{f}_max" for f in RAW_FEATURES]
    + [f"{f}_std" for f in RAW_FEATURES]
    + ["torso_angle_net_change", "head_hip_dist_net_change",
       "bbox_area_net_change", "bbox_area_raw_net_change", "aspect_ratio_net_change",
       "uprightness_score_net_change"]
)


def main():
    df = pd.read_csv(INPUT_CSV)
    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]
    deployment_threshold = bundle["threshold"]
    # Use the feature_cols actually saved with the model, not the local
    # FEATURE_COLS constant above - if they ever drift apart, the saved
    # ones are the ground truth for what this specific model expects.
    feature_cols = bundle["feature_cols"]

    clip_labels = df.groupby("clip_id")["label"].first()
    train_clips, test_clips = train_test_split(
        clip_labels.index, test_size=0.2,
        stratify=clip_labels.values, random_state=42,
    )
    test_df = df[df["clip_id"].isin(test_clips)].copy()

    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    # Use the model's ACTUAL deployment threshold (0.25), not sklearn's
    # default 0.5 - per-clip analysis should reflect real deployed
    # behavior, not a different decision rule.
    fall_class_idx = list(clf.classes_).index("FALL")
    probs = clf.predict_proba(X_test)[:, fall_class_idx]
    y_pred = np.where(probs >= deployment_threshold, "FALL", "ADL")
    test_df["predicted"] = y_pred
    test_df["fall_probability"] = probs
    test_df["correct"] = test_df["predicted"] == test_df["label"]

    # ---- Threshold sweep ----
    # Since missing a real fall is worse than a false alarm, lowering the
    # FALL threshold trades some precision for recall - let's see exactly
    # how much, across a range, to put deployment_threshold in context.
    print("=== Threshold sweep (FALL probability cutoff) ===")
    print(f"{'threshold':>10} {'recall':>8} {'precision':>10} {'f1':>8}")
    from sklearn.metrics import precision_score, recall_score, f1_score as f1_score_fn
    for thresh in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]:
        preds_at_thresh = np.where(probs >= thresh, "FALL", "ADL")
        r = recall_score(y_test, preds_at_thresh, pos_label="FALL")
        p = precision_score(y_test, preds_at_thresh, pos_label="FALL")
        f1 = f1_score_fn(y_test, preds_at_thresh, pos_label="FALL")
        marker = "  <- deployment threshold" if abs(thresh - deployment_threshold) < 1e-6 else ""
        print(f"{thresh:>10.2f} {r:>8.3f} {p:>10.3f} {f1:>8.3f}{marker}")
    print()

    print(f"=== Per-clip accuracy on FALL test clips (at deployment threshold = {deployment_threshold}) ===")
    fall_test = test_df[test_df["label"] == "FALL"]
    per_clip = fall_test.groupby("clip_id").agg(
        n_windows=("correct", "size"),
        n_correct=("correct", "sum"),
        mean_transition_speed_max=("transition_speed_max", "mean"),
    )
    per_clip["recall_in_clip"] = per_clip["n_correct"] / per_clip["n_windows"]
    print(per_clip.sort_values("recall_in_clip"))

    print("\n=== Worst-performing FALL clip: feature snapshot ===")
    worst_clip_id = per_clip.sort_values("recall_in_clip").index[0]
    worst_clip = test_df[test_df["clip_id"] == worst_clip_id]
    print(f"Clip: {worst_clip_id}")
    print(worst_clip[["window_start_frame", "torso_angle_max", "bbox_area_raw_max",
                       "transition_speed_max", "uprightness_score_mean",
                       "fall_probability", "predicted", "correct"]].to_string())

    print("\n=== Best-performing FALL clip: feature snapshot (for comparison) ===")
    best_clip_id = per_clip.sort_values("recall_in_clip").index[-1]
    best_clip = test_df[test_df["clip_id"] == best_clip_id]
    print(f"Clip: {best_clip_id}")
    print(best_clip[["window_start_frame", "torso_angle_max", "bbox_area_raw_max",
                      "transition_speed_max", "uprightness_score_mean",
                      "fall_probability", "predicted", "correct"]].to_string())


if __name__ == "__main__":
    main()