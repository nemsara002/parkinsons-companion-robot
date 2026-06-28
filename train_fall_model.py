"""
Fall Detection - Random Forest Training & Evaluation
Run this on your own machine, in the same folder as features.csv.

Splits by clip_id (not by frame) to avoid data leakage, trains a
class-weight-balanced Random Forest, and reports metrics with a
focus on FALL recall (since missing a real fall is the worst failure mode).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer, recall_score

INPUT_CSV = "windowed_features.csv"
MODEL_OUT = "fall_detector_rf.pkl"
DECISION_THRESHOLD = 0.25  # FALL probability cutoff, chosen to hit >=85%
# recall target from the proposal. Trades precision (~0.56) for recall,
# which is the correct safety priority: a false alarm costs a harmless
# voice check-in ("are you okay?"), while a missed real fall has no
# second chance. False-alarm filtering happens downstream via the voice
# check-in module, NOT by raising this threshold back up.

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
    print(f"Loaded {len(df)} rows, {df['clip_id'].nunique()} clips")
    print(df["label"].value_counts())

    # ---- Split by CLIP, not by frame, to avoid leakage ----
    clip_labels = df.groupby("clip_id")["label"].first()  # one label per clip
    train_clips, test_clips = train_test_split(
        clip_labels.index,
        test_size=0.2,
        stratify=clip_labels.values,   # keep FALL/ADL ratio similar in both splits
        random_state=42,
    )

    train_df = df[df["clip_id"].isin(train_clips)]
    test_df = df[df["clip_id"].isin(test_clips)]

    print(f"\nTrain: {len(train_df)} rows from {len(train_clips)} clips")
    print(f"Test:  {len(test_df)} rows from {len(test_clips)} clips")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["label"]

    # ---- Tune hyperparameters with clip-aware cross-validation ----
    # GroupKFold ensures no clip's windows ever appear in both train and
    # validation within a fold, preserving the no-leakage discipline from
    # the train/test split, but now applied during tuning too.
    fall_recall_scorer = make_scorer(recall_score, pos_label="FALL")

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [8, 12, None],
        "min_samples_leaf": [1, 3, 5],
    }

    groups = train_df["clip_id"].values
    cv = GroupKFold(n_splits=5)

    base_clf = RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=-1
    )

    print("\nRunning grid search (this may take a minute)...")
    grid = GridSearchCV(
        base_clf, param_grid, scoring=fall_recall_scorer,
        cv=cv, n_jobs=-1,
    )
    grid.fit(X_train, y_train, groups=groups)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV FALL-recall: {grid.best_score_:.3f}")

    clf = grid.best_estimator_

    # ---- Evaluate at the chosen DEPLOYMENT threshold, not the sklearn default of 0.5 ----
    fall_class_idx = list(clf.classes_).index("FALL")
    test_probs = clf.predict_proba(X_test)[:, fall_class_idx]
    y_pred = np.where(test_probs >= DECISION_THRESHOLD, "FALL", "ADL")

    print(f"\n(Evaluating at decision threshold = {DECISION_THRESHOLD}, "
          f"not the sklearn default of 0.5 - see DECISION_THRESHOLD comment above)")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=3))

    print("=== Confusion Matrix ===")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"Labels order: {labels}")
    print(cm)

    fall_f1 = f1_score(y_test, y_pred, pos_label="FALL")
    print(f"\nFALL-class F1 score: {fall_f1:.3f}")

    # ---- Feature importance ----
    print("\n=== Feature Importances ===")
    importances = sorted(zip(FEATURE_COLS, clf.feature_importances_),
                          key=lambda x: -x[1])
    for name, imp in importances:
        print(f"  {name:25s} {imp:.3f}")

    # ---- Save model + threshold together ----
    # Bundling these prevents the live deployment script from accidentally
    # using a stale or mismatched threshold if this gets retrained later.
    joblib.dump({"model": clf, "threshold": DECISION_THRESHOLD,
                 "feature_cols": FEATURE_COLS}, MODEL_OUT)
    print(f"\nModel + threshold ({DECISION_THRESHOLD}) saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()