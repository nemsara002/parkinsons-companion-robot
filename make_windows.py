"""
Convert per-frame features.csv into windowed (temporal) features.

Run this AFTER extract_features.py, in the same folder as features.csv.

Why: a single frame can't distinguish "falling" from "bending over" -
both can show a high torso_angle momentarily. What actually separates
a fall is the PATTERN over a short time window: a rapid, large, sustained
change followed by settling into a horizontal resting posture.

This script slides a window of WINDOW_SIZE frames (with STEP overlap)
over each clip and computes summary statistics per window:
  - mean / max / std of each raw feature within the window
  - the NET CHANGE (last - first) of torso_angle and hip y-position
    within the window, which captures "how much did posture change"

Output: windowed_features.csv  (one row per window, label = majority
label of frames in that window, clip_id preserved for proper splitting)
"""

import pandas as pd
import numpy as np

INPUT_CSV = "features.csv"
OUTPUT_CSV = "windowed_features.csv"

WINDOW_SIZE = 15   # reverted to this setting - it outperformed the smaller window in testing
STEP = 5

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


def make_windows_for_clip(clip_df):
    clip_df = clip_df.sort_values("frame").reset_index(drop=True)
    n = len(clip_df)
    rows = []

    if n < WINDOW_SIZE:
        # clip too short for even one full window; skip it
        return rows

    for start in range(0, n - WINDOW_SIZE + 1, STEP):
        window = clip_df.iloc[start:start + WINDOW_SIZE]

        row = {}
        for feat in RAW_FEATURES:
            vals = window[feat].values
            row[f"{feat}_mean"] = np.mean(vals)
            row[f"{feat}_max"] = np.max(vals)
            row[f"{feat}_std"] = np.std(vals)

        # Net change across the window = how much posture shifted overall.
        # This is the key "fall signature": large + fast + one-directional
        # change, vs. small oscillations during normal activity.
        row["torso_angle_net_change"] = (
            window["torso_angle"].iloc[-1] - window["torso_angle"].iloc[0]
        )
        row["head_hip_dist_net_change"] = (
            window["head_hip_dist"].iloc[-1] - window["head_hip_dist"].iloc[0]
        )
        # bbox_area/aspect_ratio net change specifically targets falls
        # toward/away from the camera, where torso_angle barely moves but
        # the body's projected silhouette still shrinks and changes shape.
        row["bbox_area_net_change"] = (
            window["bbox_area"].iloc[-1] - window["bbox_area"].iloc[0]
        )
        row["bbox_area_raw_net_change"] = (
            window["bbox_area_raw"].iloc[-1] - window["bbox_area_raw"].iloc[0]
        )
        row["aspect_ratio_net_change"] = (
            window["aspect_ratio"].iloc[-1] - window["aspect_ratio"].iloc[0]
        )
        # Overall collapse magnitude across the whole window, complementing
        # transition_speed_max (the steepest INSTANTANEOUS drop). A real
        # fall should show both: a high peak speed AND a large net drop
        # from window start to end (the person stays down, not a brief dip).
        row["uprightness_score_net_change"] = (
            window["uprightness_score"].iloc[-1] - window["uprightness_score"].iloc[0]
        )

        # Majority label within the window (windows can straddle a
        # fall/non-fall boundary; majority vote keeps labeling consistent)
        label_counts = window["label"].value_counts()
        row["label"] = label_counts.idxmax()
        row["is_mixed_window"] = len(label_counts) > 1
        row["clip_id"] = window["clip_id"].iloc[0]
        row["window_start_frame"] = window["frame"].iloc[0]

        rows.append(row)

    return rows


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} frame-level rows from {df['clip_id'].nunique()} clips")

    all_windows = []
    skipped_clips = 0
    for clip_id, clip_df in df.groupby("clip_id"):
        windows = make_windows_for_clip(clip_df)
        if not windows:
            skipped_clips += 1
            continue
        all_windows.extend(windows)

    out_df = pd.DataFrame(all_windows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSkipped {skipped_clips} clips (too short for one window)")
    print(f"Wrote {len(out_df)} windowed rows to {OUTPUT_CSV}")
    print(out_df["label"].value_counts())


if __name__ == "__main__":
    main()