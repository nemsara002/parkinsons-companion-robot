"""
Diagnostic script - run this in the same folder as features.csv.
Helps us see WHY fall recall is stuck, before blindly tuning parameters.
"""

import pandas as pd

df = pd.read_csv("features.csv")

print("=== Per-clip frame counts ===")
print(df.groupby(["clip_id", "label"]).size().groupby("label").describe())

print("\n=== Feature stats by label ===")
feature_cols = ["torso_angle", "knee_angle", "head_hip_dist",
                 "shoulder_hip_y_diff", "hip_velocity", "angular_velocity"]
print(df.groupby("label")[feature_cols].describe().T)

print("\n=== Overlap check: torso_angle distribution ===")
# A fall should show torso_angle skewing toward horizontal (~70-90deg)
# at SOME point in the clip. Let's see the max torso_angle reached per FALL clip
# vs per ADL clip.
max_angle_per_clip = df.groupby(["clip_id", "label"])["torso_angle"].max().reset_index()
print(max_angle_per_clip.groupby("label")["torso_angle"].describe())

print("\n=== Max hip_velocity reached per clip (capped at 99th percentile to remove jitter outliers) ===")
cap = df["hip_velocity"].quantile(0.99)
df["hip_velocity_capped"] = df["hip_velocity"].clip(upper=cap)
max_vel_per_clip = df.groupby(["clip_id", "label"])["hip_velocity_capped"].max().reset_index()
print(max_vel_per_clip.groupby("label")["hip_velocity_capped"].describe())

print(f"\n(velocity capped at 99th percentile = {cap:.3f} units/sec to remove pose-jitter outliers)")
