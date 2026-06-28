"""
Fall Detection - Feature Extraction Script
Run this on your own machine (not on the Pi).

Folder structure expected:
fall_dataset/
├── raw_videos/
│   ├── fall/
│   │   ├── fall-01-cam0.mp4
│   │   ├── fall-02-cam0.mp4
│   │   └── ...
│   └── adl/
│       ├── adl-01-cam0.mp4
│       └── ...

Output: features.csv  (one row per frame, with engineered features + label + clip_id)
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import math

mp_pose = mp.solutions.pose

# ---- CONFIG ----
DATASET_DIR = "fall_dataset/raw_videos"
OUTPUT_CSV = "features.csv"
LABEL_MAP = {"fall": "FALL", "adl": "ADL"}  # folder name -> label

# Mediapipe landmark indices we care about
LM = mp_pose.PoseLandmark


def uprightness_score(t_angle_deg, bbox_area_norm, area_calib_median):
    """Combined 0-1 'how upright is this person right now' score, used to
    derive transition_speed (how fast someone collapses from standing to
    lying down). Built from BOTH torso_angle and bbox_area, not torso_angle
    alone, because lateral falls show large torso_angle change while
    depth-axis (toward-camera) falls show little angle change but DO show
    bbox_area shrinking as the body's projected silhouette collapses
    (confirmed via visual + numeric inspection of FALL_025 earlier in this
    project). A torso_angle-only duration feature would inherit the same
    blind spot bbox features were originally added to fix.

    1.0 = fully upright, 0.0 = fully collapsed/horizontal. area_calib_median
    is the clip's typical standing bbox_area (normalized), used so the
    area-shrinkage component is scaled per-person rather than using a
    fixed absolute cutoff.
    """
    angle_component = 1.0 - min(t_angle_deg / 90.0, 1.0)  # 90deg+ = fully down
    area_ratio = bbox_area_norm / (area_calib_median + 1e-6)
    area_component = min(area_ratio, 1.0)  # area shrinking below standing = falling
    # Average the two signals; either one dropping pulls the score down,
    # so a fall detectable via EITHER mechanism still produces a clear drop.
    return (angle_component + area_component) / 2.0


def angle_between(p1, p2, p3):
    """Angle at p2, formed by p1-p2-p3, in degrees."""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def torso_angle(landmarks):
    """Angle of the torso (shoulder-hip line) relative to vertical.
    ~0 deg = upright, ~90 deg = horizontal (lying down)."""
    sh = landmarks[LM.LEFT_SHOULDER.value]
    hip = landmarks[LM.LEFT_HIP.value]
    dx = hip.x - sh.x
    dy = hip.y - sh.y
    angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
    return angle


def torso_length(landmarks):
    """Euclidean shoulder-to-hip distance in normalized image coordinates.
    Used as a per-person 'ruler' to make scale-dependent features
    (bbox_h, bbox_area, etc.) distance-invariant: a person's projected
    size depends heavily on how far they are from the camera, but their
    own torso_length scales by exactly the same factor, so dividing one
    by the other cancels out camera distance."""
    l_sh = landmarks[LM.LEFT_SHOULDER.value]
    r_sh = landmarks[LM.RIGHT_SHOULDER.value]
    l_hip = landmarks[LM.LEFT_HIP.value]
    r_hip = landmarks[LM.RIGHT_HIP.value]
    sh_x, sh_y = (l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2
    hip_x, hip_y = (l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2
    return math.sqrt((hip_x - sh_x) ** 2 + (hip_y - sh_y) ** 2)


def bbox_features(landmarks):
    """Bounding box of all visible landmarks. Captures falls where the
    body moves mainly along the camera's depth axis (toward/away from
    camera) rather than sideways within the image plane - these falls
    show little torso_angle change but DO show the body's projected
    silhouette shrinking and changing shape (tall/narrow -> short/wide)."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    bbox_h = max(ys) - min(ys)
    bbox_w = max(xs) - min(xs)
    bbox_area = bbox_h * bbox_w
    aspect_ratio = bbox_w / (bbox_h + 1e-6)
    return bbox_h, bbox_w, bbox_area, aspect_ratio


def extract_landmark_row(landmarks, frame_idx, prev_hip_y, prev_torso_angle,
                          prev_bbox_h_norm, prev_bbox_area_norm,
                          prev_bbox_h_raw, prev_bbox_area_raw, prev_frame_idx,
                          fps, reference_scale, area_calib_median,
                          prev_uprightness=None):
    nose = landmarks[LM.NOSE.value]
    l_sh = landmarks[LM.LEFT_SHOULDER.value]
    r_sh = landmarks[LM.RIGHT_SHOULDER.value]
    l_hip = landmarks[LM.LEFT_HIP.value]
    r_hip = landmarks[LM.RIGHT_HIP.value]
    l_knee = landmarks[LM.LEFT_KNEE.value]
    r_knee = landmarks[LM.RIGHT_KNEE.value]
    l_ankle = landmarks[LM.LEFT_ANKLE.value]

    hip_center_y = (l_hip.y + r_hip.y) / 2
    shoulder_center_y = (l_sh.y + r_sh.y) / 2

    t_angle = torso_angle(landmarks)
    knee_angle = angle_between(l_hip, l_knee, l_ankle)
    head_hip_dist_raw = abs(nose.y - hip_center_y)
    bbox_h_raw, bbox_w_raw, bbox_area_raw, aspect_ratio = bbox_features(landmarks)

    # Keep BOTH raw (frame-fraction) and normalized (body-relative)
    # versions. Raw features carry real signal when camera distance is
    # consistent across clips (true for URFD's fixed-camera dataset) -
    # discarding them lost the model's strongest feature in testing.
    # Normalized features add robustness to camera distance variation
    # (important for live deployment, where distance is less controlled).
    # Random Forest feature importance lets the model lean on whichever
    # signal actually helps, rather than us picking one a priori.
    head_hip_dist_norm = head_hip_dist_raw / reference_scale
    bbox_h_norm = bbox_h_raw / reference_scale
    bbox_w_norm = bbox_w_raw / reference_scale
    bbox_area_norm = bbox_area_raw / (reference_scale ** 2)

    # Velocity must be normalized by the ACTUAL elapsed time between the
    # previous successfully-detected frame and this one, not by a fixed
    # 1-frame assumption. MediaPipe drops frames during occlusion/fast
    # motion (exactly when falls happen), so gaps of several frames are
    # common. Using fps alone here would distort velocity precisely on
    # the frames that matter most.
    hip_velocity = 0.0
    angular_velocity = 0.0
    bbox_h_velocity = 0.0      # rate of vertical-extent collapse (normalized)
    bbox_area_velocity = 0.0   # rate of silhouette-area collapse (normalized)
    bbox_h_velocity_raw = 0.0
    bbox_area_velocity_raw = 0.0
    if prev_hip_y is not None and prev_frame_idx is not None:
        frame_gap = max(frame_idx - prev_frame_idx, 1)
        elapsed_seconds = frame_gap / fps
        hip_velocity = abs(hip_center_y - prev_hip_y) / reference_scale / elapsed_seconds
        angular_velocity = abs(t_angle - prev_torso_angle) / elapsed_seconds
        bbox_h_velocity = abs(bbox_h_norm - prev_bbox_h_norm) / elapsed_seconds
        bbox_area_velocity = abs(bbox_area_norm - prev_bbox_area_norm) / elapsed_seconds
        bbox_h_velocity_raw = abs(bbox_h_raw - prev_bbox_h_raw) / elapsed_seconds
        bbox_area_velocity_raw = abs(bbox_area_raw - prev_bbox_area_raw) / elapsed_seconds

    # ---- Transition-speed feature ----
    # Explicitly encodes "how fast is this person collapsing right now",
    # rather than relying on the Random Forest to infer speed implicitly
    # from several separate velocity/std columns. Built from a combined
    # angle+area uprightness score (not torso_angle alone) so it remains
    # informative for depth-axis falls, where torso_angle barely changes
    # but the body's projected silhouette area still shrinks quickly.
    current_uprightness = uprightness_score(t_angle, bbox_area_norm, area_calib_median)
    transition_speed = 0.0
    if prev_uprightness is not None and prev_hip_y is not None and prev_frame_idx is not None:
        frame_gap_u = max(frame_idx - prev_frame_idx, 1)
        elapsed_u = frame_gap_u / fps
        # Positive when COLLAPSING (uprightness decreasing); falls produce
        # large positive values, sitting/lying down slowly produces small
        # ones, and standing back up produces negative values we floor at 0
        # since we only care about collapse speed, not recovery speed.
        drop = prev_uprightness - current_uprightness
        transition_speed = max(drop, 0.0) / elapsed_u

    row = {
        "frame": frame_idx,
        "torso_angle": t_angle,
        "knee_angle": knee_angle,
        "head_hip_dist": head_hip_dist_norm,
        "head_hip_dist_raw": head_hip_dist_raw,
        "shoulder_hip_y_diff": abs(shoulder_center_y - hip_center_y) / reference_scale,
        "hip_velocity": hip_velocity,
        "angular_velocity": angular_velocity,
        "bbox_h": bbox_h_norm,
        "bbox_w": bbox_w_norm,
        "bbox_area": bbox_area_norm,
        "bbox_h_raw": bbox_h_raw,
        "bbox_w_raw": bbox_w_raw,
        "bbox_area_raw": bbox_area_raw,
        "aspect_ratio": aspect_ratio,
        "bbox_h_velocity": bbox_h_velocity,
        "bbox_area_velocity": bbox_area_velocity,
        "bbox_h_velocity_raw": bbox_h_velocity_raw,
        "bbox_area_velocity_raw": bbox_area_velocity_raw,
        "uprightness_score": current_uprightness,
        "transition_speed": transition_speed,
    }
    return (row, hip_center_y, t_angle, bbox_h_norm, bbox_area_norm,
            bbox_h_raw, bbox_area_raw, frame_idx, current_uprightness)


def process_video(video_path, label, clip_id, calibration_frames=10, max_search_frames=150):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ---- Pass 1: calibrate reference_scale ----
    # Many URFD clips open with the person not yet in frame (e.g. walking
    # in from outside the camera's view - confirmed by visual inspection
    # of FALL_025's opening frames earlier). A fixed "first 10 frames"
    # window can land entirely before the person appears, especially in
    # longer ADL clips, wrongly discarding a perfectly good clip. Instead,
    # we search forward (up to max_search_frames) for the first point
    # where a person is reliably detected, THEN average calibration_frames
    # consecutive valid detections from there. This still assumes the
    # person is roughly upright when first detected, which holds for ADL
    # clips and for FALL clips (which begin before the fall happens).
    calib_lengths = []
    calib_areas_raw = []
    frames_scanned = 0
    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while frames_scanned < max_search_frames and len(calib_lengths) < calibration_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                calib_lengths.append(torso_length(results.pose_landmarks.landmark))
                _, _, area_raw, _ = bbox_features(results.pose_landmarks.landmark)
                calib_areas_raw.append(area_raw)
            frames_scanned += 1

    if not calib_lengths:
        print(f"  WARNING: no pose detected in first {max_search_frames} frames "
              f"of {video_path}, skipping clip (cannot calibrate scale)")
        cap.release()
        return []

    reference_scale = float(np.median(calib_lengths))
    if reference_scale < 1e-4:
        print(f"  WARNING: degenerate reference_scale for {video_path}, skipping clip")
        cap.release()
        return []

    # area_calib_median is the clip's typical STANDING bbox_area (normalized
    # by reference_scale), used by uprightness_score() to scale the
    # area-shrinkage component per-person rather than against a fixed
    # absolute cutoff that wouldn't generalize across different distances.
    area_calib_median = float(np.median(calib_areas_raw)) / (reference_scale ** 2)

    # ---- Pass 2: re-open and process the full clip using reference_scale ----
    cap.release()
    cap = cv2.VideoCapture(video_path)

    rows = []
    prev_hip_y = None
    prev_torso_angle = None
    prev_bbox_h_norm = None
    prev_bbox_area_norm = None
    prev_bbox_h_raw = None
    prev_bbox_area_raw = None
    prev_frame_idx = None
    prev_uprightness = None
    frame_idx = 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                (row, prev_hip_y, prev_torso_angle, prev_bbox_h_norm,
                 prev_bbox_area_norm, prev_bbox_h_raw, prev_bbox_area_raw,
                 prev_frame_idx, prev_uprightness) = extract_landmark_row(
                    landmarks, frame_idx, prev_hip_y, prev_torso_angle,
                    prev_bbox_h_norm, prev_bbox_area_norm,
                    prev_bbox_h_raw, prev_bbox_area_raw, prev_frame_idx,
                    fps, reference_scale, area_calib_median, prev_uprightness
                )
                row["label"] = label
                row["clip_id"] = clip_id
                rows.append(row)
            # if no landmarks detected, skip the frame (occlusion/bad detection)

            frame_idx += 1

    cap.release()
    return rows


def main():
    all_rows = []
    clip_counter = 0

    for folder_name, label in LABEL_MAP.items():
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"WARNING: {folder_path} not found, skipping.")
            continue

        video_files = sorted([f for f in os.listdir(folder_path)
                               if f.lower().endswith((".mp4", ".avi"))])

        for vf in video_files:
            clip_counter += 1
            clip_id = f"{label}_{clip_counter:03d}"
            video_path = os.path.join(folder_path, vf)
            print(f"Processing {video_path} -> clip_id={clip_id}, label={label}")

            rows = process_video(video_path, label, clip_id)
            print(f"  extracted {len(rows)} frames with valid pose")
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Wrote {len(df)} rows to {OUTPUT_CSV}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()