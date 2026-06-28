"""
Fall Detection - Real-Time Deployment (Raspberry Pi)

Runs MediaPipe Pose on a live camera feed, maintains a rolling window of
the last WINDOW_SIZE frames' raw features (mirroring the exact pipeline
used in extract_features.py + make_windows.py + train_fall_model.py),
and runs the trained Random Forest on each new window to detect falls.

IMPORTANT: WINDOW_SIZE and STEP here MUST match what was used during
training (currently 15 and 5) or the model will see out-of-distribution
input shapes/statistics. If you ever change them in make_windows.py,
update them here too.

On a FALL detection, this prints/logs an event. Hook your voice
check-in module and ESP32 alert logic into the `on_fall_detected()`
function below - this script does not implement those, since this
project's fall-detector and voice/alert subsystems are separate modules.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import math
import time
from collections import deque

mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark

# ---- CONFIG: must match training pipeline exactly ----
WINDOW_SIZE = 15
STEP = 5
MODEL_PATH = "fall_detector_rf.pkl"
CAMERA_INDEX = 0   # adjust if using a USB camera other than the default
DEBUG_PRINT = True  # set False once values look sane - prints raw feature values per window

RAW_FEATURES = [
    "torso_angle", "knee_angle", "head_hip_dist", "head_hip_dist_raw",
    "shoulder_hip_y_diff", "hip_velocity", "angular_velocity",
    "bbox_h", "bbox_w", "bbox_area", "bbox_h_raw", "bbox_w_raw", "bbox_area_raw",
    "aspect_ratio", "bbox_h_velocity", "bbox_area_velocity",
    "bbox_h_velocity_raw", "bbox_area_velocity_raw",
]
CALIBRATION_FRAMES = 10     # how many valid detections to average for the ruler
CALIBRATION_MAX_SEARCH = 150  # give up if no person found within this many frames

# Baseline sanity check: the first few windows after calibration are
# assumed to be normal/idle activity (the person hasn't fallen yet right
# after starting the script). If the model's P(FALL) on THESE windows is
# already high, that's a sign the camera setup (angle/height/lens) produces
# feature statistics the model doesn't recognize as "normal" - i.e. a
# domain mismatch versus training data - rather than an actual fall.
# This won't catch every such issue, but flags the common case early
# instead of silently producing false alarms throughout a session.
BASELINE_CHECK_WINDOWS = 6
BASELINE_WARN_THRESHOLD = 0.5  # if median baseline P(FALL) exceeds this, warn


def angle_between(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def torso_angle(landmarks):
    sh = landmarks[LM.LEFT_SHOULDER.value]
    hip = landmarks[LM.LEFT_HIP.value]
    dx = hip.x - sh.x
    dy = hip.y - sh.y
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def torso_length(landmarks):
    """Shoulder-to-hip Euclidean distance - the per-person 'ruler' used
    to normalize scale-dependent features, mirroring extract_features.py.
    Must match that implementation exactly or normalized features will
    be on a different scale than what the model was trained on."""
    l_sh = landmarks[LM.LEFT_SHOULDER.value]
    r_sh = landmarks[LM.RIGHT_SHOULDER.value]
    l_hip = landmarks[LM.LEFT_HIP.value]
    r_hip = landmarks[LM.RIGHT_HIP.value]
    sh_x, sh_y = (l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2
    hip_x, hip_y = (l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2
    return math.sqrt((hip_x - sh_x) ** 2 + (hip_y - sh_y) ** 2)


def bbox_features(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    bbox_h = max(ys) - min(ys)
    bbox_w = max(xs) - min(xs)
    bbox_area = bbox_h * bbox_w
    aspect_ratio = bbox_w / (bbox_h + 1e-6)
    return bbox_h, bbox_w, bbox_area, aspect_ratio


def landmarks_reliable(landmarks, min_visibility=0.5):
    """MediaPipe attaches a visibility score (0-1) to each landmark,
    indicating confidence the point is actually visible/in-frame. When a
    joint (e.g. an ankle) is out of frame or occluded, MediaPipe still
    returns a coordinate - often an extrapolated guess that can fall
    outside the normal 0-1 normalized range entirely. Training data
    (URFD, fixed wall camera) always kept the full body in frame, so
    this was never an issue there. A live webcam framing that crops
    feet/ankles will produce exactly this failure mode, so we check
    visibility on the key joints our features depend on and reject
    the frame if any are unreliable, rather than feeding the model
    garbage coordinates it never saw in training."""
    key_indices = [
        LM.NOSE.value, LM.LEFT_SHOULDER.value, LM.RIGHT_SHOULDER.value,
        LM.LEFT_HIP.value, LM.RIGHT_HIP.value, LM.LEFT_KNEE.value,
        LM.LEFT_ANKLE.value,
    ]
    for idx in key_indices:
        vis = getattr(landmarks[idx], "visibility", 1.0)
        if vis is not None and vis < min_visibility:
            return False
    return True


def extract_raw_features(landmarks, prev_state, elapsed_seconds, reference_scale):
    """Computes one frame's raw feature dict, mirroring extract_features.py.
    prev_state is a dict holding the previous frame's hip_y/torso_angle/
    bbox_h/bbox_area (normalized AND raw), or None after a tracking gap.
    reference_scale is this person's calibrated torso_length, used to
    normalize scale-dependent features exactly as training did."""
    nose = landmarks[LM.NOSE.value]
    l_sh = landmarks[LM.LEFT_SHOULDER.value]
    r_sh = landmarks[LM.RIGHT_SHOULDER.value]
    l_hip = landmarks[LM.LEFT_HIP.value]
    r_hip = landmarks[LM.RIGHT_HIP.value]
    l_knee = landmarks[LM.LEFT_KNEE.value]
    l_ankle = landmarks[LM.LEFT_ANKLE.value]

    hip_center_y = (l_hip.y + r_hip.y) / 2
    shoulder_center_y = (l_sh.y + r_sh.y) / 2
    t_angle = torso_angle(landmarks)
    knee_angle = angle_between(l_hip, l_knee, l_ankle)
    head_hip_dist_raw = abs(nose.y - hip_center_y)
    bbox_h_raw, bbox_w_raw, bbox_area_raw, aspect_ratio = bbox_features(landmarks)

    head_hip_dist_norm = head_hip_dist_raw / reference_scale
    bbox_h_norm = bbox_h_raw / reference_scale
    bbox_w_norm = bbox_w_raw / reference_scale
    bbox_area_norm = bbox_area_raw / (reference_scale ** 2)

    hip_velocity = 0.0
    angular_velocity = 0.0
    bbox_h_velocity = 0.0
    bbox_area_velocity = 0.0
    bbox_h_velocity_raw = 0.0
    bbox_area_velocity_raw = 0.0
    if prev_state is not None and elapsed_seconds > 0:
        hip_velocity = abs(hip_center_y - prev_state["hip_y"]) / reference_scale / elapsed_seconds
        angular_velocity = abs(t_angle - prev_state["torso_angle"]) / elapsed_seconds
        bbox_h_velocity = abs(bbox_h_norm - prev_state["bbox_h_norm"]) / elapsed_seconds
        bbox_area_velocity = abs(bbox_area_norm - prev_state["bbox_area_norm"]) / elapsed_seconds
        bbox_h_velocity_raw = abs(bbox_h_raw - prev_state["bbox_h_raw"]) / elapsed_seconds
        bbox_area_velocity_raw = abs(bbox_area_raw - prev_state["bbox_area_raw"]) / elapsed_seconds

    features = {
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
    }
    new_state = {
        "hip_y": hip_center_y, "torso_angle": t_angle,
        "bbox_h_norm": bbox_h_norm, "bbox_area_norm": bbox_area_norm,
        "bbox_h_raw": bbox_h_raw, "bbox_area_raw": bbox_area_raw,
    }
    return features, new_state


def window_to_feature_vector(window_frames, feature_cols):
    """Mirrors make_windows.py: mean/max/std per raw feature, plus
    torso_angle and head_hip_dist and bbox_area/aspect_ratio net change.
    window_frames is a list of raw-feature dicts, oldest first."""
    row = {}
    for feat in RAW_FEATURES:
        vals = np.array([f[feat] for f in window_frames])
        row[f"{feat}_mean"] = np.mean(vals)
        row[f"{feat}_max"] = np.max(vals)
        row[f"{feat}_std"] = np.std(vals)

    row["torso_angle_net_change"] = (
        window_frames[-1]["torso_angle"] - window_frames[0]["torso_angle"]
    )
    row["head_hip_dist_net_change"] = (
        window_frames[-1]["head_hip_dist"] - window_frames[0]["head_hip_dist"]
    )
    row["bbox_area_net_change"] = (
        window_frames[-1]["bbox_area"] - window_frames[0]["bbox_area"]
    )
    row["bbox_area_raw_net_change"] = (
        window_frames[-1]["bbox_area_raw"] - window_frames[0]["bbox_area_raw"]
    )
    row["aspect_ratio_net_change"] = (
        window_frames[-1]["aspect_ratio"] - window_frames[0]["aspect_ratio"]
    )

    # Order matters: must match FEATURE_COLS order from training exactly.
    final_vector = [row[c] for c in feature_cols]

    if DEBUG_PRINT:
        print(f"  [window] bbox_h_mean={row['bbox_h_mean']:.3f} "
              f"bbox_h_raw_mean={row['bbox_h_raw_mean']:.4f} "
              f"torso_angle_mean={row['torso_angle_mean']:.1f} "
              f"aspect_ratio_mean={row['aspect_ratio_mean']:.2f}")

    return pd.DataFrame([final_vector], columns=feature_cols)


def calibrate_reference_scale(cap, pose):
    """Searches forward through the live feed for the first reliable
    detections of the person and averages their torso_length to get a
    'ruler' for normalizing scale-dependent features. Mirrors the
    calibration pass in extract_features.py's process_video(). Ask the
    person to stand fully in frame, upright, during this step."""
    print(f"Calibrating... please stand fully in frame, upright, facing the camera.")
    calib_lengths = []
    frames_scanned = 0
    while frames_scanned < CALIBRATION_MAX_SEARCH and len(calib_lengths) < CALIBRATION_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks and landmarks_reliable(results.pose_landmarks.landmark):
            tl = torso_length(results.pose_landmarks.landmark)
            calib_lengths.append(tl)
            if DEBUG_PRINT:
                bbox_h_raw, bbox_w_raw, bbox_area_raw, ar = bbox_features(
                    results.pose_landmarks.landmark
                )
                print(f"  [calib sample {len(calib_lengths)}] torso_length={tl:.4f}  "
                      f"bbox_h_raw={bbox_h_raw:.4f}  bbox_h_raw/torso_length={bbox_h_raw/tl:.2f}")
        cv2.imshow("Fall Detection (press q to quit)", frame)
        cv2.waitKey(1)
        frames_scanned += 1

    if not calib_lengths:
        print("ERROR: could not calibrate - no reliable pose detected. "
              "Check camera framing (full body visible?) and lighting.")
        return None

    reference_scale = float(np.median(calib_lengths))
    if DEBUG_PRINT:
        print(f"  [calib] all samples: {[round(x,4) for x in calib_lengths]}")
        print(f"  [calib] min={min(calib_lengths):.4f} max={max(calib_lengths):.4f} "
              f"median={reference_scale:.4f}")
    print(f"Calibration complete. reference_scale = {reference_scale:.4f} "
          f"(from {len(calib_lengths)} samples)")
    return reference_scale


def on_fall_detected(fall_probability):
    """Hook point: wire this into the voice check-in / alert subsystem.
    This function intentionally does nothing else here - this script's
    job is detection only."""
    print(f"*** FALL DETECTED (probability={fall_probability:.2f}) "
          f"at {time.strftime('%H:%M:%S')} ***")
    # TODO: trigger voice check-in module here
    # TODO: if no response within N seconds, send alert via ESP32/GSM


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    threshold = bundle["threshold"]
    feature_cols = bundle["feature_cols"]
    fall_idx = list(model.classes_).index("FALL")

    print(f"Loaded model. Decision threshold = {threshold}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {CAMERA_INDEX}")
        return

    frame_buffer = deque(maxlen=WINDOW_SIZE)
    frames_since_last_window = 0
    prev_state = None
    prev_time = None
    baseline_probs = []
    baseline_check_done = False

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        reference_scale = calibrate_reference_scale(cap, pose)
        if reference_scale is None or reference_scale < 1e-4:
            print("Calibration failed - exiting. Re-run once positioned in frame.")
            cap.release()
            cv2.destroyAllWindows()
            return

        print("Starting live detection. Press 'q' to quit.")
        print(f"(First {BASELINE_CHECK_WINDOWS} windows are treated as a baseline "
              f"sanity check - stand/move normally, don't simulate a fall yet.)")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: failed to read camera frame")
                break

            now = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                if not landmarks_reliable(landmarks):
                    # Key joint(s) out of frame/low-confidence - treat like
                    # a missed detection rather than feeding the model
                    # coordinates it never saw in training (see
                    # landmarks_reliable() docstring for why this matters).
                    if DEBUG_PRINT:
                        print("  [skipped frame: low-visibility landmark detected]")
                    prev_state = None
                    prev_time = None
                else:
                    elapsed = (now - prev_time) if prev_time is not None else 0.0
                    feats, prev_state = extract_raw_features(
                        landmarks, prev_state, elapsed, reference_scale
                    )
                    prev_time = now
                    frame_buffer.append(feats)
                    frames_since_last_window += 1

                    # Run inference every STEP frames once we have a full window,
                    # mirroring the sliding-window logic from make_windows.py
                    if len(frame_buffer) == WINDOW_SIZE and frames_since_last_window >= STEP:
                        frames_since_last_window = 0
                        X = window_to_feature_vector(list(frame_buffer), feature_cols)
                        fall_prob = model.predict_proba(X)[0, fall_idx]

                        label = "FALL" if fall_prob >= threshold else "ADL"
                        print(f"  P(FALL)={fall_prob:.2f} -> {label}")

                        if not baseline_check_done:
                            baseline_probs.append(fall_prob)
                            if len(baseline_probs) >= BASELINE_CHECK_WINDOWS:
                                baseline_check_done = True
                                baseline_median = float(np.median(baseline_probs))
                                print(f"\n[baseline check] median P(FALL) during "
                                      f"first {BASELINE_CHECK_WINDOWS} windows = "
                                      f"{baseline_median:.2f}")
                                if baseline_median >= BASELINE_WARN_THRESHOLD:
                                    print(
                                        "  WARNING: baseline P(FALL) is high during "
                                        "normal activity. This usually means the "
                                        "camera's angle/height/distance differs "
                                        "enough from the training setup that the "
                                        "model doesn't recognize this as 'normal'. "
                                        "Check camera mounting (height, tilt, full "
                                        "body visible) and re-run calibration. "
                                        "Detections from here on may be unreliable "
                                        "until this is resolved.\n"
                                    )
                                else:
                                    print("  Baseline looks reasonable. Continuing.\n")

                        if label == "FALL":
                            on_fall_detected(fall_prob)
            else:
                # Pose lost (occlusion, person left frame, etc).
                # Reset velocity baseline but DON'T clear frame_buffer -
                # a brief gap shouldn't throw away an otherwise-valid window;
                # only velocity needs a fresh reference point.
                prev_state = None
                prev_time = None

            cv2.imshow("Fall Detection (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()