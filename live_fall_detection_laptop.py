"""
Fall Detection - Real-Time Deployment (LAPTOP / USB WEBCAM VERSION)

Use this version on the Raspberry Pi with a USB webcam, OR on a laptop.
It uses cv2.VideoCapture (standard OpenCV webcam interface) instead of
picamera2, so it works with any UVC-compatible USB camera on any platform.

All detection logic (calibration, feature extraction, windowing,
transition_speed, uprightness_score) is identical to the Pi CSI version.
The only difference is how frames are captured.

IMPORTANT: WINDOW_SIZE and STEP must match what was used during training
(currently 15 and 5). If you change them in make_windows.py, update here too.
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
CAMERA_INDEX = 0   # change to 1 or 2 if wrong camera opens
DEBUG_PRINT = True

# ---- DEPLOYMENT THRESHOLD OVERRIDE ----
# Set this to None to use the threshold saved in fall_detector_rf.pkl (0.25,
# tuned for URFD camera geometry). If your camera geometry differs from URFD
# (different height, angle, distance), set this to a value measured from your
# actual ADL baseline - typically (your_adl_95th_percentile + small_margin).
# Run with ADL_CALIBRATION_MODE = True first to measure your baseline, then
# set this value accordingly.
DEPLOYMENT_THRESHOLD_OVERRIDE = None  # e.g. set to 0.93 after measuring baseline

# ---- ADL CALIBRATION MODE ----
# Set to True to record P(FALL) values during normal activity to a CSV file,
# so you can measure your actual ADL baseline at this camera geometry.
# Run for 60+ seconds of normal activity (standing, walking, sitting, bending)
# WITHOUT simulating any falls, then set to False and analyze adl_baseline.csv
# to find the right DEPLOYMENT_THRESHOLD_OVERRIDE value.
ADL_CALIBRATION_MODE = False
ADL_CALIBRATION_OUTPUT = "adl_baseline.csv"

# ---- HOMOGRAPHY CORRECTION ----
# Warps frames from your camera's low-angle geometry to approximate
# URFD's chest-height horizontal camera geometry, so MediaPipe sees
# the body from a similar perspective to what the model was trained on.
# Computed from 4 measured pixel-to-world correspondences:
#   right_foot=(209,476), left_foot=(258,476),
#   left_shoulder=(285,141), right_shoulder=(193,143)
# at camera height=28cm, tilt=15deg, subject distance=272cm.
# Set APPLY_HOMOGRAPHY = False to disable and compare results.
APPLY_HOMOGRAPHY = True
HOMOGRAPHY_MATRIX = np.array([
    [1.31266057,  0.49787326, -343.16555842],
    [0.05019921,  1.49489735, -186.22274293],
    [0.00012996,  0.00149162,    1.00000000],
])
IMG_WIDTH, IMG_HEIGHT = 640, 480

# Baseline sanity check constants
CALIBRATION_FRAMES = 10
CALIBRATION_MAX_SEARCH = 150
BASELINE_CHECK_WINDOWS = 6
BASELINE_WARN_THRESHOLD = 0.5


# ---- Feature helper functions ----

def angle_between(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))


def torso_angle(landmarks):
    sh = landmarks[LM.LEFT_SHOULDER.value]
    hip = landmarks[LM.LEFT_HIP.value]
    dx = hip.x - sh.x
    dy = hip.y - sh.y
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def torso_length(landmarks):
    l_sh = landmarks[LM.LEFT_SHOULDER.value]
    r_sh = landmarks[LM.RIGHT_SHOULDER.value]
    l_hip = landmarks[LM.LEFT_HIP.value]
    r_hip = landmarks[LM.RIGHT_HIP.value]
    sh_x = (l_sh.x + r_sh.x) / 2
    sh_y = (l_sh.y + r_sh.y) / 2
    hip_x = (l_hip.x + r_hip.x) / 2
    hip_y = (l_hip.y + r_hip.y) / 2
    return math.sqrt((hip_x - sh_x) ** 2 + (hip_y - sh_y) ** 2)


def bbox_features(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    bbox_h = max(ys) - min(ys)
    bbox_w = max(xs) - min(xs)
    bbox_area = bbox_h * bbox_w
    aspect_ratio = bbox_w / (bbox_h + 1e-6)
    return bbox_h, bbox_w, bbox_area, aspect_ratio


def uprightness_score(t_angle_deg, bbox_area_norm, area_calib_median):
    """Combined 0-1 uprightness metric. 1.0=standing, 0.0=lying flat.
    Combines torso_angle and bbox_area so depth-axis falls (where
    torso_angle barely changes) are still captured via area shrinkage."""
    angle_component = 1.0 - min(t_angle_deg / 90.0, 1.0)
    area_ratio = bbox_area_norm / (area_calib_median + 1e-6)
    area_component = min(area_ratio, 1.0)
    return (angle_component + area_component) / 2.0


def landmarks_reliable(landmarks, min_visibility=0.5):
    """Reject frames where key joints are out of frame or poorly visible,
    to avoid feeding the model extrapolated/garbage landmark coordinates."""
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


def extract_raw_features(landmarks, prev_state, elapsed_seconds,
                          reference_scale, area_calib_median):
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

    bbox_h_norm = bbox_h_raw / reference_scale
    bbox_w_norm = bbox_w_raw / reference_scale
    bbox_area_norm = bbox_area_raw / (reference_scale ** 2)
    head_hip_dist_norm = head_hip_dist_raw / reference_scale

    current_uprightness = uprightness_score(t_angle, bbox_area_norm, area_calib_median)

    hip_velocity = angular_velocity = 0.0
    bbox_h_velocity = bbox_area_velocity = 0.0
    bbox_h_velocity_raw = bbox_area_velocity_raw = 0.0
    transition_speed = 0.0

    if prev_state is not None and elapsed_seconds > 0:
        hip_velocity = abs(hip_center_y - prev_state["hip_y"]) / reference_scale / elapsed_seconds
        angular_velocity = abs(t_angle - prev_state["torso_angle"]) / elapsed_seconds
        bbox_h_velocity = abs(bbox_h_norm - prev_state["bbox_h_norm"]) / elapsed_seconds
        bbox_area_velocity = abs(bbox_area_norm - prev_state["bbox_area_norm"]) / elapsed_seconds
        bbox_h_velocity_raw = abs(bbox_h_raw - prev_state["bbox_h_raw"]) / elapsed_seconds
        bbox_area_velocity_raw = abs(bbox_area_raw - prev_state["bbox_area_raw"]) / elapsed_seconds
        drop = prev_state["uprightness"] - current_uprightness
        transition_speed = max(drop, 0.0) / elapsed_seconds

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
        "uprightness_score": current_uprightness,
        "transition_speed": transition_speed,
    }
    new_state = {
        "hip_y": hip_center_y, "torso_angle": t_angle,
        "bbox_h_norm": bbox_h_norm, "bbox_area_norm": bbox_area_norm,
        "bbox_h_raw": bbox_h_raw, "bbox_area_raw": bbox_area_raw,
        "uprightness": current_uprightness,
    }
    return features, new_state


def window_to_feature_vector(window_frames, feature_cols):
    RAW_FEATURES = [
        "torso_angle", "knee_angle", "head_hip_dist", "head_hip_dist_raw",
        "shoulder_hip_y_diff", "hip_velocity", "angular_velocity",
        "bbox_h", "bbox_w", "bbox_area", "bbox_h_raw", "bbox_w_raw", "bbox_area_raw",
        "aspect_ratio", "bbox_h_velocity", "bbox_area_velocity",
        "bbox_h_velocity_raw", "bbox_area_velocity_raw",
        "uprightness_score", "transition_speed",
    ]
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
    row["uprightness_score_net_change"] = (
        window_frames[-1]["uprightness_score"] - window_frames[0]["uprightness_score"]
    )

    final_vector = [row[c] for c in feature_cols]

    if DEBUG_PRINT:
        print(f"  [window] bbox_h_mean={row['bbox_h_mean']:.3f} "
              f"bbox_h_raw_mean={row['bbox_h_raw_mean']:.4f} "
              f"torso_angle_mean={row['torso_angle_mean']:.1f} "
              f"aspect_ratio_mean={row['aspect_ratio_mean']:.2f} "
              f"transition_speed_max={row['transition_speed_max']:.3f}")

    return pd.DataFrame([final_vector], columns=feature_cols)


def calibrate_reference_scale(cap, pose):
    print("Calibrating... please stand fully in frame, upright, facing the camera.")
    calib_lengths = []
    calib_areas_raw = []
    frames_scanned = 0

    while frames_scanned < CALIBRATION_MAX_SEARCH and len(calib_lengths) < CALIBRATION_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        # Webcam gives BGR — convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Apply homography correction to match URFD camera geometry
        if APPLY_HOMOGRAPHY:
            frame_rgb = cv2.warpPerspective(frame_rgb, HOMOGRAPHY_MATRIX,
                                            (IMG_WIDTH, IMG_HEIGHT))
        results = pose.process(frame_rgb)
        if results.pose_landmarks and landmarks_reliable(results.pose_landmarks.landmark):
            tl = torso_length(results.pose_landmarks.landmark)
            calib_lengths.append(tl)
            bbox_h_raw, bbox_w_raw, bbox_area_raw, ar = bbox_features(
                results.pose_landmarks.landmark
            )
            calib_areas_raw.append(bbox_area_raw)
            if DEBUG_PRINT:
                print(f"  [calib sample {len(calib_lengths)}] "
                      f"torso_length={tl:.4f}  "
                      f"bbox_h_raw={bbox_h_raw:.4f}  "
                      f"bbox_h_raw/torso_length={bbox_h_raw/tl:.2f}")
        cv2.imshow("Fall Detection (press q to quit)", frame)
        cv2.waitKey(1)
        frames_scanned += 1

    if not calib_lengths:
        print("ERROR: could not calibrate - no reliable pose detected. "
              "Check camera framing (full body visible?) and lighting.")
        return None, None

    reference_scale = float(np.median(calib_lengths))
    area_calib_median = float(np.median(calib_areas_raw)) / (reference_scale ** 2)

    if DEBUG_PRINT:
        print(f"  [calib] median torso_length = {reference_scale:.4f}")
        print(f"  [calib] area_calib_median (normalized) = {area_calib_median:.4f}")

    print(f"Calibration complete. reference_scale = {reference_scale:.4f}")
    return reference_scale, area_calib_median


def on_fall_detected(fall_probability):
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

    # Apply deployment threshold override if set
    if DEPLOYMENT_THRESHOLD_OVERRIDE is not None:
        threshold = DEPLOYMENT_THRESHOLD_OVERRIDE
        print(f"Using DEPLOYMENT_THRESHOLD_OVERRIDE = {threshold}")
    else:
        print(f"Using saved threshold = {threshold}")

    if ADL_CALIBRATION_MODE:
        import csv
        calib_log = open(ADL_CALIBRATION_OUTPUT, "w", newline="")
        calib_writer = csv.writer(calib_log)
        calib_writer.writerow(["timestamp", "fall_probability"])
        print(f"ADL CALIBRATION MODE: logging P(FALL) to {ADL_CALIBRATION_OUTPUT}")
        print("Perform NORMAL ACTIVITIES ONLY (no falls) for 60+ seconds, then press q.")
    else:
        calib_log = None
        calib_writer = None

    print(f"Loaded model. Decision threshold = {threshold}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {CAMERA_INDEX}. "
              f"Try changing CAMERA_INDEX to 1 or 2.")
        return

    frame_buffer = deque(maxlen=WINDOW_SIZE)
    frames_since_last_window = 0
    prev_state = None
    prev_time = None
    baseline_probs = []
    baseline_check_done = False

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5) as pose:

        reference_scale, area_calib_median = calibrate_reference_scale(cap, pose)
        if (reference_scale is None or reference_scale < 1e-4
                or area_calib_median is None or area_calib_median < 1e-6):
            print("Calibration failed - exiting.")
            cap.release()
            cv2.destroyAllWindows()
            return

        print("Starting live detection. Press 'q' to quit.")
        print(f"(First {BASELINE_CHECK_WINDOWS} windows: stand normally, "
              f"don't simulate a fall yet - baseline sanity check.)")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: failed to read camera frame")
                break

            now = time.time()
            # Webcam gives BGR — convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply homography correction to match URFD camera geometry
            if APPLY_HOMOGRAPHY:
                frame_rgb = cv2.warpPerspective(frame_rgb, HOMOGRAPHY_MATRIX,
                                                (IMG_WIDTH, IMG_HEIGHT))
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                if not landmarks_reliable(landmarks):
                    if DEBUG_PRINT:
                        print("  [skipped: low-visibility landmark]")
                    prev_state = None
                    prev_time = None
                else:
                    elapsed = (now - prev_time) if prev_time is not None else 0.0
                    feats, prev_state = extract_raw_features(
                        landmarks, prev_state, elapsed,
                        reference_scale, area_calib_median
                    )
                    prev_time = now
                    frame_buffer.append(feats)
                    frames_since_last_window += 1

                    if len(frame_buffer) == WINDOW_SIZE and frames_since_last_window >= STEP:
                        frames_since_last_window = 0
                        X = window_to_feature_vector(list(frame_buffer), feature_cols)
                        fall_prob = model.predict_proba(X)[0, fall_idx]

                        label = "FALL" if fall_prob >= threshold else "ADL"
                        print(f"  P(FALL)={fall_prob:.2f} -> {label}")

                        # Log to CSV in calibration mode
                        if ADL_CALIBRATION_MODE and calib_writer:
                            calib_writer.writerow([time.strftime('%H:%M:%S'), f"{fall_prob:.4f}"])

                        if not baseline_check_done:
                            baseline_probs.append(fall_prob)
                            if len(baseline_probs) >= BASELINE_CHECK_WINDOWS:
                                baseline_check_done = True
                                baseline_median = float(np.median(baseline_probs))
                                print(f"\n[baseline check] median P(FALL) = "
                                      f"{baseline_median:.2f}")
                                if baseline_median >= BASELINE_WARN_THRESHOLD:
                                    print(
                                        "  WARNING: baseline P(FALL) is high. "
                                        "Camera angle/height/distance may differ "
                                        "from training data. Check mounting and "
                                        "re-run calibration if needed.\n"
                                    )
                                else:
                                    print("  Baseline looks reasonable. Continuing.\n")

                        if label == "FALL":
                            on_fall_detected(fall_prob)
            else:
                prev_state = None
                prev_time = None

            # Show warped frame if homography is active, raw frame otherwise
            display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) if APPLY_HOMOGRAPHY else frame
            cv2.imshow("Fall Detection (press q to quit)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if ADL_CALIBRATION_MODE and calib_log:
        calib_log.close()
        print(f"\nADL calibration data saved to {ADL_CALIBRATION_OUTPUT}")
        print("Now run this to find your threshold:")
        print(f"  python3 -c \"")
        print(f"  import pandas as pd")
        print(f"  df = pd.read_csv('{ADL_CALIBRATION_OUTPUT}')")
        print(f"  print('95th percentile:', df.fall_probability.quantile(0.95))")
        print(f"  print('99th percentile:', df.fall_probability.quantile(0.99))")
        print(f"  print('max:', df.fall_probability.max())")
        print(f"  \"")
        print("Set DEPLOYMENT_THRESHOLD_OVERRIDE to (99th percentile + 0.02) in the script.")


if __name__ == "__main__":
    main()
