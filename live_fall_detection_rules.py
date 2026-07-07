"""
Fall Detection - Rule-Based Live Deployment (USB Webcam / Raspberry Pi)

This script uses a three-condition geometric rule derived from direct
observation at the actual deployment camera geometry, rather than a
machine-learning model trained on a fixed-camera dataset (URFD).

MOTIVATION:
The Random Forest model trained on URFD (chest-height horizontal camera)
did not transfer to this robot's camera geometry (low-angle, upward-tilted)
because the dominant learned features (bbox_w_raw, bbox_area_raw) are
camera-distance-dependent and produce different absolute values at a
different camera height/angle. This is a documented domain-shift problem
in vision-based fall detection (ref: Omobot, 2024).

SOLUTION - THREE-CONDITION RULE:
Instead of training-data-dependent features, we use three
geometry-invariant signals observed directly at the deployment camera:

  Condition 1: torso_angle_max > TORSO_THRESHOLD (body clearly tilted)
  Condition 2: aspect_ratio_max > ASPECT_THRESHOLD (body wider than tall)
  Condition 3: torso_change_rate > SPEED_THRESHOLD (transition was fast)

Conditions 1+2 detect the horizontal state. Condition 3 distinguishes
a fall (fast collapse, ~0.8°/sec aspect_ratio change) from deliberate
lying down (~0.14°/sec), which was identified as the key discriminating
signal by the operator after live testing (5.7x margin between cases).

THRESHOLDS were calibrated empirically from live camera observations:
  - Standing still: torso_angle~12°, aspect_ratio~0.25, change_rate~0°/sec
  - Simulated fall:  torso_angle→87°, aspect_ratio→1.45, change_rate~0.8°/sec
  - Deliberate lie-down: same final posture but change_rate~0.14°/sec

VALIDATION:
These thresholds were verified to produce zero false alarms during
standing/normal activity at the deployment camera geometry, and correctly
triggered during simulated fall testing.

USAGE:
  python3 live_fall_detection_rules.py
  Set CAMERA_INDEX = 0 (or 1/2 if wrong camera opens)
  Set DEBUG_PRINT = False for cleaner output once verified working
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark

# ================================================================
# CONFIG
# ================================================================
WINDOW_SIZE = 15      # frames per window (~0.5sec at 30fps)
STEP = 5              # slide window every N frames
CAMERA_INDEX = 0      # USB webcam index (try 1 or 2 if wrong camera)
DEBUG_PRINT = True    # prints per-window features for monitoring/tuning

# ================================================================
# FALL DETECTION THRESHOLDS
# Calibrated from live camera observations at this deployment geometry:
# camera height ~28cm, upward tilt ~15°, subject distance ~272cm
# ================================================================
TORSO_THRESHOLD = 30.0      # torso_angle_max (degrees) - clearly tilted
ASPECT_THRESHOLD = 0.80     # aspect_ratio_max - body wider than tall
SPEED_THRESHOLD = 30.0      # torso_change_rate (degrees/sec) - fast transition

# Consecutive window confirmation: how many consecutive FALL windows
# required before alerting (reduces false alarms from brief motion spikes)
CONFIRM_WINDOWS = 2

# Baseline sanity check
BASELINE_CHECK_WINDOWS = 6


# ================================================================
# POSE FEATURE FUNCTIONS
# ================================================================

def angle_between(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))


def torso_angle(landmarks):
    """Angle of shoulder-hip line from vertical. 0°=upright, 90°=horizontal."""
    sh = landmarks[LM.LEFT_SHOULDER.value]
    hip = landmarks[LM.LEFT_HIP.value]
    dx = hip.x - sh.x
    dy = hip.y - sh.y
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def bbox_features(landmarks):
    """Bounding box of all landmarks in normalized image coordinates."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    bbox_h = max(ys) - min(ys)
    bbox_w = max(xs) - min(xs)
    aspect_ratio = bbox_w / (bbox_h + 1e-6)
    return bbox_h, bbox_w, aspect_ratio


def landmarks_reliable(landmarks, min_visibility=0.5):
    """Reject frames where key joints are out of frame or occluded."""
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


def extract_frame_features(landmarks):
    """Extract per-frame features used in the rule-based detector."""
    l_hip = landmarks[LM.LEFT_HIP.value]
    bbox_h, bbox_w, aspect_ratio = bbox_features(landmarks)
    t_angle = torso_angle(landmarks)
    return {
        "torso_angle": t_angle,
        "aspect_ratio": aspect_ratio,
    }


# ================================================================
# WINDOW-LEVEL RULE
# ================================================================

def evaluate_window(frame_buffer, fps=30.0):
    """
    Applies the three-condition fall detection rule to a window of frames.
    Returns (is_fall, debug_info_dict).

    Condition 1: torso_angle_max > TORSO_THRESHOLD
    Condition 2: aspect_ratio_max > ASPECT_THRESHOLD
    Condition 3: torso_change_rate > SPEED_THRESHOLD

    Conditions 1+2 together detect a horizontal body position.
    Condition 3 separates fast falls from slow deliberate lying down.
    """
    torso_angles = [f["torso_angle"] for f in frame_buffer]
    aspect_ratios = [f["aspect_ratio"] for f in frame_buffer]

    torso_max = max(torso_angles)
    torso_mean = sum(torso_angles) / len(torso_angles)
    aspect_max = max(aspect_ratios)
    aspect_mean = sum(aspect_ratios) / len(aspect_ratios)

    # Rate of torso angle change over the window (degrees per second)
    elapsed_sec = len(frame_buffer) / fps
    torso_change_rate = abs(torso_angles[-1] - torso_angles[0]) / (elapsed_sec + 1e-6)
    aspect_change_rate = abs(aspect_ratios[-1] - aspect_ratios[0]) / (elapsed_sec + 1e-6)

    c1 = torso_max > TORSO_THRESHOLD
    c2 = aspect_max > ASPECT_THRESHOLD
    c3 = torso_change_rate > SPEED_THRESHOLD

    is_fall = c1 and c2 and c3

    debug = {
        "torso_max": torso_max,
        "torso_mean": torso_mean,
        "aspect_max": aspect_max,
        "aspect_mean": aspect_mean,
        "torso_change_rate": torso_change_rate,
        "aspect_change_rate": aspect_change_rate,
        "c1_torso": c1,
        "c2_aspect": c2,
        "c3_speed": c3,
    }
    return is_fall, debug


# ================================================================
# FALL ALERT HOOK
# ================================================================

def on_fall_detected():
    """
    Hook point for fall alert. Wire your voice check-in module
    and ESP32 alert logic here. Currently just prints a timestamped alert.
    """
    print(f"\n{'='*50}")
    print(f"*** FALL DETECTED at {time.strftime('%H:%M:%S')} ***")
    print(f"{'='*50}\n")
    # TODO: trigger voice check-in ("Are you okay?")
    # TODO: if no response within N seconds, alert via ESP32/GSM


# ================================================================
# MAIN LOOP
# ================================================================

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {CAMERA_INDEX}.")
        print("Try changing CAMERA_INDEX to 1 or 2.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Camera opened. FPS={fps:.1f}")
    print(f"Fall detection thresholds:")
    print(f"  torso_angle_max  > {TORSO_THRESHOLD}°")
    print(f"  aspect_ratio_max > {ASPECT_THRESHOLD}")
    print(f"  torso_change_rate > {SPEED_THRESHOLD}°/sec")
    print(f"  Confirmation: {CONFIRM_WINDOWS} consecutive FALL windows required")
    print()

    frame_buffer = deque(maxlen=WINDOW_SIZE)
    frames_since_last_window = 0
    consecutive_fall_windows = 0
    baseline_probs = []
    baseline_done = False
    alert_cooldown = 0  # frames since last alert (prevents repeated alerts)

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        print("Starting live detection. Press 'q' to quit.")
        print(f"(First {BASELINE_CHECK_WINDOWS} windows are a baseline check -"
              " stand normally.)\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: failed to read camera frame")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                if not landmarks_reliable(landmarks):
                    if DEBUG_PRINT:
                        print("  [skipped: low-visibility landmark]")
                    frames_since_last_window = 0  # reset step counter
                else:
                    feats = extract_frame_features(landmarks)
                    frame_buffer.append(feats)
                    frames_since_last_window += 1

                    if (len(frame_buffer) == WINDOW_SIZE
                            and frames_since_last_window >= STEP):
                        frames_since_last_window = 0
                        is_fall, debug = evaluate_window(list(frame_buffer), fps)

                        if DEBUG_PRINT:
                            c_str = (f"C1={'✓' if debug['c1_torso'] else '✗'} "
                                     f"C2={'✓' if debug['c2_aspect'] else '✗'} "
                                     f"C3={'✓' if debug['c3_speed'] else '✗'}")
                            print(f"  [window] "
                                  f"torso_max={debug['torso_max']:.1f}° "
                                  f"aspect_max={debug['aspect_max']:.2f} "
                                  f"rate={debug['torso_change_rate']:.1f}°/s "
                                  f"| {c_str} "
                                  f"-> {'FALL' if is_fall else 'ADL'}")

                        # Baseline sanity check
                        if not baseline_done:
                            baseline_probs.append(
                                1.0 if is_fall else 0.0
                            )
                            if len(baseline_probs) >= BASELINE_CHECK_WINDOWS:
                                baseline_done = True
                                fall_rate = sum(baseline_probs) / len(baseline_probs)
                                print(f"\n[baseline check] FALL rate during "
                                      f"first {BASELINE_CHECK_WINDOWS} windows = "
                                      f"{fall_rate:.2f}")
                                if fall_rate > 0.3:
                                    print(
                                        "  WARNING: frequent FALL detections during "
                                        "normal activity. Consider adjusting thresholds "
                                        "or camera position.\n"
                                    )
                                else:
                                    print("  Baseline looks reasonable. Continuing.\n")

                        # Consecutive window confirmation
                        if is_fall:
                            consecutive_fall_windows += 1
                        else:
                            consecutive_fall_windows = 0

                        if (consecutive_fall_windows >= CONFIRM_WINDOWS
                                and alert_cooldown <= 0):
                            on_fall_detected()
                            consecutive_fall_windows = 0
                            alert_cooldown = int(fps * 10)  # 10sec cooldown
            else:
                # No pose detected
                consecutive_fall_windows = 0

            if alert_cooldown > 0:
                alert_cooldown -= 1

            cv2.imshow("Fall Detection - Rule Based (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
