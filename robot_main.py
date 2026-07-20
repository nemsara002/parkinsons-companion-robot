"""
robot_main.py

Unifies three previously separate pieces into one shared pipeline:
  - patient_identifier.py    (who is the patient, where are they -- bbox)
  - live_fall_detection_rules.py  (validated 3-condition fall detector)
  - person_follow_uL_uR.py   (differential-drive steering)

DESIGN DECISION (per discussion): fall detection and steering are BOTH
gated to the identified patient only -- matches your proposal's scope
("identify the intended patient... maintain visual tracking of the
identified user"). Bystanders/visitors in frame are not monitored for
falls or followed.

HOW THE SHARING WORKS (important -- read before modifying):
  1. PatientTargetLock.update(frame) runs each frame -- cheap most of the
     time (CSRT tracking; face recognition only periodically), gives us
     target_bbox = where the identified patient roughly is.
  2. mediapipe Pose runs ONCE per frame on the FULL, UNCROPPED frame --
     exactly as live_fall_detection_rules.py does it. This is deliberate:
     cropping to the patient bbox before running Pose would distort the
     normalized x/y landmark coordinates (bbox aspect ratio != frame
     aspect ratio), which would silently invalidate the fall detector's
     empirically-calibrated thresholds (torso_angle, aspect_ratio use
     these normalized coords directly). DO NOT crop the frame before
     pose.process() for this reason.
  3. We check whether the Pose skeleton's own bounding box overlaps
     target_bbox enough (landmarks_belong_to_target()). This is what
     "gates to the patient" -- if Pose locked onto some other person
     in frame, their landmarks are ignored for both steering and fall
     detection this frame, same as your existing low-visibility skip
     logic already does for unreliable landmarks.
  4. If the overlap check passes, the SAME landmarks feed both the
     steering calculation and the fall-detection window buffer -- one
     Pose inference per frame serves both purposes.

Fall detection logic (torso_angle, bbox_features, landmarks_reliable,
extract_frame_features, evaluate_window, on_fall_detected, and all
thresholds) is imported directly from live_fall_detection_rules.py
rather than retyped here, so there is exactly one source of truth for
the validated detector and no risk of transcription drift between the
standalone script and this unified one.
"""

import time
from collections import deque

import cv2
import mediapipe as mp

from patient_identifier import PatientTargetLock, draw_status
from live_fall_detection_rules import (
    torso_angle,
    bbox_features,
    landmarks_reliable,
    extract_frame_features,
    evaluate_window,
    on_fall_detected,
    TORSO_THRESHOLD,
    ASPECT_THRESHOLD,
    SPEED_THRESHOLD,
    WINDOW_SIZE,
    STEP,
    CONFIRM_WINDOWS,
    BASELINE_CHECK_WINDOWS,
)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
LM = mp_pose.PoseLandmark


# ---------------- CONFIGURATION ----------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# NOTE: this should match whatever resolution/aspect ratio your fall
# detector's thresholds were actually calibrated at. Angle/aspect-ratio
# math uses landmark coordinates normalized to frame width/height
# separately, so if the camera's ASPECT RATIO here differs from what was
# used during calibration, the thresholds may no longer be valid even
# though absolute resolution itself doesn't matter. If your fall
# detection testing was done without explicitly setting these, check
# what resolution the webcam defaulted to and match it here.

DEBUG_PRINT = True

# Steering control (unchanged from person_follow_uL_uR.py)
STEER_DEADZONE = 0.05
STEER_GAIN = 0.8
TURN_SPEED_LIMIT = 0.6

TARGET_SIZE_MIN = 0.35
TARGET_SIZE_MAX = 0.45
FORWARD_SPEED = 0.5
BACKWARD_SPEED = 0.4

SMOOTHING_WINDOW = 5

IDLE_UL, IDLE_UR = 0.0, 0.0

# Minimum fraction of the Pose skeleton's bounding box that must overlap
# the identified patient's tracked bbox for these landmarks to be trusted
# as "this is the patient". Tune if legitimate patient landmarks are
# being rejected (lower it) or a bystander's landmarks are slipping
# through (raise it).
MIN_PATIENT_OVERLAP = 0.4


# ---------------- STEERING HELPERS (from person_follow_uL_uR.py) -------

def compute_body_center_x(landmarks):
    l_sh = landmarks[LM.LEFT_SHOULDER]
    r_sh = landmarks[LM.RIGHT_SHOULDER]
    l_hip = landmarks[LM.LEFT_HIP]
    r_hip = landmarks[LM.RIGHT_HIP]
    return (l_sh.x + r_sh.x + l_hip.x + r_hip.x) / 4.0


def compute_body_size(landmarks):
    l_sh = landmarks[LM.LEFT_SHOULDER]
    r_sh = landmarks[LM.RIGHT_SHOULDER]
    l_hip = landmarks[LM.LEFT_HIP]
    r_hip = landmarks[LM.RIGHT_HIP]

    shoulder_width = abs(l_sh.x - r_sh.x)
    hip_width = abs(l_hip.x - r_hip.x)

    sh_mid_y = (l_sh.y + r_sh.y) / 2
    hip_mid_y = (l_hip.y + r_hip.y) / 2
    torso_height = abs(hip_mid_y - sh_mid_y)

    return (0.3 * shoulder_width) + (0.3 * hip_width) + (0.4 * torso_height)


def smooth(buffer, new_value):
    buffer.append(new_value)
    return sum(buffer) / len(buffer)


def compute_motor_signals(center_x, body_size):
    error_x = center_x - 0.5

    if abs(error_x) < STEER_DEADZONE:
        steer = 0.0
    else:
        steer = error_x * STEER_GAIN
        steer = max(-TURN_SPEED_LIMIT, min(TURN_SPEED_LIMIT, steer))

    if body_size < TARGET_SIZE_MIN:
        forward = FORWARD_SPEED
    elif body_size > TARGET_SIZE_MAX:
        forward = -BACKWARD_SPEED
    else:
        forward = 0.0

    uL = forward + steer
    uR = forward - steer

    uL = max(-1.0, min(1.0, uL))
    uR = max(-1.0, min(1.0, uR))

    return uL, uR


# ---------------- PATIENT-GATING HELPER ----------------

def landmarks_belong_to_target(landmarks, target_bbox, frame_shape,
                                min_overlap=MIN_PATIENT_OVERLAP):
    """
    Computes the Pose skeleton's own bounding box (in pixel coords) and
    checks what fraction of it overlaps the identified patient's tracked
    bbox. Returns True if the overlap is high enough to trust these
    landmarks as belonging to the patient, not a bystander.
    """
    frame_h, frame_w = frame_shape[:2]

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    skel_left = min(xs) * frame_w
    skel_right = max(xs) * frame_w
    skel_top = min(ys) * frame_h
    skel_bottom = max(ys) * frame_h

    skel_area = max(0.0, skel_right - skel_left) * max(0.0, skel_bottom - skel_top)
    if skel_area <= 0:
        return False

    tx, ty, tw, th = target_bbox
    t_left, t_right = tx, tx + tw
    t_top, t_bottom = ty, ty + th

    inter_left = max(skel_left, t_left)
    inter_right = min(skel_right, t_right)
    inter_top = max(skel_top, t_top)
    inter_bottom = min(skel_bottom, t_bottom)

    inter_w = max(0.0, inter_right - inter_left)
    inter_h = max(0.0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h

    overlap_fraction = inter_area / skel_area
    return overlap_fraction >= min_overlap


# ---------------- MAIN LOOP ----------------

def main():
    print("[main] Loading PatientTargetLock...")
    lock = PatientTargetLock()

    print("[main] Opening webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[main] ERROR: could not open webcam.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[main] Webcam opened. FPS={fps:.1f}")
    print(f"[main] Fall thresholds: torso>{TORSO_THRESHOLD} deg "
          f"aspect>{ASPECT_THRESHOLD} rate>{SPEED_THRESHOLD} deg/s "
          f"confirm={CONFIRM_WINDOWS} windows")
    print("[main] Entering main loop (ESC to quit)...\n")

    center_x_buffer = deque(maxlen=SMOOTHING_WINDOW)
    body_size_buffer = deque(maxlen=SMOOTHING_WINDOW)

    frame_buffer = deque(maxlen=WINDOW_SIZE)
    frames_since_last_window = 0
    consecutive_fall_windows = 0
    baseline_probs = []
    baseline_done = False
    alert_cooldown = 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[main] webcam stopped delivering frames.")
                break

            target_bbox = lock.update(frame)
            draw_status(frame, target_bbox, lock.last_status, lock.last_similarity)

            uL, uR = IDLE_UL, IDLE_UR

            if target_bbox is None:
                # Patient not identified/locked -- stop, reset both
                # steering smoothing and the fall-detection window buffer
                # so stale data doesn't bridge across a lock gap.
                center_x_buffer.clear()
                body_size_buffer.clear()
                frame_buffer.clear()
                frames_since_last_window = 0
                consecutive_fall_windows = 0

                cv2.putText(frame, "PATIENT NOT LOCKED - STOPPED", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            else:
                # One shared Pose call on the FULL frame -- do not crop
                # (see module docstring for why).
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    is_patient = (
                        landmarks_reliable(landmarks)
                        and landmarks_belong_to_target(
                            landmarks, target_bbox, frame.shape
                        )
                    )

                    if is_patient:
                        mp_draw.draw_landmarks(
                            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                        )

                        # ---- Steering ----
                        raw_center_x = compute_body_center_x(landmarks)
                        raw_body_size = compute_body_size(landmarks)
                        center_x = smooth(center_x_buffer, raw_center_x)
                        body_size = smooth(body_size_buffer, raw_body_size)
                        uL, uR = compute_motor_signals(center_x, body_size)

                        cv2.putText(frame, f"center_x: {center_x:.2f}", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"body_size: {body_size:.3f}", (10, 105),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"uL: {uL:+.2f}  uR: {uR:+.2f}", (10, 130),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                        # ---- Fall detection (same windowing as
                        # live_fall_detection_rules.py, unchanged) ----
                        feats = extract_frame_features(landmarks)
                        frame_buffer.append(feats)
                        frames_since_last_window += 1

                        if (len(frame_buffer) == WINDOW_SIZE
                                and frames_since_last_window >= STEP):
                            frames_since_last_window = 0
                            is_fall, debug = evaluate_window(list(frame_buffer), fps)

                            if DEBUG_PRINT:
                                c_str = (f"C1={'v' if debug['c1_torso'] else 'x'} "
                                         f"C2={'v' if debug['c2_aspect'] else 'x'} "
                                         f"C3={'v' if debug['c3_speed'] else 'x'}")
                                print(f"  [window] torso_max={debug['torso_max']:.1f} "
                                      f"aspect_max={debug['aspect_max']:.2f} "
                                      f"rate={debug['torso_change_rate']:.1f} "
                                      f"| {c_str} -> {'FALL' if is_fall else 'ADL'}")

                            if not baseline_done:
                                baseline_probs.append(1.0 if is_fall else 0.0)
                                if len(baseline_probs) >= BASELINE_CHECK_WINDOWS:
                                    baseline_done = True
                                    fall_rate = sum(baseline_probs) / len(baseline_probs)
                                    print(f"[baseline] FALL rate over first "
                                          f"{BASELINE_CHECK_WINDOWS} windows = "
                                          f"{fall_rate:.2f}")

                            if is_fall:
                                consecutive_fall_windows += 1
                            else:
                                consecutive_fall_windows = 0

                            if (consecutive_fall_windows >= CONFIRM_WINDOWS
                                    and alert_cooldown <= 0):
                                on_fall_detected()
                                consecutive_fall_windows = 0
                                alert_cooldown = int(fps * 10)
                                # Fall confirmed -> override motor output
                                uL, uR = 0.0, 0.0

                    else:
                        # Pose found someone, but not the identified
                        # patient (bystander) or landmarks unreliable --
                        # same skip pattern as the standalone fall
                        # detector: don't accumulate this frame, don't
                        # steer confidently.
                        frames_since_last_window = 0
                        consecutive_fall_windows = 0
                        cv2.putText(frame, "PATIENT POSE NOT CONFIRMED", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                else:
                    consecutive_fall_windows = 0
                    cv2.putText(frame, "NO POSE DETECTED", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if alert_cooldown > 0:
                alert_cooldown -= 1

            # TODO: send uL, uR to the ESP32 over UART, e.g.:
            #   ser.write(f"{uL:.2f},{uR:.2f}\n".encode())

            cv2.imshow("Robot Main", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    lock.close()
    print("[main] Stopped.")


if __name__ == "__main__":
    main()
