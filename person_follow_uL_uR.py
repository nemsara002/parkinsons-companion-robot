"""
Camera-only person-following controller.

Computes two differential-drive motor signals (uL, uR) in range [-1, 1]
from MediaPipe Pose landmarks:

  - Steering: based on horizontal offset of body center from frame center.
  - Forward/backward: based on a fused "apparent body size" metric
    (shoulder width + hip width + torso height), with a dead zone.

This script only displays uL/uR on screen for testing. Sending these
values to the ESP32 (e.g. over UART/serial or WiFi) is a separate step —
see the note at the bottom of this file.
"""

import cv2
import mediapipe as mp
from collections import deque

# ---------------- CONFIGURATION ----------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Steering control
STEER_DEADZONE = 0.05        # normalized fraction of frame width treated as "centered"
STEER_GAIN = 0.8             # how aggressively to correct steering
TURN_SPEED_LIMIT = 0.6       # max magnitude of steering correction

# Distance control (body-size based)
TARGET_SIZE_MIN = 0.35       # below this -> person far -> move forward
TARGET_SIZE_MAX = 0.45       # above this -> person too close -> move backward
FORWARD_SPEED = 0.5          # base forward command magnitude (0-1)
BACKWARD_SPEED = 0.4         # base backward command magnitude (0-1)

# Temporal smoothing
SMOOTHING_WINDOW = 5          # number of frames averaged

# ---------------- SETUP ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

center_x_buffer = deque(maxlen=SMOOTHING_WINDOW)
body_size_buffer = deque(maxlen=SMOOTHING_WINDOW)


def compute_body_center_x(landmarks):
    """Average x of left/right shoulder and hip -> normalized [0,1]."""
    l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    return (l_sh.x + r_sh.x + l_hip.x + r_hip.x) / 4.0


def compute_body_size(landmarks):
    """
    Fuse multiple features into one 'apparent body size' metric:
    shoulder width, hip width, and torso height (shoulder-to-hip distance).
    All values are in normalized [0,1] coordinates relative to the frame.

    Torso height is weighted slightly higher since it stays more stable
    than shoulder/hip width when the person turns sideways.
    """
    l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    shoulder_width = abs(l_sh.x - r_sh.x)
    hip_width = abs(l_hip.x - r_hip.x)

    sh_mid_y = (l_sh.y + r_sh.y) / 2
    hip_mid_y = (l_hip.y + r_hip.y) / 2
    torso_height = abs(hip_mid_y - sh_mid_y)

    body_size = (0.3 * shoulder_width) + (0.3 * hip_width) + (0.4 * torso_height)
    return body_size


def smooth(buffer, new_value):
    buffer.append(new_value)
    return sum(buffer) / len(buffer)


def compute_motor_signals(center_x, body_size):
    """
    Returns (uL, uR) in range [-1, 1] for a differential-drive robot.
    Positive = forward rotation for that wheel, negative = reverse.
    """
    # ---- Steering term ----
    error_x = center_x - 0.5  # 0.5 = frame center

    if abs(error_x) < STEER_DEADZONE:
        steer = 0.0
    else:
        steer = error_x * STEER_GAIN
        steer = max(-TURN_SPEED_LIMIT, min(TURN_SPEED_LIMIT, steer))

    # ---- Forward/backward term ----
    if body_size < TARGET_SIZE_MIN:
        forward = FORWARD_SPEED          # person far -> move forward
    elif body_size > TARGET_SIZE_MAX:
        forward = -BACKWARD_SPEED        # person too close -> move backward
    else:
        forward = 0.0                    # within target range -> stay still

    # ---- Differential drive mixing ----
    # steer > 0 means person is to the right of center -> turn right
    # (slow the right wheel relative to the left)
    uL = forward + steer
    uR = forward - steer

    uL = max(-1.0, min(1.0, uL))
    uR = max(-1.0, min(1.0, uR))

    return uL, uR


# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror for natural viewing, optional
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    uL, uR = 0.0, 0.0  # default: stop if no person detected

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark

        raw_center_x = compute_body_center_x(landmarks)
        raw_body_size = compute_body_size(landmarks)

        center_x = smooth(center_x_buffer, raw_center_x)
        body_size = smooth(body_size_buffer, raw_body_size)

        uL, uR = compute_motor_signals(center_x, body_size)

        cv2.putText(frame, f"center_x: {center_x:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"body_size: {body_size:.3f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"uL: {uL:+.2f}  uR: {uR:+.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        print(f"center_x={center_x:.2f}  body_size={body_size:.3f}  uL={uL:+.2f}  uR={uR:+.2f}")
    else:
        cv2.putText(frame, "NO PERSON DETECTED - STOPPED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Person Following - uL/uR Debug", frame)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

# ----------------------------------------------------------------------
# NEXT STEP (not in this file): sending uL/uR to the ESP32.
# Typical approach over UART (matches your block diagram):
#
#   import serial
#   ser = serial.Serial('/dev/ttyUSB0', 115200)
#   ser.write(f"{uL:.2f},{uR:.2f}\n".encode())
#
# On the ESP32 side, parse the comma-separated floats and map them to
# PWM duty cycles for the TB6612FNG inputs (e.g. uL=1.0 -> max forward
# PWM on left motor, uL=-1.0 -> max reverse PWM).
# ----------------------------------------------------------------------
