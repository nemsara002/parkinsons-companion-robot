"""
Camera-only person-following controller — now sending to the ESP32.

Computes two differential-drive motor signals (uL, uR) in range [-1, 1]
from MediaPipe Pose landmarks, scales them to PWM (-255..255), and sends
them to the ESP32 over UART every frame.
"""

import cv2
import mediapipe as mp
import serial
import time
from collections import deque

# ---------------- CONFIGURATION ----------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Steering control
STEER_DEADZONE = 0.05
STEER_GAIN = 0.8
TURN_SPEED_LIMIT = 0.6

# Distance control (body-size based)
TARGET_SIZE_MIN = 0.35
TARGET_SIZE_MAX = 0.45
FORWARD_SPEED = 0.5
BACKWARD_SPEED = 0.4

# Temporal smoothing
SMOOTHING_WINDOW = 5

# ---------------- ESP32 LINK ----------------
SERIAL_PORT = "/dev/serial0"
SERIAL_BAUD = 115200
PWM_SCALE = 255          # uL/uR in [-1,1] -> PWM in [-255,255]

ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
time.sleep(0.2)


def send_motor_command(uL, uR):
    """Scale [-1,1] floats to integer PWM and send 'uL,uR\\n' to the ESP32."""
    pwmL = int(max(-1.0, min(1.0, uL)) * PWM_SCALE)
    pwmR = int(max(-1.0, min(1.0, uR)) * PWM_SCALE)
    ser.write(f"{pwmL},{pwmR}\n".encode())


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
    l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    return (l_sh.x + r_sh.x + l_hip.x + r_hip.x) / 4.0


def compute_body_size(landmarks):
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


# ---------------- MAIN LOOP ----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
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

        # send every frame, whether or not a person is detected
        # (uL=uR=0 when nobody's in frame -> ESP32 stops the motors)
        send_motor_command(uL, uR)

        cv2.imshow("Person Following - uL/uR Debug", frame)

        if cv2.waitKey(1) == 27:  # ESC to quit
            break

finally:
    send_motor_command(0.0, 0.0)  # always stop on exit
    ser.close()
    cap.release()
    cv2.destroyAllWindows()
