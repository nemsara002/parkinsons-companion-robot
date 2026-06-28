import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:

        # draw skeleton
        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # -------------------------------
        # EXTRACT LANDMARKS (IMPORTANT)
        # -------------------------------
        landmarks = result.pose_landmarks.landmark

        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        nose = landmarks[mp_pose.PoseLandmark.NOSE]

        hip_y = (left_hip.y + right_hip.y) / 2
        head_y = nose.y

        # -------------------------------
        # FALL DETECTION LOGIC (PUT HERE)
        # -------------------------------
        head_hip_diff = abs(head_y - hip_y)

        if head_hip_diff < 0.1:
            cv2.putText(frame, "FALL DETECTED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 3)

        # optional debug
        print("Head-Hip Diff:", head_hip_diff)

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()