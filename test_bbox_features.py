"""
Test whether bounding-box / silhouette-based features capture
camera-facing falls (like FALL_025) that torso_angle misses.

This is a DIAGNOSTIC only - run it on the specific problem clips first,
before committing to rebuilding the whole pipeline. If this doesn't show
a clear signal either, we need a different fix (e.g. depth camera input
isn't available, so we'd look at temporal silhouette-area collapse rate
more aggressively, or accept this as a true hardware limitation).

Run in the same folder as fall_dataset/raw_videos/.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

VIDEO_PATH_GOOD = "fall_dataset/raw_videos/fall/fall-04-cam0.mp4"   # adjust filename
VIDEO_PATH_BAD = "fall_dataset/raw_videos/fall/fall-25-cam0.mp4"    # adjust filename


def analyze_clip(video_path, label):
    cap = cv2.VideoCapture(video_path)
    print(f"\n=== {label}: {video_path} ===")
    print(f"{'frame':>6} {'bbox_h':>10} {'bbox_w':>10} {'bbox_area':>10} {'aspect_ratio':>12}")

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
                lms = results.pose_landmarks.landmark
                xs = [lm.x for lm in lms]
                ys = [lm.y for lm in lms]

                bbox_h = max(ys) - min(ys)
                bbox_w = max(xs) - min(xs)
                bbox_area = bbox_h * bbox_w
                aspect_ratio = bbox_w / (bbox_h + 1e-6)

                # print every 10th frame to keep output readable
                if frame_idx % 10 == 0:
                    print(f"{frame_idx:>6} {bbox_h:>10.4f} {bbox_w:>10.4f} "
                          f"{bbox_area:>10.4f} {aspect_ratio:>12.4f}")

            frame_idx += 1

    cap.release()


if __name__ == "__main__":
    analyze_clip(VIDEO_PATH_GOOD, "FALL_004 (good - sideways fall)")
    analyze_clip(VIDEO_PATH_BAD, "FALL_025 (bad - toward-camera fall)")
