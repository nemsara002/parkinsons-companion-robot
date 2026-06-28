"""
Extract a few sample frames from the worst-performing FALL clips as images,
so we can visually check whether the camera angle/framing explains the
poor recall (e.g. person falls toward/away from camera, or partially
out of frame, rather than falling sideways within the camera plane).

Run this in the folder that contains fall_dataset/raw_videos/...
Adjust CLIP_FILES to match your actual filenames for clips 25, 7, 9.
"""

import cv2
import os

# Map clip_id (as used in your features.csv, e.g. "FALL_025") to the
# actual video filename. Clip numbering follows alphabetical sort order
# of files in fall_dataset/raw_videos/fall/, 1-indexed.
CLIP_NUMBERS_TO_CHECK = {
    "FALL_025": 25,
    "FALL_007": 7,
    "FALL_009": 9,
    "FALL_004": 4,   # good clip, for comparison
}

VIDEO_DIR = "fall_dataset/raw_videos/fall"
OUTPUT_DIR = "sample_frames"
N_SAMPLE_FRAMES = 6  # evenly spaced frames across the clip


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_files = sorted([f for f in os.listdir(VIDEO_DIR)
                           if f.lower().endswith((".mp4", ".avi"))])

    for clip_id, clip_num in CLIP_NUMBERS_TO_CHECK.items():
        idx = clip_num - 1  # 1-indexed -> 0-indexed
        if idx >= len(video_files):
            print(f"WARNING: clip number {clip_num} out of range, skipping")
            continue

        video_path = os.path.join(VIDEO_DIR, video_files[idx])
        print(f"{clip_id} -> {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sample_indices = [
            int(total_frames * i / (N_SAMPLE_FRAMES - 1))
            if i < N_SAMPLE_FRAMES - 1 else total_frames - 1
            for i in range(N_SAMPLE_FRAMES)
        ]

        for i, frame_idx in enumerate(sample_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                out_path = os.path.join(OUTPUT_DIR, f"{clip_id}_frame{i}.jpg")
                cv2.imwrite(out_path, frame)

        cap.release()

    print(f"\nDone. Check the '{OUTPUT_DIR}' folder for sample images.")


if __name__ == "__main__":
    main()
