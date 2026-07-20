"""
capture_registration_photos.py

Standalone helper to build the patient's registration photo set directly
from the Pi's own webcam -- no shuttling image files between laptop and Pi,
and no camera-mismatch risk (same camera used for registration and live
identification).

Usage:
    python capture_registration_photos.py

Controls (while the preview window is focused):
    SPACE   -> capture a photo
    q       -> quit early (keeps whatever photos were already captured)

After capturing, optionally runs register_patient() immediately against
the captured set, so you can go straight from "stand in front of camera"
to "patient_embedding.pkl exists" in one script run.
"""

import os
import time

import cv2

# Import register_patient from the main module so this stays in sync with
# whatever REGISTERED_PATIENT_PATH / embedding logic is defined there.
from patient_identifier import register_patient, REGISTERED_PATIENT_PATH


OUTPUT_DIR = "patient_registration_photos"
NUM_PHOTOS = 8          # how many photos to collect -- 5-10 is a good range
CAMERA_INDEX = 0        # change if your USB webcam isn't index 0 on the Pi


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[capture] Opening webcam (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[capture] ERROR: could not open camera index {CAMERA_INDEX}. "
              f"Check it's not in use by another process, and that this is "
              f"the right index for your USB webcam.")
        return

    print(f"[capture] Webcam opened. Collecting {NUM_PHOTOS} photos.")
    print("[capture] Move slightly between shots -- vary angle a little "
          "(straight on, slight left, slight right, chin up/down) and "
          "keep lighting similar to normal operating conditions.")
    print("[capture] Press SPACE to capture a photo, 'q' to quit early.\n")

    captured_paths = []
    photo_num = 0

    while photo_num < NUM_PHOTOS:
        ret, frame = cap.read()
        if not ret:
            print("[capture] ERROR: webcam stopped delivering frames. Stopping.")
            break

        display = frame.copy()
        cv2.putText(
            display,
            f"Captured: {photo_num}/{NUM_PHOTOS}  (SPACE=capture, q=quit)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Registration Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            photo_num += 1
            filename = os.path.join(OUTPUT_DIR, f"{photo_num}.png")
            cv2.imwrite(filename, frame)
            captured_paths.append(filename)
            print(f"[capture] Saved {filename} ({photo_num}/{NUM_PHOTOS})")
            time.sleep(0.3)  # small debounce so one press doesn't double-fire

        elif key == ord("q"):
            print("[capture] Quit early by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if not captured_paths:
        print("[capture] No photos captured, nothing to register.")
        return

    print(f"\n[capture] Done. {len(captured_paths)} photos saved to "
          f"'{OUTPUT_DIR}/'.")

    answer = input(
        f"\nRun register_patient() now against these {len(captured_paths)} "
        f"photos and save to '{REGISTERED_PATIENT_PATH}'? [y/n]: "
    ).strip().lower()

    if answer == "y":
        register_patient(captured_paths)
        print("[capture] Registration complete. You can now run "
              "patient_identifier.py for live identification.")
    else:
        print(f"[capture] Skipped registration. Photos are saved in "
              f"'{OUTPUT_DIR}/' -- you can register later by calling "
              f"register_patient([...]) with these paths.")


if __name__ == "__main__":
    main()
