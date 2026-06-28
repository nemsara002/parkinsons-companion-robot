"""
Minimal camera + display test - isolates whether the webcam and OpenCV
GUI window work AT ALL, independent of MediaPipe/fall-detection logic.
Run this BEFORE re-running live_fall_detection_laptop.py.
"""

import cv2

print("Opening camera...")
cap = cv2.VideoCapture(0)
print(f"cap.isOpened() = {cap.isOpened()}")

if not cap.isOpened():
    print("ERROR: camera did not open at all. Try CAMERA_INDEX = 1 or 2 instead of 0.")
else:
    print("Reading 5 frames and checking they're not empty/black...")
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f"  frame {i}: READ FAILED (ret=False)")
        else:
            print(f"  frame {i}: shape={frame.shape}, mean_pixel_value={frame.mean():.1f} "
                  f"(0=pure black, 255=pure white - a real picture should NOT be near 0)")

    print("\nNow opening a live preview window for 5 seconds. "
          "A window titled 'Camera Test' should appear ON TOP of other windows.")
    print("If you don't see it, check your taskbar for a new window, "
          "or check if it opened off-screen / minimized.")
    import time
    start = time.time()
    while time.time() - start < 5:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done. Did the window appear?")
