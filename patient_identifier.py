"""
patient_identifier.py

Solves: "robot gets confused when multiple people are in frame"

Architecture (as discussed):
  Stage 1 - IDENTIFICATION (slow, periodic): face embedding match against the
            registered patient, accepted only if it clears an absolute
            similarity threshold AND wins by a clear margin over the next
            best face in frame.
  Stage 2 - TRACKING (fast, every frame): once identified, a lightweight
            OpenCV tracker follows that person's body box frame-to-frame
            without needing to re-run face recognition or even see the face.
  Stage 3 - RE-VERIFICATION (periodic): every few seconds, or when tracker
            confidence looks shaky, re-run face recognition to confirm the
            tracker hasn't drifted onto a different person (e.g. after two
            people cross paths).

This file is meant to be dropped next to person_follow_uL_uR.py and used as
a drop-in "give me the target bbox" layer in front of your existing
steering/UART code. Integration points are marked with TODO.

Dependencies:
    pip install face_recognition opencv-python numpy --break-system-packages
    (face_recognition needs dlib -- on a Pi this can take a while to build;
    see note at bottom of file for a lighter-weight alternative if build
    time / RAM becomes a problem on the Pi 4.)
"""

import os
import csv
import time
import pickle
import logging
from datetime import datetime
from enum import Enum, auto

import cv2
import numpy as np
import face_recognition


# ----------------------------------------------------------------------------
# Config -- tune these empirically on YOUR camera/lighting before trusting them
# ----------------------------------------------------------------------------

REGISTERED_PATIENT_PATH = "patient_embedding.pkl"

# Where identification events get logged (CSV -- easy to open in Excel/Sheets
# or plot later for your report). One row per VERIFIED or LOST event.
IDENTIFICATION_LOG_PATH = "identification_log.csv"

# Absolute similarity floor. face_recognition returns a distance (lower =
# more similar); we convert to a 0-1 similarity score. Below this, we don't
# trust the match at all, regardless of margin.
SIMILARITY_THRESHOLD = 0.55

# The winning face must beat the second-best candidate face by at least this
# much. This is what actually prevents mis-identifying in a crowd -- a
# narrow win against a runner-up is not a confident identification.
MARGIN_THRESHOLD = 0.12

# Run full face recognition once every N frames while searching/tracking
# (recognition is expensive; don't do it every frame on a Pi 4).
RECOGNITION_EVERY_N_FRAMES = 15

# While locked onto a tracked target, force a re-verification at least this
# often even if nothing looks wrong, to catch slow drift.
FORCED_RECHECK_INTERVAL_SEC = 4.0

# If the CSRT tracker's own confidence proxy (see note in update()) drops,
# treat it as a trigger for immediate re-verification rather than waiting
# for the timed recheck.
TRACKER_FAILURE_TRIGGERS_RECHECK = True


class State(Enum):
    SEARCHING = auto()       # no locked target, scanning every recognition cycle
    TRACKING = auto()        # locked on, tracker running every frame
    VERIFYING = auto()       # tracking, but re-running recognition this frame


# ----------------------------------------------------------------------------
# Logging: one CSV row per identification event, so you have a record to
# check after a run rather than only the live on-screen view.
# ----------------------------------------------------------------------------

class IdentificationLogger:
    """
    Appends one row per notable event (locked on, re-verified, lost) to a
    CSV file. Also mirrors the same line to the console via `logging`, so
    you get both a live scroll while testing and a saved file afterwards.

    CSV columns: timestamp, event, similarity, frame_count
        event: "VERIFIED" (recognition confirmed patient, tracker (re)seeded)
               "LOST"     (re-verification failed or tracker failed -> back
                           to SEARCHING)
    """

    def __init__(self, log_path=IDENTIFICATION_LOG_PATH):
        self.log_path = log_path
        self._file = None
        self._writer = None

        self._console = logging.getLogger("patient_identifier")
        if not self._console.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(message)s]"))
            self._console.addHandler(handler)
            self._console.setLevel(logging.INFO)

        try:
            file_exists = os.path.exists(log_path)
            self._file = open(log_path, "a", newline="")
            self._writer = csv.writer(self._file)
            if not file_exists:
                self._writer.writerow(["timestamp", "event", "similarity", "frame_count"])
                self._file.flush()
        except (PermissionError, OSError) as e:
            # File locked by another process (leftover python.exe, Excel,
            # OneDrive sync, etc). Don't let a log-file lock take down the
            # whole identification pipeline -- fall back to console-only
            # logging and keep going.
            self._console.warning(
                f"Could not open log file '{log_path}' ({e}). "
                f"Continuing with console-only logging (no CSV will be saved)."
            )
            self._file = None
            self._writer = None

    def log(self, event, similarity, frame_count):
        timestamp = datetime.now().isoformat(timespec="seconds")
        sim_str = f"{similarity:.3f}" if similarity is not None else ""

        if self._writer is not None:
            try:
                self._writer.writerow([timestamp, event, sim_str, frame_count])
                self._file.flush()  # flush immediately -- don't lose events on a crash
            except (PermissionError, OSError) as e:
                self._console.warning(f"Log write failed ({e}), disabling CSV logging.")
                self._file = None
                self._writer = None

        msg = f"{event}" + (f" similarity={sim_str}" if similarity is not None else "")
        self._console.info(msg)

    def close(self):
        if self._file is not None:
            self._file.close()


# ----------------------------------------------------------------------------
# Stage 0: Registration (run once, offline, to build the reference embedding)
# ----------------------------------------------------------------------------

def register_patient(image_paths, save_path=REGISTERED_PATIENT_PATH):
    """
    Build the patient's reference embedding by averaging embeddings across
    several photos (different angles/lighting improves robustness). Run this
    once as a setup step, not during normal operation.

    image_paths: list of file paths, e.g. 5-10 photos of the patient's face
                 taken under conditions similar to real deployment (indoor,
                 same rough lighting as their home).
    """
    embeddings = []
    for path in image_paths:
        img = face_recognition.load_image_file(path)
        locations = face_recognition.face_locations(img)
        if not locations:
            print(f"[register_patient] WARNING: no face found in {path}, skipping")
            continue
        encs = face_recognition.face_encodings(img, known_face_locations=locations)
        if encs:
            embeddings.append(encs[0])

    if not embeddings:
        raise RuntimeError("No usable faces found in provided images.")

    avg_embedding = np.mean(embeddings, axis=0)
    with open(save_path, "wb") as f:
        pickle.dump(avg_embedding, f)

    print(f"[register_patient] Saved averaged embedding from {len(embeddings)} "
          f"images to {save_path}")
    return avg_embedding


# ----------------------------------------------------------------------------
# Stage 1: Identification with margin check
# ----------------------------------------------------------------------------

class PatientIdentifier:
    def __init__(self, embedding_path=REGISTERED_PATIENT_PATH):
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(
                f"{embedding_path} not found -- run register_patient() first."
            )
        with open(embedding_path, "rb") as f:
            self.reference_embedding = pickle.load(f)

        self._last_snapshot_time = 0.0

    def _save_debug_snapshot(self, frame_bgr, min_interval_sec=2.0):
        """Saves the current frame to disk for visual inspection, throttled
        so it doesn't spam-write a file every failed frame."""
        now = time.time()
        if now - self._last_snapshot_time < min_interval_sec:
            return
        self._last_snapshot_time = now
        try:
            cv2.imwrite("no_face_debug_frame.jpg", frame_bgr)
            print("[find_patient] Saved current frame to "
                  "'no_face_debug_frame.jpg' -- open it (e.g. via VNC file "
                  "manager or 'eog no_face_debug_frame.jpg') to see exactly "
                  "what the camera is capturing right now.")
        except Exception as e:
            print(f"[find_patient] Could not save debug snapshot: {e}")

    @staticmethod
    def _distance_to_similarity(distance):
        # face_recognition's distance is roughly 0 (identical) to ~1.0+
        # (very different). Clamp and invert to a 0-1 similarity score.
        return max(0.0, 1.0 - distance)

    def find_patient(self, frame_bgr, debug=True, upsample=1):
        """
        Runs face detection + recognition on a frame that may contain
        multiple people. Returns (bbox, similarity, reason).

        bbox is (top, right, bottom, left) in face_recognition's convention,
        matching the pixel coords of `frame_bgr`. bbox/similarity are None
        unless reason == "MATCHED".

        reason is one of:
          "MATCHED"   - confident match found, bbox/similarity are valid
          "NO_FACE"   - no face detected in the frame at all. This is
                        INCONCLUSIVE, not evidence of a mismatch -- e.g. the
                        patient's face just isn't visible right now (turned
                        away, too far for face detection at this camera's
                        geometry, etc). Callers should NOT treat this the
                        same as a rejection when deciding whether to keep
                        trusting an existing tracker lock.
          "REJECTED"  - a face WAS detected, but failed the similarity or
                        margin check. This IS real evidence -- either it's
                        genuinely not the patient, or the patient's face
                        didn't score well enough to trust. Callers should
                        treat this more seriously than NO_FACE.

        debug: prints diagnostics to console (face count found, best/second
        similarity, and why a match was rejected). Also saves a snapshot of
        the live frame to `no_face_debug_frame.jpg` on failure (throttled to
        once every ~2 sec) so you can visually inspect exactly what the
        pipeline is receiving -- catches issues like wrong camera index,
        an unexpectedly dark/blurry feed, or a frame that isn't what you
        think it is. Useful while calibrating; turn off once tuned.

        upsample: passed to face_recognition.face_locations as
        number_of_times_to_upsample. Bump this to 2 if the patient's face
        is small/far from camera -- HOG's default sensitivity can miss
        smaller faces, and upsampling the image improves detection at the
        cost of speed.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(
            rgb, number_of_times_to_upsample=upsample
        )

        if not face_locations:
            if debug:
                print("[find_patient] No face detected in frame at all "
                      "(check camera index / lighting / face_locations model).")
                self._save_debug_snapshot(frame_bgr)
            return None, None, "NO_FACE"

        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        scored = []
        for loc, enc in zip(face_locations, face_encodings):
            dist = face_recognition.face_distance(
                [self.reference_embedding], enc
            )[0]
            sim = self._distance_to_similarity(dist)
            scored.append((sim, loc))

        # Sort best-first
        scored.sort(key=lambda x: x[0], reverse=True)

        best_sim, best_loc = scored[0]
        second_sim = scored[1][0] if len(scored) > 1 else 0.0

        margin = best_sim - second_sim

        if debug:
            print(f"[find_patient] faces_found={len(scored)} "
                  f"best_sim={best_sim:.3f} second_sim={second_sim:.3f} "
                  f"margin={margin:.3f} "
                  f"(threshold={SIMILARITY_THRESHOLD}, margin_min={MARGIN_THRESHOLD})")

        if best_sim < SIMILARITY_THRESHOLD:
            # Not confident enough at all, regardless of how many people are around
            if debug:
                print("[find_patient] REJECTED: best_sim below SIMILARITY_THRESHOLD")
            return None, None, "REJECTED"

        if len(scored) > 1 and margin < MARGIN_THRESHOLD:
            # Ambiguous -- too close to call against the runner-up.
            # Safer to say "not found" than to guess wrong in a crowd.
            if debug:
                print("[find_patient] REJECTED: margin below MARGIN_THRESHOLD "
                      "(too close to call against runner-up face)")
            return None, None, "REJECTED"

        return best_loc, best_sim, "MATCHED"


# ----------------------------------------------------------------------------
# Stage 2: Frame-to-frame body tracking (no face needed once locked on)
# ----------------------------------------------------------------------------

class PatientTracker:
    """
    Thin wrapper around an OpenCV tracker so the main loop doesn't need to
    know which tracker backend is in use.

    Default is CSRT: slower than KCF (roughly 2-4x more CPU per frame) but
    notably more robust to partial occlusion, scale change (patient walking
    toward/away from camera), and general drift. Since this pipeline isn't
    running at a high frame rate anyway (one shared MediaPipe Pose call per
    frame already), correctness matters more here than raw tracker speed --
    a hijacked or drifted lock is a worse failure mode than a slightly
    slower loop. KCF remains available as a fallback if CSRT turns out to be
    too slow on your Pi 4 in practice.
    """

    def __init__(self, backend="CSRT"):
        self.backend = backend
        self.tracker = None
        self._init_tracker()

    @staticmethod
    def _create_tracker(name):
        """
        OpenCV has changed how trackers are constructed across versions:
          - old (<=4.4):      cv2.TrackerCSRT_create()
          - mid (4.5.1-4.10): cv2.legacy.TrackerCSRT_create()
          - new (4.11+):      cv2.TrackerCSRT.create()
        Try each in turn so this works regardless of which OpenCV build is
        installed (this is exactly why the Pi hit a different error than
        the laptop -- different opencv-python versions between the two).
        """
        candidates = [
            lambda: getattr(cv2, f"Tracker{name}_create")(),
            lambda: getattr(cv2.legacy, f"Tracker{name}_create")(),
            lambda: getattr(cv2, f"Tracker{name}").create(),
        ]
        errors = []
        for attempt in candidates:
            try:
                return attempt()
            except AttributeError as e:
                errors.append(str(e))

        raise RuntimeError(
            f"Could not create cv2 Tracker{name} with any known OpenCV API "
            f"style. This usually means opencv-contrib-python is needed "
            f"instead of plain opencv-python (trackers live in the contrib "
            f"module). Try:\n"
            f"    pip install opencv-contrib-python --break-system-packages\n"
            f"(uninstall plain opencv-python first if both are present, "
            f"they can conflict). Errors tried: {errors}"
        )

    def _init_tracker(self):
        self.tracker = self._create_tracker(self.backend)

    def start(self, frame_bgr, body_bbox_xywh):
        """
        body_bbox_xywh: (x, y, w, h) in pixel coords.

        Returns True/False for whether initialization succeeded.

        Note: some OpenCV builds' tracker.init() returns None on success
        rather than True (the return contract isn't fully consistent across
        the legacy vs new class-based tracker APIs -- same inconsistency we
        already had to work around for tracker *creation* above). Treat
        only an explicit False as failure; None or True both count as
        success.
        """
        self._init_tracker()
        bbox_int = tuple(int(v) for v in body_bbox_xywh)
        raw = self.tracker.init(frame_bgr, bbox_int)
        success = raw is not False
        print(f"[PatientTracker] init(bbox={bbox_int}) raw return={raw!r} "
              f"-> treated as {'SUCCESS' if success else 'FAILURE'}")
        return success

    def update(self, frame_bgr):
        """
        Returns (success, bbox_xywh).

        Same caution as start(): don't fully trust a falsy-but-not-False
        return as failure. Additionally sanity-check the returned bbox has
        positive width/height -- a degenerate bbox is a more reliable
        failure signal than the raw success flag on some OpenCV builds.
        """
        raw_success, bbox = self.tracker.update(frame_bgr)
        success = raw_success is not False

        if success and bbox is not None:
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                success = False

        return success, bbox


# ----------------------------------------------------------------------------
# Helper: map a face bbox to a full-body bbox using your existing pose output
# ----------------------------------------------------------------------------

def face_bbox_to_body_bbox(face_loc, pose_landmarks, frame_shape):
    """
    face_loc: (top, right, bottom, left) from face_recognition
    pose_landmarks: your existing MediaPipe Pose result for the SAME frame
                     (a list of per-person landmark sets if you're running
                     pose on multiple detected people -- see TODO below)
    frame_shape: frame.shape (for converting normalized coords to pixels)

    Returns (x, y, w, h) body bbox for whichever pose skeleton's
    head/shoulder region overlaps the identified face box.

    TODO: this assumes your pose pipeline can return landmarks per-detected
    person. If person_follow_uL_uR.py currently only runs single-person
    MediaPipe Pose (pose.process() gives one skeleton for the whole frame),
    you have two options:
      1. Crop candidate person regions (e.g. via a person detector / simple
         contour-based body boxes) and run Pose per-crop, or
      2. Simpler short-term fix: once a face is identified, take a bbox
         directly around the face and expand it downward by a fixed ratio
         (e.g. 1x width, 4x height) as an approximate body box to seed the
         tracker -- less precise than true pose-based mapping, but works and
         is much less code. Given your timeline, I'd start here and only
         build the full per-person pose mapping if tracking proves too loose.
    """
    top, right, bottom, left = face_loc
    face_w = right - left
    face_h = bottom - top

    # Simple expansion approach (option 2 above) -- swap in pose-based
    # mapping later if needed.
    x = max(0, left - int(face_w * 0.5))
    y = top
    w = int(face_w * 2.0)
    h = int(face_h * 5.0)  # rough face-to-full-body height ratio

    frame_h, frame_w = frame_shape[:2]
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)

    return (x, y, w, h)


# ----------------------------------------------------------------------------
# Main state machine -- integrate this loop with your existing capture +
# steering code in person_follow_uL_uR.py
# ----------------------------------------------------------------------------

class PatientTargetLock:
    """
    Call `update(frame)` once per captured frame. It returns the current
    body bbox to feed into your existing steering/UART logic, or None if
    the patient isn't currently locked (robot should idle / search state).
    """

    def __init__(self, log_path=IDENTIFICATION_LOG_PATH):
        self.identifier = PatientIdentifier()
        self.tracker = PatientTracker(backend="CSRT")
        self.logger = IdentificationLogger(log_path)

        self.state = State.SEARCHING
        self.frame_count = 0
        self.last_verified_time = 0.0

        # For visual feedback only (see demo loop below).
        # last_status is one of: "SEARCHING", "VERIFIED", "TRACKING", "LOST"
        #   VERIFIED  = face recognition just confirmed this frame
        #   TRACKING  = CSRT following, no fresh confirmation this frame
        #               (this now also covers "no face visible right now,
        #               but the tracker is still confidently following" --
        #               see update()'s docstring)
        #   LOST      = dropped due to a REAL mismatch (a face was seen
        #               and didn't match) or the tracker itself failed
        #               with nothing to recover onto. No longer drops
        #               just because a face happened to be out of view.
        self.last_status = "SEARCHING"
        self.last_similarity = None

    def update(self, frame_bgr):
        """
        Returns the current body bbox, or None if the patient isn't
        currently locked.

        RE-VERIFICATION POLICY (see also find_patient()'s "reason" values):
          While TRACKING, this used to force a face-recognition check every
          FORCED_RECHECK_INTERVAL_SEC and drop the lock if no face was
          found -- which meant the lock would be dropped simply because
          the patient was too far away or turned away for face detection
          to work, even though the CSRT tracker was still following them
          correctly. That's the wrong failure mode for a robot meant to
          follow someone at a few meters' distance.

          Now: face recognition is still attempted periodically
          (opportunistically) while tracking, but the outcome is handled
          based on WHY it failed:
            - NO_FACE (no face visible at all): inconclusive, not evidence
              of anything wrong. Keep trusting the tracker, keep TRACKING.
            - REJECTED (a face WAS seen, but didn't match): real evidence
              something's wrong -- either this isn't the patient, or the
              patient's face didn't score well enough. Drop the lock.
            - MATCHED: confirms + re-seeds the tracker to correct drift,
              same as before.
          The lock is only otherwise dropped if the CSRT tracker itself
          reports failure (lost the visual target entirely) AND no face
          is available to recover onto.
        """
        self.frame_count += 1
        now = time.time()

        attempt_recognition_this_frame = (
            self.state == State.SEARCHING
            and self.frame_count % RECOGNITION_EVERY_N_FRAMES == 0
        ) or (
            self.state == State.TRACKING
            and now - self.last_verified_time > FORCED_RECHECK_INTERVAL_SEC
        )

        if self.state == State.SEARCHING:
            self.last_status = "SEARCHING"
            self.last_similarity = None

            if attempt_recognition_this_frame or self.frame_count == 1:
                face_loc, sim, reason = self.identifier.find_patient(frame_bgr)
                if reason == "MATCHED":
                    body_bbox = face_bbox_to_body_bbox(
                        face_loc, None, frame_bgr.shape
                    )
                    ok = self.tracker.start(frame_bgr, body_bbox)
                    if ok:
                        self.state = State.TRACKING
                        self.last_verified_time = now
                        self.last_status = "VERIFIED"
                        self.last_similarity = sim
                        self.logger.log("VERIFIED", sim, self.frame_count)
                        print(f"[PatientTargetLock] Locked on. similarity={sim:.2f}")
                        return body_bbox
            return None  # still searching, robot should idle/rotate to scan

        elif self.state == State.TRACKING:
            tracker_success, body_bbox = self.tracker.update(frame_bgr)

            # Only force a recognition attempt right now if the tracker
            # itself just failed -- otherwise recognition attempts happen
            # on the normal opportunistic timer above.
            do_recognition_now = (
                (not tracker_success) or attempt_recognition_this_frame
            )

            if do_recognition_now:
                face_loc, sim, reason = self.identifier.find_patient(frame_bgr)

                if reason == "MATCHED":
                    # Confirmed patient still visible; re-seed the tracker
                    # off the fresh face detection to correct any drift.
                    body_bbox = face_bbox_to_body_bbox(
                        face_loc, None, frame_bgr.shape
                    )
                    self.tracker.start(frame_bgr, body_bbox)
                    self.last_verified_time = now
                    self.last_status = "VERIFIED"
                    self.last_similarity = sim
                    self.logger.log("VERIFIED", sim, self.frame_count)
                    return body_bbox

                elif reason == "REJECTED":
                    # A face WAS visible and it did NOT match -- real
                    # mismatch evidence, not just "face out of view".
                    # Safety-critical: don't keep following a possibly
                    # wrong person.
                    print("[PatientTargetLock] Re-verification found a "
                          "face that did NOT match -- dropping lock "
                          "-> SEARCHING")
                    self.state = State.SEARCHING
                    self.last_status = "LOST"
                    self.last_similarity = None
                    self.logger.log("LOST", None, self.frame_count)
                    return None

                else:  # reason == "NO_FACE"
                    if tracker_success:
                        # Inconclusive -- no face visible right now (too
                        # far, turned away, etc), but the tracker is still
                        # confidently following the same body. Keep going;
                        # this is the fix for losing lock at distance.
                        self.last_status = "TRACKING"
                        # Don't reset last_verified_time -- we still want
                        # to keep trying opportunistically on the normal
                        # schedule, not restart a fresh countdown every
                        # single NO_FACE frame.
                        return body_bbox
                    else:
                        # Tracker failed AND no face to recover onto --
                        # genuinely nothing to continue tracking with.
                        print("[PatientTargetLock] Tracker failed and no "
                              "face available to recover -- dropping lock "
                              "-> SEARCHING")
                        self.state = State.SEARCHING
                        self.last_status = "LOST"
                        self.last_similarity = None
                        self.logger.log("LOST", None, self.frame_count)
                        return None

            if tracker_success:
                self.last_status = "TRACKING"
                return body_bbox

            # Tracker failed on a frame where we didn't attempt recognition
            # (shouldn't normally happen since tracker failure forces
            # do_recognition_now above, but handled defensively).
            self.state = State.SEARCHING
            self.last_status = "LOST"
            self.last_similarity = None
            self.logger.log("LOST", None, self.frame_count)
            return None

        return None

    def close(self):
        """Call when shutting down so the log file is closed cleanly."""
        self.logger.close()


# ----------------------------------------------------------------------------
# Example integration skeleton -- adapt to your actual capture/UART code
# ----------------------------------------------------------------------------


# Box color per status, so you can see at a glance whether the robot is
# actively confirming identity, coasting on the tracker, or has lost lock:
#   VERIFIED  -> green  : face recognition just confirmed this IS the patient
#   TRACKING  -> yellow : CSRT following, no recognition run this frame
#   LOST      -> red    : lock dropped, robot is back to SEARCHING
_STATUS_COLORS = {
    "VERIFIED": (0, 255, 0),
    "TRACKING": (0, 220, 255),
    "LOST": (0, 0, 255),
    "SEARCHING": (0, 0, 255),
}


def draw_status(frame, target_bbox, status, similarity):
    """Draws the tracked box (if any) color-coded by status, plus a text
    label in the corner so it's readable even when the box is small."""
    color = _STATUS_COLORS.get(status, (255, 255, 255))

    if target_bbox is not None:
        x, y, w, h = target_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = status if similarity is None else f"{status} ({similarity:.2f})"
        cv2.putText(frame, label, (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Always-visible status line, useful even while SEARCHING (no box yet)
    cv2.putText(frame, f"state: {status}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


if __name__ == "__main__":
    # One-time setup (run this separately, comment out afterwards):
    # register_patient(["patient_photo_1.jpg", "patient_photo_2.jpg", ...])

    print("[main] Loading PatientIdentifier (registered embedding)...")
    lock = PatientTargetLock()
    print("[main] PatientTargetLock ready.")

    print("[main] Opening webcam (index 0)...")
    cap = cv2.VideoCapture(0)  # your USB webcam

    if not cap.isOpened():
        print("[main] ERROR: cv2.VideoCapture(0) failed to open. "
              "Camera index 0 may not exist, may be in use by another "
              "app (close Zoom/Teams/browser tabs using the camera), "
              "or you may have more than one camera where index 0 isn't "
              "the one you expect. Try index 1 instead: cv2.VideoCapture(1)")
        raise SystemExit(1)

    print("[main] Webcam opened successfully. Entering main loop "
          "(press 'q' in the video window to quit)...")

    frame_num = 0
    while True:
        ret, frame = cap.read()
        frame_num += 1

        if not ret:
            print(f"[main] cap.read() failed at frame {frame_num} "
                  f"(ret=False) -- webcam stopped delivering frames. Exiting.")
            break

        if frame_num == 1:
            print(f"[main] First frame captured successfully, shape={frame.shape}")

        target_bbox = lock.update(frame)
        draw_status(frame, target_bbox, lock.last_status, lock.last_similarity)

        if target_bbox is not None:
            pass
            # TODO: feed target_bbox into your existing steering logic from
            # person_follow_uL_uR.py, e.g.:
            #   uL, uR = compute_steering(target_bbox, frame.shape)
            #   send_motor_command(uL, uR)
        else:
            # TODO: idle / scan-in-place behavior while searching for patient
            pass

        cv2.imshow("Patient Lock", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    lock.close()


# ----------------------------------------------------------------------------
# Note on face_recognition / dlib build time on Pi 4
# ----------------------------------------------------------------------------
# dlib compiles from source on ARM and can take 30-60+ min on a Pi 4, and
# needs decent free RAM (add swap if the build gets OOM-killed). If that
# becomes a blocker given your timeline, a lighter alternative is an
# ONNX-exported MobileFaceNet model run via onnxruntime -- same
# similarity-+-margin logic above applies, just swap out
# PatientIdentifier's internals for a different embedding call. Flagging
# this now so it's not a surprise mid-integration.
