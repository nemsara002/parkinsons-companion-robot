# Semi-Autonomous Assistive Companion Robot for Parkinson's Patients

An individual design project: a differential-drive mobile robot that follows a Parkinson's
patient around the home, detects falls from a camera feed, and provides AI-based companionship.
The platform runs on a Raspberry Pi 4 (high-level vision and AI) paired with an ESP32
(low-level real-time motor control).

## Subsystems

| Subsystem | Status | Description |
|---|---|---|
| **Fall detection** | Complete | Vision-based fall detection using MediaPipe pose, handcrafted skeletal + bounding-box temporal features, and a Random Forest classifier. |
| **Person following** | Implemented | Follows the identified user at a safe distance using a fused body-size distance metric and horizontal-offset steering, output as differential-drive `uL`/`uR` signals. |
| **Face recognition** | Planned | Identifies the registered patient under normal lighting. |
| **AI companion** | Planned | Voice interaction, medication reminders, and routine notifications. |

## Repository structure

```
fall_detection/
  training/      Dataset feature extraction, windowing, model training, clip diagnostics
  deployment/    Live detection scripts (Raspberry Pi and laptop)
  evaluation/    Generates evaluation plots for the report
  tests/         Feature and camera sanity checks
  models/        Trained classifier (.pkl)
person_following/  Person-following controller (outputs uL/uR to ESP32)
docs/              Report assets, including evaluation plots
```

## Fall detection pipeline

1. **`training/extract_features.py`** — runs MediaPipe pose over the URFD dataset clips and
   extracts per-frame skeletal and bounding-box features.
2. **`training/make_windows.py`** — builds windowed temporal features so the model can see the
   *motion signature* of a collapse rather than single frames.
3. **`training/train_fall_model.py`** — trains a Random Forest with GroupKFold cross-validation.
4. **`evaluation/generate_eval_plots.py`** — produces the confusion matrix, ROC curve,
   precision-recall curve, feature importance, and threshold-sweep plots.
5. **`deployment/live_fall_detection.py`** — runs live on the Pi with per-session calibration and
   an `on_fall_detected()` hook for the voice check-in and ESP32 alert.

### Design notes
- **Recall over precision.** The decision threshold is set low (0.25) to favour catching real
  falls; a downstream voice check-in acts as the false-alarm filter.
- **Temporal features matter.** Frame-by-frame classification misses the collapse motion, so
  windowed features are used.
- **Bounding-box features matter.** Skeletal angles alone are blind to falls toward/away from the
  camera; bbox/silhouette features capture that depth axis.

## Hardware
Raspberry Pi 4 · ESP32 · Raspberry Pi Camera Module 3 NoIR · N20 geared motors (TB6612FNG driver)
· HC-SR04 ultrasonic sensors · MPU6050 IMU

## Setup
```bash
pip install -r requirements.txt
```
Note: `picamera2` is pre-installed on Raspberry Pi OS and is not required on the laptop.

## Author
R.M.K.N.B. Ranasinghe — Individual Mechatronics Design Project
