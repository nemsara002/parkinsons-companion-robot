from ultralytics import YOLO

print("Loading model...")

model = YOLO("yolov8n.pt")

print("YOLO loaded successfully!")