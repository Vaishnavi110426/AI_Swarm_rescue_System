import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv, Bottleneck, C3, SPPF
from torch.nn.modules.container import Sequential
import cv2

# Allow YOLO modules during torch.load
torch.serialization.add_safe_globals([DetectionModel, Conv, Bottleneck, C3, SPPF, Sequential])

# Load model
model = YOLO("yolov8n.pt")  # CPU-only works fine

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            if label in ["person", "chair", "bottle", "dog", "cat"]:
                cv2.putText(frame, "🚨 Obstacle Ahead!", (50,50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("Drone Vision Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
