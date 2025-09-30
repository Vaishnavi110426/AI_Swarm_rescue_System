import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv, Bottleneck, C3, SPPF
from torch.nn.modules.container import Sequential
import cv2

# Allow YOLO modules during torch.load
torch.serialization.add_safe_globals([DetectionModel, Conv, Bottleneck, C3, SPPF, Sequential])

model = YOLO("yolov8n.pt")

def detect_from_camera(device=0):
    cap = cv2.VideoCapture(device)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        cv2.imshow("detect", annotated[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_from_camera()
