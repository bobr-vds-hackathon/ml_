import traceback
import cv2
from ultralytics import YOLO
import time
import sys


class VideoStream:
    def __init__(self, url: str):
        self.url = url
        self.video_ = cv2.VideoCapture(self.url)
        self.model = YOLO(sys.path[0]+"/video/best.pt")
        self.detection_active = True
        self.last_detection_time = None

    def real_time_detection(self):
        try:
            cap = cv2.VideoCapture(self.url)
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    current_time = time.time()
                    if self.detection_active:
                        if self.last_detection_time and current_time - self.last_detection_time < 60:
                            self.detection_active = False
                        else:
                            results = self.model(frame)[0]
                            if not results or len(results) == 0:
                                continue
                            else:
                                self.detection_active = False
                                for result in results:
                                    detection_count = result.boxes.shape[0]
                                    for i in range(detection_count):
                                        self.last_detection_time = current_time
                                        bounding_box = result.boxes.xyxy[i].cpu().numpy()
                                        # Draw bounding box
                                        cv2.rectangle(frame,
                                                      (int(bounding_box[0]), int(bounding_box[1])),
                                                      (int(bounding_box[2]), int(bounding_box[3])),
                                                      (0, 255, 0), 2)
                                        print(bounding_box, flush=True)
                                        if len(result.boxes) != 0:
                                            yield frame
                    else:
                        if current_time - self.last_detection_time >= 100:
                            self.detection_active = True
            finally:
                cap.release()
        except FileNotFoundError:
            traceback.print_exc()
