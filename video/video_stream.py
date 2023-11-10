import cv2
from ultralytics import YOLO
import time


class VideoStream:
    def __init__(self, url: str):
        self.url = url
        self.video_ = cv2.VideoCapture(self.url)
        self.model = YOLO("best.pt")
        self.detection_active = True
        self.last_detection_time = None

    def real_time_detection(self):
        cap = cv2.VideoCapture(self.url)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = time.time()
            if self.detection_active:
                if self.last_detection_time and current_time - self.last_detection_time < 60:
                    self.detection_active = False
                else:
                    results = self.model(frame)
                    if results:
                        self.last_detection_time = current_time
                        self.detection_active = False
            else:
                if current_time - self.last_detection_time >= 60:
                    self.detection_active = True


# VideoStream('pexels_videos_1338598 (1080p).mp4').real_time_detection()
