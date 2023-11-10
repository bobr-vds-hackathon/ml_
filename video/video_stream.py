import cv2
from ultralytics import YOLO
import time


class VideoStream:
    def __init__(self, url: str):
        self.url = url
        self.video_ = cv2.VideoCapture(self.url)
        self.model = YOLO("video/best.pt")
        self.detection_active = True
        self.last_detection_time = None

    def real_time_detection(self):
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
                                    tmp_pic = result.orig_img
                                    bounding_box = result.boxes.xyxy[i].cpu().numpy()
                                    print(bounding_box)
                                    if len(result.boxes) != 0:
                                        yield result.orig_img
                else:
                    if current_time - self.last_detection_time >= 2:
                        self.detection_active = True
        finally:
            cap.release()

