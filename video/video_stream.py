import cv2
from ultralytics import YOLO
import datetime


class VideoStream:
    def __init__(self, url: str):
        self.url = url
        self.video_ = cv2.VideoCapture(self.url)
        self.model = YOLO("best.pt")

    def rtd(self):
        while True:
            ret, frame = self.video_.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 640))
            results = self.model(frame)
            for pred in results.pred[0]:
                confidence = pred[4]
                if confidence > 0.7:
                    x_min, y_min, x_max, y_max = map(int, pred[:4])
                    yield frame[y_min:y_max, x_min:x_max]

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f'Class: 0', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)
            cv2.imshow('Object Detection', frame)


VideoStream('236.VC19.1.7 Набережная Табло 2023-09-26 18-50-00_000+0300 [8m0s].mp4').rtd()
