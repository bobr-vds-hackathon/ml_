import cv2

data = {"login": "admin",
        "password": "A1234567"}


class VideoStream:
    def __init__(self, url: str, location: []):
        self.url = url
        self.location = location

        self.video_ = cv2.VideoCapture(self.url)
        self.x, self.y = self.location[0], self.location[1]
