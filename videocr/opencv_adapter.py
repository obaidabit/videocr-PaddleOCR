from __future__ import annotations

import cv2


class Capture:
    def __init__(self, video_path: str):
        self.path = video_path

    def __enter__(self) -> 'cv2.VideoCapture':
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError('Can not open video {}.'.format(self.path))
        return self.cap

    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()
