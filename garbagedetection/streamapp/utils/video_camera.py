import threading

import cv2


class VideoCamera(object):

    def __init__(self, position=0):
        self.video = cv2.VideoCapture(position)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        return self.frame

    # def to_frame(self, frame):
    #     _, jpeg = cv2.imencode('jpg', frame)
    #     return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()
