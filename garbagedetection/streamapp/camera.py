import cv2
import imutils
from imutils.video import FPS
from imutils.video import VideoStream

from . import extract_embeddings


class FaceDetect(object):
    def __init__(self):
        extract_embeddings.embeddings()
        # train_model.model_train()
        # initialize the video stream, then allow the camera sensor to warm up
        self.vs = VideoStream(src=3).start()
        # start the FPS throughput estimator
        self.fps = FPS().start()

    def __del__(self):
        cv2.destroyAllWindows()

    def get_frame(self):
        # grab the frame from the threaded video stream
        frame = self.vs.read()
        frame = cv2.flip(frame, 1)

        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        while True:
            self.fps.update()
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

        # loop over the detections
        # for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        # confidence = detections[0, 0, i, 2]

        # filter out weak detections
        # if confidence > 0.5:
        # compute the (x, y)-coordinates of the bounding box for
        # the face
        # box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        # (startX, startY, endX, endY) = box.astype("int")

        # extract the face ROI
        # face = frame[startY:endY, startX:endX]
        # (fH, fW) = face.shape[:2]

        # ensure the face width and height are sufficiently large
        # if fW < 20 or fH < 20:
        #     continue

        # construct a blob for the face ROI, then pass the blob
        # through our face embedding model to obtain the 128-d
        # quantification of the face
        # faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
        #                                  (96, 96), (0, 0, 0), swapRB=True, crop=False)
        # embedder.setInput(faceBlob)
        # vec = embedder.forward()

        # perform classification to recognize the face
        # preds = recognizer.predict_proba(vec)[0]
        # j = np.argmax(preds)
        # proba = preds[j]
        # name = le.classes_[j]

        # draw the bounding box of the face along with the
        # associated probability
        # text = "{}: {:.2f}%".format(name, proba * 100)
        # y = startY - 10 if startY - 10 > 10 else startY + 10
        # cv2.rectangle(frame, (startX, startY), (endX, endY),
        #               (0, 0, 255), 2)
        # cv2.putText(frame, text, (startX, y),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # update the FPS counter
        # self.fps.update()
        # ret, jpeg = cv2.imencode('.jpg', frame)
        # return jpeg.tobytes()
