import os

import cv2
from ultralytics import YOLO
import pandas as pd
from .tracker import Tracker
import uuid
import os
import torch

def gradien(titik1, titik2):
    x1, y1 = titik1
    x2, y2 = titik2
    
    return ((y2 - y1) / (x2 - x1))
    

def is_point_on_line(x, y, ujung_kiri, ujung_kanan):
    # Tentukan gradien antara titik acak (x,y) terhadap titik ujung kiri.
  m_titik = round(gradien((x,y), ujung_kanan))

  # Tentukan gradien dari ujung kiri ke ujung kanan.
  m_garis_batas = round(gradien(ujung_kiri, ujung_kanan))
  
  # Periksa apakah kedua gradien tersebut sama.
  if m_titik == m_garis_batas:
    return True
  else:
    return False

class Model:
    """Model class"""
    weight_name = "best (2).pt"
    #weight_name = "yolov8x.pt"
    model = None

    def __init__(self):
        for f in os.listdir('streamapp/utils'):
            if f == self.weight_name:
                self.model = YOLO(f'streamapp/utils/{f}')
            else:
                print("Model is not found!")

    def detect(self, original_image):
        tracker = Tracker()
        # Load Model
        if self.model is None:
            raise Exception("NO MODEL!")
        # else:
            # print("Model is", self.model)

        try:
            frame = original_image
            # Run inference from input
            results = self.model.predict(frame, device=0, verbose=False)
            a = results[0].boxes.data.cpu()
            px = pd.DataFrame(a).astype("float")
            
            list = []
                                                
            if not px.empty:
                for index, row in px.iterrows():
                    x1 = int(row[0])
                    y1 = int(row[1])
                    x2 = int(row[2])
                    y2 = int(row[3])
                    d = int(row[5])
                    list.append([x1, y1, x2, y2])
                
                bbox_id = tracker.update(list)
                for bbox in bbox_id:
                    x3, y3, x4, y4, id = bbox
                    cx = int(x3 + x4) // 2
                    cy = int(y3 + y4) // 2
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                    
                if is_point_on_line(cx, cy, (1, 268), (534, 476)):
                    cv2.imwrite(f'{os.getcwd()}/garbagedetection/output/{uuid.uuid4().hex}.jpg', original_image)
                    print("Recorded!")

            else:
                print("Data kosong")                
            
            cv2.line(frame, (1, 268), (534, 476), (0,255,0), 2)
            # Visualize results on the frame
            # annotated_frame = results[0].plot()
            # annotated_frame = cv2.rectangle(annotated_frame, (900, 900), (1080, 1920), (255, 0, 0), 2)

            # return as image
            _, jpg = cv2.imencode('.jpg', frame)

            return jpg.tobytes()
        except TypeError as e:
            print("Error: ", e)
