from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
import json


def mask_prediction(image:np.ndarray, bbox:(float, float, float, float), mask_net):
    x, y, w, h = map(int, bbox)
    image_crop = image[y:y+h, x:x+w]
    face = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)

    face = np.expand_dims(face, axis=0)
    preds = mask_net.predict(face, batch_size=32)

    return preds


def face_prediction(image_original:np.ndarray, face_net):
    (h, w) = image_original.shape[:2]
    blob = cv2.dnn.blobFromImage(image_original, 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    
    face_net.setInput(blob)
    detections = face_net.forward() 

    faces = []
    locations = []

    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]

        if conf > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")    

            (x1, y1) = (max(0, x1), max(0, y1))
            (x2, y2) = (min(w - 1, x2), min(h - 1, y2))

            face = image_original[y1:y2, x1:x2]
            if face.any():
                faces.append(face)  
                locations.append((x1, y1, x2, y2))
    
    return (locations, faces)


# utils
def get_list_bbox(bbox_pd) -> (float, float, float, float):
    json_bbox = json.loads(bbox_pd)
    float_test_bbox = [float(i) for i in json_bbox]
    return float_test_bbox


def get_masked_or_unmasked(preds_mask):
    for pred in preds_mask:
        mask, without_mask = pred
        if mask > without_mask:
            return "masked"  
        else:
            return "unmasked"