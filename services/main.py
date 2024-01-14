import numpy as np  
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
from keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.applications.mobilenet_v2
from emotion_main import EmotionHandler

def race_prediction(image_crop:np.ndarray, race_model):
    IMAGE_HEIGHT = 100
    IMAGE_WIDTH = 100
    ID_RACE_MAP = {0: 'caucasian', 1: 'mongoloid', 2: 'negroid'}

    resized_image = cv2.resize(image_crop, (IMAGE_HEIGHT, IMAGE_WIDTH))
    resized_image = np.array(resized_image).reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))/255
    prediction = race_model.predict(resized_image)
    y_pred = ID_RACE_MAP[np.argmax(prediction)]

    return y_pred

def skintone_prediction(detected_face, model):
    classes = ['dark', 'mid-dark', 'mid-light', 'light']

    detected_face = cv2.resize(detected_face, (120, 90))
    detected_face = preprocess_input(detected_face[np.newaxis, ...])
    
    predictions = model.predict(detected_face)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = classes[predicted_class_idx]
    return predicted_class

def mask_prediction(image_crop:np.ndarray, mask_net):
    face = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)

    face = np.expand_dims(face, axis=0)
    preds = mask_net.predict(face, batch_size=32)

    for pred in preds:
        mask, unmasked = pred
        label = "masked" if mask > unmasked else "unmasked"
        return label


def main():
    # Model paths
    race_model_path = "../models/race/race_classification_resnet50v2.h5"
    skintone_model_path = "../models/skin_tone/skin_tone_model.h5"
    mask_model_path = "../models/mask/mask_detector.model"
    emotion_model_path = "../models/emotion/fer2013_mini_XCEPTION.110-0.65.hdf5"
    face_model_path = ""
    gender_model_path = ""
    age_model_path = ""

    # Load models
    race_model = load_model(race_model_path)
    skintone_model = load_model(skintone_model_path)
    mask_model = load_model(mask_model_path)

    # Image path 
    image_path = "/Users/lequangnhat/My Study/3-CNN-Tensorflow/15-Race-Detection/data/images/mongoloid/10123220.jpg"
    image = cv2.imread(image_path)
    # Face detection
    bbox = (1,2,3,4)
    image_crop = np.ones((100,100,3))

    # Race detection
    race = race_prediction(image_crop=image, race_model=race_model)
    # Skintone detection
    skintone = skintone_prediction(detected_face=image, model=skintone_model)
    # Mask detection
    mask = mask_prediction(image_crop=image, mask_net=mask_model)
    emotion_handler = EmotionHandler()
    emotion_prediction = emotion_handler.emotions_predict(image)
    print(emotion_prediction)
    print(mask, skintone, race)
    
    
if __name__ == "__main__":
    main()