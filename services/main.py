import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
import tensorflow as tf
from emotion_main import EmotionHandler


def race_prediction(image_crop: np.ndarray, race_model):
    IMAGE_HEIGHT = 100
    IMAGE_WIDTH = 100
    ID_RACE_MAP = {0: 'caucasian', 1: 'mongoloid', 2: 'negroid'}

    resized_image = cv2.resize(image_crop, (IMAGE_HEIGHT, IMAGE_WIDTH))
    resized_image = np.array(resized_image).reshape(
        (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))/255
    prediction = race_model.predict(resized_image)
    y_pred = ID_RACE_MAP[np.argmax(prediction)]

    return y_pred


def detect_skin_tone(detected_face, model):
    classes = ['dark', 'mid-dark', 'mid-light', 'light']

    detected_face = cv2.resize(detected_face, (120, 90))
    detected_face = tf.keras.applications.mobilenet_v2.preprocess_input(
        detected_face[np.newaxis, ...])

    predictions = model.predict(detected_face)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = classes[predicted_class_idx]
    return predicted_class


def main():
    race_model_path = ""
    skintone_model_path = ""
    mask_model_path = ""

    emotion_handler = EmotionHandler()
    img = cv2.imread(os.path.join(os.path.dirname(
        __file__), '../data/mask/test/968261.jpg'))
    emotion_prediction = emotion_handler.emotions_predict(img)
    print(emotion_prediction)


if __name__ == "__main__":
    main()
