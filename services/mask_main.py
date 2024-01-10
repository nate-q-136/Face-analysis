from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
import json


from common.mask.mask_utils import mask_prediction, face_prediction, get_list_bbox

def main():
    # Prepare model
    prototxt_path = "../models/face/v1/deploy.prototxt"
    weights_path = "../models/face/v1/res10_300x300_ssd_iter_140000.caffemodel"
    mask_model_path = "../models/mask/mask_detector.model"

    face_net = cv2.dnn.readNet(prototxt_path, weights_path)
    mask_net = load_model(mask_model_path)

    # Prepare data test
    base_folder = "../data/mask/test"
    image_paths = glob.glob(os.path.join(base_folder, "*.jpg"))

    # lấy labels
    labels_path = "../data_contest/labels/labels.csv"
    df = pd.read_csv(labels_path)
    filtered_masked_df = df[df["masked"] == 'masked']

    print(filtered_masked_df.head())
    # chọn ảnh đầu tiên
    test_path = image_paths[0]
    test_image = cv2.imread(test_path)
    
    cv2.imshow("ádasd", test_image)
    cv2.waitKey(0)
    print(test_path)
    test_bbox = filtered_masked_df[filtered_masked_df['file_name'] == os.path.basename(test_path)]['bbox'].values

    float_test_bbox = get_list_bbox(test_bbox[0])
    print(float_test_bbox)
    x,y,w,h = map(int, float_test_bbox)  



    pred_mask = mask_prediction(test_image, float_test_bbox, mask_net)

    for pred in pred_mask:
        mask, unmasked = pred

        label = "Mask" if mask > unmasked else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, unmasked) * 100)

        cv2.putText(test_image, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        
        cv2.rectangle(test_image, (x, y), (x + w, y + h), color, 2)
        
    print(pred_mask)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB) 
    plt.tight_layout() 
    plt.figure(figsize=(20, 10))  
    plt.imshow(test_image)
    plt.title(os.path.basename(test_path))
    plt.show()

    pass


if __name__ == "__main__":
    main()
