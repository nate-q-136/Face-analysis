from keras import models
import numpy as np
import cv2
import os

model_path = os.path.join(os.path.dirname(
    __file__), '../models/emotion/fer2013_mini_XCEPTION.110-0.65.hdf5')
CLASS_MAPPING = [
    'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral'
]


class EmotionHandler:

    def emotions_predict(self, bboxes: list):
        """
        bboxes: list of image inside bounding box representations
        return: list of predicted labels matching with faces in input faces
        """
        faces = [self.standardize_face(np.ndarray(
            face), grayscale=True) for face in bboxes]
        faces = np.expand_dims(faces, -1)
        emotion_model = models.load_model(model_path)
        predictions = emotion_model.predict(faces)
        prediction_indices = np.argmax(predictions, axis=1)
        prediction_labels = [CLASS_MAPPING[prediction_index]
                             for prediction_index in prediction_indices]
        return prediction_labels

    def standardize_face(self,
                         face: np.ndarray,
                         target_size: tuple = (64, 64),
                         grayscale: bool = False,
                         normalize=True
                         ) -> list:

        if face.shape[0] > 0 and face.shape[1] > 0:
            if grayscale is True:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # resize and padding
            factor_0 = target_size[0] / face.shape[0]
            factor_1 = target_size[1] / face.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (
                int(face.shape[1] * factor),
                int(face.shape[0] * factor),
            )
            face = cv2.resize(face, dsize)

            diff_0 = target_size[0] - face.shape[0]
            diff_1 = target_size[1] - face.shape[1]
            if grayscale is False:
                # Put the base image in the middle of the padded image
                face = np.pad(
                    face,
                    (
                        (diff_0 // 2, diff_0 - diff_0 // 2),
                        (diff_1 // 2, diff_1 - diff_1 // 2),
                        (0, 0),
                    ),
                    "constant",
                )
            else:
                face = np.pad(
                    face,
                    (
                        (diff_0 // 2, diff_0 - diff_0 // 2),
                        (diff_1 // 2, diff_1 - diff_1 // 2),
                    ),
                    "constant",
                )

            # double check: if target image is not still the same size with target.
            if face.shape[0:2] != target_size:
                face = cv2.resize(face, target_size)

            # normalizing the image pixels
            if normalize:
                face = face.astype(np.float32)
                face /= 255.0

        return face
