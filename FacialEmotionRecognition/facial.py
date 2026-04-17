import os, sys
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort

sys.path.append(str(Path(__file__).resolve().parents[1]))
from globalSettings import *

base_path = os.path.dirname(os.path.realpath(__file__))

_session    = ort.InferenceSession(os.path.join(base_path, 'fer.onnx'))
_input_name = _session.get_inputs()[0].name
print("Loaded FER model from disk")

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def detect_emotion(full_size_image):
    face  = cv2.CascadeClassifier(os.path.join(base_path, 'haarcascade_frontalface_default.xml'))
    faces = face.detectMultiScale(full_size_image, 1.3, 10)

    for (x, y, w, h) in faces:
        roi     = full_size_image[y:y + h, x:x + w]
        cropped = cv2.resize(roi, (48, 48)).astype(np.float32)
        cv2.normalize(cropped, cropped, alpha=0, beta=1,
                      norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        inp  = cropped.reshape(1, 48, 48, 1)
        pred = _session.run(None, {_input_name: inp})[0]
        return labels[int(np.argmax(pred))]

    return None
