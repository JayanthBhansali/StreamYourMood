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
_input_shape = _session.get_inputs()[0].shape  # e.g. [None,48,48,1] or [batch,1,48,48]

# Detect format: NHWC (Keras, last dim == 1) vs NCHW (PyTorch, second dim == 1)
_is_nchw = _input_shape[-1] != 1

# Label order matches model training:
# - Keras model: trained with a fixed label list
# - PyTorch model: ImageFolder sorts classes alphabetically
if _is_nchw:
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
else:
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print(f"Loaded FER model  |  format: {'NCHW (PyTorch)' if _is_nchw else 'NHWC (Keras)'}")


def detect_emotion(full_size_image):
    face  = cv2.CascadeClassifier(os.path.join(base_path, 'haarcascade_frontalface_default.xml'))
    faces = face.detectMultiScale(full_size_image, 1.3, 10)

    for (x, y, w, h) in faces:
        roi     = full_size_image[y:y + h, x:x + w]
        cropped = cv2.resize(roi, (48, 48)).astype(np.float32)

        if _is_nchw:
            # PyTorch model: pixel → [0,1] → normalise with mean=0.5 std=0.5
            inp = ((cropped / 255.0 - 0.5) / 0.5).reshape(1, 1, 48, 48)
        else:
            # Keras model: L2 normalise
            cv2.normalize(cropped, cropped, alpha=0, beta=1,
                          norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            inp = cropped.reshape(1, 48, 48, 1)

        pred = _session.run(None, {_input_name: inp})[0]
        return labels[int(np.argmax(pred))]

    return None
