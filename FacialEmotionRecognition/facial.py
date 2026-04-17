# load json and create model
import os, sys
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2
from tensorflow import keras

sys.path.append(str(Path(__file__).resolve().parents[1]))
from globalSettings import *

base_path = os.path.dirname(os.path.realpath(__file__))


def _build_fer_model():
    """Rebuild the FER architecture with original Keras 2 layer names so that
    weights can be loaded by name from the legacy fer.h5 file."""
    return keras.Sequential([
        keras.layers.Input(shape=(48, 48, 1)),
        keras.layers.Conv2D(64, (3, 3), activation='relu',
                            kernel_regularizer=keras.regularizers.L2(0.01),
                            name='conv2d_1'),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',  name='conv2d_2'),
        keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001,        name='batch_normalization_1'),
        keras.layers.MaxPooling2D((2, 2),                                    name='max_pooling2d_1'),
        keras.layers.Dropout(0.5,                                            name='dropout_1'),

        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_3'),
        keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001,        name='batch_normalization_2'),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_4'),
        keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001,        name='batch_normalization_3'),
        keras.layers.MaxPooling2D((2, 2),                                    name='max_pooling2d_2'),
        keras.layers.Dropout(0.5,                                            name='dropout_2'),

        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_5'),
        keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001,        name='batch_normalization_4'),
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_6'),
        keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001,        name='batch_normalization_5'),
        keras.layers.MaxPooling2D((2, 2),                                    name='max_pooling2d_3'),
        keras.layers.Dropout(0.5,                                            name='dropout_3'),

        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_7'),
        keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001,        name='batch_normalization_6'),
        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_8'),
        keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001,        name='batch_normalization_7'),
        keras.layers.MaxPooling2D((2, 2),                                    name='max_pooling2d_4'),
        keras.layers.Dropout(0.5,                                            name='dropout_4'),

        keras.layers.Flatten(name='flatten_1'),
        keras.layers.Dense(512, activation='relu',    name='dense_1'),
        keras.layers.Dropout(0.4,                     name='dropout_5'),
        keras.layers.Dense(256, activation='relu',    name='dense_2'),
        keras.layers.Dropout(0.4,                     name='dropout_6'),
        keras.layers.Dense(128, activation='relu',    name='dense_3'),
        keras.layers.Dropout(0.5,                     name='dropout_7'),
        keras.layers.Dense(7,   activation='softmax', name='dense_4'),
    ], name='sequential_1')


if useVGG:
    IMG_SIZE = 48
    model = keras.models.load_model(os.path.join(base_path, "VGG16.hdf5"))
else:
    loaded_model = _build_fer_model()
    loaded_model.load_weights(os.path.join(base_path, "fer.h5"), by_name=True)

print("Loaded model from disk")

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#loading image
#full_size_image = cv2.imread("sad-face.jpg",0)
def detect_emotion(full_size_image):
    emotion = None
    gray = full_size_image
    print("Image Loaded")
    #gray=cv2.cvtColor(full_size_image,cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier(os.path.join(base_path,'haarcascade_frontalface_default.xml'))
    faces = face.detectMultiScale(full_size_image, 1.3  , 10)

    #detecting faces
    for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            #predicting the emotion
            if useVGG:
                #test_image = np.array(cropped_img).reshape( -1, IMG_SIZE, IMG_SIZE, 1)
                prediction = model.predict({'input_1': cropped_img })
                emotion = ''
                for i in range(0,6):
                    if(prediction[0][i] == 1):
                        emotion = labels[i]
                    
                    #print("Emotion : ", emotion)
            else:
                yhat= loaded_model.predict(cropped_img)
                #cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                #print("Emotion: "+labels[int(np.argmax(yhat))])
                emotion = labels[int(np.argmax(yhat))]

    #cv2.imshow('Emotion', full_size_image)
    cv2.destroyAllWindows()
    return emotion