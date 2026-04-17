"""
One-time conversion script — run locally (TensorFlow must be installed).

    python convert_models.py

Converts Keras .h5 models to ONNX format so the app can run without TensorFlow.
Uses from_function (concrete function) to work with Keras 3.x / TF 2.16+.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def convert_fer():
    import tensorflow as tf
    from tensorflow import keras
    import tf2onnx
    import onnx

    model = keras.Sequential([
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

    model.load_weights('FacialEmotionRecognition/fer.h5', by_name=True)

    spec = tf.TensorSpec([None, 48, 48, 1], tf.float32)
    func = tf.function(lambda x: model(x, training=False))
    proto, _ = tf2onnx.convert.from_function(func, input_signature=[spec], opset=13)

    out = 'FacialEmotionRecognition/fer.onnx'
    onnx.save(proto, out)
    print(f"✓  FER model  →  {out}")


def convert_audio():
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import tf2onnx
    import onnx

    genres = {
        'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
        'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9,
    }
    model = load_model(
        'AudioClassification/models/custom_cnn_2d.h5',
        custom_objects=genres,
        compile=False,
    )
    input_shape = model.input_shape[1:]
    print(f"\nAudio CNN input shape: {input_shape}")

    spec = tf.TensorSpec([None] + list(input_shape), tf.float32)
    func = tf.function(lambda x: model(x, training=False))
    proto, _ = tf2onnx.convert.from_function(func, input_signature=[spec], opset=13)

    out = 'AudioClassification/models/audio_cnn.onnx'
    onnx.save(proto, out)
    print(f"✓  Audio CNN  →  {out}")


if __name__ == '__main__':
    print("Converting Keras models to ONNX...\n")
    convert_fer()
    convert_audio()
    print("\nConversion complete.")
    print("Commit the .onnx files, then push to deploy without TensorFlow.")
