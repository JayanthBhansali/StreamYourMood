import os
import numpy as np
import librosa
import onnxruntime as ort

genres = {
    'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
    'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9,
}

base_path   = os.path.dirname(os.path.realpath(__file__))
_session    = ort.InferenceSession(os.path.join(base_path, 'models/audio_cnn.onnx'))
_input_name = _session.get_inputs()[0].name


def majority_voting(scores, dict_genres):
    preds  = np.argmax(scores, axis=1)
    values, counts = np.unique(preds, return_counts=True)
    counts = np.round(counts / np.sum(counts), 2)
    votes  = dict(sorted(zip(values, counts), key=lambda x: x[1], reverse=True))
    return [(get_genres(k, dict_genres), v) for k, v in votes.items()]


def get_genres(key, dict_genres):
    return {v: k for k, v in dict_genres.items()}[key]


def splitsongs(X, overlap=0.5):
    chunk  = 33000
    offset = int(chunk * (1. - overlap))
    return np.array([
        X[i:i + chunk]
        for i in range(0, X.shape[0] - chunk + offset, offset)
        if X[i:i + chunk].shape[0] == chunk
    ])


def to_melspectrogram(songs, n_fft=1024, hop_length=256):
    melspec = lambda x: librosa.feature.melspectrogram(
        y=x, n_fft=n_fft, hop_length=hop_length, n_mels=128
    )[:, :, np.newaxis]
    return np.array(list(map(melspec, songs)))


def classify_audio(song):
    signal, _ = librosa.load(song, sr=None)
    specs = to_melspectrogram(splitsongs(signal)).astype(np.float32)
    preds = _session.run(None, {_input_name: specs})[0]
    return majority_voting(preds, genres)[:3]
