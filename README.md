# StreamYourMood 🎵

A mood-based music recommendation system that detects your facial emotion in real-time and plays music that matches how you feel.

## How It Works

1. **Mood Detection** — Captures your face via the browser camera and classifies your emotion using a pre-trained deep learning model.
2. **Genre Mapping** — Maps the detected emotion to compatible music genres.
3. **Music Analysis** — Analyzes your music library and classifies each song into genres using a CNN trained on mel-spectrograms.
4. **Playback** — Plays a song from the matched genre with a built-in music player.

### Emotion → Genre Mapping

| Emotion  | Genres             |
|----------|--------------------|
| Happy    | Rock, Pop          |
| Sad      | Blues              |
| Angry    | Hip-Hop            |
| Disgust  | Hip-Hop            |
| Fear     | Classical, Country |
| Surprise | Metal, Country     |
| Neutral  | Jazz, Reggae       |

## Project Structure

```
StreamYourMood/
├── streamlit_app.py                   # Streamlit web app (main entry point)
├── main.py                            # CLI entry point
├── globalSettings.py                  # Config: emotion-genre mappings, DB path
├── db.py                              # SQLite database initialization
├── convert_models.py                  # One-time Keras → ONNX conversion utility
├── requirements.txt                   # Deployment dependencies
├── packages.txt                       # System packages for Streamlit Cloud
│
├── FacialEmotionRecognition/
│   ├── facial.py                      # Inference: face detection + emotion classification
│   ├── train.py                       # Training: CNN on FER-2013 → exports fer.onnx
│   ├── fer.onnx                       # Deployed model (ONNX Runtime)
│   ├── fer.h5                         # Legacy Keras weights (kept for reference)
│   └── haarcascade_frontalface_default.xml  # Haar Cascade face detector
│
├── AudioClassification/
│   ├── audio.py                       # Audio genre classification
│   └── models/
│       ├── audio_cnn.onnx             # Deployed CNN model (ONNX Runtime)
│       ├── custom_cnn_2d.h5           # Legacy Keras weights (kept for reference)
│       └── pipe_svm.joblib            # SVM pipeline model
│
├── dataset/                           # FER-2013 dataset (not in git — download from Kaggle)
│   ├── train/  angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
│   └── test/   angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
│
└── assets/
    ├── images/loading.gif             # Loading animation
    └── fonts/GothamLight.ttf          # Custom UI font
```

## Model Performance

### Emotion (Mood) Detection

| Metric              | Details                                                   |
|---------------------|-----------------------------------------------------------|
| Face Detection      | Pre-trained HAAR Frontal-Face Classifier                  |
| Classification      | Fisherface Algorithm + VGG16 (low-latency inference)      |
| Accuracy            | **92%** (Fisherface) · **83.2%** (Haar Cascades + VGG16) |
| Pre-processing      | Face crop + grayscale conversion                          |
| Training Data       | 16 images per mood category, collected over 5 seconds     |
| Evaluation          | Confusion matrix + precision                              |

### Audio Classification

| Metric              | Details                                                        |
|---------------------|----------------------------------------------------------------|
| Libraries           | PyAudio · librosa                                              |
| Features Extracted  | **36 features** across 4 dimensions                           |
| Feature Dimensions  | Dynamic · Harmony · Rhythm · Spectral                         |
| Best Feature Combo  | Spectral + Dynamic + Harmony (via Feed Forward selection)      |
| Classifier          | SVM with RBF kernel                                            |
| Accuracy            | **81.6%**                                                      |
| Audio File Size     | 5 MB – 10 MB · 30–60 seconds per file                        |

## Tech Stack

| Layer                | Technology                              |
|----------------------|-----------------------------------------|
| Language             | Python 3                                |
| Web UI               | Streamlit                               |
| Emotion Detection    | ONNX Runtime + OpenCV Haar Cascade      |
| Audio Classification | ONNX Runtime (CNN on mel-spectrograms)  |
| Audio Playback       | Browser-native (`st.audio`)             |
| Database             | SQLite3 (song cache)                    |
| Audio Features       | librosa                                 |

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd StreamYourMood
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the App

### Streamlit (recommended)

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501` in your browser. Grant camera permission when prompted.

### CLI

```bash
python main.py
```

## App Flow

1. **Home** — Enter your music folder path and click **Submit**, or click **Use Existing** if you've run the app before.
2. **Analyzing** — Take a photo using your browser camera. The app detects your emotion and classifies any new songs in your folder.
3. **Player** — A song matching your mood plays automatically. Use the controls to Pause, Next, Stop, or Restart.

## Configuration

Edit `globalSettings.py` to change behaviour:

| Setting        | Default    | Description                             |
|----------------|------------|-----------------------------------------|
| `save_images`  | `True`     | Save captured frames to `saved_images/` |
| `DBPath`       | `audio.db` | Path to the SQLite database             |

## Supported Audio Formats

- `.mp3`
- `.wav`

## Notes

- Songs are analyzed and cached in `audio.db` on first run — subsequent runs are faster.
- If no songs match the detected genre, the app falls back to playing the highest-rated song in the database.
- Emotion detection retries up to 3 times if no face is found.
- Camera access requires a browser with permission granted (works on `localhost` without HTTPS).

## Dataset

The facial emotion model was trained on the **FER-2013** dataset.

| Detail       | Info |
|--------------|------|
| Source       | [Kaggle — FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) |
| Classes      | 7 emotions — Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral |
| Images       | ~35,000 grayscale 48×48 images |
| Split        | `train/` and `test/` folders, one sub-folder per class |

> The dataset folder is not included in this repository. Download it from Kaggle and place it under `FacialEmotionRecognition/data/` if you want to retrain the model.

---

## TensorFlow → ONNX Runtime Migration

### Why it was needed

TensorFlow was the original inference engine for both the facial emotion model and the audio genre CNN. It worked locally but caused repeated failures on Streamlit Community Cloud:

- `tensorflow-cpu` was removed as a standalone package in TF 2.16+ — only `tensorflow` exists now.
- Even plain `tensorflow` returned *"No matching distribution found (from versions: none)"* on the cloud, because TensorFlow has no wheels for Python 3.13, and Streamlit Cloud's default Python had moved past 3.11/3.12.
- Pinning `runtime.txt` / `.python-version` to Python 3.11 did not resolve it reliably.

### What changed

| Before | After |
|--------|-------|
| `tensorflow` (~500 MB install) | `onnxruntime` (~8 MB install) |
| `facial.py` loaded `fer.h5` via Keras | `facial.py` loads `fer.onnx` via `ort.InferenceSession` |
| `audio.py` loaded `custom_cnn_2d.h5` via Keras | `audio.py` loads `audio_cnn.onnx` via `ort.InferenceSession` |
| Required Python 3.9–3.12 | Works on Python 3.9–3.13 |
| `runtime.txt` / `.python-version` pins needed | No Python version constraints |

### How it helps

- **Deployment** — `onnxruntime` has pre-built wheels for every platform and Python version from 3.9 to 3.13. No compilation, no CUDA, no version conflicts.
- **Speed** — ONNX Runtime's inference engine is typically faster than TF for single-sample forward passes.
- **Size** — Removes a ~500 MB dependency. The app image on Streamlit Cloud builds in seconds instead of minutes.
- **Portability** — `.onnx` files are a vendor-neutral format; the same files run on TF, PyTorch, Core ML, or any other ONNX-compatible runtime.

### One-time model conversion (local)

The pre-trained weights are stored as Keras `.h5` files. A local conversion script exports them to ONNX once — TensorFlow is only needed for this step, not for running the app.

```bash
pip install tf2onnx onnx
python convert_models.py
# produces: FacialEmotionRecognition/fer.onnx
#           AudioClassification/models/audio_cnn.onnx
```

The generated `.onnx` files are committed to the repository so the deployed app never needs TensorFlow.

---

## Research Paper

This project is based on the following published paper:

**[Music Recommendation Based on Facial Expression](https://www.ijltemas.in/DigitalLibrary/Vol.9Issue11/18-23.pdf)**
Mrudula K, Harsh R Jain, Amogha R Chandra, Jayanth Bhansali
*International Journal of Latest Technology in Engineering, Management & Applied Science (IJLTEMAS)*
Vol. IX, Issue XI, pp. 18–23, December 2020
B.N.M Institute of Technology, Bangalore
