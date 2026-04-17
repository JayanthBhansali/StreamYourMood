# StreamYourMood 🎵

A mood-based music recommendation system that detects your facial emotion in real-time and plays music that matches how you feel.

## How It Works

1. **Mood Detection** — Captures your face via webcam (or a static image) and classifies your emotion using a pre-trained deep learning model.
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
├── main.py                            # CLI entry point
├── stream_your_mood.py                # GUI entry point (Tkinter)
├── globalSettings.py                  # Config: emotion-genre mappings, DB path, webcam toggle
├── db.py                              # SQLite database initialization
├── requirements.txt                   # Python dependencies
├── FacialEmotionRecognition/
│   ├── facial.py                      # Face detection + emotion classification
│   ├── fer.json                       # DNN model architecture
│   ├── fer.h5                         # Pre-trained model weights
│   └── haarcascade_frontalface_default.xml  # Haar Cascade face detector
├── AudioClassification/
│   ├── audio.py                       # Audio genre classification
│   └── models/
│       ├── custom_cnn_2d.h5           # CNN model for genre classification
│       └── pipe_svm.joblib            # SVM pipeline model
└── assets/
    ├── images/loading.gif             # Loading animation
    └── fonts/GothamLight.ttf          # Custom UI font
```

## Tech Stack

| Layer              | Technology                        |
|--------------------|-----------------------------------|
| Language           | Python 3                          |
| Emotion Detection  | Keras DNN + OpenCV Haar Cascade   |
| Audio Classification | Custom CNN (mel-spectrograms)   |
| GUI                | Tkinter + Pygame                  |
| Database           | SQLite3 (song cache)              |
| Audio Features     | librosa                           |

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

### GUI (recommended)

```bash
python stream_your_mood.py
```

### CLI

```bash
python main.py
```

## Configuration

Edit `globalSettings.py` to change behaviour:

| Setting               | Default  | Description                                      |
|-----------------------|----------|--------------------------------------------------|
| `use_webcam`          | `False`  | Set to `True` to use live webcam input           |
| `save_images`         | `True`   | Save captured frames to `saved_images/`          |
| `DBPath`              | `audio.db` | Path to the SQLite database                    |

## Supported Audio Formats

- `.mp3`
- `.wav`

## Notes

- Songs are analyzed and cached in `audio.db` on first run — subsequent runs are faster.
- If no songs match the detected genre, the app falls back to playing the highest-rated song in the database.
- Emotion detection retries up to 3 times if no face is found.
