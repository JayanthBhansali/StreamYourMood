import streamlit as st
import os
import cv2
import uuid
import random
import tempfile
import numpy as np

from db import create_connection
from globalSettings import DBPath, save_images, emotion_genre_mappings
from FacialEmotionRecognition import facial
from AudioClassification import audio as audio_module


EMOTION_EMOJI = {
    'Angry':    '😠',
    'Disgust':  '🤢',
    'Fear':     '😨',
    'Happy':    '😄',
    'Sad':      '😢',
    'Surprise': '😲',
    'Neutral':  '😐',
}

GENRE_EMOJI = {
    'rock':      '🎸',
    'pop':       '🎤',
    'blues':     '🎷',
    'hiphop':    '🎧',
    'classical': '🎻',
    'country':   '🤠',
    'metal':     '🤘',
    'jazz':      '🎺',
    'reggae':    '🌿',
    'disco':     '🪩',
}


def _inject_css():
    st.markdown("""
    <style>
    /* Page background */
    .stApp { background-color: #0e1117; }

    /* Centered hero title */
    .hero-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }

    /* Step cards on home page */
    .step-card {
        background: #1c1f2e;
        border: 1px solid #2d3148;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        height: 100%;
    }
    .step-icon { font-size: 2rem; }
    .step-num  { color: #a78bfa; font-weight: 700; font-size: 0.8rem;
                 letter-spacing: 0.08em; text-transform: uppercase; }
    .step-text { color: #d1d5db; font-size: 0.9rem; margin-top: 0.3rem; }

    /* Now-playing card */
    .now-playing-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #1e3a5f 100%);
        border: 1px solid #3730a3;
        border-radius: 16px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .np-emotion { font-size: 3.5rem; line-height: 1; }
    .np-label   { color: #a5b4fc; font-size: 0.75rem; letter-spacing: 0.1em;
                  text-transform: uppercase; margin-top: 0.6rem; }
    .np-emotion-name { color: #e0e7ff; font-size: 1.4rem; font-weight: 600; }
    .np-song    { color: #f3f4f6; font-size: 1rem; font-weight: 500;
                  margin-top: 0.8rem; word-break: break-all; }
    .genre-tag  {
        display: inline-block;
        background: #312e81;
        color: #c7d2fe;
        border-radius: 999px;
        padding: 0.2rem 0.9rem;
        font-size: 0.78rem;
        margin-top: 0.5rem;
        letter-spacing: 0.05em;
    }

    /* Playlist row */
    .playlist-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: #1c1f2e;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        cursor: pointer;
        border: 1px solid transparent;
    }
    .playlist-row.active {
        border-color: #6366f1;
        background: #1e1b4b;
    }
    .playlist-name { color: #d1d5db; font-size: 0.88rem; flex: 1; }
    .playlist-name.active { color: #a5b4fc; font-weight: 600; }

    /* Progress step bar */
    .step-bar {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .step-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #374151;
    }
    .step-dot.active { background: #6366f1; }

    /* Hide default Streamlit header padding */
    section[data-testid="stSidebar"] { display: none; }
    </style>
    """, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def createFolderIfnotExists(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o777)


def analyze_songs(music_folder_dir, files):
    conn = create_connection(DBPath)
    c = conn.cursor()
    c.execute("SELECT path, song FROM songs")
    rows = c.fetchall()
    existing_paths = {row[0] for row in rows}
    existing_songs = {row[1] for row in rows}

    for file in files:
        song_path = os.path.join(music_folder_dir, file)
        if song_path not in existing_paths and file not in existing_songs:
            results = audio_module.classify_audio(song_path)
            i = 0
            for genre, val in results:
                if val > 0.5 or i == 0:
                    c.execute(
                        "INSERT INTO songs(path, genre, prediction, song) VALUES(?,?,?,?)",
                        [song_path, genre, val, file]
                    )
                    conn.commit()
                else:
                    break
                i += 1
    conn.close()


def get_songs_for_emotion(detected_emotion):
    if detected_emotion not in emotion_genre_mappings:
        return [], None

    genre_to_play = random.choice(emotion_genre_mappings[detected_emotion])
    conn = create_connection(DBPath)
    c = conn.cursor()

    c.execute("SELECT path FROM songs WHERE genre = ?", (genre_to_play,))
    song_paths = [row[0] for row in c if os.path.isfile(row[0])]

    if not song_paths:
        c.execute("SELECT path FROM songs ORDER BY prediction DESC")
        song_paths = [row[0] for row in c if os.path.isfile(row[0])]

    conn.close()
    return song_paths, genre_to_play


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_home():
    st.markdown('<p class="hero-title">🎵 Stream Your Mood</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Detects your facial emotion and plays music that matches how you feel.</p>', unsafe_allow_html=True)

    # How it works
    c1, c2, c3 = st.columns(3)
    steps = [
        ("📸", "Capture", "Take a photo — we detect your facial emotion in real time."),
        ("🎵", "Match",   "Your emotion is mapped to the perfect music genre."),
        ("▶️",  "Play",    "A matching song starts playing automatically."),
    ]
    for col, (icon, num, text) in zip([c1, c2, c3], steps):
        col.markdown(f"""
        <div class="step-card">
            <div class="step-icon">{icon}</div>
            <div class="step-num">{num}</div>
            <div class="step-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload your music files",
        type=["mp3", "wav"],
        accept_multiple_files=True,
        help="Supports .mp3 and .wav — up to 200 MB per file",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")

    st.write("")
    if st.button("Let's Go →", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("Please upload at least one .mp3 or .wav file.")
        else:
            tmp_dir = tempfile.mkdtemp()
            files = []
            for uf in uploaded_files:
                dest = os.path.join(tmp_dir, uf.name)
                with open(dest, "wb") as f:
                    f.write(uf.getbuffer())
                files.append(uf.name)
            st.session_state.update({
                'music_folder': tmp_dir,
                'files':        files,
                'page':         'analyzing',
            })
            st.rerun()


def page_analyzing():
    st.markdown('<p class="hero-title">🎵 Stream Your Mood</p>', unsafe_allow_html=True)

    # Step indicator: step 2 of 3 active
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot active"></div>
        <div class="step-dot active"></div>
        <div class="step-dot"></div>
    </div>
    """, unsafe_allow_html=True)

    music_folder = st.session_state.get('music_folder', '')
    files        = st.session_state.get('files', [])

    st.subheader("📸 Take a photo")
    st.caption("Make sure your face is clearly visible and well-lit.")
    img_file = st.camera_input("", label_visibility="collapsed")
    if img_file is None:
        st.info("Allow camera access in your browser, then click the capture button.")
        return

    img_array = np.frombuffer(img_file.getvalue(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if save_images:
        createFolderIfnotExists('saved_images')
        cv2.imwrite(os.path.join('saved_images', str(uuid.uuid4()) + '.png'), frame)

    with st.spinner("Detecting your emotion..."):
        detected_emotion = None
        for _ in range(3):
            detected_emotion = facial.detect_emotion(image)
            if detected_emotion:
                break

    if not detected_emotion:
        st.error("No face detected. Make sure your face is fully visible and try again.")
        if st.button("← Try Again"):
            st.session_state['page'] = 'home'
            st.rerun()
        return

    emoji = EMOTION_EMOJI.get(detected_emotion, '🎭')
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:4rem">{emoji}</div>
        <div style="color:#a5b4fc; font-size:0.8rem; text-transform:uppercase;
                    letter-spacing:0.1em; margin-top:0.4rem">Detected</div>
        <div style="color:#e0e7ff; font-size:1.8rem; font-weight:700">{detected_emotion}</div>
    </div>
    """, unsafe_allow_html=True)

    if files:
        with st.spinner(f"Classifying {len(files)} song(s) — this may take a moment..."):
            analyze_songs(music_folder, files)

    with st.spinner("Matching songs to your mood..."):
        song_paths, genre_to_play = get_songs_for_emotion(detected_emotion)

    if not song_paths:
        st.error("No songs found in the database. Go back and upload some music.")
        if st.button("← Go Back"):
            st.session_state['page'] = 'home'
            st.rerun()
        return

    st.session_state.update({
        'detected_emotion': detected_emotion,
        'genre_to_play':    genre_to_play,
        'song_paths':       song_paths,
        'current_song':     random.choice(song_paths),
        'page':             'player',
    })
    st.rerun()


def page_player():
    song_paths   = st.session_state.get('song_paths', [])
    current_song = st.session_state.get('current_song', '')
    emotion      = st.session_state.get('detected_emotion', '')
    genre        = st.session_state.get('genre_to_play', '')

    st.markdown('<p class="hero-title">🎵 Stream Your Mood</p>', unsafe_allow_html=True)

    # Step indicator: all steps done
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot active"></div>
        <div class="step-dot active"></div>
        <div class="step-dot active"></div>
    </div>
    """, unsafe_allow_html=True)

    # Now-playing card
    emoji      = EMOTION_EMOJI.get(emotion, '🎭')
    genre_icon = GENRE_EMOJI.get(genre, '🎵')
    song_name  = os.path.basename(current_song) if current_song else "—"
    # strip extension for display
    display_name = os.path.splitext(song_name)[0]

    st.markdown(f"""
    <div class="now-playing-card">
        <div class="np-emotion">{emoji}</div>
        <div class="np-label">Mood</div>
        <div class="np-emotion-name">{emotion}</div>
        <div class="np-song">♪ {display_name}</div>
        <div class="genre-tag">{genre_icon} {genre.title() if genre else "—"}</div>
    </div>
    """, unsafe_allow_html=True)

    if current_song:
        st.audio(current_song, autoplay=True)

    # Controls
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⏭  Next Song", use_container_width=True):
            st.session_state['current_song'] = random.choice(song_paths)
            st.rerun()
    with c2:
        if st.button("🔄  Start Over", use_container_width=True):
            for key in ['music_folder', 'files', 'detected_emotion',
                        'genre_to_play', 'song_paths', 'current_song']:
                st.session_state.pop(key, None)
            st.session_state['page'] = 'home'
            st.rerun()

    # Playlist
    st.divider()
    st.markdown(f"**Playlist** · {len(song_paths)} song(s)")
    st.write("")

    for i, path in enumerate(song_paths):
        name       = os.path.splitext(os.path.basename(path))[0]
        is_current = (path == current_song)
        active_cls = "active" if is_current else ""
        icon       = "▶" if is_current else "○"

        # Render a visual row, then an invisible button to handle clicks
        st.markdown(f"""
        <div class="playlist-row {active_cls}">
            <span style="color:{'#6366f1' if is_current else '#6b7280'}">{icon}</span>
            <span class="playlist-name {active_cls}">{name}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button(name, key=f"track_{i}", use_container_width=True,
                     type="primary" if is_current else "secondary"):
            st.session_state['current_song'] = path
            st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Stream Your Mood",
        page_icon="🎵",
        layout="centered",
    )
    _inject_css()

    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'

    pages = {
        'home':      page_home,
        'analyzing': page_analyzing,
        'player':    page_player,
    }
    pages[st.session_state['page']]()


if __name__ == '__main__':
    main()
