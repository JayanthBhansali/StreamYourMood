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
    st.title("🎵 Stream Your Mood")
    st.caption("Detects your facial emotion and plays music that matches how you feel.")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload your music files (.mp3 or .wav)",
        type=["mp3", "wav"],
        accept_multiple_files=True,
    )

    if st.button("Submit", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("Please upload at least one audio file.")
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
                'files': files,
                'page': 'analyzing',
            })
            st.rerun()


def page_analyzing():
    st.title("🎵 Stream Your Mood")

    music_folder = st.session_state.get('music_folder', '')
    files        = st.session_state.get('files', [])

    # ── Step 1: Get image via browser camera ──
    st.subheader("📸 Capture your photo")
    img_file = st.camera_input("Take a photo for mood detection")
    if img_file is None:
        st.info("Allow camera access in your browser, then take a photo to continue.")
        return
    img_array = np.frombuffer(img_file.getvalue(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if save_images:
        createFolderIfnotExists('saved_images')
        cv2.imwrite(os.path.join('saved_images', str(uuid.uuid4()) + '.png'), frame)

    # ── Step 2: Detect emotion ──
    with st.spinner("Analyzing your mood..."):
        detected_emotion = None
        for _ in range(3):
            detected_emotion = facial.detect_emotion(image)
            if detected_emotion:
                break

    if not detected_emotion:
        st.error("No face detected after 3 attempts.")
        if st.button("Try Again"):
            st.session_state['page'] = 'home'
            st.rerun()
        return

    st.success(f"Emotion detected: **{detected_emotion}**")

    # ── Step 3: Analyze songs ──
    if files:
        with st.spinner(f"Classifying {len(files)} song(s) — this may take a moment..."):
            analyze_songs(music_folder, files)

    # ── Step 4: Match songs to emotion ──
    with st.spinner("Matching songs to your mood..."):
        song_paths, genre_to_play = get_songs_for_emotion(detected_emotion)

    if not song_paths:
        st.error("No songs found in the database.")
        if st.button("Go Back"):
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

    st.title("🎵 Stream Your Mood")

    col1, col2 = st.columns(2)
    col1.metric("Detected Emotion", emotion)
    col2.metric("Playing Genre",    genre.title() if genre else "—")

    st.divider()

    song_name = os.path.basename(current_song) if current_song else "—"
    st.subheader("▶  Now Playing")
    st.write(f"**{song_name}**")

    if current_song:
        st.audio(current_song, autoplay=True)

    # ── Controls ──
    c1, c2 = st.columns(2)

    with c1:
        if st.button("⏭  Next", use_container_width=True):
            st.session_state['current_song'] = random.choice(song_paths)
            st.rerun()

    with c2:
        if st.button("🔄  Restart", use_container_width=True):
            for key in ['music_folder', 'files', 'detected_emotion',
                        'genre_to_play', 'song_paths', 'current_song']:
                st.session_state.pop(key, None)
            st.session_state['page'] = 'home'
            st.rerun()

    # ── Playlist ──
    st.divider()
    st.subheader("Playlist")
    for i, path in enumerate(song_paths):
        name       = os.path.basename(path)
        is_current = (path == current_song)
        label      = f"▶  {name}" if is_current else f"　{name}"
        if st.button(label, key=f"track_{i}", use_container_width=True):
            st.session_state['current_song'] = path
            st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Stream Your Mood",
        page_icon="🎵",
        layout="centered",
    )

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
