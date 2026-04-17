import streamlit as st
import os
import cv2
import uuid
import random
import numpy as np
import pygame
from mutagen.mp3 import MP3

from db import create_connection
from globalSettings import DBPath, use_webcam, save_images, emotion_genre_mappings
from FacialEmotionRecognition import facial
from AudioClassification import audio as audio_module


# ── Helpers ───────────────────────────────────────────────────────────────────

def createFolderIfnotExists(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o777)


def init_pygame():
    if not st.session_state.get('pygame_initialized'):
        pygame.mixer.init()
        st.session_state['pygame_initialized'] = True


def save_folder_path(music_folder_dir):
    conn = create_connection(DBPath)
    c = conn.cursor()
    c.execute("SELECT path FROM folder_paths")
    existing = [row[0] for row in c]
    if os.path.normpath(music_folder_dir) not in [os.path.normpath(p) for p in existing]:
        c.execute("INSERT INTO folder_paths(path) VALUES(?)", [music_folder_dir])
        conn.commit()
    conn.close()


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

    conn = create_connection(DBPath)
    c = conn.cursor()
    c.execute("SELECT path FROM folder_paths")
    existing_paths = [row[0] for row in c]
    conn.close()

    music_folder = st.text_input("Enter the full path of your music directory")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit", type="primary", use_container_width=True):
            if not music_folder:
                st.error("Please enter a folder path.")
            elif not os.path.isdir(music_folder):
                st.error("Invalid folder path.")
            else:
                files = [f for f in os.listdir(music_folder)
                         if f.endswith('.mp3') or f.endswith('.wav')]
                if not files:
                    st.error("Folder contains no .mp3 or .wav files.")
                else:
                    st.session_state.update({
                        'music_folder': music_folder,
                        'files': files,
                        'use_existing': False,
                        'page': 'analyzing',
                    })
                    st.rerun()

    with col2:
        if existing_paths and st.button("Use Existing", use_container_width=True):
            st.session_state.update({
                'music_folder': '',
                'files': [],
                'use_existing': True,
                'page': 'analyzing',
            })
            st.rerun()

    if existing_paths:
        st.subheader("Saved Paths")
        for i, path in enumerate(existing_paths, 1):
            st.text(f"{i}  →  {path}")


def page_analyzing():
    st.title("🎵 Stream Your Mood")

    music_folder = st.session_state.get('music_folder', '')
    files        = st.session_state.get('files', [])
    use_existing = st.session_state.get('use_existing', False)

    # ── Step 1: Get image ──
    if use_webcam:
        st.subheader("Capture your photo")
        img_file = st.camera_input("Take a photo for mood detection")
        if img_file is None:
            st.info("Take a photo to continue.")
            return
        img_array = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imread("happy.png", 0)
        frame = image

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
    if not use_existing and files:
        with st.spinner(f"Classifying {len(files)} song(s) — this may take a moment..."):
            analyze_songs(music_folder, files)
            save_folder_path(music_folder)

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
        'song_loaded':      False,
        'is_playing':       False,
        'is_paused':        False,
        'page':             'player',
    })
    st.rerun()


def page_player():
    init_pygame()

    song_paths   = st.session_state.get('song_paths', [])
    current_song = st.session_state.get('current_song', '')
    emotion      = st.session_state.get('detected_emotion', '')
    genre        = st.session_state.get('genre_to_play', '')
    is_playing   = st.session_state.get('is_playing', False)
    is_paused    = st.session_state.get('is_paused', False)

    # Auto-play on first arrival at player
    if not st.session_state.get('song_loaded') and current_song:
        pygame.mixer.music.load(current_song)
        pygame.mixer.music.play()
        st.session_state['song_loaded'] = True
        st.session_state['is_playing']  = True
        is_playing = True

    # Detect if song finished naturally
    if is_playing and not is_paused and not pygame.mixer.music.get_busy():
        st.session_state['is_playing'] = False
        is_playing = False

    st.title("🎵 Stream Your Mood")

    col1, col2 = st.columns(2)
    col1.metric("Detected Emotion", emotion)
    col2.metric("Playing Genre",    genre.title() if genre else "—")

    st.divider()

    song_name = os.path.basename(current_song) if current_song else "—"
    if is_playing and not is_paused:
        status_label = "▶  Playing"
    elif is_paused:
        status_label = "⏸  Paused"
    else:
        status_label = "⏹  Stopped"

    st.subheader(status_label)
    st.write(f"**{song_name}**")

    # ── Controls ──
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        pause_label = "⏸  Pause" if (is_playing and not is_paused) else "▶  Resume"
        if st.button(pause_label, use_container_width=True):
            if is_paused:
                pygame.mixer.music.unpause()
                st.session_state['is_paused'] = False
            else:
                pygame.mixer.music.pause()
                st.session_state['is_paused'] = True
            st.rerun()

    with c2:
        if st.button("⏭  Next", use_container_width=True):
            next_song = random.choice(song_paths)
            pygame.mixer.music.load(next_song)
            pygame.mixer.music.play()
            st.session_state.update({
                'current_song': next_song,
                'is_playing':   True,
                'is_paused':    False,
            })
            st.rerun()

    with c3:
        if st.button("⏹  Stop", use_container_width=True):
            pygame.mixer.music.stop()
            st.session_state.update({'is_playing': False, 'is_paused': False})
            st.rerun()

    with c4:
        if st.button("🔄  Restart", use_container_width=True):
            pygame.mixer.music.stop()
            for key in ['music_folder', 'files', 'use_existing', 'detected_emotion',
                        'genre_to_play', 'song_paths', 'current_song',
                        'song_loaded', 'is_playing', 'is_paused']:
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
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            st.session_state.update({
                'current_song': path,
                'is_playing':   True,
                'is_paused':    False,
            })
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
