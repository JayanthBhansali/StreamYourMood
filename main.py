import os
from globalSettings import *
import uuid
from db import *
import sys, cv2
from sys import exit
import traceback
import random
import time

try:
    from mutagen.mp3 import MP3
except:
    os.system("pip install mutagen")
    from mutagen.mp3 import MP3

try:
    import pygame
except:
    os.system('pip install pygame')
    import pygame

def createFolderIfnotExists(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o777)

print("\n<============= LOADING LIBRARIES =============>\n")

#with suppress_stdout():
from FacialEmotionRecognition import facial
from AudioClassification import audio

print("\n<============= LIBRARIES LOADED =============>\n")

def main_func():

    conn = create_connection(DBPath)
    c=conn.cursor()

    while True:
        music_folder_dir = input("Enter the full path of music directory : ")
        if os.path.isdir(music_folder_dir):
            break
        else:
            print("\nPlease enter a valid folder!\n")
            time.sleep(2)

    files = [f for f in os.listdir(music_folder_dir) if f.endswith('.mp3') or f.endswith('.wav')]
    if not files:
        raise NameError("\n The folder doesn't contain any audio files")

    print("\n<============= ANALYZING MOOD =============>\n")
    if use_webcam:
        video_capture = cv2.VideoCapture(0)
        frame = video_capture.read()[1]
        video_capture.release()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
    else:
        image = cv2.imread("happy.png",0)
        frame = image

    if save_images:
        createFolderIfnotExists('saved_images')
        name = os.path.join('saved_images', str(uuid.uuid4()) + '.png')
        cv2.imwrite(name, frame)
    
    detected_emotion = facial.detect_emotion(image)
    print("\nEmotion detected : ", detected_emotion)

    print("\n<============= ANALYZING MUSIC =============>\n")

    existing_paths = []
    existing_songs = []
    query="select path,song from songs"
    c.execute(query)
    for col in c:
        existing_paths.append(col[0])
        existing_songs.append(col[1])

    for file in files:
        song = os.path.join(music_folder_dir,file)

        if song not in existing_paths and file not in existing_songs:
            sys.stdout.write("Analyzing song ... %s%%\r" % (file))
            sys.stdout.flush()

            results = audio.classify_audio(song)
            i = 0
            print("results : ", results)
            for genre, val in results:
                if val > 0.5 or i==0:
                    c.execute("insert into songs(path, genre, prediction, song) values(?,?,?, ?)",[song, genre, val, file])
                    conn.commit()
                else:
                    break
                i+=1
    
    if detected_emotion in emotion_genre_mappings.keys():
        genre_to_play = emotion_genre_mappings[detected_emotion]
        genre_to_play = random.choice(genre_to_play)
    else:
        raise NameError("unknown genre")
    

    query = "SELECT path, song FROM songs WHERE genre = ?"
    c.execute(query, (genre_to_play,))
    song_paths = []
    i = 0
    for col in c:
        path = col[0]
        song = col[1]
        if os.path.isfile(path):
            song_paths.append(path)
            i=1
            

    if i==0:
        print("Couldn't find any appropriate songs. Playing a random song : ")
        query = "select path, song from songs order by prediction desc"
        c.execute(query)
        for col in c:
            path = col[0]
            song = col[1]
            if os.path.isfile(path):
                i=1
                song_paths.append(path)

    conn.close()

    if i==0 and not song_paths:
        print("No songs in the DB.")
    else:
        path = random.choice(song_paths)
        audio_file = MP3(path)
        sleep_time = int(audio_file.info.length)
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        print("PLAYING {}. PLEASE ENJOY!".format(os.path.basename(path)))
        pygame.mixer.music.play()
        #pygame.event.wait()
        time.sleep(sleep_time)
        pygame.mixer.music.stop()
        pygame.display.quit()
        pygame.quit()
        choice= input("Press x to exit, anything else to play another song : ")
        if str(choice).lower() == 'x':
            exit()
        main_func()

if __name__ == '__main__':
    try:
        main_func()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        exit()