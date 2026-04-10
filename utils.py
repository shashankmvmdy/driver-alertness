from scipy.spatial import distance
import threading
from playsound import playsound
import pandas as pd
import os
from datetime import datetime

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[0], mouth[1])
    B = distance.euclidean(mouth[2], mouth[3])
    return A / B

def play_alarm(sound):
    threading.Thread(target=playsound, args=(sound,), daemon=True).start()

def log_event(event, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    data = {
        "timestamp": [datetime.now()],
        "event": [event]
    }

    df = pd.DataFrame(data)

    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)