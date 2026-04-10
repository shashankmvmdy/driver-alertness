import cv2
import mediapipe as mp
from utils import eye_aspect_ratio, mouth_aspect_ratio, play_alarm, log_event
import config


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

cap = cv2.VideoCapture(0)

frame_counter = 0
yawn_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            left_eye = [points[i] for i in LEFT_EYE]
            right_eye = [points[i] for i in RIGHT_EYE]
            mouth = [points[i] for i in MOUTH]

            EAR = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            MAR = mouth_aspect_ratio(mouth)

            # 💤 Drowsiness Detection
            if EAR < config.EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= config.ALERT_FRAMES:
                    cv2.putText(frame, "DROWSY ALERT!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    play_alarm(config.ALARM_SOUND)
                    log_event("Drowsiness Detected", config.LOG_FILE)
            else:
                frame_counter = 0

            # 😮 Yawning Detection
            if MAR > config.MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= config.YAWN_FRAMES:
                    cv2.putText(frame, "YAWNING ALERT!", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    play_alarm(config.ALARM_SOUND)
                    log_event("Yawning Detected", config.LOG_FILE)
            else:
                yawn_counter = 0

    cv2.imshow("Driver Alert System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()