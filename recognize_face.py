import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def eye_ratio(landmarks, eye_points, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]

    v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    h1 = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    return (v1 + v2) / (2.0 * h1)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

blink_count = 0
blink_detected = False
counter = 0
EAR_THRESHOLD = 0.25

CONFIDENCE_THRESHOLD = 60  # FIX: Added confidence threshold to avoid labeling unknown faces

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("trained_model.xml")

# Load label mapping
labels = np.load("labels.npy", allow_pickle=True).item()

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = model.predict(face)

        # FIX: Only label face if confidence is within threshold
        if confidence < CONFIDENCE_THRESHOLD:
            name = labels[label]
        else:
            name = "Unknown"

        # Display name and bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # FIX: Moved blink detection INSIDE the while loop (was outside before — never ran during live feed)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    h_frame, w_frame, _ = frame.shape

    # FIX: results is now always defined before this check (was causing potential NameError before)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            leftEAR = eye_ratio(face_landmarks.landmark, LEFT_EYE, w_frame, h_frame)
            rightEAR = eye_ratio(face_landmarks.landmark, RIGHT_EYE, w_frame, h_frame)

            ear = (leftEAR + rightEAR) / 2.0

            if ear < EAR_THRESHOLD:
                counter += 1
            else:
                if counter >= 2:
                    blink_count += 1
                    blink_detected = True
                    print("Blink detected")
                counter = 0

    # Display blink count on frame
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()