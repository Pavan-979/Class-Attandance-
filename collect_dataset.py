import cv2
import os
import json

name     = input("Enter your name: ")
roll_no  = input("Enter your roll number: ")
path     = f"dataset/{name}"

os.makedirs(path, exist_ok=True)

# ── Save roll number to roll_numbers.json ─────────────────────────────────────
ROLL_FILE = "roll_numbers.json"

if os.path.exists(ROLL_FILE):
    with open(ROLL_FILE, "r") as f:
        roll_data = json.load(f)
else:
    roll_data = {}

roll_data[name] = roll_no

with open(ROLL_FILE, "w") as f:
    json.dump(roll_data, f, indent=4)

print(f"Roll number {roll_no} saved for '{name}'.")

# ── Collect face images ───────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        cv2.imwrite(f"{path}/{count}.jpg", face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.putText(frame, f"Collected: {count}/50", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) == 27:
        break

    if count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Collected {count} images for '{name}'.")