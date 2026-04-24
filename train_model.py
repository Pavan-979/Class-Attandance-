import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []
label_map = {}

current_label = 0

# Loop through each person folder
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
           continue

        img = cv2.resize(img, (200, 200))

        faces.append(img)
        labels.append(current_label)

    current_label += 1

# Convert to numpy arrays
faces = np.array(faces)
labels = np.array(labels)

# Train model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)

# Save model
model.save("trained_model.xml")

# Save label mapping
np.save("labels.npy", label_map)

print("Training completed successfully")