"""
=============================================================
  RECORD YOUR OWN FACE DATA
=============================================================
  This is Step 2 of the project: creating the dataset.

  HOW TO USE - run it once for each label:

      python record_faces.py Awake
      python record_faces.py Sleeping
      python record_faces.py Talking
      python record_faces.py Yawning

  A window opens showing your webcam with a green box on your face.
  Make that face and press SPACE to save a picture.
  Hold SPACE to save many quickly. Press Q when you are done.

  Aim for about 100 pictures per label. Move your head a little,
  turn slightly, change the light - the more variety, the better
  the model works.

  Everything is added to faces.csv in this folder.
=============================================================
"""
import sys
import os
import cv2
import numpy as np
import pandas as pd

SIZE = 24
LABELS = ["Awake", "Sleeping", "Talking", "Yawning"]

if len(sys.argv) < 2 or sys.argv[1] not in LABELS:
    print("Please give one label. Example:  python record_faces.py Awake")
    print("Labels:", LABELS)
    sys.exit()

label = sys.argv[1]

# the face detector ships inside opencv - nothing is downloaded
face_finder = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    "haarcascade_frontalface_default.xml")

camera = cv2.VideoCapture(0)
rows = []

print("Recording label:", label)
print("SPACE = save a picture   Q = quit")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Camera not working. Close other apps using the camera and try again.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_finder.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

    for (x, y, w, h) in faces[:1]:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, label + "   saved: " + str(len(rows)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Recording - SPACE to save, Q to quit", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" ") and len(faces) > 0:
        x, y, w, h = faces[0]
        face = gray[y:y + h, x:x + w]
        small = cv2.resize(face, (SIZE, SIZE))
        rows.append(small.flatten())

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

if len(rows) > 0:
    new_data = pd.DataFrame(rows, columns=["p" + str(i) for i in range(SIZE * SIZE)])
    new_data["label"] = label

    if os.path.exists("faces.csv"):
        old = pd.read_csv("faces.csv")
        new_data = pd.concat([old, new_data], ignore_index=True)

    new_data.to_csv("faces.csv", index=False)
    print("Saved", len(rows), "pictures. faces.csv now has", len(new_data), "rows.")
    print(new_data["label"].value_counts())
else:
    print("Nothing saved.")
