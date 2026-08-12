"""
=============================================================
  RECORD YOUR OWN HAND DATA
=============================================================
  This is Step 2 of the project: creating the dataset.

  HOW TO USE - run it once for each number of fingers:

      python record_fingers.py 0
      python record_fingers.py 1
      ... up to ...
      python record_fingers.py 5

  A window opens with a BLUE BOX. Put your hand inside the box,
  show that many fingers, and press SPACE to save.

  TIPS for a model that actually works:
   - use a plain wall or a plain sheet of paper behind your hand
   - keep the light steady
   - move your hand a little between shots
   - about 100 pictures per number

  Everything is added to fingers.csv in this folder.
=============================================================
"""
import sys
import os
import cv2
import numpy as np
import pandas as pd

SIZE = 24
LABELS = ["0", "1", "2", "3", "4", "5"]

if len(sys.argv) < 2 or sys.argv[1] not in LABELS:
    print("Please give the number of fingers. Example:  python record_fingers.py 3")
    sys.exit()

label = sys.argv[1]

camera = cv2.VideoCapture(0)
rows = []

# the fixed box where the hand must go
BOX_X, BOX_Y, BOX_SIZE = 340, 100, 260

print("Recording label:", label, "fingers")
print("SPACE = save a picture   Q = quit")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Camera not working.")
        break

    cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_SIZE, BOX_Y + BOX_SIZE),
                  (255, 120, 0), 2)
    cv2.putText(frame, label + " fingers   saved: " + str(len(rows)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 120, 0), 2)
    cv2.imshow("Put your hand in the box - SPACE to save, Q to quit", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        box = frame[BOX_Y:BOX_Y + BOX_SIZE, BOX_X:BOX_X + BOX_SIZE]
        gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (SIZE, SIZE))
        rows.append(small.flatten())

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

if len(rows) > 0:
    new_data = pd.DataFrame(rows, columns=["p" + str(i) for i in range(SIZE * SIZE)])
    new_data["label"] = label

    if os.path.exists("fingers.csv"):
        old = pd.read_csv("fingers.csv")
        new_data = pd.concat([old, new_data], ignore_index=True)

    new_data.to_csv("fingers.csv", index=False)
    print("Saved", len(rows), "pictures. fingers.csv now has", len(new_data), "rows.")
    print(new_data["label"].value_counts())
else:
    print("Nothing saved.")
