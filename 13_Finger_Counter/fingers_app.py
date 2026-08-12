"""
=============================================================
  FINGER COUNTER - CAMERA WEB APP
=============================================================

  BEFORE RUNNING:
    1. Run Finger_Counter_12_Steps.ipynb
    2. Make sure fingers_model.pkl is in this folder

  RUN:   streamlit run fingers_app.py

  Your browser will ask permission to use the camera. Say yes.

  IMPORTANT: if you trained on the drawn sample data, this will not
  recognise your real hand. Record your own data with
  record_fingers.py and re-run the notebook first.
=============================================================
"""

# =============================================================
# STEP 2 - IMPORT LIBRARIES
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
import os


# =============================================================
# STEP 3 - LOAD THE SAVED MODEL
# =============================================================
if os.path.exists("fingers_model.pkl") == False:
    st.error("fingers_model.pkl not found in this folder.")
    st.info("Run Finger_Counter_12_Steps.ipynb first (Step 12 creates this file).")
    st.stop()

package = joblib.load("fingers_model.pkl")

model = package["model"]
columns = package["columns"]
SIZE = package["size"]
labels = package["labels"]


# =============================================================
# STEP 4 - PAGE TITLE
# =============================================================
st.set_page_config(page_title="Finger Counter", page_icon="✋")

st.title("✋ Finger Counter")
st.write("Hold up some fingers inside the box, take a photo, and the model will count them.")

col1, col2 = st.columns(2)
col1.metric("Accuracy on test pictures", str(round(package["accuracy"] * 100, 1)) + " %")
col2.metric("Labels it knows", len(labels))

st.caption("It can answer: " + ", ".join(str(l) for l in labels))

st.divider()


# =============================================================
# STEP 5 - TAKE INPUT FROM THE USER (the camera!)
# =============================================================
# st.camera_input opens the laptop camera and takes ONE photo.
# It works on a laptop and also when the app is hosted online.
st.subheader("Take a photo")
st.caption("Hold your hand in the middle of the picture, against a plain wall.")

photo = st.camera_input("Camera")

if photo is None:
    st.info("Waiting for a photo. Click the button above.")
    st.stop()


# =============================================================
# STEP 6 - TURN THE PHOTO INTO A ROW OF NUMBERS
# =============================================================
# This must match the notebook EXACTLY:
# grey -> cut out the hand -> resize to SIZE x SIZE -> flatten -> divide by 255

file_bytes = np.asarray(bytearray(photo.getvalue()), dtype=np.uint8)
picture = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)

# There is no hand detector built into OpenCV, so we use the same fixed
# box that record_fingers.py used. The hand must be inside the box.
height, width = gray.shape
box_size = min(height, width) - 40
start_x = (width - box_size) // 2
start_y = (height - box_size) // 2

cut_out = gray[start_y:start_y + box_size, start_x:start_x + box_size]

boxed = picture.copy()
cv2.rectangle(boxed, (start_x, start_y),
              (start_x + box_size, start_y + box_size), (255, 120, 0), 3)

small = cv2.resize(cut_out, (SIZE, SIZE))


# =============================================================
# STEP 7 - SCALE
# =============================================================
row = small.flatten() / 255.0
new_row = pd.DataFrame([row], columns=columns)


# =============================================================
# STEP 8 - PREDICT
# =============================================================
prediction = model.predict(new_row)[0]
probability = model.predict_proba(new_row)[0]
chance = probability.max()


# =============================================================
# STEP 9 - SHOW THE RESULT
# =============================================================
st.divider()
st.subheader("The model says")

st.markdown("<h1 style='color:#1f9d55'>" + str(prediction) + "</h1>", unsafe_allow_html=True)

st.write("How sure it is:", str(round(chance * 100, 1)) + " %")
st.progress(float(chance))

if chance < 0.6:
    st.warning("The model is not confident. More light, or more recorded training "
               "pictures, usually fixes this.")


# =============================================================
# STEP 10 - SHOW WHAT THE MODEL RECEIVED
# =============================================================
with st.expander("See exactly what the model received"):

    left, right = st.columns(2)

    with left:
        st.write("**What it cut out of your photo:**")
        st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB), width=260)

    with right:
        st.write("**Shrunk to " + str(SIZE) + " x " + str(SIZE) + ":**")
        st.image(small, width=180, clamp=True)

    st.write("**All the chances:**")
    st.dataframe(pd.DataFrame({"label": model.classes_,
                               "chance_%": (probability * 100).round(2)}),
                 hide_index=True)

    st.caption("That small grey square is the entire input. "
               + str(SIZE * SIZE) + " numbers, nothing else.")

st.caption("3-Day AI Training | Computer Vision")
