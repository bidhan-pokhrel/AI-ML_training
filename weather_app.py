"""
=============================================================
  WEATHER PREDICTION - WEB APP
=============================================================

This file turns the model we trained in the notebook into a
real website that anybody can open in a browser.

BEFORE RUNNING THIS FILE:
  1. Run the notebook Weather_Prediction_12_Steps.ipynb first
  2. Make sure weather_model.pkl is in this same folder

TO RUN THIS APP:
  Open Anaconda Prompt / Terminal in this folder and type:

      streamlit run weather_app.py

  The browser will open at http://localhost:8501
  To stop the app, press Ctrl + C in the terminal.

IMPORTANT IDEA:
  No training happens in this file. Training was done once in
  the notebook. Here we only LOAD the trained model and USE it.
  That is why the app answers instantly.
=============================================================
"""

# =============================================================
# STEP 1 - INSTALL STREAMLIT
# =============================================================
# Run this ONCE in Anaconda Prompt / Terminal (not inside this file):
#
#     pip install streamlit


# =============================================================
# STEP 2 - IMPORT LIBRARIES
# =============================================================
# streamlit  -> builds the website
# pandas     -> to make the input table for the model
# joblib     -> to open the saved .pkl file

import streamlit as st
import pandas as pd
import joblib
import os


# =============================================================
# STEP 3 - LOAD THE SAVED MODEL
# =============================================================
# In Step 12 of the notebook we saved three things inside
# weather_model.pkl:
#
#     model    -> makes the prediction
#     scaler   -> scales the new values the same way as training
#     columns  -> the column names in the correct order

# A small safety check: if the .pkl file is missing, show a clear
# message instead of a long red error.
if os.path.exists("weather_model.pkl") == False:
    st.error("weather_model.pkl not found in this folder.")
    st.info("Run the notebook Weather_Prediction_12_Steps.ipynb first (Step 12 creates this file).")
    st.stop()

package = joblib.load("weather_model.pkl")

model = package["model"]
scaler = package["scaler"]
columns = package["columns"]


# =============================================================
# STEP 4 - PAGE TITLE AND HEADING
# =============================================================
# set_page_config must always be the FIRST streamlit command.

st.set_page_config(page_title="Weather Prediction", page_icon="🌦️")

st.title("🌦️ Will it rain tomorrow?")

st.write("Fill in the details below and the AI model will give the answer.")

st.divider()


# =============================================================
# STEP 5 - TAKE INPUT FROM THE USER
# =============================================================
# st.slider draws a slider on the website.
# The 4 values inside are:  label, minimum, maximum, starting value
# Whatever the user chooses is stored in our variable.
#
# These are the SAME columns we used to train the model.

st.subheader("Enter the details")

temperature = st.slider("Temperature (°C)", 0.0, 50.0, 26.0)
humidity = st.slider("Humidity (%)", 0.0, 100.0, 92.0)
wind_speed = st.slider("Wind speed (km/h)", 0.0, 100.0, 15.0)
pressure = st.slider("Air pressure (hPa)", 980.0, 1045.0, 1004.0)
cloud_cover = st.slider("Cloud cover (%)", 0.0, 100.0, 88.0)


# =============================================================
# STEP 6 - PUT THE INPUT INTO A DATAFRAME
# =============================================================
# The model was trained on a TABLE, so we must give it a table.
# One row = one case.
#
# We use columns from the .pkl file, so the order is always
# exactly the same as during training.

new_row = pd.DataFrame([[temperature, humidity, wind_speed, pressure, cloud_cover]], columns=columns)


# =============================================================
# STEP 7 - SCALE THE INPUT
# =============================================================
# We use transform, NOT fit_transform.
# The scaler already learned in the notebook. fit_transform here
# would make it learn again from one single row and the
# prediction would be wrong.

new_row_scaled = scaler.transform(new_row)
new_row_scaled = pd.DataFrame(new_row_scaled, columns=columns)


# =============================================================
# STEP 8 - PREDICT
# =============================================================
# predict()       -> gives the answer
# predict_proba() -> gives the chance behind that answer

prediction = model.predict(new_row_scaled)[0]

probability = model.predict_proba(new_row_scaled)[0]


# =============================================================
# STEP 9 - SHOW THE RESULT
# =============================================================
# The if / else below only decides which colour and message to
# show on the screen. It does not change the prediction.

st.divider()
st.subheader("The model says")

if prediction == "Yes":
    st.error("🌧️  RAIN EXPECTED - carry an umbrella")
else:
    st.success("☀️  NO RAIN EXPECTED - it should be a dry day")

# Find the chance of "Yes" and show it as a bar
classes = list(model.classes_)
position = classes.index("Yes")
chance = probability[position]

st.metric("Chance of rain", str(round(chance * 100, 2)) + " %")
st.progress(float(chance))


# =============================================================
# STEP 10 - SHOW WHAT WE SENT TO THE MODEL
# =============================================================
# This proves that the values on the sliders are exactly what
# reaches the model.

with st.expander("See what the model received"):

    st.write("**The values you entered:**")
    st.dataframe(new_row)

    st.write("**The same values after scaling:**")
    st.dataframe(new_row_scaled.round(2))

    st.write("**All possible answers and their chances:**")
    st.dataframe(pd.DataFrame({"answer": model.classes_,
                               "chance_%": (probability * 100).round(2)}))


# =============================================================
# HOW TO PUT THIS APP ON THE INTERNET (FREE)
# =============================================================
#  1. Make a free account on github.com
#  2. Create a NEW PUBLIC repository
#  3. Upload these 3 files into it:
#         weather_app.py
#         weather_model.pkl
#         requirements.txt
#  4. Go to share.streamlit.io and sign in with GitHub
#  5. Click "New app", choose your repository
#  6. Set Main file path to:  weather_app.py
#  7. Click Deploy and wait 2-3 minutes
#
#  You will get a link like https://your-app-name.streamlit.app
#  Anyone in the world can open it, even on a mobile phone.
# =============================================================

st.divider()
st.caption("Built in the 3-Day AI Training  |  Data -> Model -> .pkl -> Web App")
