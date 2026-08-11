"""
=============================================================
  FESTIVAL SALE PLANNER - WEB APP
=============================================================

This file turns the model we trained in the notebook into a
real website that anybody can open in a browser.

BEFORE RUNNING THIS FILE:
  1. Run the notebook Festival_Sale_Planner_12_Steps.ipynb first
  2. Make sure festival_model.pkl is in this same folder

TO RUN THIS APP:
  Open Anaconda Prompt / Terminal in this folder and type:

      streamlit run festival_app.py

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
import numpy as np
import joblib
import os


# =============================================================
# STEP 3 - LOAD THE SAVED MODEL
# =============================================================
# In Step 12 of the notebook we saved three things inside
# festival_model.pkl:
#
#     model    -> makes the prediction
#     scaler   -> scales the new values the same way as training
#     columns  -> the column names in the correct order

# A small safety check: if the .pkl file is missing, show a clear
# message instead of a long red error.
if os.path.exists("festival_model.pkl") == False:
    st.error("festival_model.pkl not found in this folder.")
    st.info("Run the notebook Festival_Sale_Planner_12_Steps.ipynb first (Step 12 creates this file).")
    st.stop()

package = joblib.load("festival_model.pkl")

model = package["model"]
scaler = package["scaler"]
columns = package["columns"]


# =============================================================
# STEP 4 - PAGE TITLE AND HEADING
# =============================================================
# set_page_config must always be the FIRST streamlit command.

st.set_page_config(page_title="Festival Sale Planner", page_icon="🪔")

st.title("🪔 Should we put this item on sale for this festival?")

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

item_code_name = st.selectbox("Item", ['Clothing', 'Electronics', 'Sweets & Mithai', 'Home Decor', 'Footwear', 'Groceries', 'Jewellery', 'Toys'])
item_code = {'Clothing': 1, 'Electronics': 2, 'Sweets & Mithai': 3, 'Home Decor': 4, 'Footwear': 5, 'Groceries': 6, 'Jewellery': 7, 'Toys': 8}[item_code_name]

festival_code_name = st.selectbox("Festival / Occasion", ['Dashain', 'Tihar', 'Christmas', 'Eid', 'Normal Day'])
festival_code = {'Dashain': 1, 'Tihar': 2, 'Christmas': 3, 'Eid': 4, 'Normal Day': 5}[festival_code_name]

city_code_name = st.selectbox("Store city", ['Kathmandu', 'Pokhara'])
city_code = {'Kathmandu': 1, 'Pokhara': 2}[city_code_name]

price = st.slider("Price (NPR)", 150, 20000, 1500)
discount_percent = st.slider("Discount we will give (%)", 0, 60, 15)
stock_available = st.slider("Stock available (pieces)", 20, 600, 250)
advertisement = st.slider("Will we advertise it? (0=No, 1=Yes)", 0, 1, 1)
last_year_units_sold = st.slider("Units sold in the same festival last year", 10, 500, 200)


# =============================================================
# STEP 6 - PUT THE INPUT INTO A DATAFRAME
# =============================================================
# The model was trained on a TABLE, so we must give it a table.
# One row = one case.
#
# We use columns from the .pkl file, so the order is always
# exactly the same as during training.

new_row = pd.DataFrame([[item_code, festival_code, city_code, price, discount_percent, stock_available, advertisement, last_year_units_sold]], columns=columns)

# The notebook changed this column with a log transform in Step 4.3,
# so we MUST do exactly the same here. If we forget this, the model
# receives a number 10 times too big and the answer will be wrong.
new_row["price"] = np.log(new_row["price"] + 1)


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

if prediction == "High":
    st.success("🔥  HIGH SALE EXPECTED - put this item on the front display")
elif prediction == "Medium":
    st.warning("🟡  MEDIUM SALE - keep normal stock, no big promotion needed")
else:
    st.error("❄️  LOW SALE - do not waste the festival budget on this item")

# There are three possible answers, so we show all three chances
chances = pd.DataFrame({"answer": model.classes_,
                        "chance_%": (probability * 100).round(2)})
st.dataframe(chances)


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
#         festival_app.py
#         festival_model.pkl
#         requirements.txt
#  4. Go to share.streamlit.io and sign in with GitHub
#  5. Click "New app", choose your repository
#  6. Set Main file path to:  festival_app.py
#  7. Click Deploy and wait 2-3 minutes
#
#  You will get a link like https://your-app-name.streamlit.app
#  Anyone in the world can open it, even on a mobile phone.
# =============================================================

st.divider()
st.caption("Built in the 3-Day AI Training  |  Data -> Model -> .pkl -> Web App")
