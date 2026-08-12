"""
=============================================================
  SHELF PLACEMENT PREDICTOR - WEB APP
=============================================================

This file turns the model we trained in the notebook into a
real website that anybody can open in a browser.

BEFORE RUNNING THIS FILE:
  1. Run the notebook Shelf_Placement_12_Steps.ipynb first
  2. Make sure shelf_model.pkl is in this same folder

TO RUN THIS APP:
  Open Anaconda Prompt / Terminal in this folder and type:

      streamlit run shelf_app.py

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
# shelf_model.pkl:
#
#     model    -> makes the prediction
#     scaler   -> scales the new values the same way as training
#     columns  -> the column names in the correct order

# A small safety check: if the .pkl file is missing, show a clear
# message instead of a long red error.
if os.path.exists("shelf_model.pkl") == False:
    st.error("shelf_model.pkl not found in this folder.")
    st.info("Run the notebook Shelf_Placement_12_Steps.ipynb first (Step 12 creates this file).")
    st.stop()

package = joblib.load("shelf_model.pkl")

model = package["model"]
scaler = package["scaler"]
columns = package["columns"]


# =============================================================
# STEP 4 - PAGE TITLE AND HEADING
# =============================================================
# set_page_config must always be the FIRST streamlit command.

st.set_page_config(page_title="Shelf Placement Predictor", page_icon="🛒")

st.title("🛒 If we place this item here, will it sell fast?")

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

zone_code_name = st.selectbox("Where in the shop?", ['Entrance', 'Near Billing Counter', 'Front Aisle', 'Middle Aisle', 'Back Aisle', 'Corner Shelf'])
zone_code = {'Entrance': 1, 'Near Billing Counter': 2, 'Front Aisle': 3, 'Middle Aisle': 4, 'Back Aisle': 5, 'Corner Shelf': 6}[zone_code_name]

height_code_name = st.selectbox("How high on the rack?", ['Eye Level', 'Floor Level', 'Top Shelf'])
height_code = {'Eye Level': 1, 'Floor Level': 2, 'Top Shelf': 3}[height_code_name]

item_category_code_name = st.selectbox("Item category", ['Clothing', 'Electronics', 'Sweets & Mithai', 'Home Decor', 'Footwear', 'Groceries', 'Jewellery', 'Toys'])
item_category_code = {'Clothing': 1, 'Electronics': 2, 'Sweets & Mithai': 3, 'Home Decor': 4, 'Footwear': 5, 'Groceries': 6, 'Jewellery': 7, 'Toys': 8}[item_category_code_name]

price = st.slider("Price (NPR)", 100, 20000, 900)
discount_percent = st.slider("Discount (%)", 0, 50, 10)
daily_customer_traffic = st.slider("Customers passing this spot daily", 20, 600, 320)
nearby_promotion = st.slider("Promotion banner nearby? (0=No, 1=Yes)", 0, 1, 1)
city_code_name = st.selectbox("Store city", ['Kathmandu', 'Pokhara'])
city_code = {'Kathmandu': 1, 'Pokhara': 2}[city_code_name]



# =============================================================
# STEP 6 - PUT THE INPUT INTO A DATAFRAME
# =============================================================
# The model was trained on a TABLE, so we must give it a table.
# One row = one case.
#
# We use columns from the .pkl file, so the order is always
# exactly the same as during training.

new_row = pd.DataFrame([[zone_code, height_code, item_category_code, price, discount_percent, daily_customer_traffic, nearby_promotion, city_code]], columns=columns)

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

if prediction == "Fast":
    st.success("🚀  SELLS FAST - a very good spot for this item")
elif prediction == "Slow":
    st.warning("🐌  SELLS SLOWLY - it will move, but not quickly")
else:
    st.error("😔  IGNORED BY CUSTOMERS - move this item to a better spot")

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
#         shelf_app.py
#         shelf_model.pkl
#         requirements.txt
#  4. Go to share.streamlit.io and sign in with GitHub
#  5. Click "New app", choose your repository
#  6. Set Main file path to:  shelf_app.py
#  7. Click Deploy and wait 2-3 minutes
#
#  You will get a link like https://your-app-name.streamlit.app
#  Anyone in the world can open it, even on a mobile phone.
# =============================================================

st.divider()
st.caption("Built in the 3-Day AI Training  |  Data -> Model -> .pkl -> Web App")
