# Shelf Placement Predictor

**Question:** If we place this item here, will it sell fast?
**Answer:** Fast, Slow or Ignored
**Model:** Logistic Regression

## Files

| File | What it is |
|---|---|
| `shelf_placement.csv` | the dataset |
| `Shelf_Placement_12_Steps.ipynb` | the notebook — 12 steps |
| `shelf_app.py` | the web app |
| `shelf_model.pkl` | the trained model (created by Step 12) |
| `requirements.txt` | library list (created by Step 12.3) |

## How to use

1. Open the notebook and run every cell from top to bottom
2. Step 12 creates `shelf_model.pkl` and `requirements.txt`
3. Run the app:

```
streamlit run shelf_app.py
```

## To host it on the internet

Upload `shelf_app.py`, `shelf_model.pkl` and `requirements.txt` to a **public** GitHub repository,
then deploy on share.streamlit.io with Main file path = `shelf_app.py`.
