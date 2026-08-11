# Festival Sale Planner

**Question:** Should we put this item on sale for this festival?
**Answer:** High, Medium or Low
**Model:** Decision Tree

## Files

| File | What it is |
|---|---|
| `festival_sales.csv` | the dataset |
| `Festival_Sale_Planner_12_Steps.ipynb` | the notebook — 12 steps |
| `festival_app.py` | the web app |
| `festival_model.pkl` | the trained model (created by Step 12) |
| `requirements.txt` | library list (created by Step 12.3) |

## How to use

1. Open the notebook and run every cell from top to bottom
2. Step 12 creates `festival_model.pkl` and `requirements.txt`
3. Run the app:

```
streamlit run festival_app.py
```

## To host it on the internet

Upload `festival_app.py`, `festival_model.pkl` and `requirements.txt` to a **public** GitHub repository,
then deploy on share.streamlit.io with Main file path = `festival_app.py`.
