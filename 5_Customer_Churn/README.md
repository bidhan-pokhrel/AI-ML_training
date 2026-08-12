# Customer Churn Prediction

**Question:** Will this customer leave us?
**Answer:** Yes or No
**Model:** Decision Tree

## Files

| File | What it is |
|---|---|
| `churn.csv` | the dataset |
| `Customer_Churn_12_Steps.ipynb` | the notebook — 12 steps |
| `churn_app.py` | the web app |
| `churn_model.pkl` | the trained model (created by Step 12) |
| `requirements.txt` | library list (created by Step 12.3) |

## How to use

1. Open the notebook and run every cell from top to bottom
2. Step 12 creates `churn_model.pkl` and `requirements.txt`
3. Run the app:

```
streamlit run churn_app.py
```

## To host it on the internet

Upload `churn_app.py`, `churn_model.pkl` and `requirements.txt` to a **public** GitHub repository,
then deploy on share.streamlit.io with Main file path = `churn_app.py`.
