# Credit Score Ranking

**Question:** Is this customer Poor, Average or Good?
**Answer:** Poor, Average or Good
**Model:** Logistic Regression

## Files

| File | What it is |
|---|---|
| `credit.csv` | the dataset |
| `Credit_Score_12_Steps.ipynb` | the notebook — 12 steps |
| `credit_app.py` | the web app |
| `credit_model.pkl` | the trained model (created by Step 12) |
| `requirements.txt` | library list (created by Step 12.3) |

## How to use

1. Open the notebook and run every cell from top to bottom
2. Step 12 creates `credit_model.pkl` and `requirements.txt`
3. Run the app:

```
streamlit run credit_app.py
```

## To host it on the internet

Upload `credit_app.py`, `credit_model.pkl` and `requirements.txt` to a **public** GitHub repository,
then deploy on share.streamlit.io with Main file path = `credit_app.py`.
