# Student Exam Result Prediction

**Question:** Will this student pass the exam?
**Answer:** Pass or Fail
**Model:** Logistic Regression

## Files

| File | What it is |
|---|---|
| `exam.csv` | the dataset |
| `Exam_Result_12_Steps.ipynb` | the notebook — 12 steps |
| `exam_app.py` | the web app |
| `exam_model.pkl` | the trained model (created by Step 12) |
| `requirements.txt` | library list (created by Step 12.3) |

## How to use

1. Open the notebook and run every cell from top to bottom
2. Step 12 creates `exam_model.pkl` and `requirements.txt`
3. Run the app:

```
streamlit run exam_app.py
```

## To host it on the internet

Upload `exam_app.py`, `exam_model.pkl` and `requirements.txt` to a **public** GitHub repository,
then deploy on share.streamlit.io with Main file path = `exam_app.py`.
