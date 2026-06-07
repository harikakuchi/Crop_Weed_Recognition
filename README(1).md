# Marketing and Product Performance: Sales Revenue Prediction
### DSC01 Machine Learning 120 A — Submission 1

---

## Project Overview

This project applies supervised machine learning to predict **Sales Revenue** from a marketing and product performance dataset. It investigates 7 research questions spanning model accuracy, feature importance, advertising spend patterns, seasonal/regional effects, demographics, engagement metrics, and discount impact.

**Task Type:** Regression  
**Target Variable:** `Sales_Revenue`  
**Dataset:** [Marketing and Product Performance Dataset — Kaggle](https://www.kaggle.com/datasets/imranalishahh/marketing-and-product-performance-dataset)

---

## Repository Structure

```
├── RQ1_Model_Comparison.ipynb           # Which ML model predicts revenue best?
├── RQ2_Feature_Importance.ipynb         # Which features matter most?
├── RQ3_AdSpend_Revenue_Relationship.ipynb  # Ad spend vs revenue & diminishing returns
├── RQ4_Seasonal_Regional_Factors.ipynb  # Seasonal and regional prediction accuracy
├── RQ5_Demographics_Impact.ipynb        # Do Age/Gender improve predictions?
├── RQ6_CTR_CVR_AdSpend_Revenue.ipynb    # CTR/CVR/Ad Spend interaction with revenue
├── RQ7_Discount_Effect.ipynb            # Discount impact across categories/regions
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Research Questions

| # | Research Question |
|---|---|
| RQ1 | How accurately can ML models predict sales revenue, and which performs best? |
| RQ2 | Which features are the most important predictors of sales revenue? |
| RQ3 | How does ad spend relate to revenue across campaign types (diminishing returns)? |
| RQ4 | How do seasonal and regional factors influence prediction accuracy? |
| RQ5 | Can customer demographics (Age, Gender) improve revenue predictions? |
| RQ6 | How do CTR and CVR interact with ad spend to influence sales revenue? |
| RQ7 | How does discount application affect revenue across categories and regions? |

---

## Models Used

- Linear Regression (baseline)
- Ridge & Lasso Regression
- Decision Tree Regressor
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- Support Vector Regressor (SVR)

---

## Evaluation Metrics

- MAE, MSE, RMSE
- R² Score and Adjusted R²
- 5-Fold Cross-Validation R²

---

## How to Run (on Kaggle)

1. Go to [Kaggle](https://www.kaggle.com) and create a new Notebook.
2. Add the dataset: **Marketing and Product Performance Dataset** by `imranalishahh`.
3. Upload each `.ipynb` file one by one and run all cells.
4. Each notebook saves figures as `.pdf` and tables as `.csv` in the working directory.

**Dataset path on Kaggle:**
```
/kaggle/input/marketing-and-product-performance-dataset/marketing_data.xlsx
```

> If the filename differs, run `import os; os.listdir('/kaggle/input/marketing-and-product-performance-dataset/')` to find the correct name.

---

## Output Files Per Notebook

| Notebook | Figures (PDF) | Tables (CSV) |
|---|---|---|
| RQ1 | Figure_1_1, Figure_1_2 | Table_1_1 |
| RQ2 | Figure_2_1, Figure_2_2, Figure_2_3* | Table_2_1 |
| RQ3 | Figure_3_1, Figure_3_2, Figure_3_3 | Table_3_1 |
| RQ4 | Figure_4_1, Figure_4_2, Figure_4_3 | Table_4_1 |
| RQ5 | Figure_5_1, Figure_5_2 | Table_5_1 |
| RQ6 | Figure_6_1, Figure_6_2, Figure_6_3 | Table_6_1 |
| RQ7 | Figure_7_1, Figure_7_2, Figure_7_3 | Table_7_1, Table_7_2 |

*Figure_2_3 (SHAP) requires `shap` package.

---

## Instructor
**Prof. Raja Hashim Ali** — DSC01 Machine Learning 120 A
