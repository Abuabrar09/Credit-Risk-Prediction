# 🏦 Credit Risk Prediction — Machine Learning for Loan Decisions

[![View on Kaggle](https://img.shields.io/badge/Kaggle-View%20Full%20Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/abrarahmedsyed313/credit-risk-prediction)

**Author:** Abrar Ahmed Syed — Business & Financial Data Analyst | AI

## 🎯 Problem
Can we predict which loan applicants will default, and choose an approval rule that **minimises the bank's losses**?

## 📊 Data
- **Source:** German Credit dataset (UCI Machine Learning Repository / OpenML)
- **Size:** 1,000 real loan applicants, 20 features
- **Features:** checking account status, credit history, loan amount and duration, savings, employment, housing and more
- **Target:** whether the applicant defaulted (30% default rate)

## 🛠️ Approach
1. **Exploratory analysis:** default rates by account balance, credit history, savings, loan amount and duration
2. **Responsible AI:** removed gender/marital status and foreign-worker status so the model doesn't base lending decisions on protected characteristics
3. **Models:** Logistic Regression and Random Forest, with class balancing for the minority default class
4. **Bayesian Optimization (Optuna):** tuned Random Forest hyperparameters over 40 trials with 5-fold cross-validation
5. **Business-cost threshold tuning:** approving a defaulter is treated as 5x more costly than rejecting a good customer, and the approval threshold is chosen to minimise total cost
6. **Explainability:** feature importance to identify the main drivers of default risk

## 📈 Results
| Metric | Result |
|---|---|
| Best model | Random Forest |
| Test ROC-AUC | **0.814** |
| Cost-optimal threshold | **0.18** (vs default 0.50) |
| Defaults caught | **57 of 60 (95%)** |
| Reduction in lending losses | **48%** (cost 187 → 98) |

## 💡 Key Insights
1. **Checking account is a strong warning sign:** applicants with a negative balance default **49%** of the time, vs **12%** for those without a checking account
2. **Savings protect:** applicants with savings ≥1,000 default far less than those with under 100
3. **Top risk drivers:** loan amount, loan duration and checking-account status
4. **Threshold beats tuning:** moving the threshold from 0.50 to 0.18 cut losses by **48%**, while Bayesian Optimization didn't improve test AUC (0.811 vs 0.814), which is expected with a small dataset
5. **Data quality check:** "no credits / all paid" showed an unexpectedly high default rate, a known quirk of this dataset worth questioning before acting on it

## 📝 Conclusion
In credit risk, **aligning the model with real business costs matters more than squeezing out extra accuracy**. A well-chosen approval threshold turned a good model into one that catches 95% of defaulters and nearly halves lending losses.

## 🧰 Tools
Python · pandas · scikit-learn · Optuna · matplotlib · seaborn · Kaggle

## ▶️ How to Run
Click **View on Kaggle** above → **Copy & Edit** → **Run All** (internet must be turned on in notebook settings).

---
*Educational project using a public dataset. Not financial advice.*
