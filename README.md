# Customer Churn Prediction using Machine Learning 📊

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)

## Project Overview
Predict whether a telecom customer will churn (cancel service) using supervised learning. The notebook implements data cleaning, feature engineering, model training, evaluation and a short insights report to support retention decisions.

## Dataset
- Telco Customer Churn (Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
## Dataset Summary (variable index)

A compact table summarizing the Telco Customer Churn dataset variables. Use this as a quick reference / variable index.

| idx | Variable | Type | Description | Example / Notes |
|-----:|----------|------|-------------|-----------------|
| 1 | customerID | string | Unique customer identifier | 7590-VHVEG |
| 2 | gender | categorical | Customer gender | Male / Female |
| 3 | SeniorCitizen | int (0/1) | Whether customer is senior citizen | 0 / 1 |
| 4 | Partner | categorical | Has partner? | Yes / No |
| 5 | Dependents | categorical | Has dependents? | Yes / No |
| 6 | tenure | numeric | Months with the company | 1..72 |
| 7 | PhoneService | categorical | Has phone service | Yes / No |
| 8 | MultipleLines | categorical | Multiple phone lines | Yes / No / No phone service |
| 9 | InternetService | categorical | Internet service provider | DSL / Fiber optic / No |
| 10 | OnlineSecurity | categorical | Online security subscription | Yes / No / No internet service |
| 11 | OnlineBackup | categorical | Online backup subscription | Yes / No / No internet service |
| 12 | DeviceProtection | categorical | Device protection plan | Yes / No / No internet service |
| 13 | TechSupport | categorical | Tech support subscription | Yes / No / No internet service |
| 14 | StreamingTV | categorical | Streaming TV service | Yes / No / No internet service |
| 15 | StreamingMovies | categorical | Streaming movies service | Yes / No / No internet service |
| 16 | Contract | categorical | Contract type | Month-to-month / One year / Two year |
| 17 | PaperlessBilling | categorical | Paperless billing enabled | Yes / No |
| 18 | PaymentMethod | categorical | Payment method | Electronic check / Mailed check / Bank transfer / Credit card |
| 19 | MonthlyCharges | numeric | Monthly charge amount (USD) | e.g., 29.85 |
| 20 | TotalCharges | numeric | Total charges to date (may contain blanks) | e.g., 29.85; blank treated as missing |
| 21 | Churn | target (binary) | Whether the customer churned | Yes / No (map to 1/0) |

Notes:
- TotalCharges may contain blank strings; convert to numeric and impute as needed.
- Many "No internet service" / "No phone service" values encode feature absence — consider consolidating or one-hot encoding.
- Target: map "Yes"→1, "No"→0 for modeling.
- Check class balance before training; apply resampling or class weights if heavily imbalanced.

## Project Task
- Binary classification: predict `Churn` (Yes/No → 1/0).  
- Evaluate models using Accuracy, Precision, Recall, F1, ROC-AUC and confusion matrix.  
- Prioritize Recall for business use-cases (catch potential churners).

## Methodology
1. Data loading & exploration  
2. Clean missing / blank values (e.g., TotalCharges)  
3. Encode categorical variables (one-hot / label where appropriate)  
4. Scale numeric features (StandardScaler) for distance-based methods  
5. Train and tune models:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest
6. Evaluate models and compare performance; inspect feature importances and produce an insights report.

## Model Summary Comparison
| Model                           |   Accuracy |   Precision |   Recall |       F1 |   ROC_AUC |   TP |   FP |   TN |   FN |
|:--------------------------------|-----------:|------------:|---------:|---------:|----------:|-----:|-----:|-----:|-----:|
| Random Forest                   |   0.800568 |    0.66787  | 0.494652 | 0.568356 |  0.84463  |  185 |   92 |  943 |  189 |
| Logistic Regression             |   0.800568 |    0.654485 | 0.526738 | 0.583704 |  0.842396 |  197 |  104 |  931 |  177 |
| Decision Tree (depth=5, leaf=3) |   0.79418  |    0.62963  | 0.545455 | 0.584527 |  0.828841 |  204 |  120 |  915 |  170 |
| KNN (K=9)                       |   0.77005  |    0.569444 | 0.548128 | 0.558583 |  0.799145 |  205 |  155 |  880 |  169 |


## Model Performance Discussion

1. **Random Forest**
   - **ROC-AUC:** 0.8446 (highest) – excellent class distinction
   - **Precision:** 0.6679 – most reliable positive predictions
   - **Recall:** 0.4947 – slightly lower than Logistic Regression
   - Robust and less prone to overfitting

2. **Logistic Regression**
   - Best balance between precision and recall
   - **F1-Score:** 0.5837 (highest)
   - Identifies more true positives than Random Forest but produces more false positives
   - Strong alternative when recall is important

3. **Decision Tree**
   - Performs respectably with recall close to Logistic Regression
   - Slightly lower precision and ROC-AUC than Random Forest
   - More prone to misclassification (higher false positives)
   - Useful for interpretability but weaker generalization

4. **KNN (k=9)**
   - Weakest overall performance
   - Lowest accuracy, precision, and ROC-AUC
   - Produces the highest number of false positives
   - Least suitable for reliable classification

---

## Best Model Recommendation

1. **Recommended: Random Forest**
   - **ROC-AUC:** highest, strong discriminative power
   - **Precision:** highest, reliable positive predictions
   - Stable performance, resistant to overfitting
   - Balanced trade-off between accuracy and reliability

2. **Alternative (if recall-focused): Logistic Regression**
   - Best for identifying as many positive cases as possible
   - Slightly higher false positives are acceptable
   - Interpretable and consistent performance


## Libraries Used
- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  
- notebook / ipython  
(See [requirements.txt](requirements.txt) for pinned versions.)

## How to run (Windows)
1. Clone the repository:  
```bash
git clone https://github.com/Rezaul33/Customer-Churn-Prediction-using-Machine-Learning
```
2. Open PowerShell / cmd at project root:
   - ```python -m venv .venv```
   - ```.venv\Scripts\activate```
3. Install dependencies:
   - ```pip install -r requirements.txt```
4. Launch the notebook:
   - jupyter notebook Notebook.ipynb
5. Run cells sequentially. Save any updated outputs or export the insights report via the notebook code cells.

## Folder Structure
```
Customer-Churn-Prediction-using-Machine-Learning
├── Customer-Churn-Prediction.ipynb
├── README.md
├── requirements.txt
├── License
└── Data/
    └── Data Data/WA_Fn-UseC_-Telco-Customer-Churn.csv

```

## Notes 
- Consider addressing class imbalance (class_weight or resampling) if needed.

## Author & License
Author: Rezaul Islam. [Linkedin](https://www.linkedin.com/in/md-rezaul-islam-cse/)
  

[License: MIT](License)



