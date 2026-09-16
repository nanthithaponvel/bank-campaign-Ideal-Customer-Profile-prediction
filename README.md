# Bank Campaign Ideal Customer Profile Prediction

## Project Overview

Bank Campaign Ideal Customer Profile Prediction is a Machine Learning project that predicts the likelihood of customers subscribing to a bank term deposit during direct marketing campaigns.

The project uses customer, campaign, contact, previous campaign, and economic indicators to generate customer-level subscription probabilities. These probabilities are then used to prioritize customers into different targeting categories.

The objective is not only to predict whether a customer will subscribe, but also to support **data-driven campaign targeting and customer prioritization**.

---

## Business Problem

Banks may contact a large number of customers during marketing campaigns. Treating every customer equally can result in inefficient use of campaign resources.

This project addresses the problem by:

- Predicting customer subscription propensity
- Ranking customers based on predicted probability
- Identifying high-priority customers
- Analyzing important campaign and customer patterns
- Supporting more focused marketing campaigns

---

## Dataset

The project uses the **Bank Marketing dataset** from the UCI Machine Learning Repository.

- Dataset: Bank Marketing (`bank-additional-full.csv`)
- Records: **41,188**
- Original input features: **20**
- Target variable: `y`
- Target classes:
  - `no`
  - `yes`

The dataset contains information related to customer demographics, campaign interactions, previous campaign outcomes, contact methods, and economic indicators.

### Target Distribution

- `No`: approximately **88.7%**
- `Yes`: approximately **11.3%**

This indicates that the target variable is imbalanced, making metrics such as Precision, Recall, F1-score, and ROC-AUC important in addition to accuracy.

---

## Machine Learning Approach

### 1. Data Understanding

The dataset was examined to understand:

- Number of observations and features
- Feature types
- Target variable
- Categorical and numerical variables
- Unknown values
- Target distribution

### 2. Exploratory Data Analysis

EDA was performed to identify patterns related to:

- Customer characteristics
- Job categories
- Campaign month
- Contact method
- Previous campaign outcomes
- Campaign contact frequency
- Economic indicators

### 3. Data Preprocessing

Numerical and categorical variables were processed using a Scikit-learn preprocessing pipeline.

**Numerical features:**
- StandardScaler was used for numerical scaling.

**Categorical features:**
- One-Hot Encoding was used to convert categorical variables into numerical form.

### 4. Data Leakage Prevention

The `duration` feature was excluded from the final model.

`duration` represents the length of the current marketing call and is only known during or after the current call. Since the business objective is to identify customers for targeting **before the call**, including this variable could introduce data leakage.

Therefore, the final model uses **19 features**.

---

## Features Used

### Numerical Features

- age
- campaign
- pdays
- previous
- emp.var.rate
- cons.price.idx
- cons.conf.idx
- euribor3m
- nr.employed

### Categorical Features

- job
- marital
- education
- default
- housing
- loan
- contact
- month
- day_of_week
- poutcome

---

## Machine Learning Models

Four classification algorithms were compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

### Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 83.50% | 36.79% | 64.66% | 46.89% | 0.8009 |
| Decision Tree | 83.14% | 35.99% | 63.79% | 46.02% | 0.7921 |
| Random Forest | **85.87%** | **41.73%** | **64.12%** | **50.55%** | **0.8146** |
| Gradient Boosting | 90.09% | 67.50% | 23.28% | 34.62% | 0.8092 |

## Final Model

**Random Forest** was selected as the final model based on its overall balance between Precision and Recall and its ROC-AUC of **0.8146** among the tested models.

### Test Set Performance

The final Random Forest model achieved:

- **Accuracy:** 85.87%
- **Precision:** 41.73%
- **Recall:** 64.12%
- **F1-score:** 50.55%
- **ROC-AUC:** 0.8146

The model was evaluated on a held-out test set of **8,238 records**.

---

## Customer Targeting

The final model generates a subscription probability for each customer.

Instead of using only a yes/no prediction, customers can be ranked according to their predicted propensity.

The dashboard categorizes customers into:

- **High Priority**
- **Medium Priority**
- **Low Priority**

This allows campaign teams to focus resources on customers with higher predicted propensity.

### Full Dataset Scoring

The trained model was used to score **41,188 historical customer records**.

- Predicted subscribers: **7,207**
- Average predicted probability: **37.09%**
- High Priority: **5,290**
- Medium Priority: **5,298**
- Low Priority: **30,600**

### Top 10% Targeting Opportunity

The top 10% of customers represent **4,119 customers** ranked by predicted propensity.

Their average predicted probability was approximately **85.48%**.

These scores represent model-estimated propensity and should not be interpreted as guaranteed future subscriptions.

---

## Business Insights

The project provides insights into factors associated with campaign response, including:

### Campaign Month

Subscription propensity varies across campaign months, providing a basis for understanding seasonal campaign patterns.

### Previous Campaign Outcome

Previous campaign outcomes provide useful information for customer targeting.

### Job Analysis

Different customer job categories show differences in historical subscription patterns.

### Contact Method

The dataset contains two major contact methods:

- Cellular
- Telephone

The project analyzes their relationship with campaign response.

### Campaign Contact Frequency

The historical data shows a general decline in predicted propensity as the number of contacts in the current campaign increases.

This is treated as a historical association rather than a causal relationship.

---

## Feature Importance

The Random Forest model identified several influential predictive features, including:

- `euribor3m`
- `nr.employed`
- `emp.var.rate`
- `cons.conf.idx`
- `pdays`
- `cons.price.idx`
- `poutcome_success`
- `age`
- `month_may`
- `contact_telephone`

Feature importance indicates predictive contribution and does not imply that a feature directly causes subscription behavior.

---

## Dashboard

An interactive **Streamlit dashboard** was developed with the following sections:

- Overview
- Customer Targeting
- Business Insights
- Model Performance
- Project Information

The dashboard provides:

- KPI summaries
- Customer propensity scores
- Customer prioritization
- Model performance metrics
- Feature importance
- Campaign insights
- Targeting analysis

---

## Technology Stack

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Machine Learning

- Scikit-learn

### Visualization

- Matplotlib
- Seaborn

### Model Persistence

- Joblib

### Dashboard

- Streamlit

---

## Project Structure

```text
bank_marketing_project/
│
├── app.py
├── business_prediction_analysis.py
├── check_dataset.py
├── README.md
├── .gitignore
│
├── data/
│   └── bank-additional-full.csv
│
└── models/
    ├── best_model.joblib
    ├── full_customer_predictions.csv
    └── test_predictions.csv
