# Loan Approval Prediction & Clustering System

An end-to-end Data Mining project that implements Supervised and Unsupervised Machine Learning algorithms to automate, evaluate, and segment banking loan approval applications. 

---

## 📑 Table of Contents
1. [Dataset Overview](#-dataset-overview)
2. [Data Preprocessing Pipeline](#-data-preprocessing-pipeline)
3. [Supervised Learning (Classification)](#-supervised-learning-classification)
4. [Unsupervised Learning (Clustering)](#-unsupervised-learning-clustering)
5. [Tech Stack & Libraries](#-tech-stack--libraries)

---

## 📊 Dataset Overview

The primary dataset `loan_approval_dataset.csv` contains **4,269 records** and **12 attributes** covering applicants' financial background, asset valuation, and historical credit ratings.

### Features Definition
| Feature Name | Description | Data Type |
| :--- | :--- | :--- |
| `loan_id` | Unique Identifier for each loan application | Numeric (Dropped during training) |
| `education` | Applicant's educational level (*Graduate* / *Not Graduate*) | Categorical |
| `self_employed` | Employment status (*Yes* / *No*) | Categorical |
| `income_annum` | Annual income of the applicant | Numeric |
| `loan_amount` | Requested loan amount | Numeric |
| `loan_term` | Repayment timeline (in months) | Numeric |
| `cibil_score` | Credit Rating / CIBIL Score | Numeric |
| `residential_assets_value` | Value of residential property | Numeric |
| `commercial_assets_value` | Value of commercial property | Numeric |
| `luxury_assets_value` | Value of luxury personal property | Numeric |
| `bank_asset_value` | Value of cash deposits/assets held in bank | Numeric |
| `loan_status` | **Target Variable**: Approval status (*Approved* / *Rejected*) | Categorical (Target) |

---


## 🛠 Data Preprocessing Pipeline

Initial Exploratory Data Analysis (EDA) revealed a highly clean dataset with **zero missing values** and **no duplicated entries**. The preprocessing workflow consists of:
1.  **Feature Dropping**: The `loan_id` attribute was discarded as it acts merely as an index and adds no predictive value.
2.  **Label Encoding**: Categorical parameters (`education`, `self_employed`, `loan_status`) were mapped to binary numeric values (`0` and `1`).
3.  **Data Normalization**: Applied `MinMaxScaler` to scale all quantitative financial attributes into a tight `[0, 1]` range to prevent attributes with larger scales from dominating the gradients during model training.

---

## 🤖 Supervised Learning (Classification)

The dataset was split using a standard **80% Train / 20% Test** ratio. The `LazyPredict` library was deployed to quickly bench-test 29 distinct classification models to select the best architecture.

### 🏆 Top 3 Performing Models:
1.  **LightGBM (LGBMClassifier)**: Highest performer with **Accuracy: 98.9%** and Balanced Accuracy: 98.8%.
2.  **Decision Tree**: Reached an Accuracy of **97.8%**.
3.  **XGBoost (XGBClassifier)**: Secured an Accuracy of **97.7%**.

### 🔍 XGBoost In-depth Performance:
* **Accuracy**: 97.66%
* **Precision / Recall / F1-Score**: ~96.86%
* **Confusion Matrix**:
    * True Positives (Correctly Approved): 526
    * True Negatives (Correctly Rejected): 308
    * Type I & Type II Errors: Restricted to only 10 cases each.
* **Feature Importance**: Based on the `plot_importance` output, **`cibil_score` (Credit Score)** stands out as the single most critical dominant factor dictating loan approvals, followed heavily by `loan_term` and asset profiles.

---

## 🌀 Unsupervised Learning (Clustering)

To partition and profile distinct customer personas, the data was condensed via **PCA (Principal Component Analysis)**, focusing primarily on highly correlated vectors: `loan_amount` and `luxury_assets_value`.

Three clustering frameworks were evaluated and compared:
1.  **K-Means Clustering**: Utilizing the **Elbow Method**, the optimal number of clusters was determined to be $K = 2$. It clearly segregates low-asset/low-demand applicants from high-net-worth premium tiers.
2.  **DBSCAN**: A density-based method that successfully discovered 2 primary highly dense clusters (4,185 core points), mapped out 56 border points, and effectively isolated 28 extreme outliers as noise points.
3.  **Agglomerative Clustering**: A hierarchical bottom-up clustering technique that provided a clear, multi-tiered structural look at applicant wealth demographics.

**Performance Evaluation (Silhouette Score):**
* **K-Means** and **Agglomerative Clustering** yielded the highest Silhouette Scores, proving to be the cleanest fit for partitioning this specific distribution of financial data.

---

## 💻 Tech Stack & Libraries

The system was fully implemented using **Python 3** and leverages the following core ecosystem:
* **Data Manipulation**: `pandas`, `numpy`
* **Data Visualization**: `matplotlib`, `seaborn`, `plotly`, `missingno`
* **Machine Learning**: `scikit-learn`, `lazypredict`, `xgboost`, `lightgbm`, `catboost`
