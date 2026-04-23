# Healthcare Provider Fraud Detection - Phase 1

This project explores healthcare provider fraud detection using classical machine learning on aggregated claims and beneficiary data. The notebook walks through the full Phase 1 workflow: loading raw data, performing exploratory analysis, engineering provider-level features, training baseline models, and comparing results.

## Project Objective

The goal of this phase is to identify whether a healthcare provider is potentially fraudulent based on historical claim behavior and beneficiary patterns.

This notebook focuses on:

- understanding the structure of the raw datasets
- combining inpatient, outpatient, beneficiary, and provider label data
- engineering provider-level features
- handling class imbalance
- training and evaluating machine learning models
- extracting early business insights from feature importance and clustering

## Dataset Files

Place the following files inside a `dataset/` folder in the same directory as the notebook:

- `Train_Beneficiarydata-1542865627584.csv`
- `Train_Inpatientdata-1542865627584.csv`
- `Train_Outpatientdata-1542865627584.csv`
- `Train-1542865627584.csv`

## Notebook Structure

### Phase 1: Data Exploration
The notebook begins by loading and inspecting four training datasets:

- **Beneficiary data**: patient demographics and chronic conditions
- **Inpatient claims**
- **Outpatient claims**
- **Provider fraud labels**

Initial exploration includes:

- dataset shapes
- sample rows
- data types
- descriptive statistics
- missing value checks
- fraud class distribution

### Phase 2: Feature Engineering and Preprocessing
The project then builds a provider-level modeling table by:

1. concatenating inpatient and outpatient claims
2. merging claims with beneficiary data using `BeneID`
3. aggregating records to the **provider level**
4. merging the engineered features with fraud labels

Examples of engineered features include:

- claim count, sum, mean, std, max, and min of reimbursed amount
- number of unique diagnosis codes
- number of unique procedure codes
- inpatient claim ratio
- number of unique beneficiaries served
- beneficiary demographic summaries
- chronic condition counts

The final modeling dataset contains:

- **5,410 providers**
- **35 input features**
- **1 binary target column (`PotentialFraud`)**

### Phase 3: Model Development and Evaluation
The notebook trains and compares multiple approaches:

- **Logistic Regression** as a baseline classifier
- **Random Forest** as the main supervised model
- **K-Means Clustering** for unsupervised pattern exploration

It also includes:

- stratified train/test splitting
- standardization
- SMOTE for class imbalance handling
- ROC curve comparison
- confusion matrices
- feature importance analysis

## Class Distribution

The provider fraud labels are imbalanced:

- **Non-fraud**: 4,904 providers
- **Fraud**: 506 providers
- **Fraud rate**: 9.35%

Because of this imbalance, the notebook uses **SMOTE** to oversample the minority class during model development.

## Model Results

### Logistic Regression
- Recall: **0.8812**
- Precision: **0.4120**
- F1-score: **0.5615**
- ROC-AUC: **0.9546**

This model captures a large share of fraudulent providers, but produces many false positives.

### Random Forest
- Recall: **1.0000**
- Precision: **0.8938**
- F1-score: **0.9439**
- ROC-AUC: **1.0000**

The Random Forest model performs much better than Logistic Regression in this notebook and is identified as the best-performing model in Phase 1.

### K-Means Clustering
Using `k = 3`, clustering reveals groups with very different fraud concentrations:

- Cluster 0: **4.51%** fraud
- Cluster 1: **40.28%** fraud
- Cluster 2: **89.71%** fraud

This suggests that provider behavior naturally separates into meaningful risk segments.

## Top Fraud Indicators

Based on Random Forest feature importance, the strongest fraud-related signals include:

- total reimbursed claim amount
- number of unique procedure codes
- maximum reimbursed amount
- inpatient/outpatient claim mix
- provider-level utilization patterns

These features suggest that unusually large claim activity and complex procedure behavior may be strong indicators of fraudulent providers.

## Technologies Used

- Python
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn (SMOTE)

## How to Run

1. Clone the repository.
2. Place the required CSV files inside a `dataset/` folder.
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```
4. Open the notebook:
   ```bash
   jupyter notebook fraud_detection_phase1.ipynb
   ```
5. Run the cells from top to bottom.

## Repository Output

By the end of the notebook, you will have:

- a provider-level fraud modeling dataset
- exploratory statistics and visualizations
- trained Logistic Regression and Random Forest models
- ROC and confusion matrix comparisons
- Random Forest feature importance rankings
- K-Means cluster analysis for fraud segmentation

## Notes and Limitations

- This notebook is a **Phase 1 exploration** and should be treated as an initial modeling baseline.
- The current implementation appears to apply scaling and SMOTE using the full feature matrix rather than only the training split. That can introduce **data leakage** and make evaluation results look better than they would be in a stricter production pipeline.
- Some engineered feature names come from lambda aggregations (for example, `ClaimType_<lambda>`), which should be renamed more clearly in a later cleanup pass.
- The very strong Random Forest results are promising, but they should be revalidated with a fully leakage-safe pipeline and cross-validation before making deployment claims.

## Future Improvements

Suggested next steps:

- fix preprocessing leakage by fitting scaling and SMOTE only on training data
- rename engineered columns for readability
- add cross-validation and hyperparameter tuning
- test XGBoost, LightGBM, or CatBoost
- analyze threshold tuning for fraud recall vs precision tradeoffs
- add SHAP or permutation importance for explainability
- package preprocessing and modeling into reusable pipelines

## Author / Purpose

This notebook was created as part of an early-stage healthcare fraud detection project to explore whether provider-level claim patterns can be used to distinguish potentially fraudulent providers from non-fraudulent ones.
