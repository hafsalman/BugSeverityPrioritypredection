# Bug Severity & Priority Prediction

A machine learning project for predicting bug severity and priority levels in frontend/UI/UX applications using structured feature engineering and multiple classification models.

## Overview

This project implements an end-to-end machine learning pipeline to automatically classify and prioritize software bugs based on their characteristics. By automating bug severity and priority assessment, development teams can allocate resources more efficiently and focus on critical issues.

**Dataset Size:** 500,000 bug records from Frontend UI/UX applications  
**Source:** [Kaggle - Frontend UI/UX Bug Dataset](https://www.kaggle.com/datasets/hahafsalman/frontend-uiux-bug-dataset-500k).

## Problem Statement

In large-scale software development, bug triage—the process of assigning severity levels and priorities to reported issues—is time-consuming and subjective. This project addresses this challenge by:

- **Automating bug severity classification** (Low, Medium, High, Critical)
- **Automating bug priority assignment** (P3, P2, P1)
- **Reducing manual triage effort** through machine learning predictions
- **Ensuring consistent prioritization** across development teams

## 📊 Dataset

| Aspect | Details |
|--------|---------|
| **Total Records** | 500,000 bug reports |
| **Domain** | Frontend UI/UX applications |
| **Target Variables** | Severity (4 classes), Priority (3 classes) |
| **Feature Count** | 10 selected features after analysis |
| **Processing** | 15-step cleaning pipeline |

### Key Dataset Features

- `user_impact_score` - Impact on end users
- `time_to_fix_sec` - Estimated fix time
- `bug_type` - Type of bug (UI, Performance, Security, etc.)
- `time_to_detect_sec` - Time between deployment and detection
- `module` - Application module affected
- `screenshot_available` - Whether screenshot evidence exists
- `resolved` - Resolution status
- `reported_by` - Reporter information
- `device_type` - Device category (Mobile, Desktop, Tablet)
- `app_name` - Application identifier

## Project Structure

```
BugSeverityPrioritypredection/
│
├── Milestone_1: Data Cleaning & Preprocessing
│   ├── dataset_clean.py              # 15-step data cleaning pipeline
│   ├── frontend_uiux_bug_dataset_500k.csv   # Raw dataset
│   ├── frontend_uiux_bug_dataset_cleaned.csv # Cleaned dataset
│   └── plots/                        # Initial exploratory plots
│
├── Milestone_2: Feature Analysis & Visualization
│   ├── Analysis.py                   # Statistical analysis & feature selection
│   ├── selected_features.json        # Top features with MI scores
│   ├── selected_features_correlation.json
│   └── project_visualizations/       # Correlation heatmaps, distributions
│
├── Milestone_3: Model Training & Prediction
│   ├── model_training.py             # Train 7 classification models
│   ├── model_training_correlation.py # Alternative training pipeline
│   ├── predict_bug.py                # Inference script
│   ├── severity_rf_model.joblib      # Trained Random Forest (Severity)
│   ├── priority_rf_model.joblib      # Trained Random Forest (Priority)
│   ├── data_encoders.joblib          # Feature encoders
│   └── model_results/
│       ├── model_summary.json        # Performance metrics for all models
│       └── model_summary_correlation.json
│
└── README.md
```

## Key Features

- **Comprehensive Data Pipeline:** 15-step cleaning process handling missing values, outliers, and data normalization
- **Feature Engineering:** Mutual information-based feature selection for optimal model input
- **Multi-Model Approach:** Evaluation of 7 different algorithms:
  - Logistic Regression
  - Naive Bayes
  - Decision Tree
  - Random Forest
  - SVM
  - Passive Aggressive Classifier
  - XGBoost
- **Visualization:** Correlation analysis, distribution plots, and model performance comparisons
- **Production Ready:** Trained models saved as serialized joblib objects for deployment

## Model Performance

### Severity Classification (4 Classes: Low, Medium, High, Critical)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | 75.82% | 0.759 |
| Naive Bayes | 75.77% | 0.759 |
| SVM | 80.48% | 0.806 |
| Random Forest | 88.34% | 0.884 |
| **Decision Tree** | **88.71%** | **0.887** ✓ |
| XGBoost | 88.65% | 0.887 |
| Passive Aggressive | 61.89% | 0.598 |

**Best Model:** Decision Tree (88.71% accuracy)

### Priority Classification (3 Classes: P3, P2, P1)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Passive Aggressive | 58.08% | 0.578 |
| Naive Bayes | 80.68% | 0.807 |
| Logistic Regression | 82.31% | 0.822 |
| SVM | 85.62% | 0.855 |
| Random Forest | 89.43% | 0.894 |
| Decision Tree | 89.56% | 0.895 |
| **XGBoost** | **89.63%** | **0.896** ✓ |

**Best Model:** XGBoost (89.63% accuracy)

## Installation

### Prerequisites

- Python 3.7+
- pip or conda

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/BugSeverityPrioritypredection.git
   cd BugSeverityPrioritypredection
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
   ```

## Usage

### 1. Data Cleaning (Milestone 1)

```bash
cd Milestone_1
python dataset_clean.py
```

**Output:** `frontend_uiux_bug_dataset_cleaned.csv`

### 2. Feature Analysis (Milestone 2)

```bash
cd ../Milestone_2
python Analysis.py
```

**Output:** Feature importance scores and visualizations in `project_visualizations/`

### 3. Model Training (Milestone 3)

```bash
cd ../Milestone_3
python model_training.py
```

**Output:** Trained models and performance metrics in `model_results/`

### 4. Make Predictions

```bash
python predict_bug.py
```

**Example prediction script:**
```python
import pandas as pd
import joblib

# Load trained models
severity_model = joblib.load("severity_rf_model.joblib")
priority_model = joblib.load("priority_rf_model.joblib")
encoders = joblib.load("data_encoders.joblib")

# Prepare your bug data
# ...

# Make predictions
severity_pred = severity_model.predict(X)
priority_pred = priority_model.predict(X)
```

## Results Summary

- **Best Severity Model:** Decision Tree with **88.71% accuracy**
- **Best Priority Model:** XGBoost with **89.63% accuracy**
- **Feature Set:** 10 optimized features selected through mutual information analysis
- **Dataset:** 500,000 bug records from production applications

### Key Insights

1. **Decision Tree excels at severity classification** - Captures complex feature interactions for severity levels
2. **XGBoost outperforms on priority** - Better handles the three-way priority classification
3. **Feature reduction improves efficiency** - From full feature set to 10 key features without significant accuracy loss
4. **High correlation detected** - Severity and priority are moderately correlated in the dataset

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/hahafsalman/frontend-uiux-bug-dataset-500k)
