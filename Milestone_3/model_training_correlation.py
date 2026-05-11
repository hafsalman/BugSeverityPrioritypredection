import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.linear_model  import LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.naive_bayes   import GaussianNB
from sklearn.svm           import SVC
from xgboost import XGBClassifier

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "Milestone_1", "frontend_uiux_bug_dataset_cleaned.csv")
FEATURES_PATH = os.path.join(SCRIPT_DIR, "..", "Milestone_2", "selected_features_correlation.json")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "model_results")
RANDOM_STATE = 42
TEST_SIZE    = 0.20

SEV_LABELS = ["Low", "Medium", "High", "Critical"]
PRI_LABELS = ["P3", "P2", "P1"]

print("="*65)
print("DS4SE – Milestone 3: Model Training & Evaluation")
print("="*65)

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

print("\nStep 1: Loading dataset…")
df = pd.read_csv(DATASET_PATH)
print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Severity distribution:\n{df['severity'].value_counts().to_string()}")
print(f"\n  Priority distribution:\n{df['priority'].value_counts().to_string()}")

print("\nStep 2: Loading selected features from Milestone 2…")

# Load pre-computed features and correlation scores from Milestone_2
with open(FEATURES_PATH, "r") as f:
    features_data = json.load(f)

SELECTED_FEATURES = features_data["selected_features"]
sev_correlation = features_data["severity_correlation_scores"]
pri_correlation = features_data["priority_correlation_scores"]

print(f"  Loaded {len(SELECTED_FEATURES)} selected features from Milestone 2")
print(f"  Features: {SELECTED_FEATURES}")

# Drop text/NLP and identifier columns – they cannot be used directly
nlp_drop = ["bug_id", "steps_to_reproduce", "expected_behavior",
            "actual_behavior", "tags"]
df.drop(columns=[c for c in nlp_drop if c in df.columns], inplace=True)

# Boolean → integer
for col in ["screenshot_available", "resolved"]:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Label-encode categorical columns
cat_cols = ["app_name", "module", "bug_type", "device_type",
            "browser", "os", "reported_by"]
le_registry = {}
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_registry[col] = le

# Ordinal-encode targets (preserves natural ordering)
sev_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
pri_map = {"P3": 0, "P2": 1, "P1": 2}
df["sev_enc"] = df["severity"].map(sev_map)
df["pri_enc"] = df["priority"].map(pri_map)

# Use only the selected features from Milestone 2
FEATURE_COLS = [f for f in SELECTED_FEATURES if f in df.columns]
print(f"  Using {len(FEATURE_COLS)} selected features for training")
print(f"  Features: {FEATURE_COLS}")

# Create correlation dataframes for visualization from loaded data
corr_s_list = [sev_correlation.get(f, 0) for f in FEATURE_COLS]
corr_p_list = [pri_correlation.get(f, 0) for f in FEATURE_COLS]

df_corr_s = pd.DataFrame({"Feature": FEATURE_COLS, "Correlation_Severity": corr_s_list}).sort_values("Correlation_Severity", ascending=False)
df_corr_p = pd.DataFrame({"Feature": FEATURE_COLS, "Correlation_Priority": corr_p_list}).sort_values("Correlation_Priority", ascending=False)

print("\n  Top features for Severity (from Milestone 2):")
print(df_corr_s.to_string(index=False))
print("\n  Top features for Priority (from Milestone 2):")
print(df_corr_p.to_string(index=False))

X  = df[FEATURE_COLS].values
y_s = df["sev_enc"].values
y_p = df["pri_enc"].values

# ──────────────────────────────────────────────────────────────────
# STEP 3 – Feature Selection (loaded from Milestone 2)
# ──────────────────────────────────────────────────────────────────
print("\nStep 3: Feature selection completed (loaded from Milestone 2)")
print(f"  {len(FEATURE_COLS)} features selected for training")

# ──────────────────────────────────────────────────────────────────
# STEP 4 – Train / Test Split
# ──────────────────────────────────────────────────────────────────
print("\nStep 4: Splitting data (80/20 stratified)…")
Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
    X, y_s, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_s)
Xtr_p, Xte_p, ytr_p, yte_p = train_test_split(
    X, y_p, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_p)
print(f"  Train: {Xtr_s.shape[0]:,}  |  Test: {Xte_s.shape[0]:,}")

# Scale for distance/probability-based models
sc_s = StandardScaler(); sc_p = StandardScaler()
Xtr_s_sc = sc_s.fit_transform(Xtr_s); Xte_s_sc = sc_s.transform(Xte_s)
Xtr_p_sc = sc_p.fit_transform(Xtr_p); Xte_p_sc = sc_p.transform(Xte_p)

# ──────────────────────────────────────────────────────────────────
# STEP 5 – Define Models
# ──────────────────────────────────────────────────────────────────
# Each entry: (model_sev, model_pri, use_scaled_data)
MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
        LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
        True
    ),
    "Naive Bayes": (
        GaussianNB(),
        GaussianNB(),
        True
    ),
    "Decision Tree": (
        DecisionTreeClassifier(max_depth=15, min_samples_leaf=20, random_state=RANDOM_STATE),
        DecisionTreeClassifier(max_depth=15, min_samples_leaf=20, random_state=RANDOM_STATE),
        False
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=10,
                               n_jobs=-1, random_state=RANDOM_STATE),
        RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=10,
                               n_jobs=-1, random_state=RANDOM_STATE),
        False
    ),
    "Passive Aggressive": (
        PassiveAggressiveClassifier(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
        PassiveAggressiveClassifier(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
        True
    ),
    "SVM": (
        SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE),
        SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE),
        True
    ),
    "XGBoost": (
        XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                      random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                      random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        False
    ),
}

# ──────────────────────────────────────────────────────────────────
# STEP 6 – Train & Evaluate
# ──────────────────────────────────────────────────────────────────
print("\nStep 4: Training & Evaluating Models…")
print("-"*65)

results = {}
for name, (m_s, m_p, scaled) in MODELS.items():
    print(f"\n  [{name}]")
    Xtr_s_use = Xtr_s_sc if scaled else Xtr_s
    Xte_s_use = Xte_s_sc if scaled else Xte_s
    Xtr_p_use = Xtr_p_sc if scaled else Xtr_p
    Xte_p_use = Xte_p_sc if scaled else Xte_p

    # Severity
    m_s.fit(Xtr_s_use, ytr_s)
    pred_s = m_s.predict(Xte_s_use)
    acc_s  = accuracy_score(yte_s, pred_s)
    f1_s   = f1_score(yte_s, pred_s, average="weighted")

    # Priority
    m_p.fit(Xtr_p_use, ytr_p)
    pred_p = m_p.predict(Xte_p_use)
    acc_p  = accuracy_score(yte_p, pred_p)
    f1_p   = f1_score(yte_p, pred_p, average="weighted")

    print(f"    Severity  →  Accuracy: {acc_s:.4f}   F1(weighted): {f1_s:.4f}")
    print(f"    Priority  →  Accuracy: {acc_p:.4f}   F1(weighted): {f1_p:.4f}")
    results[name] = {
        "sev": {"acc": acc_s, "f1": f1_s, "pred": pred_s, "model": m_s},
        "pri": {"acc": acc_p, "f1": f1_p, "pred": pred_p, "model": m_p},
    }

# ──────────────────────────────────────────────────────────────────
# STEP 7 – Detailed Reports for Best Models
# ──────────────────────────────────────────────────────────────────
best_s = max(results, key=lambda k: results[k]["sev"]["acc"])
best_p = max(results, key=lambda k: results[k]["pri"]["acc"])

print("\n" + "="*65)
print("Step 5: Detailed Classification Reports…")
print(f"\n  Best model for SEVERITY : {best_s}")
print(classification_report(yte_s, results[best_s]["sev"]["pred"],
                             target_names=SEV_LABELS))
print(f"\n  Best model for PRIORITY : {best_p}")
print(classification_report(yte_p, results[best_p]["pri"]["pred"],
                             target_names=PRI_LABELS))

# ──────────────────────────────────────────────────────────────────
# STEP 8 – Visualizations
# ──────────────────────────────────────────────────────────────────
print("\nStep 6: Generating Visualizations…")
model_names = list(results.keys())

def bar_annotations(ax, bars):
    for bar in bars:
        ax.annotate(f"{bar.get_height():.1f}%",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

x = np.arange(len(model_names)); w = 0.35

# Plot 1 – Accuracy comparison
sev_accs = [results[m]["sev"]["acc"]*100 for m in model_names]
pri_accs = [results[m]["pri"]["acc"]*100 for m in model_names]
fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - w/2, sev_accs, w, label="Severity", color="steelblue")
b2 = ax.bar(x + w/2, pri_accs, w, label="Priority",  color="coral")
bar_annotations(ax, list(b1) + list(b2))
ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=15, ha="right")
ax.set_xlabel("Model"); ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy Comparison – Severity vs Priority",
             fontsize=14, fontweight="bold")
ax.legend(); ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_accuracy_comparison_correlation.png", dpi=150); plt.close()

# Plot 2 – F1 comparison
sev_f1s = [results[m]["sev"]["f1"]*100 for m in model_names]
pri_f1s = [results[m]["pri"]["f1"]*100 for m in model_names]
fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - w/2, sev_f1s, w, label="Severity", color="mediumpurple")
b2 = ax.bar(x + w/2, pri_f1s, w, label="Priority", color="mediumseagreen")
bar_annotations(ax, list(b1) + list(b2))
ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=15, ha="right")
ax.set_xlabel("Model"); ax.set_ylabel("Weighted F1 (%)")
ax.set_title("Model Weighted F1-Score Comparison", fontsize=14, fontweight="bold")
ax.legend(); ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_f1_comparison_correlation.png", dpi=150); plt.close()

# Plot 3 & 4 – Confusion Matrices
for task, name, y_true, y_pred, labels, fname in [
    ("Severity", best_s, yte_s, results[best_s]["sev"]["pred"],
     SEV_LABELS, "03_confusion_severity_correlation"),
    ("Priority", best_p, yte_p, results[best_p]["pri"]["pred"],
     PRI_LABELS, "04_confusion_priority_correlation"),
]:
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix – {task}\n({name}, % per true class)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{fname}.png", dpi=150); plt.close()

# Plot 5 – Random Forest feature importance (severity)
rf_model = results["Random Forest"]["sev"]["model"]
fi_df = pd.DataFrame({"Feature": FEATURE_COLS,
                       "Importance": rf_model.feature_importances_*100})
fi_df = fi_df.sort_values("Importance")
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(fi_df["Feature"], fi_df["Importance"], color="steelblue")
ax.set_xlabel("Feature Importance (%)"); ax.set_ylabel("Feature")
ax.set_title("Random Forest – Feature Importance for Severity",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_rf_feature_importance_correlation.png", dpi=150); plt.close()

# Plot 6 – Correlation Analysis (both targets, from Milestone 2)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, df_corr, col, title, color in [
    (axes[0], df_corr_s.sort_values("Correlation_Severity"),     "Correlation_Severity", "Severity", "tomato"),
    (axes[1], df_corr_p.sort_values("Correlation_Priority"),     "Correlation_Priority", "Priority", "mediumorchid"),
]:
    ax.barh(df_corr["Feature"], df_corr[col], color=color)
    ax.set_xlabel("Absolute Correlation")
    ax.set_title(f"Correlation – {title}", fontweight="bold")
plt.suptitle("Feature Importance via Correlation Analysis (from Milestone 2)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_correlation_analysis_correlation.png", dpi=150); plt.close()

# Plot 7 – Model summary table image
fig, ax = plt.subplots(figsize=(13, 4))
ax.axis("off")
table_data = [["Model", "Sev Acc", "Sev F1", "Pri Acc", "Pri F1"]]
for m in model_names:
    table_data.append([
        m,
        f"{results[m]['sev']['acc']*100:.1f}%",
        f"{results[m]['sev']['f1']*100:.1f}%",
        f"{results[m]['pri']['acc']*100:.1f}%",
        f"{results[m]['pri']['f1']*100:.1f}%",
    ])
tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               cellLoc="center", loc="center",
               colColours=["#4472C4"]*5)
tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 2.5)
for j in range(5):
    tbl[0, j].set_text_props(color="white", fontweight="bold")
ax.set_title("Model Performance Summary", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_model_summary_table_correlation.png", dpi=150); plt.close()

print(f"  Saved 7 plots to '{OUTPUT_DIR}/'")

# ──────────────────────────────────────────────────────────────────
# STEP 9 – Export JSON summary
# ──────────────────────────────────────────────────────────────────
summary = {
    "project":      "DS4SE – Bug Severity and Priority Prediction",
    "dataset_size": 500_000,
    "features":     FEATURE_COLS,
    "results": {
        task_key: {
            m: {
                "accuracy":    round(results[m][tk]["acc"], 4),
                "f1_weighted": round(results[m][tk]["f1"],  4),
            }
            for m in model_names
        }
        for task_key, tk in [("severity", "sev"), ("priority", "pri")]
    },
    "best_model": {
        "severity": best_s,
        "priority": best_p,
    }
}
json_path = f"{OUTPUT_DIR}/model_summary_correlation.json"
with open(json_path, "w") as f:
    json.dump(summary, f, indent=4)
print(f"  Saved JSON summary → {json_path}")

# ──────────────────────────────────────────────────────────────────
# STEP 10 – Save best models & encoders for predict_bug.py
# ──────────────────────────────────────────────────────────────────
print("\nStep 8: Saving best models and encoders for prediction…")

joblib.dump(results[best_s]["sev"]["model"], os.path.join(SCRIPT_DIR, "severity_rf_model_correlation.joblib"))
joblib.dump(results[best_p]["pri"]["model"], os.path.join(SCRIPT_DIR, "priority_rf_model_correlation.joblib"))

encoders = {
    "features":        le_registry,
    "cat_cols":        cat_cols,
    "feature_cols":    FEATURE_COLS,
    "sev_map":         sev_map,
    "pri_map":         pri_map,
    "best_sev_model":  best_s,
    "best_pri_model":  best_p,
}
joblib.dump(encoders, os.path.join(SCRIPT_DIR, "data_encoders_correlation.joblib"))

print(f"  Saved: severity_rf_model_correlation.joblib  (model: {best_s})")
print(f"  Saved: priority_rf_model_correlation.joblib  (model: {best_p})")
print(f"  Saved: data_encoders_correlation.joblib")

print("\n" + "="*65)
print("Milestone 3 Complete!")
print(f"  Best Severity model : {best_s}  ({results[best_s]['sev']['acc']*100:.1f}% accuracy)")
print(f"  Best Priority model : {best_p}  ({results[best_p]['pri']['acc']*100:.1f}% accuracy)")
print("  Model files saved → run predict_bug.py to make predictions.")
print("="*65)