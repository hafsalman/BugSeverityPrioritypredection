# DS4SE Project: Milestone 2: Statistical Analysis & Graph Generation
# Group Leader: Hira Zahoor (22K-4823)
# Members:   Hafsa Salman (22K-5161)
#            Jaysha Iqbal (22K-5178)
#            Amna Mansoor (22K-5159)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import json
import os

print("="*60)
print("DS4SE Project - Statistical Analysis & Graph Generation")
print("="*60)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIZ_DIR    = os.path.join(SCRIPT_DIR, "project_visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

print("\nStep 1: Loading dataset...")

CSV_PATH = os.path.join(SCRIPT_DIR, "..", "Milestone_1", "frontend_uiux_bug_dataset_cleaned.csv")


df = pd.read_csv(CSV_PATH)

print("\nStep 2: Preparing data for Statistical Analysis...")
target_cols = ['severity', 'priority']

nlp_cols  = ['page_url', 'steps_to_reproduce', 'expected_behavior', 'actual_behavior']
drop_cols = [c for c in nlp_cols if c in df.columns]

feature_cols = [c for c in df.columns if c not in target_cols + drop_cols]
print(f" -> Analyzing {len(feature_cols)} structured features.")

df_encoded      = df.copy()
non_numeric_cols = df_encoded.select_dtypes(exclude=[np.number]).columns
for col in non_numeric_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

X     = df_encoded[feature_cols]
y_sev = df_encoded['severity']
y_pri = df_encoded['priority']

print("\nStep 3: Calculating Information Gain (Mutual Information)...")
mi_sev = mutual_info_classif(X, y_sev, random_state=42)
mi_pri = mutual_info_classif(X, y_pri, random_state=42)

df_mi_sev = (pd.DataFrame({'Feature': feature_cols, 'Importance': mi_sev})
               .sort_values(by='Importance', ascending=False))
df_mi_pri = (pd.DataFrame({'Feature': feature_cols, 'Importance': mi_pri})
               .sort_values(by='Importance', ascending=False))

print("\nStep 3b: Calculating Pearson Correlation Analysis...")
# Calculate Pearson correlation between features and targets
corr_sev = np.array([np.corrcoef(X[feature_cols[i]], y_sev)[0, 1] for i in range(len(feature_cols))])
corr_pri = np.array([np.corrcoef(X[feature_cols[i]], y_pri)[0, 1] for i in range(len(feature_cols))])

# Replace NaN values (from constant features) with 0
corr_sev = np.nan_to_num(corr_sev, nan=0.0)
corr_pri = np.nan_to_num(corr_pri, nan=0.0)

# Use absolute correlation values for ranking
df_corr_sev = (pd.DataFrame({'Feature': feature_cols, 'Correlation': np.abs(corr_sev)})
                 .sort_values(by='Correlation', ascending=False))
df_corr_pri = (pd.DataFrame({'Feature': feature_cols, 'Correlation': np.abs(corr_pri)})
                 .sort_values(by='Correlation', ascending=False))

print("\nStep 4: Exporting selected_features.json (Mutual Information)...")
combined_importance = (
    df_mi_sev.set_index('Feature')['Importance'] +
    df_mi_pri.set_index('Feature')['Importance']
).sort_values(ascending=False)

selected_features = combined_importance.head(10).index.tolist()

output_json = {
    "project":           "DS4SE - Bug Severity and Priority Prediction",
    "targets":           target_cols,
    "selected_features": selected_features,
    "severity_importance_scores": df_mi_sev.set_index('Feature')['Importance'].to_dict(),
    "priority_importance_scores": df_mi_pri.set_index('Feature')['Importance'].to_dict(),
}

json_path = os.path.join(SCRIPT_DIR, "selected_features.json")
with open(json_path, "w") as f:
    json.dump(output_json, f, indent=4)
print(" -> selected_features.json saved successfully.")

print("\nStep 4b: Exporting selected_features_correlation.json (Pearson Correlation)...")
combined_corr = (
    df_corr_sev.set_index('Feature')['Correlation'] +
    df_corr_pri.set_index('Feature')['Correlation']
).sort_values(ascending=False)

selected_features_corr = combined_corr.head(10).index.tolist()

output_json_corr = {
    "project":           "DS4SE - Bug Severity and Priority Prediction",
    "targets":           target_cols,
    "selected_features": selected_features_corr,
    "severity_correlation_scores": df_corr_sev.set_index('Feature')['Correlation'].to_dict(),
    "priority_correlation_scores": df_corr_pri.set_index('Feature')['Correlation'].to_dict(),
}

json_path_corr = os.path.join(SCRIPT_DIR, "selected_features_correlation.json")
with open(json_path_corr, "w") as f:
    json.dump(output_json_corr, f, indent=4)
print(" -> selected_features_correlation.json saved successfully.")

print("\nStep 5: Generating Visualizations...")

severity_order = [s for s in ["Critical", "High", "Medium", "Low"]
                  if s in df['severity'].unique()]
priority_order = [p for p in ["P1", "P2", "P3"]
                  if p in df['priority'].unique()]

def save(filename):
    path = os.path.join(VIZ_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Saved: project_visualizations/{filename}")

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='severity', order=severity_order,
              hue='severity', hue_order=severity_order,
              palette="Reds_r", legend=False)
plt.title('Distribution of Bug Severity', fontsize=14, fontweight='bold')
plt.xlabel('Severity Level')
plt.ylabel('Number of Bugs')
plt.tight_layout()
save('01_severity_distribution.png')

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='priority', order=priority_order,
              hue='priority', hue_order=priority_order,
              palette="Oranges_r", legend=False)
plt.title('Distribution of Bug Priority', fontsize=14, fontweight='bold')
plt.xlabel('Priority Level')
plt.ylabel('Number of Bugs')
plt.tight_layout()
save('02_priority_distribution.png')

plt.figure(figsize=(8, 6))
crosstab = pd.crosstab(df['severity'], df['priority'])
crosstab = crosstab.reindex(index=severity_order, columns=priority_order)
sns.heatmap(crosstab, annot=True, fmt='d', cmap="YlOrRd", linewidths=.5)
plt.title('Severity vs. Priority Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
save('03_severity_priority_matrix.png')

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='device_type',
              hue='severity', hue_order=severity_order, palette="Set2")
plt.title('Bug Severity Breakdown by Device Type', fontsize=14, fontweight='bold')
plt.xlabel('Device Type')
plt.ylabel('Number of Bugs')
plt.legend(title='Severity')
plt.tight_layout()
save('04_device_vs_severity.png')

plt.figure(figsize=(12, 6))
top_modules    = df['module'].value_counts().head(10).index
df_top_modules = df[df['module'].isin(top_modules)]
sns.countplot(data=df_top_modules, y='module',
              hue='priority', hue_order=priority_order, palette="Set1")
plt.title('Top 10 Modules by Bug Priority', fontsize=14, fontweight='bold')
plt.xlabel('Number of Bugs')
plt.ylabel('Application Module')
plt.legend(title='Priority')
plt.tight_layout()
save('05_module_vs_priority.png')

plt.figure(figsize=(10, 6))
df_sev_imp = df_mi_sev[df_mi_sev['Importance'] > 0].copy()
sns.barplot(data=df_sev_imp, x='Importance', y='Feature',
            hue='Feature', palette='viridis', legend=False)
plt.title('Feature Importance for SEVERITY (Mutual Information)',
          fontsize=14, fontweight='bold')
plt.xlabel('Information Gain')
plt.ylabel('Features')
plt.tight_layout()
save('06_severity_feature_importance.png')

plt.figure(figsize=(10, 6))
df_pri_imp = df_mi_pri[df_mi_pri['Importance'] > 0].copy()
sns.barplot(data=df_pri_imp, x='Importance', y='Feature',
            hue='Feature', palette='magma', legend=False)
plt.title('Feature Importance for PRIORITY (Mutual Information)',
          fontsize=14, fontweight='bold')
plt.xlabel('Information Gain')
plt.ylabel('Features')
plt.tight_layout()
save('07_priority_feature_importance.png')

plt.figure(figsize=(10, 6))
df_sev_corr = df_corr_sev[df_corr_sev['Correlation'] > 0].copy()
sns.barplot(data=df_sev_corr, x='Correlation', y='Feature',
            hue='Feature', palette='coolwarm', legend=False)
plt.title('Feature Importance for SEVERITY (Pearson Correlation)',
          fontsize=14, fontweight='bold')
plt.xlabel('Absolute Correlation')
plt.ylabel('Features')
plt.tight_layout()
save('08_severity_correlation.png')

plt.figure(figsize=(10, 6))
df_pri_corr = df_corr_pri[df_corr_pri['Correlation'] > 0].copy()
sns.barplot(data=df_pri_corr, x='Correlation', y='Feature',
            hue='Feature', palette='twilight', legend=False)
plt.title('Feature Importance for PRIORITY (Pearson Correlation)',
          fontsize=14, fontweight='bold')
plt.xlabel('Absolute Correlation')
plt.ylabel('Features')
plt.tight_layout()
save('09_priority_correlation.png')

print("\n -> 9 Graphs saved successfully in 'project_visualizations/' folder.")
print("\n" + "="*60)
print("Process Complete! You are ready for the ML Modeling phase.")
print("="*60)