# DS4SE Project: Milestone 1: Data Cleaning Script
# Group Leader: Hira Zahoor (22K-4823)
# Members:   Hafsa Salman (22K-5161)
#            Jaysha Iqbal (22K-5178)
#            Amna Mansoor (22K-5159)
# Dataset: Frontend UI/UX Bug Dataset (500,000 records)
# We have performed data cleaning in 15 steps and additionally our code generates some visuals as well 
# The data set link from Kaggle that we are using for our project: https://www.kaggle.com/datasets/mirzayasirabdullah07/frontend-uiux-bug-dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Step no 1: Loading Dataset")

df = pd.read_csv('frontend_uiux_bug_dataset_500k.csv')

print(f"Dataset has been loaded successfully!")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn names: {list(df.columns)}")

print("Step 2: Initial exploration")

print("\nFirst 5 rows from the kaggle dataset are:")
print(df.head())

print("\nData Types for each column:")
print(df.dtypes)

print("\nBasic Statistics for only the numerical Columns")
print(df.describe())

print("\nBasic Statistics for the categorical columns")
print(df.describe(include="object"))

print("Step 3: Checking for missing values:")

missing     = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df  = pd.DataFrame({
    "Missing Count": missing,
    "Missing %":     missing_pct
})

has_missing = missing_df[missing_df["Missing Count"] > 0]

if len(has_missing) == 0:
    print("No missing values have been found in any column of our dataset. Nothing to fill.")
else:
    print("Following are the columns with missing values found:")
    print(has_missing)

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled '{col}' missing values with median: {median_val}")

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  Filled '{col}' missing values with mode: {mode_val}")

print(f"\nThe total missing values our code found after handling: {df.isnull().sum().sum()}")

print("Step 4: Checking for duplicate rows")

before     = len(df)
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows found through our code: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    print(f"Removed {duplicates} duplicate rows.")
    print(f"Rows before: {before} | Rows after: {len(df)}")
else:
    print("No duplicate rows were found so there is nothing to remove.")

print("Step 5: Standardizing the column names:")

print(f"Before: {list(df.columns)}")
df.columns = (df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_", regex=False))
print(f"After : {list(df.columns)}")

print("Step 6: Cleaning categorical columns:")

categorical_cols = [
    "app_name", "module", "bug_type", "severity",
    "priority", "device_type", "browser", "os", "reported_by"
]

for col in categorical_cols:
    if col in df.columns:
        before_unique = df[col].nunique()
        df[col] = df[col].str.strip()   
        df[col] = df[col].str.title()   
        after_unique = df[col].nunique()
        print(f"  '{col}': {before_unique} → {after_unique} unique values")

if "bug_type" in df.columns:
    df["bug_type"] = df["bug_type"].str.replace(
        "Animation Bug", "Animation", regex=False
    )
    print(f"\n  Fixed: 'Animation Bug' → 'Animation' in bug_type")
    print(f"  Unique bug_type values now: {sorted(df['bug_type'].unique().tolist())}")

print("Step 7: Validating target columns:")

valid_severity    = ["Critical", "High", "Medium", "Low"]
invalid_sev_mask  = ~df["severity"].isin(valid_severity)
invalid_sev_count = invalid_sev_mask.sum()

print(f"Valid severity levels  : {valid_severity}")
print(f"Unique values found    : {sorted(df['severity'].unique().tolist())}")
print(f"Invalid severity rows  : {invalid_sev_count}")

if invalid_sev_count > 0:
    print(f"  Removing {invalid_sev_count} rows with invalid severity values...")
    df = df[~invalid_sev_mask].reset_index(drop=True)

print(f"\nSeverity distribution after validation:")
print(df["severity"].value_counts())

valid_priority    = ["P1", "P2", "P3"]
invalid_pri_mask  = ~df["priority"].isin(valid_priority)
invalid_pri_count = invalid_pri_mask.sum()

print(f"\nValid priority levels  : {valid_priority}")
print(f"Unique values found    : {sorted(df['priority'].unique().tolist())}")
print(f"Invalid priority rows  : {invalid_pri_count}")

if invalid_pri_count > 0:
    print(f"  Removing {invalid_pri_count} rows with invalid priority values")
    df = df[~invalid_pri_mask].reset_index(drop=True)

print(f"\nPriority distribution after validation:")
print(df["priority"].value_counts())

print("Step 8: Cleaning Numerical columns:")

numerical_cols = ["time_to_detect_sec", "time_to_fix_sec", "user_impact_score"]

for col in numerical_cols:
    if col in df.columns:

        df[col] = pd.to_numeric(df[col], errors="coerce")

        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"  '{col}': {neg_count} negative values → set to NaN")
            df.loc[df[col] < 0, col] = np.nan

        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  '{col}': {nan_count} NaN values filled with median ({median_val:.2f})")

        print(f"  '{col}' → min: {df[col].min():.2f} | "
              f"max: {df[col].max():.2f} | "
              f"mean: {df[col].mean():.2f} | "
              f"nulls remaining: {df[col].isnull().sum()}")

print("Step 9: Cleaning the boolean columns:")

bool_cols = ["screenshot_available", "resolved"]

for col in bool_cols:
    if col in df.columns:
        print(f"  '{col}' raw unique values: {df[col].unique().tolist()}")
        df[col] = (df[col].astype(str)
                          .str.strip()
                          .str.title()
                          .map({
                              "True":  True,  "False": False,
                              "Yes":   True,  "No":    False,
                              "1":     True,  "0":     False
                          }))
        print(f"  '{col}' after cleaning: {df[col].value_counts().to_dict()}")
        print(f"  Nulls after mapping  : {df[col].isnull().sum()}")

print("Step 10: Cleaning tag columns")


if "tags" in df.columns:
    df["tags"] = df["tags"].astype(str).str.strip()
    df["tags"] = df["tags"].replace("nan",  "")
    df["tags"] = df["tags"].replace("None", "")
    df["tags"] = df["tags"].replace("NaN",  "")
    print(f"  Tags column cleaned.")
    print(f"  Sample values : {df['tags'].head(10).tolist()}")
    print(f"  Empty tags    : {(df['tags'] == '').sum()}")

print("Step 11: Dropping irrelevant columns:")

cols_to_drop  = ["page_url"]
cols_actually_dropped = [c for c in cols_to_drop if c in df.columns]

df.drop(columns=cols_actually_dropped, inplace=True)
print(f"  Dropped columns : {cols_actually_dropped}")
print(f"  Columns remaining ({len(df.columns)}): {list(df.columns)}")


print("Step 12: Resetting index")
print("=" * 60)

df.reset_index(drop=True, inplace=True)
print(f"  Index has been reset now. Final dataset shape: {df.shape}")

print("Step 13: Final cleaning summary report:")

print(f"  Total records        : {len(df)}")
print(f"  Total columns        : {len(df.columns)}")
print(f"  Missing values left  : {df.isnull().sum().sum()}")
print(f"  Duplicate rows left  : {df.duplicated().sum()}")
print(f"\n  Column names & types:")
print(df.dtypes)
print(f"\n  Sample of cleaned data (first 3 rows):")
print(df.head(3).to_string())

print("Step 14: Generating Visualizations:")

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 5))
order = [s for s in ["Critical", "High", "Medium", "Low"]
         if s in df["severity"].unique()]
sns.countplot(data=df, x="severity", order=order, palette="Set2")
plt.title("Bug Severity Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Severity Level")
plt.ylabel("Number of Bugs")
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("plots/01_severity_distribution.png", dpi=150)
plt.close()
print("  Saved: plots/01_severity_distribution.png")

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="priority",
              order=["P1", "P2", "P3"], palette="Set1")
plt.title("Bug Priority Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Priority Level")
plt.ylabel("Number of Bugs")
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("plots/02_priority_distribution.png", dpi=150)
plt.close()
print("  Saved: plots/02_priority_distribution.png")

plt.figure(figsize=(13, 5))
bug_counts = df["bug_type"].value_counts()
bug_counts.plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Bug Type Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Bug Type")
plt.ylabel("Count")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig("plots/03_bug_type_distribution.png", dpi=150)
plt.close()
print("  Saved: plots/03_bug_type_distribution.png")

plt.figure(figsize=(7, 7))
df["device_type"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    colors=["#66b3ff", "#99ff99", "#ffcc99"],
    startangle=90,
    textprops={"fontsize": 12}
)
plt.title("Bugs by Device Type", fontsize=14, fontweight="bold")
plt.ylabel("")
plt.tight_layout()
plt.savefig("plots/04_device_type_distribution.png", dpi=150)
plt.close()
print("  Saved: plots/04_device_type_distribution.png")

plt.figure(figsize=(8, 5))
pivot = pd.crosstab(df["severity"], df["priority"])
row_order = [r for r in ["Critical", "High", "Medium", "Low"]
             if r in pivot.index]
pivot = pivot.reindex(row_order)
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5)
plt.title("Severity vs Priority Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/05_severity_vs_priority_heatmap.png", dpi=150)
plt.close()
print("  Saved: plots/05_severity_vs_priority_heatmap.png")

plt.figure(figsize=(9, 5))
df["browser"].value_counts().plot(
    kind="bar", color="coral", edgecolor="black"
)
plt.title("Bugs Reported by Browser", fontsize=14, fontweight="bold")
plt.xlabel("Browser")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/06_browser_distribution.png", dpi=150)
plt.close()
print("  Saved: plots/06_browser_distribution.png")

plt.figure(figsize=(9, 5))
sns.histplot(df["user_impact_score"], bins=10, kde=True, color="purple")
plt.title("User Impact Score Distribution", fontsize=14, fontweight="bold")
plt.xlabel("User Impact Score (1–10)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/07_user_impact_score_distribution.png", dpi=150)
plt.close()
print("  Saved: plots/07_user_impact_score_distribution.png")

plt.figure(figsize=(9, 5))
df["os"].value_counts().plot(kind="bar", color="mediumseagreen", edgecolor="black")
plt.title("Bugs by Operating System", fontsize=14, fontweight="bold")
plt.xlabel("Operating System")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/08_os_distribution.png", dpi=150)
plt.close()
print("  Saved: plots/08_os_distribution.png")

plt.figure(figsize=(6, 6))
df["resolved"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    labels=["Unresolved", "Resolved"],
    colors=["#ff9999", "#99ff99"],
    startangle=90,
    textprops={"fontsize": 12}
)
plt.title("Resolved vs Unresolved Bugs", fontsize=14, fontweight="bold")
plt.ylabel("")
plt.tight_layout()
plt.savefig("plots/09_resolved_distribution.png", dpi=150)
plt.close()
print("  Saved: plots/09_resolved_distribution.png")

plt.figure(figsize=(10, 5))
df["module"].value_counts().head(10).plot(
    kind="barh", color="mediumpurple", edgecolor="black"
)
plt.title("Top 10 Modules by Bug Count", fontsize=14, fontweight="bold")
plt.xlabel("Number of Bugs")
plt.ylabel("Module")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plots/10_top_modules.png", dpi=150)
plt.close()
print("  Saved: plots/10_top_modules.png")

# Step 15: Here we are finally saving the cleaned dataset as a new csv file

print("\n" + "=" * 60)
print("Step 15: Saving cleaned dataset:")
print("=" * 60)

output_file = "frontend_uiux_bug_dataset_cleaned.csv"
df.to_csv(output_file, index=False)

size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"  Saved as  : {output_file}")
print(f"  File size : {size_mb:.2f} MB")
print(f"  Total rows: {len(df)}")
print(f"  Total cols: {len(df.columns)}")

print("\n" + "=" * 60)
print("All the steps for data cleaning have been performed by our code")
print("=" * 60)