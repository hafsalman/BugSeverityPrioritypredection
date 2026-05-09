# DS4SE Project – Bug Severity & Priority Predictor
# Group Leader: Hira Zahoor (22K-4823)
# Members:   Hafsa Salman (22K-5161)
#            Jaysha Iqbal (22K-5178)
#            Amna Mansoor (22K-5159)
#
# USAGE:
#   1. Run model_training.py first — it saves:
#        severity_rf_model.joblib
#        priority_rf_model.joblib
#        data_encoders.joblib
#   2. Then run:  python predict_bug.py

import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Always look for model files next to this script,
# regardless of which directory the terminal is in.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Valid options (must match what the dataset was trained on) ──────
VALID_OPTIONS = {
    "app_name":    ["Shopease", "Financetracker", "Healthportal", "Edulearn", "Taskmaster"],
    "module":      ["Navbar", "Footer", "Login Form", "Dashboard", "Sidebar",
                    "Editor", "Toolbar", "Checkout", "Search"],
    "bug_type":    ["Layout", "Text Overflow", "Color Mismatch", "Broken Link",
                    "Responsiveness", "Animation", "Accessibility", "Performance", "Crash"],
    "device_type": ["Desktop", "Mobile", "Tablet"],
    "browser":     ["Chrome", "Firefox", "Safari", "Edge"],
    "os":          ["Windows", "Macos", "Ios", "Android"],
    "reported_by": ["Qa", "Developer", "User"],
}

def show_options(field):
    """Print the numbered list of valid choices for a field."""
    opts = VALID_OPTIONS[field]
    print(f"  Options: {', '.join(opts)}")

def ask_choice(prompt, field):
    """
    Ask the user to type a value for a categorical field.
    If the value isn't in the valid list, default to the first option and warn.
    """
    show_options(field)
    raw = input(f"  → {prompt}: ").strip().title()
    opts = VALID_OPTIONS[field]
    if raw in opts:
        return raw
    # Try case-insensitive match
    lower_map = {o.lower(): o for o in opts}
    if raw.lower() in lower_map:
        return lower_map[raw.lower()]
    print(f"  [!] '{raw}' not recognised. Defaulting to '{opts[0]}'.")
    return opts[0]

def ask_int(prompt, lo, hi, default):
    """Ask for an integer in [lo, hi]. Use default on bad input."""
    try:
        val = int(input(f"  → {prompt} ({lo}–{hi}): ").strip())
        if lo <= val <= hi:
            return val
        print(f"  [!] Out of range. Using default ({default}).")
        return default
    except ValueError:
        print(f"  [!] Not a number. Using default ({default}).")
        return default


def main():
    print("\n" + "=" * 60)
    print("  AI Frontend UI/UX Bug Predictor")
    print("  Predicts: Severity  &  Priority")
    print("=" * 60)

    # ── Load models and encoders ────────────────────────────────────
    try:
        model_severity = joblib.load(os.path.join(SCRIPT_DIR, "severity_rf_model.joblib"))
        model_priority = joblib.load(os.path.join(SCRIPT_DIR, "priority_rf_model.joblib"))
        encoders       = joblib.load(os.path.join(SCRIPT_DIR, "data_encoders.joblib"))
    except FileNotFoundError as e:
        print(f"\n  Error: {e}")
        print("  Please run model_training.py first to generate the model files.")
        return

    feature_encoders = encoders["features"]     # dict {col: LabelEncoder}
    feature_cols     = encoders["feature_cols"] # ordered list of feature names
    sev_map          = encoders["sev_map"]      # {"Low":0, "Medium":1, ...}
    pri_map          = encoders["pri_map"]      # {"P3":0, "P2":1, "P1":2}

    # Invert the maps so we can decode integer predictions → label strings
    inv_sev = {v: k for k, v in sev_map.items()}   # {0:"Low", 1:"Medium", ...}
    inv_pri = {v: k for k, v in pri_map.items()}   # {0:"P3", 1:"P2", 2:"P1"}

    # ── Collect user input ──────────────────────────────────────────
    print("\nEnter the details of the new bug:")
    print("(Type the value and press Enter. Invalid inputs default to the first option.)\n")

    app_name    = ask_choice("App Name",    "app_name")
    module      = ask_choice("Module",      "module")
    bug_type    = ask_choice("Bug Type",    "bug_type")
    device_type = ask_choice("Device Type", "device_type")
    browser     = ask_choice("Browser",     "browser")
    os_name     = ask_choice("OS",          "os")
    reported_by = ask_choice("Reported By", "reported_by")

    print("\n  Screenshot Available? (1 = Yes, 0 = No)")
    screenshot = ask_int("Screenshot available", 0, 1, 1)

    print()
    time_to_detect = ask_int("Time to detect (seconds)",      5,   60,  30)
    time_to_fix    = ask_int("Estimated time to fix (seconds)", 60, 1000, 400)
    user_impact    = ask_int("User impact score",              1,   10,   5)

    # resolved is always False for a new / just-reported bug
    resolved = 0

    # ── Build the raw input row ─────────────────────────────────────
    raw = {
        "app_name":             app_name,
        "module":               module,
        "bug_type":             bug_type,
        "device_type":          device_type,
        "browser":              browser,
        "os":                   os_name,
        "screenshot_available": screenshot,
        "time_to_detect_sec":   time_to_detect,
        "time_to_fix_sec":      time_to_fix,
        "user_impact_score":    user_impact,
        "reported_by":          reported_by,
        "resolved":             resolved,
    }
    input_df = pd.DataFrame([raw])

    # ── Encode categorical columns with the fitted LabelEncoders ────
    for col, le in feature_encoders.items():
        if col not in input_df.columns:
            continue
        val = input_df[col].astype(str).iloc[0]
        if val in le.classes_:
            input_df[col] = le.transform([val])
        else:
            # Unseen label → fall back to the first known class
            input_df[col] = le.transform([le.classes_[0]])

    # Reorder columns to exactly match training order
    input_df = input_df[feature_cols]

    # ── Predict ─────────────────────────────────────────────────────
    pred_sev_int = model_severity.predict(input_df)[0]
    pred_pri_int = model_priority.predict(input_df)[0]

    severity_label = inv_sev[pred_sev_int]
    priority_label = inv_pri[pred_pri_int]

    # Severity → colour hint for readability
    sev_colour = {
        "Critical": "🔴 CRITICAL",
        "High":     "🟠 HIGH",
        "Medium":   "🟡 MEDIUM",
        "Low":      "🟢 LOW",
    }.get(severity_label, severity_label)

    pri_colour = {
        "P1": "🔴 P1 (Urgent)",
        "P2": "🟠 P2 (Normal)",
        "P3": "🟢 P3 (Low)",
    }.get(priority_label, priority_label)

    # ── Print results ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  AI PREDICTION RESULTS")
    print("=" * 60)
    print(f"  App / Module  : {app_name}  →  {module}")
    print(f"  Bug Type      : {bug_type}")
    print(f"  Device / OS   : {device_type} / {os_name}")
    print(f"  User Impact   : {user_impact}/10   |   Fix Time: {time_to_fix}s")
    print("-" * 60)
    print(f"  Predicted Severity : {sev_colour}")
    print(f"  Predicted Priority : {pri_colour}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()