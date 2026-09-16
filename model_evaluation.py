import pandas as pd
import numpy as np
from pathlib import Path

import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)


# ============================================================
# BANK MARKETING PROJECT
# DETAILED MODEL EVALUATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_FILE = PROJECT_DIR / "data" / "train.csv"
TEST_FILE = PROJECT_DIR / "data" / "test.csv"

MODEL_FILE = PROJECT_DIR / "models" / "best_model.joblib"
MODEL_RESULTS_FILE = PROJECT_DIR / "models" / "model_comparison.csv"

OUTPUT_DIR = PROJECT_DIR / "eda_outputs"
MODEL_DIR = PROJECT_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


print("=" * 75)
print("BANK MARKETING PROJECT - DETAILED MODEL EVALUATION")
print("=" * 75)


# ============================================================
# 1. CHECK REQUIRED FILES
# ============================================================

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test file not found:\n{TEST_FILE}\n\n"
        "Run split.py first."
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Best model not found:\n{MODEL_FILE}\n\n"
        "Run train_models.py first."
    )


# ============================================================
# 2. LOAD DATA AND MODEL
# ============================================================

test_df = pd.read_csv(
    TEST_FILE,
    sep=";"
)

model = joblib.load(
    MODEL_FILE
)


print("\n1. MODEL AND TEST DATA LOADED")
print("-" * 75)

print(
    f"Test observations: {len(test_df)}"
)

print(
    f"Model file: {MODEL_FILE.name}"
)


# ============================================================
# 3. LOAD MODEL FEATURES
# ============================================================

features_file = (
    MODEL_DIR /
    "model_features.txt"
)

if not features_file.exists():
    raise FileNotFoundError(
        "model_features.txt not found."
    )


with open(
    features_file,
    "r",
    encoding="utf-8"
) as file:

    lines = file.readlines()


MODEL_FEATURES = []

for line in lines:

    line = line.strip()

    if (
        line
        and line != "MODEL FEATURES"
        and line != "==============="
    ):
        MODEL_FEATURES.append(line)


print("\n2. MODEL FEATURES")
print("-" * 75)

print(
    f"Number of features: "
    f"{len(MODEL_FEATURES)}"
)

for feature in MODEL_FEATURES:
    print(f"- {feature}")


# ============================================================
# 4. PREPARE TEST DATA
# ============================================================

X_test = test_df[
    MODEL_FEATURES
].copy()


# Robust target conversion
y_test_raw = (
    test_df["y"]
    .astype(str)
    .str.strip()
    .str.lower()
)


if set(y_test_raw.unique()).issubset(
    {"yes", "no"}
):

    y_test = y_test_raw.map({
        "no": 0,
        "yes": 1
    })


elif set(y_test_raw.unique()).issubset(
    {"0", "1"}
):

    y_test = y_test_raw.map({
        "0": 0,
        "1": 1
    })


else:

    raise ValueError(
        "Unexpected target values: "
        f"{sorted(y_test_raw.unique())}"
    )


print("\n3. TEST DATA PREPARED")
print("-" * 75)

print(
    f"X_test shape: {X_test.shape}"
)

print(
    f"y_test shape: {y_test.shape}"
)


# ============================================================
# 5. DATA LEAKAGE CHECK
# ============================================================

print("\n4. DATA LEAKAGE CHECK")
print("-" * 75)

if "duration" in X_test.columns:

    raise ValueError(
        "ERROR: duration is present."
    )

print(
    "[OK] duration is excluded."
)


# ============================================================
# 6. GENERATE PREDICTIONS
# ============================================================

print("\n5. GENERATING PREDICTIONS")
print("-" * 75)

y_pred = model.predict(
    X_test
)

y_probability = (
    model
    .predict_proba(X_test)[:, 1]
)


print(
    "[OK] Class predictions generated."
)

print(
    "[OK] Subscription probabilities generated."
)


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)

print(
    "CLASSIFICATION REPORT"
)

print("=" * 75)

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "No Subscription",
        "Subscription"
    ],
    digits=4
)

print(report)


# Save classification report
with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "BANK MARKETING PROJECT\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        "=====================\n\n"
    )

    file.write(report)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("=" * 75)

print(
    "CONFUSION MATRIX"
)

print("=" * 75)

print(cm)


tn, fp, fn, tp = cm.ravel()


print("\nInterpretation:")

print(
    f"True Negatives : {tn}"
)

print(
    f"False Positives: {fp}"
)

print(
    f"False Negatives: {fn}"
)

print(
    f"True Positives  : {tp}"
)


# Plot confusion matrix
fig, ax = plt.subplots(
    figsize=(7, 6)
)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No",
        "Yes"
    ]
).plot(
    ax=ax
)

ax.set_title(
    "Random Forest - Confusion Matrix"
)

plt.tight_layout()

confusion_file = (
    OUTPUT_DIR /
    "confusion_matrix.png"
)

plt.savefig(
    confusion_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

roc_auc = auc(
    fpr,
    tpr
)


print("\n" + "=" * 75)

print(
    "ROC-AUC"
)

print("=" * 75)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Random Forest - ROC Curve"
)

plt.legend()

plt.tight_layout()

roc_file = (
    OUTPUT_DIR /
    "roc_curve.png"
)

plt.savefig(
    roc_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. PRECISION-RECALL CURVE
# ============================================================

precision, recall, pr_thresholds = (
    precision_recall_curve(
        y_test,
        y_probability
    )
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall,
    precision
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Random Forest - Precision-Recall Curve"
)

plt.tight_layout()

pr_file = (
    OUTPUT_DIR /
    "precision_recall_curve.png"
)

plt.savefig(
    pr_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 75)

print(
    "FEATURE IMPORTANCE"
)

print("=" * 75)


preprocessor = (
    model
    .named_steps["preprocessor"]
)

trained_model = (
    model
    .named_steps["model"]
)


# Get transformed feature names
feature_names = (
    preprocessor
    .get_feature_names_out()
)


if hasattr(
    trained_model,
    "feature_importances_"
):

    importances = (
        trained_model
        .feature_importances_
    )

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importances

    })


    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )


    print(
        importance_df.head(20)
        .to_string(
            index=False
        )
    )


    importance_file = (
        MODEL_DIR /
        "feature_importance.csv"
    )

    importance_df.to_csv(
        importance_file,
        index=False
    )


    # Top 15 feature importance chart

    top_features = (
        importance_df
        .head(15)
        .sort_values(
            "Importance"
        )
    )


    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Random Forest - Top 15 Feature Importance"
    )

    plt.tight_layout()


    importance_plot = (
        OUTPUT_DIR /
        "feature_importance.png"
    )

    plt.savefig(
        importance_plot,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


else:

    print(
        "[INFO] Feature importance is not available "
        "for this model."
    )


# ============================================================
# 12. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = test_df.copy()


prediction_df[
    "predicted_class"
] = y_pred


prediction_df[
    "subscription_probability"
] = y_probability


prediction_df[
    "predicted_subscription"
] = (
    prediction_df["predicted_class"]
    .map({
        0: "no",
        1: "yes"
    })
)


prediction_file = (
    MODEL_DIR /
    "test_predictions.csv"
)


prediction_df.to_csv(
    prediction_file,
    index=False
)


print("\n" + "=" * 75)

print(
    "TEST PREDICTIONS SAVED"
)

print("=" * 75)

print(
    f"File: {prediction_file}"
)


# ============================================================
# 13. BUSINESS INTERPRETATION
# ============================================================

print("\n" + "=" * 75)

print(
    "BUSINESS INTERPRETATION"
)

print("=" * 75)


print(
    "\nTrue Positive:"
)

print(
    "Customer predicted as likely to subscribe "
    "and actually subscribed."
)


print(
    "\nFalse Positive:"
)

print(
    "Customer predicted as likely to subscribe "
    "but did not subscribe."
)


print(
    "\nFalse Negative:"
)

print(
    "Customer predicted as unlikely to subscribe "
    "but actually subscribed."
)


print(
    "\nTrue Negative:"
)

print(
    "Customer predicted as unlikely to subscribe "
    "and did not subscribe."
)


print(
    "\nFor a future campaign:"
)

print(
    "The subscription probability can be used "
    "to prioritize customers."
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)

print(
    "MODEL EVALUATION COMPLETE"
)

print("=" * 75)


print("\nGenerated files:")

print(
    "- eda_outputs/classification_report.txt"
)

print(
    "- eda_outputs/confusion_matrix.png"
)

print(
    "- eda_outputs/roc_curve.png"
)

print(
    "- eda_outputs/precision_recall_curve.png"
)

print(
    "- eda_outputs/feature_importance.png"
)

print(
    "- models/feature_importance.csv"
)

print(
    "- models/test_predictions.csv"
)


print("\nNext stage:")

print(
    "Build the customer-level prediction system."
)


print(
    "\n" + "=" * 75
)
