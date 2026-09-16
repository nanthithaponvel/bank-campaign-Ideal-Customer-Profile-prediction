import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import joblib

from semantic_schema import (
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


# ============================================================
# BANK MARKETING PROJECT - MODEL TRAINING
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_FILE = PROJECT_DIR / "data" / "train.csv"
TEST_FILE = PROJECT_DIR / "data" / "test.csv"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


print("=" * 75)
print("BANK MARKETING PROJECT - MODEL TRAINING")
print("=" * 75)


# ============================================================
# 1. CHECK FILES
# ============================================================

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}\n\n"
        "Run split.py first."
    )

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Testing file not found:\n{TEST_FILE}\n\n"
        "Run split.py first."
    )


# ============================================================
# 2. LOAD DATA
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE,
    sep=";"
)

test_df = pd.read_csv(
    TEST_FILE,
    sep=";"
)


X_train = train_df[MODEL_FEATURES].copy()
X_test = test_df[MODEL_FEATURES].copy()

y_train_raw = train_df["y"].copy()
y_test_raw = test_df["y"].copy()


print("\n1. DATA LOADED")
print("-" * 75)

print(
    f"Training observations: {len(X_train)}"
)

print(
    f"Testing observations : {len(X_test)}"
)

print(
    f"Model features       : {len(MODEL_FEATURES)}"
)


# ============================================================
# 3. CHECK ORIGINAL TARGET VALUES
# ============================================================

print("\n2. ORIGINAL TARGET VALUES")
print("-" * 75)

print(
    "Training target values:"
)

print(
    y_train_raw.value_counts(dropna=False)
)

print(
    "\nTesting target values:"
)

print(
    y_test_raw.value_counts(dropna=False)
)


# ============================================================
# 4. ROBUST TARGET ENCODING
# ============================================================

def encode_target(series):

    # Convert to string first so both numeric
    # and text targets can be handled safely.
    cleaned = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # yes/no format
    if set(cleaned.unique()).issubset(
        {"yes", "no"}
    ):

        return cleaned.map({
            "no": 0,
            "yes": 1
        })

    # 0/1 format
    if set(cleaned.unique()).issubset(
        {"0", "1"}
    ):

        return cleaned.map({
            "0": 0,
            "1": 1
        })

    # Sometimes datasets may contain
    # True/False
    if set(cleaned.unique()).issubset(
        {"true", "false"}
    ):

        return cleaned.map({
            "false": 0,
            "true": 1
        })

    # If we reach here, stop and show
    # the actual unexpected values.
    raise ValueError(
        "Unexpected target values found: "
        f"{sorted(cleaned.unique().tolist())}"
    )


y_train = encode_target(
    y_train_raw
)

y_test = encode_target(
    y_test_raw
)


if y_train.isna().any():
    raise ValueError(
        "Missing values found in training target."
    )

if y_test.isna().any():
    raise ValueError(
        "Missing values found in testing target."
    )


print("\n3. TARGET ENCODING")
print("-" * 75)

print(
    "[OK] Target successfully encoded."
)

print(
    "no  -> 0"
)

print(
    "yes -> 1"
)

print(
    "\nTraining target distribution:"
)

print(
    y_train.value_counts()
)

print(
    "\nTesting target distribution:"
)

print(
    y_test.value_counts()
)


# ============================================================
# 5. DATA LEAKAGE CHECK
# ============================================================

print("\n4. DATA LEAKAGE CHECK")
print("-" * 75)

if "duration" in X_train.columns:

    raise ValueError(
        "ERROR: duration is present in training features."
    )

if "duration" in X_test.columns:

    raise ValueError(
        "ERROR: duration is present in testing features."
    )

print(
    "[OK] duration is excluded from the model."
)


# ============================================================
# 6. FEATURE GROUP CHECK
# ============================================================

print("\n5. FEATURE GROUP CHECK")
print("-" * 75)

missing_numeric = [
    feature
    for feature in NUMERICAL_FEATURES
    if feature not in X_train.columns
]

missing_categorical = [
    feature
    for feature in CATEGORICAL_FEATURES
    if feature not in X_train.columns
]


if missing_numeric:

    raise ValueError(
        f"Missing numerical features: "
        f"{missing_numeric}"
    )


if missing_categorical:

    raise ValueError(
        f"Missing categorical features: "
        f"{missing_categorical}"
    )


print(
    f"[OK] Numerical features: "
    f"{len(NUMERICAL_FEATURES)}"
)

print(
    f"[OK] Categorical features: "
    f"{len(CATEGORICAL_FEATURES)}"
)


# ============================================================
# 7. NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 8. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# 9. COMBINED PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            numeric_pipeline,
            NUMERICAL_FEATURES
        ),

        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ]
)


print("\n6. PREPROCESSING PIPELINE")
print("-" * 75)

print(
    "[OK] Numerical missing values -> median"
)

print(
    "[OK] Numerical features -> StandardScaler"
)

print(
    "[OK] Categorical missing values -> most frequent"
)

print(
    "[OK] Categorical features -> OneHotEncoder"
)

print(
    "[OK] Unknown categories -> ignored safely"
)


# ============================================================
# 10. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ============================================================
# 11. TRAIN MODELS
# ============================================================

results = []

trained_models = {}


for model_name, model in models.items():

    print("\n" + "=" * 75)

    print(
        f"TRAINING: {model_name}"
    )

    print("=" * 75)

    pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )


    print(
        "Fitting model..."
    )


    pipeline.fit(
        X_train,
        y_train
    )


    print(
        "Generating predictions..."
    )


    y_pred = pipeline.predict(
        X_test
    )


    y_probability = (
        pipeline
        .predict_proba(X_test)[:, 1]
    )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": roc_auc
    })


    trained_models[
        model_name
    ] = pipeline


    print("\nMODEL PERFORMANCE")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by="ROC_AUC",
    ascending=False
)


print("\n" + "=" * 75)

print(
    "MODEL COMPARISON"
)

print("=" * 75)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 13. SELECT BEST MODEL
# ============================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)


best_model = trained_models[
    best_model_name
]


best_row = results_df.iloc[0]


print("\n" + "=" * 75)

print(
    "BEST MODEL"
)

print("=" * 75)


print(
    f"Selected model: "
    f"{best_model_name}"
)


print(
    f"ROC-AUC: "
    f"{best_row['ROC_AUC']:.4f}"
)


print(
    f"Precision: "
    f"{best_row['Precision']:.4f}"
)


print(
    f"Recall: "
    f"{best_row['Recall']:.4f}"
)


print(
    f"F1-score: "
    f"{best_row['F1']:.4f}"
)


# ============================================================
# 14. SAVE MODEL COMPARISON
# ============================================================

results_file = (
    MODEL_DIR /
    "model_comparison.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# 15. SAVE BEST MODEL
# ============================================================

model_file = (
    MODEL_DIR /
    "best_model.joblib"
)


joblib.dump(
    best_model,
    model_file
)


# ============================================================
# 16. SAVE BEST MODEL NAME
# ============================================================

model_name_file = (
    MODEL_DIR /
    "best_model_name.txt"
)


with open(
    model_name_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        best_model_name
    )


# ============================================================
# 17. SAVE MODEL FEATURES
# ============================================================

feature_file = (
    MODEL_DIR /
    "model_features.txt"
)


with open(
    feature_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "MODEL FEATURES\n"
    )

    file.write(
        "===============\n\n"
    )

    for feature in MODEL_FEATURES:

        file.write(
            f"{feature}\n"
        )


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)

print(
    "MODEL TRAINING COMPLETE"
)

print("=" * 75)


print("\nModels trained:")

for model_name in models:

    print(
        f"- {model_name}"
    )


print(
    f"\nSelected model: "
    f"{best_model_name}"
)


print("\nSaved files:")

print(
    "- models/model_comparison.csv"
)

print(
    "- models/best_model.joblib"
)

print(
    "- models/best_model_name.txt"
)

print(
    "- models/model_features.txt"
)


print("\nNext stage:")

print(
    "Detailed model evaluation and "
    "customer-level prediction."
)


print(
    "\n" + "=" * 75
)