# Machine Learning Project - AIN3101
# Task: Classification
# Datasets: Raisin (UCI id 850) and Dry Bean (UCI id 602)
# Methods: Logistic Regression, SVM, Random Forest

import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)

os.makedirs("results", exist_ok=True)

datasets = {
    "Raisin": "data/Raisin_Dataset.xlsx",
    "DryBean": "data/Dry_Bean_Dataset.xlsx",
}

# the 3 models and the hyper-parameters we tune for each one
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=5000),
        {"model__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "SVM": (
        SVC(),
        {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20]},
    ),
}

all_results = []

for data_name, path in datasets.items():
    print("\n==============================")
    print("Dataset:", data_name)
    print("==============================")

    # 1) load data
    df = pd.read_excel(path)
    print("Shape:", df.shape)

    # 2) cleaning: check missing values and remove duplicate rows
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())
    df = df.drop_duplicates()
    print("Shape after cleaning:", df.shape)

    # 3) features and target (target is the "Class" column)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    print("Classes:", y.value_counts().to_dict())

    # class distribution figure
    y.value_counts().plot(kind="bar", color="steelblue", figsize=(7, 4))
    plt.title(f"Class distribution - {data_name}")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"results/classes_{data_name}.png")
    plt.close()

    # 4) encode the class names to numbers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # 5) same train/test split for all models (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print("Train size:", len(X_train), " Test size:", len(X_test))

    results = []

    for model_name, (model, params) in models.items():
        # pipeline: fill missing values -> scale -> model
        # (the imputer and scaler learn only from the training data)
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        # 6) tune hyper-parameters with 5-fold cross validation on train only
        grid = GridSearchCV(pipe, params, cv=5, scoring="f1_macro", n_jobs=-1)
        start = time.time()
        grid.fit(X_train, y_train)
        tune_time = time.time() - start

        # 7) evaluate on the test set
        y_pred = grid.predict(X_test)
        row = {
            "Dataset": data_name,
            "Model": model_name,
            "Best Params": str(grid.best_params_).replace("model__", ""),
            "CV F1": round(grid.best_score_, 4),
            "Train Accuracy": round(accuracy_score(y_train, grid.predict(X_train)), 4),
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average="macro"), 4),
            "Recall": round(recall_score(y_test, y_pred, average="macro"), 4),
            "F1": round(f1_score(y_test, y_pred, average="macro"), 4),
            "Tuning Time (s)": round(tune_time, 1),
        }
        results.append(row)
        print(f"\n{model_name}: best params = {row['Best Params']}")
        print(f"  CV F1={row['CV F1']}  Train Acc={row['Train Accuracy']}")
        print(f"  Test: Accuracy={row['Accuracy']}  Precision={row['Precision']}  "
              f"Recall={row['Recall']}  F1={row['F1']}")

        # confusion matrix figure
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=encoder.classes_,
            cmap="Blues", xticks_rotation=45)
        plt.title(f"{data_name} - {model_name}")
        plt.tight_layout()
        plt.savefig(f"results/cm_{data_name}_{model_name.replace(' ', '_')}.png")
        plt.close()

    # 8) comparison table for this dataset
    table = pd.DataFrame(results)
    print("\nComparison table -", data_name)
    print(table[["Model", "Accuracy", "Precision", "Recall", "F1"]].to_string(index=False))
    table.to_csv(f"results/table_{data_name}.csv", index=False)
    all_results += results

    # 9) comparison figure for this dataset
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    table.set_index("Model")[metrics].plot(kind="bar", figsize=(8, 5))
    plt.title(f"Model comparison - {data_name}")
    plt.ylabel("Score")
    plt.ylim(table[metrics].min().min() - 0.05, 1)
    plt.xticks(rotation=0)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"results/compare_{data_name}.png")
    plt.close()

pd.DataFrame(all_results).to_csv("results/all_results.csv", index=False)
print("\nDone. All tables and figures are in the results folder.")
