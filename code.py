# Task 16: Hyperparameter Tuning using GridSearchCV (FIXED VERSION)

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Default Model
# -------------------------------
default_model = RandomForestClassifier(random_state=42)
default_model.fit(X_train, y_train)

default_preds = default_model.predict(X_test)
default_accuracy = accuracy_score(y_test, default_preds)

# GridSearchCV (SAFE CONFIG)
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=1,        
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Tuned model evaluation
tuned_preds = best_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test, tuned_preds)

# Results
print("Default Model Accuracy:", round(default_accuracy, 4))
print("Tuned Model Accuracy:", round(tuned_accuracy, 4))

print("\nBest Hyperparameters:")
print(grid_search.best_params_)

comparison_df = pd.DataFrame({
    "Model": ["Default Random Forest", "Tuned Random Forest"],
    "Accuracy": [default_accuracy, tuned_accuracy]
})

print("\nPerformance Comparison:")
print(comparison_df)

print("\nClassification Report (Tuned Model):")
print(classification_report(y_test, tuned_preds))
