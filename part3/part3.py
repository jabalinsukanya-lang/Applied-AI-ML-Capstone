import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

print("=" * 70)
print("PART 3 - DECISION TREE MODELS")
print("=" * 70)
#Load Dataset
df=pd.read_csv("../part1/cleaned_data.csv")
print("\nDataset Shape")
print(df.shape)
# Targets
y_clf = df["Churn"].map({
    "No":0,
    "Yes":1
})
X = df.drop(
    columns=[
        "customerID",
        "TotalCharges",
        "Churn"
    ]
)
# Label Encoding (Ordered)
contract_map = {
    "Month-to-month":0,
    "One year":1,
    "Two year":2
}

X["Contract"] = X["Contract"].map(contract_map)

# One Hot Encoding
categorical = X.select_dtypes(include="object").columns

X = pd.get_dummies(
    X,
    columns=categorical,
    drop_first=True
)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_clf,

    test_size=0.20,

    random_state=42,

    stratify=y_clf

)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# DECISION TREE (BASELINE)
print("\n")
print("=" * 70)
print("BASELINE DECISION TREE")
print("=" * 70)

tree_default = DecisionTreeClassifier(
    random_state=42
)

tree_default.fit(
    X_train_scaled,
    y_train
)

train_pred = tree_default.predict(
    X_train_scaled
)

test_pred = tree_default.predict(
    X_test_scaled
)

train_prob = tree_default.predict_proba(
    X_train_scaled
)[:,1]

test_prob = tree_default.predict_proba(
    X_test_scaled
)[:,1]

train_acc = accuracy_score(
    y_train,
    train_pred
)

test_acc = accuracy_score(
    y_test,
    test_pred
)

auc_default = roc_auc_score(
    y_test,
    test_prob
)

print("\nTraining Accuracy :", round(train_acc,4))
print("Test Accuracy     :", round(test_acc,4))
print("ROC AUC           :", round(auc_default,4))

# CONTROLLED TREE
print("\n")
print("=" * 70)
print("CONTROLLED DECISION TREE")
print("=" * 70)

tree_control = DecisionTreeClassifier(

    max_depth=5,

    min_samples_split=20,

    random_state=42

)

tree_control.fit(
    X_train_scaled,
    y_train
)

train_pred2 = tree_control.predict(
    X_train_scaled
)

test_pred2 = tree_control.predict(
    X_test_scaled
)

test_prob2 = tree_control.predict_proba(
    X_test_scaled
)[:,1]

train_acc2 = accuracy_score(
    y_train,
    train_pred2
)

test_acc2 = accuracy_score(
    y_test,
    test_pred2
)

auc_control = roc_auc_score(
    y_test,
    test_prob2
)

print("\nTraining Accuracy :", round(train_acc2,4))
print("Test Accuracy     :", round(test_acc2,4))
print("ROC AUC           :", round(auc_control,4))

# GINI VS ENTROPY
print("\n")
print("=" * 70)
print("GINI vs ENTROPY")
print("=" * 70)

tree_gini = DecisionTreeClassifier(

    criterion="gini",

    max_depth=5,

    random_state=42

)

tree_entropy = DecisionTreeClassifier(

    criterion="entropy",

    max_depth=5,

    random_state=42

)

tree_gini.fit(
    X_train_scaled,
    y_train
)

tree_entropy.fit(
    X_train_scaled,
    y_train
)

gini_pred = tree_gini.predict(
    X_test_scaled
)

entropy_pred = tree_entropy.predict(
    X_test_scaled
)

gini_acc = accuracy_score(
    y_test,
    gini_pred
)

entropy_acc = accuracy_score(
    y_test,
    entropy_pred
)

gini_auc = roc_auc_score(
    y_test,
    tree_gini.predict_proba(X_test_scaled)[:,1]
)

entropy_auc = roc_auc_score(
    y_test,
    tree_entropy.predict_proba(X_test_scaled)[:,1]
)

comparison = pd.DataFrame({

    "Model":[
        "Default Decision Tree",
        "Controlled Decision Tree",
        "Gini Tree",
        "Entropy Tree"
    ],

    "Train Accuracy":[
        train_acc,
        train_acc2,
        np.nan,
        np.nan
    ],

    "Test Accuracy":[
        test_acc,
        test_acc2,
        gini_acc,
        entropy_acc
    ],

    "ROC AUC":[
        auc_default,
        auc_control,
        gini_auc,
        entropy_auc
    ]

})

print("\nDecision Tree Comparison")

print(comparison)

comparison.to_csv(
    "decision_tree_results.csv",
    index=False
)
# RANDOM FOREST & GRADIENT BOOSTING
print("\n" + "="*70)
print("RANDOM FOREST")
print("="*70)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

rf_train_pred = rf.predict(X_train)

rf_test_pred = rf.predict(X_test)

rf_train_prob = rf.predict_proba(X_train)[:,1]

rf_test_prob = rf.predict_proba(X_test)[:,1]

rf_train_acc = accuracy_score(
    y_train,
    rf_train_pred
)

rf_test_acc = accuracy_score(
    y_test,
    rf_test_pred
)

rf_auc = roc_auc_score(
    y_test,
    rf_test_prob
)

print("\nTraining Accuracy :", round(rf_train_acc,4))
print("Test Accuracy     :", round(rf_test_acc,4))
print("ROC AUC           :", round(rf_auc,4))

# ----------------------------------------------------------
# Feature Importance
# ----------------------------------------------------------

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": rf.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 5 Important Features")

print(
    importance.head(5)
)

importance.to_csv(
    "feature_importance.csv",
    index=False
)

# Gradient Boosting
print("\n" + "="*70)
print("GRADIENT BOOSTING")
print("="*70)

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(
    X_train,
    y_train
)
gb_train_pred = gb.predict(
    X_train
)
gb_test_pred = gb.predict(
    X_test
)
gb_train_prob = gb.predict_proba(
    X_train
)[:,1]
gb_test_prob = gb.predict_proba(
    X_test
)[:,1]
gb_train_acc = accuracy_score(
    y_train,
    gb_train_pred
)
gb_test_acc = accuracy_score(
    y_test,
    gb_test_pred
)
gb_auc = roc_auc_score(
    y_test,
    gb_test_prob
)
print("\nTraining Accuracy :", round(gb_train_acc,4))
print("Test Accuracy     :", round(gb_test_acc,4))
print("ROC AUC           :", round(gb_auc,4))
# Comparison Table
ensemble = pd.DataFrame({
    "Model":[
        "Random Forest",
        "Gradient Boosting"
    ],
    "Training Accuracy":[
        rf_train_acc,
        gb_train_acc
    ],
    "Test Accuracy":[
        rf_test_acc,
        gb_test_acc
    ],
    "ROC AUC":[
        rf_auc,
        gb_auc
    ]
})
print("\nEnsemble Comparison")
print(ensemble)
ensemble.to_csv(
    "ensemble_results.csv",
    index=False
)
# FEATURE ABLATION & CROSS VALIDATION
print("\n" + "="*70)
print("FEATURE ABLATION STUDY")
print("="*70)
# Five Least Important Features
lowest_features = importance.sort_values(
    by="Importance",
    ascending=True
).head(5)
print("\nFive Least Important Features")
print(lowest_features)
remove_features = lowest_features["Feature"].tolist()
print("\nRemoving Features")
print(remove_features)
# Reduced Dataset
X_train_reduced = X_train.drop(
    columns=remove_features
)
X_test_reduced = X_test.drop(
    columns=remove_features
)
# Train Reduced Random Forest
rf_reduced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_reduced.fit(
    X_train_reduced,

    y_train
)
reduced_prob = rf_reduced.predict_proba(
    X_test_reduced
)[:,1]
auc_reduced = roc_auc_score(
    y_test,
    reduced_prob
)
print("\nRandom Forest AUC (Full Model)")
print(round(rf_auc,4))
print("\nRandom Forest AUC (Reduced Model)")
print(round(auc_reduced,4))
ablation = pd.DataFrame({
    "Model":[
        "Full Feature Model",
        "Reduced Feature Model"
    ],
    "ROC AUC":[
        rf_auc,
        auc_reduced
    ]
})
print("\nFeature Ablation Results")
print(ablation)
ablation.to_csv(
    "feature_ablation.csv",
    index=False
)

# CROSS VALIDATION
print("\n" + "="*70)
print("5-FOLD CROSS VALIDATION")
print("="*70)

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)
# Logistic Regression
log_cv = LogisticRegression(

    class_weight="balanced",

    max_iter=1000,

    random_state=42

)

log_scores = cross_val_score(

    log_cv,

    X,

    y_clf,

    cv=cv,

    scoring="roc_auc"

)

# Controlled Decision Tree
tree_scores = cross_val_score(

    tree_control,

    X,

    y_clf,

    cv=cv,

    scoring="roc_auc"

)

# Random Forest
rf_scores = cross_val_score(

    rf,

    X,

    y_clf,

    cv=cv,

    scoring="roc_auc"

)

# Gradient Boosting
gb_scores = cross_val_score(

    gb,

    X,

    y_clf,

    cv=cv,

    scoring="roc_auc"

)

# Comparison Table
cv_results = pd.DataFrame({

    "Model":[
        "Logistic Regression",
        "Controlled Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],

    "Mean CV AUC":[
        log_scores.mean(),
        tree_scores.mean(),
        rf_scores.mean(),
        gb_scores.mean()
    ],

    "Std CV AUC":[
        log_scores.std(),
        tree_scores.std(),
        rf_scores.std(),
        gb_scores.std()
    ]

})
print("\nCross Validation Results")
print(cv_results)
cv_results.to_csv(

    "cross_validation_results.csv",

    index=False)

# PIPELINE + GRID SEARCH
print("\n" + "=" * 70)
print("PIPELINE + GRID SEARCH")
print("=" * 70)

# Pipeline
pipeline = make_pipeline(

    SimpleImputer(strategy="median"),

    StandardScaler(),

    RandomForestClassifier(
        random_state=42
    )

)
# Parameter Grid
param_grid = {

    "randomforestclassifier__n_estimators":[
        50,
        100,
        200
    ],

    "randomforestclassifier__max_depth":[
        5,
        10,
        None
    ],

    "randomforestclassifier__min_samples_leaf":[
        1,
        5
    ]

}
# Cross Validation
cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)
# Grid Search
grid = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    scoring="roc_auc",

    cv=cv,

    n_jobs=-1,

    verbose=1

)
print("\nRunning Grid Search...")
grid.fit(

    X_train,

    y_train

)

# Results
print("\nBest Parameters")
print(grid.best_params_)
print("\nBest Cross Validation AUC")
print(round(grid.best_score_,4))
best_pipeline = grid.best_estimator_
# Number of Models Evaluated
total_models = (
    len(param_grid["randomforestclassifier__n_estimators"])
    *
    len(param_grid["randomforestclassifier__max_depth"])
    *
    len(param_grid["randomforestclassifier__min_samples_leaf"])
)
print("\nTotal Hyperparameter Combinations")
print(total_models)
print("\nTotal Models Trained")
print(total_models * 5)

# Test Performance
best_probability = best_pipeline.predict_proba(
    X_test
)[:,1]
best_prediction = best_pipeline.predict(
    X_test
)
best_auc = roc_auc_score(
    y_test,
    best_probability
)
best_accuracy = accuracy_score(
    y_test,
    best_prediction
)
print("\nBest Pipeline Test Accuracy")
print(round(best_accuracy,4))
print("\nBest Pipeline Test ROC AUC")
print(round(best_auc,4))

# Summary Table
grid_summary = pd.DataFrame({

    "Best Accuracy":[
        best_accuracy
    ],

    "Best ROC AUC":[
        best_auc
    ],

    "Cross Validation AUC":[
        grid.best_score_
    ]
})
print("\nGrid Search Summary")
print(grid_summary)
grid_summary.to_csv(
    "grid_search_results.csv",
    index=False
)

# MANUAL LEARNING CURVE
print("\n" + "=" * 70)
print("MANUAL LEARNING CURVE")
print("=" * 70)

from sklearn.metrics import roc_auc_score

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

learning_results = []

for fraction in fractions:

    size = int(fraction * len(X_train))

    X_subset = X_train.iloc[:size]
    y_subset = y_train.iloc[:size]

    # Train the best pipeline from GridSearchCV
    best_pipeline.fit(
        X_subset,
        y_subset
    )

    # Training AUC
    train_prob = best_pipeline.predict_proba(
        X_subset
    )[:,1]

    train_auc = roc_auc_score(
        y_subset,
        train_prob
    )

    # Test AUC
    test_prob = best_pipeline.predict_proba(
        X_test
    )[:,1]

    test_auc = roc_auc_score(
        y_test,
        test_prob
    )

    learning_results.append([

        f"{int(fraction*100)}%",

        round(train_auc,4),

        round(test_auc,4)

    ])

learning_curve = pd.DataFrame(

    learning_results,

    columns=[

        "Training Fraction",

        "Training AUC",

        "Test AUC"

    ]

)

print("\nLearning Curve")

print(learning_curve)

learning_curve.to_csv(

    "learning_curve.csv",

    index=False

)

# ----------------------------------------------------------
# Trend Analysis
# ----------------------------------------------------------

print("\nTrend Analysis")

train_first = learning_curve.iloc[0]["Training AUC"]
train_last = learning_curve.iloc[-1]["Training AUC"]

test_first = learning_curve.iloc[0]["Test AUC"]
test_last = learning_curve.iloc[-1]["Test AUC"]

if train_last < train_first:
    print("Training AUC decreases as training data increases.")
else:
    print("Training AUC does not decrease.")

if test_last > test_first:
    print("Test AUC increases with more training data.")
else:
    print("Test AUC does not increase.")

difference = test_last - learning_curve.iloc[-2]["Test AUC"]

print("\nBias-Variance Conclusion")

if difference > 0.005:
    print("Model appears DATA-LIMITED.")
    print("Collecting additional training data may improve performance.")
else:
    print("Model appears CAPACITY-LIMITED.")
    print("Increasing model complexity is more likely to improve performance.")

print("\n" + "=" * 70)
print("MODEL SERIALIZATION")
print("=" * 70)

# ----------------------------------------------------------
# Save Best Pipeline
# ----------------------------------------------------------

joblib.dump(
    best_pipeline,
    "best_model.pkl"
)

print("\nModel saved successfully as best_model.pkl")

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------

loaded_model = joblib.load(
    "best_model.pkl"
)

print("Model loaded successfully.")

# ----------------------------------------------------------
# Create Two Sample Test Rows
# ----------------------------------------------------------

sample_rows = X_test.iloc[:2].copy()

print("\nSample Rows")

print(sample_rows)

# ----------------------------------------------------------
# Predict
# ----------------------------------------------------------

predictions = loaded_model.predict(
    sample_rows
)

probabilities = loaded_model.predict_proba(
    sample_rows
)

print("\nPredicted Class")

print(predictions)

print("\nPrediction Probabilities")

print(probabilities)

# ----------------------------------------------------------
# Save Prediction Results
# ----------------------------------------------------------

prediction_results = sample_rows.copy()

prediction_results["Predicted_Class"] = predictions

prediction_results["Probability_No"] = probabilities[:,0]

prediction_results["Probability_Yes"] = probabilities[:,1]

prediction_results.to_csv(

    "sample_predictions.csv",

    index=False

)

print("\nPrediction results saved as sample_predictions.csv")

# ----------------------------------------------------------
# Verify Loaded Model
# ----------------------------------------------------------

print("\nVerification")

if len(predictions) == 2:
    print("Loaded model prediction successful.")
else:
    print("Verification failed.")
print("=" * 70)