import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

#Load Dataset
df=pd.read_csv("../part1/cleaned_data.csv")
print("="*60)
print("\nDataset Loaded Successfully")
print("="*60)
print(df.head())
print("\nShape:",df.shape)
print("\nDatatypes")
print(df.dtypes)
# Define Targets
# Regression Target
y_reg = df["TotalCharges"]
# Classification Target
# Convert Yes/No to 1/0
y_clf = df["Churn"].map({
    "No":0,
    "Yes":1
})
# Feature Matrix
X = df.drop(
    columns=[
        "customerID",
        "TotalCharges",
        "Churn"
    ]
)
print("\nFeature Matrix Shape :", X.shape)
# Encode Ordered Category
contract_mapping = {
    "Month-to-month":0,
    "One year":1,
    "Two year":2
}
X["Contract"] = X["Contract"].map(contract_mapping)
# One-Hot Encoding
categorical_cols = X.select_dtypes(include="object").columns
X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)
print("\nEncoded Feature Shape :", X.shape)
# Regression Train/Test Split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X,
    y_reg,
    test_size=0.20,
    random_state=42
)
# Classification Train/Test Split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X,
    y_clf,
    test_size=0.20,
    random_state=42,
    stratify=y_clf
)
# Scaling (Regression)
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(
    X_train_reg
)
X_test_reg_scaled = scaler_reg.transform(
    X_test_reg
)
# Scaling (Classification)
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(
    X_train_clf
)
X_test_clf_scaled = scaler_clf.transform(
    X_test_clf
)
print("\nScaling Completed")
#Class Distribution
print("\nClassification Target Distribution")
print(y_train_clf.value_counts())
print("\nPercentage")
print(
    y_train_clf.value_counts(normalize=True) * 100
)

# LINEAR REGRESSION & RIDGE REGRESSION
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
print("\n" + "=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)
# Train Linear Regression
linear_model = LinearRegression()
linear_model.fit(
    X_train_reg_scaled,
    y_train_reg
)
# Prediction
y_pred_reg = linear_model.predict(
    X_test_reg_scaled
)
# Evaluation
mse_linear = mean_squared_error(
    y_test_reg,
    y_pred_reg
)
r2_linear = r2_score(
    y_test_reg,
    y_pred_reg
)
print("\nLinear Regression Results")
print(f"MSE : {mse_linear:.2f}")
print(f"R²  : {r2_linear:.4f}")
# Coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_
})
coef_df["Absolute"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values(
    by="Absolute",
    ascending=False
)
print("\nTop 10 Features")
print(
    coef_df[
        ["Feature","Coefficient"]
    ].head(10)
)
print("\nTop 3 Most Important Features")
top3 = coef_df.head(3)
print(
    top3[
        ["Feature","Coefficient"]
    ]
)
# Ridge Regression
print("\n" + "=" * 60)
print("RIDGE REGRESSION")
print("=" * 60)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(
    X_train_reg_scaled,
    y_train_reg
)
ridge_pred = ridge_model.predict(
    X_test_reg_scaled
)
mse_ridge = mean_squared_error(
    y_test_reg,
    ridge_pred
)
r2_ridge = r2_score(
    y_test_reg,
    ridge_pred
)
print("\nRidge Regression Results")
print(f"MSE : {mse_ridge:.2f}")
print(f"R²  : {r2_ridge:.4f}")
# Comparison Table
comparison = pd.DataFrame({
    "Model":[
        "Linear Regression",
        "Ridge Regression"
    ],
    "MSE":[
        mse_linear,
        mse_ridge
    ],
    "R2 Score":[
        r2_linear,
        r2_ridge
    ]
})
print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(comparison)
# Ridge Coefficients
ridge_coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": ridge_model.coef_
})
ridge_coef["Absolute"] = ridge_coef["Coefficient"].abs()
ridge_coef = ridge_coef.sort_values(
    by="Absolute",
    ascending=False
)
print("\nTop Ridge Features")
print(
    ridge_coef[
        ["Feature","Coefficient"]
    ].head(10)
)

# LOGISTIC REGRESSION CLASSIFICATION
# Create plots folder
os.makedirs("plots", exist_ok=True)
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)
# Check Class Distribution
print("\nClass Distribution Before Training")
print(y_train_clf.value_counts())
print("\nPercentage")
print((y_train_clf.value_counts(normalize=True) * 100).round(2))

# Logistic Regression
log_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
log_model.fit(
    X_train_clf_scaled,
    y_train_clf
)
# Prediction
y_pred = log_model.predict(
    X_test_clf_scaled
)
y_prob = log_model.predict_proba(
    X_test_clf_scaled
)[:,1]
# Confusion Matrix
cm = confusion_matrix(
    y_test_clf,
    y_pred
)
print("\nConfusion Matrix")
print(cm)
# Accuracy
accuracy = accuracy_score(
    y_test_clf,
    y_pred
)
precision = precision_score(
    y_test_clf,
    y_pred
)
recall = recall_score(
    y_test_clf,
    y_pred
)
f1 = f1_score(
    y_test_clf,
    y_pred
)
print("\nAccuracy :", round(accuracy,4))
print("Precision :", round(precision,4))
print("Recall    :", round(recall,4))
print("F1 Score  :", round(f1,4))
# Classification Report
print("\nClassification Report")
print(
    classification_report(
        y_test_clf,
        y_pred
    )
)
# ROC Curve
fpr, tpr, thresholds = roc_curve(
    y_test_clf,
    y_prob
)
auc = roc_auc_score(
    y_test_clf,
    y_prob
)
print("\nAUC Score :", round(auc,4))
# Plot ROC
plt.figure(figsize=(8,6))
plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {auc:.3f}"
)
plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.savefig(
    "plots/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
# Summary Table
metrics = pd.DataFrame({
    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "AUC"
    ],
    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]
})
print("\n")
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(metrics)

# THRESHOLD ANALYSIS & REGULARIZATION
print("\n" + "=" * 60)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 60)
# Threshold Analysis
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

threshold_results = []

for threshold in thresholds:

    pred = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_test_clf,
        pred
    )

    recall = recall_score(
        y_test_clf,
        pred
    )

    f1 = f1_score(
        y_test_clf,
        pred
    )

    threshold_results.append([
        threshold,
        precision,
        recall,
        f1
    ])

threshold_df = pd.DataFrame(

    threshold_results,

    columns=[
        "Threshold",
        "Precision",
        "Recall",
        "F1 Score"
    ]

)

print("\nThreshold Performance")

print(threshold_df)

best_row = threshold_df.loc[
    threshold_df["F1 Score"].idxmax()
]

print("\nBest Threshold")

print(best_row)
# STRONG REGULARIZATION
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION (C=0.01)")
print("=" * 60)

log_model_smallC = LogisticRegression(

    C=0.01,

    class_weight="balanced",

    max_iter=1000,

    random_state=42

)

log_model_smallC.fit(

    X_train_clf_scaled,

    y_train_clf

)

pred_small = log_model_smallC.predict(

    X_test_clf_scaled

)

prob_small = log_model_smallC.predict_proba(

    X_test_clf_scaled

)[:,1]

precision_small = precision_score(

    y_test_clf,

    pred_small

)

recall_small = recall_score(

    y_test_clf,

    pred_small

)

auc_small = roc_auc_score(

    y_test_clf,

    prob_small

)
# Comparison Table
comparison = pd.DataFrame({

    "Model":[
        "Logistic C=1.0",
        "Logistic C=0.01"
    ],

    "Precision":[
        precision,
        precision_small
    ],

    "Recall":[
        recall,
        recall_small
    ],

    "AUC":[
        auc,
        auc_small
    ]

})

print("\nRegularization Comparison")

print(comparison)

# BOOTSTRAP
print("\n" + "=" * 60)
print("BOOTSTRAP CONFIDENCE INTERVAL")
print("=" * 60)

np.random.seed(42)

auc_difference = []

for i in range(500):

    idx = np.random.choice(

        len(y_test_clf),

        size=len(y_test_clf),

        replace=True

    )

    sample_y = y_test_clf.iloc[idx]

    sample_prob1 = y_prob[idx]

    sample_prob2 = prob_small[idx]

    try:

        auc1 = roc_auc_score(
            sample_y,
            sample_prob1
        )

        auc2 = roc_auc_score(
            sample_y,
            sample_prob2
        )

        auc_difference.append(
            auc1-auc2
        )

    except:

        pass

auc_difference = np.array(auc_difference)

mean_diff = auc_difference.mean()

lower = np.percentile(
    auc_difference,
    2.5
)

upper = np.percentile(
    auc_difference,
    97.5
)

print("\nMean Difference")

print(mean_diff)

print("\n95% Confidence Interval")

print(lower)

print(upper)

print("\nInterval Excludes Zero?")

if lower > 0 or upper < 0:

    print("YES")

else:

    print("NO")

# SAVE TABLES
threshold_df.to_csv(
    "threshold_results.csv",
    index=False
)
comparison.to_csv(
    "regularization_comparison.csv",
    index=False
)
print("\nCSV files saved.")