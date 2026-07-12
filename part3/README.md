#part  - Advanced Modeling: Ensembles, Hyperparameter Tuning and ML Pipeline

##Overview: This project extends the supervised machine learning models developed in Part 2 by implementing ensemble learning techniques, hyperparameter tuning, feature selection, cross-validation, model serialization, and a production-ready machine learning pipeline. The objective is to compare multiple classification algorithms and identify the model that provides the best balance between predictive performance and generalization.

#Dataset: cleaned_data.csv
#Source: dataset generated during Part1 cleaning the IBM Telco Customer dataset

#Data Preprocessing:
    -label encoding of ircered contract feature.
    -One-Hot encoding of nominal categorial variales
    -Train-test split using an 80:20 ratio
    -Standaization using StandardScaler.
#Decision Tree Baseline: The first model trained was an unconstrained Decision Tree using the default sklearn parameters.
##Results
Training Accuracy : 0.9979
Test Accuracy     : 0.7346
ROC AUC           : 0.6631
#Controlled Decision Tree: A second Decision Tree was trained
##Results
Training Accuracy : 0.8003
Test Accuracy     : 0.7956
ROC AUC           : 0.8353
GINI vs ENTROPY
======================================================================

Decision Tree Comparison
                      Model  Train Accuracy  Test Accuracy   ROC AUC
0     Default Decision Tree        0.997870       0.734564  0.663077
1  Controlled Decision Tree        0.800319       0.795600  0.835304
2                 Gini Tree             NaN       0.795600  0.835304
3              Entropy Tree             NaN       0.794890  0.833979

======================================================================
RANDOM FOREST
======================================================================

Training Accuracy : 0.869
Test Accuracy     : 0.8006
ROC AUC           : 0.8367

Top 5 Important Features
                           Feature  Importance
1                           tenure    0.228838
2                         Contract    0.144183
3                   MonthlyCharges    0.136857
10     InternetService_Fiber optic    0.062807
26  PaymentMethod_Electronic check    0.059450

======================================================================
GRADIENT BOOSTING
======================================================================

Training Accuracy : 0.8229
Test Accuracy     : 0.7984
ROC AUC           : 0.8424

Ensemble Comparison
               Model  Training Accuracy  Test Accuracy   ROC AUC
0      Random Forest           0.869010       0.800568  0.836706
1  Gradient Boosting           0.822861       0.798439  0.842391

======================================================================
FEATURE ABLATION STUDY
======================================================================

Five Least Important Features
                                Feature  Importance
8        MultipleLines_No phone service    0.004286
7                      PhoneService_Yes    0.005114
18      TechSupport_No internet service    0.008920
12   OnlineSecurity_No internet service    0.009683
22  StreamingMovies_No internet service    0.009888

Removing Features
['MultipleLines_No phone service', 'PhoneService_Yes', 'TechSupport_No internet service', 'OnlineSecurity_No internet service', 'StreamingMovies_No internet service']

Random Forest AUC (Full Model)
0.8367

Random Forest AUC (Reduced Model)
0.836

Feature Ablation Results Model   
ROC 
AUC
0     Full Feature Model  0.836706
1  Reduced Feature Model  0.835985