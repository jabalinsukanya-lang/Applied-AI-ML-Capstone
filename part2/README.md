#part  - Supervised Machine learning: Build, Train and Evaluate
##Overview
This project builds supervised machine learning models using cleaned dataset generated in part 
-**Regression:** predicting customer **TtoalCharge**.
-**Classification:**Predicting whether a customer will **churn (Yes/No)**.
#Dataset used: cleaned_data.csv
#Source: Generated from part after cleaning the IBM TELCO CUSTOMER CHURN dataset
##Regression Target: TotalCharges- This is a continuous numerical variable representing the total amount charged to each customer.
##Classification Targt: The original dataset already contains a neutral binary target. Mapping used: Originally No and Yes Encoded to 0 and 1

#Feature Engineering: 
-customerID
-TotalCharges
-Churn
customerID was removed because it is only an identifier and carries no predictive information.

#Encoding Stratergy
    ##Label Encoding: the **Contract** column has a batural ordr. Month to month -> one Year -> two year. Therefoer it is encoded as Month to month as 0, One year as 1 and Two years as 2.
    ##One-Hot Encoding: the remaining categorial variables were one-hot encoded.