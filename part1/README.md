</>
#Part 1 -Data Acquisition, Cleaning and Explonatory Data Analysis
##Dataset
-Loaded the dataset
-removed duplicates
-Converted datatypes
-Created visualizations

**Datase Name:** IBM Telco Customer Churn Dataset
**Source:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Dataset Justification: This dataset was selected because it satisfies all the requirements of the assignment. It contains more than 500 observations, includes both categorial and numerical variables, and is suitable for data analysis, exploratory data analysis, visualization and predictive machine learning. Additionally customer churn prediction is a widely studied business problem, making this dataset appropriate for demonstrating data preprocessing and analytical techniques.
Scatter Plot Interpretation: The scatter plot comparing MonthlyCharges and TotalCharges demonstrates a strong positive relationship between the two variables. Customers who pay higher monthly charges generally accumulate higher total charges over time. The overall trend suggests that increases in monthly billing are associated with increases in the total amount paid by customers. Although the relationship appears strong, the spread of points indicates that customer tenure also influences total charges. Customers with similar monthly charges may have different total charges depending on how long they have remained with the company.
HeatMap Interpretation: the Pearson correlation heatmap reveled that the strongest correlation exists between Tenure and Total Charges, with a corretlation cooefficient of approximately 0.826. This strong postive correlation indicates that customers with longer service duration tend to accumulate higher totla charges.
Why median ws chiosed for implutation: the two most slewed numerical variables were Senior Citizen and Total Charges. For both vatiables the median was chosen for imputation because the distribution are positively skewed. The positively skewed distribution the mean is pulled upward by relatively large values, whereas the median remains more stable and accurately represents the center of the data. Using median therefore reduxs the influence of extreme obsevation and provides a more robust estimate when filing missing values.