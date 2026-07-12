import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")
#Load Dataset
df=pd.read_csv("../dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.head())
print("\nData Types")
print(df.dtypes)
print("\nShape")
print(df.shape)
#Null Value Ananlysis
null_count=df.isnull().sum()
null_percent=(df.isnull().sum()/df.shape[0])*100
print(type(null_count))
print(type(null_percent))
print(null_count.shape)
print(null_percent.shape)
null_table=pd.DataFrame({"Missig Values":null_count, "Percentage":null_percent})
print(null_table)
print("\nColumn with >20% Null Values")
print(null_table[null_table["Percentage"]>20])
numeric_cols=df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    if null_percent[col]<20:
        df[col]=df[col].fillna(df[col].median())
#Duplicate Detection
duplicates=df.duplicated().sum()
print("Duplicate Rows:",duplicates)
before=df.shape[0]
df=df.drop_duplicates()
after=df.shape[0]
print("Rows removed",before-after)
new_null=(df.isnull().sum()/df.shape[0])*100
print(new_null)
#Datatype Correction
memory_before=df.memory_usage(deep=True).sum()
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
df["Contract"]=df["Contract"].astype("category")
memory_after=df.memory_usage(deep=True).sum()
print("Memory Before:",memory_before)
print("Memory After",memory_after)
#Statics & Skewness
print(df.describe())
skewness={}
for col in df.select_dtypes(include=np.number).columns:
    skewness[col]=df[col].skew()
skew_df=pd.DataFrame(skewness.items(),columns=["Column","Skewness"])
print(skew_df)
highest=skew_df.iloc[skew_df["Skewness"].abs().idxmax()]
print(highest)
#outlier Detection
cols=["MonthlyCharges", "TotalCharges"]
for col in cols:
    q1=df[col].quantile(.25)
    q3=df[col].quantile(.75)
    iqr=q3-q1
    lower=q1-1.5*iqr
    upper=q3+1.5*iqr
    outliers=df[(df[col]<lower)|(df[col]>upper)]
    print(col)
    print("Outliers:",len(outliers))
#Visualizations
#Line Plot
plt.figure(figsize=(10,5))
plt.plot(df.index,df["MonthlyCharges"])
plt.title("MonthlyCharges")
plt.xlabel("Index")
plt.ylabel("MonthlyCharges")
plt.savefig("plots/line_plot.png")
plt.show()
#Bar Chart
group=df.groupby("Contract")["MonthlyCharges"].mean()
group.plot.bar()
plt.title("Average Monbthly Charges by Contract")
plt.xlabel("Contracts")
plt.ylabel("Average Charges")
plt.savefig("plots/bar_plot.png")
plt.show()
#Histogram
plt.figure(figsize=(8,5))
sns.histplot(df["TotalCharges"],bins=20)
plt.title("Distribution of Total Charges")
plt.savefig("plots/histogram.png")
plt.show()
#Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="MonthlyCharges",
    y="TotalCharges")
plt.title("Monthly vs Total Charges")
plt.savefig("plots/scatter.png")
plt.show()
#Box Plot
plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Contract",
    y="MonthlyCharges"
)
plt.title("Monthly Charges by Contract")
plt.savefig("plots/boxplot.png")
plt.show()
#Correlation Heatmap
numeric=df.select_dtypes(include=np.number)
pearson=numeric.corr()
plt.figure(figsize=(8,6))
sns.heatmap(
    pearson,
    annot=True,
    cmap="coolwarm"
)
plt.title("Pearson Correlation")
plt.savefig("plots/heatmap.png")
plt.show()
#Mean and Median
top2 = skew_df.reindex(skew_df["Skewness"].abs().sort_values(ascending=False).index).head(2)
for col in top2["Column"]:
    print(col)
    print("Mean :",df[col].mean())
    print("Median :",df[col].median())
    df[col] = df[col].fillna(df[col].median())
print(df[top2["Column"]].isnull().sum())
#Spearman Correlation
spearman = numeric.corr(method="spearman")
print("Pearson")
print(pearson)
print("Spearman")
print(spearman)
pairs=[]
cols=list(pearson.columns)
for i in range(len(cols)):
    for j in range(i+1,len(cols)):
        p=pearson.iloc[i,j]
        s=spearman.iloc[i,j]
        pairs.append([
            cols[i],
            cols[j],
            p,
            s,
            abs(s-p)
        ])
difference=pd.DataFrame(
    pairs,
    columns=[
        "Column1",
        "Column2",
        "Pearson",
        "Spearman",
        "Difference"
    ]
)
difference=difference.sort_values(
    by="Difference",
    ascending=False
)
print(difference.head(3))
#Grouped Aggregation
agg=df.groupby("Contract")["MonthlyCharges"].agg([
    "mean",
    "std",
    "count"
])
print(agg)
highest_mean=agg["mean"].idxmax()
highest_std=agg["std"].idxmax()
ratio=agg["mean"].max()/agg["mean"].min()
print("Highest Mean:",highest_mean)
print("Highest Std:",highest_std)
print("Ratio:",ratio)
#Save Clean Dataset
df.to_csv("cleaned_data.csv",index=False)
print("cleaned_data.csv created successfully")