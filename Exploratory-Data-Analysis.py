import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Upload dataset
#df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls", skiprows=1)
df= pd.read_csv("data/UCI_Credit_Card.csv")
df.columns = ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + [f"PAY_{i}" for i in range(0, 6)] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)] + ["default.payment.next.month"]
df.drop(columns=["ID"], inplace=True)

# Rename target column
df.rename(columns={"default.payment.next.month": "DEFAULT"}, inplace=True)

# Visualizing the distribution of the target variable
fig = px.histogram(df, x="DEFAULT", title="Distribution of customers in default")
fig.show()

# Correlation analysis between variables
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation analysis between variables")
plt.show()

# Credit limits distribution
fig = px.histogram(df, x="LIMIT_BAL", nbins=50, title="Credit limits distribution")
fig.show()

# Age comparison between defaulted and non-defaulted customers
fig = px.box(df, x="DEFAULT", y="AGE", title="Age comparison between defaulted and non-defaulted customers")
fig.show()



