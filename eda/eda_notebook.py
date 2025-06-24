import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/creditcard.csv')

# Overview
print(df.head())
print(df.info())
print(df['Class'].value_counts())

# Distribution
sns.countplot(x='Class', data=df)
plt.title("Class Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm", vmax=0.8)
plt.title("Correlation Heatmap")
plt.show()
