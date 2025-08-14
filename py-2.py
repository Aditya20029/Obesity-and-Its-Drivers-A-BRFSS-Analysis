# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset
file_path = "C:\\Users\\Adity\\Downloads\\Cleaned_Data.csv" # Replace with the correct path to the dataset
data = pd.read_csv(file_path)
# Display the first few rows to understand the structure
print("First few rows of the dataset:")
print(data.head())
# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())
# Drop rows where 'data_value' is NaN (as this is the main analysis target)
data = data.dropna(subset=["data_value"])
# Fill missing education values with "Unknown"
data["education"] = data["education"].fillna("Unknown")
# Check unique values in income and education
print("\nUnique income groups:", data["income"].unique())
print("Unique education groups:", data["education"].unique())
# ----------------------- Univariate Analysis -----------------------
# Plot obesity rates by income levels
plt.figure(figsize=(10, 6))
sns.boxplot(x="income", y="data_value", data=data)
plt.title("Obesity Rates by Income Groups")
plt.xticks(rotation=45)
plt.ylabel("Obesity Rate (%)")
plt.xlabel("Income Groups")
plt.show()
# Plot obesity rates by education levels
plt.figure(figsize=(10, 6))
sns.boxplot(x="education", y="data_value", data=data)
plt.title("Obesity Rates by Education Levels")
plt.xticks(rotation=45)
plt.ylabel("Obesity Rate (%)")
plt.xlabel("Education Levels")
plt.show()
# ----------------------- Multivariate Analysis -----------------------
# Create a pivot table to calculate mean obesity rates for each income-education group
pivot_table = data.pivot_table(values="data_value", index="education",
columns="income", aggfunc="mean")
# Plot a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".1f")
plt.title("Mean Obesity Rate by Income and Education Level")
plt.xlabel("Income Groups")
plt.ylabel("Education Levels")
plt.show()
# ----------------------- Regression Analysis -----------------------
# Prepare the data for regression
X = data[["income", "education"]]
y = data["data_value"]
# One-hot encode categorical variables (income and education)
preprocessor = ColumnTransformer(transformers=[
("cat", OneHotEncoder(drop="first"), ["income", "education"])
])
# Create a linear regression pipeline
pipeline = Pipeline(steps=[
("preprocessor", preprocessor),
("regressor", LinearRegression())
])
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Fit the model
pipeline.fit(X_train, y_train)
# Predict on the test set
y_pred = pipeline.predict(X_test)
# Evaluate the model
print("\nRegression Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))
# ----------------------- Feature Importance -----------------------
# Get feature names after one-hot encoding
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
coefficients = pipeline.named_steps['regressor'].coef_
# Combine feature names and coefficients
importance = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
# Sort by absolute value of coefficient
importance["Importance"] = importance["Coefficient"].abs()
importance = importance.sort_values(by="Importance", ascending=False)
print("\nFeature Importance:")
print(importance)
# ----------------------- Summary -----------------------
print("\nSummary of Analysis:")
print("- Lower income and lower education groups tend to have higher obesity
rates.")
print("- Regression analysis quantifies the impact of income and education on
obesity rates.")
print("- Feature importance highlights which factors (income or education) are more
influential in predicting obesity rates.")