# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset
file_path = "C:\\Users\\Adity\\Downloads\\Cleaned_Data.csv" # Replace with your actual file path

data = pd.read_csv(file_path)
# Display the first few rows of the dataset
print(data.head())
# Handle missing values
data.replace("NA", np.nan, inplace=True)
data.dropna(subset=["data_value"], inplace=True) # Drop rows where 'data_value' is NaN

data["data_value"] = data["data_value"].astype(float)
# Univariate Analysis
# 1. Distribution of Obesity Rates
plt.figure(figsize=(10, 6))
sns.histplot(data[data["topic"] == "Obesity / Weight Status"]["data_value"],
kde=True, color="blue")
plt.title("Distribution of Obesity Rates")
plt.xlabel("Obesity Rate (%)")
plt.ylabel("Frequency")
plt.show()
# 2. Distribution of Physical Activity Rates
plt.figure(figsize=(10, 6))
sns.histplot(data[data["topic"] == "Physical Activity - Behavior"]["data_value"],
kde=True, color="green")
plt.title("Distribution of Physical Activity Rates")
plt.xlabel("Physical Activity Rate (%)")
plt.ylabel("Frequency")
plt.show()
# Multivariate Analysis
# 1. Correlation between Physical Activity and Obesity by Age
# Filter relevant data for obesity and physical activity
obesity_data = data[data["topic"] == "Obesity / Weight Status"]
physical_activity_data = data[data["topic"] == "Physical Activity - Behavior"]
# Merge the datasets on common columns
merged_data = pd.merge(obesity_data, physical_activity_data, on=["locationdesc",
"age_years", "income", "education"], suffixes=('_obesity', '_physical'))
# Scatter plot of Obesity vs Physical Activity
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_data, x="data_value_physical", y="data_value_obesity",
hue="age_years", palette="viridis")
plt.title("Obesity vs Physical Activity by Age")
plt.xlabel("Physical Activity Rate (%)")
plt.ylabel("Obesity Rate (%)")
plt.legend(title="Age")
plt.show()
# Correlation Coefficient
correlation, p_value = pearsonr(merged_data["data_value_physical"],
merged_data["data_value_obesity"])
print(f"Correlation between Physical Activity and Obesity: {correlation:.2f} (pvalue:
{p_value:.2e})")
# 2. Regression Analysis
# Prepare data for regression
X = merged_data[["data_value_physical", "age_years"]]
y = merged_data["data_value_obesity"]
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Create and train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predictions
y_pred = regressor.predict(X_test)
# Regression Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
# Coefficients
print("Regression Coefficients:")
print(f"Intercept: {regressor.intercept_}")
print(f"Physical Activity Coefficient: {regressor.coef_[0]}")
print(f"Age Coefficient: {regressor.coef_[1]}")
# 3. Heatmap to show relationships
plt.figure(figsize=(10, 6))
correlation_matrix = merged_data[["data_value_physical", "data_value_obesity",
"age_years"]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
# 4. Obesity and Physical Activity by Income Groups
plt.figure(figsize=(12, 6))
sns.boxplot(data=merged_data, x="income", y="data_value_obesity", palette="Set3")
plt.title("Obesity Rates by Income Group")
plt.xlabel("Income Group")
plt.ylabel("Obesity Rate (%)")
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(data=merged_data, x="income", y="data_value_physical", palette="Set2")
plt.title("Physical Activity Rates by Income Group")
plt.xlabel("Income Group")
plt.ylabel("Physical Activity Rate (%)")
plt.xticks(rotation=45)
plt.show()