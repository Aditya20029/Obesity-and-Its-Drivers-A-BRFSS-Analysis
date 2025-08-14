# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene, bartlett, ttest_ind
from statsmodels.formula.api import ols
import statsmodels.api as sm
# Load the dataset
file_path = "C:\\Users\\Adity\\Downloads\\Cleaned_Data.csv" # Replace with the
file path for the dataset
data = pd.read_csv(file_path)
# Preview the data
print(data.head())
# --- Preprocessing ---
# Rename columns for easier access
data.columns = data.columns.str.lower().str.replace(" ", "_")
# Handle missing values
data['data_value'] = pd.to_numeric(data['data_value'], errors='coerce') # Convert
'data_value' to numeric
data = data.dropna(subset=['data_value', 'latitude', 'longitude']) # Drop rows with missing key data
# --- Urban/Rural Classification ---
# Define a proxy for urban vs rural (e.g., based on latitude/longitude or a predefined classification)
# Here, we use an arbitrary cutoff for latitude as a proxy for urban/rural (you can adjust this based on actual data)
data['urban_rural'] = np.where(data['latitude'] > 60, 'Rural', 'Urban')
# --- Univariate Analysis ---
# Summary statistics for physical activity, eating habits, and obesity by urban/rural
physical_activity = data[data['topic'] == 'Physical Activity - Behavior']
obesity = data[data['topic'] == 'Obesity / Weight Status']
print("\nSummary Statistics for Physical Activity:")
print(physical_activity.groupby('urban_rural')['data_value'].describe())
print("\nSummary Statistics for Obesity:")
print(obesity.groupby('urban_rural')['data_value'].describe())
# Visualize distributions
sns.histplot(data=physical_activity, x='data_value', hue='urban_rural', kde=True)
plt.title("Physical Activity by Urban/Rural")
plt.xlabel("Physical Activity (% Adults)")
plt.show()
sns.histplot(data=obesity, x='data_value', hue='urban_rural', kde=True)
plt.title("Obesity by Urban/Rural")
plt.xlabel("Obesity Prevalence (% Adults)")
plt.show()
# --- Variance Comparison ---
# Compare variances between urban and rural populations for obesity and physical activity
stat, p_physical = levene(
physical_activity[physical_activity['urban_rural'] == 'Urban']['data_value'],
physical_activity[physical_activity['urban_rural'] == 'Rural']['data_value']
)
print(f"Levene's Test for Physical Activity Variance: stat={stat}, pvalue={
p_physical}")
stat, p_obesity = levene(
obesity[obesity['urban_rural'] == 'Urban']['data_value'],
obesity[obesity['urban_rural'] == 'Rural']['data_value']
)
print(f"Levene's Test for Obesity Variance: stat={stat}, p-value={p_obesity}")
# --- Multivariate Analysis ---
# Combine physical activity and obesity data
combined_data = physical_activity[['locationabbr', 'urban_rural',
'data_value']].rename(columns={'data_value': 'physical_activity'})
combined_data = combined_data.merge(
obesity[['locationabbr', 'urban_rural',
'data_value']].rename(columns={'data_value': 'obesity'}),
on=['locationabbr', 'urban_rural'],
how='inner'
)
# Fit a regression model to test the relationship between physical activity, obesity, and urban/rural
model = ols("obesity ~ physical_activity + urban_rural", data=combined_data).fit()
print("\nRegression Model Summary:")
print(model.summary())
# Visualize the relationship
sns.scatterplot(data=combined_data, x='physical_activity', y='obesity',
hue='urban_rural')
plt.title("Obesity vs Physical Activity by Urban/Rural")
plt.xlabel("Physical Activity (% Adults)")
plt.ylabel("Obesity Prevalence (% Adults)")
plt.show()