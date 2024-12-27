import pandas as pd

# Load the dataset to examine its structure
df = pd.read_csv("Customer_chrun.py/rainfall in india 1901-2015.csv")
# Display initial rows and data info for understanding
df.head()
df.info()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Analyzing annual rainfall trends
annual_rainfall = df.groupby('YEAR')['ANNUAL'].mean()
mean_rainfall = annual_rainfall.mean()

# Visualization: Annual Rainfall Trend
plt.figure(figsize=(12, 8))
plt.plot(annual_rainfall.index, annual_rainfall.values, color='blue', label='Annual Rainfall')
plt.axhline(mean_rainfall, color='red', linestyle='--', label='Mean Rainfall')
plt.title('Trend in Annual Rainfall in India (1901-2015)')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.grid(True)
plt.show()

# Rolling Average
rolling_avg = annual_rainfall.rolling(window=10).mean()

plt.figure(figsize=(12, 8))
plt.plot(annual_rainfall.index, annual_rainfall.values, color='blue', alpha=0.5, label='Annual Rainfall')
plt.plot(rolling_avg.index, rolling_avg.values, color='green', label='10-Year Rolling Average')
plt.axhline(mean_rainfall, color='red', linestyle='--', label='Mean Rainfall')
plt.title('10-Year Rolling Average of Annual Rainfall (1901-2015)')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.grid(True)
plt.show()

# Observations for threshold analysis (extreme rainfall)
std_dev = annual_rainfall.std()
threshold_high = mean_rainfall + 1.5 * std_dev
threshold_low = mean_rainfall - 1.5 * std_dev

extreme_high = annual_rainfall[annual_rainfall > threshold_high]
extreme_low = annual_rainfall[annual_rainfall < threshold_low]

# Extract top extreme rainfall years
extreme_low_df = extreme_low.reset_index()[['YEAR', 'ANNUAL']].sort_values(by='ANNUAL').head()
extreme_high_df = extreme_high.reset_index()[['YEAR', 'ANNUAL']].sort_values(by='ANNUAL', ascending=False).head()

# Results for extremes
(extreme_low_df, extreme_high_df)
# Monthly Average Rainfall Analysis
monthly_avg_rainfall = df[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                           'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].mean()
overall_mean_rainfall = monthly_avg_rainfall.mean()

# Visualization: Monthly Rainfall with Mean Comparison
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=monthly_avg_rainfall.index, y=monthly_avg_rainfall.values, color='skyblue', label='Monthly Avg Rainfall')
plt.axhline(overall_mean_rainfall, color='red', linestyle='--', label='Mean Rainfall')
plt.title('Average Monthly Rainfall with Mean Comparison')
plt.xlabel('Month')
plt.ylabel('Rainfall (mm)')

for container in ax.containers:
    ax.bar_label(container)

plt.legend()
plt.grid(True)
plt.show()

# Seasonal Analysis
seasonal_rainfall = df[['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']].mean()
overall_seasonal_mean = seasonal_rainfall.mean()

# Visualization: Seasonal Rainfall with Mean Comparison
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=seasonal_rainfall.index, y=seasonal_rainfall.values, color='skyblue', label='Seasonal Avg Rainfall')
plt.axhline(overall_seasonal_mean, color='red', linestyle='--', label='Mean Rainfall')
plt.title('Average Seasonal Rainfall with Mean Comparison')
plt.xlabel('Season')
plt.ylabel('Rainfall (mm)')

for container in ax.containers:
    ax.bar_label(container)

plt.legend()
plt.grid(True)
plt.show()

# Seasonal correlation with annual rainfall
seasonal_corr = df[['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']].corrwith(df['ANNUAL']).to_frame(name='Correlation')
seasonal_corr
from sklearn.ensemble import IsolationForest

# Anomaly Detection - Annual Rainfall
iso_forest_annual = IsolationForest(contamination=0.05, random_state=42)
annual_rainfall_df = annual_rainfall.reset_index()
annual_rainfall_df['Anomaly'] = iso_forest_annual.fit_predict(annual_rainfall_df[['ANNUAL']])

anomalies_annual = annual_rainfall_df[annual_rainfall_df['Anomaly'] == -1]
normal_annual = annual_rainfall_df[annual_rainfall_df['Anomaly'] == 1]

# Visualization: Anomalies in Annual Rainfall
plt.figure(figsize=(12, 8))
plt.plot(annual_rainfall_df['YEAR'], annual_rainfall_df['ANNUAL'], color='blue', label='Annual Rainfall')
plt.scatter(anomalies_annual['YEAR'], anomalies_annual['ANNUAL'], color='red', label='Anomalous Years')
plt.axhline(mean_rainfall, color='green', linestyle='--', label='Mean Rainfall')
plt.title('Annual Rainfall Anomalies in India (1901-2015)')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.grid(True)
plt.show()

# Display detected anomalies
anomalies_annual[['YEAR', 'ANNUAL']].sort_values(by='ANNUAL')
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Clustering Analysis
rainfall_features = df[['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec', 'ANNUAL']].dropna()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(rainfall_features)

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rainfall_features['Cluster'] = kmeans.fit_predict(scaled_features)

# Mapping clusters to categories
cluster_labels = {0: 'Dry', 1: 'Normal', 2: 'Wet'}
rainfall_features['Category'] = rainfall_features['Cluster'].map(cluster_labels)

# Add category to the main dataframe
df['Rainfall_Category'] = rainfall_features['Category']

# Visualization: Clustering Results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['YEAR'], df['ANNUAL'], c=df['Rainfall_Category'].astype('category').cat.codes, cmap='viridis')
plt.title('Clustering of Years Based on Rainfall Patterns')
plt.xlabel('Year')
plt.ylabel('Annual Rainfall (mm)')
plt.legend(handles=scatter.legend_elements()[0], labels=['Dry', 'Normal', 'Wet'], title="Rainfall Category")
plt.grid(True)
plt.show()

# Display clustered categories
rainfall_features[['Cluster', 'Category']].value_counts()
from sklearn.linear_model import LinearRegression

# Forecasting Future Rainfall Trends
data = df[['YEAR', 'ANNUAL']].dropna()
X = data[['YEAR']]
y = data['ANNUAL']

# Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Predictions for the next 20 years (2016–2035)
future_years = np.arange(2016, 2036).reshape(-1, 1)
future_rainfall = model.predict(future_years)

# Visualization: Forecasted Rainfall
plt.figure(figsize=(12, 8))
plt.scatter(data['YEAR'], data['ANNUAL'], color='blue', label='Actual Rainfall')
plt.plot(future_years, future_rainfall, color='red', label='Forecasted Rainfall')
plt.title('Forecasted Annual Rainfall for the Next 20 Years (2016-2035)')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.grid(True)
plt.show()

# Forecasted Data
forecast_data = pd.DataFrame({'YEAR': future_years.flatten(), 'Forecasted_Rainfall': future_rainfall})
forecast_data.head(20)
# Re-import required libraries and reload data due to environment reset.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Reload dataset
file_path = '/mnt/data/rainfall in india 1901-2015.csv'
df = pd.read_csv(file_path)

# Forecasting Future Rainfall Trends
data = df[['YEAR', 'ANNUAL']].dropna()
X = data[['YEAR']]
y = data['ANNUAL']

# Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Predictions for the next 20 years (2016–2035)
future_years = np.arange(2016, 2036).reshape(-1, 1)
future_rainfall = model.predict(future_years)

# Visualization: Forecasted Rainfall
plt.figure(figsize=(12, 8))
plt.scatter(data['YEAR'], data['ANNUAL'], color='blue', label='Actual Rainfall')
plt.plot(future_years, future_rainfall, color='red', label='Forecasted Rainfall')
plt.title('Forecasted Annual Rainfall for the Next 20 Years (2016-2035)')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.grid(True)
plt.show()

# Forecasted Data
forecast_data = pd.DataFrame({'YEAR': future_years.flatten(), 'Forecasted_Rainfall': future_rainfall})
forecast_data.head(20)
