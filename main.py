import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import zscore

# Load the dataset
file_path = "./CSV_files/Student_Performance_Data_Wide_Version.xlsx"
df = pd.read_excel(file_path, "Sheet1")

# Data Cleaning
# Handling missing values
df = df.head(15)
df = df.copy()  # Work on a copy of the DataFrame to avoid SettingWithCopyWarning
df.fillna(0, inplace=True)  # Fill missing values with 0

# Removing outliers using Z-score
df_numeric = df.select_dtypes(include=[np.number])
df = df[(np.abs(zscore(df_numeric)) < 3).all(axis=1)]

# Data Transformation
# Min-max normalization
scaler = MinMaxScaler()
df[df_numeric.columns] = scaler.fit_transform(df[df_numeric.columns])

# Z-score normalization
scaler = StandardScaler()
df[df_numeric.columns] = scaler.fit_transform(df[df_numeric.columns])

# Decimal scaling
df[df_numeric.columns] = df[df_numeric.columns] / 10**np.ceil(np.log10(df[df_numeric.columns].abs().max()))

# Data Discretization - Binning
# Binning continuous data into 4 bins
for col in df_numeric.columns:
    df[col + '_binned'] = pd.cut(df[col], bins=4, labels=False)

# Data Analysis and Visualization

# Line Chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=df.drop(columns=[col for col in df.columns if 'binned' in col]))
plt.title('Line Chart')
plt.savefig('line_chart.png')
plt.close()

# Example Bar Graph (needs specific categorical and numerical columns)
plt.figure(figsize=(10, 6))
sns.barplot(x='Student_ID', y='Paper 1', data=df)
plt.title('Bar Graph of Paper 1 Scores by Student ID')
plt.xticks(rotation=90)
plt.savefig('bar_graph.png')
plt.close()

# Example Histogram

plt.figure(figsize=(10, 6))
sns.histplot(df['Paper 1'], bins=30)
plt.title('Histogram of Paper 1 Scores')
plt.savefig('histogram.png')
plt.close()

# Example Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Semster_Name', y='Paper 1', data=df)
plt.title('Box Plot of Paper 1 Scores by Semester')
plt.xticks(rotation=90)
plt.savefig('box_plot.png')
plt.close()

# Example Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Paper 1', y='Paper 2', data=df)
plt.title('Scatter Plot of Paper 1 vs Paper 2 Scores')
plt.savefig('scatter_plot.png')
plt.close()

# Heat Map
numeric_df = df.select_dtypes(include=[np.number])  # Ensure only numeric columns are used
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Heat Map of Correlations')
plt.savefig('heat_map.png')
plt.close()
