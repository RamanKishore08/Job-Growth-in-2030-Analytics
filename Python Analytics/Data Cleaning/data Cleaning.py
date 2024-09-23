import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\91962\Documents\Data Analytics Project\Dataset.csv")

missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Remove rows with missing values
data_cleaned = data.dropna()

data_cleaned['Projected Growth by 2030'] = data_cleaned['Projected Growth by 2030'].str.rstrip('%').astype('float')

# 3. Removing Duplicates
duplicates = data_cleaned.duplicated().sum()
print(f"Number of duplicates: {duplicates}")

# Remove duplicates
data_cleaned = data_cleaned.drop_duplicates()

Q1 = data_cleaned['Projected Growth by 2030'].quantile(0.25)
Q3 = data_cleaned['Projected Growth by 2030'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
data_cleaned = data_cleaned[(data_cleaned['Projected Growth by 2030'] >= lower_bound) & 
                            (data_cleaned['Projected Growth by 2030'] <= upper_bound)]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_cleaned['Projected Growth by 2030'] = scaler.fit_transform(data_cleaned[['Projected Growth by 2030']])

# Final Cleaned Data
print("Cleaned Data Head:\n", data_cleaned.head())
