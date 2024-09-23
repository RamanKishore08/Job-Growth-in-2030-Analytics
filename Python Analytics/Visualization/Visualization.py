import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Load the dataset
data = pd.read_csv(r"C:\Users\91962\Documents\Data Analytics Project\Dataset.csv")

# Display the first few rows of the dataset
print(data.head())

# Check dataset information and for missing values
print(data.info())
print(data.isnull().sum())

# Data Cleaning
data['Projected Growth by 2030'] = data['Projected Growth by 2030'].str.rstrip('%').astype('float')

# Drop rows with missing values
data_cleaned = data.dropna()

# Now you can visualize the cleaned data
plt.figure(figsize=(8, 6))
sns.histplot(data_cleaned['Projected Growth by 2030'], bins=30, kde=True, color='blue')
plt.title('Distribution of Projected Growth by 2030', fontsize=16)
plt.xlabel('Projected Growth by 2030 (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()
