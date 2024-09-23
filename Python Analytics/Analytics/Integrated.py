import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV file
data = pd.read_csv(r"C:\Users\91962\Documents\Data Analytics Project\Dataset.csv")
print(data.head())


data['Projected Growth by 2030'] = data['Projected Growth by 2030'].str.rstrip('%').astype('float')
print(data[['Domain', 'Job Title', 'Projected Growth by 2030']].head())


# Convert categorical columns to numeric using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['Domain', 'Job Title'])
print(data_encoded.head())


# Define the features (independent variables) and the target (dependent variable)
X = data_encoded.drop(columns=['Projected Growth by 2030'])
y = data_encoded['Projected Growth by 2030']

# Split the data into training (80%) and testing sets (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")



# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Projected Growth by 2030')
plt.ylabel('Predicted Projected Growth by 2030')
plt.title('Actual vs Predicted Growth')
plt.show()
