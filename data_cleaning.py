import pandas as pd

# Load raw dataset
data = pd.read_csv('data/athlete_events.csv')

# Drop rows with missing values in critical columns
data = data.dropna(subset=['Height', 'Weight', 'Age'])

# Normalize continuous variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['Height', 'Weight', 'Age']] = scaler.fit_transform(data[['Height', 'Weight', 'Age']])

# One-hot encode the 'Medal' column
data = pd.get_dummies(data, columns=['Medal'], prefix='', prefix_sep='')

# Save cleaned data
data.to_csv('data/cleaned_athlete_data.csv', index=False)
print("Data cleaning complete. Cleaned data saved to 'data/cleaned_athlete_data.csv'.")

