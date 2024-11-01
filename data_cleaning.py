import pandas as pd

# Load the dataset
data = pd.read_csv('athlete_performance_data.csv')

# Display initial info
print("Initial Data Info:")
print(data.info())

# Drop rows with missing values
data.dropna(inplace=True)

# Reset index after dropping
data.reset_index(drop=True, inplace=True)

# Display cleaned data info
print("Cleaned Data Info:")
print(data.info())

# Save the cleaned data
data.to_csv('cleaned_athlete_performance_data.csv', index=False)
