import pandas as pd

# Example: Load dataset from Kaggle
# Replace 'YOUR_DATASET_PATH' with the actual path to your dataset on Kaggle
dataset_url = 'https://www.kaggle.com/datasets/someuser/athlete-performance-dataset'
data = pd.read_csv(dataset_url)

# Display the first few rows of the dataset
print(data.head())

# Save the dataset locally for further processing
data.to_csv('athlete_performance_data.csv', index=False)
