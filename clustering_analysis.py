import pandas as pd
from sklearn.cluster import KMeans

# Load cleaned data
data = pd.read_csv('data/cleaned_athlete_data.csv')

# Select features for clustering
features = ['Height', 'Weight', 'Age', 'Gold', 'Silver', 'Bronze']

# Apply KMeans clustering
k = 5  # Optimal number of clusters determined via elbow method
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[features])

# Save clustered data
data.to_csv('data/clustered_athlete_data.csv', index=False)
print(f"Clustering complete. Clustered data saved to 'data/clustered_athlete_data.csv'.")
