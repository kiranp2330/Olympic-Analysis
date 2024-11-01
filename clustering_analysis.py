import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
data = pd.read_csv('cleaned_athlete_performance_data.csv')

# Select features for clustering
features = data[['Height', 'Weight', 'YearsActive', 'PerformanceMetric1', 'PerformanceMetric2']]

# Determine optimal number of clusters (k)
# Here, we use the elbow method for simplicity
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

# Plotting the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Perform KMeans clustering with the chosen k (e.g., k=3)
k = 3  # Assume 3 clusters based on the elbow method
kmeans = KMeans(n_clusters=k, random_state=0)
data['Cluster'] = kmeans.fit_predict(features)

# Save the clustered data
data.to_csv('clustered_athlete_performance_data.csv', index=False)
