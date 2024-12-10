import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load cleaned data
data = pd.read_csv('data/cleaned_athlete_data.csv')

# Normalize data for clustering (if not already normalized)
scaler = StandardScaler()
features = ['Height', 'Weight', 'Age']
normalized_data = scaler.fit_transform(data[features])

# Elbow Method for Optimal k
inertia = []
k_values = range(1, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_data)
    inertia.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig('images/elbow_graph.png', dpi=300)
plt.show()

# Apply KMeans with optimal k
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(normalized_data)

# Scatter plot to visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Height', y='Weight', hue='Cluster', palette='Set1')
plt.title('Athlete Performance Clusters')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend(title='Cluster')
plt.grid(True)
plt.savefig('images/athlete_performance_clusters.png', dpi=300)
plt.show()
