import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load clustered data
data = pd.read_csv('clustered_athlete_performance_data.csv')

# Plotting the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='PerformanceMetric1', y='PerformanceMetric2', hue='Cluster', palette='Set1')
plt.title('Athlete Performance Clusters')
plt.xlabel('Performance Metric 1')
plt.ylabel('Performance Metric 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.savefig('athlete_performance_clusters.png')
plt.show()
