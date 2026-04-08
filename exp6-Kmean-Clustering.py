import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Given data points
X = np.array([
    [2, 3],
    [3, 4],
    [6, 6],
    [7, 7]
])

# Apply K-Means with k=2
# Correction: Changed 'k means' to 'kmeans' (variable names cannot have spaces)
kmeans = KMeans(n_clusters=2, init='random', n_init=10, random_state=0)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Get centroids
centroids = kmeans.cluster_centers_

print("Cluster labels:", labels)
print("Centroids:\n", centroids)

# Visualization
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red', label='Centroids')

plt.title("K-Means Clustering")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

# Output
Cluster labels: [1 1 0 0]
Centroids:
[[6.5 6.5]
[2.5 3.5]]
