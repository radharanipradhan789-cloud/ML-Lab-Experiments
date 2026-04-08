import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. GIVEN DATA POINTS
point_labels = ['a', 'b', 'c', 'd']
point_values = np.array([
    [1, 1],
    [2, 1],
    [4, 3],
    [5, 4]
])

print("Data points and values:")
for i, label in enumerate(point_labels):
    print(f"{label}: {point_values[i]}")

# 2. APPLY AGGLOMERATIVE CLUSTERING
# We use single linkage as specified in the objective
model = AgglomerativeClustering(n_clusters=2, linkage='single')
cluster_labels = model.fit_predict(point_values)

print("\n--- Agglomerative Clustering Results ---")
print(f"Cluster Labels: {cluster_labels}")

# 3. GROUPING POINTS BY CLUSTER FOR DISPLAY
num_clusters = len(np.unique(cluster_labels))

for i in range(num_clusters):
    # Get labels for points in this cluster
    cluster_points = [point_labels[j] for j, label in enumerate(cluster_labels) if label == i]
    # Get values for points in this cluster
    cluster_vals = [point_values[j].tolist() for j, label in enumerate(cluster_labels) if label == i]
    
    print(f"Cluster {i + 1}: Points {cluster_points}, Values {cluster_vals}")

# 4. VISUALIZATION: DENDROGRAM
# Linkage matrix using the single linkage method
Z = linkage(point_values, method='single')

plt.figure(figsize=(8, 5))
dendrogram(Z, labels=point_labels)
plt.title("Dendrogram (Single Linkage)")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()

#Output
Data points and values:
a: [1 1]
b: [2 1]
c: [4 3]
d: [5 4]

--- Agglomerative Clustering Results ---
Cluster Labels: [1 1 0 0]
Cluster 1: Points ['c', 'd'], Values [[4, 3], [5, 4]]
Cluster 2: Points ['a', 'b'], Values [[1, 1], [2, 1]]
