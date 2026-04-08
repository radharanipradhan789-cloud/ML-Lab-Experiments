import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ==========================================
# 1. GIVEN 1D DATA 
# ==========================================
# Data points provided in the lab sheet
X = np.array([2, 4, 10, 12, 3, 20, 30, 11, 25])

# Reshape data (Required for Sklearn - needs a 2D array)
X_reshaped = X.reshape(-1, 1)

# Apply K-Means with k=2
# Correction: variable names cannot have spaces; used 'kmeans' instead of 'k means'
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
kmeans.fit(X_reshaped)

# ==========================================
# 2. GET RESULTS 
# ==========================================
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Data Points:", X)
print("Cluster Labels:", labels)
print("Centroids:\n", centroids)

# ==========================================
# 3. VISUALIZATION
# ==========================================
plt.figure(figsize=(10, 2))

# Plotting the data points on a 1D line (y=0 for all points)
plt.scatter(X, np.zeros_like(X), c=labels, cmap='viridis', s=100, label='Data Points')

# Plotting the Centroids
plt.scatter(centroids, np.zeros_like(centroids), marker='x', s=200, c='red', label='Centroids')

plt.title("K-means Clustering (1D Data)")
plt.xlabel("Data Points")

# Formatting to hide the Y-axis as it's 1D data
plt.yticks([]) 
plt.legend()
plt.show()

#Output
Data Points: [ 2  4 10 12  3 20 30 11 25]
Cluster Labels: [0 0 0 0 0 1 1 0 1]
Centroids:
 [[ 7.]
 [25.]]
