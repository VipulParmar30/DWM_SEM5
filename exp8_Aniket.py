# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data using make_blobs (for demonstration)
# You can replace this with your own data, e.g., X = your_data_array
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Create and fit the K-Means model
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

# Get the cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the data points and the cluster centers
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='X')  # Red markers for cluster centers
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()