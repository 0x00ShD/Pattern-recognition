# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the breast cancer dataset
# cancer_data = pd.read_csv('cancer_data.csv')
# X = cancer_data.iloc[:, :-1].values  # Extract features, excluding the last column (target)

# K = 2  # Set the desired number of clusters

# # Randomly initialize centroids
# centroids = X[np.random.choice(X.shape[0], K, replace=False), :]

# def euclidean_distance(x1, x2):
#     return np.sqrt(np.sum((x1 - x2)**2))

# def assign_clusters(X, centroids):
#     clusters = np.zeros(X.shape[0])
#     for i in range(X.shape[0]):
#         distances = [euclidean_distance(X[i], centroid) for centroid in centroids]
#         cluster = np.argmin(distances)
#         clusters[i] = cluster
#     return clusters

# def update_centroids(X, clusters, K):
#     centroids = np.zeros((K, X.shape[1]))
#     for k in range(K):
#         cluster_points = X[clusters == k]
#         if len(cluster_points) > 0:
#             centroids[k] = np.mean(cluster_points, axis=0)
#     return centroids

# def kmeans(X, K, max_iters=100):
#     centroids = X[np.random.choice(X.shape[0], K, replace=False), :]
#     for _ in range(max_iters):
#         clusters = assign_clusters(X, centroids)
#         new_centroids = update_centroids(X, clusters, K)
#         if np.all(centroids == new_centroids):
#             break
#         centroids = new_centroids
#     return clusters, centroids

# clusters, centroids = kmeans(X, K)

# colors = ['red', 'blue']
# labels = ['Cluster 1', 'Cluster 2']
# for i in range(K):
#     cluster_points = X[clusters == i]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=labels[i])

# plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Centroids')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('K-means Clustering of Cancer Dataset')
# plt.legend()
# plt.show()

# for loop 
for i in range(29, 74):
    print(i)