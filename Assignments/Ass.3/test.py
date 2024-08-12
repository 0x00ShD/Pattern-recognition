import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


iris_data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

X = iris_data.iloc[:, :-1].values
y_true = iris_data.iloc[:, -1].values 

K = 3

centroids = X[np.random.choice(X.shape[0], K, replace=False), :]

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def assign_clusters(X, centroids):
    clusters = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        distances = [euclidean_distance(X[i], centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters[i] = cluster
    return clusters

def update_centroids(X, clusters, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        cluster_points = X[clusters == k]
        centroids[k] = np.mean(cluster_points, axis=0)
    return centroids

def kmeans(X, K, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], K, replace=False), :]
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, K)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

clusters, centroids = kmeans(X, K)

correct = 0
for i in range(K):
    cluster_labels = y_true[clusters == i]
    if len(cluster_labels) > 0:
        most_common_label = np.bincount(cluster_labels).argmax()
        correct += np.sum(cluster_labels == most_common_label)

colors = ['red', 'green', 'blue']
labels = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
for i in range(K):
    cluster_points = X[clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=labels[i])
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Centroids')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-means Clustering of Iris Dataset')
plt.legend()
plt.show()
accuracy = correct / len(y_true)
print("Accuracy:", accuracy)

