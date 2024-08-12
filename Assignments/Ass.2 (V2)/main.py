# Import required libraries
import os
import numpy as np
from skimage import io, feature
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Define the path to the dataset
data_path = '/path/to/FMD/'

# Define the categories and number of images per category
categories = ['fabric', 'foliage', 'glass', 'leather', 'metal', 'paper', 'plastic', 'stone', 'water', 'wood']
num_images = 100

# Define the GLCM features to use
glcm_features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

def calculate_glcm(image, d=1, theta=0, levels=256):
    
    # Convert the input image to a 1D array
    image = np.ravel(image)
    
    # Define the offset between the pixel pairs
    dx = d * np.cos(np.deg2rad(theta))
    dy = d * np.sin(np.deg2rad(theta))
    offset = int(dx + dy * image.shape[1])
    
    # Compute the co-occurrence matrix
    glcm = np.zeros((levels, levels))
    for i in range(len(image) - offset):
        x = int(image[i])
        y = int(image[i + offset])
        glcm[x, y] += 1
    
    # Normalize the GLCM
    glcm = glcm / np.sum(glcm)
    
    return glcm

import numpy as np

def calculate_contrast(glcm):
    
    # Compute the contrast value
    contrast = np.sum(glcm * np.square(np.arange(glcm.shape[0]) - np.arange(glcm.shape[1])))

    return contrast

def calculate_correlation(glcm):
    
    # Compute the correlation value
    i, j = np.meshgrid(np.arange(glcm.shape[0]), np.arange(glcm.shape[1]))
    mu_i = np.sum(i * glcm)
    mu_j = np.sum(j * glcm)
    sigma_i = np.sqrt(np.sum(np.square(i - mu_i) * glcm))
    sigma_j = np.sqrt(np.sum(np.square(j - mu_j) * glcm))
    correlation = np.sum(((i - mu_i) * (j - mu_j) * glcm) / (sigma_i * sigma_j))

    return correlation

def calculate_energy(glcm):
    
    # Compute the energy value
    energy = np.sum(np.square(glcm))

    return energy

def calculate_homogeneity(glcm):
    
    # Compute the homogeneity value
    homogeneity = np.sum(glcm / (1 + np.abs(np.arange(glcm.shape[0]) - np.arange(glcm.shape[1]))))

    return homogeneity


def get_glcm_features(image):
    # Convert the image to grayscale
    gray = np.mean(image, axis=2)

    # Compute the GLCM matrix
    glcm = calculate_glcm(gray,5,0,256)
    
    
    # Compute the contrast, correlation, energy, and homogeneity features
    contrast = calculate_contrast(glcm)
    correlation = calculate_correlation(glcm)
    energy = calculate_energy(glcm)
    homogeneity = calculate_homogeneity(glcm)
    
    # Return the feature vector
    return np.array([contrast, correlation, energy, homogeneity])

# Initialize lists to store features and labels
features = []
labels = []

# Loop over each category
for category in categories:
    # Loop over each image in the category
    for i in range(1, num_images+1):
        # Define the path to the image
        img_path = os.path.join(data_path, category, f'{category}_{i}.jpg')
        # Load the image
        img = io.imread(img_path, as_gray=True)
        # Calculate the GLCM matrix and its features
        glcm = feature.greycomatrix(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        glcm_feats = feature.greycoprops(glcm, glcm_features).ravel()
        # Add the features and label to the lists
        features.append(glcm_feats)
        labels.append(category)

# Convert the features and labels to arrays
X = np.array(features)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the KNN classifier with k=1
class KNN:
    distances = [[] * 2]
    final_label = []

    def getDistance(self, test_vector, train_feature_vectors, train_labels):
        for i in range(len(train_feature_vectors)):
            distance = np.linalg.norm(
                test_vector - train_feature_vectors[i]
            )  
            self.distances.append(
                [distance, train_labels[i]]
            )  
        self.distances = sorted(
            self.distances, key=lambda x: x[0]
        )  
        return self.distances

    def getLabel(self, k):
        labels = []
        for i in range(k):
            labels.append(self.distances[i][1])
        return labels

    def getNearestNeighbor(self, k):
        labels = self.getLabel(k)
        return max(set(labels), key=labels.count)

    def Classifier(self, k, train_features, test_features, Trainlabels):
        for i in range(len(test_features)):
            self.distances = []
            self.getDistance(
                test_features[i], train_features, Trainlabels
            )  
            self.final_label.append(
                self.getNearestNeighbor(k)
            ) 
        return self.final_label

self = KNN()
# Predict the labels for the testing data

y_pred = KNN.predict(X_test)

# Test the classifier using the testing data
accuracy = KNN.score(X_test, y_test)

# Print the accuracy
print(f'Accuracy {accuracy:.2f}')
