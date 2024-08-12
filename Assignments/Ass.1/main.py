#Shady Mohamed Abdelgawad ========> 20200246
#Abdallah Mohamed Gamal   ========> 20200306
#Youssef Samir            ========> 20200653
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def imaged_grid(img, row, col):

    x, y = img.shape
    assert x % row == 0, x % row.format(x, row)
    assert y % col == 0, y % col.format(y, col)

    return img.reshape(x // row, row, -1, col).swapaxes(1, 2).reshape(-1, row, col)


def get_centroid(img):

    feature_vector = []

    for grid in imaged_grid(img, 2, 2):

        X_center = 0
        Y_center = 0
        sum = 0

        for x in range(2):
            for y in range(2):
                X_center = X_center + x * grid[x][y]
                Y_center = Y_center + y * grid[x][y]
                sum += grid[x][y]

        if sum == 0:
            feature_vector.append(0)
            feature_vector.append(0)

        else:
            feature_vector.append(X_center / sum)
            feature_vector.append(Y_center / sum)

    return np.array(feature_vector)



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
    

(Datatrain, Trainlabels), (Datateast, Teastlabels) = mnist.load_data()

Datatrain = Datatrain[0:10000]
Trainlabels = Trainlabels[0:10000]
Datateast = Datateast[0:1000]
Teastlabels = Teastlabels[0:1000]

self = KNN()
train_features = [get_centroid(img) for img in Datatrain]
test_features = [get_centroid(img) for img in Datateast]
KNN_Prediction = KNN.Classifier(self, 3, train_features, test_features, Trainlabels)

wrong_classifier = 0
for i in range(len(Teastlabels)):
    if KNN_Prediction[i] != Teastlabels[i]:
       wrong_classifier += 1
accuracy = 100 - (wrong_classifier / len(Teastlabels)) * 100
print("Scratch Accuracy", accuracy, "%")