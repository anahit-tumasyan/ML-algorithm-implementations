import numpy as np
import statistics

class KNN:
    def __init__(self, k, distance_metric):
        self.k = k
        self.distance_metric = distance_metric
    
    def distance_metrics(self, new_X, X):
        if self.distance_metric == "Euclidean":
            return np.sqrt(np.sum((new_X  - X) ** 2))
        if self.distance_metric == "Manhattan":
            return np.sum(np.abs(new_X - X))
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test, problem):
        distances = []

        for i in range(len(self.X_train)):
            dist = self.distance_metrics(X_test, self.X_train[i])
            distances.append(dist)
        sorted_by_indices = np.argsort(distances)[:self.k]

        if problem == "Classification":
            label = self.y_train[sorted_by_indices]
            return statistics.mode(label)
        
        if problem == "Regression":
            label = self.y_train[sorted_by_indices]
            return np.average(label)
    