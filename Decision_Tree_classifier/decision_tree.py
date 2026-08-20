import numpy as np
import statistics

class Node:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

class DecisionTree:
    def __init__(self, max_depth):
        self.root = None
        self.max_depth = max_depth

    def gini(self, y):
        classes, counts = np.unique(y, return_counts = True)
        probabilities = []
        for i in range(len(classes)):
            probability = counts[i] / len(y)
            probabilities.append(probability)
        probabilities = np.array(probabilities)
        return 1 - np.sum((probabilities) ** 2)

    def best_split(self, X, y):
        best_gini = np.inf
        best_feature = 0
        best_threshold = 0

        for i in range(X.shape[1]):
            feature_values = X[:, i]
            sorted_values = np.sort(feature_values)

            for j in range(len(feature_values) - 1):
                threshold = np.mean([sorted_values[j], sorted_values[j+1]])

                left_X = X[feature_values < threshold]
                right_X = X[feature_values >= threshold]
                left_y = y[feature_values < threshold]
                right_y = y[feature_values >= threshold]

                gini_left = self.gini(left_y)
                gini_right = self.gini(right_y)

                gini_weighted = (gini_left * (len(left_y) / len(X))) + (gini_right * (len(right_y) / len(X)))
                if gini_weighted < best_gini:
                    best_gini = gini_weighted
                    best_feature = i
                    best_threshold = threshold

        return best_gini, best_feature, best_threshold
    
    def build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            node = Node()
            node.value = statistics.mode(y)
            return node
        else:
            best_gini, best_feature, best_threshold = self.best_split(X, y)

            left_X = X[X[:, best_feature]< best_threshold]
            left_y = y[X[:, best_feature]< best_threshold]
            right_X = X[X[:, best_feature] >= best_threshold]
            right_y = y[X[:, best_feature] >= best_threshold]

            left_child = self.build_tree(left_X, left_y, depth + 1)
            right_child = self.build_tree(right_X, right_y, depth + 1)

            node = Node()
            node.feature = best_feature
            node.threshold = best_threshold
            node.left = left_child
            node.right = right_child
            return node

    def fit(self, X, y):
        self.root = self.build_tree(X, y, 0)
        return self.root

    def predict(self, X):
        predictions = []
        for sample in X:
            node = self.root

            while node.value is None:
                if sample[node.feature] < node.threshold:
                    node = node.left
                else:
                    node = node.right

            predictions.append(node.value)

        return np.array(predictions)