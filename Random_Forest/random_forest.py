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
    def __init__(self, max_depth, max_features):
        self.root = None
        self.max_depth = max_depth
        self.max_features = max_features

    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)

        probabilities = counts / len(y)

        return 1 - np.sum(probabilities ** 2)

    def best_split(self, X, y):
        best_gini = np.inf
        best_threshold = 0
        best_feature = 0

        features = np.arange(X.shape[1])

        selected_features = np.random.choice(
            features,
            size=self.max_features,
            replace=False
        )

        for feature in selected_features:
            feature_values = X[:, feature]
            sorted_values = np.sort(feature_values)

            for j in range(len(feature_values) - 1):
                threshold = np.mean(
                    [sorted_values[j], sorted_values[j + 1]]
                )

                left_mask = feature_values < threshold
                right_mask = feature_values >= threshold

                left_y = y[left_mask]
                right_y = y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gini_left = self.gini(left_y)
                gini_right = self.gini(right_y)

                gini_weighted = (
                    gini_left * (len(left_y) / len(X))
                    + gini_right * (len(right_y) / len(X))
                )

                if gini_weighted < best_gini:
                    best_gini = gini_weighted
                    best_feature = feature
                    best_threshold = threshold

        return best_gini, best_feature, best_threshold

    def build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            node = Node()
            node.value = statistics.mode(y)
            return node

        best_gini, best_feature, best_threshold = self.best_split(X, y)

        if best_gini == np.inf:
            node = Node()
            node.value = statistics.mode(y)
            return node

        left_mask = X[:, best_feature] < best_threshold
        right_mask = X[:, best_feature] >= best_threshold

        left_X = X[left_mask]
        left_y = y[left_mask]

        right_X = X[right_mask]
        right_y = y[right_mask]

        left_child = self.build_tree(
            left_X,
            left_y,
            depth + 1
        )

        right_child = self.build_tree(
            right_X,
            right_y,
            depth + 1
        )

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


class RandomForest:
    def __init__(self, max_depth, max_features, n_trees=100):
        self.forest = []
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features

    def fit(self, X, y):
        self.forest = []

        sample_indices = np.arange(X.shape[0])

        for _ in range(self.n_trees):
            bootstrap_sample = np.random.choice(
                sample_indices,
                size=X.shape[0],
                replace=True
            )

            X_bootstrap = X[bootstrap_sample]
            y_bootstrap = y[bootstrap_sample]

            tree = DecisionTree(
                self.max_depth,
                self.max_features
            )

            tree.fit(X_bootstrap, y_bootstrap)

            self.forest.append(tree)

        return self.forest

    def predict(self, X):
        predictions = []

        for tree in self.forest:
            predictions.append(tree.predict(X))

        predictions = np.array(predictions)

        final_predictions = []

        for sample_predictions in predictions.T:
            final_predictions.append(
                statistics.mode(sample_predictions)
            )

        return np.array(final_predictions)