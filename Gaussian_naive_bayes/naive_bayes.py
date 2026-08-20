import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.priors = []
        self.mean = []
        self.var = []

    def fit(self, X, y):
        self.classes, count = np.unique(y, return_counts = True)

        for i in range(len(self.classes)):

            prob_class_i = count[i] / sum(count)
            self.priors.append(prob_class_i)

            class_rows = X[y == self.classes[i]]

            for j in range(X.shape[1]):
                feature_values = class_rows[:, j]

                mean = np.mean(feature_values)
                self.mean.append(mean)

                variance = np.var(feature_values)
                self.var.append(variance)

        self.mean = np.array(self.mean).reshape(len(self.classes), X.shape[1])
        self.var = np.array(self.var).reshape(len(self.classes), X.shape[1])

    def gaussian_probability(self, x, mean, var):
        return (1 / (np.sqrt(2 * np.pi * var))) * np.exp(-(((x - mean) ** 2) / (2 * var)))

    def predict(self, x_new):
        scores = []

        for i in range(len(self.classes)):       # classes
            probabilities = []

            for j in range(len(x_new)):          # features
                prob = self.gaussian_probability(x_new[j], self.mean[i, j], self.var[i, j])
                probabilities.append(prob)

            score = np.log(self.priors[i]) + np.sum(np.log(probabilities))
            scores.append(score)

        return self.classes[np.argmax(scores)]

