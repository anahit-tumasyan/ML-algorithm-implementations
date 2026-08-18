import numpy as np

class LogisticRegression:
    def __init__(self, lr, epochs):
        self.w = 0.0
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1/ (1 + np.exp(-z))

    def fit(self, X, y):
        for epoch in range(self.epochs):

            z = self.w * X + self.b
            p = self.sigmoid(z)
            L = -(np.sum(y * np.log(p) + (1-y)*np.log(1-p)))

            dL_dp = ((1 - y) / (1 - p)) - (y / p)
            dp_dz = p * (1 - p)
            dz_dw = X
            dL_dw = np.sum(dL_dp * dp_dz * dz_dw) # simplified: np.sum((p-y) * X) 
            dz_db = 1
            dL_db = np.sum(dL_dp * dp_dz * dz_db) # simplified: np.sum(p-y)

            new_w = self.w - self.lr * (dL_dw)
            new_b = self.b - self.lr * (dL_db)

            self.w = new_w
            self.b = new_b
            if epoch % 10 == 0:
                print(epoch, L)



    def predict(self, X):
        z = self.w * X + self.b
        p = self.sigmoid(z)
        return (p >= 0.5).astype(int)