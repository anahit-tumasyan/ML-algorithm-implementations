import numpy as np

class LinearRegression:
    def __init__(self):
        #initialize model parameters
        self.w = 0.0
        self.b = 0.0
        self.loss_history = []

    def mse(self, y_pred, y_true):
        # Calculate Mean squared error 
        n = len(y_true)
        return 1/n * (np.sum((y_pred - y_true) ** 2))

    def fit(self, X, y, epochs, lr):
       #Train the model using Gradient descent
        print("Training started ...")
        n = len(y)
        for epoch in range(epochs):
            y_pred = self.w * X + self.b

            loss = self.mse(y_pred, y)
            self.loss_history.append(loss)

            new_w = self.w - (lr * (2/n * np.sum((y_pred - y) * X)))
            new_b = self.b - (lr * (2/n * np.sum(y_pred - y)))

            self.w = new_w
            self.b = new_b

        print(f"Current loss is {self.loss_history[-1]}")

            

    def predict(self, X):
        #Make predictions using learned parameters
        y_predict = self.w * X + self.b
        return y_predict
    
X = np.array([1, 2, 3, 4, 5])
y = 3 * X + 5
# X = 1 -> y = 8
# X = 2 -> y = 11
# X = 3 -> y = 14
# X = 4 -> y = 17
# X = 5 -> y = 20 

X_test = np.array([6, 7, 8])

model = LinearRegression()
model.fit(X, y, epochs = 1000, lr = 0.01)
print(model.w)
print(model.b)
print(model.predict(X_test))