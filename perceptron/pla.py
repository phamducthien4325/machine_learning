import numpy as np

class perceptron:
    def __init__(self):
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_train = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
        n_features, n_samples = X_train.shape
        self.weights = self.__initialize_weights(n_features + 1)
        while not self.__has_converged(X_train, y):
            for i in range(n_samples):
                if y[i] * self.__cost(X_train[i]) <= 0:
                    self.weights += y[i] * X_train[i]
        

    def __has_converged(self, X: np.ndarray, y: np.ndarray) -> bool:
        predictions = self.predict(X)
        return np.all(predictions == y)

    def __initialize_weights(self, n_features: int):
        return np.random.randn(n_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_test = np.hstack((np.ones((X.shape[0], 1)), X))
        predictions = self.__cost(X_test)
        return np.sign(predictions).astype(int)

    def __cost(self, X: np.ndarray)  -> float:
        return np.dot(self.weights.reshape(-1, 1), X)

if __name__ == "__main__":
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([1, -1, 1])
    model = perceptron()
    model.fit(X, y)