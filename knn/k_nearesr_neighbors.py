import numpy as np


class K_nearest_neighbors:
    def __init__(self, n_neighbors: int = 3, p: int = 2):
        self.n_neighbors = n_neighbors
        self.p = p # norm type
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            distances = self.__compute_distances(X[i])
            nearest = np.argpartition(distances, self.n_neighbors - 1)[:self.n_neighbors]
            nearest_labels = self.y_train[nearest]
            most_common = np.bincount(nearest_labels).argmax()
            y_pred.append(most_common)
        return np.array(y_pred)
    
    def __compute_distances(self, X):
        return np.linalg.norm(self.X_train - X, ord=self.p, axis=1)