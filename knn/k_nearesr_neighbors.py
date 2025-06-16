import numpy as np
from scipy.spatial.distance import cdist

class K_nearest_neighbors:
    def __init__(self, n_neighbors: int = 3, p: int = 2, weights: str = 'uniform'):
        self.n_neighbors = n_neighbors
        self.p = p # norm type
        self.X_train = None
        self.y_train = None
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        distances = cdist(X, self.X_train, metric='minkowski', p=self.p)
        y_pred = []
        for i in range(X.shape[0]):
            nearest = np.argpartition(distances[i], self.n_neighbors - 1)[:self.n_neighbors]
            nearest_labels = self.y_train[nearest]
            label_pred = self.__find_label(distances[i], nearest_labels)
            y_pred.append(label_pred)
        return np.array(y_pred)
    
    def __find_label(self, distances, labels):
        match self.weights:
            case 'uniform':
                labels, counts = np.unique(labels, return_counts=True)
                most_common = labels[np.argmax(counts)]
                return most_common
            case 'distance':
                weights = 1 / (distances + 1e-5)  # Avoid division by zero
                weights_counts = dict()
                for label, weight in zip(labels, weights):
                    weights_counts[label] = weights_counts.get(label, 0.0) + weight
                most_common = max(weights_counts, key=weights_counts.get)
                return most_common
            case _:
                raise ValueError("Unknown weights type. Use 'uniform' or 'distance'.")
