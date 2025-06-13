import numpy as np
import math

class SoftmaxRegression:
    def __init__(self, max_iter: int = 10000, learning_rate: float = 0.03, tol: float = 1e-4):
        self.max_iter = max_iter
        self.w = None
        self.w_0 = 0.0
        self.learning_rate = learning_rate
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.w = np.zeros(X.shape[1])
        self.w_0 = 0.0
        epoch = 1
        while epoch <= self.max_iter:
            delta_w, delta_w_0 = self.get_gradient(X, y)
            self.w -= self.learning_rate * delta_w
            self.w_0 -= self.learning_rate * delta_w_0
            epoch += 1
            if np.linalg.norm(delta_w) < self.tol:
                print(f"Converged at epoch {epoch}")
                break

        print(f"Final weights: w = {self.w}, w_0 = {self.w_0}")

    def get_gradient(self, X: np.ndarray, y: np.ndarray):
        miu = softmax(self.w_0 + np.dot(X, self.w.transpose()))
        error = miu - y  # (n_samples,)
        delta_w = np.dot(X.T, error)  # (n_features,)
        delta_w_0 = np.sum(error)  # Scalar

        return delta_w, delta_w_0

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.w_0 + np.dot(X, self.w)
        probabilities = softmax(scores)
        return np.argmax(probabilities, axis=0)  # Chọn lớp có xác suất cao nhất

def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Compute softmax values for each sets of scores in V.
    each column of V is a set of score.    
    """
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 0)
    return A