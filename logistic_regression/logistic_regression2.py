import numpy as np
import math
import unittest
    

class LogisticRegression2:
    def __init__(self, max_iter: int = 10000, learning_rate: float = 1, tol: float = 1e-4):
        self.max_iter = max_iter
        self.w = None
        self.w_0 = 0.0
        self.learning_rate = learning_rate
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.w = np.zeros(X.shape[1])
        self.w_0 = 0.0
        epoch = 1
        check_w_after = 20
        while epoch <= self.max_iter:
            lr = self.learning_rate / math.sqrt(epoch)
            miu = sigmoid(self.w_0 + np.dot(X, self.w.transpose()))
            delta_w = np.dot(X.transpose(), (miu - y))
            delta_w_0 = np.sum(miu - y)
            if epoch % check_w_after == 0:
                if np.linalg.norm(delta_w) < self.tol and abs(delta_w_0) < self.tol:
                    break
            self.w -= lr * delta_w
            self.w_0 -= lr * delta_w_0
            epoch += 1
            if epoch == self.max_iter:
                print("Max iterations reached")
                print(f"w: {self.w}, w_0: {self.w_0}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (sigmoid(self.w_0 + np.dot(X, self.w.transpose())) >= 0.5).astype(int)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)  
    return 1 / (1 + np.exp(-z))
