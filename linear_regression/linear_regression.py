import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        x0 = np.ones((X.shape[0], 1))
        Xbar = np.concatenate((x0, X), axis=1)
        A = np.dot(Xbar.T, Xbar)
        b = np.dot(Xbar.T, y)
        A_dagger = np.linalg.inv(A)
        self.w = np.dot(A_dagger, b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        x0 = np.ones((X.shape[0], 1))
        Xbar = np.concatenate((x0, X), axis=1)
        y_pred = np.dot(Xbar, self.w)
        return y_pred
    
if __name__ == "__main__":
    data = load_diabetes()
    X = data.data
    y = data.target
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(np.sum((y_val - y_pred) ** 2))
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_val)
    print(np.sum((y_val - y_pred_sklearn) ** 2))
    # Test on test set
    y_pred = model.predict(X_test)
    print(np.sum((y_test - y_pred) ** 2))
    y_pred_sklearn = sklearn_model.predict(X_test)
    print(np.sum((y_test - y_pred_sklearn) ** 2))

    print("Model weights: ", model.w)
    print("Sklearn model weights: ", sklearn_model.coef_)