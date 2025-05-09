import numpy as np

class LogisticRegression():
    def __init__(self):...

    def fit(self):...

    def predict(self):...

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum() # tra ve 1 vector cung shape voi vector dau vao


if __name__ == "__main__":
    # Example usage
    y = np.array([0, 1, 1])
    print(np.exp(y - np.max(y)))