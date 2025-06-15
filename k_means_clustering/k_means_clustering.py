import numpy as np
from scipy.spatial.distance import cdist

class KMeansClustering:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None

    def fit(self, X: np.ndarray):
        """
        Fit the KMeans model to the data.

        Parameters
        ----------
        X : np.ndarray
            The dataset to fit the model to.

        """
        centers = self._initialize_centroids(X)
        it = 0
        while True:
            label = self._assign_labels(X, centers)
            new_centers = self._update_centers(X, label, self.n_clusters)
            if self.has_converged(centers, new_centers):
                break
            centers = new_centers
            self._update_centers(X, label, self.n_clusters)
            it += 1
            # print(f"Iteration {it}: centers = {centers}")
        # print(f"Final centers: {centers}")
        self.cluster_centers_ = centers

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Randomly initialize k centers from the dataset X.

        Parameters
        ----------
        X : np.ndarray
            The dataset from which to initialize the centers.
            
        k : int
            The number of centers to initialize.
            
        Returns
        -------
        np.ndarray
            The initialized centers.     
        """
        # Randomly select k data points as initial centroids
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False) # shape (n_clusters,)
        return X[random_indices] # shape (n_clusters, n_features)
    
    def _assign_labels(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Assign labels to each data point based on the distance to each center.

        Parameters
        ----------
        X : np.ndarray
            The dataset to assign labels to.
        centers : np.ndarray
            The current centers of the clusters.
        
        Returns
        -------
        np.ndarray
            The labels assigned to each data point.
        """
        distances = cdist(X, centers) # shape (n_samples, n_clusters)
        return np.argmin(distances, axis=1) # shape (n_samples,)
    
    def _update_centers(self, X: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
        """
        Update the centers based on the current labels.

        Parameters
        ----------
        x : np.ndarray
            The dataset to update the centers from.
        labels : np.ndarray
            The current labels of the data points.
        K : int
            The number of clusters.

        Returns
        -------
        np.ndarray
            The updated centers.
        """
        centers = np.zeros((K, X.shape[1]))
        for k in range(K):
            # collect all points assigned to the k-th cluster 
            Xk = X[labels == k, :]
            # take average
            centers[k,:] = np.mean(Xk, axis = 0)
        return centers
    
    def has_converged(self, centers: np.ndarray, new_centers: np.ndarray) -> bool:
        """
        Check if the centers have converged.

        Parameters
        ----------
        centers : np.ndarray
            The current centers of the clusters.
        new_centers : np.ndarray
            The new centers of the clusters.

        Returns
        -------
        bool
            True if the centers have converged, False otherwise.
        """
        # return True if two sets of centers are the same
        return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the dataset.

        Returns
        -------
        np.ndarray
            The predicted labels for each data point.
        """
        return self._assign_labels(X, self.cluster_centers_)