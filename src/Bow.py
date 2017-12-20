import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import os
from sklearn.externals import joblib


class Bow():
    """This class implements a bag of words model for given data sample. IDeally, the feature vectors are 128 dimensional
    SIFT descriptors. Basically, bag of words model for images works in the following way:
    - Training data (Nx128 matrix of SIFT descriptors) is clustered into K clusters using K-means with L1 distance.
    - For each feature matrix of image/patch, histogram vector is calculated based on nearest cluster means for each data
    sample in the matrix
    - Histogram is normalized into the 'transformed feature vector' of the given patch/image
    """

    def __init__(self, num_centers=500, norm="l1", subsample=0.2, dim=128):
        """parameter are initialized to default values
        """
        self.num_centers = num_centers
        self.norm = norm
        self.dim = dim
        self.features = None
        self.kmeans = None
        self.subsample = subsample

    def fit(self, features):
        """Takes in the training feature matrix and builds the K-Means model with specified number of centers

        :type features: ndarray (N-by-128)
        """
        self.features = features  # store training data for future trials
        if self.subsample != 1:
            self.subsampled_indices = np.random.permutation(np.arange(0, self.features.shape[0]))[
                                      :int(self.features.shape[0] * self.subsample)]
            self.subsampled_features = self.features[self.subsampled_indices, :]
        else:
            self.subsampled_features = self.features
        self.centers = np.empty((self.num_centers, self.features.shape[1]))
        assert (self.subsampled_features.shape[0] > self.num_centers)

        self.kmeans = KMeans(n_clusters=self.num_centers, n_init=10, max_iter=200, verbose=1).fit(self.subsampled_features)
        # self.nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.centers)

        self.dim = self.kmeans.cluster_centers_.shape[1]

    def save(self, path):
        file = os.path.join(path, "kmeans_ncenters_%d_dim_%d_subsample_%f.pkl" % (
            self.kmeans.cluster_centers_.shape[0], self.kmeans.cluster_centers_.shape[1], self.subsample))
        joblib.dump(self.kmeans, file)

    def load(self, path):
        file = os.path.join(path, "kmeans_ncenters_%d_dim_%d_subsample_%f.pkl" % (
            self.num_centers, self.dim, self.subsample))
        self.kmeans = joblib.load(file)

    def transform(self, feature):
        """Given a feature matrix for an image/patch, transform into normalized histogram of cluster centers. Each sample
        vector is assigned to the closest center

        :type feature: ndarray (N-by-128)
        :rtype hist: np.array (1-by-num_centers)
        """

        # shape assertion for avoiding shapes (n,) and (,n)
        assert (feature.ndim == 2)
        if self.kmeans is None:
            raise AssertionError

        # predict the cluster for each data row, and transform into normalized histogram feature vector
        labels = self.kmeans.predict(feature)
        hist, _ = np.histogram(labels.reshape(-1, 1), bins=np.arange(0, self.num_centers, 1))
        return self.__normalize(hist.reshape(1,-1))

    def __normalize(self, feature):
        """given a histogram, normalizes with respect to some norm (L1 by default)"""
        if self.norm is None:
            return feature
        feature_norm = normalize(feature.astype(np.float64), norm=self.norm)
        return feature_norm.reshape(1,-1)


if __name__ == "__main__":
    features = np.random.random((1000, 100))
    bow = Bow(num_centers=100)
    bow.fit(features)
    print(bow.transform(features))
