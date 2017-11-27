import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class Bow():
    def __init__(self, num_centers=1000, norm="l1"):
        self.num_centers = num_centers
        self.norm = norm
        self.features = None
        self.kmeans = None

    def fit(self, features):
        self.features = features
        self.centers = np.empty((self.num_centers, self.features.shape[1]))
        assert (self.features.shape[0] > self.num_centers)

        self.kmeans = KMeans(n_clusters=self.num_centers, verbose=1).fit(self.features)
        #self.nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.centers)

    def transform(self, feature):
        assert (feature.ndim == 2)
        if self.kmeans is None:
            raise AssertionError
        labels = self.kmeans.predict(feature)
        hist, _ = np.histogram(labels.reshape(-1,1), bins=np.arange(0,self.num_centers,1))
        return self.__normalize(hist)

    def __normalize(self, feature):
        if self.norm is None:
            return feature
        feature_norm = normalize(feature.astype(np.float64), norm=self.norm)
        return feature_norm.reshape(-1,)

if __name__=="__main__":
    features = np.random.random((1000,100))
    bow = Bow(num_centers=100)
    bow.fit(features)
    print(bow.transform(features))




