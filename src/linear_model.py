import numpy as np
from numpy import ndarray
from sklearn import linear_model, svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from abc import ABC


class LinearModel(ABC):
    def __init__(self, X, y):
        """
        This is abstract class for all LiearModels

        :param X: The training X data, n * p array
        :param y: The training labels, n * 1 array  [1, 3, 5, 3, 1], not binarized

        :type y: ndarray
        :type X: ndarray

        :return: accuracy on test
        :rtype: float
        """
        self.X = X
        self.y = label_binarize(y, classes=np.unique(y))

        self.number_of_classes = self.y.shape[1]
        self.number_of_samples, self.number_of_features = X.shape

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=.1
        )

        self.test_size = len(self.y_test)
        self.train_size = len(self.X_test)

        self.model = None

    @property
    def accuracy(self):
        """ Return mean accuracy for every clas label"""
        return np.mean(sum(self.model.predict(self.X_test) == self.y_test) / self.test_size)

    @property
    def roc(self):
        """ Return mean accuracy for every class label
        :returns: It return 4 dict for false positive, true positive, and thresholds for these values
        :rtype: dict, dict, dict
        """

        y_score = self.model.decision_function(self.X_test)

        fpr = {}
        tpr = {}
        thresholds = {}

        for i in range(self.number_of_classes):
            fpr[i], tpr[i], thresholds[i] = roc_curve(self.y_test[:, i], y_score[:, i])

        return fpr, tpr, thresholds


class SupportVectorMachine(LinearModel):

    def __init__(self, X, y):
        """This is very basic logistic regression method to investigate linear model success on data """

        super().__init__(X, y)
        self.classifier = OneVsRestClassifier(
            svm.SVC(
                kernel='linear',
                probability=True)
        )

        self.model = self.classifier.fit(self.X_train, self.y_train)
