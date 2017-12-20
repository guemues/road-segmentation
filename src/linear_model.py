import numpy as np
from numpy import ndarray
from sklearn import linear_model, svm
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from abc import ABC
from sklearn.model_selection import KFold


def require_train(f):
    def wrapper(self, *args):
        if self.model is None:
            raise Exception("Model is not trained")
        return f(self, *args)

    return wrapper


class LinearModel:

    def __init__(self, X, y, classifier):
        """
        This is abstract class for all LinearModels

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

        self.classifier = classifier

        self.model = None
        # self.model = self.classifier.fit(self.X_train, self.y_train)

    def cross_validation(self):
        accuracies = []
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(self.X_train):
            X_train, X_validation = self.X_train[train_index], self.X_train[test_index]
            y_train, y_validation = self.y_train[train_index], self.y_train[test_index]

            model_validation = self.classifier.fit(X_train, y_train)
            accuracies.append(np.mean(sum(model_validation.predict(X_validation) == y_validation) / len(y_validation)))
        cross_validation_accuracy = np.mean(accuracies)
        print("""Cross validation accuracy with 10 fold: {}""".format(cross_validation_accuracy))
        return cross_validation_accuracy

    def train(self):
        self.model = self.classifier.fit(self.X_train, self.y_train)

    @property
    @require_train
    def accuracy(self):
        """ Return mean accuracy for every class label"""
        if self.model is None:
            raise Exception("Model is not trained")

        return np.mean(sum(self.model.predict(self.X_test) == self.y_test) / self.test_size)

    @property
    @require_train
    def roc(self):
        """ Return mean accuracy for every class label
        :returns: It returns 4 dict for false positive, true positive, and thresholds for these values
        :rtype: dict, dict, dict
        """
        if self.model is None:
            raise Exception("Model is not trained")

        y_score = self.model.decision_function(self.X_test)

        fpr = {}
        tpr = {}
        thresholds = {}

        for i in range(self.number_of_classes):
            fpr[i], tpr[i], thresholds[i] = roc_curve(self.y_test[:, i], y_score[:, i])

        return fpr, tpr, thresholds
