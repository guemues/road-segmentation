import numpy as np
from numpy import ndarray
from sklearn import linear_model


def logistic_regression(X, y):
    """
    This is very basic logistic regression method to investigate linear model success on data
    :param X: The training X data
    :type X: ndarray
    :param y: The training labels
    :type y: ndarray
    :return: accuracy on test
    :rtype: float
    """

    logistic = linear_model.LogisticRegression(C=1e5)

    indices = np.random.permutation(len(X))

    test_len = int(len(indices) / 10)

    X_train = X[indices[:-test_len]]
    y_train = y[indices[:-test_len]]
    X_test = X[indices[-test_len:]]
    y_test = y[indices[-test_len:]]

    model = logistic.fit(X_train, y_train)

    return sum(model.predict(X_test) == y_test) / test_len


