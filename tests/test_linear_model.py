from sklearn import datasets
from linear_model import logistic_regression


def test_logistic_regression():
    """Test logistic_regression method from logistic regression"""
    iris = datasets.load_iris()

    iris_X = iris.data
    iris_y = iris.target

    test_accuracy = logistic_regression(iris_X, iris_y)

    assert test_accuracy > 0.5  # test accuracy must be bigger then 0.5
    assert test_accuracy <= 1.0   # test accuracy must be less then 1

    print("""Test accuracy of the logistic_regression on iris data is: {}""".format(test_accuracy))