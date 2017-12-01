from sklearn import datasets
from linear_model import LogisticRegression


def test_logistic_regression():
    """Test logistic_regression method from logistic regression"""
    iris = datasets.load_iris()

    iris_X = iris.data
    iris_y = iris.target

    linear_model = LogisticRegression(iris_X, iris_y)

    test_accuracy = linear_model.accuracy
    assert test_accuracy > 0.5  # test accuracy must be bigger then 0.5
    assert test_accuracy <= 1.0   # test accuracy must be less then 1

    print("""Test accuracy of the logistic_regression on iris data is: {}""".format(test_accuracy))