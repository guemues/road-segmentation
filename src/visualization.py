import matplotlib.pyplot as plt


def visualize_roc(fpr, tpr):
    """
    A method for visualize ROC

    :param tpr: True positive ratios
    :type tpr: dict
    :param tpr: False positive ratios
    :type fpr: dict
    """

    plt.close('all')
    for i, _ in enumerate(fpr):
        plt.plot(fpr[i], tpr[i])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
