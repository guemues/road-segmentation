
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC

import cv2, os
import sys, time
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isfile

from utilities import *
from Bow import *
from Sift import *
from Trainer import *
from Tester import *
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

def visualize_image(pred, I, key, threshold, plot_original=True, plot_threshold=False):

    if plot_original:
        fig, ax = plt.subplots(1, 2)
        img_test = ax[0].imshow(I)
        img_pred = ax[1].imshow(pred.T)
        plt.title('Image {} - Predicted'.format(key))
        # fig.colorbar(img_pred, ax=ax[1])

        fig_man = plt.get_current_fig_manager()
        fig_man.window.showMaximized()
        plt.show()

    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1
    pred = pred.astype(np.uint8)

    if plot_threshold:
        plt.imshow(pred.T, cmap='Greys',  interpolation='nearest')
        plt.title('Image {} - Thresholded at {}'.format(key, threshold))
        plt.colorbar()
        plt.show()


def main():

    if len(sys.argv) != 4:
        print('Enter the path to test images, predictions file, and threshold for binarizing the image within the range [0, 1]')
        return -1

    path_img = str(sys.argv[1])
    path_pred = str(sys.argv[2])
    threshold = np.float(sys.argv[3])

    assert (os.path.isdir(path_img))
    images = {}
    image_files = [f for f in listdir(path_img) if isfile(join(path_img, f))]
    for i, f in enumerate(image_files):
        images[f] = cv2.imread(join(path_img, image_files[i]), 0)

    predictions = joblib.load(path_pred)

    if type(predictions) != dict:
        print('Data structure is not a dictionary!\nAborting')
        return -1

    for key, pred in predictions.items():
        visualize_image(pred, images[key], key, threshold, plot_original=True, plot_threshold=False)


if __name__ == "__main__":
    main()
