
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

from mask_to_submission import *


patch_size = 16


def threshold_image(image, threshold):

    if threshold > 255 or threshold < 0:
        print('Invalid threshold value, enter threshold within [0 255] range')
        return image

    img = image.copy()
    img[img < threshold] = np.uint8(0)
    img[img >= threshold] = np.uint8(1)

    return img


def binary_to_submission_image(image, step_size=patch_size):
    prediction_image = np.zeros(image.shape)
    for x in range(0, image.shape[1], step_size):
        for y in range(0, image.shape[0], step_size):
            prediction_image[y: y + step_size, x: x + step_size] = patch_to_label(image[y: y + step_size, x: x + step_size])\
                                                                  * np.ones((step_size, step_size))

    return prediction_image


def sift_main():

    if len(sys.argv) != 4:
        print('Enter the path to prediction grids, binarization threshold and directory for saving prediction images')
        return -1

    file = str(sys.argv[1])
    threshold = float(sys.argv[2])
    path_img = str(sys.argv[3])

    assert (os.path.isfile(file))
    assert (os.path.isdir(path_img))

    images = {}
    image_files = [f for f in listdir(path_img) if isfile(join(path_img, f))]
    for ii, f in enumerate(image_files):
        images[f] = cv2.imread(join(path_img, image_files[ii]), 0)

    prediction_images = joblib.load(file)
    for key, pred_img in prediction_images.items():
        threshold = 1.6 * np.sum(pred_img)/(pred_img.shape[0]*pred_img.shape[1])

        bin_img = threshold_image(pred_img.T, threshold)

        size = 20
        kernel_closure = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        bin_img_ = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_closure)
        bin_img_ = cv2.morphologyEx(bin_img_, cv2.MORPH_OPEN, kernel_opening)

        # sub_img = binary_to_submission_image(bin_img)
        sub_img_ = binary_to_submission_image(bin_img_)

        cv2.imwrite(join(path_img, key), 255*sub_img_)

        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(images[key])
        # ax[1].imshow(sub_img)
        # ax[2].imshow(sub_img_)
        #
        # fig_man = plt.get_current_fig_manager()
        # fig_man.window.showMaximized()
        # plt.show()


def main():

    if len(sys.argv) != 3:
        print('Enter submission file name and directory to predictions of test images')

    submission_filename = str(sys.argv[1])
    prediction_path = str(sys.argv[2])

    assert (os.path.isdir(prediction_path))

    predictions = {}
    prediction_files = [f for f in listdir(prediction_path) if isfile(join(prediction_path, f))]
    for ii, f in enumerate(prediction_files):
        predictions[f] = cv2.imread(join(prediction_path, prediction_files[ii]), 0)

    masks_to_submission(submission_filename, *prediction_files)


if __name__ == '__main__':
    sift_main()