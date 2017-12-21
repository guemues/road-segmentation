
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import f1_score

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
    '''Given a unint8 2D numpy array (grayscale image) with values 0 to 255, threshold it into a binary uint8 image'''

    if threshold > 255 or threshold < 0:
        print('Invalid threshold value, enter threshold within [0 255] range')
        return image

    img = image.copy()
    img[img < threshold] = np.uint8(0)
    img[img >= threshold] = np.uint8(1)

    return img


def binary_to_submission_image(image, step_size=patch_size):
    '''The method reproduces the submission structure with 16x16 patches classified as road/background so that we conduct
    postprocessing accordingly (binary image morphology) as the submission procedure demands smoothing of predictions'''

    prediction_image = np.zeros(image.shape)
    for x in range(0, image.shape[1], step_size):
        for y in range(0, image.shape[0], step_size):
            # write the smoothed patch into the output
            prediction_image[y: y + step_size, x: x + step_size] = patch_to_label(image[y: y + step_size, x: x + step_size])\
                                                                  * np.ones((step_size, step_size))

    return prediction_image


def compute_validation_threshold(path_predictions, path_groundtruth, epoch_tail='epoch1.png'):
    '''Given the path to pixel predictions for validation data with confidence values, path to validation groundtruth
    folder and epoch of the neural net predictions, compute the optimal threshold that maximizes mean f score (micro
    averaged f score to be formal) over all validation set'''

    # if len(sys.argv) != 3:
    #     print('Enter path to predictions for validation set and path to groundtruth for validation examples')
    #     return -1
    #
    # path_predictions = str(sys.argv[1])
    # path_groundtruth = str(sys.argv[2])

    assert (os.path.isdir(path_predictions))
    assert (os.path.isdir(path_groundtruth))

    # read files in the directories into lists
    prediction_files = [f for f in listdir(path_predictions) if isfile(join(path_predictions, f)) and f.endswith(epoch_tail)]
    groundtruth_files = [f for f in listdir(path_groundtruth) if isfile(join(path_groundtruth, f))]

    # define a linear space of threshold values to be checked
    thresholds = np.linspace(10, 250, 25)

    mean_fscore_mat = np.zeros((len(groundtruth_files), thresholds.shape[0]))
    i = 0
    for ii, f in enumerate(prediction_files):
        original_file = f[:f.find('.png')+4] # extract the name of the original training image file
        if original_file in groundtruth_files:

            # read the confidence image and respective groundtruth(binarize it with threshold 128)
            prediction = cv2.imread(join(path_predictions, f), 0)
            truth = cv2.imread(join(path_groundtruth, original_file), 0)
            truth = threshold_image(truth, 128)

            # for all possible threshold values, binarize confidence predictions and compute Mean F-Score using the
            # groundtruth image. Record the value to designated (image,threshold) bin
            for j, threshold in enumerate(thresholds):
                prediction_binary = threshold_image(prediction, threshold)
                prediction_binary = binary_to_submission_image(prediction_binary)
                mean_fscore = f1_score(truth.reshape(-1), prediction_binary.reshape(-1), average='micro')
                mean_fscore_mat[i, j] = mean_fscore

            i = i + 1

    # get the threshold for which the average of Mean F-Score is the greatest for all validation images
    average_fscore_thresholds = np.mean(mean_fscore_mat, axis=0)
    best_threshold_idx = np.argmax(average_fscore_thresholds)
    best_threshold = thresholds[best_threshold_idx]

    return best_threshold_idx, best_threshold, average_fscore_thresholds


def confidence_to_submission_image(path_predictions, path_groundtruth, path_test_predictions, path_save, epoch_tail):
    '''Given path to all predicted images, path to groundtruth of validation set, path to test_predictions(may be the same
    as path to all predicted images), path to save the submission images and the epoch whose results to be used; the
    function computes the best validation threshold and generates binary submission images based on that value. Optionally,
    user can choose to apply basic binary image morphology such as opening and closing as post processing step'''

    assert (os.path.isdir(path_test_predictions))
    assert (os.path.isdir(path_save))

    # get the best performing validation threshold
    best_threshold_idx, best_threshold, average_fscore_thresholds = compute_validation_threshold(path_predictions,
                                                                                                 path_groundtruth,
                                                                                                 epoch_tail=epoch_tail)

    print('Average Mean F-Scores for various thresholds:\n{}'.format(average_fscore_thresholds))
    print('\nBest Threshold {} ==> {}\n'.format(best_threshold, average_fscore_thresholds[best_threshold_idx]))

    best_threshold = 160

    # store the test confidence prediction image file names
    prediction_files = [f for f in listdir(path_test_predictions) if isfile(join(path_test_predictions, f))
                        and f.startswith('test') and f.endswith(epoch_tail)]

    for f in prediction_files:
        original_file = f[:f.find('.png')+4]

        # read the test confidence image and binarize
        prediction = cv2.imread(join(path_predictions, f), 0)
        prediction_binary = threshold_image(prediction, best_threshold)

        # apply closing then opening to connect nearby road segments and remove random road predictions that are just a
        # few pixels and relatively separate from main segments of road (possible noisy predictions)
        size = 5
        kernel_closure = cv2.getStructuringElement(cv2.MORPH_RECT, (size+10, size+10))
        kernel_opening_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        kernel_opening_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (size+5, size+5))
        # prediction_binary = cv2.morphologyEx(prediction_binary, cv2.MORPH_OPEN, kernel_opening_1)
        prediction_binary = cv2.morphologyEx(prediction_binary, cv2.MORPH_CLOSE, kernel_closure)
        prediction_binary = cv2.morphologyEx(prediction_binary, cv2.MORPH_OPEN, kernel_opening_2)

        # write the binary image into file for submission
        cv2.imwrite(join(path_save, original_file), prediction_binary)


def prediction_main():
    '''Pseudo main function that generates submission image files using the previous 2 methods'''

    if len(sys.argv) != 6:
        print('Please enter the following in order:\nPath to all predictions\nPath to validation groundtruth'
              + '\nPath to test confidence predictions\nPath to save final predictions\nExtention of epoch to be valdiated')
        return -1

    path_predictions = str(sys.argv[1])
    path_groundtruth = str(sys.argv[2])
    path_test_predictions = str(sys.argv[3])
    path_save = str(sys.argv[4])
    epoch_tail = str(sys.argv[5])

    confidence_to_submission_image(path_predictions, path_groundtruth, path_test_predictions, path_save,
                                   epoch_tail=epoch_tail)


def sift_main():
    '''Pseudo main method that read prediction confidence images, adaptively thresholds them (refer to the report), applies
    simple binary image morphological operator (first closing then opening) to merge nearby chunks of road segment but
    remove noisey small distant predictions of road segments. Save the images to saving directory'''

    if len(sys.argv) != 4:
        print('Enter the path to prediction grids, binarization threshold and directory for saving prediction images')
        return -1

    # get the file that stores confidence prediction images in a dict indexed by file names, threshold and path to save
    file = str(sys.argv[1])
    threshold = float(sys.argv[2])
    path_img = str(sys.argv[3])

    assert (os.path.isfile(file))
    assert (os.path.isdir(path_img))

    # read the confidence prediction files
    images = {}
    image_files = [f for f in listdir(path_img) if isfile(join(path_img, f))]
    for ii, f in enumerate(image_files):
        images[f] = cv2.imread(join(path_img, image_files[ii]), 0)

    # load predictions as confidence images into memory
    prediction_images = joblib.load(file)
    for key, pred_img in prediction_images.items():
        threshold = 1.6 * np.sum(pred_img)/(pred_img.shape[0]*pred_img.shape[1])

        # threshold the images into binary
        bin_img = threshold_image(pred_img.T, threshold)

        # apply morphological operators to smooth road segment and get rid of noise as much as possible
        size = 20
        kernel_closure = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        bin_img_ = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_closure)
        bin_img_ = cv2.morphologyEx(bin_img_, cv2.MORPH_OPEN, kernel_opening)

        # sub_img = binary_to_submission_image(bin_img)
        # sub_img_ = binary_to_submission_image(bin_img_)

        cv2.imwrite(join(path_img, key), bin_img_)

        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(images[key])
        # ax[1].imshow(sub_img)
        # ax[2].imshow(sub_img_)
        #
        # fig_man = plt.get_current_fig_manager()
        # fig_man.window.showMaximized()
        # plt.show()


def main():
    '''Main function that generates submission images from confidence images outputted by classifier'''

    if len(sys.argv) != 3:
        print('Enter submission file name and directory to predictions of test images')

    submission_filename = str(sys.argv[1])
    prediction_path = str(sys.argv[2])

    assert (os.path.isdir(prediction_path))

    prediction_files = [join(prediction_path, f) for f in listdir(prediction_path) if isfile(join(prediction_path, f))]
    prediction_files.sort(key=lambda x: int(re.search(r"\d+", x).group(0)))
    masks_to_submission(submission_filename, *prediction_files)


if __name__ == '__main__':
    prediction_main()
    # main()