
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

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
from submit_predictions import *

'''This file is the ultimate train/test script for SIFT-BOW-LOGISTIC model. For model training part it does the following:
- reads the images,
- generates SIFT features,
- trains bag of words model,
- generates training image patches by producing respective normalized cluster centroid histogram based on features in the patch,
- trains a logistic regression model

For model testing, it does the following:
- extracts SIFT features from test images
- for each patch (via a sliding window), create normalized cluster centroid histogram feature vector
- using the prediction confidence per patch, generate a grid of confidence values as output'''

def main():

    if len(sys.argv) != 5:
        print('Please enter directories for training images, groundtruth images, test images and path to saving models')
        return

    data_dir = str(sys.argv[1])
    label_dir = str(sys.argv[2])
    test_dir = str(sys.argv[3])
    path = str(sys.argv[4])

    load_bow = False
    save_bow = True
    load_sift = False
    save_sift = True
    load_model = False
    save_model = True
    init = True

    # Initialize the Tester class, which takes care of processing images, extracting SIFT features, constructing bag
    # of words model, generating image crops from training images and annotating them as road/background/none, training
    # linear classifier and predicting pixel labels on the test images using sliding window
    tester = Tester()
    tester.fit(data_dir, label_dir, load_bow=load_bow, save_bow=save_bow, load_sift=load_sift, save_sift=save_sift,
               load_model=load_model, save_model=save_model, path=path, init=init)

    # use validation data for threshold optimization and mean f score computation for the validation set.
    validation_data = joblib.load('validation_data.pkl')


    # store predictions in dictionary indexed by test file names
    predictions = {}
    image_files = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]
    test_count = len(image_files)
    file = os.path.join(path, "prediction_grids.pkl")
    for i, f in enumerate(image_files):
        print('Transforming test image({}) {}/{}'.format(f, i + 1, test_count))
        # threshold uint8 test images into {0,1}
        tic = time.time()
        I = cv2.imread(join(test_dir, image_files[i]), 0)
        predictions[f] = tester.extract_transform(I)
        print('Transforming test image {}/{} done in {}'.format(i + 1, test_count, time.time() - tic))

    joblib.dump(predictions, file)


    # # FOR TEST PURPOSES ONLY, DO NOT RUN THE COMMENTED PART
    # # Cross Validation of the linear model
    #
    # X = 0
    # y = 0
    # if tester.data_generation_mode == 'image':
    #     # check if training set is populated beforehand
    #     assert (tester.model.training_set_road.size > 0)
    #     assert (tester.model.training_set_background.size > 0)
    #
    #     # prepare training feature matrix and labels
    #     X = np.concatenate((tester.model.training_set_road, tester.model.training_set_background), axis=0)
    #     y = np.concatenate(
    #         (np.ones(tester.model.training_set_road.shape[0]), np.zeros(tester.model.training_set_background.shape[0])), axis=0)
    # else:
    #     # check if training patches are populated beforehand
    #     assert (tester.model.training_patch_road.size > 0)
    #     assert (tester.model.training_patch_background.size > 0)
    #
    #     # prepare training feature matrix and labels
    #     X = np.concatenate((tester.model.training_patch_road, tester.model.training_patch_background), axis=0)
    #     y = np.concatenate(
    #         (np.ones(tester.model.training_patch_road.shape[0]), np.zeros(tester.model.training_patch_background.shape[0])), axis=0)
    #
    # print('Road to Background ratio: {}'.format(np.sum(y)/y.shape[0]))

    # print('Cross Validation Started...')
    # tic = time.time()
    # cv_acc = cross_val_score(tester.model.log_reg_model, X, y, cv=5, verbose=1)
    # print('Cross validation completed in {}'.format(tic - time.time()))
    # print('Cross Validation Accuracies:\n {}'.format(cv_acc))
    # print('CV Mean Accuracy: {}, (+/- {})'.format(cv_acc.mean(), cv_acc.std()))

if __name__ == "__main__":
    main()

