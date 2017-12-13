
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

def main():

    data_dir = '/home/ali/Dropbox/Courses/CS-433/road-segmentation/training_images'
    label_dir = '/home/ali/Dropbox/Courses/CS-433/road-segmentation/training_groundtruth'
    test_dir = '/home/ali/Dropbox/Courses/CS-433/road-segmentation/test_images'
    path = '/home/ali/Dropbox/Courses/CS-433/road-segmentation/src'
    load_bow = True
    save_bow = False
    load_sift = True
    save_sift = False
    load_model = True
    save_model = False
    init = True

    tester = Tester()
    tester.fit(data_dir, label_dir, load_bow=load_bow, save_bow=save_bow, load_sift=load_sift, save_sift=save_sift,
               load_model=load_model, save_model=save_model, path=path, init=init)

    predictions = {}
    image_files = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]
    test_count = len(image_files)
    file = os.path.join(path, "prediction_grids.pkl")
    for i, f in enumerate(image_files):
        print('Transforming test image({}) {}/{}'.format(f, i + 1, test_count))
        tic = time.time()
        I = cv2.imread(join(test_dir, image_files[i]), 0)
        I[I < 128] = np.uint8(0)
        I[I >= 128] = np.uint8(1)
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

