
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC

import cv2, os
import sys
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isfile

from utilities import *
from Bow import *
from Sift import *
from Trainer import *

class Tester(object):

    MODEL_INIT = ['initialize', 'none']
    DATA_GENERATION_MODE = ['image', 'patch']
    PREDICTION_MODE = ['sliding_window', 'pixel_prediction']
    WINDOW_SIZE = 20
    CONFIDENCE = 4
    STEP_SIZE = 2
    CLASSIFIER = ['SVM', 'Logistic_Regression']

    def __init__(self, test_dir=None, init=False):
        self.model = Trainer()
        self.test_set = {}
        self.initialize_model = False
        self.data_generation_mode = 'image'
        self.prediction_mode ='sliding_window'
        self.classifier = 'SVM'

    def populate_test_examples(self, test_dir):

        assert (os.path.isdir(test_dir))
        image_files = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]

        # populate corpus of images and groundtruth
        for i, f in enumerate(image_files):
            self.test_set[image_files[i]] = cv2.imread(join(test_dir, image_files[i]), 0)


    def fit(self, data_dir, label_dir, initialize_model=False, data_generation_mode='image',
                         prediction_mode='sliding_window', classifier='SVM'):

        assert (isinstance(initialize_model, bool))

        if data_generation_mode not in Tester.DATA_GENERATION_MODE:
            self.data_generation_mode = Tester.DATA_GENERATION_MODE[0]
        else:
            self.data_generation_mode = data_generation_mode

        if prediction_mode not in Tester.PREDICTION_MODE:
            self.prediction_mode = Tester.PREDICTION_MODE[0]
        else:
            self.prediction_mode = prediction_mode

        if classifier not in self.CLASSIFIER:
            self.classifier = self.CLASSIFIER[0]
        else:
            self.classifier = classifier

        self.model.fit_sift(data_dir, label_dir, True)
        self.model.fit_bow()

        if self.data_generation_mode == 'image':
            self.model.generate_image_training_set()
        else:
            self.model.generate_patch_training_set(self.WINDOW_SIZE, self.CONFIDENCE, self.STEP_SIZE)

        if self.classifier == 'SVM'
            self.model.fit_svm(self.data_generation_mode)
        elif self.classifier == 'Logistic_Regression':
            self.model.fit_logistic_regression(self.data_generation_mode)

    def predict(self, I, dense=True, patch_size=4, step_size=2):

        # generate test image's SIFT descriptors
        # I = cv2.imread(test_image_file, 0)

        sift = cv2.xfeatures2d.SIFT_create()  # create a SIFT descriptor instance
        keypoints = sift.detect(I, None)

        # if dense option is True, compute dense overlapping SIFT descriptor
        if dense:
            keypoint_dense_grid = []
            for x in range(0, I.shape[0] - patch_size, step_size):
                for y in range(0, I.shape[1] - patch_size, step_size):
                    keypoint_dense_grid.append(cv2.KeyPoint(x, y, patch_size))

            keypoints.extend(keypoint_dense_grid)
            keypoints.sort(key=lambda p: p.pt)

        # compute SIFT descriptors for provided key points and save
        keypoints, descriptors = sift.compute(I, keypoints)

        prediction_grid = np.zeros(I.shape)
        for x in range(0, I.shape[1] - self.WINDOW_SIZE, 1):
            for y in range(0, I.shape[0] - self.WINDOW_SIZE, 1):
                feature = ()
                for i, kp in enumerate(keypoints):
                    if is_within_window(kp.pt, (x, y), self.WINDOW_SIZE):
                        feature = feature + (descriptors[i, :],)

                image_features = self.bow_model.transform(np.concatenate(feature, axis=0))
                prob = self.model.svm_model.predict_proba(image_features)
                prob_window = prob * np.ones((self.WINDOW_SIZE, self.WINDOW_SIZE))
                prediction_grid[y:self.WINDOW_SIZE, x:self.WINDOW_SIZE] = \
                    np.maximum(prediction_grid[y:self.WINDOW_SIZE, x:self.WINDOW_SIZE], prob_window)

        return prediction_grid

    def transform(self, data):

        self.populate_test_examples(data)

        image_ids = []
        predictions = {}
        for key, I in self.test_set.items():
            image_ids.append(key)
            predictions[key] = self.predict(I)

        return image_ids, predictions


