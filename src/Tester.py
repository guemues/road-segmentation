
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
    """This class sets up the environment for training and testing this particular model, where we make use of SIFT descriptors,
    bag of words model and robust binary classifiers. One can submit training images, define mode of training data generation,
    train binary classifiers and make predictions on unseen test images."""

    MODEL_INIT = ['initialize', 'none'] # tells wether to initialize the SIFT and BOW models or not
    DATA_GENERATION_MODE = ['image', 'patch'] # defines the way training data is generated from training images
    PREDICTION_MODE = ['sliding_window', 'pixel_prediction'] # prediction strategy (More on report and comments)
    WINDOW_SIZE = 20 # window size for generating training data in patch mode
    CONFIDENCE = 4 # size of the confidence region in center of patches, required for patch mode training example generation
    STEP_SIZE = 2 # step size of patch generation
    CLASSIFIER = ['SVM', 'Logistic_Regression'] # classifier type specification

    def __init__(self, test_dir=None, init=False):
        self.model = Trainer()
        self.test_set = {}
        self.initialize_model = False
        self.data_generation_mode = 'image'
        self.prediction_mode ='sliding_window'
        self.classifier = 'SVM'

    def populate_test_examples(self, test_dir):
        """Populates the test images in the given directory in the instance variable"""

        assert (os.path.isdir(test_dir))
        image_files = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]

        # populate corpus of images and groundtruth
        for i, f in enumerate(image_files):
            self.test_set[image_files[i]] = cv2.imread(join(test_dir, image_files[i]), 0)

    def fit(self, data_dir, label_dir, initialize_model=False, data_generation_mode='image',
                         prediction_mode='sliding_window', classifier='SVM'):
        """Prepares the environment for testing. Trains the SIFT and BOW models initially, with the images and groundtruth
        in the given directories. Training data is produces with respect to the choice of data generation mode. Finally,
        binary classifier is trained based on the data samples generated"""

        assert (isinstance(initialize_model, bool))

        # set data generation mode
        if data_generation_mode not in Tester.DATA_GENERATION_MODE:
            self.data_generation_mode = Tester.DATA_GENERATION_MODE[0]
        else:
            self.data_generation_mode = data_generation_mode

        # set prediction mode
        if prediction_mode not in Tester.PREDICTION_MODE:
            self.prediction_mode = Tester.PREDICTION_MODE[0]
        else:
            self.prediction_mode = prediction_mode

        # set classifier to be trained
        if classifier not in self.CLASSIFIER:
            self.classifier = self.CLASSIFIER[0]
        else:
            self.classifier = classifier

        # train SIFT and BOW models based in input training images and SIFT descriptors, respectively.
        self.model.fit_sift(data_dir, label_dir, True)
        self.model.fit_bow()

        # generate training data based on previously set data generation mode
        if self.data_generation_mode == 'image':
            self.model.generate_image_training_set()
        else:
            self.model.generate_patch_training_set(self.WINDOW_SIZE, self.CONFIDENCE, self.STEP_SIZE)

        # finallt, train the binary classifier
        if self.classifier == 'SVM':
            self.model.fit_svm(self.data_generation_mode)
        elif self.classifier == 'Logistic_Regression':
            self.model.fit_logistic_regression(self.data_generation_mode)

    def predict(self, I, dense=True, patch_size=4, step_size=2):
        """Predicts the labels for each pixels in the provided image. Depending on the prediction mode, computation differs

        :type dense: bool,
        :type patch_size: int,
        :type step_size, int
        :rtype prediction_grid: ndarray"""

        # generate test image's SIFT descriptors
        # I = cv2.imread(test_image_file, 0)

        sift = cv2.xfeatures2d.SIFT_create()  # create a SIFT descriptor instance
        keypoints = sift.detect(I, None)

        # sample the keypoints in the test image
        # if dense option is True, compute dense overlapping SIFT descriptor
        if dense:
            keypoint_dense_grid = []
            for x in range(0, I.shape[0] - patch_size, step_size):
                for y in range(0, I.shape[1] - patch_size, step_size):
                    keypoint_dense_grid.append(cv2.KeyPoint(x, y, patch_size))

            keypoints.extend(keypoint_dense_grid)
            keypoints.sort(key=lambda p: p.pt)

        # compute SIFT descriptors for provided key points for the test image
        keypoints, descriptors = sift.compute(I, keypoints)

        # Using a sliding window, go over the whole image, compute normalized histogram feature for the patches using
        # the SIFT descriptors extracted from the test image. Predict the label of the patch and compute the probability
        # of belonging to that label. For each pixel, assign the maximum of current probability and previous maximum
        # Refer to the report for more details on the method
        prediction_grid = np.zeros(I.shape)
        for x in range(0, I.shape[1] - self.WINDOW_SIZE, 1):
            for y in range(0, I.shape[0] - self.WINDOW_SIZE, 1):
                feature = ()
                for i, kp in enumerate(keypoints):
                    # compute the descriptors that fall into the patch
                    if is_within_window(kp.pt, (x, y), self.WINDOW_SIZE):
                        feature = feature + (descriptors[i, :].reshape(1,-1),)

                # compute feature vector out of descriptors and predict the label together with confidence score (probability)
                image_features = self.model.bow_model.transform(np.concatenate(feature, axis=0))
                prob = self.model.svm_model.predict_proba(image_features)
                prob_window = prob[0, 1] * np.ones((self.WINDOW_SIZE, self.WINDOW_SIZE))
                prediction_grid[y:y+self.WINDOW_SIZE, x:x+self.WINDOW_SIZE] = \
                    np.maximum(prediction_grid[y:y+self.WINDOW_SIZE, x:x+self.WINDOW_SIZE], prob_window)

        return prediction_grid

    def transform(self, data):
        """Predict the pixel labels for each image in the test set corpus of the Tester instance"""

        self.populate_test_examples(data)

        image_ids = []
        predictions = {}
        for key, I in self.test_set.items():
            image_ids.append(key)
            predictions[key] = self.predict(I)

        return image_ids, predictions


