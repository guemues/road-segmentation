
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


class Trainer(object):

    def __init__(self):
        """ Constructor
        Trainer instances store sift_model, which keeps key points and associated
        SIFT descriptors in images of the training corpus. Keypoints are densely
        sampled.

        Bag of words model generates clusters based on the vectorized SIFT descriptors.
        Any training image patch with keypoints and descriptors is converted to histogram
        of cluster means.

        training data for roads and background is kept separately for training binary classifiers
        """
        self.bow_model = Bow()
        self.sift_model = DenseSift()
        self.training_patch_road = np.array([])
        self.training_patch_background = np.array([])
        self.training_set_road = np.array([])
        self.training_set_background = np.array([])
        self.svm_model = SVC(probability=True)

    def fit_sift(self, data_dir=None, label_dir=None, init=False):
        """SIFT model is constructed with training data and respective groundtruth images
        given the directories that store training images anf groundtruth in separate folders
        """
        self.sift_model.populate_corpus(data_dir, label_dir, init)
        self.sift_model.extract_sift_descriptors()

    def fit_bow(self):
        """Given SIFT model, extract descriptors and training BOW model with KMeans on descriptors"""
        self.bow_model.fit(self.descriptor2features())

    def populate_corpus(self, data, label):
        self.sift_model.populate_corpus(data, label, True)

    def descriptor2features(self):
        if not self.sift_model.get_sift_points():
            return 0

        tup = ()
        for key, kp_desc_pair in self.sift_model.get_sift_points().items():
            tup = tup + (kp_desc_pair[1],)

        return np.concatenate(tup, axis=0)

    def generate_patch_training_set(self, window_size, confidence, step_size=2):

        assert (window_size >= confidence)

        data_road = ()
        data_background = ()
        for key, I in self.sift_model.get_sift_points().items():
            image = self.sift_model.get_corpus()[key]
            truth = self.sift_model.get_groundtruth()[key]
            keypoints, descriptors = I

            assert (window_size <= I.shape[0] & window_size <= I.shape[1])

            for x in range(0, image.shape[1] - window_size, step_size):
                for y in range(0, image.shape[0] - window_size, step_size):
                    if check_patch_confidence(truth, (x,y), window_size, confidence) >= 0:
                        feature = ()
                        for i, kp in enumerate(keypoints):
                            if is_within_window(kp.pt, (x, y), window_size):
                                feature = feature + (descriptors[i, :],)

                        feature = np.concatenate(feature, axis=0)

                        if check_patch_confidence(truth, (x,y), window_size, confidence) == 0:
                            data_road = data_road + (self.bow_model.transform(feature),)
                        else:
                            data_background = data_background + (self.bow_model.transform(feature),)

        self.training_patch_road = np.concatenate(data_road, axis=0)
        self.training_patch_background = np.concatenate(data_background, axis=0)

    def generate_image_training_set(self):

        data_set_road = ()
        data_set_background = ()
        for key, I in self.sift_model.get_sift_points().items():
            data_road = ()
            data_background = ()
            truth = self.sift_model.get_groundtruth()[key]
            keypoints, descriptors = I

            for i, kp in enumerate(keypoints):
                if truth[np.int(kp.pt[1]), np.int(kp.pt[0])] == 1:
                    data_road = data_road + (descriptors[i].reshape(1,-1),)
                else:
                    data_background = data_background + (descriptors[i].reshape(1,-1),)

            tmp_road = np.concatenate(data_road, axis=0)
            tmp_back = np.concatenate(data_background, axis=0)

            data_set_road = data_set_road + (self.bow_model.transform(tmp_road).reshape(1,-1),)
            data_set_background = data_set_background + (self.bow_model.transform(tmp_back).reshape(1,-1),)

        self.training_set_road = np.concatenate(data_set_road, axis=0)
        self.training_set_background = np.concatenate(data_set_background, axis=0)

    def fit_svm(self, data_generation_mode='image'):

        X = 0
        y = 0
        if data_generation_mode == 'image':
            X = np.concatenate((self.training_set_road, self.training_set_background), axis=0)
            y = np.concatenate(
                (np.ones(self.training_set_road.shape[0]), np.zeros(self.training_set_background.shape[0])), axis=0)
        else:
            X = np.concatenate((self.training_patch_road, self.training_patch_background), axis=0)
            y = np.concatenate(
                (np.ones(self.training_patch_road.shape[0]), np.zeros(self.training_patch_background.shape[0])), axis=0)

        self.svm_model.fit(X, y)

    def fit_logistic_regresion(self, data_generation_mode='image'):
        return

