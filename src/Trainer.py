
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.externals import joblib
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression

import cv2, os
import sys, time
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
        self.log_reg_model = LogisticRegression()

    def fit_sift(self, data_dir=None, label_dir=None, load=False, save=False, path='/', init=False):
        """SIFT model is constructed with training data and respective groundtruth images
        given that directories that store training images and groundtruth are separate folders

        :type data_dir: str
        :type label_dir: str
        :type init: bool
        """

        self.sift_model.populate_corpus(data_dir, label_dir, init)

        if load:
            self.sift_model.load(path)
        else:
            self.sift_model.extract_sift_descriptors()

        if save:
            self.sift_model.save(path)

    def fit_bow(self, load=False, save=False, path='/'):
        """Given SIFT model, extract descriptors and training BOW model with KMeans on descriptors"""

        # make sure Sift descriptors are extracted properly
        if not self.sift_model.get_sift_points():
            raise AssertionError

        if load:
            self.bow_model.load(path)
        else:
            self.bow_model.fit(self.descriptor2features())

        if save:
            self.bow_model.save(path)

    def populate_corpus(self, data, label):
        """Populate the DenseSift instance with training images and respective groundtruth maps"""
        self.sift_model.populate_corpus(data, label, True)

    def descriptor2features(self):
        """Convert extracted descriptors for all images into a single training data matrix for bag of words model

        :rtype: ndarray
        """

        # make sure Sift descriptors are extracted properly
        if not self.sift_model.get_sift_points():
            raise AssertionError

        tup = ()
        for key, kp_desc_pair in self.sift_model.get_sift_points().items():
            tup = tup + (kp_desc_pair[1],)

        return np.concatenate(tup, axis=0)

    def generate_patch_training_set(self, window_size, confidence, step_size=2):
        """Takes the images in the sift model corpus, and extracts training 'patches' based on th structure of each patch
        cropped from the image. Each image is scanned through with a sliding window. For each window, if the the group
        of pixels in the middle (within a square window of size 'confidence') are all 1s or 0s, then this patch is taken
        as a training example for the respective class. This redundant sampling multiplies the training examples we have,
        and it also captures the spatial information in the images. (See the report for more detailed explanation)

        :type window_size: int,
        :type confidence: int,
        :type step_size: int
        """

        assert (window_size >= confidence)

        # populate normalized histogram features of each patch in tuples according to their class label, determined by
        # the pixels in the center of the patch
        data_road = ()
        data_background = ()

        j = 1
        # patches are computed for each image iteratively
        for key, I in self.sift_model.get_sift_points().items():
            print('Training: image patches generation for image {} is started'.format(j))
            tic = time.time()
            image = self.sift_model.get_corpus()[key]
            truth = self.sift_model.get_groundtruth()[key]
            keypoints, descriptors = I

            # print('Image: {}, Image Size: {}, Window Size: {}'.format(key, image.shape, window_size))
            assert (window_size <= image.shape[0] and window_size <= image.shape[1])

            # determine the sift descriptors that lie inside each patch and contruct a normalized histogram feature vector
            # for each patch where all the pixels in the center belong to the same class
            for x in range(0, image.shape[1] - window_size, step_size):
                for y in range(0, image.shape[0] - window_size, step_size):

                    # check if all the pixels in the center, within a square of size 'confidence', belong to the same label
                    patch_label = check_patch_confidence(truth, (x,y), window_size, confidence)
                    if patch_label >= 0:
                        feature = ()
                        for i, kp in enumerate(keypoints):
                            if is_within_window(kp, (x, y), window_size):
                                feature = feature + (descriptors[i, :].reshape(1,-1),)

                        # convert from tuple to ndarray
                        feature = np.concatenate(feature, axis=0)

                        # store feature vector in respective class' data matrix
                        if patch_label == 0:
                            data_road = data_road + (self.bow_model.transform(feature),)
                        else:
                            data_background = data_background + (self.bow_model.transform(feature),)

            toc = time.time() - tic
            print('Training: image patches generation for image {} is done in {}\n'.format(j, toc))
            j = j+1

        # store training examples in instance variable
        self.training_patch_road = np.concatenate(data_road, axis=0)
        self.training_patch_background = np.concatenate(data_background, axis=0)

    def generate_image_training_set(self):
        """Basically, performs analogous operations with function 'generate_patch_training_set',but per image, it generates
        two training sample, one for each class. Per image, descriptors that belong to regions of road and background are
        separated. For each separated set of descriptors, it generates a normalized histogram feature vector. (Please refer
        to the report for detailed explanation of the idea behind)"""

        data_set_road = ()
        data_set_background = ()
        # iterate over all images and group descriptors into the class label for their keypoint
        for key, I in self.sift_model.get_sift_points().items():
            data_road = ()
            data_background = ()
            truth = self.sift_model.get_groundtruth()[key]
            keypoints, descriptors = I

            # store each keypoints/descriptor in respective class label's matrix
            for i, kp in enumerate(keypoints):
                if truth[np.int(kp[1]), np.int(kp[0])] == 1:
                    data_road = data_road + (descriptors[i].reshape(1,-1),)
                else:
                    data_background = data_background + (descriptors[i].reshape(1,-1),)

            # tranform from tuple to ndarray
            tmp_road = np.concatenate(data_road, axis=0)
            tmp_back = np.concatenate(data_background, axis=0)

            # compute normalized histogram features
            data_set_road = data_set_road + (self.bow_model.transform(tmp_road),)
            data_set_background = data_set_background + (self.bow_model.transform(tmp_back),)

        # store training feature matrices into respective instance variable
        self.training_set_road = np.concatenate(data_set_road, axis=0)
        self.training_set_background = np.concatenate(data_set_background, axis=0)

    def fit_svm(self, data_generation_mode='image'):
        """Train the binary SVM model for the training data generated and stored. Two training modes: image and patch.
        In the image training mode, training images are generated based in descriptors in the whole image. In the patch
        mode training images are the overlapping patches in the images. A patch is considered as a proper training image
        if pixels in the center, within a square of size 'confidence', belong to the same label in the groundtruth"""

        X = 0
        y = 0
        if data_generation_mode == 'image':
            # check if training set is populated beforehand
            assert (self.training_set_road.size > 0)
            assert (self.training_set_background.size > 0)

            # prepare training feature matrix and labels
            X = np.concatenate((self.training_set_road, self.training_set_background), axis=0)
            y = np.concatenate(
                (np.ones(self.training_set_road.shape[0]), np.zeros(self.training_set_background.shape[0])), axis=0)
        else:
            # check if training patches are populated beforehand
            assert (self.training_patch_road.size > 0)
            assert (self.training_patch_background.size > 0)

            # prepare training feature matrix and labels
            X = np.concatenate((self.training_patch_road, self.training_patch_background), axis=0)
            y = np.concatenate(
                (np.ones(self.training_patch_road.shape[0]), np.zeros(self.training_patch_background.shape[0])), axis=0)

        self.svm_model.fit(X, y)

    def save(self, path, data_generation_mode='patch'):
        file = os.path.join(path, "svm_data_mode_{}.pkl".format(data_generation_mode))
        joblib.dump((self.svm_model, self.training_patch_road, self.training_patch_background), file)

    def load(self, path, data_generation_mode='patch'):
        file = os.path.join(path, "svm_data_mode_{}.pkl".format(data_generation_mode))
        (self.svm_model, self.training_patch_road, self.training_patch_background) = joblib.load(file)


    def fit_logistic_regresion(self, data_generation_mode='image'):
        return

