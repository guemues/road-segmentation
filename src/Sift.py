import sys

sys.path.append('/home/epfl/ftp/insta-network/src')

import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isfile


class DenseSift(object):
    """ This class implements a denseSIFT algorithm that samples key points from a given image and computes their SIFT
    descriptors. True SIFT keypoints and densely sampled key points are merged and their descriptors are computed
    collectively
    """
    def __init__(self, data=None, label=None, init=False):
        """Constructor for the DenseSift class. All the instance variables are initialized empty"""
        self.corpus = {}
        self.groundtruth = {}
        self.SIFT_points = {}

        self.populate_corpus(data, label, init)

    def populate_corpus(self, data, label, init):
        """Given directories that store training data and their groundtruth images, populates them into respective
        dictionary structures, identified by image file name. If init=False, then the function return immediately

        :type data: str
        :type label: str
        :type init: bool
        """

        # read images in given directories
        if init:
            # apply necessary checks to input arguments
            assert (isinstance(init, bool))
            assert (os.path.isdir(data))
            assert (os.path.isdir(label))

            image_files = [f for f in listdir(data) if isfile(join(data, f))]
            label_files = [f for f in listdir(label) if isfile(join(label, f))]

            # populate corpus of images and groundtruth
            for i, f in enumerate(image_files):
                self.corpus[image_files[i]] = cv2.imread(join(data, image_files[i]), 0)
                self.groundtruth[label_files[i]] = cv2.imread(join(label, label_files[i]), 0)

        else:
            return

    def extract_sift_descriptors(self, dense=True, patch_size=4, step_size=2):
        """Extracts true SIFT keypoints from the image using OpenCV implementation of SIFT. In addition, densely sameples
        the image for overlapping SIFT descriptors. Densely sampled SIFT points have the same size and unknown scale.
        SIFT descriptors are computed for all the keypoints and stored in sorted order with respect to x and y coordinates
        """

        # return immediately if the corpus does not have any image for training
        if not self.corpus:
            return

        sift = cv2.xfeatures2d.SIFT_create()  # create a SIFT descriptor instance
        for key, I in self.corpus.items():
            keypoints = sift.detect(I, None)

            # if dense option is True, sample dense overlapping SIFT ketpoints
            if dense:
                keypoint_dense_grid = []
                for x in range(0, I.shape[0] - patch_size, step_size):
                    for y in range(0, I.shape[1] - patch_size, step_size):
                        keypoint_dense_grid.append(cv2.KeyPoint(x, y, patch_size))

                # sort keypoints to foster efficient search during training and testing
                keypoints.extend(keypoint_dense_grid)
                keypoints.sort(key=lambda p: p.pt)

            # compute SIFT descriptors for provided key points and save
            keypoints_, descriptors = sift.compute(I, keypoints)
            self.SIFT_points[key] = (keypoints_, descriptors)

    def get_corpus(self):
        return self.corpus

    def get_groundtruth(self):
        return self.groundtruth

    def get_sift_points(self):
        return self.SIFT_points
