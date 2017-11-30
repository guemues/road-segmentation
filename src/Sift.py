
import sys
sys.path.append('/home/epfl/ftp/insta-network/src')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isfile


class DenseSift(object):
    def __init__(self, data=None, label=None, init=False):
        self.corpus = {}
        self.groundtruth = {}
        self.SIFT_points = {}

        self.populate_corpus(data, label, init)

    def populate_corpus(self, data, label, init):

        # read images in given directories
        if init:
            # apply necessary checks to input arguments
            assert (is_instance(init, str))
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

        if not self.corpus:
            return

        sift = cv2.xfeatures2d.SIFT_create()  # create a SIFT descriptor instance
        for key, I in self.corpus.items():
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
            self.SIFT_points[key] = (keypoints, descriptors)

    def get_corpus(self):
        return self.corpus

    def get_groundtruth(self):
        return self.groundtruth

    def get_sift_points(self):
        return self.SIFT_points




