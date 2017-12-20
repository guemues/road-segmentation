import sys

sys.path.append('/home/epfl/ftp/insta-network/src')

import cv2, os, time
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isfile
from utilities import *
from sklearn.externals import joblib


class DenseSift(object):
    """ This class implements a denseSIFT algorithm that samples key points from a given image and computes their SIFT
    descriptors. True SIFT keypoints and densely sampled key points are merged and their descriptors are computed
    collectively
    """
    PATCH_SIZE = 16
    STEP_SIZE = 2

    def __init__(self, data=None, label=None, patch_size=PATCH_SIZE, step_size=STEP_SIZE, init=False):
        """Constructor for the DenseSift class. All the instance variables are initialized empty"""
        self.corpus = {} # test images stored in a dictionary indexed by file names
        self.groundtruth = {} # groundtruth training images (0-1 uint8 ndarray) in a dictionary indexed by file name
        self.SIFT_points = {}
        self.patch_size = patch_size
        self.step_size = step_size

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
                I = cv2.imread(join(label, label_files[i]), 0)
                I[I < 128] = np.uint8(0)
                I[I >= 128] = np.uint8(1)
                self.groundtruth[label_files[i]] = I

        else:
            return

    def extract_sift_descriptors(self, dense=True, patch_size=PATCH_SIZE, step_size=STEP_SIZE):
        """Extracts true SIFT keypoints from the image using OpenCV implementation of SIFT. In addition, densely samples
        the image for overlapping SIFT descriptors. Densely sampled SIFT points have the same size and unknown scale.
        SIFT descriptors are computed for all the keypoints and stored in sorted order with respect to x and y coordinates
        """

        # return immediately if the corpus does not have any image for training
        if not self.corpus:
            return
        i = 1
        sift = cv2.xfeatures2d.SIFT_create()  # create a SIFT descriptor instance
        for key, img in self.corpus.items():
            tic = time.time()
            # reflect image borders so that we can extract features at the border pixels.
            I = cv2.copyMakeBorder(img, np.int(patch_size/2), np.int(patch_size/2), np.int(patch_size/2), np.int(patch_size/2),
                                   cv2.BORDER_REFLECT_101)
            keypoints = sift.detect(I, None)
            keypoints = [kp for kp in keypoints if is_within_real_image(kp.pt, img.shape, np.int(patch_size/2))]

            # if dense option is True, sample dense overlapping SIFT ketpoints
            if dense:
                keypoint_dense_grid = []
                for x in range(np.int(patch_size/2), I.shape[1] - np.int(patch_size/2), step_size):
                    for y in range(np.int(patch_size/2), I.shape[0] - patch_size, step_size):
                        keypoint_dense_grid.append(cv2.KeyPoint(x, y, patch_size))

                # sort keypoints to foster efficient search during training and testing
                keypoints.extend(keypoint_dense_grid)
                keypoints.sort(key=lambda p: p.pt)

            # compute SIFT descriptors for provided key points and save
            keypoints_, descriptors = sift.compute(I, keypoints)
            keypoints_ = [(kp.pt[0] - np.int(patch_size/2), kp.pt[1] - np.int(patch_size/2)) for kp in keypoints_]
            self.SIFT_points[key] = (keypoints_, descriptors)
            print('image {} sift descriptors done in {}'.format(i, time.time() - tic))
            i = i+1

    def save(self, path='/home/ali/Dropbox/Courses/CS-433/road-segmentation/src'):
        file = os.path.join(path, "patchsize_%d_stepsize_%d_.pkl" % (self.patch_size, self.step_size))
        joblib.dump(self.SIFT_points, file)

    def load(self, path='/home/ali/Dropbox/Courses/CS-433/road-segmentation/src'):
        file = os.path.join(path, "patchsize_%d_stepsize_%d_.pkl" % (self.patch_size, self.step_size))
        self.SIFT_points = joblib.load(file)

    def get_corpus(self):
        return self.corpus

    def get_groundtruth(self):
        return self.groundtruth

    def get_sift_points(self):
        return self.SIFT_points
