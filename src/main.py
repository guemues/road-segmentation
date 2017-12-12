
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

def main():

    data_dir = '/home/ali/Dropbox/Courses/CS-433/road-segmentation/training_images'
    label_dir = '/home/ali/Dropbox/Courses/CS-433/road-segmentation/training_groundtruth'
    test_dir = '/home/ali/Dropbox/Courses/CS-433/road-segmentation/test_images'
    path = '/home/ali/Dropbox/Courses/CS-433/road-segmentation/src'
    load_bow = True
    save_bow = False
    load_sift = True
    save_sift = False
    load_svm = True
    save_svm = False
    init = True

    tester = Tester()
    tester.fit(data_dir, label_dir, load_bow=load_bow, save_bow=save_bow, load_sift=load_sift, save_sift=save_sift,
               load_svm=load_svm, save_svm=save_svm, path=path, init=init)

    predictions = []
    image_files = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]
    test_count = len(image_files)
    file = os.path.join(path, "prediction_grids.pkl")
    for i, f in enumerate(image_files):
        print('Transforming test image {}/{}'.format(i+1, test_count))
        tic = time.time()
        I = cv2.imread(join(test_dir, image_files[i]), 0)
        I[I < 128] = np.uint8(0)
        I[I >= 128] = np.uint8(1)
        predictions.append(tester.extract_transform(I))
        print('Transforming test image {}/{} done in {}'.format(i+1, test_count, time.time() - tic))
        joblib.dump(predictions, file)

if __name__ == "__main__":
    main()

