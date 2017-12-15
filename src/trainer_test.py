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
from Tester import *

def main():

    data_dir = '/home/ali/Dropbox/Courses/CS-433/road-segmentation/training_images'
    label_dir = '/home/ali/Dropbox/Courses/CS-433/road-segmentation/training_groundtruth'

    trainer = Trainer()
    trainer.fit_sift(data_dir, label_dir, init=True)
    trainer.fit_bow()
    trainer.generate_image_training_set()
    trainer.fit_svm

if __name__ == "__main__":
    main()

