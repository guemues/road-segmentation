
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC

import cv2
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

    
    tester = Tester()


