[![Build Status](https://travis-ci.org/guemues/road-segmentation.svg?branch=master)](https://travis-ci.org/guemues/road-segmentation)

# Starting
- [x] Understand the data, how to make submission
- [x] Push data to git
- [ ] Methods for loading data. Read as tensor or as a filename keyed dictionary
# BOW
- [ ] how to extract sift? Which library to use? Need to look at skimage. patch size is a hyperparameter
- [ ] Extract sift descriptors from all the data using sliding windows. Windows should largely overlap.
- [ ] Use k-means to find clusters. Number of clusters should be around 1000, this is a hyperparameter.
- [ ] For every patch take the sift descriptor of it and neighbor patches, turn to histogram of nearest cluster centers.
- [ ] Train svm or logistic regression to recognize road and not road
# Utility
- [ ] Cross validation code
# Search
- [ ] Look for similar datasets. We can fuse two datasets.
- [ ] Some state of the art network? 
- [ ] Use fisher vectors or vlad instead of bow?
- [ ] Look for old reports on the internet. I have seen at least one.
