import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from Bow import Bow
import numpy as np
import cv2
from torch.autograd import Variable
import PIL
import os
from skimage.transform import rotate


class ImageNetModel(object):
    def __init__(self, num_centers=1000, window=128, step=10, label_window=7, save_path="./"):
        self.num_centers = num_centers
        self.step = step
        self.window = window
        self.label_window = label_window
        self.bow = Bow(num_centers=self.num_centers)
        self.transform = \
            transforms.Compose([
                transforms.Scale(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.save_path = save_path
        self.num_features = 4096

    def fit(self, train_im):
        #assert (train_im.shape[:-1] == label_im.shape)  # except the number of channels
        self.__init__dnn()

        batch, height, width, channels = train_im.shape
        batch = 1
        feature_map = None
        for b in range(batch):  # for all images
            unique_x = int(height / self.step)
            if feature_map is None:
                feature_map = np.empty(shape=(batch, unique_x, unique_x, self.num_features))
            for patch, x, y, _ in self.__generate_patches(train_im[b, :, :, :], self.window, self.step, im_label=None):
                # print("Processing batch %d patch %d" % (b, idx))
                feature_map[b, int(x / self.step), int(y / self.step), :] = self.__dnn_transform(patch)

            print("Saving ./train_features_%d_%d.npy" % (self.window, self.step))
            np.save(os.path.join(self.save_path, "./train_features_window_%d_step_%d.npy" % (self.window, self.step)),
                    feature_map)

        features = feature_map.reshape((-1, self.num_features))
        self.bow.fit(features)
        self.bow.save(self.save_path)

    def save(self):
        self.bow.save(self.save_path)

    def load(self):
        self.bow.load(self.save_path)

    def transform(self, im, window, step=1):
        self.__init__dnn()

        height, width, channels = im.shape
        feature_map = None
        unique_x = int(height / self.step)
        if feature_map is None:
            feature_map = np.empty(shape=(unique_x, unique_x, self.num_features))
        for patch, x, y, _ in self.__generate_patches(im, window, step):
            # print("Processing batch %d patch %d" % (b, idx))
            feature_map[int(x / self.step), int(y / self.step), :] = self.__dnn_transform(patch)

        histogram_map = None
        height, width, channels = feature_map.shape
        unique_x = int(height / self.step)
        if feature_map is None:
            histogram_map = np.empty(shape=(unique_x, unique_x, self.num_features))
        for patch, x, y, _ in self.__generate_patches(im, window, step):
            # print("Processing batch %d patch %d" % (b, idx))
            histogram_map[int(x / self.step), int(y / self.step), :] = self.bow.transform(patch.reshape((-1,self.num_features)))

        return histogram_map


    def __generate_patches(self, im, window, step, im_label=None, rotate_list=[0]):
        height, width, channels = im.shape
        x_iter = np.arange(0, height, step)
        y_iter = np.arange(0, width, step)
        border = int(window / 2)
        im = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_REFLECT)
        im_label = cv2.copyMakeBorder(im_label, border, border, border, border, cv2.BORDER_REFLECT)
        # image_patches = np.empty(shape=(x_iter.shape[0] * y_iter.shape[0], window, window, channels))

        idx = 0
        patch_location = []
        for r in rotate_list:
            im_rotate = rotate(im, r)
            label_rotate = rotate(im_label, r)
            for x in x_iter:
                for y in y_iter:
                    x_min, x_max = x, x + window
                    y_min, y_max = y, y + window
                    patch = im_rotate[x_min:x_max, y_min:y_max]
                    label_patch = None
                    if label_rotate is not None:
                        label_patch = label_rotate[x_min:x_max, y_min:y_max]
                    patch_location.append((x, y))
                    idx = idx + 1
                    yield patch, x, y, label_patch

    def __init__dnn(self, name="alexnet"):
        if name == "alexnet":
            self.dnn_model = models.alexnet(pretrained=True)
            # remove last fully-connected layer
            new_classifier = nn.Sequential(*list(self.dnn_model.classifier.children())[:-1])
            self.dnn_model.classifier = new_classifier
        else:
            raise NotImplementedError

    def __dnn_transform(self, im):
        patch = PIL.Image.fromarray(np.squeeze(im).astype(np.uint8))
        tensor = Variable(self.transform(patch))
        output = np.squeeze(self.dnn_model.forward(tensor.unsqueeze(0)).data.numpy())
        return output


if __name__ == "__main__":
    from data_loader import load_data

    model = ImageNetModel(step=40, num_centers=10)
    train_im, ground_im, test_im = load_data(path="../../")
    model.fit(train_im)
