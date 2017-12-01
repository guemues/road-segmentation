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

    def fit(self, train_im, label_im):
        assert (train_im.shape[:-1] == label_im.shape)  # except the number of channels
        self.__init__dnn()

        batch, height, width, channels = train_im.shape
        batch = 1
        feature_map = None
        for b in range(batch):  # for all images
            train_patches, loc = self.__generate_patches(train_im[b, :, :, :], self.window, self.step)
            unique_x = len(set([l[0] for l in loc]))
            if feature_map is None:
                feature_map = np.empty(shape=(batch, unique_x, unique_x, self.num_features))
            for idx, patch in enumerate(train_patches):
                print("Processing batch %d patch %d" % (b, idx))
                patch = PIL.Image.fromarray(np.squeeze(patch).astype(np.uint8))
                tensor = Variable(self.transform(patch))
                output = np.squeeze(self.dnn_model.forward(tensor.unsqueeze(0)).data.numpy())
                feature_map[b, int(loc[idx][0] / self.window), int(loc[idx][1] / self.window), :] = output

            print("Saving ./train_features_%d_%d.npy" % (self.window, self.step))
            np.save(os.path.join(self.save_path,"./train_features_%d_%d.npy" % (self.window, self.step)), feature_map)

        features = feature_map.reshape((-1,self.num_features))
        self.bow.fit(features)
        self.bow.save(self.save_path)


    def __generate_patches(self, im, window, step):
        height, width, channels = im.shape
        x_iter = np.arange(0, height, step)
        y_iter = np.arange(0, width, step)
        border = int(window / 2)
        im = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT)
        image_patches = np.empty(shape=(x_iter.shape[0] * y_iter.shape[0], window, window, channels))

        idx = 0
        patch_location = []
        for x in x_iter:
            for y in y_iter:
                x_min, x_max = x, x + window
                y_min, y_max = y, y + window
                image_patches[idx, :, :] = im[x_min:x_max, y_min:y_max]
                patch_location.append((x, y))
                idx = idx + 1
        return image_patches, patch_location

    def __init__dnn(self, name="alexnet"):
        if name == "alexnet":
            self.dnn_model = models.alexnet(pretrained=True)
            # remove last fully-connected layer
            new_classifier = nn.Sequential(*list(self.dnn_model.classifier.children())[:-1])
            self.dnn_model.classifier = new_classifier
        else:
            raise NotImplementedError

    def transform(self):
        pass

    def __dnn_transform(self):
        pass


if __name__ == "__main__":
    from data_loader import load_data

    model = ImageNetModel(step=40, num_centers=10)
    train_im, ground_im, test_im = load_data(path="../../")
    model.fit(train_im, label_im=ground_im)
