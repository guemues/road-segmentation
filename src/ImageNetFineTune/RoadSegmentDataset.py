from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from os import listdir
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class RoadSegmentDataset(Dataset):

    def __init__(self, path, target_path, window_size, step_size, confidence_window):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.window_size = window_size
        self.step_size = step_size
        self.images_list = [Image.open(os.path.join(path, image_name)) for image_name in os.listdir(path) if
                            image_name.endswith(".png")]
        self.target_images_list = [Image.open(os.path.join(target_path, image_name)) for image_name in
                                   os.listdir(target_path) if image_name.endswith(".png")]
        self.confidence_window = confidence_window
        self.transform = transforms.Compose([transforms.Scale((224, 224)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
                                             ])

    def __len__(self):
        """Number of training samples"""
        return len(self.images_list) * int(self.images_list[0].height / self.step_size) * int(
            self.images_list[0].width / self.step_size)

    def __getitem__(self, idx):
        crops_per_image = int(self.images_list[0].height / self.step_size) * int(
            self.images_list[0].width / self.step_size)
        image_idx = int(idx / crops_per_image)
        crop_idx = idx % crops_per_image
        crop_x_idx = crop_idx % int(self.images_list[0].height / self.step_size)
        crop_y_idx = crop_idx / int(self.images_list[0].width / self.step_size)

        crop_x_loc = int(crop_x_idx * self.step_size)
        crop_y_loc = int(crop_y_idx * self.step_size)

        crop_image = self.images_list[image_idx].crop((
            int(crop_x_loc - self.window_size / 2), int(crop_y_loc - self.window_size / 2),
            int(crop_x_loc + self.window_size / 2), int(crop_y_loc + self.window_size / 2)))
        crop_image = self.transform(crop_image)

        target_image = self.target_images_list[image_idx].crop((int(crop_x_loc - self.confidence_window / 2),
                                                                int(crop_y_loc - self.confidence_window / 2),
                                                                int(crop_x_loc + self.confidence_window / 2),
                                                                int(crop_y_loc + self.confidence_window / 2)))

        road = 0  # TODO
        return crop_image, road
