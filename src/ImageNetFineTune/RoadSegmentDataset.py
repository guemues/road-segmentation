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
        self.target_images_list = [im.point(lambda x: 0 if x < 125 else 255, '1') for im in self.target_images_list] # turn into black and white


        self.confidence_window = confidence_window
        self.transform = transforms.Compose([transforms.Scale((224, 224)),
                                             #transforms.RandomHorizontalFlip(),
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
        #crop_image.resize((224,224)).show()
        crop_image = self.transform(crop_image)

        target_image_crop = self.target_images_list[image_idx].crop((int(crop_x_loc - self.confidence_window / 2),
                                                                int(crop_y_loc - self.confidence_window / 2),
                                                                int(crop_x_loc + self.confidence_window / 2),
                                                                int(crop_y_loc + self.confidence_window / 2)))

        """
        self.target_images_list[image_idx].crop((int(crop_x_loc - self.window_size / 2),
                                                 int(crop_y_loc - self.window_size / 2),
                                                 int(crop_x_loc + self.window_size / 2),
                                                 int(crop_y_loc + self.window_size / 2))).show()
        """

        sum = 0
        for i in range(target_image_crop.width):
            for j in range(target_image_crop.height):
                sum = sum + target_image_crop.getpixel((i, j))/255
        road = sum > target_image_crop.height * target_image_crop.width * 0.3
        road = int(road)

        #print(image_idx, crop_x_loc, crop_y_loc, road)
        return crop_image, road

if __name__=="__main__":
    data_dir = './'
    window_size = 40
    step_size = 5
    confidence_window = 5

    dataset =  RoadSegmentDataset(os.path.join(data_dir, "train"),
                                            os.path.join(data_dir, "train" + "_label"), window_size, step_size, confidence_window)


    while True:
        idx= np.random.randint(len(dataset))
        crop_image, road = dataset[idx]
        print(road)
        image = crop_image.numpy().transpose((1,2,0)) *[0.25, 0.25, 0.25] + [0.5, 0.5, 0.5]
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.title(str(road))
        plt.show()
