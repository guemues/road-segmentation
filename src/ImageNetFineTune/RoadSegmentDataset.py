from __future__ import print_function, division

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RoadSegmentDataset(Dataset):
    """
    Class used by pytorch DataLoader.
    When asked object[idx], returns a crop of the image with the binary label.
    Also returns to image index and the crop locations.
    """
    def __init__(self, path, target_path, window_size, step_size, confidence_window):
        self.window_size = window_size
        self.step_size = step_size
        self.path = path
        # name of the loaded images
        self.image_names_list = [image_name for image_name in os.listdir(path) if
                                 image_name.endswith(".png")]
        # list of images as PIL objects
        self.images_list = [Image.open(os.path.join(path, image_name)) for image_name in os.listdir(path) if
                            image_name.endswith(".png")]
        # whether this is the test set
        self.have_labels = os.path.isdir(target_path)
        if self.have_labels:
            # if this is not the test set, load the grountruth images
            self.target_images_list = [Image.open(os.path.join(target_path, image_name)) for image_name in
                                       os.listdir(target_path) if image_name.endswith(".png")]
            # since ground truth images are not binary, we need the threshold them. We select 125 as the naive threshold value.
            self.target_images_list = [im.point(lambda x: 0 if x < 125 else 255, '1') for im in
                                       self.target_images_list]  # turn into black and white
        # confidence window corresponds to extend of the center crop to determine whether current crop is a rod
        self.confidence_window = confidence_window
        # list of preprocessing operations applied to each crop
        self.transform = transforms.Compose([transforms.Scale((224, 224)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                             ])

    def __len__(self):
        """Number of training samples"""
        return int(len(self.images_list) * (self.images_list[0].height / self.step_size) *
                   (self.images_list[0].width / self.step_size) * 1.1)

    def __getitem__(self, idx):
        """
        Getitem for this dataset returns a crop of an input image.
        The index of the image and the crop location are calculated from single idx
        :param idx: crop idx
        :return: cropped image, boolean value whether this crop is a road instance, index of the image, location of the crop
        """
        try:
            crops_per_image = int((self.images_list[0].height / self.step_size) *
                                  (self.images_list[0].width / self.step_size))
            image_idx = int(idx / crops_per_image)
            # we need to load the image from strach everytime due to corruption issues with PIL crops
            self.images_list[image_idx] = Image.open(os.path.join(self.path, self.image_names_list[image_idx]))
            crop_idx = idx % crops_per_image
            crop_x_idx = crop_idx % round(self.images_list[0].height / self.step_size)
            crop_y_idx = crop_idx / round(self.images_list[0].width / self.step_size)

            # center crop locations
            crop_x_loc = int(crop_x_idx * self.step_size)
            crop_y_loc = int(crop_y_idx * self.step_size)

            # do the cropping with the previously computed image index and crop locations
            crop_image = self.images_list[image_idx].crop((
                int(crop_x_loc - self.window_size / 2), int(crop_y_loc - self.window_size / 2),
                int(crop_x_loc + self.window_size / 2), int(crop_y_loc + self.window_size / 2)))

            # do the preprocessing defined in the constructor
            crop_image = self.transform(crop_image)

            # if this is not a test image, calculated whether this is a road instance
            if self.have_labels:
                # make the center cropping with much smaller confidence window
                target_image_crop = self.target_images_list[image_idx].crop(
                    (int(crop_x_loc - self.confidence_window / 2),
                     int(crop_y_loc - self.confidence_window / 2),
                     int(crop_x_loc + self.confidence_window / 2),
                     int(crop_y_loc + self.confidence_window / 2)))

                # count the number of positive pixels in the center crop to determine whether this is a road instance
                sum = 0
                for i in range(target_image_crop.width):
                    for j in range(target_image_crop.height):
                        sum = sum + target_image_crop.getpixel((i, j)) / 255
                road = sum > target_image_crop.height * target_image_crop.width * 0.2
                road = int(road)
            else:
                road = False

            return crop_image, road, image_idx, crop_x_loc, crop_y_loc
        except BaseException as e:
            # print(e)
            return self[np.random.randint(0, int(self.__len__() * 0.8))]


if __name__ == "__main__":
    data_dir = './'
    window_size = 40
    step_size = 5
    confidence_window = 5

    dataset = RoadSegmentDataset(os.path.join(data_dir, "val"),
                                 os.path.join(data_dir, "val" + "_label"), window_size, step_size, confidence_window)

    idx = 0
    while True:
        idx = idx + 1
        crop_image, road, image_idx, crop_x_loc, crop_y_loc = dataset[idx]
        print(road)
        image = crop_image.numpy().transpose((1, 2, 0)) * [0.25, 0.25, 0.25] + [0.5, 0.5, 0.5]
        import matplotlib.pyplot as plt

        plt.imshow(image)
        plt.title(str(road))
        plt.show()
