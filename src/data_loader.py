import os
import numpy as np
from skimage.io import imread

def load_data():
    folder_list = ["../training_images/", "../training_groundtruth/", "../test_images/"]
    images_list = []
    for folder in folder_list:
        listdir = os.listdir(folder)
        images_path  = [os.path.join(folder, image_name) for image_name in listdir if image_name.endswith(".png")]
        if "groundtruth" in folder:
            images = np.empty(shape=(len(images_path), 400, 400))
        else:
            images = np.empty(shape=(len(images_path), 400, 400, 3))

        for idx, path in enumerate(images_path):
            if "groundtruth" in folder:
                images[idx, :, :] = imread(path)
            else:
                images[idx,:,:,:] = imread(path)

        images_list.append(images)

    return images_list

if __name__=="__main__":
    images_list = load_data()
    pass
