# This is  a tutorial file
import numpy as np
from numpy import ndarray, empty
from error import CenterException, SizeException


def test():
    """Say hello to developers """
    print('Hello world3!')


def crop_image(image, corner, size, size_exception=True, center_exception=True):
    """ Crop the numpy matrix
    :param size_exception: If it is True function may return
    :type image: ndarray
    :type corner: tuple
    :type size: tuple
    :type size_exception: bool
    :type center_exception: bool
    :rtype: ndarray
    """
    max_x, max_y = image.shape

    start_x, start_y = corner

    if start_x >= max_x or start_y >= max_y:
        if center_exception:
            raise CenterException()
        else:
            return empty(shape=(0, 0))

    w, h = size, size

    if size_exception and (start_x + w >= max_x or start_y + h >= max_y):
        raise SizeException()

    end_x = start_x + w if start_x + w < max_x else max_x - 1
    end_y = start_y + h if start_y + h < max_y else max_y - 1

    return image[start_y:end_y, start_x:end_x]


def is_within_window(pt, corner, size):
    # TODO: Documentation
    if (pt[0] >= corner[0]) & (pt[0] <= corner[0] + size) & (pt[1] >= corner[1]) & (pt[1] <= corner[1] + size):
        return True
    else:
        return False


def check_patch_confidence(truth, corner, window_size, confidence):
    # TODO: Documentation

    assert (window_size >= confidence)

    conf_corner = (np.int(corner[0] + (window_size - confidence) / 2),
                   np.int(corner[1] + (window_size - confidence) / 2))

    cropped_image = truth[conf_corner[1]:conf_corner[1]+confidence, conf_corner[0]:conf_corner[0]+confidence]
    pixel_sum = np.sum(np.sum(cropped_image))
    max_sum = cropped_image.shape[0] * cropped_image.shape[1]

    if pixel_sum == 0:
        return 0
    elif pixel_sum == max_sum:
        return 1
    else:
        return -1


def is_within_real_image(keypoint, image_size, reflection):
    return (keypoint[0] >= reflection) & (keypoint[1] >= reflection) & \
           (keypoint[0] < image_size[1] + reflection) & (keypoint[1] < image_size[0] + reflection)

