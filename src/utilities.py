# This is  a tutorial file
from numpy import ndarray


def test():
    """Say hello to developers """
    print('Hello world!')


def crop_image(image, corner, size):
    """ Crop the numpy matrix
    :type image: ndarray
    :type corner: tuple
    :type size: tuple
    :rtype: ndarray
    """

    max_x, max_y = image.shape

    start_x, start_y = corner
    w, h = size

    end_x = start_x + w if start_x + w < max_x else max_x - 1
    end_y = start_y + h if start_y + h < max_y else max_y - 1

    return image[start_x:end_x, start_y:end_y]

def is_within_window(pt, corner, size):

    if (pt[0] >= corner[0]) & (pt[0] <= corner[0] + size[0]) & (pt[1] >= corner[1]) & (pt[1] <= corner[1] + size[1]):
        return True
    else:
        return False

def check_patch_confidence(truth, corner, window_size, confidence):

    assert (window_size >= confidence)

    conf_corner = (corner[0] + np.floor((window_size - confidence) / 2),
                          corner[1] + np.floor((window_size - confidence) / 2))

    cropped_image = crop_image(truth, conf_corner, confidence)
    pixel_sum = np.sum(np.sum(cropped_image))
    max_sum = cropped_image.shape[0] * cropped_image.shape[1]

    if pixel_sum == 0:
        return 0
    elif pixel_sum == max_sum:
        return 1
    else:
        return -1

