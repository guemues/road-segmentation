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
