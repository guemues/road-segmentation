# This is  a tutorial file
from numpy import ndarray
from error import CenterException, SizeException


def test():
    """Say hello to developers """
    print('Hello world3!')


def crop_image(image, corner, size, size_exception=True, center_exception=True):
    """ Crop the numpy matrix
    :type image: ndarray
    :type corner: tuple
    :type size: tuple
    :type size_exception: bool
    :type center_exception: bool
    :rtype: ndarray
    """
    max_x, max_y = image.shape

    start_x, start_y = corner

    if center_exception and (start_x >= max_x or start_y >= max_y):
        raise CenterException()

    w, h = size

    if size_exception and (start_x + w >= max_x or start_y + h >= max_y):
        raise SizeException()

    end_x = start_x + w if start_x + w < max_x else max_x - 1
    end_y = start_y + h if start_y + h < max_y else max_y - 1

    return image[start_x:end_x, start_y:end_y]
