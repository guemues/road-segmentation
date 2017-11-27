from utilities import crop_image
import numpy as np


def test_crop_image():
    A = np.arange(9).reshape((3,3))
    assert crop_image(A, (1, 1), (1, 1))[0] == 4
