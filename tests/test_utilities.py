from utilities import crop_image
from nose.tools import assert_raises

from error import SizeException
from error import CenterException

import numpy as np


def test_crop_image():
    A = np.arange(9).reshape((3,3))
    assert crop_image(A, (1, 1), (1, 1))[0] == 4

    assert_raises(SizeException, crop_image, A, (1, 1), (4, 4))
    assert_raises(CenterException, crop_image, A, (5, 5), (1, 1))
