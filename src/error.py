class CropException(Exception):
    """Something is wrong with the crop process crop size is larger then it can be"""
    pass


class SizeException(CropException):
    """User crop size is larger then it can be"""
    pass


class CenterException(CropException):
    """Center is not in the image"""
    pass
