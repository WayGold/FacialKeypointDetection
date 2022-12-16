import cv2
import torch
import numpy as np
from dataLoader import FacialKptsDataSet

# Helper Global Vars
original_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                  28, 29]
flip_index = [2, 3, 0, 1, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 21, 24, 25, 22, 23, 26, 27, 28,
              29]


def mirror(img, kpts, param=None):
    """
    Creates a mirror of the input img and kpts data.

    Args:
        img:            Torch Tensor
        kpts:           Numpy Array
        param:          Extra Params needed to pass in to dataset class,
                        Not needed for this particular function.

    Returns:            Torch.tensor    -   img data after mirror
                        Numpy.ndarray   -   kpts data after mirror

    """
    img = np.array(img)
    img = img[:, :, ::-1]
    img = torch.tensor(img.copy(), dtype=torch.float32).reshape(1, 96, 96)

    kpts = kpts[flip_index]
    kpts[::2] = 96 - kpts[::2]

    return img, kpts


def add_noise(img, kpts, param=0.008):
    """
    Create a img with random noise * input noise factor.

    Args:
        img (torch.Tensor):             Input Image
        kpts (Numpy.ndarray):           Input Kpts
        param (float):                  Noise Factor

    Returns:            Torch.tensor    -   img data after mirror
                        Numpy.ndarray   -   kpts data after mirror

    """
    img = np.array(img)
    img = img + param * np.random.randn(1, 96, 96)
    img = torch.tensor(img.copy(), dtype=torch.float32).reshape(1, 96, 96)
    return img, kpts


def brightness_trim(img, kpts, param):
    """
        Create a img with param degree of brightness trim.

        Args:
            img (torch.Tensor):             Input Image
            kpts (Numpy.ndarray):           Input Kpts
            param (float):                  brightness trim factor

        Returns:            Torch.tensor    -   img data after brightness trim
                            Numpy.ndarray   -   kpts data after brightness trim

    """
    img = np.array(img)
    img = np.clip(img + param, -1, 1)
    img = torch.tensor(img.copy(), dtype=torch.float32).reshape(1, 96, 96)
    return img, kpts


def rotate(img, kpts, param):
    """
    Create a img with param degree of rotation. Angle(param) is positive for
    counter-clockwise and negative for clockwise.

    Args:
        img (torch.Tensor):             Input Image
        kpts (Numpy.ndarray):           Input Kpts
        param (float):                  Rotation Factor, in degree

    Returns:            Torch.tensor    -   img data after rotation
                        Numpy.ndarray   -   kpts data after rotation

    """
    # Rotation Image
    img = np.array(img)
    rotMat = cv2.getRotationMatrix2D((48, 48), param, 1)
    img = cv2.warpAffine(np.array(img).reshape(96, 96), rotMat, (96, 96), flags=cv2.INTER_CUBIC)
    img = torch.tensor(img.copy(), dtype=torch.float32).reshape(1, 96, 96)

    # Rotate Kpts
    radian = -param / 180 * np.pi
    # Center to origin
    kpts -= 48

    # Loop through each point to rotation them, skip 2 since x,y
    # coords are not coupled
    for i in range(0, len(kpts), 2):
        x, y = kpts[i], kpts[i + 1]
        kpts[i] = x * np.cos(radian) - y * np.sin(radian)
        kpts[i + 1] = x * np.sin(radian) + y * np.cos(radian)

    # Re-center back
    kpts += 48

    return img, kpts


def create_augs_from_transform(i_df, transform, params):
    """
    Create a list of all augmented datasets.

    Args:
        i_df (pandas.DataFrame):
        transform (callable):
        params (list):

    Returns:            List(FacialKptsDataSet)

    """
    augs = []
    for param in params:
        augs.append(FacialKptsDataSet(i_df, transform_func=transform, transform_param=param))
    return augs
