import torch
import numpy as np

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
    img = torch.tensor(img.copy()).reshape(1, 96, 96)
    kpts = kpts[flip_index]
    kpts[::2] = 96 - kpts[::2]
    return img, kpts
