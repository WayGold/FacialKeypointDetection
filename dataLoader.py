import numpy as np
from csvLoader import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FacialKptsDataSet(Dataset):
    def __init__(self, i_df, transform_func=None, transform_param=None):
        self.img_arr, self.kpts_df = getImgArrAndKptsDf(i_df)
        self.transform_func = transform_func
        self.transform_param = transform_param

        # Norm to [-1, 1]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

    def __getitem__(self, index):
        img = self.img_arr[index].astype(np.uint8).reshape(96, 96)
        img = self.transform(img)

        # Extracting rows
        kpts = np.array(self.kpts_df.iloc[index])

        if self.transform_func is not None:
            img, kpts = self.transform_func(img, kpts, self.transform_param)

        return img, kpts

    def __len__(self):
        return len(self.img_arr)


def getTrainValidationDataSet(i_df, train_val_percentage):
    """

    Args:
        i_df (pandas.DataFrame):            Input DataFrame to split into training
                                            and validation set.
        train_val_percentage (float):       Percentage to split.

    Returns:    (pandas.DataFrame, pandas.DataFrame)   -    training set and
                                                            validation set

    """
    len_df = len(i_df)
    rand_idx = list(range(len_df))
    np.random.shuffle(rand_idx)

    split_idx = int(np.floor(len_df * train_val_percentage))
    train_idx, val_idx = rand_idx[:split_idx], rand_idx[split_idx:]

    train_df = i_df.iloc[train_idx]
    val_df = i_df.iloc[val_idx]

    return train_df, val_df
