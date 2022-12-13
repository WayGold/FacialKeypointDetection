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
