import os
import torch
import train
import model
import numpy as np
import csvLoader as cl
import visualizer as vs
import dataLoader as dl
import dataAugmentation as da
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

TRAIN_CSV_PATH = 'facial-keypoints-detection/training/training.csv'
TEST_CSV_PATH = 'facial-keypoints-detection/test/test.csv'


def vis_test():
    USE_GPU, device = train.check_GPU()
    print('GPU mode: {}'.format(USE_GPU))

    train_csv = cl.load_csv(TRAIN_CSV_PATH)

    print(f'Len of train csv: {len(np.array(train_csv.Image))}')
    csv_allValid, csv_autoFill, csv_missingOnly = cl.clean_csv(train_csv)

    print('Loading Dataset...')
    allValid_dataset = dl.FacialKptsDataSet(csv_allValid)
    print('Randomly Visualizing...')
    vs.rand_vis_dataset(allValid_dataset, 5)


def train_test():
    lr = 1e-3
    train_csv = cl.load_csv(TRAIN_CSV_PATH)
    print(f'Len of train csv: {len(np.array(train_csv.Image))}')
    csv_allValid, csv_autoFill, csv_missingOnly = cl.clean_csv(train_csv)

    print('Loading Dataset...')
    # allValid_dataset = dl.FacialKptsDataSet(csv_allValid)
    # allValidTrain, allValidVal = dl.getTrainValidationDataSet(csv_allValid, 0.85)
    autoFillTrain, autoFillVal = dl.getTrainValidationDataSet(csv_allValid, 0.75)
    val_dataset = dl.FacialKptsDataSet(autoFillVal)

    if os.path.exists('./Aug_set.pt'):
        print('Pre-saved aug data set found, loading...')
        train_datasets = torch.load('./Aug_set.pt')
    else:
        # Data Augmentation
        all_datasets = []
        print('Loading training set...')
        train_dataset = dl.FacialKptsDataSet(autoFillTrain)
        print('Size of set: {}'.format(len(train_dataset)))

        print('Augmenting training set using mirror...')
        mirror_set = da.create_augs_from_transform(autoFillTrain, da.mirror, params=[None])
        print('Size of set: {}'.format(len(mirror_set[0])))

        print('Augmenting training set using noise...')
        noise_set = da.create_augs_from_transform(autoFillTrain, da.add_noise, params=[0.1, 0.08])
        print('Size of set: {}'.format(len(noise_set[0])))

        print('Augmenting training set using brightness trim...')
        brightTrim_set = da.create_augs_from_transform(autoFillTrain, da.brightness_trim, params=[0.7, -0.7])
        print('Size of set: {}'.format(len(brightTrim_set[0])))

        print('Augmenting training set using rotation...')
        rotation_set = da.create_augs_from_transform(autoFillTrain, da.rotate, params=[10, -10, 5, -5])

        all_datasets += [train_dataset]
        all_datasets += mirror_set
        all_datasets += noise_set
        all_datasets += brightTrim_set
        all_datasets += rotation_set

        print('Num of datasets after augmentation: {}'.format(len(all_datasets)))

        print('Concatenating all sets...')
        train_datasets = torch.utils.data.ConcatDataset(all_datasets)
        torch.save(train_datasets, './Aug_set.pt')

    print('Num of samples after concatenation: {}'.format(len(train_datasets)))

    train_sampler = SubsetRandomSampler(range(len(train_datasets)))
    val_sampler = SequentialSampler(range(len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=256, sampler=train_sampler, num_workers=2,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, sampler=val_sampler, num_workers=2,
                                             pin_memory=True)

    print('Size of training loader batches: {}\nSize of validation loader batches: {}'.format(len(train_loader),
                                                                                              len(val_loader)))
    fc_model = model.FullyConnectedNet()
    resnet32_model = model.resnet32()
    resnet47_model = model.resnet47()
    resnet50 = models.resnet50(num_classes=30)

    resnet50.inplanes = 96
    resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

    use_model = resnet50

    optimizer = optim.Adam(use_model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5)
    train.train_model(use_model, optimizer, train_loader, val_loader, scheduler=scheduler, loss_fn=train.RMSELoss,
                      to_mask=False, epochs=50)


if __name__ == '__main__':
    train_test()
