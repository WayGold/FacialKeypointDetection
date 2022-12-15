import torch
import train
import model
import numpy as np
import csvLoader as cl
import visualizer as vs
import dataLoader as dl
import torch.optim as optim
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
    lr = 2e-2
    train_csv = cl.load_csv(TRAIN_CSV_PATH)
    print(f'Len of train csv: {len(np.array(train_csv.Image))}')
    csv_allValid, csv_autoFill, csv_missingOnly = cl.clean_csv(train_csv)

    print('Loading Dataset...')
    # allValid_dataset = dl.FacialKptsDataSet(csv_allValid)
    # allValidTrain, allValidVal = dl.getTrainValidationDataSet(csv_allValid, 0.85)
    allValid_dataset = dl.FacialKptsDataSet(csv_autoFill)
    allValidTrain, allValidVal = dl.getTrainValidationDataSet(csv_autoFill, 0.85)

    train_sampler = SubsetRandomSampler(range(len(allValidTrain)))
    val_sampler = SubsetRandomSampler(range(len(allValidVal)))

    train_loader = torch.utils.data.DataLoader(allValidTrain, batch_size=128, sampler=train_sampler, num_workers=3)
    val_loader = torch.utils.data.DataLoader(allValidVal, batch_size=128, sampler=val_sampler, num_workers=3)

    print('Size of training loader batches: {}\nSize of validation loader batches: {}'.format(len(train_loader),
                                                                                              len(val_loader)))
    fc_model = model.FullyConnectedNet()
    resnet32_model = model.resnet32()
    resnet47_model = model.resnet47()

    optimizer = optim.Adam(resnet47_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5)
    train.train_model(resnet47_model, optimizer, train_loader, val_loader, scheduler=scheduler, loss_fn=train.RMSELoss,
                      to_mask=True, epochs=50)


if __name__ == '__main__':
    train_test()
