import numpy as np
import matplotlib.pyplot as plt


def vis_img_kpts(img, kpts):
    """
    Draw image and kpts to plot.

    Args:
        img:            Image data to draw
        kpts:           Keypoint data to draw

    Returns:            None

    """
    plt.imshow(img.reshape(96, 96), cmap='gray')
    # Kpts is formatted [left_eye_center_x, left_eye_center_y, ...]
    plt.scatter(kpts[::2], kpts[1::2], marker='x', s=10)


def rand_vis_dataset(i_dataset, num_vis):
    """
    Given the input dataset, randomly visualize num_vis amount of samples.

    Args:
        i_dataset (FacialKptsDataSet):      Input dataset to draw sample from
        num_vis (int):                      Num of samples to visualize

    Returns:                                Plots of visualizations

    """
    fig = plt.figure(figsize=(10, 20))
    plt.tight_layout()

    for i in range(num_vis):
        rand_img = np.random.randint(0, len(i_dataset))
        fig.add_subplot(1, num_vis, i + 1)
        vis_img_kpts(i_dataset[rand_img][0], i_dataset[rand_img][1])

    plt.show()


def rand_vis_compare_orig_augset(orig_ds, aug_ds_list, num_vis):
    """
    Given the original and augmented sets, randomly draw corresponding samples.

    Args:
        orig_ds (FacialKptsDataSet):                Original Dataset
        aug_ds_list (list(FacialKptsDataSet)):      List of Augmented Datasets
        num_vis (int):                              Num of samples to visualize

    Returns:                                        Plots of visualizations

    """
    fig = plt.figure(figsize=(10, 20))
    plt.tight_layout()
    num_datasets = len(aug_ds_list)

    for index, aug_ds in enumerate(aug_ds_list):
        print('Visualizing dataset #{}...'.format(index))
        for i in range(num_vis):
            rand_img = np.random.randint(0, len(orig_ds))
            # Original Image
            fig.add_subplot(num_vis * num_datasets, 2, index * num_vis * 2 + (i + 1) * 2 - 1)
            vis_img_kpts(orig_ds[rand_img][0], orig_ds[rand_img][1])
            # Augmented Image
            fig.add_subplot(num_vis * num_datasets, 2, index * num_vis * 2 + (i + 1) * 2)
            vis_img_kpts(aug_ds[rand_img][0], aug_ds[rand_img][1])
    plt.show()


def vis_loss(train_losses, val_losses):
    """

    Args:
        train_losses:
        val_losses:

    Returns:

    """
    plt.tick_params(colors='black')
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend(frameon=False)
    plt.show()


def vis_predication(img, orig_kpts, pred_kpts):
    fig = plt.figure(figsize=(10, 20))
    plt.tight_layout()

    fig.add_subplot(1, 2, 1)
    plt.imshow(img.reshape(96, 96), cmap='gray')
    plt.scatter(orig_kpts[::2], orig_kpts[1::2], marker='o', s=100, color='green')

    fig.add_subplot(1, 2, 2)
    plt.imshow(img.reshape(96, 96), cmap='gray')
    plt.scatter(pred_kpts[::2], pred_kpts[1::2], marker='x', s=100, color='red')

    plt.show()
