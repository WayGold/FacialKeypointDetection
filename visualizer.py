import numpy as np
import matplotlib.pyplot as plt


def vis_img_kpts(img, kpts):
    plt.imshow(img.reshape(96, 96), cmap='gray')
    # Kpts is formatted [left_eye_center_x, left_eye_center_y, ...]
    plt.scatter(kpts[::2], kpts[1::2], marker='x', s=10)


def rand_vis_dataset(i_dataset, num_vis):
    fig = plt.figure(figsize=(10, 20))
    plt.tight_layout()

    for i in range(num_vis):
        rand_img = np.random.randint(0, len(i_dataset))
        fig.add_subplot(1, num_vis, i + 1)
        vis_img_kpts(i_dataset[rand_img][0], i_dataset[rand_img][1])

    plt.show()
