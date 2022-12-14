import torch
import torch.optim as optim
import torch.nn.functional as F


def train_model(model, loader_train, optimizer, epochs=1, log_every=100):
    """

    Args:
        model (torch.nn.Sequential):
        loader_train (torch.utils.data.DataLoader):
        optimizer (torch.optim.Optimizer):
        epochs (int):
        log_every (int):

    Returns:

    """
    USE_GPU, device = check_GPU()
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    for e in range(epochs):
        for i, img, kpts in enumerate(loader_train):
            if USE_GPU:
                img = img.cuda()
                kpts = kpts.cuda()

            prediction = model(img)
            loss = F.cross_entropy(prediction, kpts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_every == 0:
                print('Iteration %d, loss = %.4f' % (i, loss.item()))


def check_GPU():
    USE_GPU = False
    device = torch.device('cpu')

    if torch.cuda.is_available():
        USE_GPU = True
        device = torch.device('cuda')

    print('using device:', device)

    return USE_GPU, device
