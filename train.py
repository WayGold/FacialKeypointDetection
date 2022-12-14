import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


def train_model(model, loader_train, loader_val, optimizer, scheduler,
                loss_fn=F.cross_entropy, epochs=1, log_every=100):
    """
    Args:
        model (torch.nn.Sequential):
        loader_train (torch.utils.data.DataLoader):
        loader_val (torch.utils.data.DataLoader):
        optimizer (torch.optim.Optimizer):
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau):
        loss_fn (callable)
        epochs (int):
        log_every (int):
    Returns:
    """
    train_losses, val_losses = [], []
    min_val_loss = np.Inf

    USE_GPU, device = check_GPU()
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    for e in range(epochs):
        train_loss = 0
        val_loss = 0

        for i, img, kpts in enumerate(loader_train):
            model.float().train()

            if USE_GPU:
                img = img.cuda()
                kpts = kpts.cuda()

            prediction = model(img)
            loss = loss_fn(prediction, kpts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i % log_every == 0:
                print(f'Iteration {i}, loss = {loss.item():.4f}')

            val_loss = evaluate(model, loader_val)
            scheduler.step(val_loss)

        train_losses.append(train_loss / len(loader_train))
        val_losses.append(val_loss / len(loader_val))

        print("Epoch: {}/{} ".format(e, epochs),
              "Average Training Loss: {:.4f}".format(train_losses[-1]),
              "Average Val Loss: {:.4f}".format(val_losses[-1]))

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), './best_model.pt')
            print('Improvement Detected, Saving to ./best_model.pt')


def evaluate(model, val_loader):
    if val_loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')

    USE_GPU, device = check_GPU()
    val_loss = 0

    with torch.no_grad():
        model.eval()
        for img, kpts in val_loader:
            if USE_GPU:
                img = img.cuda()
                kpts = kpts.cuda()
            prediction = model(img)
            loss = RMSELoss(prediction, kpts)
            val_loss += loss.item()
        print('Total Loss - {}\nAverage Loss - {}'.format(val_loss, val_loss / len(val_loader)))

    return val_loss


def RMSELoss(prediction, y):
    return torch.sqrt(torch.mean((prediction - y) ** 2))


def check_GPU():
    USE_GPU = False
    device = torch.device('cpu')

    if torch.cuda.is_available():
        USE_GPU = True
        device = torch.device('cuda')

    print('using device:', device)

    return USE_GPU, device
