import torch
import visualizer
import numpy as np


def RMSELoss(prediction, y):
    return torch.sqrt(torch.mean((prediction - y) ** 2))


def train_model(model, optim, loader_train, loader_val, scheduler=None,
                loss_fn=RMSELoss, epochs=1, log_every=50):
    """
    Args:
        optim (torch.optim.Optimizer):
        model (torch.nn.Module):
        loader_train (torch.utils.data.DataLoader):
        loader_val (torch.utils.data.DataLoader):
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau):
        loss_fn (callable)
        epochs (int):
        log_every (int):
        lr (float):
    Returns:
    """
    train_losses, val_losses = [], []
    min_val_loss = np.Inf

    USE_GPU, device = check_GPU()
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    for e in range(epochs):
        train_loss = 0
        val_loss = 0

        for i, (img, kpts) in enumerate(loader_train):
            model.float().train()

            if USE_GPU:
                img = img.cuda()
                kpts = kpts.cuda()

            prediction = model(img)
            loss = loss_fn(prediction, kpts)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item()
            val_loss = evaluate(model, loader_val, loss_fn)

        if scheduler is not None:
            scheduler.step(val_loss)

        train_losses.append(train_loss / len(loader_train))
        val_losses.append(val_loss / len(loader_val))

        print("Epoch: {}/{} ".format(e + 1, epochs),
              "Average Training Loss: {:.4f}".format(train_losses[-1]),
              "Average Val Loss: {:.4f}".format(val_losses[-1]))

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), './best_model.pt')
            print('Improvement Detected, Saving to ./best_model.pt')

    visualizer.vis_loss(train_losses, val_losses)


def evaluate(model, val_loader, loss_fn):
    USE_GPU, device = check_GPU()
    val_loss = 0

    with torch.no_grad():
        model.eval()
        for img, kpts in val_loader:
            if USE_GPU:
                img = img.cuda()
                kpts = kpts.cuda()
            prediction = model(img)
            loss = loss_fn(prediction, kpts)
            val_loss += loss.item()

    return val_loss


def check_GPU():
    USE_GPU = False
    device = torch.device('cpu')

    if torch.cuda.is_available():
        USE_GPU = True
        device = torch.device('cuda')

    return USE_GPU, device
