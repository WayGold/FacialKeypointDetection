import torch
import visualizer


def loadModel(path, model_class):
    model = model_class
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def predict(model, imgs, kpts=None, vis=True):
    with torch.no_grad():
        pred_kpts = model(imgs)
    if vis:
        visualizer.vis_predication(imgs, pred_kpts, kpts)
    return pred_kpts
