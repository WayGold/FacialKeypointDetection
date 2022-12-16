import torch
import visualizer


def loadModel(path, model_class):
    model = model_class
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def predict(model, img, kpts, vis=True):
    pred_kpts = model(img)
    if vis:
        visualizer.vis_predication(img, kpts, pred_kpts)
    return pred_kpts
