import torch
import visualizer


def loadModel(path, model_class):
    model = model_class
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def predict(model, imgs, kpts=None, vis=True, comp_kpts=None):
    with torch.no_grad():
        pred_kpts = model(imgs)
    if vis:
        if comp_kpts is not None:
            visualizer.vis_predication(imgs, pred_kpts, comp_kpts)
        else:
            visualizer.vis_predication(imgs, pred_kpts, kpts)

    return pred_kpts
