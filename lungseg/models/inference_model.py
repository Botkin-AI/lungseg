import numpy as np
import torch
from torch import nn

from torch.utils.data import DataLoader
from scipy.ndimage import zoom

from .datagen import DataReader2DFrom3D
from .postprocess import lung_top_volume_filter
from .model_register import ModelRegister, NeedToFlip


def inference_lung_model(model_lungseg, slices, batch_size, need_to_flip: NeedToFlip = NeedToFlip()):
    """Return probability """

    scale_zoom_out = (slices.shape[0] / 384., slices.shape[1] / 384., 1)
    slices_copy = slices.copy()
    if need_to_flip.x:
        slices_copy = np.flip(slices_copy, axis=0)
    if need_to_flip.y:
        slices_copy = np.flip(slices_copy, axis=1)

    lungseg_transform = ModelRegister.CT_LUNG['transform']
    lungseg_datagen = DataReader2DFrom3D(volume=slices_copy, transform=lungseg_transform)

    dataloader = DataLoader(lungseg_datagen, batch_size=batch_size, shuffle=False, num_workers=0)

    lung_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            if torch.cuda.is_available():
                pred = model_lungseg(batch['input'].cuda()).argmax(dim=1).cpu().numpy()
            else:
                pred = model_lungseg(batch['input']).argmax(dim=1).numpy()

            lung_predictions.append(pred)
    lung_predictions = np.concatenate(lung_predictions, axis=0)
    lung_predictions = np.moveaxis(lung_predictions, 0, 2)

    lung_predictions = lung_top_volume_filter(lung_predictions)
    lung_predictions = zoom(lung_predictions, scale_zoom_out, order=0)

    if need_to_flip.x:
        lung_predictions = np.flip(lung_predictions, axis=0)
    if need_to_flip.y:
        lung_predictions = np.flip(lung_predictions, axis=1)

    return lung_predictions


def inference_orientation_model(model_orientation, slices):
    # thin out input volume (regularly) to get only 10 slices
    step = int(slices.shape[2] / 10)
    slc_indexes = list(range(0, slices.shape[2], step))[:10]

    slices_chosen = np.zeros((10, 256, 256), dtype=np.int16)

    for n, i_slc in enumerate(slc_indexes):
        slc = slices[:, :, i_slc]
        slc = zoom(slc, (256 / slc.shape[0], 256 / slc.shape[1]))
        slices_chosen[n] = slc.copy()

    orientation_transformer = ModelRegister.CT_BODYORIENT['transform']
    data = orientation_transformer({'input': slices_chosen})

    with torch.no_grad():
        if torch.cuda.is_available():
            pred = nn.Sigmoid()(model_orientation(data['input'].unsqueeze(0).cuda())).detach().cpu().numpy()
        else:
            pred = nn.Sigmoid()(model_orientation(data['input'].unsqueeze(0))).numpy()

    pred = tuple(pred.flatten() > 0.5)

    return NeedToFlip(x=pred[1], y=pred[2], z=pred[0])
