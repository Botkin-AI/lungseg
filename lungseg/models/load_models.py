from typing import AnyStr, Optional

import torch

from lungseg.models.architectures import UNet, resnest18
from lungseg.models.model_register import ModelRegister


def load_model(model_name: str, ckpt_path: Optional[AnyStr] = None):
    """
    Load model from local path either from url
    :param model_name: Currently supported: ['CT_LUNG', 'CT_BODYORIENT']
    :param ckpt_path: load path there stored pre-downloaded model checkpoint (You should download it manually!)
    :return: requested model with pretrained weights
    """
    if model_name == 'CT_LUNG':
        model_info = ModelRegister.CT_LUNG
        model = UNet(in_ch=1, out_ch=model_info['n_class'], multiclass=model_info['n_class'] > 2)
    elif model_name == 'CT_BODYORIENT':
        model_info = ModelRegister.CT_BODYORIENT
        model = resnest18(input_channels=10, num_classes=model_info['n_class'], return_features=False)
    else:
        raise NameError("Unknown model name. Currently supported: ['CT_LUNG', 'CT_BODYORIENT']")

    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        if not model_info['ckpt_local_path'].is_file() or model_info['ckpt_local_path'].stat().st_size / 1024 / 1024 < 1:
            # second condition checks if file with such name exist, but less then 1 mb
            torch.hub.download_url_to_file(model_info['ckpt_url'], model_info['ckpt_local_path'], progress=True)
        state_dict = torch.load(model_info['ckpt_local_path'], map_location='cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model
