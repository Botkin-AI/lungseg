from pathlib import Path

from torchvision.transforms import Compose

from .transform import Normalizer, Zoomer, ChannelsChecker, ToTensor

PATH_CKPT_DIR = Path(__file__).parent.resolve() / 'checkpoints'


class NeedToFlip:  # output of body orientation model
    def __init__(self, x=False, y=False, z=False):
        self.x: bool = x  # spine <-> stomach
        self.y: bool = y  # left <-> right
        self.z: bool = z  # head <-> legs


class ModelRegister:
    CT_LUNG = dict(
        ckpt_url='https://storage.yandexcloud.net/botkin-opensource/model_lungseg_v1.ckpt',
        ckpt_local_path=PATH_CKPT_DIR / 'model_lungseg_v1.ckpt',
        n_class=3,
        label_mapping={1: 'LEFT_LUNG', 2: 'RIGHT_LUNG'},
        type_mapping={'LEFT_LUNG': 1, 'RIGHT_LUNG': 2},
        transform=Compose([Zoomer(new_shape=(384, 384), zoom_order=2, with_target=False),
                           Normalizer(with_target=False),
                           ChannelsChecker(with_target=False),
                           ToTensor(with_target=False)]))
    CT_BODYORIENT = dict(
        ckpt_url='https://storage.yandexcloud.net/botkin-opensource/model_bodyorient_v1.ckpt',
        ckpt_local_path=PATH_CKPT_DIR / 'model_bodyorient_v1.ckpt',
        n_class=3,
        transform=Compose([Normalizer(), ToTensor()]))
