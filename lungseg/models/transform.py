from typing import Dict, Tuple

import numpy as np
import torch
from scipy.ndimage.interpolation import zoom


class Transformer:
    """
    Base for Transformer class. Should be implemented for both cases: with target and without (for inference)
    """

    def __init__(self, with_target: bool = True):
        self.input_key = 'input'
        self.target_key = 'target'
        self.with_target = with_target

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Normalizer(Transformer):
    """NormalizeInput - cast array to distribution [0 - pixel_mean, 1 - pixel_mean]"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_bound = -1000
        self.max_bound = 400.0
        self.mean = 0.25

    def __call__(self, batch_dict: Dict[str, np.array]):
        sample = batch_dict[self.input_key].astype('float32')
        sample = (sample - self.min_bound) / (self.max_bound - self.min_bound)
        sample = np.clip(sample, 0., 1.)
        sample = sample - self.mean
        batch_dict[self.input_key] = sample
        return batch_dict

    def __repr__(self):
        msg = f"Normalizer(min_bound={self.min_bound}, max_bound={self.max_bound}, mean={self.mean})"
        return msg


class Zoomer(Transformer):
    """Zoom to expected shape"""

    def __init__(self, new_shape=None, zoom_order=1, **kwargs):
        super().__init__(**kwargs)
        self.new_shape = new_shape
        self.zoom_order_input = zoom_order

    def __call__(self, batch_dict: Dict[str, np.array]):
        sample = batch_dict[self.input_key].astype('float32')
        old_shape = sample.shape
        assert len(old_shape) == len(self.new_shape), "Mismatch of dimensions for old and new size in Zoomer"

        zoom_factor = [new / old for new, old in zip(self.new_shape, old_shape)]
        batch_dict[self.input_key] = zoom(sample, zoom_factor, order=self.zoom_order_input).astype(np.int16)

        if self.with_target:
            target = batch_dict[self.target_key]
            batch_dict[self.target_key] = zoom(target, zoom_factor, order=0)
        return batch_dict

    def __repr__(self):
        msg = f"Zoomer(new_shape={self.new_shape})"
        return msg


def get_dim_param(array: np.array) -> Tuple[int, int]:
    """
    Return ndim and dim of picture channels
    :param array: input array
    :return: (ndim, dim of channels). dim of channels = -1, if all dimensions equal.
    """
    ndim = np.ndim(array)
    if np.unique(array.shape).size == 1 and ndim == 2:  # input is 2-dim square picture like (512, 512)
        channel_dim = -1
    else:
        channel_dim = np.argmin(array.shape).item()  # lowest dim assumed to be channel_dim [ex. (1, 512, 512)]
    return ndim, channel_dim


class ChannelsChecker(Transformer):
    """
    Convert input and target to expected format [channels, W, H]
    """

    def __init__(self, ndim: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.ndim_expected = ndim

    def __call__(self, batch_dict: Dict[str, np.array]) -> Dict[str, np.array]:

        sample = batch_dict[self.input_key]

        ndim, which_ch = get_dim_param(sample)

        if which_ch == -1 and ndim == self.ndim_expected - 1:
            sample = np.expand_dims(sample, axis=0)
        elif which_ch == 0 and ndim == self.ndim_expected:
            pass
        elif which_ch == 2 and ndim == self.ndim_expected:
            sample = np.moveaxis(sample, 2, 0)
        else:
            pass

        batch_dict[self.input_key] = sample.astype('float32')

        if self.with_target:
            target = batch_dict[self.target_key]
            ndim_tar, which_ch_tar = get_dim_param(target)
            if which_ch_tar == -1 and ndim_tar == self.ndim_expected - 1:
                target = np.expand_dims(target, axis=0)
            elif which_ch_tar == 0 and ndim_tar == self.ndim_expected:
                pass
            elif which_ch_tar == 2 and ndim_tar == self.ndim_expected:
                target = np.moveaxis(target, 2, 0)
            else:
                pass
            batch_dict[self.target_key] = target.astype('float32')

        return batch_dict

    def __repr__(self):
        msg = f"ChannelsChecker(ndim={self.ndim_expected})"
        return msg


class ToTensor(Transformer):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, batch_dict):
        return {k: torch.from_numpy(batch_dict[k]) for k in batch_dict.keys()}

    def __repr__(self):
        msg = f"ToTensor()"
        return msg
