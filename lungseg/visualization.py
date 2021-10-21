from typing import Union, Tuple, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def prepare_mask(mask: np.array) -> np.array:
    mask_c = np.copy(mask).astype('float')
    mask_c[mask_c <= 0.1] = np.nan
    return mask_c


def plot_seria_to_10(img_vol: np.array, masks_vol: Optional[np.array],
                     img_extrems: Sequence[Tuple[Union[int, float], Union[int, float]]] = (-1000, 400),
                     msk_extrems: Tuple[int, int] = (0, 1), alpha: float = 0.3) -> Figure:
    plt.close('all')

    if img_vol.shape[2] >= 10:
        step = int(img_vol.shape[2] / 10)
        slcs = list(range(0, img_vol.shape[2], step))[:10]
    else:  # replicate last slices
        slcs = list(range(0, img_vol.shape[2], 1)) + [img_vol.shape[2] - 1] * (10 - img_vol.shape[2])

    img_min, img_max = img_extrems
    msk_min, msk_max = msk_extrems

    fig, axes = plt.subplots(2, 5, figsize=(50, 20))
    for i, slc in enumerate(slcs):
        ax = axes.reshape(-1, 1)[i][0]
        ax.imshow(img_vol[:, :, slc], cmap='gray', vmin=img_min, vmax=img_max)
        if masks_vol is not None:
            ax.imshow(prepare_mask(masks_vol[:, :, slc]), cmap='YlOrRd', interpolation='none', vmax=msk_max,
                      vmin=msk_min, alpha=alpha)
        ax.axis('off')
    fig.tight_layout()
    return fig
