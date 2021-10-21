import numpy as np
from skimage.measure import label


def lung_top_volume_filter(lungmask: np.array, return_count_top=False) -> np.array:
    n_top = 1

    vois = label(lungmask, background=0)
    unique_labels, count_labels = np.unique(vois, return_counts=True)
    # exclude background label
    count_labels = count_labels[unique_labels != 0]
    unique_labels = unique_labels[unique_labels != 0]
    if len(unique_labels) < 2:  # if where only 1 connected volume -- do nothing
        if return_count_top:
            return lungmask, n_top
        else:
            return lungmask

    indexes_top_2_vol = count_labels.argsort()[-2:][::-1]
    top_1_vol_label, top_2_vol_label = tuple(unique_labels[indexes_top_2_vol])
    top_1_vol_count, top_2_vol_count = tuple(count_labels[indexes_top_2_vol])

    filtered_lungmask = np.zeros_like(lungmask)
    filtered_lungmask[vois == top_1_vol_label] = 1

    # assumed that if lungs separated, second lung should be at list as 15% at first
    if top_2_vol_count >= top_1_vol_count * 0.15:
        filtered_lungmask[vois == top_2_vol_label] = 1
        n_top = 2

    filtered_lungmask = lungmask * filtered_lungmask  # apply mask to lungmask
    if return_count_top:
        return filtered_lungmask, n_top
    else:
        return filtered_lungmask
