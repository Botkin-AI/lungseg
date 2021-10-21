from typing import AnyStr, Optional

import numpy as np

from lungseg.models.inference_model import inference_orientation_model, inference_lung_model
from lungseg.models.load_models import load_model


class LungSegmentor:
    """
    Predict lungs (thoracic regions) on CT-volume. Label 1 relates to left lung, label 2 relates to right lung.
    Recommended to use method 'predict' to process whole CT series as 1 volume. Under the hood on this method is
    automatic detecting of body orientation to stable results regardless to position on scan.
    In case of small CT series (less then 10 slices) or a single slice it's better to use method predict_one_slice. But
    in this case be sure that body oriented in a right way - spine at bottom, right lung on left part of image.
    All method expect to get data in a Hounsfield values.
    In case of repeatedly usage you can load model's weights on local path. In this case point this paths as below
    :param local_path_bodyorient: path to weights of body orientation model
    :param local_path_lungseg: path to weights of lung segmentation model
    """

    def __init__(self, local_path_bodyorient: Optional[AnyStr] = None, local_path_lungseg: Optional[AnyStr] = None):
        self.model_bodyorient = load_model(model_name='CT_BODYORIENT', ckpt_path=local_path_bodyorient)
        self.model_lungseg = load_model(model_name='CT_LUNG', ckpt_path=local_path_lungseg)

    def predict(self, slices: np.array, batch_size: int = 8) -> np.array:
        """
        Predict lungs segmentation mask for CT volume with values in Hounsfield units.
        Expected shapes of slices [H x W x n_slice], where each slice in axial projection. n_slice should be >10 for
        needs of body orientation detection.
        :param slices: np.array - volume CT data in HU scale. 3 dims, with last dim along axial slices.
        :param batch_size: size of batch (depends on GPU memory)
        :return: predicted lung mask shape of slices with labels: 0 - background, 1 - left lung, 2 - right lung.
        """
        if slices.shape[2] < 10:
            raise ValueError(
                "Expected to get volume grater than 10 slices. Consider to use method predict_one_slice for "
                "smaller number")

        need_to_flip = inference_orientation_model(self.model_bodyorient, slices)
        lung_predictions = inference_lung_model(self.model_lungseg, slices, batch_size=batch_size,
                                                need_to_flip=need_to_flip)
        return lung_predictions

    def predict_one_slice(self, one_slice: np.array) -> np.array:
        """
        Predict lungs segmentation mask for CT slice with values in Hounsfield units.
        Expected axial slice [H x W].
        :param one_slice: np.array - slice of CT data in HU scale. 2 dims
        :return: predicted lung mask shape of slices with labels: 0 - background, 1 - left lung, 2 - right lung.
        """
        one_slice_3d = np.expand_dims(np.squeeze(one_slice), 2)
        lung_predictions = inference_lung_model(self.model_lungseg, one_slice_3d, batch_size=1)
        return lung_predictions.reshape(one_slice)
