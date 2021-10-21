import glob
import logging
import os
from typing import AnyStr

import numpy as np
import pydicom
from scipy import stats

logger = logging.getLogger('SeriesLoader')


def convert2hu(slices: np.array, slope=1., intercept=0):
    """
    Convert dicom-data to Hounsfield Units
    """
    slices_c = slope * slices + intercept
    slices_c = np.clip(slices_c.astype(np.int16), a_min=-1000, a_max=1000)
    return slices_c


class SeriesLoader:
    def __init__(self, folder_to_dicoms: AnyStr, filter_dcm=False):
        """
        Read DICOM data from folder. Volumetric data stored in self.slices, dicom tags - in related variables.
        self.slices stacked on the 3rd dim of array, so output dimension would be [H x W x C], there [H x W] - shape in
        an axial projection, C - amount of slices.
        :param folder_to_dicoms: path to folder with series of DICOM files
        :param filter_dcm: if any other files extensions on dicom folder
        """
        self.dicom_folder = folder_to_dicoms

        files_wildcard = '*.dcm' if filter_dcm else '*'
        path_dicoms_list = sorted(glob.glob(os.path.join(folder_to_dicoms, files_wildcard)))
        path_dicoms_list = [f for f in path_dicoms_list if os.path.isfile(f)]

        self.dicom_fnames = [os.path.basename(x) for x in path_dicoms_list]
        self.dicoms_raw = [pydicom.dcmread(x, force=True) for x in path_dicoms_list]
        self.instance_nums = [int(dcm.InstanceNumber) for dcm in self.dicoms_raw]

        self.instance_nums, self.dicom_fnames, self.dicoms_raw = zip(
            *sorted(zip(self.instance_nums, self.dicom_fnames, self.dicoms_raw), key=lambda x: x[0]))

        self.rescale_slope = np.float(self._read_dicom_static_tag('RescaleSlope', default=1.))
        self.rescale_intercept = np.int16(self._read_dicom_static_tag('RescaleIntercept', default=0))
        self.slices = np.stack(self._read_dicom_dynamic_tag('pixel_array'), axis=2).astype(np.int16)
        self.slices = convert2hu(self.slices, slope=self.rescale_slope, intercept=self.rescale_intercept)

        self.patient_id = str(self._read_dicom_static_tag('PatientID', default='Unknown'))
        self.study_id = str(self._read_dicom_static_tag('StudyInstanceUID', default='Unknown'))
        self.series_id = str(self._read_dicom_static_tag('SeriesInstanceUID', default='Unknown'))
        self.study_date = str(self._read_dicom_static_tag('StudyDate', default='Unknown'))
        self.series_date = str(self._read_dicom_static_tag('SeriesDate', default='Unknown'))

        self.slice_thickness = float(self._read_dicom_static_tag('SliceThickness', default=1.))
        self.slice_distance = float(self._determine_slice_distance(default=1.))
        self.pixel_spacing = list(map(float, self._read_dicom_static_tag('PixelSpacing', default=[1., 1.])))

    def _determine_slice_distance(self, default=None, verbose=True):
        """
        Calculate slice distance from slice distance. If not regular - returns most common.
        """
        try:
            slice_location = [float(dcm.SliceLocation) for dcm in self.dicoms_raw]
            slice_location = np.array(slice_location)
            distances = np.abs(np.round(np.diff(slice_location), 2))
            if np.unique(distances).shape[0] != 1 and verbose:
                logging.warning("SeriesLoader: Distances between slices are not uniform")
            return float(stats.mode(distances, axis=None).mode)
        except (AttributeError, ValueError):
            if verbose:
                logging.warning(f"SeriesLoader: Couldn't calculate slice distance, default value will be used instead")
            return default

    def _read_dicom_dynamic_tag(self, tag_name, verbose=True):
        """
        Read tag which changes in series. Returns all tag values.
        """
        try:
            tag_values = [getattr(dcm, tag_name) for dcm in self.dicoms_raw]
            return tag_values
        except (AttributeError, ValueError):
            if verbose:
                logger.warning(f"Couldn't read {tag_name}, default value will be used instead")
            return None

    def _read_dicom_static_tag(self, tag_name, default=None, verbose=True):
        """
        Read tag which remains static for whole series. Returns only first tag value.
        """
        try:
            tag_values = [getattr(dcm, tag_name) for dcm in self.dicoms_raw]
            return tag_values[0]
        except (AttributeError, ValueError):
            if verbose:
                logger.warning(f"Couldn't read {tag_name}, default value will be used instead")
            return default
