from typing import List

import numpy as np
import xarray as xr

from netcdf_to_gltf_converter.config import Config
from netcdf_to_gltf_converter.netcdf.wrapper import UgridDataset

xr.set_options(keep_attrs=True)
"""Attributes need to be preserved when creating a new DataArray with a transformation."""


def shift(dataset: UgridDataset):
    """
    Shift the x- and y-coordinates in the data set, such that the smallest x and y become the origin (0,0).
    The original data set is updated with the new coordinates.

    Args:
        dataset (xr.UgridDataset): The Ugrid dataset to transform the coordinates for.
        config (Config): The converter configuration.
    """

    node_x_var, node_y_var = dataset.node_coord_vars
    edge_x_var, edge_y_var = dataset.edge_coord_vars
    face_x_var, face_y_var = dataset.face_coord_vars

    shift_x = node_x_var.values.min()
    shift_y = node_y_var.values.min()

    _shift(node_x_var, shift_x, dataset)
    _shift(node_y_var, shift_y, dataset)
    _shift(edge_x_var, shift_x, dataset)
    _shift(edge_y_var, shift_y, dataset)
    _shift(face_x_var, shift_x, dataset)
    _shift(face_y_var, shift_y, dataset)


def _shift(variable: xr.DataArray, shift: float, dataset: UgridDataset):
    shifted_coords_var = variable - shift
    dataset.set_variable(shifted_coords_var)


def scale(dataset: UgridDataset, variables: List[str], scale: float):
    """
    If scaling is required, scale the x- and y-coordinates and the data values, with the scaling factor that is specified in the Config.
    The original data set is updated with the new coordinates.

    Args:
        dataset (xr.UgridDataset): The Ugrid dataset to transform the coordinates for.
        variables (List[str]): The names of the variables to scale.
    """

    node_x_var, node_y_var = dataset.node_coord_vars
    edge_x_var, edge_y_var = dataset.edge_coord_vars
    face_x_var, face_y_var = dataset.face_coord_vars

    _scale(node_x_var, dataset, scale)
    _scale(node_y_var, dataset, scale)
    _scale(edge_x_var, dataset, scale)
    _scale(edge_y_var, dataset, scale)
    _scale(face_x_var, dataset, scale)
    _scale(face_y_var, dataset, scale)

    _scale_data(dataset, variables, scale)


def _scale(variable: xr.DataArray, dataset: UgridDataset, scale: float):
    scaled_coords_var = variable * scale
    dataset.set_variable(scaled_coords_var)


def _scale_data(dataset: UgridDataset, variables: List[str], scale: float):
    for variable in variables:
        _scale(dataset.get_variable(variable), dataset, scale)


def swap_yz(data: np.ndarray):
    """If required, swap the y and z axes.

    The original data is updated.

    Args:
        data (np.ndarray): The data to swap the y and z axes for.
        config (Config): The converter configuration.
    """
    copy = data.copy()
    data[:, 1], data[:, 2] = copy[:, 2], copy[:, 1]
