from enum import Enum
from typing import List

import numpy as np
import xarray as xr

from netcdf_to_gltf_converter.config import Config
from netcdf_to_gltf_converter.netcdf.wrapper import DatasetWrapper, GridWrapper, VariableWrapper

class XBeachGrid(GridWrapper):
    
    def __init__(self, x_coord_var: xr.DataArray, y_coord_var: xr.DataArray):

        self.node_x = x_coord_var.values.flatten()
        self.node_y = y_coord_var.values.flatten()

        n_vertix_cols = len(x_coord_var.data[0])
        
        triangles = []

        for node_index in range(len(self.node_x)):
            if node_index >= (n_vertix_cols - 1) * n_vertix_cols:
                continue

            if (node_index + 1) % n_vertix_cols == 0:
                continue

            triangle1 = [
                node_index,
                node_index + n_vertix_cols,
                node_index + n_vertix_cols + 1,
            ]
            triangles.append(triangle1)

            triangle2 = [node_index, triangle1[2], triangle1[2] - n_vertix_cols]
            triangles.append(triangle2)

        self.face_node_connectivity = triangles

class XBeachVariable(VariableWrapper):
    """Class that serves as a wrapper object for an xarray.DataArray with UGrid conventions.
    The wrapper allows for easier retrieval of relevant data.
    """

    def __init__(self, data: xr.DataArray) -> None:
        """Initialize a UgridVariable with the specified data.

        Args:
            data (xr.DataArray): The variable data.
        """
        self._data = data
        self._coordinates = self._get_coordinates()

    @property
    def time_index_max(self) -> int:
        """Get the maximum time step index for this data variable.

        Returns:
            int: An integer specifying the maximum time step index.
        """
        return self._data.sizes["globaltime"] - 1

    def get_data_at_time(self, time_index: int) -> np.ndarray:
        """Get the variable values at the specified time index.

        Args:
            time_index (int): The time index.

        Returns:
            np.ndarray: A 1D np.ndarray of floats.
        """
        return self._data.isel(globaltime=time_index).to_numpy()

class XBeachDataset(DatasetWrapper):
    """Class that serves as a wrapper object for an xarray.Dataset with UGrid conventions.
    The wrapper allows for easier retrieval of relevant data.
    """

    def __init__(self, dataset: xr.Dataset) -> None:
        """Initialize a UgridDataset with the specified arguments.

        Args:
            dataset (xr.Dataset): The xarray Dataset.
        """
        self._dataset = dataset

    @property
    def grid(self) -> XBeachGrid:
        """Get the XBeachGrid from the data set.

        Returns:
            XBeachGrid: A XBeachGrid created from the data set.
        """        
        return XBeachGrid(self.x_coord_vars[0], self.y_coord_vars[0])
        
    @property
    def min_x(self) -> float:
        """Gets the smallest x-coordinate of the grid.

        Returns:
            float: A floating value with the smallest x-coordinate.
        """
        return self.x_coord_vars[0].values.min()

    @property
    def min_y(self) -> float:
        """Gets the smallest y-coordinate of the grid.

        Returns:
            float: A floating value with the smallest y-coordinate.
        """
        return self.y_coord_vars[0].values.min()

    def get_variable(self, variable_name: str) -> XBeachVariable:
        """Get the variable with the specified name from the data set.

        Args:
            variable_name (str): The variable name.

        Returns:
            UgridVariable: A UgridVariable.

        Raises:
            ValueError: When the dataset does not contain a variable with the name.
        """
        data = self.get_array(variable_name)
        return XBeachVariable(data)