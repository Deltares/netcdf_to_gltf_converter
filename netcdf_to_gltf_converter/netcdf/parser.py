from typing import List

import numpy as np
import xarray as xr
from xugrid import Ugrid2d

from netcdf_to_gltf_converter.config import Config, Variable
from netcdf_to_gltf_converter.custom_types import Color
from netcdf_to_gltf_converter.data.mesh import MeshAttributes, TriangularMesh
from netcdf_to_gltf_converter.netcdf.wrapper import UgridDataset, UgridVariable
from netcdf_to_gltf_converter.preprocessing.interpolation import Interpolator, Location
from netcdf_to_gltf_converter.preprocessing.transformation import scale, shift
from netcdf_to_gltf_converter.preprocessing.triangulation import Triangulator
from netcdf_to_gltf_converter.utils.arrays import uint32_array
from netcdf_to_gltf_converter.utils.sequences import inclusive_range


class Parser:
    """Class to parse a xr.DataArray into a set of TriangularMeshes."""

    def __init__(self) -> None:
        """Initialize a Parser."""

        self._interpolator = Interpolator()
        self._triangulator = Triangulator()

    def parse(self, dataset: xr.Dataset, config: Config) -> List[TriangularMesh]:
        """Parse the provided data set to a list of TriangularMeshes.

        Args:
            dataset (xr.Dataset): The NetCDF dataset.
            config (Config): The converter configuration.
        """
        ugrid_dataset = UgridDataset(dataset, config)
        Parser._transform_grid(config, ugrid_dataset)
        triangulated_grid = self._triangulator.triangulate(ugrid_dataset.ugrid2d)

        triangular_meshes = []

        for variable in config.variables:
            data_mesh = self._parse_variable(
                variable, triangulated_grid, ugrid_dataset, config
            )
            triangular_meshes.append(data_mesh)

            if variable.use_threshold:
                threshold_mesh = Parser._get_threshold_mesh(
                    data_mesh,
                    variable.threshold_height,
                    variable.threshold_color,
                    config,
                )
                triangular_meshes.append(threshold_mesh)

        return triangular_meshes

    def _parse_variable(
        self,
        variable: Variable,
        grid: Ugrid2d,
        ugrid_dataset: UgridDataset,
        config: Config,
    ):
        data = ugrid_dataset.get_ugrid_variable(variable.name)
        base_vertex_positions = self._interpolate(data, config.time_index_start, grid)

        transformations = []

        for time_index in Parser._get_time_indices(data.time_index_max, config):
            trfm_vertex_positions = self._interpolate(data, time_index, grid)
            vertex_displacements = Parser.calculate_displacements(
                trfm_vertex_positions, base_vertex_positions
            )
            transformation = MeshAttributes(vertex_displacements, variable.color)
            transformations.append(transformation)

        mesh = TriangularMesh(
            MeshAttributes(base_vertex_positions, variable.color),
            uint32_array(grid.face_node_connectivity),
            transformations,
            variable.metallic_factor,
            variable.roughness_factor,
        )

        if config.swap_yz:
            mesh.swap_yz()

        return mesh

    @staticmethod
    def _get_threshold_mesh(
        triangular_mesh: TriangularMesh, height: float, color: Color, config: Config
    ) -> TriangularMesh:
        vertex_positions = triangular_mesh.base.vertex_positions.copy()
        height *= config.scale

        if config.swap_yz:
            vertex_positions[:, 1] = height
        else:
            vertex_positions[:, -1] = height

        mesh_attributes = MeshAttributes(
            vertex_positions=vertex_positions, mesh_color=color
        )

        return TriangularMesh(
            base=mesh_attributes,
            triangles=triangular_mesh.triangles,
            transformations=[],
            metallic_factor=0.0,
            roughness_factor=1.0,
        )

    @staticmethod
    def _get_time_indices(time_index_max: int, config: Config):
        start = config.time_index_start + config.times_per_frame

        if config.time_index_end is not None:
            end = config.time_index_end
        else:
            end = time_index_max

        return inclusive_range(start, end, config.times_per_frame)

    @staticmethod
    def _transform_grid(config: Config, ugrid_dataset: UgridDataset):
        if config.shift_coordinates:
            shift(ugrid_dataset)

        if config.scale != 1.0:
            variables = [var.name for var in config.variables]
            scale(ugrid_dataset, variables, config.scale)

    def _interpolate(self, data: UgridVariable, time_index: int, grid: Ugrid2d):
        return self._interpolator.interpolate_nearest(
            data.coordinates, data.get_data_at_time(time_index), grid, Location.nodes
        )

    @staticmethod
    def calculate_displacements(data: np.ndarray, vertex_positions: np.ndarray):
        return np.subtract(
            data,
            vertex_positions,
            dtype=np.float32,
        )
