from pathlib import Path

import numpy as np
import pytest

from netcdf_to_gltf_converter.config import Config, Variable
from netcdf_to_gltf_converter.netcdf.importer import Importer
from tests.utils import resources


class TestImporter:
    def test_import_from(self):
        file_path = resources / "3x3nodes_rectilinear_map.nc"

        variable = Variable(
            name="Mesh2d_waterdepth",
            color=[0.38, 0.73, 0.78, 1.0],
            metallic_factor=0.0,
            roughness_factor=0.11,
            use_threshold=True,
            threshold_height=0.01,
            threshold_color=[1.0, 1.0, 1.0, 1.0],
        )

        config = Config(
            time_index_start=0,
            times_per_frame=1,
            shift_coordinates=True,
            scale=1.0,
            swap_yz=False,
            variables=[variable],
        )

        importer = Importer()
        triangular_meshes = importer.import_from(file_path, config)
        data_mesh = triangular_meshes[0]

        exp_triangles = np.array(
            [
                [5, 2, 0],
                [5, 0, 1],
                [1, 0, 3],
                [1, 3, 6],
                [2, 7, 4],
                [2, 4, 0],
                [0, 4, 8],
                [0, 8, 3],
            ],
            dtype=np.uint32,
        )

        exp_vertex_positions = np.array(
            [
                [1.0, 1.0, 5.0],
                [0.0, 1.0, 5.0],
                [1.0, 0.0, 5.0],
                [1.0, 2.0, 5.0],
                [2.0, 1.0, 4.0],
                [0.0, 0.0, 5.0],
                [0.0, 2.0, 5.0],
                [2.0, 0.0, 4.0],
                [2.0, 2.0, 4.0],
            ],
            dtype=np.float32,
        )

        exp_vertex_transformations = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        assert np.array_equal(data_mesh.triangles, exp_triangles)
        assert np.array_equal(data_mesh.base.vertex_positions, exp_vertex_positions)
        assert len(data_mesh.transformations) == 4
        assert np.array_equal(
            data_mesh.transformations[2].vertex_positions,
            exp_vertex_transformations,
        )

    def test_import_from_netcdf_does_not_exist_raises_error(self):
        netcdf = Path("path/to/file.netcdf")
        config = Config(
            time_index_start=0,
            times_per_frame=1,
            shift_coordinates=True,
            scale=1.0,
            swap_yz=False,
            variables=[],
        )

        importer = Importer()

        with pytest.raises(ValueError) as error:
            _ = importer.import_from(netcdf, config)

        assert str(error.value) == rf"NetCDF file does not exist: {netcdf}"
