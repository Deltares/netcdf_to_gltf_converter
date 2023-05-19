import numpy as np

from netcdf_to_gltf_converter.data.mesh import MeshAttributes, TriangularMesh
from netcdf_to_gltf_converter.utils.arrays import float32_array, uint32_array


class TestTriangularMesh:
    def test_initializer(self):
        vertex_positions = float32_array(
            [
                [0, 0, 1],
                [1, 0, 2],
                [1, 1, 3],
                [0, 1, 4],
            ]
        )
        base_geometry = MeshAttributes(
            vertex_positions=vertex_positions, mesh_color=[0.38, 0.73, 0.78, 1.0]
        )

        triangles = uint32_array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )

        transformation = MeshAttributes(
            vertex_positions=float32_array(
                [
                    [0, 0, 0.5],
                    [0, 0, -0.5],
                    [0, 0, 0.5],
                    [0, 0, -1.0],
                ]
            ),
            mesh_color=[0.38, 0.73, 0.78, 1.0],
        )

        metallic_factor = 0.5
        roughness_factor = 0.75

        triangular_mesh = TriangularMesh(
            base_geometry,
            triangles,
            [transformation],
            metallic_factor,
            roughness_factor,
        )

        assert triangular_mesh.base == base_geometry
        assert np.array_equal(triangular_mesh.triangles, triangles)
        assert len(triangular_mesh.transformations) == 1
        assert triangular_mesh.transformations[0] == transformation
        assert triangular_mesh.metallic_factor == metallic_factor
        assert triangular_mesh.roughness_factor == roughness_factor
