from typing import List

import numpy as np

from netcdf_to_gltf_converter.custom_types import Color
from netcdf_to_gltf_converter.preprocessing.transformation import swap_yz as swap_yz
from netcdf_to_gltf_converter.utils.arrays import float32_array, validate_2d_array


class MeshAttributes:
    def __init__(self, vertex_positions: np.ndarray, mesh_color: Color) -> None:
        """Initialize a MeshAttributes with the specified arguments.

        Args:
            vertex_positions (np.ndarray): The vertex positions, an ndarray of floats with shape (n, 3). Each row represents one vertex and contains the x, y and z coordinate of this vertex.
            mesh_color (Color): The mesh color, a list of floats with shape (, 4). The four values represent a color defined by its normalized red, green, blue and alpha (RGBA) values.
        Raises:
            AssertionError: When the shape or dtype of the arrays do not match the requirements.
            AssertionError: When the number of vertex colors does not correspond with the number of vertex positions (number of vertices).
        """
        self.vertex_positions = vertex_positions
        self.vertex_colors = float32_array(len(vertex_positions) * [mesh_color])
        self._validate()

    def _validate(self):
        validate_2d_array(self.vertex_positions, np.float32, n_col=3)
        validate_2d_array(self.vertex_colors, np.float32, n_col=4)
        assert len(self.vertex_positions) == len(self.vertex_colors)


class TriangularMesh:
    def __init__(
        self,
        base: MeshAttributes,
        triangles: np.ndarray,
        transformations: List[MeshAttributes],
        metallic_factor: float,
        roughness_factor: float,
    ) -> None:
        """Initialize a TriangularMesh with the specified arguments.

        Args:
            base (MeshAttributes): The base attributes of the mesh.
            triangles (np.ndarray): The vertex indices per triangle, an ndarray of integers with shape (m, 3). Each row represents one triangle and contains the three vertex indices of this triangle.
            transformations (List[MeshAttributes]): The mesh transformations containing the vertex displacements.
            metallic_factor (float): The metallic factor defining the degree of metallicity or non-metallicity of the mesh material.
            roughness_factor (float):  The roughness factor defining the smoothness or roughness of the mesh material.

        Raises:
            AssertionError: When the shape or dtype of the `triangles` does not match the described requirements.
            AssertionError: When the size of the attributes in any of the mesh transformations do not match the size in the base mesh geometry.
        """

        self.base = base
        self.triangles = triangles
        self.transformations = transformations
        self.metallic_factor = metallic_factor
        self.roughness_factor = roughness_factor

        self._validate()

    def swap_yz(self):
        swap_yz(self.base.vertex_positions)
        swap_yz(self.triangles)

        for transformation in self.transformations:
            swap_yz(transformation.vertex_positions)

    def _validate(self):
        validate_2d_array(self.triangles, np.uint32, n_col=3)

        n_vertices = len(self.base.vertex_positions)
        for transformation in self.transformations:
            assert len(transformation.vertex_positions) == n_vertices
