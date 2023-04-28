from typing import List

import numpy as np

from netcdf_to_gltf_converter.data.colors import DEFAULT_MESH_COLOR, PLANE_MESH_COLOR
from netcdf_to_gltf_converter.utils.arrays import float32_array, validate_2d_array


class MeshAttributes:
    def __init__(
        self, vertex_positions: np.ndarray, mesh_color: List = DEFAULT_MESH_COLOR
    ) -> None:
        """Initialize a MeshAttributes with the specified arguments.

        Args:
            vertex_positions (np.ndarray): The vertex positions, an ndarray of floats with shape (n, 3). Each row represents one vertex and contains the x, y and z coordinate of this vertex.
            mesh_color (List): The mesh color, a list of floats with shape (, 4). The four values represent a color defined by its normalized red, green, blue and alpha (RGBA) values.
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
    ) -> None:
        """Initialize a TriangularMesh with the specified arguments.

        Args:
            base (MeshAttributes): The base attributes of the mesh.
            triangles (np.ndarray): The vertex indices per triangle, an ndarray of integers with shape (m, 3). Each row represents one triangle and contains the three vertex indices of this triangle.
            transformations (List[MeshAttributes]): The mesh transformations containing the vertex displacements.

        Raises:
            AssertionError: When the shape or dtype of the `triangles` does not match the described requirements.
            AssertionError: When the size of the attributes in any of the mesh transformations do not match the size in the base mesh geometry.
        """

        self.base = base
        self.triangles = triangles
        self.transformations = transformations
        self.plane = self._get_plane_mesh(height=0.01)

        self._validate()

    def _get_plane_mesh(self, height: float) -> MeshAttributes:
        vertex_positions = self.base.vertex_positions.copy()
        vertex_positions[:, -1] = height
        return MeshAttributes(
            vertex_positions=vertex_positions, mesh_color=PLANE_MESH_COLOR
        )

    def _validate(self):
        validate_2d_array(self.triangles, np.uint32, n_col=3)

        for transformation in self.transformations:
            assert (
                transformation.vertex_positions.size == self.base.vertex_positions.size
            )
            assert transformation.vertex_colors.size == self.base.vertex_colors.size