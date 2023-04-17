from pathlib import Path


class Converter:
    """Converter class for converting the NetCDF file to a glTF file."""
    
    def __init__(self, netcdf: Path, gltf: Path) -> None:
        """Initialize a Converter with the specified arguments.

        Args:
            netcdf (Path): Path to the source NetCDF file
            gltf (Path): Path to the destination glTF file
        """
        self._netcdf = netcdf
        self._gltf = gltf

    
    def run(self):
        """Run the conversion."""
        pass