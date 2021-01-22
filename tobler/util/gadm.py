"""Utilities for fetching data from GADM."""
import os
import tempfile
import time
from warnings import warn

import geopandas as gpd


def get_gadm(code, level=0, use_fsspec=True, gpkg=True, n_retries=3):
    """Collect data from GADM as a geodataframe.

    Parameters
    ----------
    code : str
        three character ISO code for a country
    level : int, optional
        which geometry level to collect, by default 0
    use_fsspec : bool
        whether to use the `fsspec` library
    gpkg : bool
        whether to read from a geopackage or shapefile. If True,
        geopackage will be read; shapefile if False. Ignored if using fsspec
    n_retries : int optional
        number of retries in case read fails from direct stream from GADM.
        Ignored if using fsspec.

    Returns
    -------
    geopandas.GeoDataFrame
        geodataframe containing GADM data

    Notes
    -------
    If not using the fsspec package, this function uses fiona's syntax to read a geodataframe directly with
    geopandas `read_file` function. Unfortunately, sometimes the operation fails
    before the read is complete resulting in an error--or occasionally, a
    geodataframe with missing rows. Repeating the call sometimes helps.

    When using fsspec, the function doesn't suffer these issues, but requires an additional dependency.
    If fsspec is available, this function its syntax to store a temporary file which is then
    read in by geopandas. In theory, the file could be read into fsspec directly
    without storing it in a temporary directory, but when reading a bytestream of GPKG,
    geopandas does not allow the specification of a particular layer (so reading GPKG
    with this method would always returns the layer with index 0 in the geopackage file).
    """
    code = code.upper()

    try:
        import fsspec

        has_fsspec = True
    except ImportError:
        has_fsspec = False

    if use_fsspec and has_fsspec:
        with tempfile.TemporaryDirectory() as temp_path:
            with fsspec.open(
                f"simplecache::zip://*.gpkg::https://biogeo.ucdavis.edu/data/gadm3.6/gpkg/gadm36_{code}_gpkg.zip",
                simplecache={"cache_storage": temp_path},
            ):
                gdf = gpd.read_file(
                    os.path.join(temp_path, os.listdir(temp_path)[0]),
                    layer=f"gadm36_{code}_{level}",
                )
                return gdf
    else:
        warn(
            "Reading data directly from gadm can be unstable. For more predictable "
            "performance install the fsspec package and set `use_fsspec=True`. "
            "See the function notes for more information"
        )
        url = "zip+http://biogeo.ucdavis.edu/data/gadm3.6/"

        attempts = 0
        while attempts < n_retries:
            try:
                if not gpkg:
                    gdf = gpd.read_file(
                        url + f"shp/gadm36_{code}_shp.zip!gadm36_{code}_{level}.shp",
                    )

                else:
                    gdf = gpd.read_file(
                        url + f"gpkg/gadm36_{code}_gpkg.zip!gadm36_{code}.gpkg",
                        layer=f"gadm36_{code}_{level}",
                    )
                return gdf
            except Exception:
                warn("Direct read attempt failed. Retrying.")
                attempts += 1
                time.sleep(1)
        raise ValueError(
            "Unable to read data. Try calling the function again or use fsspec"
        )
