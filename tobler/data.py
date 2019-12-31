from urllib.parse import unquote, urlparse
from warnings import warn
from requests.exceptions import Timeout

import quilt3


def store_rasters():
    """Save raster data to the local quilt package storage.

    Returns
    -------
    None
        Data will be available in the local quilt registry and available
        for use in interpolation functions from the `harmonize` module.

    """
    quilt3.Package.install("rasters/nlcd", "s3://quilt-cgs")


def fetch_quilt_path(path):
    """utility for getting the path to a raster stored with quilt.

    Parameters
    ----------
    path : str
        string identifying raster from CGS quilt database, or full path to
        a local raster file. Current options include "nlcd_2001", "nlcd_2011",
        or the path to a local file.

    Returns
    -------
    str
        If the input is in the quilt database, then return the full path,
        otherwise return the original path untouched

    """

    if path in ["nlcd_2011", "nlcd_2001"]:
        try:
            from quilt3.data.rasters import nlcd

        except ImportError:
            warn(
                "Unable to locate local raster data. If you would like to use "
                "raster data from the National Land Cover Database, you can "
                "store it locally using the `data.store_rasters()` function"
            )
            try:
                nlcd = quilt3.Package.browse(
                    "rasters/nlcd", "s3://quilt-cgs"
                )

            except Timeout:
                warn(
                    "Unable to locate local raster data and unable to reach s3 bucket."
                    "You will be unable to use built-in data during this session. "
                    "If you need these data, please try downloading a local copy "
                    "with the data.store_rasters function, then restart your "
                    "python kernel and try again."
                )
        full_path = unquote(nlcd[path + ".tif"].get())
        full_path = urlparse(full_path).path

    else:
        return path
    return full_path
