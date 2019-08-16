import quilt3
from urllib.parse import unquote

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
        a local raster file.

    Returns
    -------
    str
        If the input is in the quilt database, then return the full path,
        otherwise return the original path untouched

    """

    if path in ["nlcd_2011", "nlcd_2001"]:
        try:
            from quilt3.data.rasters import nlcd
            full_path = unquote(nlcd[path+'.tif'].get())
        except ImportError:
            raise(
                "Unable to locate local raster data. You store it locally for "
                "use with the `data.store_rasters()` function"
            )

    else:
        return path
    return full_path
