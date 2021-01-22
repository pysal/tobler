"""tools for dealing with rasters."""

import ast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import features
from shapely.geometry import shape


def _parse_geom(geom_str):
    return shape(ast.literal_eval(geom_str))


def extract_raster_features(gdf, raster_path, pixel_types=None, nodata=255):
    """Generate a geodataframe by extracting shapes of raster features using rasterio's features module.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        geodataframe  defining the area of interest. The input raster will be
        clipped to the extent of the geodataframe
    raster_path : str
        path to raster file, such as downloaded from <https://lcviewer.vito.be/download>
    pixel_types : list-like, optional
        subset of pixel values to extract, by default None. If None, this function
        may generate a very large geodataframe
    nodata : int, optional
        pixel value denoting "no data" in input raster

    Returns
    -------
    geopandas.GeoDataFrame
        geodataframe whose rows are the zones extracted by the rasterio.features module.
        The geometry of each zone is the boundary of a contiguous group of pixels with 
        the same value; the `value` column contains the pixel value of each zone.
    """
    with rio.open(raster_path) as src:

        raster_crs = src.crs.data
        gdf = gdf.to_crs(raster_crs)
        geomask = [gdf.unary_union.__geo_interface__]

        out_image, out_transform = rio.mask.mask(
            src, geomask, nodata=nodata, crop=True
        )  # clip to AoI using a vector layer

        if pixel_types:
            mask = np.isin(out_image, pixel_types)  # only include developed pixels
            shapes = list(
                features.shapes(out_image, mask=mask, transform=out_transform)
            )  # convert regions to polygons
        else:
            shapes = list(features.shapes(out_image, transform=out_transform))
    res = list(zip(*shapes))
    geoms = pd.Series(res[0], name="geometry").apply(str).apply(_parse_geom)
    vals = pd.Series(res[1], name="value")
    gdf = gpd.GeoDataFrame(vals, geometry=geoms)
    gdf.crs = raster_crs

    return gdf
