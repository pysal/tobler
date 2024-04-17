"""tools for working with rasters."""

import ast
import multiprocessing
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rasterstats as rs
from joblib import Parallel, delayed
from packaging.version import Version
from rasterio import features
from rasterio.mask import mask
from shapely.geometry import shape

from ..util.util import _check_presence_of_crs

GPD_10 = Version(gpd.__version__) >= Version("1.0.0dev")


def _chunk_dfs(geoms_to_chunk, n_jobs):
    chunk_size = geoms_to_chunk.shape[0] // n_jobs + 1
    for i in range(n_jobs):
        start = i * chunk_size
        yield geoms_to_chunk.iloc[start : start + chunk_size]


def _parse_geom(geom_str):
    return shape(ast.literal_eval(geom_str))


def _apply_parser(df):
    return df.apply(_parse_geom)


def _fast_append_profile_in_gdf(geodataframe, raster_path, force_crs_match=True):
    """Append categorical zonal statistics (counts by pixel type) as columns to an input geodataframe.

    geodataframe : geopandas.GeoDataFrame
        geodataframe that has overlay with the raster. If some polygon do not overlay the raster,
        consider a preprocessing step using the function subset_gdf_polygons_from_raster.
    raster_path : str
        path to the raster image.
    force_crs_match : bool, Default is True.
        Whether the Coordinate Reference System (CRS) of the polygon will be reprojected to
        the CRS of the raster file. It is recommended to let this argument as True.

    Notes
    -----
    The generated geodataframe will input the value 0 for each Type that is not present in the raster
    for each polygon.
    """

    _check_presence_of_crs(geodataframe)
    if force_crs_match:
        with rio.open(raster_path) as raster:
            geodataframe = geodataframe.to_crs(crs=raster.crs.data)
    else:
        warnings.warn(
            "The GeoDataFrame is not being reprojected. The clipping might be being performing on unmatching polygon to the raster."
        )

    zonal_gjson = rs.zonal_stats(
        geodataframe, raster_path, prefix="Type_", geojson_out=True, categorical=True
    )

    zonal_ppt_gdf = gpd.GeoDataFrame.from_features(zonal_gjson)

    return zonal_ppt_gdf


def extract_raster_features(
    gdf, raster_path, pixel_values=None, nodata=255, n_jobs=-1, collapse_values=False
):
    """Generate a geodataframe from raster data by polygonizing contiguous pixels with the same value using rasterio's features module.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        geodataframe  defining the area of interest. The input raster will be
        clipped to the extent of the geodataframe
    raster_path : str
        path to raster file, such as downloaded from <https://lcviewer.vito.be/download>
    pixel_values : list-like, optional
        subset of pixel values to extract, by default None. If None, this function
        may generate a very large geodataframe
    nodata : int, optional
        pixel value denoting "no data" in input raster
    n_jobs : int
        [Optional. Default=-1] Number of processes to run in parallel. If -1,
        this is set to the number of CPUs available
    collapse_values : bool, optional
        If True, multiple values passed to the pixel_values argument are treated
        as a single type. I.e. polygons will be generated from any contiguous collection
        of values from pixel_types, instead of unique polygons generated for each value
        This can dramatically reduce the complexity of the resulting geodataframe a
        fewer polygons are required to represent the study area.

    Returns
    -------
    geopandas.GeoDataFrame
        geodataframe whose rows are the zones extracted by the rasterio.features module.
        The geometry of each zone is the boundary of a contiguous group of pixels with
        the same value; the `value` column contains the pixel value of each zone.
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    with rio.open(raster_path) as src:
        raster_crs = src.crs.to_dict()
        gdf = gdf.to_crs(raster_crs)
        if GPD_10:
            geomask = [gdf.union_all().__geo_interface__]
        else:
            geomask = [gdf.unary_union.__geo_interface__]

        out_image, out_transform = mask(
            src, geomask, nodata=nodata, crop=True
        )  # clip to AoI using a vector layer

        if pixel_values:
            if collapse_values:
                out_image = np.where(
                    np.isin(out_image, pixel_values), pixel_values[0], out_image
                )  #  replace values to generate fewer polys
            pixel_values = np.isin(
                out_image, pixel_values
            )  # only include requested pixels

        shapes = list(
            features.shapes(out_image, mask=pixel_values, transform=out_transform)
        )  # convert regions to polygons
    res = list(zip(*shapes))
    geoms = pd.Series(res[0], name="geometry").astype(str)
    pieces = _chunk_dfs(geoms, n_jobs)
    geoms = pd.concat(
        Parallel(n_jobs=n_jobs)(delayed(_apply_parser)(i) for i in pieces)
    )
    geoms = gpd.GeoSeries(geoms).buffer(0)  # we sometimes get self-intersecting rings
    vals = pd.Series(res[1], name="value")
    gdf = gpd.GeoDataFrame(vals, geometry=geoms, crs=raster_crs)
    if collapse_values:
        gdf = gdf.drop(columns=["value"])  # values col is misleading in this case

    return gdf
