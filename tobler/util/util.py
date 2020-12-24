"""
Useful functions for spatial interpolation methods of tobler
"""

import math
from warnings import warn

import geopandas
import numpy as np
import pandas
from pyproj import CRS
from shapely.geometry import Polygon


def _check_crs(source_df, target_df):
    """check if crs is identical"""
    if not (source_df.crs == target_df.crs):
        print("Source and target dataframes have different crs. Please correct.")
        return False
    return True


def _nan_check(df, column):
    """Check if variable has nan values.

    Warn and replace nan with 0.0.
    """
    values = df[column].values
    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        wherenan = np.isnan(values)
        values[wherenan] = 0.0
        warn(f"nan values in variable: {column}, replacing with 0")
    return values


def _inf_check(df, column):
    """Check if variable has nan values.

    Warn and replace inf with 0.0.
    """
    values = df[column].values
    if np.any(np.isinf(values)):
        wherenan = np.isinf(values)
        values[wherenan] = 0.0
        warn(f"inf values in variable: {column}, replacing with 0")
    return values


def _check_presence_of_crs(geoinput):
    """check if there is crs in the polygon/geodataframe"""
    if geoinput.crs is None:
        raise KeyError("Geodataframe must have a CRS set before using this function.")


def is_crs_utm(crs):
    """
    Determine if a CRS is a UTM CRS
    Parameters
    ----------
    crs : dict or string or pyproj.CRS
        a coordinate reference system
    Returns
    -------
    bool
        True if crs is UTM, False otherwise
    """
    if not crs:
        return False
    crs_obj = CRS.from_user_input(crs)
    if crs_obj.coordinate_operation and crs_obj.coordinate_operation.name.upper().startswith(
        "UTM"
    ):
        return True
    return False


def project_gdf(gdf, to_crs=None, to_latlong=False):
    """
    lovingly copied from OSMNX <https://github.com/gboeing/osmnx/blob/master/osmnx/projection.py>

    Project a GeoDataFrame to the UTM zone appropriate for its geometries'
    centroid.
    The simple calculation in this function works well for most latitudes, but
    won't work for some far northern locations like Svalbard and parts of far
    northern Norway.

    Parameters
    ----------
    gdf : GeoDataFrame
        the gdf to be projected
    to_crs : dict or string or pyproj.CRS
        if not None, just project to this CRS instead of to UTM
    to_latlong : bool
        if True, projects to latlong instead of to UTM

    Returns
    -------
    GeoDataFrame
    """
    assert len(gdf) > 0, "You cannot project an empty GeoDataFrame."

    # else, project the gdf to UTM
    # if GeoDataFrame is already in UTM, just return it
    if is_crs_utm(gdf.crs):
        return gdf

    # calculate the centroid of the union of all the geometries in the
    # GeoDataFrame
    avg_longitude = gdf["geometry"].unary_union.centroid.x

    # calculate the UTM zone from this avg longitude and define the UTM
    # CRS to project
    utm_zone = int(math.floor((avg_longitude + 180) / 6.0) + 1)
    utm_crs = "+proj=utm +zone={} +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(
        utm_zone
    )

    # project the GeoDataFrame to the UTM CRS
    projected_gdf = gdf.to_crs(utm_crs)

    return projected_gdf


def hexify(source, resolution=6, clip=False):
    """Generate a hexgrid geodataframe that covers the face of a source geodataframe.

    Parameters
    ----------
    source : geopandas.GeoDataFrame
        GeoDataFrame to transform into a hexagonal grid
    resolution : int, optional
        resolution of output h3 hexgrid. 
        See <https://h3geo.org/docs/core-library/restable> for more information
    clip : bool, optional
        if True, hexagons are clipped to the precise boundary of the source gdf. Otherwise,
        heaxgons along the boundary will be left intact.

    Returns
    -------
    geopandas.GeoDataFrame
        a GeoDataFrame whose rows comprise a hexagonal h3 grid. 
    """
    try:
        from h3 import h3
    except ImportError:
        raise ImportError(
            "This function requires the `h3` library. "
            "You can install it with `conda install h3` or "
            "`pip install h3`"
        )
    orig_crs = source.crs.to_string()

    if not source.crs.name == "WGS 84":
        source = source.copy().to_crs(4326)

    hexids = pandas.Series(
        list(
            h3.polyfill(
                source.unary_union.__geo_interface__,
                resolution,
                geo_json_conformant=True,
            )
        ),
        name="hex_id",
    )

    polys = hexids.apply(
        lambda hex_id: Polygon(h3.h3_to_geo_boundary(hex_id, geo_json=True)),
    )

    hexs = geopandas.GeoDataFrame(hexids, geometry=polys, crs=source.crs).set_index(
        "hex_id"
    )

    if clip:
        hexs = geopandas.clip(hexs, source)

    if source.crs.to_string() != orig_crs:
        hexs = hexs.to_crs(orig_crs)

    return hexs