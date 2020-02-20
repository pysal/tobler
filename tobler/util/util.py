"""
Useful functions for spatial interpolation methods of tobler
"""

import numpy as np
import math


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
    if np.any(np.isnan(values)):
        wherenan = np.isnan(values)
        values[wherenan] = 0.0
        print("nan values in variable: {var}, replacing with 0.0".format(var=column))
    return values


def _check_presence_of_crs(geoinput):
    """check if there is crs in the polygon/geodataframe"""
    if geoinput.crs is None:
        raise KeyError(
            "The polygon/geodataframe does not have a Coordinate Reference System (CRS). This must be set before using this function."
        )


def project_gdf(gdf, to_crs=None, to_latlong=False):
    """Reproject gdf into the appropriate UTM zone.

    Project a GeoDataFrame to the UTM zone appropriate for its geometries'
    centroid.
    The simple calculation in this function works well for most latitudes, but
    won't work for some far northern locations like Svalbard and parts of far
    northern Norway.

    This function is lovingly modified from osmnx:
    https://github.com/gboeing/osmnx/

    Parameters
    ----------
    gdf : GeoDataFrame
        the gdf to be projected
    to_crs : dict
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
    if (gdf.crs is not None) and ("+proj=utm " in gdf.crs):
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
