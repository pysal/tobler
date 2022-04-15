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


# from https://h3geo.org/docs/core-library/restable/
hexvals = pandas.DataFrame(data=np.array([
    [0, 4250546.8477000, 1107.712591000, 122],
    [1, 607220.9782429, 418.676005500, 842],
    [2, 86745.8540347, 158.244655800, 5882],
    [3, 12392.2648621, 59.810857940, 41162],
    [4, 1770.3235517, 22.606379400, 288122],
    [5, 252.9033645, 8.544408276, 2016842],
    [6, 36.1290521, 3.229482772, 14117882],
    [7, 5.1612932, 1.220629759, 98825162],
    [8, 0.7373276, 0.461354684, 691776122],
    [9, 0.1053325, 0.174375668, 4842432842],
    [10, 0.0150475, 0.065907807, 33897029882],
    [11, 0.0021496, 0.024910561, 237279209162],
    [12, 0.0003071, 0.009415526, 1660954464122],
    [13, 0.0000439, 0.003559893, 11626681248842],
    [14, 0.0000063, 0.001348575, 81386768741882],
    [15, 0.0000009, 0.000509713, 569707381193162]
]), columns=["resolution", "area",
             'edge_length', 'number'])


def circumradius(resolution, hexvals=hexvals):
    """Find the circumradius of an h3 hexagon at given resolution.

     Parameters
    ----------
    resolution : int
        h3 grid resolution
    
    hexvals : DataFrame
        statistics on h3py hexagons at different resolutions

    Returns
    -------
    circumradius : float
        circumradius in meters
    """
    cr = hexvals[hexvals.resolution == resolution].edge_length.values.item()
    return 1000 * cr


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



def h3fy(source, resolution=6, clip=False, buffer=False, return_geoms=True):
    """Generate a hexgrid geodataframe that covers the face of a source geodataframe.

    Parameters
    ----------
    source : geopandas.GeoDataFrame
        GeoDataFrame to transform into a hexagonal grid
    resolution : int, optional (default is 6)
        resolution of output h3 hexgrid.
        See <https://h3geo.org/docs/core-library/restable> for more information
    clip : bool, optional (default is False)
        if True, hexagons are clipped by the boundary of the source gdf. Otherwise,
        heaxgons along the boundary will be left intact.
    buffer : bool, optional (default is False)
        if True, force hexagons to completely fill the interior of the source area.
        if False, (h3 default) may result in empty areas within the source area.
    return_geoms: bool, optional (default is True)
        whether to generate hexagon geometries as a geodataframe or simply return
        hex ids as a pandas.Series

    Returns
    -------
    pandas.Series or geopandas.GeoDataFrame
        if `return_geoms` is True, a geopandas.GeoDataFrame whose rows comprise a hexagonal h3 grid (indexed on h3 hex id).
        if `return_geoms` is False, a pandas.Series of h3 hexagon ids
    """
    # h3 hexes only work on polygons, not multipolygons
    if source.crs is None:
        raise ValueError(
            "source geodataframe must have a valid CRS set before using this function"
        )

    orig_crs = source.crs
    clipper = source

    if not source.crs.is_geographic:
        if buffer:
            clipper = source.to_crs(4326)
            distance = circumradius(resolution)
            source = source.buffer(distance).to_crs(4326)
        else:
            source = source.to_crs(4326)

    source_unary = source.unary_union


    if type(source_unary) == Polygon:
        hexagons = _to_hex(
            source_unary, resolution=resolution, return_geoms=return_geoms
        )
    else:
        output = []
        for geom in source_unary.geoms:
            hexes = _to_hex(geom, resolution=resolution, return_geoms=return_geoms)
            output.append(hexes)
            hexagons = pandas.concat(output)

    if return_geoms and clip:
        hexagons = geopandas.clip(hexagons, clipper)

    if return_geoms and not hexagons.crs.equals(orig_crs):
        hexagons = hexagons.to_crs(orig_crs)

    return hexagons


def _to_hex(source, resolution=6, return_geoms=True, buffer=True):
    """Generate a hexgrid geodataframe that covers the face of a source geometry.

    Parameters
    ----------
    source : geometry
        geometry to transform into a hexagonal grid (needs to support __geo_interface__)
    resolution : int, optional (default is 6)
        resolution of output h3 hexgrid.
        See <https://h3geo.org/docs/core-library/restable> for more information
    return_geoms: bool, optional (default is True)
        whether to generate hexagon geometries as a geodataframe or simply return
        hex ids as a pandas.Series

    Returns
    -------
    pandas.Series or geopandas.GeoDataFrame
        if `return_geoms` is True, a geopandas.GeoDataFrame whose rows comprise a hexagonal h3 grid (indexed on h3 hex id).
        if `return_geoms` is False, a pandas.Series of h3 hexagon ids
    """
    try:
        import h3
    except ImportError:
        raise ImportError(
            "This function requires the `h3` library. "
            "You can install it with `conda install h3-py` or "
            "`pip install h3`"
        )



    hexids = pandas.Series(
        list(
            h3.polyfill(
                source.__geo_interface__,
                resolution,
                geo_json_conformant=True,
            )
        ),
        name="hex_id",
    )
        
    if not return_geoms:
        return hexids


    polys = hexids.apply(
        lambda hex_id: Polygon(h3.h3_to_geo_boundary(hex_id, geo_json=True)),
    )

    hexs = geopandas.GeoDataFrame(hexids, geometry=polys, crs=4326).set_index("hex_id")

    return hexs
