"""Useful functions to support tobler's interpolation methods."""

from warnings import warn

import geopandas
import numpy as np
import pandas
import shapely
from packaging.version import Version
from shapely.geometry import Polygon

GPD_10 = Version(geopandas.__version__) >= Version("1.0.0dev")


def circumradius(resolution):
    """Find the circumradius of an h3 hexagon at given resolution.

     Parameters
    ----------
    resolution : int
        h3 grid resolution

    Returns
    -------
    circumradius : float
        circumradius in meters
    """
    try:
        import h3
    except ImportError:
        raise ImportError(
            "This function requires the `h3` library. "
            "You can install it with `conda install h3-py` or "
            "`pip install h3`"
        )

    return h3.edge_length(resolution, "m")


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

    if source.crs.is_geographic:
        if buffer:  # if CRS is geographic but user wants a buffer, we need to estimate
            warn(
                "The source geodataframe is stored in a geographic CRS. Falling back to estimated UTM zone "
                "to generate desired buffer. If this produces unexpected results, reproject the input data "
                "prior to using this function"
            )
            source = (
                source.to_crs(source.estimate_utm_crs())
                .buffer(circumradius(resolution))
                .to_crs(4326)
            )

    else:  # if CRS is projected, we need lat/long
        crs_units = source.crs.to_dict()["units"]
        if buffer:  #  we can only convert between units we know
            if not crs_units in ["m", "us-ft"]:
                raise ValueError(
                    f"The CRS of source geodataframe uses an unknown measurement unit: `{crs_units}`. "
                    "The `buffer` argument requires either a geographic CRS or a projected one measured "
                    "in meters or feet (U.S.)"
                )
            clipper = source.to_crs(4326)
            distance = circumradius(resolution)
            if crs_units == "ft-us":
                distance = distance * 3.281
            source = source.buffer(distance).to_crs(4326)
        else:
            source = source.to_crs(4326)

    if GPD_10:
        source_unary = shapely.force_2d(source.union_all())
    else:
        source_unary = shapely.force_2d(source.unary_union)

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

    hexs = geopandas.GeoDataFrame(hexids, geometry=polys.values, crs=4326).set_index(
        "hex_id"
    )

    return hexs
