import pandas as pd
import geopandas as gpd
import numpy as np
from pandas.api.types import is_list_like
from shapely.geometry import MultiPoint


def poly_to_dots(
    gdf, scale=1, method="uniform", category=None, rng=None, method_kwargs=None
):
    if method_kwargs is None:
        method_kwargs = {}
    if category is None:
        raise ValueError("must pass a category to expand")
    if category not in gdf.columns.tolist():
        raise ValueError(f"{category} not in columns of the passed geodataframe")
    size = (gdf[category] * scale).round(0).astype(int).to_numpy()
    if method == "uniform":
        pts = gpd.GeoSeries(
            gdf.sample_points(
                size=size, method=method, rng=rng, **method_kwargs
            ).explode()
        ).to_frame()
    else:
        pts = (
            gdf.apply(
                lambda row: _draw_pointpats(row, category, method, method_kwargs),
                axis=1,
            )
            .explode()
            .to_frame()
        )
        pts = pts.rename(columns={0: "sampled_points"})
        pts = pts.set_geometry("sampled_points")
        pts = pts.set_crs(gdf.crs)

    if pts.index.name is None:
        pts.index.name = "poly_id"
    return pts.reset_index()


def poly_to_multidots(
    gdf, scale=1, method="uniform", categories=None, rng=None, method_kwargs=None
):
    """Simulate points-in-polygon for multiple categories.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        a geodataframe with columns of numeric data, a selection of which will be used
        to simulate a point-process within each geometry. The data in each column defines
        the number of points to simulate in each geometry
    scale : int, optional
        scalar coefficient used to increase or decrease the number of simulated points in
        each geometry. For example a number less than 1 is used to create a proportional
        dot-density map; a stochastic realization of the population in each polygon would use
        1, resulting in the same number of points generated as the numeric value in the dataframe.
        By default 1
    method : str, optional
        name of the distribution used to simulate point locations. The default is  "uniform", in which
        every location within a polygon has an equal chance of being chosen. Alternatively, other
    categories : list-like, optional
        a list or array of columns in the dataframe holding the desired size of the set of points in each
        category. For example this would hold a set of mutually-exclusive racial groups, or employment
        industries
    rng : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        A random generator or seed to initialize the numpy BitGenerator. If None, then fresh,
        unpredictable entropy will be pulled from the OS.
    method_kwargs : dict, optional
        additional keyword arguments passed to the pointpats.random generator.

    Returns
    -------
    GeoDataFrame
        a geodataframe with simulated points in the geometry column, with each row containing the index
        of the containing polygon, and the category to which the point belongs.

    Raises
    ------
    ValueError
        raises an error if the specified categories are not columns in the geodataframe
    ValueError
        raises an error of the categories argument is not list-like
    """
    if not is_list_like(categories):
        raise ValueError("`categories` should be a list of columns")
    if not set(categories).issubset(set(gdf.columns)):
        raise ValueError(f"{categories} not in gdf columns")
    pts = []
    for cat in categories:
        dots = poly_to_dots(
            gdf,
            scale=scale,
            method=method,
            category=cat,
            rng=rng,
            method_kwargs=method_kwargs,
        )
        dots["category"] = cat
        pts.append(dots)
    pts = gpd.GeoDataFrame(pd.concat(pts))
    pts["category"] = pts["category"].astype("category")

    return pts


def _draw_pointpats(row, column, method, method_kwargs):
    try:
        import pointpats as pps
    except:
        raise ImportError(
            "you must have `pointpats` installed to draw from other distributions"
        )
    sample_function = getattr(pps.random, method)
    pts = (
        gpd.points_from_xy(
            *sample_function(row.geometry, size=int(row[column]), **method_kwargs).T
        ).union_all()
        if not (
            row.geometry.is_empty
            or row["geometry"] is None
            or "Polygon" not in row["geometry"].geom_type
            or row[column] < 1
        )
        else MultiPoint()
    )
    return pts
