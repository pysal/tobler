from warnings import warn

import geopandas as gpd
import pandas as pd
from geopandas.tools._random import uniform
from pandas.api.types import is_list_like
from shapely.geometry import MultiPoint


def _poly_to_dots(
    gdf, scale=1., method="uniform", category=None, rng=None, method_kwargs=None
):
    """this is just a function that wraps geopandas sample_points, but returns
    a single point from a uniform distribution when clustered pointpattern
    DGPs would raise an error
    """
    if method_kwargs is None:
        method_kwargs = {}
    if category is None:
        raise ValueError("must pass a category to expand")
    if category not in gdf.columns.tolist():
        raise ValueError(f"{category} not in columns of the passed geodataframe")
    if not pd.api.types.is_numeric_dtype(gdf[category]):
        raise ValueError(
            f"The column {category} must hold numeric data, but is "
            + f" type {gdf[category].dtype}."
        )
    # this could be a parallel apply, like with hex-generation
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
                lambda row: _draw_pointpats(
                    row, category, scale, method, rng, method_kwargs
                ),
                axis=1,
            )
            .explode()
            .to_frame()
        )
        pts = pts.rename(columns={0: "sampled_points"})
        pts = pts.set_geometry("sampled_points")
        pts = pts[~pts.is_empty]
        pts = pts.set_crs(gdf.crs)
        pts = pts.explode()

    if pts.index.name is None:
        pts.index.name = "poly_id"
    return pts.reset_index()


def dot_density(
    gdf, columns, scale=1.0, method="uniform", rng=None, method_kwargs=None
):
    """Draw a sample of points inside each polygon/multipolygon row of a
    geodataframe, where sample size is one or more columns on the dataframe,
    optionally scaled by a constant. For example to create a proportional dot density
    map, pass a polygon geodataframe storing total population counts for
    mutually-exclusive groups. 

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        a geodataframe with columns of numeric data, a selection of which will be used
        to simulate a point-process within each geometry. The data in each column defines
        the number of points to simulate in each geometry
    columns : list-like,
        a list or array of columns in the dataframe holding the desired size of the set
        of points in each category. For example this would hold a set of
        mutually-exclusive racial groups, or employment industries, etc.
        industries
    scale : float, optional
        scalar coefficient used to increase or decrease the number of simulated points
        in each geometry. For example a number less than 1 is used to create a
        proportional dot-density map; a stochastic realization of the population in each
        polygon would use 1, resulting in the same number of points generated as the
        numeric value in the dataframe.
        By default 1
    method : str, optional
        name of the distribution used to simulate point locations. The default is
        "uniform", in which every location within a polygon has an equal chance of being
        chosen. Alternatively, other methods are implemented in the `pointpats` package,
        including {'normal', 'cluster_normal', 'poisson', 'cluster_poisson'}
    rng : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        A random generator or seed to initialize the numpy BitGenerator. If None, then
        fresh, unpredictable entropy will be pulled from the OS.
    method_kwargs : dict, optional
        additional keyword arguments passed to the pointpats.random generator.

    Returns
    -------
    GeoDataFrame
        a geodataframe with simulated points in the geometry column, with each row/point
        holding the index of its containing polygon, and the column to which the
        point belongs.

    Raises
    ------
    ValueError
        raises an error if the specified categories are not columns in the geodataframe
    ValueError
        raises an error of the categories argument is not list-like
    """
    if not is_list_like(columns):
        raise ValueError("`categories` should be a list of columns")
    pts = []
    for cat in columns:
        dots = _poly_to_dots(
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


def _draw_pointpats(row, column, scale, method, rng, method_kwargs):
    try:
        import pointpats as pps
    except ImportError as e:
        raise ImportError(
            "you must have `pointpats` installed to draw from other distributions"
        ) from e
    sample_function = getattr(pps.random, method)
    pts = []
    val = int(round(row[column] * scale, 0))
    if not (
        row.geometry.is_empty
        or row["geometry"] is None
        or "Polygon" not in row["geometry"].geom_type
        or val <= 1
    ):
        pts.append(
            gpd.points_from_xy(
                *sample_function(row.geometry, size=val, **method_kwargs).T
            ).union_all()
        )
    elif val == 1:
        warn("drawing size=1, resorting to uniform for single draw", stacklevel=2)
        pts.append(uniform(row["geometry"], 1, rng))
    else:
        pts.append(MultiPoint())
    return pts
