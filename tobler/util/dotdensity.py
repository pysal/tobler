from warnings import warn

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from geopandas.tools._random import uniform
from pandas.api.types import is_list_like
from shapely.geometry import MultiPoint

__all__ = ["dot_density"]

_POLYGONAL = (3, 6)  # shapely type ids of Polygon and MultiPolygon


def _uniform_in_polygons(geoms, size, rng):
    """Draw ``size[i]`` points uniformly inside each polygon ``geoms[i]``.

    This is the rejection sampling geopandas uses: each candidate is drawn
    uniformly in its polygon's bounding box and kept if the polygon contains
    it. Rather than sampling one geometry at a time, every point is drawn in a
    single vectorized pass, and only the rejected candidates are redrawn.

    Returns the position of each point's polygon and the point coordinates.
    """
    which = np.repeat(np.arange(len(geoms)), size)
    bounds = shapely.bounds(geoms)[which]
    x = np.empty(len(which))
    y = np.empty(len(which))
    todo = np.arange(len(which))
    # prepared polygons answer containment far faster when they are large; undo
    # it afterward so the caller's geometries are left as they were
    fresh = geoms[~shapely.is_prepared(geoms)]
    shapely.prepare(fresh)
    try:
        while todo.size:
            x[todo] = rng.uniform(bounds[todo, 0], bounds[todo, 2])
            y[todo] = rng.uniform(bounds[todo, 1], bounds[todo, 3])
            inside = shapely.contains_xy(geoms[which[todo]], x[todo], y[todo])
            todo = todo[~inside]
    finally:
        shapely.destroy_prepared(fresh)
    return which, x, y


def _uniform_points(gdf, size, rng):
    """Sample ``size[i]`` points uniformly within each row of ``gdf``.

    Polygons are sampled with :func:`_uniform_in_polygons`; any other geometry
    type falls back to :meth:`geopandas.GeoSeries.sample_points`.
    """
    geoms = np.asarray(gdf.geometry.values)
    size = np.asarray(size, dtype=int)
    polygonal = np.isin(shapely.get_type_id(geoms), _POLYGONAL) & ~shapely.is_empty(
        geoms
    )
    fill = polygonal & (shapely.area(geoms) > 0)

    degenerate = polygonal & ~fill & (size > 0)
    if degenerate.any():
        warn(
            f"{degenerate.sum()} polygon(s) have zero area and receive no points.",
            UserWarning,
            stacklevel=4,
        )

    rows = np.flatnonzero(fill & (size > 0))
    which, x, y = _uniform_in_polygons(geoms[rows], size[rows], rng)
    pts = gpd.GeoSeries(
        gpd.points_from_xy(x, y),
        index=gdf.index[rows[which]],
        crs=gdf.crs,
        name="sampled_points",
    )

    other = np.flatnonzero(~polygonal & (size > 0))
    if other.size:
        extra = gdf.iloc[other].sample_points(size=size[other], rng=rng).explode()
        pts = pd.concat([pts, extra[~extra.is_empty]])
    return pts


def _sizes_by_group(values, groups, scale, rng):
    """Round each group's total once, then split it across the group's rows.

    The whole-number total of each group is divided among its rows by a
    multinomial draw with probabilities proportional to the rows' values, so
    every group receives exactly ``round(total * scale)`` points even when each
    row holds only a fraction of one.
    """
    values = np.nan_to_num(np.asarray(values, dtype=float))
    if (values < 0).any():
        raise ValueError("values must be non-negative to sample by group")
    sizes = np.zeros(len(values), dtype=int)
    indices = pd.DataFrame({"g": np.asarray(groups)}).groupby("g", dropna=False).indices
    for idx in indices.values():
        total = values[idx].sum()
        n = int(round(total * scale))
        if n > 0:
            sizes[idx] = rng.multinomial(n, values[idx] / total)
    return sizes


def _poly_to_dots(
    gdf,
    scale=1.0,
    method="uniform",
    category=None,
    rng=None,
    method_kwargs=None,
    by=None,
):
    """Simulate the points of one category in every row of ``gdf``.

    Uniform samples come from :func:`_uniform_points`. Other methods wrap the
    pointpats generators, but draw a single point from a uniform distribution
    when clustered point pattern data-generating processes would raise an error.
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
    if by is not None and by not in gdf.columns.tolist():
        raise ValueError(f"{by} not in columns of the passed geodataframe")
    # one generator for every row: geopandas re-seeds each geometry from `rng`,
    # so an integer seed would otherwise repeat the same draws in every row
    rng = np.random.default_rng(rng)
    # this could be a parallel apply, like with hex-generation
    if by is None:
        size = (gdf[category] * scale).round(0).astype(int).to_numpy()
    else:
        size = _sizes_by_group(gdf[category], gdf[by], scale, rng)
    if method == "uniform":
        pts = _uniform_points(gdf, size, rng).to_frame()
    else:
        sized = gdf.assign(**{category: size})
        pts = (
            sized.apply(
                lambda row: _draw_pointpats(
                    row, category, 1.0, method, rng, method_kwargs
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
    gdf,
    columns,
    scale=1.0,
    method="uniform",
    rng=None,
    method_kwargs=None,
    by=None,
):
    """Draw a sample of points inside each polygon/multipolygon row of a
    geodataframe, where sample size is one or more columns on the dataframe,
    optionally scaled by a constant. For example to create a proportional dot density
    map, pass a polygon geodataframe storing total population counts for
    mutually-exclusive groups.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        a geodataframe with columns of numeric data, a selection of which will be
        used to simulate a point-process within each geometry. The data in each column
        defines the number of points to simulate in each geometry
    columns : list-like
        a list or array of columns in the dataframe holding the desired size of the
        set of points in each category. For example this would hold a set of
        mutually-exclusive racial groups, or employment industries, etc.
        industries
    scale : float, optional
        scalar coefficient used to increase or decrease the number of simulated points
        in each geometry. For example a number less than 1 is used to create a
        proportional dot-density map; a stochastic realization of the population in
        each polygon would use 1, resulting in the same number of points generated as
        the numeric value in the dataframe.
        By default 1
    method : str, optional
        name of the distribution used to simulate point locations. The default is
        "uniform", in which every location within a polygon has an equal chance of
        being chosen. Alternatively, other methods are implemented in the
        `pointpats` package, including
        {'normal', 'cluster_normal', 'poisson', 'cluster_poisson'}
    rng : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        A random generator or seed to initialize the numpy BitGenerator. If None, then
        fresh, unpredictable entropy will be pulled from the OS.
    method_kwargs : dict, optional
        additional keyword arguments passed to the pointpats.random generator.
    by : str, optional
        column identifying groups of rows that share one whole-number count, such
        as the floors of the buildings in a census tract. If given, each group's
        total (times ``scale``) is rounded once and divided among the group's rows
        by a random draw in proportion to their values, so each group receives
        exactly ``round(total * scale)`` points. Without it, each row's value is
        rounded separately, which loses points when rows hold fractions of one.
        By default None.

    Returns
    -------
    geopandas.GeoDataFrame
        a geodataframe with simulated points in the geometry column, with each
        row/point holding the index of its containing polygon, and the column
        to which the point belongs.

    Raises
    ------
    ValueError
        raises an error if the specified categories are not columns in the geodataframe
    ValueError
        raises an error of the categories argument is not list-like
    """
    if not is_list_like(columns):
        raise ValueError("`categories` should be a list of columns")
    # one generator across categories, so each category gets fresh draws
    rng = np.random.default_rng(rng)
    pts = []
    for cat in columns:
        dots = _poly_to_dots(
            gdf,
            scale=scale,
            method=method,
            category=cat,
            rng=rng,
            method_kwargs=method_kwargs,
            by=by,
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
