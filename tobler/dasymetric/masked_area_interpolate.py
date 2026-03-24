from warnings import warn

import geopandas as gpd

from ..area_weighted import area_interpolate
from ..util import dot_density
from .raster_tools import extract_raster_features

__all__ = ["masked_area_interpolate", "masked_dot_density"]


def masked_area_interpolate(
    source_df,
    target_df,
    raster,
    pixel_values,
    extensive_variables=None,
    intensive_variables=None,
    categorical_variables=None,
    allocate_total=True,
    nodata=255,
    n_jobs=-1,
    fill_nan=0.0,
):
    """Interpolate data between two polygonal datasets using an
    auxiliary raster to mask out uninhabited land.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        source data to be converted to another geometric representation.
    target_df : geopandas.GeoDataFrame
        target geometries that will form the new representation of the input data
    raster : str
        path to raster file that contains ancillary data
    pixel_values : list of ints
        list of pixel values that should be considered part of the mask. For example if
        using data from NLCD Land Cover Database <https://www.mrlc.gov/data>, a common
        input might be [21,22,23,24], which match the "developed" land types in
        that dataset
    extensive_variables : list
        Columns of the input dataframe containing extensive variables to interpolate
    intensive_variables : list
        Columns of the input dataframe containing intensive variables to interpolate
    categorical_variables : list
        [Optional. Default=None] Columns in dataframes for categorical variables`
    allocate_total : bool
        whether to allocate the total from the source geometries (the default is True).
    nodata : int
        value in raster that indicates null or missing values. Default is 255
    n_jobs : int
        [Optional. Default=-1] Number of processes to run in parallel to
        generate the area allocation. If -1, this is set to the number of CPUs
        available.
    fill_nan : numeric, str, or None
        [Optional. Default=0.0] Value to replace NaN values in the source variables.
        If None, NaN values are not replaced and will propagate through the interpolation.
        If a string is passed, it should be one of 'mean', 'median', 'max', or 'min',
        and NaN values will be replaced with the corresponding aggregate value from the
        source variable.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with geometries matching the target_df and extensive and intensive
        variables as the columns
    """

    source_df = source_df.copy()
    assert not any(
        source_df.index.duplicated()
    ), "The index of the source_df cannot contain duplicates."

    #  create a vector mask from the raster data
    raster_mask = extract_raster_features(
        source_df, raster, pixel_values, nodata, n_jobs, collapse_values=True
    )
    #  create a column in the source_df to dissolve on
    idx_name = source_df.index.name if source_df.index.name else "idx"
    source_df[idx_name] = source_df.index

    #  clip source_df by its mask (overlay/dissolve is faster than gpd.clip here)
    source_df = gpd.overlay(
        source_df, raster_mask.to_crs(source_df.crs), how="intersection"
    ).dissolve(idx_name)

    #  continue with standard areal interpolation using the clipped source
    interpolation = area_interpolate(
        source_df,
        target_df.copy(),
        extensive_variables=extensive_variables,
        intensive_variables=intensive_variables,
        n_jobs=n_jobs,
        categorical_variables=categorical_variables,
        allocate_total=allocate_total,
        fill_nan=fill_nan
    )
    return interpolation


def masked_dot_density(
    source_df,
    raster,
    pixel_values,
    scale=1,
    method="uniform",
    columns=None,
    rng=None,
    method_kwargs=None,
    nodata=255,
    n_jobs=-1,
):
    """Simulate a point pattern process within each source polygon while using raster
    data to mask out uninhabited areas of the each geometry.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        source data to be converted to another geometric representation.
    raster : str
        path to raster file that contains ancillary data
    pixel_values : list of ints
        list of pixel values that should be considered part of the mask. For example if
        using data from NLCD Land Cover Database <https://www.mrlc.gov/data>, a common
        input might be [21,22,23,24], which match the "developed" land types in
        that dataset
    scale : int, optional
        scalar coefficient used to increase or decrease the number of simulated points in
        each geometry. For example a number less than 1 is used to create a proportional
        dot-density map; a stochastic realization of the population in each polygon would use
        1, resulting in the same number of points generated as the numeric value in the dataframe.
        By default 1
    method : str, optional
        name of the distribution used to simulate point locations. The default is  "uniform", in which
        every location within a polygon has an equal chance of being chosen. Alternatively, other
    columns : list-like, optional
        a list or array of columns in the dataframe holding the desired size of the set of points in each
        category. For example this would hold a set of mutually-exclusive racial groups, or employment
        industries
    rng : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        A random generator or seed to initialize the numpy BitGenerator. If None, then fresh,
        unpredictable entropy will be pulled from the OS.
    method_kwargs : dict, optional
        additional keyword arguments passed to the pointpats.random generator.
    nodata : int
        value in raster that indicates null or missing values. Default is 255
    n_jobs : int
        [Optional. Default=-1] Number of processes to run in parallel to
        generate the area allocation. If -1, this is set to the number of CPUs
        available.

    Returns
    -------
    GeoDataFrame
        a geodataframe with simulated points in the geometry column, with each row containing the index
        of the containing polygon, and the category to which the point belongs.
    """
    if columns is None:
        raise ValueError("must provide a set of categories to draw from")
    source_df = source_df.copy()
    assert not any(
        source_df.index.duplicated()
    ), "The index of the source_df cannot contain duplicates."

    #  create a vector mask from the raster data
    raster_mask = extract_raster_features(
        source_df, raster, pixel_values, nodata, n_jobs, collapse_values=True
    )
    #  create a column in the source_df to dissolve on
    idx_name = source_df.index.name if source_df.index.name else "idx"
    source_df[idx_name] = source_df.index

    #  clip source_df by its mask (overlay/dissolve is faster than gpd.clip here)
    source_df = gpd.overlay(
        source_df, raster_mask.to_crs(source_df.crs), how="intersection"
    ).dissolve(idx_name)

    gdf = dot_density(
        source_df,
        scale=scale,
        method=method,
        columns=columns,
        rng=rng,
        method_kwargs=method_kwargs,
    )
    return gdf
