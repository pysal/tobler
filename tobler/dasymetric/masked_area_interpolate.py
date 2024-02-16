from warnings import warn

import geopandas as gpd

from ..area_weighted import area_interpolate
from .raster_tools import extract_raster_features


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
    codes=None,
):
    """Interpolate data between two polygonal datasets using an auxiliary raster to mask out uninhabited land.

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
        input might be [21,22,23,24], which match the "developed" land types in that dataset
    extensive_variables : list
        Columns of the input dataframe containing extensive variables to interpolate
    intensive_variables : list
        Columns of the input dataframe containing intensive variables to interpolate
    categorical_variables : list
        [Optional. Default=None] Columns in dataframes for categorical variables
    allocate_total : bool
        whether to allocate the total from the source geometries (the default is True).
    nodata : int
        value in raster that indicates null or missing values. Default is 255
    n_jobs : int
        [Optional. Default=-1] Number of processes to run in parallel to
        generate the area allocation. If -1, this is set to the number of CPUs
        available.


    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with geometries matching the target_df and extensive and intensive
        variables as the columns

    """
    if codes:
        warn(
            "The `codes` keyword is deprecated and will be removed shortly. Please use `pixel_values` instead"
        )
        pixel_values = codes
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
    )
    return interpolation
