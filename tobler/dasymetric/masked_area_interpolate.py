from ..area_weighted import _slow_area_interpolate, area_tables_raster

from ..area_weighted.vectorized_raster_interpolation import *


def masked_area_interpolate(
    source_df,
    target_df,
    raster_path="nlcd_2011",
    codes=None,
    force_crs_match=True,
    extensive_variables=None,
    intensive_variables=None,
    allocate_total=True,
    tables=None,
):
    """Short summary.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        source data to be converted to another geometric representation.
    target_df : geopandas.GeoDataFrame
        target geometries that will form the new representation of the input data
    raster_path : str
        path to raster file that contains ancillary data. 
        alternatively a user can pass `ncld_2001` or `nlcd_2011` to use built-in data from the
        National Land Cover Database
    codes : list of ints
        list of pixel values that should be considered part of the mask (the default is None).
        If no codes are passed, this defaults to  [21, 22, 23, 24] which are the developed land use
        codes from the NLCD data
    force_crs_match : bool
        whether to force the input and target data to share the same CRS (the default is True).
    extensive_variables : list
        Columns of the input dataframe containing extensive variables to interpolate
    intensive_variables : type
        Columns of the input dataframe containing intensive variables to interpolate
    allocate_total : bool
        whether  (the default is True).
    tables : type
        Description of parameter `tables` (the default is None).

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with geometries matching the target_df and extensive and intensive 
        variables as the columns

    """

    if not tables:
        tables = area_tables_raster(
            source_df,
            target_df.copy(),
            raster_path=raster_path,
            codes=codes,
            force_crs_match=force_crs_match,
        )
    if not codes:
        codes = [21, 22, 23, 24]

    # In area_interpolate, the resulting variable has same length as target_df
    interpolation = _slow_area_interpolate(
        source_df,
        target_df.copy(),
        extensive_variables=extensive_variables,
        intensive_variables=intensive_variables,
        allocate_total=allocate_total,
        tables=tables,
    )

    return interpolation
