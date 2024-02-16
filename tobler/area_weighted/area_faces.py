import numpy as np
import pandas as pd
import geopandas as gpd
from .area_interpolate import _area_interpolate_binning as area_interpolate

__author__ = "Serge Rey <sjsrey@gmail.com>"


def area_faces(source_df, target_df,
               extensive_variables=[],
               intensive_variables=[]):
    """
    Interpolation of source_df values to faces formed by the union of
               the source and target dataframes.


    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        GeoDataFrame containing source values
    target_df : geopandas.GeoDataFrame
        GeoDataFrame containing target values
    extensive_variables : string or list-like
        column(s) in source_df dataframe for extensive variable(s) to
    be interpolated
    intensive_variables : string or list-like
        column(s) in source_df dataframe for intensive variable(s) to
    be interpolated

    Returns
    -------
    results : geopandas.GeoDataFrame
        GeoDataFrame with interpolated values as additional columns
    
    """

    union = gpd.overlay(source_df, target_df, how="union")
    results = area_interpolate(source_df, union, extensive_variables, intensive_variables)
    return results
