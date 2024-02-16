import numpy as np
import pandas as pd
import geopandas as gpd
from .area_interpolate import _area_interpolate_binning as area_interpolate

__author__ = "Serge Rey <sjsrey@gmail.com>"


def area_buffer(source_df, buffer_df, in_place=False):
    """
    Classify spatial relationship of source_df geometries relative to buffer_df geometries

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        GeoDataFrame containing source values
    buffer_df : geopandas.GeoDataFrame
        GeoDataFrame containing buffer geometries
    in_place : boolean
        If True, the source_df is modified in place, otherwise a copy
    is returned (default)

    Returns
    -------
    source_df : geopandas.GeoDataFrame
        GeoDataFrame with additional column `right_relation` that
        takes three possible values ['within', 'partial', 'disjoint']
        specifying the spatial predicate of source to buffer
        geometries.

    """
    within = buffer_df.sindex.query(source_df.geometry, predicate="within")[0]
    intersects = buffer_df.sindex.query(source_df.geometry, predicate="intersects")[0]
    partial = [i for i in intersects if i not in within]
    if not in_place:
        source_df = source_df.copy()
    source_df['right_relation'] = 'disjoint'
    source_df.loc[partial, 'right_relation'] = 'partial'
    source_df.loc[within, 'right_relation'] = 'within'
    return source_df
