"""
Area Weighted Interpolation

"""

import numpy as np
import geopandas as gpd

def area_tables(source_df, target_df):
    """
    Construct area allocation and source-target correspondence tables
    Parameters
    ----------

    source_df: geopandas GeoDataFrame with geometry column of polygon type

    source_df: geopandas GeoDataFrame with geometry column of polygon type

    Returns
    -------
    tables: tuple (optional)
            two 2-D numpy arrays
            SU: area of intersection of source geometry i with union geometry j
            UT: binary mapping of union geometry i to target geometry j



    Notes
    -----
    The assumption is both dataframes have the same coordinate reference system.

    SU Maps source geometry to union geometry, UT maps union geometry to target geometry



    """
    n_s = source_df.shape[0]
    n_t = target_df.shape[0]
    _left = np.arange(n_s)
    _right = np.arange(n_t)
    source_df['_left'] = _left  # create temporary index for union
    target_df['_right'] = _right # create temporary index for union
    res_union = gpd.overlay(source_df, target_df, how='union')
    n_u, _ = res_union.shape
    SU = np.zeros((n_s, n_u)) # holds area of intersection of source geom with union geom
    UT = np.zeros((n_u, n_t)) # binary table mapping union geom to target geom
    for index, row in res_union.iterrows():
        # only union polygons that intersect both a source and a target geometry matter 
        if not np.isnan(row['_left']) and not np.isnan(row['_right']):
            s_id = int(row['_left'])
            t_id = int(row['_right'])
            SU[s_id, index] = row['geometry'].area
            UT[index, t_id] = 1

    return SU, UT


def area_extensive(source_df, target_df, att_name, tables=None):
    """
    Interpolate extensive attribute values from source features to target features

    Parameters
    ----------

    source_df: geopandas GeoDataFrame with geometry column of polygon type

    source_df: geopandas GeoDataFrame with geometry column of polygon type

    att_name: string
              column name in source_df to interpolate over target_df 


    tables: tuple (optional)
            two 2-D numpy arrays
            SU: area of intersection of source geometry i with union geometry j
            UT: binary mapping of union geometry i to target geometry j


    Returns
    -------
    estimates: array
              values of attribute in target polygons

    Notes
    -----
    The assumption is both dataframes have the same coordinate reference system.


    Estimate at target polygon j:

    v_j = \sum_i v_i w_{i,j}

    w_{i,j} = a_{i,j} / \sum_k a_{i,k}





    """
    att = source_df[att_name]
    if tables is None:
        SU, UT  = area_tables(source_df, target_df)
    else:
        SU, UT = tables
    den = source_df['geometry'].area.values
    den = den + (den==0)
    weights = np.dot(np.diag(1/den), SU)
    estimates = np.dot(np.diag(att), weights)
    estimates = np.dot(estimates, UT)
    return estimates.sum(axis=0)


def area_intensive(source_df, target_df, att_name, tables=None):
    """
    Interpolate intensive attribute values from source features to target features

    Parameters
    ----------

    source_df: geopandas GeoDataFrame with geometry column of polygon type

    source_df: geopandas GeoDataFrame with geometry column of polygon type

    att_name: string
              column name in source_df to interpolate over target_df 


    table: array (optional)
           area mapping reporting intersection of target polygon i with source polygon j for all polygons i in source_df and j in target_df

    Returns
    -------
    estimates: array
              values of attribute in target polygons


    Notes
    -----
    The assumption is both dataframes have the same coordinate reference system.

    Estimate at target polygon j:

    v_j = \sum_i v_i w_{i,j}

    w_{i,j} = a_{i,j} / \sum_k a_{k,j}




    """
    att = source_df[att_name]
    if tables is None:
        SU, UT = area_tables(source_df, target_df)
    else:
        SU, UT = tables
    area = source_df['geometry'].area.values
    ST = np.dot(SU, UT)
    area = ST.sum(axis=0)
    den = np.diag(1./ (area + (area == 0)))
    weights = np.dot(ST, den)
    vals = att.values
    vals.shape = (len(vals), 1)
    return  (vals * weights).sum(axis=0)

