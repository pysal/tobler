"""
Area Weighted Interpolation

"""

import numpy as np
import geopandas as gpd
from nlcd import *

def area_tables(source_df, target_df):
    """
    Construct area allocation and source-target correspondence tables
    Parameters
    ----------

    source_df: geopandas GeoDataFrame with geometry column of polygon type

    target_df: geopandas GeoDataFrame with geometry column of polygon type

    Returns
    -------
    tables: tuple (optional)
            two 2-D numpy arrays
            SU: area of intersection of source geometry i with union geometry j
            UT: binary mapping of union geometry j to target geometry t



    Notes
    -----
    The assumption is both dataframes have the same coordinate reference system.

    Union geometry is a geometry formed by the intersection of a source geometry and a target geometry

    SU Maps source geometry to union geometry, UT maps union geometry to target geometry



    """
    n_s = source_df.shape[0]
    n_t = target_df.shape[0]
    _left = np.arange(n_s)
    _right = np.arange(n_t)
    source_df.loc[:, '_left'] = _left  # create temporary index for union
    target_df.loc[:, '_right'] = _right # create temporary index for union
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
    source_df.drop(['_left'], axis=1, inplace=True)
    target_df.drop(['_right'], axis=1, inplace=True)
    return SU, UT


def area_interpolate(source_df, target_df, extensive_variables=[], intensive_variables=[], tables=None, allocate_total=True):
    """
    Area interpolation for extensive and intensive variables.

    Parameters
    ----------

    source_df: geopandas GeoDataFrame with geometry column of polygon type

    target_df: geopandas GeoDataFrame with geometry column of polygon type

    extensive_variables: list of columns in dataframes for extensive variables

    intensive_variables: list of columns in dataframes for intensive variables


    tables: tuple (optional)
            two 2-D numpy arrays
            SU: area of intersection of source geometry i with union geometry j
            UT: binary mapping of union geometry j to target geometry t


    allocate_total: boolean
                    True if total value of source area should be allocated.
                    False if denominator is area of i. Note that the two cases
                    would be identical when the area of the source polygon is
                    exhausted by intersections. See Notes for more details.

    Returns
    -------
    estimates: tuple (2)
              (extensive variable array, intensive variables array) 

    Notes
    -----
    The assumption is both dataframes have the same coordinate reference system.


    For an extensive variable, the estimate at target polygon j (default case) is:

    v_j = \sum_i v_i w_{i,j}

    w_{i,j} = a_{i,j} / \sum_k a_{i,k}


    If the area of the source polygon is not exhausted by intersections with
    target polygons and there is reason to not allocate the complete value of
    an extensive attribute, then setting allocate_total=False will use the
    following weights:


    v_j = \sum_i v_i w_{i,j}

    w_{i,j} = a_{i,j} / a_i

    where a_i is the total area of source polygon i.


    For an intensive variable, the estimate at target polygon j is:

    v_j = \sum_i v_i w_{i,j}

    w_{i,j} = a_{i,j} / \sum_k a_{k,j}


    """
    if tables is None:
        SU, UT  = area_tables(source_df, target_df)
    else:
        SU, UT = tables
    den = source_df['geometry'].area.values
    if allocate_total:
        den = SU.sum(axis=1)
    den = den + (den==0)
    weights = np.dot(np.diag(1/den), SU)

    extensive = [] 
    for variable in extensive_variables:
        att = source_df[variable]
        estimates = np.dot(np.diag(att), weights)
        estimates = np.dot(estimates, UT)
        estimates = estimates.sum(axis=0)
        extensive.append(estimates)
    extensive = np.array(extensive)

    ST = np.dot(SU, UT)
    area = ST.sum(axis=0)
    den = np.diag(1./ (area + (area == 0)))
    weights = np.dot(ST, den)
    intensive = []
    for variable in intensive_variables:
        att = source_df[variable]
        vals = att.values
        vals.shape = (len(vals), 1)
        est = (vals * weights).sum(axis=0)
        intensive.append(est)
    intensive = np.array(intensive)

    return (extensive, intensive)


def area_tables_nlcd(source_df, target_df, raster, codes = [21, 22, 23, 24], crs = {'init': 'epsg:4326'}):
    """
    Construct area allocation and source-target correspondence tables according to National Land Cover Data (NLCD) 'populated' areas
    Parameters
    ----------

    source_df : geopandas GeoDataFrame with geometry column of polygon type

    target_df : geopandas GeoDataFrame with geometry column of polygon type
    
    raster    : the associated NLCD raster (from rasterio.open)
    
    codes     : an integer list of codes values that should be considered as 'populated' from the National Land Cover Database (NLCD).
                The description of each code can be found here: https://www.mrlc.gov/sites/default/files/metadata/landcover.html
                The default is 21 (Developed, Open Space), 22 (Developed, Low Intensity), 23 (Developed, Medium Intensity) and 24 (Developed, High Intensity).
    
    crs       : the original crs of both GeoDataFrames given by a dict (this is gonna be projected later, as this must be the same as the raster).


    Returns
    -------
    tables: tuple (optional)
            two 2-D numpy arrays
            SU: area of intersection of source geometry i with union geometry j
            UT: binary mapping of union geometry j to target geometry t



    Notes
    -----
    The assumption is both dataframes have the same coordinate reference system given by crs parameter.

    Union geometry is a geometry formed by the intersection of a source geometry and a target geometry

    SU Maps source geometry to union geometry, UT maps union geometry to target geometry



    """
    
    source_df.crs = crs
    target_df.crs = crs
    
    n_s = source_df.shape[0]
    n_t = target_df.shape[0]
    _left = np.arange(n_s)
    _right = np.arange(n_t)
    source_df.loc[:, '_left'] = _left  # create temporary index for union
    target_df.loc[:, '_right'] = _right # create temporary index for union
    
    res_union_pre = gpd.overlay(source_df, target_df, how='union')
    
    # The 'append_profile_in_gdf' function is present in nlcd.py script
    res_union = append_profile_in_gdf(res_union_pre, raster = raster, polygon_crs = crs)
    
    str_codes     = [str(i) for i in codes]
    str_list      = ['Type_' + i for i in str_codes]
    
    # Extract list of code names that actually appear in the appended dataset
    str_list_ok = [col for col in res_union.columns if col in str_list]
    
    res_union['Populated_Pixels'] = res_union[str_list_ok].sum(axis = 1)
    
    n_u, _ = res_union.shape
    SU = np.zeros((n_s, n_u)) # holds area of intersection of source geom with union geom
    UT = np.zeros((n_u, n_t)) # binary table mapping union geom to target geom
    
    for index, row in res_union.iterrows():
        # only union polygons that intersect both a source and a target geometry matter 
        if not np.isnan(row['_left']) and not np.isnan(row['_right']):
            s_id = int(row['_left'])
            t_id = int(row['_right'])
            SU[s_id, index] = row['Populated_Pixels']
            UT[index, t_id] = 1
    source_df.drop(['_left'], axis=1, inplace=True)
    target_df.drop(['_right'], axis=1, inplace=True)
    return SU, UT