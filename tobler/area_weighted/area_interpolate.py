"""
Area Weighted Interpolation

"""

import numpy as np
import geopandas as gpd
from ._vectorized_raster_interpolation import _fast_append_profile_in_gdf
import warnings
from scipy.sparse import dok_matrix, diags, coo_matrix
import pandas as pd

from tobler.util.util import _check_crs, _nan_check, _inf_check, _check_presence_of_crs


def _area_tables_binning(source_df, target_df, spatial_index):
    """Construct area allocation and source-target correspondence tables using a spatial indexing approach
    ...

    NOTE: this currently relies on pygeos.STRTree

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        GeoDataFrame containing input data and polygons
    target_df : geopandas.GeoDataFramee
        GeoDataFrame defining the output geometries
    spatial_index : str
        Spatial index to use to build the allocation of area from source to
        target tables. It currently support the following values:
            - "source": build the spatial index on `source_df`
            - "target": build the spatial index on `target_df`
            - "auto": attempts to guess the most efficient alternative.
              Currently, this option uses the largest table to build the
              index, and performs a `bulk_query` on the shorter table.

    Returns
    -------
    tables : scipy.sparse.dok_matrix

    """
    if _check_crs(source_df, target_df):
        pass
    else:
        return None

    df1 = source_df.copy()
    df2 = target_df.copy()

    # it is generally more performant to use the longer df as spatial index
    if spatial_index == "auto":
        if df1.shape[0] > df2.shape[0]:
            spatial_index = "source"
        else:
            spatial_index = "target"

    if spatial_index == "source":
        ids_tgt, ids_src = df1.sindex.query_bulk(df2.geometry, predicate="intersects")
    elif spatial_index == "target":
        ids_src, ids_tgt = df2.sindex.query_bulk(df1.geometry, predicate="intersects")
    else:
        raise ValueError(
            f"'{spatial_index}' is not a valid option. Use 'auto', 'source' or 'target'."
        )

    areas = df1.geometry.values[ids_src].intersection(df2.geometry.values[ids_tgt]).area

    table = coo_matrix(
        (areas, (ids_src, ids_tgt),),
        shape=(df1.shape[0], df2.shape[0]),
        dtype=np.float32,
    )

    table = table.todok()

    return table


def _area_tables(source_df, target_df):
    """
    Construct area allocation and source-target correspondence tables.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
    target_df : geopandas.GeoDataFrame

    Returns
    -------
    tables : tuple (optional)
            two 2-D numpy arrays
            SU: area of intersection of source geometry i with union geometry j
            UT: binary mapping of union geometry j to target geometry t



    Notes
    -----
    The assumption is both dataframes have the same coordinate reference system.

    Union geometry is a geometry formed by the intersection of a source geometry and a target geometry

    SU Maps source geometry to union geometry, UT maps union geometry to target geometry

    """
    if _check_crs(source_df, target_df):
        pass
    else:
        return None
    source_df = source_df.copy()
    source_df = source_df.copy()

    n_s = source_df.shape[0]
    n_t = target_df.shape[0]
    _left = np.arange(n_s)
    _right = np.arange(n_t)
    source_df.loc[:, "_left"] = _left  # create temporary index for union
    target_df.loc[:, "_right"] = _right  # create temporary index for union
    res_union = gpd.overlay(source_df, target_df, how="union")
    n_u, _ = res_union.shape
    SU = np.zeros(
        (n_s, n_u)
    )  # holds area of intersection of source geom with union geom
    UT = np.zeros((n_u, n_t))  # binary table mapping union geom to target geom
    for index, row in res_union.iterrows():
        # only union polygons that intersect both a source and a target geometry matter
        if not np.isnan(row["_left"]) and not np.isnan(row["_right"]):
            s_id = int(row["_left"])
            t_id = int(row["_right"])
            SU[s_id, index] = row[row.geometry.name].area
            UT[index, t_id] = 1
    source_df.drop(["_left"], axis=1, inplace=True)
    target_df.drop(["_right"], axis=1, inplace=True)
    return SU, UT


def _area_interpolate_binning(
    source_df,
    target_df,
    extensive_variables=None,
    intensive_variables=None,
    table=None,
    allocate_total=True,
    spatial_index="auto",
):
    """
    Area interpolation for extensive and intensive variables.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
    target_df : geopandas.GeoDataFrame
    extensive_variables : list
        [Optional. Default=None] Columns in dataframes for extensive variables
    intensive_variables : list
        [Optional. Default=None] Columns in dataframes for intensive variables
    table : scipy.sparse.dok_matrix
        [Optional. Default=None] Area allocation source-target correspondence
        table. If not provided, it will be built from `source_df` and
        `target_df` using `tobler.area_interpolate._area_tables_binning`
    allocate_total : boolean
        [Optional. Default=True] True if total value of source area should be
        allocated. False if denominator is area of i. Note that the two cases
        would be identical when the area of the source polygon is exhausted by
        intersections. See Notes for more details.
    spatial_index : str
        [Optional. Default="auto"] Spatial index to use to build the
        allocation of area from source to target tables. It currently support
        the following values:
            - "source": build the spatial index on `source_df`
            - "target": build the spatial index on `target_df`
            - "auto": attempts to guess the most efficient alternative.
              Currently, this option uses the largest table to build the
              index, and performs a `bulk_query` on the shorter table.

    Returns
    -------
    estimates : geopandas.GeoDataFrame
         new geodaraframe with interpolated variables as columns and target_df geometry
         as output geometry

    Notes
    -----
    The assumption is both dataframes have the same coordinate reference system.
    For an extensive variable, the estimate at target polygon j (default case) is:

    .. math::
     v_j = \\sum_i v_i w_{i,j}

     w_{i,j} = a_{i,j} / \\sum_k a_{i,k}

    If the area of the source polygon is not exhausted by intersections with
    target polygons and there is reason to not allocate the complete value of
    an extensive attribute, then setting allocate_total=False will use the
    following weights:

    .. math::
     v_j = \\sum_i v_i w_{i,j}

     w_{i,j} = a_{i,j} / a_i

    where a_i is the total area of source polygon i.
    For an intensive variable, the estimate at target polygon j is:

    .. math::
     v_j = \\sum_i v_i w_{i,j}

     w_{i,j} = a_{i,j} / \\sum_k a_{k,j}
    """
    source_df = source_df.copy()
    target_df = target_df.copy()

    if _check_crs(source_df, target_df):
        pass
    else:
        return None

    if table is None:
        table = _area_tables_binning(source_df, target_df, spatial_index)

    den = source_df[source_df.geometry.name].area.values
    if allocate_total:
        den = np.asarray(table.sum(axis=1))
    den = den + (den == 0)
    den = 1.0 / den
    n = den.shape[0]
    den = den.reshape((n,))
    den = diags([den], [0])
    weights = den.dot(table)  # row standardize table

    dfs = []
    extensive = []
    if extensive_variables:
        for variable in extensive_variables:
            vals = _nan_check(source_df, variable)
            vals = _inf_check(source_df, variable)
            estimates = diags([vals], [0]).dot(weights)
            estimates = estimates.sum(axis=0)
            extensive.append(estimates.tolist()[0])

        extensive = np.asarray(extensive)
        extensive = np.array(extensive)
        extensive = pd.DataFrame(extensive.T, columns=extensive_variables)

    area = np.asarray(table.sum(axis=0))
    den = 1.0 / (area + (area == 0))
    n, k = den.shape
    den = den.reshape((k,))
    den = diags([den], [0])
    weights = table.dot(den)

    intensive = []
    if intensive_variables:
        for variable in intensive_variables:
            vals = _nan_check(source_df, variable)
            vals = _inf_check(source_df, variable)
            n = vals.shape[0]
            vals = vals.reshape((n,))
            estimates = diags([vals], [0])
            estimates = estimates.dot(weights).sum(axis=0)
            intensive.append(estimates.tolist()[0])

        intensive = np.asarray(intensive)
        intensive = pd.DataFrame(intensive.T, columns=intensive_variables)

    if extensive_variables:
        dfs.append(extensive)
    if intensive_variables:
        dfs.append(intensive)

    df = pd.concat(dfs, axis=1)
    df["geometry"] = target_df[target_df.geometry.name].reset_index(drop=True)
    df = gpd.GeoDataFrame(df.replace(np.inf, np.nan))
    return df


def _area_interpolate(
    source_df,
    target_df,
    extensive_variables=None,
    intensive_variables=None,
    tables=None,
    allocate_total=True,
):
    """
    Area interpolation for extensive and intensive variables.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame (required)
        geodataframe with polygon geometries
    target_df : geopandas.GeoDataFrame (required)
        geodataframe with polygon geometries
    extensive_variables : list, (optional)
        columns in dataframes for extensive variables
    intensive_variables : list, (optional)
        columns in dataframes for intensive variables
    tables : tuple (optional)
        two 2-D numpy arrays
        SU: area of intersection of source geometry i with union geometry j
        UT: binary mapping of union geometry j to target geometry t
    allocate_total : boolean
        True if total value of source area should be allocated.
        False if denominator is area of i. Note that the two cases
        would be identical when the area of the source polygon is
        exhausted by intersections. See Notes for more details.

    Returns
    -------
    estimates : geopandas.GeoDataFrame
        new geodaraframe with interpolated variables as columns and target_df geometry
        as output geometry

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
    source_df = source_df.copy()
    target_df = target_df.copy()

    if _check_crs(source_df, target_df):
        pass
    else:
        return None

    if tables is None:
        SU, UT = _area_tables(source_df, target_df)
    else:
        SU, UT = tables
    den = source_df[source_df.geometry.name].area.values
    if allocate_total:
        den = SU.sum(axis=1)
    den = den + (den == 0)
    weights = np.dot(np.diag(1 / den), SU)

    dfs = []
    extensive = []
    if extensive_variables:
        for variable in extensive_variables:
            vals = _nan_check(source_df, variable)
            vals = _inf_check(source_df, variable)
            estimates = np.dot(np.diag(vals), weights)
            estimates = np.dot(estimates, UT)
            estimates = estimates.sum(axis=0)
            extensive.append(estimates)
        extensive = np.array(extensive)
        extensive = pd.DataFrame(extensive.T, columns=extensive_variables)

    ST = np.dot(SU, UT)
    area = ST.sum(axis=0)
    den = np.diag(1.0 / (area + (area == 0)))
    weights = np.dot(ST, den)
    intensive = []
    if intensive_variables:
        for variable in intensive_variables:
            vals = _nan_check(source_df, variable)
            vals = _inf_check(source_df, variable)
            vals.shape = (len(vals), 1)
            est = (vals * weights).sum(axis=0)
            intensive.append(est)
        intensive = np.array(intensive)
        intensive = pd.DataFrame(intensive.T, columns=intensive_variables)

    if extensive_variables:
        dfs.append(extensive)
    if intensive_variables:
        dfs.append(intensive)

    df = pd.concat(dfs, axis=1)
    df["geometry"] = target_df[target_df.geometry.name].reset_index(drop=True)
    df = gpd.GeoDataFrame(df.replace(np.inf, np.nan))
    return df


def _area_tables_raster(
    source_df, target_df, raster_path, codes=[21, 22, 23, 24], force_crs_match=True
):
    """
    Construct area allocation and source-target correspondence tables according to a raster 'populated' areas

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        geeodataframe with geometry column of polygon type
    target_df : geopandas.GeoDataFrame
        geodataframe with geometry column of polygon type
    raster_path : str
        the path to the associated raster image.
    codes : list
        list of integer code values that should be considered as 'populated'.
        Since this draw inspiration using the National Land Cover Database (NLCD), the default is 21 (Developed, Open Space), 22 (Developed, Low Intensity), 23 (Developed, Medium Intensity) and 24 (Developed, High Intensity).
        The description of each code can be found here: https://www.mrlc.gov/sites/default/files/metadata/landcover.html
        Only taken into consideration for harmonization raster based.
    force_crs_match : bool (default is True)
        Whether the Coordinate Reference System (CRS) of the polygon will be reprojected to the CRS of the raster file.
        It is recommended to let this argument as True.

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

    if _check_crs(source_df, target_df):
        pass
    else:
        return None
    source_df = source_df.copy()
    target_df = target_df.copy()
    n_s = source_df.shape[0]
    n_t = target_df.shape[0]
    _left = np.arange(n_s)
    _right = np.arange(n_t)
    source_df.loc[:, "_left"] = _left  # create temporary index for union
    target_df.loc[:, "_right"] = _right  # create temporary index for union

    res_union_pre = gpd.overlay(source_df, target_df, how="union")

    # Establishing a CRS for the generated union
    warnings.warn(
        "The CRS for the generated union will be set to be the same as source_df."
    )
    res_union_pre.crs = source_df.crs

    # The 'append_profile_in_gdf' function is present in nlcd.py script
    res_union = _fast_append_profile_in_gdf(
        res_union_pre, raster_path, force_crs_match=force_crs_match
    )

    str_codes = [str(i) for i in codes]
    str_list = ["Type_" + i for i in str_codes]

    # Extract list of code names that actually appear in the appended dataset
    str_list_ok = [col for col in res_union.columns if col in str_list]

    res_union["Populated_Pixels"] = res_union[str_list_ok].sum(axis=1)

    n_u, _ = res_union.shape
    SU = np.zeros(
        (n_s, n_u)
    )  # holds area of intersection of source geom with union geom
    UT = np.zeros((n_u, n_t))  # binary table mapping union geom to target geom

    for index, row in res_union.iterrows():
        # only union polygons that intersect both a source and a target geometry matter
        if not np.isnan(row["_left"]) and not np.isnan(row["_right"]):
            s_id = int(row["_left"])
            t_id = int(row["_right"])
            SU[s_id, index] = row["Populated_Pixels"]
            UT[index, t_id] = 1
    source_df.drop(["_left"], axis=1, inplace=True)
    target_df.drop(["_right"], axis=1, inplace=True)
    return SU, UT
