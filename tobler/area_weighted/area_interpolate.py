"""
Area Weighted Interpolation

"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags

from tobler.util.util import _check_crs, _inf_check, _nan_check


def _chunk_dfs(geoms_to_chunk, geoms_full, n_jobs):
    chunk_size = geoms_to_chunk.shape[0] // n_jobs + 1
    for i in range(n_jobs):
        start = i * chunk_size
        yield geoms_to_chunk.iloc[start : start + chunk_size], geoms_full


def _index_n_query(geoms1, geoms2):
    # Pick largest for STRTree, query the smallest
    if geoms1.shape[0] > geoms2.shape[0]:
        large = geoms1
        small = geoms2
    else:
        large = geoms2
        small = geoms1
    # Build tree + query
    qry_polyIDs, tree_polyIDs = large.sindex.query(small, predicate="intersects")
    # Remap IDs to global
    large_global_ids = large.iloc[tree_polyIDs].index.values
    small_global_ids = small.iloc[qry_polyIDs].index.values
    # Return always global IDs for geoms1, geoms2
    if geoms1.shape[0] > geoms2.shape[0]:
        return np.array([large_global_ids, small_global_ids]).T
    else:
        return np.array([small_global_ids, large_global_ids]).T


def _chunk_polys(id_pairs, geoms_left, geoms_right, n_jobs):
    chunk_size = id_pairs.shape[0] // n_jobs + 1
    for i in range(n_jobs):
        start = i * chunk_size
        chunk1 = geoms_left.array[id_pairs[start : start + chunk_size, 0]]
        chunk2 = geoms_right.array[id_pairs[start : start + chunk_size, 1]]
        yield chunk1, chunk2


def _intersect_area_on_chunk(geoms1, geoms2):
    areas = geoms1.intersection(geoms2).area
    return areas


def _area_tables_binning_parallel(source_df, target_df, n_jobs=-1):
    """Construct area allocation and source-target correspondence tables using
    a parallel spatial indexing approach
    ...

    NOTE: currently, the largest df is chunked and the other one is shipped in
    full to each core; within each process, the spatial index is built for the
    largest set of geometries, and the other one used for `query`

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        GeoDataFrame containing input data and polygons
    target_df : geopandas.GeoDataFramee
        GeoDataFrame defining the output geometries
    n_jobs : int
        [Optional. Default=-1] Number of processes to run in parallel. If -1,
        this is set to the number of CPUs available

    Returns
    -------
    tables : scipy.sparse.csr_matrix

    """
    from joblib import Parallel, delayed, parallel_backend

    if _check_crs(source_df, target_df):
        pass
    else:
        return None
    if n_jobs == -1:
        n_jobs = os.cpu_count()

    df1 = source_df.copy()
    df2 = target_df.copy()

    # Chunk the largest, ship the smallest in full
    if df1.shape[0] > df2.shape[1]:
        to_chunk = df1
        df_full = df2
    else:
        to_chunk = df2
        df_full = df1

    # Spatial index query
    ## Reindex on positional IDs
    to_workers = _chunk_dfs(
        gpd.GeoSeries(to_chunk.geometry.values, crs=to_chunk.crs),
        gpd.GeoSeries(df_full.geometry.values, crs=df_full.crs),
        n_jobs,
    )

    with parallel_backend("loky", inner_max_num_threads=1):
        worker_out = Parallel(n_jobs=n_jobs)(
            delayed(_index_n_query)(*chunk_pair) for chunk_pair in to_workers
        )

    ids_src, ids_tgt = np.concatenate(worker_out).T

    # Intersection + area calculation
    chunks_to_intersection = _chunk_polys(
        np.vstack([ids_src, ids_tgt]).T, df1.geometry, df2.geometry, n_jobs
    )
    with parallel_backend("loky", inner_max_num_threads=1):
        worker_out = Parallel(n_jobs=n_jobs)(
            delayed(_intersect_area_on_chunk)(*chunk_pair)
            for chunk_pair in chunks_to_intersection
        )
    areas = np.concatenate(worker_out)

    # Build CSR table
    table = coo_matrix(
        (
            areas,
            (ids_src, ids_tgt),
        ),
        shape=(df1.shape[0], df2.shape[0]),
        dtype=np.float32,
    )
    table = table.tocsr()
    return table


def _area_tables_binning(source_df, target_df, spatial_index):
    """Construct area allocation and source-target correspondence tables using a spatial indexing approach
    ...

    NOTE: this currently relies on Geopandas' spatial index machinery

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
    tables : scipy.sparse.csr_matrix

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
        ids_tgt, ids_src = df1.sindex.query(df2.geometry, predicate="intersects")
    elif spatial_index == "target":
        ids_src, ids_tgt = df2.sindex.query(df1.geometry, predicate="intersects")
    else:
        raise ValueError(
            f"'{spatial_index}' is not a valid option. Use 'auto', 'source' or 'target'."
        )

    areas = df1.geometry.values[ids_src].intersection(df2.geometry.values[ids_tgt]).area

    table = coo_matrix(
        (
            areas,
            (ids_src, ids_tgt),
        ),
        shape=(df1.shape[0], df2.shape[0]),
        dtype=np.float32,
    )

    table = table.tocsr()

    return table


def _area_interpolate_binning(
    source_df,
    target_df,
    extensive_variables=None,
    intensive_variables=None,
    table=None,
    allocate_total=True,
    spatial_index="auto",
    n_jobs=1,
    categorical_variables=None,
    categorical_frequency=True,
):
    """
    Area interpolation for extensive, intensive and categorical variables.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
    target_df : geopandas.GeoDataFrame
    extensive_variables : list
        [Optional. Default=None] Columns in dataframes for extensive variables
    intensive_variables : list
        [Optional. Default=None] Columns in dataframes for intensive variables
    table : scipy.sparse.csr_matrix
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
        This argument is ignored if n_jobs>1 (or n_jobs=-1).
    n_jobs : int
        [Optional. Default=1] Number of processes to run in parallel to
        generate the area allocation. If -1, this is set to the number of CPUs
        available. If `table` is passed, this is ignored.
    categorical_variables : list
        [Optional. Default=None] Columns in dataframes for categorical variables
    categorical_frequency : Boolean
        [Optional. Default=True] If True, `estimates` returns the frequency of each
        value in a categorical variable in every polygon of `target_df` (proportion of
        area). If False, `estimates` contains the area in every polygon of `target_df`
        that is occupied by each value of the categorical

    Returns
    -------
    estimates : geopandas.GeoDataFrame
         new geodataframe with interpolated variables as columns and target_df geometry
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

    For categorical variables, the estimate returns ratio of presence of each
    unique category.
    """
    source_df = source_df.copy()
    target_df = target_df.copy()

    if _check_crs(source_df, target_df):
        pass
    else:
        return None

    if table is None:
        if n_jobs == 1:
            table = _area_tables_binning(source_df, target_df, spatial_index)
        else:
            table = _area_tables_binning_parallel(source_df, target_df, n_jobs=n_jobs)

    dfs = []
    extensive = []
    if extensive_variables:
        den = source_df.area.values
        if allocate_total:
            den = np.asarray(table.sum(axis=1))
        den = den + (den == 0)
        den = 1.0 / den
        n = den.shape[0]
        den = den.reshape((n,))
        den = diags([den], [0])
        weights = den.dot(table)  # row standardize table

        for variable in extensive_variables:
            vals = _nan_check(source_df, variable)
            vals = _inf_check(source_df, variable)
            estimates = diags([vals], [0]).dot(weights)
            estimates = estimates.sum(axis=0)
            extensive.append(estimates.tolist()[0])

        extensive = np.asarray(extensive)
        extensive = np.array(extensive)
        extensive = pd.DataFrame(extensive.T, columns=extensive_variables)

    intensive = []
    if intensive_variables:
        area = np.asarray(table.sum(axis=0))
        den = 1.0 / (area + (area == 0))
        n, k = den.shape
        den = den.reshape((k,))
        den = diags([den], [0])
        weights = table.dot(den)

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

    if categorical_variables:
        categorical = {}
        for variable in categorical_variables:
            unique = source_df[variable].unique()
            for value in unique:
                mask = source_df[variable] == value
                categorical[f"{variable}_{value}"] = np.asarray(
                    table[mask].sum(axis=0)
                )[0]

        categorical = pd.DataFrame(categorical)
        if categorical_frequency is True:
            categorical = categorical.div(target_df.area.values, axis="rows")

    if extensive_variables:
        dfs.append(extensive)
    if intensive_variables:
        dfs.append(intensive)
    if categorical_variables:
        dfs.append(categorical)

    df = pd.concat(dfs, axis=1)
    df["geometry"] = target_df[target_df.geometry.name].reset_index(drop=True)
    df = gpd.GeoDataFrame(df.replace(np.inf, np.nan))

    return df.set_index(target_df.index)
