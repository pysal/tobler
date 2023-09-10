"""
Area Weighted Interpolation, out-of-core and parallel through Dask
"""

import pandas
import geopandas
import numpy as np
from .area_interpolate import _area_interpolate_binning as area_interpolate


def area_interpolate_dask(
    source_dgdf,
    target_dgdf,
    id_col,
    extensive_variables=None,
    intensive_variables=None,
    categorical_variables=None,
    categorical_frequency=True,
):
    """
    Out-of-core and parallel area interpolation for categorical variables.

    Parameters
    ----------
    source_dgdf : dask_geopandas.GeoDataFrame
        Dask-geopandas GeoDataFrame
        IMPORTANT: the table needs to be spatially shuffled and with spatial partitions.
        This is required so only overlapping partitions are checked for interpolation. See
        more on spatial shuffling at: https://dask-geopandas.readthedocs.io/en/stable/guide/spatial-partitioning.html
    target_dgdf : dask_geopandas.GeoDataFrame
        Dask-geopandas GeoDataFrame
        IMPORTANT: the table needs to be spatially shuffled and with spatial partitions.
        This is required so only overlapping partitions are checked for interpolation. See
        more on spatial shuffling at: https://dask-geopandas.readthedocs.io/en/stable/guide/spatial-partitioning.html
    id_col : str
        Name of the column in `target_dgdf` with unique IDs to be used in output table
    extensive_variables : list
        [Optional. Default=None] Columns in `source_dgdf` for extensive variables.
        IMPORTANT: currently NOT implemented.
    intensive_variables : list
        [Optional. Default=None] Columns in `source_dgdf` for intensive variables
        IMPORTANT: currently NOT implemented.
    categorical_variables : list
        [Optional. Default=None] Columns in `source_dgdf` for categorical variables
        IMPORTANT: categorical variables must be of type `'category[known]'`. This is so
        all categories are known ahead of time and Dask can run lazily.
    categorical_frequency : Boolean
        [Optional. Default=True] If True, `estimates` returns the frequency of each
        value in a categorical variable in every polygon of `target_df` (proportion of
        area). If False, `estimates` contains the area in every polygon of `target_df`
        that is occupied by each value of the categorical


    Returns
    -------
    estimates : dask_geopandas.GeoDataFrame
         new dask-geopandas geodaraframe with interpolated variables and `id_col` as
         columns and target_df geometry as output geometry

    """
    try:
        import dask_geopandas
        from dask.base import tokenize
        from dask.highlevelgraph import HighLevelGraph
    except ImportError:
        raise ImportError(
            "Area interpolation with Dask requires `dask` and "
            "`dask_geopandas` installed to run. Please install them "
            "before importing this functionality."
        )

    if intensive_variables is not None:
        raise NotImplementedError(
            (
                "Dask-based interpolation of intensive variables is "
                "not implemented yet. Please remove intensive variables to "
                "be able to run the rest."
            )
        )
    if extensive_variables is not None:
        raise NotImplementedError(
            (
                "Dask-based interpolation of extensive variables is "
                "not implemented yet. Please remove intensive variables to "
                "be able to run the rest."
            )
        )
    # Categoricals must be Dask's known categorical
    if categorical_variables is not None:
        category_vars = []
        for cat_var in categorical_variables:
            var_names = [f"{cat_var}_{c}" for c in source_dgdf[cat_var].cat.categories]
            category_vars.extend(var_names)
    else:
        category_vars = None
    # Build tasks by joining pairs of chunks from left/right
    dsk = {}
    new_spatial_partitions = []
    parts = geopandas.sjoin(
        source_dgdf.spatial_partitions.to_frame("geometry"),
        target_dgdf.spatial_partitions.to_frame("geometry"),
        how="inner",
        predicate="intersects",
    )
    parts_left = np.asarray(parts.index)
    parts_right = np.asarray(parts["index_right"].values)
    name = "area_interpolate-" + tokenize(target_dgdf, source_dgdf)
    for i, (l, r) in enumerate(zip(parts_left, parts_right)):
        dsk[(name, i)] = (
            id_area_interpolate,
            (source_dgdf._name, l),
            (target_dgdf._name, r),
            id_col,
            extensive_variables,
            intensive_variables,
            None,
            True,
            "auto",
            1,
            categorical_variables,
            category_vars,
        )
        lr = source_dgdf.spatial_partitions.iloc[l]
        rr = target_dgdf.spatial_partitions.iloc[r]
        extent = lr.intersection(rr)
        new_spatial_partitions.append(extent)
    # Create geometries for new spatial partitions
    new_spatial_partitions = geopandas.GeoSeries(
        data=new_spatial_partitions, crs=source_dgdf.crs
    )
    # Build Dask graph
    graph = HighLevelGraph.from_collections(
        name, dsk, dependencies=[source_dgdf, target_dgdf]
    )
    # Get metadata for the outcome table
    meta = id_area_interpolate(
        source_dgdf._meta,
        target_dgdf._meta,
        id_col,
        extensive_variables=extensive_variables,
        intensive_variables=intensive_variables,
        table=None,
        allocate_total=True,
        spatial_index="auto",
        n_jobs=1,
        categorical_variables=categorical_variables,
        category_vars=category_vars,
    )
    # Build output table
    transferred = dask_geopandas.GeoDataFrame(
        graph, name, meta, [None] * (len(dsk) + 1), new_spatial_partitions
    )
    # Merge chunks
    out = target_dgdf[[id_col, "geometry"]]
    ## Extensive --> Not implemented (DAB: the below does not match single-core)
    """
    if extensive_variables is not None:
        out_extensive = (
            transferred
            .groupby(id_col)
            [extensive_variables]
            .agg({v: 'sum' for v in extensive_variables})
        )
        out = out.join(out_extensive, on=id_col)
    """
    ## Intensive --> Weight by area of the chunk (Not implemented)
    ## Categorical --> Add up proportions
    if categorical_variables is not None:
        out_categorical = (
            transferred[category_vars]
            .astype(float)
            .groupby(transferred[id_col])
            .agg({v: "sum" for v in category_vars})
        )
        out = out.join(out_categorical, on=id_col)
        if categorical_frequency is True:
            cols = out_categorical.columns.tolist()
            out[cols] = out[cols].div(out.area, axis="index")
    return out


def id_area_interpolate(
    source_df,
    target_df,
    id_col,
    extensive_variables=None,
    intensive_variables=None,
    table=None,
    allocate_total=True,
    spatial_index="auto",
    n_jobs=1,
    categorical_variables=None,
    category_vars=None,
):
    """
    Light wrapper around single-core area interpolation to be run on distributed workers

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
    target_df : geopandas.GeoDataFrame
    id_col : str
        Name of the column in `target_dgdf` with unique IDs to be used in output table
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
    categories : list
        [Optional. Default=None] Full list of category names in the format
        `f'{var_name}_{cat_name}'`

    Returns
    -------
    estimates : geopandas.GeoDataFrame
         new geodaraframe with interpolated variables as columns and target_df geometry
         as output geometry

    """
    estimates = area_interpolate(
        source_df,
        target_df,
        extensive_variables=extensive_variables,
        intensive_variables=intensive_variables,
        table=table,
        allocate_total=allocate_total,
        spatial_index=spatial_index,
        n_jobs=n_jobs,
        categorical_variables=categorical_variables,
        categorical_frequency=False,
    )
    estimates[id_col] = target_df[id_col].values

    if categorical_variables is not None:
        category_vars_to_add = []
        for category_var in category_vars:
            if category_var not in estimates.columns:
                category_vars_to_add.append(category_var)
        estimates = estimates.join(
            pandas.DataFrame(index=estimates.index, columns=category_vars_to_add)
        )
    return estimates


def area_interpolate_dask_native(
    source_dgdf,
    target_dgdf,
    source_id,
    target_id,
    extensive_variables=None,
    intensive_variables=None,
    categorical_variables=None,
    categorical_frequency=True
):
    # Categoricals must be Dask's known categorical
    if categorical_variables is not None:
        category_vars = []
        for cat_var in categorical_variables:
            var_names = [f'{cat_var}_{c}' for c in source_dgdf[cat_var].cat.categories]
            category_vars.extend(var_names)
    else:
        category_vars = None
    #----------------------------------------
    # Build cross-over table
    #----------------------------------------
    # Build tasks by joining pairs of chunks from left/right
    dsk = {}
    new_spatial_partitions = []
    parts = geopandas.sjoin(
        source_dgdf.spatial_partitions.to_frame('geometry'),
        target_dgdf.spatial_partitions.to_frame('geometry'),
        how='inner',
        predicate='intersects'
    )
    parts_left = np.asarray(parts.index)
    parts_right = np.asarray(parts['index_right'].values)
    name = 'area_interpolate-' + tokenize(
        target_dgdf, source_dgdf
    )   
    # Create computation items for workers
    for i, (l, r) in enumerate(zip(parts_left, parts_right)):
        dsk[(name, i)] = (
            id_area_table,
            (source_dgdf._name, l),
            (target_dgdf._name, r),
            source_id,
            target_id,
            'auto',
        )
        lr = source_dgdf.spatial_partitions.iloc[l]
        rr = target_dgdf.spatial_partitions.iloc[r]
        extent = lr.intersection(rr)
        new_spatial_partitions.append(extent)
    # Create geometries for new spatial partitions
    new_spatial_partitions = geopandas.GeoSeries(
        data=new_spatial_partitions, crs=source_dgdf.crs
    )
    # Build Dask graph
    graph = HighLevelGraph.from_collections(
        name, dsk, dependencies=[source_dgdf, target_dgdf]
    )
    # Get metadata for the outcome table
    meta = id_area_table(
        source_dgdf._meta,
        target_dgdf._meta,
        source_id,
        target_id,
        spatial_index='auto',
    )
    #----------------------------------------
    # Build output table
    areas = dask_geopandas.GeoDataFrame(
        graph, 
        name,
        meta,
        [None] * (len(dsk) + 1),
        new_spatial_partitions
    )
    # Merge chunks
    out = target_dgdf[[target_id, 'geometry']]
    ## Extensive --> Not implemented (DAB: the below does not match single-core)
    if extensive_variables is not None:
        weights = areas / (
            areas.groupby(source_id).transform('sum', meta=areas._meta)
        )
        tmp = (
            weights
            .join(
                source_df.set_index(source_id)[extensive_variables], 
                on=source_id
            )
        )
        for ev in extensive_variables:
            tmp[ev] = tmp[[ev, 'weight']].prod(axis='columns')
        out_extensive = tmp.groupby(target_id)[extensive_variables].sum()
        out = out.join(out_extensive, on=target_id)
    '''
    ## Intensive --> Weight by area of the chunk (Not implemented)
    ## Categorical --> Add up proportions
    if categorical_variables is not None:
        out_categorical = (
            transferred
            [category_vars]
            .astype(float)
            .groupby(transferred[target_id])
            .agg({v: 'sum' for v in category_vars})
        )    
        out = out.join(out_categorical, on=target_id)
        if categorical_frequency is True:
            cols = out_categorical.columns.tolist()
            out[cols] = out[cols].div(
                out.area, axis='index'
            )
    '''
    return out

def id_area_table(
    source_df,
    target_df,
    source_id,
    target_id,
    spatial_index='auto',
):
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
    
    table = pandas.DataFrame({
        source_id: source_df[source_id].iloc[ids_src].values,
        target_id: target_df[target_id].iloc[ids_tgt].values,
        'overlap': areas
    })
    table['uid'] = table[source_id].astype(str) + '-' + table[source_id].astype(str)
    
def area_interpolate_dask_native(
    source_dgdf,
    target_dgdf,
    source_id,
    target_id,
    extensive_variables=None,
    intensive_variables=None,
    categorical_variables=None,
    categorical_frequency=True
):
    # Categoricals must be Dask's known categorical
    if categorical_variables is not None:
        category_vars = []
        for cat_var in categorical_variables:
            var_names = [f'{cat_var}_{c}' for c in source_dgdf[cat_var].cat.categories]
            category_vars.extend(var_names)
    else:
        category_vars = None
    #----------------------------------------
    # Build cross-over table
    #----------------------------------------
    # Build tasks by joining pairs of chunks from left/right
    dsk = {}
    new_spatial_partitions = []
    parts = geopandas.sjoin(
        source_dgdf.spatial_partitions.to_frame('geometry'),
        target_dgdf.spatial_partitions.to_frame('geometry'),
        how='inner',
        predicate='intersects'
    )
    parts_left = np.asarray(parts.index)
    parts_right = np.asarray(parts['index_right'].values)
    name = 'area_interpolate-' + tokenize(
        target_dgdf, source_dgdf
    )   
    # Create computation items for workers
    for i, (l, r) in enumerate(zip(parts_left, parts_right)):
        dsk[(name, i)] = (
            id_area_table,
            (source_dgdf._name, l),
            (target_dgdf._name, r),
            source_id,
            target_id,
            'auto',
        )
        lr = source_dgdf.spatial_partitions.iloc[l]
        rr = target_dgdf.spatial_partitions.iloc[r]
        extent = lr.intersection(rr)
        new_spatial_partitions.append(extent)
    # Create geometries for new spatial partitions
    new_spatial_partitions = geopandas.GeoSeries(
        data=new_spatial_partitions, crs=source_dgdf.crs
    )
    # Build Dask graph
    graph = HighLevelGraph.from_collections(
        name, dsk, dependencies=[source_dgdf, target_dgdf]
    )
    # Get metadata for the outcome table
    meta = id_area_table(
        source_dgdf._meta,
        target_dgdf._meta,
        source_id,
        target_id,
        spatial_index='auto',
    )
    #----------------------------------------
    # Build output table
    areas = dask_geopandas.GeoDataFrame(
        graph, 
        name,
        meta,
        [None] * (len(dsk) + 1),
        new_spatial_partitions
    )
    # Merge chunks
    out = target_dgdf[[target_id, 'geometry']]
    ## Extensive --> Not implemented (DAB: the below does not match single-core)
    if extensive_variables is not None:
        weights = areas / (
            areas.groupby(source_id).transform('sum', meta=areas._meta)
        )
        tmp = (
            weights
            .join(
                source_df.set_index(source_id)[extensive_variables], 
                on=source_id
            )
        )
        for ev in extensive_variables:
            tmp[ev] = tmp[[ev, 'weight']].prod(axis='columns')
        out_extensive = tmp.groupby(target_id)[extensive_variables].sum()
        out = out.join(out_extensive, on=target_id)
    '''
    ## Intensive --> Weight by area of the chunk (Not implemented)
    ## Categorical --> Add up proportions
    if categorical_variables is not None:
        out_categorical = (
            transferred
            [category_vars]
            .astype(float)
            .groupby(transferred[target_id])
            .agg({v: 'sum' for v in category_vars})
        )    
        out = out.join(out_categorical, on=target_id)
        if categorical_frequency is True:
            cols = out_categorical.columns.tolist()
            out[cols] = out[cols].div(
                out.area, axis='index'
            )
    '''
    return out

def id_area_table(
    source_df,
    target_df,
    source_id,
    target_id,
    spatial_index='auto',
):
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
    
    table = pandas.DataFrame({
        source_id: source_df[source_id].iloc[ids_src].values,
        target_id: target_df[target_id].iloc[ids_tgt].values,
        'overlap': areas
    })
    table['uid'] = table[source_id].astype(str) + '-' + table[source_id].astype(str)
    return table.set_index('uid')