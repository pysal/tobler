import pandas
import geopandas
import dask_geopandas
import warnings
import numpy as np
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from tobler.area_weighted import area_interpolate

def area_interpolate_dask(
    left_dgdf,
    right_dgdf,
    id_col,
    extensive_variables=None,
    intensive_variables=None,
    categorical_variables=None,
):
    if intensive_variables is not None:
        raise NotImplementedError((
            "Dask-based interpolation of intensive variables is "
            "not implemented yet. Please remove intensive variables to "
            "be able to run the rest."
        ))
    # Categoricals must be Dask's known categorical
    if categorical_variables is not None:
        category_vars = []
        for cat_var in categorical_variables:
            var_names = [f'{cat_var}_{c}' for c in left_dgdf[cat_var].cat.categories]
            category_vars.extend(var_names)
    else:
        category_vars = None
    # Build tasks by joining pairs of chunks from left/right
    dsk = {}
    new_spatial_partitions = []
    parts = geopandas.sjoin(
        left_dgdf.spatial_partitions.to_frame('geometry'),
        right_dgdf.spatial_partitions.to_frame('geometry'),
        how='inner',
        predicate='intersects'
    )
    parts_left = np.asarray(parts.index)
    parts_right = np.asarray(parts['index_right'].values)
    name = 'area_interpolate-' + tokenize(
        right_dgdf, left_dgdf
    )   
    for i, (l, r) in enumerate(zip(parts_left, parts_right)):
        dsk[(name, i)] = (
            id_area_interpolate,
            (left_dgdf._name, l),
            (right_dgdf._name, r),
            id_col,
            extensive_variables,
            intensive_variables,
            None,
            True,
            'auto',
            1,
            categorical_variables,
            category_vars
        )
        lr = left_dgdf.spatial_partitions.iloc[l]
        rr = right_dgdf.spatial_partitions.iloc[r]
        extent = lr.intersection(rr)
        new_spatial_partitions.append(extent)
    # Create geometries for new spatial partitions
    new_spatial_partitions = geopandas.GeoSeries(
        data=new_spatial_partitions, crs=left_dgdf.crs
    )
    # Build Dask graph
    graph = HighLevelGraph.from_collections(
        name, dsk, dependencies=[left_dgdf, right_dgdf]
    )
    # Get metadata for the outcome table
    meta = id_area_interpolate(
        left_dgdf._meta,
        right_dgdf._meta,
        id_col,
        extensive_variables=extensive_variables,
        intensive_variables=intensive_variables,
        table=None,
        allocate_total=True,
        spatial_index='auto',
        n_jobs=1,
        categorical_variables=categorical_variables,
        category_vars=category_vars
    )
    # Build output table
    transferred = dask_geopandas.GeoDataFrame(
        graph, 
        name,
        meta,
        [None] * (len(dsk) + 1),
        new_spatial_partitions
    )
    # Merge chunks
    out = right_dgdf[[id_col, 'geometry']]
    ## Extensive --> Add up estimates by ID
    if extensive_variables is not None:
        out_extensive = (
            transferred
            .groupby(id_col)
            [extensive_variables]
            .agg({v: 'sum' for v in extensive_variables})
        )
        out = out.join(out_extensive, on=id_col)
    ## Intensive --> Weight by area of the chunk (Not implemented)
    ## Categorical --> Add up proportions
    if categorical_variables is not None:
        out_categorical = (
            transferred
            [category_vars + [id_col]]
            .groupby(id_col)
            .agg({v: 'sum' for v in category_vars})
        )    
        out = out.join(out_categorical, on=id_col)
    return out

def id_area_interpolate(
    source_df,
    target_df,
    id_col,
    extensive_variables=None,
    intensive_variables=None,
    table=None,
    allocate_total=True,
    spatial_index='auto',
    n_jobs=1,
    categorical_variables=None,
    category_vars=None
):
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
