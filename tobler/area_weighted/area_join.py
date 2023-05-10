import numpy as np
import pandas as pd
import warnings

__author__ = "Martin Fleischmann <martin@martinfleischmann.net>"


def area_join(source_df, target_df, variables):
    """
    Join variables from source_df based on the largest intersection. In case of a tie it picks the first one.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        GeoDataFrame containing source values
    target_df : geopandas.GeoDataFrame
        GeoDataFrame containing source values
    variables : string or list-like
        column(s) in source_df dataframe for variable(s) to be joined

    Returns
    -------
    joined : geopandas.GeoDataFrame
         target_df GeoDataFrame with joined variables as additional columns
    
    """
    if not pd.api.types.is_list_like(variables):
        variables = [variables]

    for v in variables:
        if v in target_df.columns:
            raise ValueError(f"Column '{v}' already present in target_df.")

    target_df = target_df.copy()
    target_ix, source_ix = source_df.sindex.query(
        target_df.geometry, predicate="intersects"
    )
    areas = (
        target_df.geometry.values[target_ix]
        .intersection(source_df.geometry.values[source_ix])
        .area
    )

    main = []
    for i in range(len(target_df)):  # vectorise this loop?
        mask = target_ix == i
        if np.any(mask):
            main.append(source_ix[mask][np.argmax(areas[mask])])
        else:
            main.append(np.nan)

    main = np.array(main, dtype=float)
    mask = ~np.isnan(main)

    for v in variables:
        arr = np.empty(len(main), dtype=object)
        arr[mask] = source_df[v].values[main[mask].astype(int)]
        try:
            arr = arr.astype(source_df[v].dtype)
        except TypeError:
            warnings.warn(
                f"Cannot preserve dtype of '{v}'. Falling back to `dtype=object`.",
            )
        target_df[v] = arr

    return target_df
