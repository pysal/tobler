"""Kriging inerpolation wrappers."""

def ordinary_kriging(
    source_df,
    target_df,
    variables=None,
    variogram_model="exponential",
    kriging_kwargs=None,
    show_variogram=False,
    return_model=False,
    backend="vectorized",
    n_closest_points=None,
    rescale=False,
):
    """Kriging Interpolation using `pykrige <https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/index.html>`_

    Parameters
    ----------
    source_df : [type]
        [description]
    target_df : [type]
        [description]
    variables : [type], optional
        [description], by default None
    variogram_model : str, optional
        [description], by default "exponential"
    kriging_kwargs : [type], optional
        [description], by default None
    show_variogram : bool, optional
        [description], by default False
    return_model : bool, optional
        [description], by default False
    backend : str, optional
        [description], by default "vectorized"
    n_closest_points : [type], optional
        [description], by default None
    rescale : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    Exception
        [description]
    """    
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError:
        raise Exception("You must have pykrige installed to use kriging interpolation")
    estimates = target_df.copy()[[target_df.geometry.name]]
    if not kriging_kwargs:
        kriging_kwargs = {}

    for variable in variables:

        krig = OrdinaryKriging(
            x=source_df.centroid.x,
            y=source_df.centroid.y,
            z=source_df[variable],
            variogram_model=variogram_model,
            enable_plotting=show_variogram,
            enable_statistics=True,
            **kriging_kwargs
        )
        estimate, ss = krig.execute(
            "points",
            target_df.centroid.x,
            target_df.centroid.y,
            backend=backend,
            n_closest_points=n_closest_points,
        )
        estimates[variable] = estimate
        if rescale:
            scaler = source_df[variable].sum() / estimates[variable].sum()
            estimates[variable] = estimates[variable] * scaler
    if return_model:
        return estimates, krig
    return estimates


def universal_kriging(
    source_df,
    target_df,
    variables=None,
    variogram_model="exponential",
    kriging_kwargs=None,
    show_variogram=False,
    return_model=False,
    backend="vectorized",
    rescale=False,
):
    try:
        from pykrige.ok import UniversalKriging
    except ImportError:
        raise Exception("You must have pykrige installed to use kriging interpolation")
    estimates = target_df.copy()[[target_df.geometry.name]]
    if not kriging_kwargs:
        kriging_kwargs = {}

    for variable in variables:

        krig = UniversalKriging(
            x=source_df.centroid.x,
            y=source_df.centroid.y,
            z=source_df[variable],
            variogram_model=variogram_model,
            enable_plotting=show_variogram,
            enable_statistics=True,
            **kriging_kwargs
        )
        estimate, ss = krig.execute(
            "points",
            target_df.centroid.x,
            target_df.centroid.y,
            backend=backend,
        )
        estimates[variable] = estimate
        if rescale:
            scaler = source_df[variable].sum() / estimates[variable].sum()
            estimates[variable] = estimates[variable] * scaler
    if return_model:
        return estimates, krig
    return estimates

