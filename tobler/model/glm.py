"""Model-based methods for areal interpolation."""

import warnings

import geopandas as gpd
import numpy as np  # noqa: F401 -- needed for numpy's logarithm in the formula string
import statsmodels.formula.api as smf
from statsmodels.genmod.families import NegativeBinomial, Poisson

from ..dasymetric import extract_raster_features
from ..util.util import _check_presence_of_crs, _rename_type

__all__ = ["glm"]


def glm(
    source_df=None,
    target_df=None,
    raster="nlcd_2011",
    pixel_values=None,
    extensive_variable=None,
    formula=None,
    likelihood="poisson",
    return_model=False,
    nodata=255,
    n_jobs=-1,
    force_crs_match=None,
):
    """Train a generalized linear model to predict polygon attributes
    based on the collection of pixel values they contain.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        geodataframe containing source original data to
        be represented by another geometry
    target_df : geopandas.GeoDataFrame
        geodataframe containing target boundaries that
        will be used to represent the source data
    raster : str, default="nlcd_2011"
        path to raster file that will be used to input data to the regression model.
        i.e. a coefficients refer to the relationship between pixel
        counts and population counts. Defaults to 2011 NLCD
    pixel_values : list, default=[21, 22, 23, 24, 41, 42, 52]
        list of integers that represent different types of raster cells. If no formula
        is given, the model will be fit from a linear combination of the logged count
        of each cell type listed here. Defaults to [21, 22, 23, 24, 41, 42, 52] which
        are informative land type cells from the NLCD
    extensive_variable : str
        name of the variable (column) to be modeled from the `source_df`.
        Currently, only one extensive variable is available.
    formula : str, optional
        patsy-style model formula that specifies the model. Raster codes should be
        prefixed with "Type_", e.g.
        `"n_total_pop ~ -1 + Type_21 + Type_22`
    likelihood : {"poisson", "neg_binomial"}, default="poisson"
        the likelihood function used in the model
    return_model : bool
        whether to return the fitted model in addition to the interpolated geodataframe.
        If true, this will return (geodataframe, model)
    nodata : int
        value in raster that indicates null or missing values. Default is 255
    n_jobs : int, default=-1
        Number of processes to run in parallel to
        generate the area allocation. If -1, this is set to the number of CPUs
        available.
    force_crs_match : None
        whether to coerce geodataframe and raster to the same CRS
        -- Scheduled for removal -- no longer used.

    Returns
    -------
    interpolated : geopandas.GeoDataFrame
        a new geopandas dataframe with boundaries from `target_df` and modeled
        attribute data from the `source_df`. If `return_model` is true, the
        function will also return the fitted regression model for further diagnostics
        The new geopandas will present a column named given by extensive_variable
    """

    if force_crs_match is not None:
        warnings.warn(
            (
                "The 'force_crs_match' is not longer used in 'glm()'"
                "and will be removed in a future release."
            ),
            UserWarning,
            stacklevel=2,
        )

    source_df = source_df.copy()
    target_df = target_df.copy()
    _check_presence_of_crs(source_df)
    liks = {"poisson": Poisson, "neg_binomial": NegativeBinomial}

    if likelihood not in liks:
        raise ValueError(f"likelihood must one of {liks.keys()}")

    if not pixel_values:
        pixel_values = [21, 22, 23, 24, 41, 42, 52]
    pixel_values_str = ["Type_" + str(i) for i in pixel_values]

    if not formula:
        formula = extensive_variable + "~ -1 +" + " + ".join(list(pixel_values_str))

    #  create a vector mask from the raster data
    # Pass collapse_values=False to obtain each pixel type
    raster_mask = extract_raster_features(
        source_df, raster, pixel_values, nodata, n_jobs, collapse_values=False
    )

    raster_mask = raster_mask.to_crs(source_df.crs)

    raster_mask = raster_mask.copy()

    source_df["source_id"] = source_df.index
    target_df["target_id"] = target_df.index
    raster_mask["mask_id"] = raster_mask.index

    # Intersect the two layers (source + raster)
    source_intersections = gpd.overlay(
        raster_mask[["mask_id", "value", "geometry"]],
        source_df[["source_id", "geometry"]],
        how="intersection",
    )

    # Since each intersection is only comprised of a single value,
    # The logic will be adjusted to the area (instead of counts)
    # of each pixel type

    # Area of intersected pieces of each category fall in each source polygon

    source_intersections["area"] = source_intersections.geometry.area

    area_by_source_id_values = (
        source_intersections.groupby(["source_id", "value"])["area"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    source_result = source_df.merge(
        area_by_source_id_values, on="source_id", how="left"
    ).fillna(0)
    source_result = _rename_type(source_result)
    results_regression = smf.glm(
        formula, data=source_result, family=liks[likelihood]()
    ).fit()

    # Extract the log from predict to get the weights sums
    # "raw" is because this prediction is only generated through the regression
    # (that is, without enforcing pycnophylactic property)
    source_result["pred_variable_on_source_id_raw"] = np.log(
        results_regression.predict(source_result.fillna(0))
    )

    # Append source predictions and original
    # variable for the pycnophylactic property step
    source_intersections_2 = source_intersections.merge(
        source_result[
            ["source_id", extensive_variable, "pred_variable_on_source_id_raw"]
        ],
        on="source_id",
        how="left",
    ).fillna(0)

    # Match each pixel to its corresponding weight
    coefs = results_regression.params
    type_coefs = coefs[coefs.index.str.match(r"^Type_\d+$")]
    mapping = {  # build mapping: 21 -> coef(Type_21), 22 -> coef(Type_22), ...
        int(name.split("_", 1)[1]): coef for name, coef in type_coefs.items()
    }

    # add coefficient as a new column
    source_intersections_2["type_coef"] = source_intersections_2["value"].map(mapping)

    # Ensure pycnophylactic property
    source_intersections_2["pred_variable_on_pixel"] = source_intersections_2[
        "type_coef"
    ] * (
        source_intersections_2[extensive_variable]
        / source_intersections_2["pred_variable_on_source_id_raw"]
    )

    # Predictions should be adjusted by area instead of pixels (appends "_wa")
    # "wa" stands for weighted area
    source_intersections_2["area"] = source_intersections_2.geometry.area
    source_intersections_2["pred_variable_on_pixel_wa"] = (
        source_intersections_2["pred_variable_on_pixel"]
        * source_intersections_2["area"]
    )

    # Intersect the two layers (target + raster)
    target_intersections = (
        gpd.overlay(
            raster_mask[["mask_id", "geometry"]],
            target_df[["target_id", "geometry"]],
            how="intersection",
        )
        .dissolve("mask_id")
        .reset_index()
    )  # Dissolve here is only to get unique mask_ids

    # Recover pixel estimated population for each mask id within every target id
    # Use the weighted variable!
    target_intersections_2 = target_intersections.merge(
        source_intersections_2[["mask_id", "pred_variable_on_pixel_wa"]],
        on="mask_id",
        how="inner",
    )

    # Sum the weights for each target polygon to get the
    # estimated population with pycnophylactic property
    # Use the weighted variable!
    sum_by_target = target_intersections_2.groupby("target_id", as_index=False)[
        "pred_variable_on_pixel_wa"
    ].sum()

    # Append these estimates into the original target geopandas
    interpolated = target_df.merge(sum_by_target, on="target_id", how="inner").rename(
        columns={"pred_variable_on_pixel_wa": extensive_variable}
    )

    # Return only relevant columns
    interpolated = interpolated[["geometry", extensive_variable]]

    if return_model:
        return interpolated, results_regression

    return interpolated
