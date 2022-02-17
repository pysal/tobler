"""Model-based methods for areal interpolation."""

import statsmodels.formula.api as smf
from statsmodels.genmod.families import Gaussian, NegativeBinomial, Poisson
from ..util.util import _check_presence_of_crs
from ..dasymetric import _fast_append_profile_in_gdf
import numpy as np


def glm(
    source_df=None,
    target_df=None,
    raster="nlcd_2011",
    raster_codes=None,
    variable=None,
    formula=None,
    likelihood="poisson",
    force_crs_match=True,
    return_model=False,
):
    """Train a generalized linear model to predict polygon attributes based on the collection of pixel values they contain.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame, required
        geodataframe containing source original data to be represented by another geometry
    target_df : geopandas.GeoDataFrame, required
        geodataframe containing target boundaries that will be used to represent the source data
    raster : str, required (default="nlcd_2011")
        path to raster file that will be used to input data to the regression model.
        i.e. a coefficients refer to the relationship between pixel counts and population counts.
        Defaults to 2011 NLCD
    raster_codes : list, required (default =[21, 22, 23, 24, 41, 42, 52])
        list of integers that represent different types of raster cells. If no formula is given,
        the model will be fit from a linear combination of the logged count of each cell type
        listed here. Defaults to [21, 22, 23, 24, 41, 42, 52] which
        are informative land type cells from the NLCD
    variable : str, required
        name of the variable (column) to be modeled from the `source_df`
    formula : str, optional
        patsy-style model formula that specifies the model. Raster codes should be prefixed with
        "Type_", e.g. `"n_total_pop ~ -1 + np.log1p(Type_21) + np.log1p(Type_22)`
    likelihood : str, {'poisson', 'gaussian', 'neg_binomial'} (default = "poisson")
        the likelihood function used in the model
    force_crs_match : bool
        whether to coerce geodataframe and raster to the same CRS
    return model : bool
        whether to return the fitted model in addition to the interpolated geodataframe.
        If true, this will return (geodataframe, model)

    Returns
    --------
    interpolated : geopandas.GeoDataFrame
        a new geopandas dataframe with boundaries from `target_df` and modeled attribute
        data from the `source_df`. If `return_model` is true, the function will also return
        the fitted regression model for further diagnostics


    """
    source_df = source_df.copy()
    target_df = target_df.copy()
    _check_presence_of_crs(source_df)
    liks = {"poisson": Poisson, "gaussian": Gaussian, "neg_binomial": NegativeBinomial}

    if likelihood not in liks.keys():
        raise ValueError(f"likelihood must one of {liks.keys()}")

    if not raster_codes:
        raster_codes = [21, 22, 23, 24, 41, 42, 52]
    raster_codes = ["Type_" + str(i) for i in raster_codes]

    if not formula:
        formula = (
            variable
            + "~ -1 +"
            + "+".join(["np.log1p(" + code + ")" for code in raster_codes])
        )

    profiled_df = _fast_append_profile_in_gdf(
        source_df[[source_df.geometry.name, variable]], raster, force_crs_match
    )

    results = smf.glm(formula, data=profiled_df, family=liks[likelihood]()).fit()

    out = target_df[[target_df.geometry.name]]
    temp = _fast_append_profile_in_gdf(
        out[[out.geometry.name]], raster, force_crs_match
    )

    out[variable] = results.predict(temp.drop(columns=[temp.geometry.name]).fillna(0))

    if return_model:
        return out, results

    return out
