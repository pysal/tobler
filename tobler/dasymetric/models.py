from ..area_weighted.vectorized_raster_interpolation import (
    calculate_interpolated_population_from_correspondence_table,
    return_weights_from_regression,
    create_non_zero_population_by_pixels_locations,
    _check_presence_of_crs,
)
from ..data import fetch_quilt_path
import rasterio
import warnings


def linear_model(
    source_df=None,
    target_df=None,
    raster="nlcd_2011",
    raster_codes=None,
    variable=None,
    formula=None,
    likelihood="poisson",
    force_crs_match=True,
):
    """Interpolate data between two polygonal datasets using an auxiliary raster to as inut to a linear regression model.

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
    raster_codes : list, required (default =[21, 22, 23, 24])
        list of inteegers that represent different types of raster cells. Defaults to [21, 22, 23, 24] which
        are considered developed land types in the NLCD
    variable : str, required
        name of the variable (column) to be modeled from the `source_df`
    formula : str, optional
        patsy-style model formula
    likelihood : str, {'poisson', 'gaussian'} (default = "poisson")
        the likelihood function used in the model

    Returns
    --------
    interpolated: geopandas.GeoDataFrame
        a new geopandas dataframe with boundaries from `target_df` and modeled attribute data from the `source_df`

    """
    if not raster_codes:
        raster_codes = [21, 22, 23, 24]

        # build weights from raster and vector data
        weights = return_weights_from_regression(
            geodataframe=source_df,
            raster_path=raster,
            pop_string=variable,
            formula_string=formula,
            codes=raster_codes,
            force_crs_match=force_crs_match,
            likelihood=likelihood,
        )

        # match vector population to pixel counts
        correspondence_table = create_non_zero_population_by_pixels_locations(
            geodataframe=source_df, raster=raster, pop_string=variable, weights=weights
        )

        # estimate the model
        interpolated = calculate_interpolated_population_from_correspondence_table(
            target_df, raster, correspondence_table
        )

    return interpolated
