"""Functions to perform Population Interpolation using a vectorized version of
a raster file. This is a generic framework that can be used to distribute
population more accurately in harmonized spatial structures.

Inspired by Reibel, Michael, and Aditya Agrawal. "Areal interpolation of population counts using pre-classified land cover data." Population Research and Policy Review 26.5-6 (2007): 619-633.

Note: This study was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Process number 88881.170553/2018-01
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import json
import pandas as pd
from geopandas import GeoDataFrame
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
import rasterstats as rs
import warnings
from tqdm.auto import tqdm
from tobler.util.util import _check_presence_of_crs

import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson, Gaussian, NegativeBinomial

__all__ = [
    "getFeatures",
    "_fast_append_profile_in_gdf",
    "_return_weights_from_regression",
    "_return_weights_from_xgboost",
    "create_lon_lat",
    "_create_non_zero_population_by_pixels_locations",
    "_calculate_interpolated_polygon_population_from_correspondence_table",
    "_calculate_interpolated_population_from_correspondence_table",
]


def getFeatures(gdf):

    """Function to parse features from GeoDataFrame in such a manner that
    rasterio wants them.

    Notes
    -----
    This function was obtained at https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html.
    """

    return [json.loads(gdf.to_json())["features"][0][gdf.geometry.name]]


def _fast_append_profile_in_gdf(geodataframe, raster_path, force_crs_match=True):

    """Function that appends the columns of the profile in a geopandas
    according to a given raster taking advantage of rasterstats.

    geodataframe    : geopandas.GeoDataFrame
        geodataframe that has overlay with the raster. The variables of the profile will be appended in this data.
        If some polygon do not overlay the raster, consider a preprocessing step using the function subset_gdf_polygons_from_raster.
    raster_path     : str
        the path to the associated raster image.
    force_crs_match : bool, Default is True.
        Wheter the Coordinate Reference System (CRS) of the polygon will be reprojected to the CRS of the raster file.
        It is recommended to let this argument as True.

    Notes
    -----
    The generated geodataframe will input the value 0 for each Type that is not present in the raster for each polygon.
    """

    _check_presence_of_crs(geodataframe)
    if force_crs_match:
        with rasterio.open(raster_path) as raster:
            # raster =
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                geodataframe = geodataframe.to_crs(crs=raster.crs.data)
    else:
        warnings.warn(
            "The GeoDataFrame is not being reprojected. The clipping might be being performing on unmatching polygon to the raster."
        )

    zonal_gjson = rs.zonal_stats(
        geodataframe, raster_path, prefix="Type_", geojson_out=True, categorical=True
    )

    zonal_ppt_gdf = GeoDataFrame.from_features(zonal_gjson)

    return zonal_ppt_gdf


def _return_weights_from_regression(
    geodataframe,
    raster_path,
    pop_string,
    codes=[21, 22, 23, 24],
    likelihood="poisson",
    formula_string=None,
    n_pixels_option_values=256,
    force_crs_match=True,
    na_value=255,
    ReLU=True,
):

    """Function that returns the weights of each land type according to NLCD
    types/codes.

    Parameters
    ----------
    geodataframe :  geopandas.GeoDataFrame 
        used to build regression
    raster_path : str
        the path to the associated raster image.
    formula_string : str
        patsy-style model formula
    pop_string : str
        the name of the variable on geodataframe that the regression shall be conducted
    codes : list
        an integer list of codes values that should be considered as 'populated' from the National Land Cover Database (NLCD).
        The description of each code can be found here: https://www.mrlc.gov/sites/default/files/metadata/landcover.html
        The default is 21 (Developed, Open Space), 22 (Developed, Low Intensity), 23 (Developed, Medium Intensity) and 24 (Developed, High Intensity).
    likelihood : str, {'Poisson', 'Gaussian'}
        the likelihood assumed for the dependent variable (population). It can be 'Poisson' or 'Gaussian'.
        With the 'Poisson' a Generalized Linear Model with log as link function will be fitted and 'Gaussian' an ordinary least squares will be fitted.
    n_pixels_option_values : int
        number of options of the pixel values of rasterior. Default is 256.
    force_crs_match   : bool. Default is True.
        Wheter the Coordinate Reference System (CRS) of the polygon will be reprojected to the CRS of the raster file.
        It is recommended to let this argument as True.
    na_value : int. Default is 255.
        The number which is considered to be 'Not a Number' (NaN) in the raster pixel values.
    ReLU : bool. Default is True.
         Whether the Rectified Linear Units (ReLU) transformation will be used to avoid negative weights for the land types.

    Notes
    -----
    1) The formula uses a substring called 'Type_' before the code number due to the 'append_profile_in_gdf' function.
    2) The pixel value, usually, ranges from 0 to 255. That is why the default of 'n_pixels_option_values' is 256.
    """

    _check_presence_of_crs(geodataframe)

    if na_value in codes:
        raise ValueError("codes should not assume the na_value value.")

    if not likelihood in ["poisson", "gaussian"]:
        raise ValueError("likelihood must one of 'poisson', 'gaussian'")

    profiled_df = _fast_append_profile_in_gdf(
        geodataframe[[geodataframe.geometry.name, pop_string]],
        raster_path,
        force_crs_match,
    )  # Use only two columns to build the weights (this avoids error, if the original dataset has already types appended on it).

    # If the list is unsorted, the codes will be sorted to guarantee that the position of the weights will match
    codes.sort()

    if not formula_string:
        # Formula WITHOUT intercept
        str_codes = [str(i) for i in codes]
        formula_string = (
            pop_string + " ~ -1 + " + " + ".join(["Type_" + s for s in str_codes])
        )

    if likelihood == "poisson":
        results = smf.glm(formula_string, data=profiled_df, family=Poisson()).fit()

    if likelihood == "gaussian":
        results = smf.ols(formula_string, data=profiled_df).fit()

    weights = np.zeros(n_pixels_option_values)
    weights[codes] = results.params

    if ReLU:
        weights = np.where(weights < 0, 0, weights)

    return weights


def _return_weights_from_xgboost(
    geodataframe,
    raster_path,
    pop_string,
    codes=[21, 22, 23, 24],
    n_pixels_option_values=256,
    tuned_xgb=False,
    gbm_hyperparam_grid={
        "learning_rate": [0.001, 0.01, 0.1],
        "n_estimators": [200],
        "subsample": [0.3, 0.5],
        "max_depth": [4, 5, 6],
        "num_boosting_rounds": [10, 20],
    },
    force_crs_match=True,
    na_value=255,
    ReLU=True,
):

    """Function that returns the weights of each land type according to NLCD
    types/codes given by Extreme Gradient Boost model (XGBoost)

    Parameters
    ----------

    geodataframe           : a geopandas geoDataFrame used to build regression

    raster_path            : the path to the associated raster image.

    pop_string             : the name of the variable on geodataframe that the regression shall be conducted

    codes                  : an integer list of codes values that should be considered as 'populated' from the National Land Cover Database (NLCD).
                             The description of each code can be found here: https://www.mrlc.gov/sites/default/files/metadata/landcover.html
                             The default is 21 (Developed, Open Space), 22 (Developed, Low Intensity), 23 (Developed, Medium Intensity) and 24 (Developed, High Intensity).

    n_pixels_option_values : number of options of the pixel values of rasterior. Default is 256.

    tuned_xgb              : bool. Default is False.
                             If True the XGBoost model will be tuned making a grid search using gbm_hyperparam_grid dictionary a picking the best model in terms of mean squared error with some pre-defined number of cross-validation.
                             Otherwise, the XGBoost model is fitted with default values of xgboost.train function from xgboost Python library.

    gbm_hyperparam_grid    : a dictionary that represent the grid for the grid search of XGBoost.

    force_crs_match        : bool. Default is True.
                             Wheter the Coordinate Reference System (CRS) of the polygon will be reprojected to the CRS of the raster file.
                             It is recommended to let this argument as True.

    na_value               : int. Default is 255.
                             The number which is considered to be 'Not a Number' (NaN) in the raster pixel values.

    ReLU                   : bool. Default is True.
                             Wheter the Rectified Linear Units (ReLU) transformation will be used to avoid negative weights for the land types.

    Notes
    -----
    1) The formula uses a substring called 'Type_' before the code number due to the 'append_profile_in_gdf' function.
    1) The formula uses a substring called 'Type_' before the code number due to the 'append_profile_in_gdf' function.
    2) The pixel value, usually, ranges from 0 to 255. That is why the default of 'n_pixels_option_values' is 256.
    3) The returning weights represent the average of the Shapley's values from each feature.
    """
    try:
        import xgboost as xgb
        import shap
    except ImportError as e:
        raise ImportError("xgboost and shap are required to perform this.")

    _check_presence_of_crs(geodataframe)

    if na_value in codes:
        raise ValueError("codes should not assume the na_value value.")

    profiled_df = _fast_append_profile_in_gdf(
        geodataframe[[geodataframe.geometry.name, pop_string]],
        raster_path,
        force_crs_match,
    )  # Use only two columns to build the weights (this avoids error, if the original dataset has already types appended on it).

    # If the list is unsorted, the codes will be sorted to guarantee that the position of the weights will match
    codes.sort()

    str_codes = [str(i) for i in codes]
    feature_names = ["Type_" + s for s in str_codes]

    y = profiled_df[pop_string]
    X = profiled_df[feature_names]

    if tuned_xgb == False:

        # Create the DMatrix
        xgb_dmatrix = xgb.DMatrix(X, y)

        # Create the parameter dictionary
        params = {"objective": "reg:linear"}

        # Train the model
        xg_reg = xgb.train(params=params, dtrain=xgb_dmatrix)

    if tuned_xgb == True:

        try:
            from sklearn.model_selection import GridSearchCV
        except ImportError as e:
            raise ImportError("sklearn is required to perform this.")

        gbm = xgb.XGBRegressor()
        grid_mse = GridSearchCV(
            estimator=gbm,
            param_grid=gbm_hyperparam_grid,
            scoring="neg_mean_squared_error",
            cv=4,  # 4-fold crossvalidation
            verbose=3,  # Prints the grid search profile
            n_jobs=-1,
        )  # Process the GridSearch in parallel all cores availables

        # Fit the grid to the data
        grid_mse.fit(X, y)

        best_params = grid_mse.best_params_
        best_params["objective"] = "reg:linear"

        # Create the DMatrix
        xgb_dmatrix = xgb.DMatrix(X, y)

        # Train the model from the best parameters of the grid search
        xg_reg = xgb.train(params=best_params, dtrain=xgb_dmatrix)

    # Build explainer and fit Shapley's values (https://github.com/slundberg/shap)
    explainer = shap.TreeExplainer(xg_reg, feature_dependence="independent")
    shap_values = explainer.shap_values(X)
    weights_from_xgb = shap_values.mean(axis=0)  # This is already sorted by pixel Type

    weights = np.zeros(n_pixels_option_values)
    weights[codes] = list(weights_from_xgb)  # Convert to list a dict_values

    if ReLU:
        weights = np.where(weights < 0, 0, weights)

    return weights


def create_lon_lat(out_img, out_transform):

    """Function that returns a tuple of longitudes and latitudes from numpy
    array and Affline.

    Parameters
    ----------

    out_img       : numpy array generated by the mask function (first tuple element)

    out_transform : Affline transformation generated by the mask function (second tuple element)

    Notes
    -----

    Inside the inner loop there is an important thing to consider which is that the dimensions
    out_img is (1, lat, lon), whereas the Affline transformation gives (lon, lat) (or, accordingly
    to their documentation is (x,y)) that is why it is swapped the indexes.
    Also, note the repetition in the indexes of (j, j) and (i, i) inside the inner loop.
    """

    lons = np.empty(shape=(1, out_img.shape[1], out_img.shape[2]))
    lats = np.empty(shape=(1, out_img.shape[1], out_img.shape[2]))

    for i in range(out_img.shape[1]):
        for j in range(out_img.shape[2]):
            lons[0][i][j] = (out_transform * (j, j))[0]  # Only index j.
            lats[0][i][j] = (out_transform * (i, i))[1]  # Only indexes of longitudes!

    # Return two arrays: one is the longitudes of each pixel and one is the latitudes of each pixel

    return lons, lats


def _create_non_zero_population_by_pixels_locations(
    geodataframe, raster, pop_string, weights=None, force_crs_match=True
):

    """Function that returns the actual population of each pixel from a given
    geodataframe and variable.

    geodataframe       : a geopandas dataframe that it is contained in the raster

    raster             : the raster used from rasterio

    pop_string         : a string of the column name of the geodataframe that the estimation will be made

    weights            : vector of weights in each position of the pixel values according 'return_weights_from_regression' function. This must be provided by the user.

    force_crs_match    : bool. Default is True.
                         Wheter the Coordinate Reference System (CRS) of the polygon will be reprojected to the CRS of the raster file.
                         It is recommended to let this argument as True.
    """

    _check_presence_of_crs(geodataframe)

    if not force_crs_match:
        warnings.warn(
            "The polygon is not being reprojected. The clipping might be being performing on unmatching polygon to the raster."
        )

    else:
        with rasterio.open(raster) as raster:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                geodataframe_projected = geodataframe.to_crs(crs=raster.crs.data)
            result_pops_array = np.array([])
            result_lons_array = np.array([])
            result_lats_array = np.array([])

            pbar = tqdm(
                total=len(geodataframe_projected),
                desc="Estimating population per pixel",
            )

            for line_index in range(len(geodataframe_projected)):
                polygon_projected = geodataframe_projected.iloc[[line_index]]

                coords = getFeatures(polygon_projected)

                out_img, out_transform = mask(dataset=raster, shapes=coords, crop=True)

                """Calculating the population for each pixel"""
                trans_numpy = weights[out_img]  # Pixel population from regression
                orig_estimate = polygon_projected[
                    pop_string
                ]  # Original Population Value of The polygon
                correction_term = orig_estimate / trans_numpy.sum()
                final_pop_numpy_pre = trans_numpy * np.array(correction_term)

                flatten_final_pop_numpy_pre = np.ndarray.flatten(final_pop_numpy_pre)

                non_zero_pop_index = np.where(flatten_final_pop_numpy_pre != 0)

                final_pop_numpy = flatten_final_pop_numpy_pre[non_zero_pop_index]

                """Retrieving location of each pixel"""
                lons, lats = create_lon_lat(out_img, out_transform)

                final_lons = np.ndarray.flatten(lons)[non_zero_pop_index]
                final_lats = np.ndarray.flatten(lats)[non_zero_pop_index]

                """Append all flattens numpy arrays"""
                result_pops_array = np.append(result_pops_array, final_pop_numpy)
                result_lons_array = np.append(result_lons_array, final_lons)
                result_lats_array = np.append(result_lats_array, final_lats)

                pbar.update(1)

            data = {
                "pop_value": result_pops_array,
                "lons": result_lons_array.round().astype(int).tolist(),
                "lats": result_lats_array.round().astype(int).tolist(),
            }

            corresp = pd.DataFrame.from_dict(data)
        pbar.close()

    return corresp


def _calculate_interpolated_polygon_population_from_correspondence_table(
    polygon, raster, corresp_table, force_crs_match=True, na_value=255
):

    """Function that returns the interpolated population of a given polygon
    according to a correspondence table previous built.

    Parameters
    ----------

    polygon         : polygon for the profile (it can be a row of a geopandas)

    raster          : the associated raster (from rasterio.open)

    corresp_table   : correspondence table that has the interpolated population for each pixel. This object is created with the function 'create_non_zero_population_by_pixels_locations'.

    force_crs_match : bool. Default is True.
                      Wheter the Coordinate Reference System (CRS) of the polygon will be reprojected to the CRS of the raster file.
                      It is recommended to let this argument as True.

    na_value        : int. Default is 255.
                      The number which is considered to be 'Not a Number' (NaN) in the raster pixel values.

    Notes
    -----
    When you clip a specific polygon, there are pixels that lie beyond the polygon extent, because the clipping is rectangular.
    Therefore, the population could be wrongly summed from another spatial unit.
    The solution is to build a pandas and filter the pixels different than 255. This is done during the construction of the polygon summary for the resulting population of this function.
    """

    _check_presence_of_crs(polygon)

    if force_crs_match:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            polygon_projected = polygon.to_crs(crs=raster.crs.data)
    else:
        warnings.warn(
            "The polygon is not being reprojected. The clipping might be being performing on unmatching polygon to the raster."
        )

    coords = getFeatures(polygon_projected)
    out_img, out_transform = mask(dataset=raster, shapes=coords, crop=True)
    lons, lats = create_lon_lat(out_img, out_transform)
    data = {
        "lons": np.ndarray.flatten(lons).round().astype(int).tolist(),
        "lats": np.ndarray.flatten(lats).round().astype(int).tolist(),
        "pixel_value": np.ndarray.flatten(out_img),
    }
    polygon_summary_full = pd.DataFrame.from_dict(data)

    # Remove pixels of the polygon that do not belong to the spatial unit, but might be from another one
    polygon_summary = polygon_summary_full[polygon_summary_full.pixel_value != na_value]

    merged_polygon = corresp_table.merge(polygon_summary, on=["lons", "lats"])

    pop = merged_polygon["pop_value"].sum()

    return pop


def _calculate_interpolated_population_from_correspondence_table(
    geodataframe, raster, corresp_table, variable_name=None, force_crs_match=True
):

    """Function that returns the interpolated population of an entire geopandas
    according to a correspondence table previous built.

    Parameters
    ----------

    geodataframe    : a GeoDataFrame from geopandas

    raster          : the associated raster (from rasterio.open)

    corresp_table   : correspondence table that has the interpolated population for each pixel. This object is created with the function 'create_non_zero_population_by_pixels_locations'.

    force_crs_match : bool. Default is True.
                      Wheter the Coordinate Reference System (CRS) of the polygon will be reprojected to the CRS of the raster file.
                      It is recommended to let this argument as True.

    Notes
    -----
    This function returns the same GeoDataFrame used as input with the addition of a new variable called 'interpolated_population', which is the resulting population.
    """

    _check_presence_of_crs(geodataframe)

    final_geodataframe = geodataframe.copy()[[geodataframe.geometry.name]]
    pop_final = np.empty(len(geodataframe))
    with rasterio.open(raster) as raster:

        pbar = tqdm(total=len(geodataframe), desc="Estimating target polygon values")

        for line_index in range(len(geodataframe)):
            polygon = geodataframe.iloc[[line_index]]
            pop_aux = _calculate_interpolated_polygon_population_from_correspondence_table(
                polygon, raster, corresp_table, force_crs_match
            )
            pop_final[line_index] = pop_aux

            pbar.update(1)

        pbar.close()
        final_geodataframe[variable_name] = pop_final

    return final_geodataframe


def subset_gdf_polygons_from_raster(geodataframe, raster, force_crs_match=True):
    """Function that returns only the polygons that actually have some
    intersection with a given raster.

    Parameters
    ----------

    geodataframe    : a GeoDataFrame from geopandas

    raster          : the associated raster (from rasterio.open)

    force_crs_match : bool. Default is True.
                      Wheter the Coordinate Reference System (CRS) of the polygon will be reprojected to the CRS of the raster file.
                      It is recommended to let this argument as True.
    """

    _check_presence_of_crs(geodataframe)
    if force_crs_match:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reprojected_gdf = geodataframe.to_crs(crs=raster.crs.data)
    else:
        warnings.warn(
            "The geodataframe is not being reprojected. The clipping might be being performing on unmatching polygon to the raster."
        )

    # has_intersection is a boolean vector: True if the polygon has some overlay with raster, False otherwise
    has_intersection = []

    pbar = tqdm(total=len(reprojected_gdf), desc="Subsetting polygons")
    for i in list(range(len(reprojected_gdf))):
        pbar.update(1)
        coords = getFeatures(reprojected_gdf.iloc[[i]])
        try:
            out_img = mask(dataset=raster, shapes=coords, crop=True)[0]
            has_intersection.append(True)
        except:
            has_intersection.append(False)
    pbar.close()

    overlayed_subset_gdf = reprojected_gdf.iloc[has_intersection]
    overlayed_subset_gdf = overlayed_subset_gdf.set_geometry(
        overlayed_subset_gdf.geometry.name
    )

    return overlayed_subset_gdf
