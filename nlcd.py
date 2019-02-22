"""
Functions to perform Population Interpolation using the National Land Cover Data (NLCD).
This is a generic framework that can be used to distribute population more accurately in harmonized spatial structures. 

Inspired by Reibel, Michael, and Aditya Agrawal. "Areal interpolation of population counts using pre-classified land cover data." Population Research and Policy Review 26.5-6 (2007): 619-633.
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import json
import pandas as pd
import geopandas as gpd
import numpy as np
from rasterio.mask import mask

import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson

import xgboost as xgb
from sklearn.model_selection import GridSearchCV


__all__ = ['getFeatures', 
           'return_area_profile', 
           'append_profile_in_gdf', 
           'return_weights_from_regression',
           'return_weights_from_xgboost',
           'create_lon_lat',
           'create_non_zero_population_by_pixels_locations',
           'calculate_interpolated_polygon_population_from_correspondence_NLCD_table',
           'calculate_interpolated_population_from_correspondence_NLCD_table']


def getFeatures(gdf):
    
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them
    
    Notes
    -----
    This function was obtained at https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html.
    
    """
    
    return [json.loads(gdf.to_json())['features'][0]['geometry']]









def return_area_profile(polygon, raster, polygon_crs = {'init': 'epsg:3395'}):
    
    """Function that counts the amount of pixel types it is inside a polygon within a given raster
    
    Parameters
    ----------
    
    polygon     : polygon for the profile (it can be a row of a geopandas)
    
    raster      : the associated raster (from rasterio.open)
    
    polygon_crs : the original crs code given by a dict of the polygon (this is gonna be projected later, as this must be the same as the raster). 
                  This argument is only to avoid passing a polygon without crs and, therefore, raising an error.

    """
    
    polygon.crs = polygon_crs
    polygon_projected = polygon.to_crs(crs = raster.crs.data)
    
    coords = getFeatures(polygon_projected)
    out_img = mask(dataset = raster, shapes = coords, crop = True)[0]
    
    x = np.ndarray.flatten(out_img)
    y = np.bincount(x)
    ii = np.nonzero(y)[0]

    profile = pd.DataFrame.from_dict(dict(zip(np.core.defchararray.add('Type_', ii.astype(str)),y[ii].reshape(len(y[ii]),1)))) # pandas like
    
    polygon_with_profile = pd.concat([polygon.reset_index(drop=True), profile], axis = 1) # Appends in the end

    return polygon_with_profile






def append_profile_in_gdf(geodataframe, raster):
    
    """Function that appends the columns of the profile in a geopandas according to a given raster
    
    geodataframe : a GeoDataFrame from geopandas. The variables of the profile will be appended in this data.
    
    raster       : the associated raster (from rasterio.open)
    
    """
    
    final_geodata = gpd.GeoDataFrame()
    
    for i in range(len(geodataframe)):
        
        aux = return_area_profile(geodataframe.iloc[[i]], raster = raster)
        final_geodata = pd.concat([final_geodata.reset_index(drop = True), aux], axis = 0, sort = False) # sort = False means that the profile will be appended in the end of the result
        print('Polygon profile {} appended out of {}'.format(i + 1, len(geodataframe)), end = "\r")
    
    return final_geodata





def return_weights_from_regression(geodataframe, raster, pop_string, codes = [21, 22, 23, 24], likelihood = 'Poisson', n_pixels_option_values = 256):
    
    """Function that returns the weights of each land type according to NLCD types/codes
    
    Parameters
    ----------
    
    geodataframe           : a geopandas geoDataFrame used to build regression
    
    raster                 : a raster (from rasterio.open) that has the types of each pixel in the geodataframe
    
    pop_string             : the name of the variable on geodataframe that the regression shall be conducted
    
    codes                  : an integer list of codes values that should be considered as 'populated' from the National Land Cover Database (NLCD).
                             The description of each code can be found here: https://www.mrlc.gov/sites/default/files/metadata/landcover.html
                             The default is 21 (Developed, Open Space), 22 (Developed, Low Intensity), 23 (Developed, Medium Intensity) and 24 (Developed, High Intensity).
                             
    likelihood             : the likelihood assumed for the dependent variable (population). 
                             It can be 'Poisson' or 'Gaussian'. 
                             With the 'Poisson' a Generalized Linear Model with log as link function will be fitted and 'Gaussian' an ordinary least squares will be fitted. 
                             
    n_pixels_option_values : number of options of the pixel values of rasterior. Default is 256.
    
    Notes
    -----
    1) The formula uses a substring called 'Type_' before the code number due to the 'append_profile_in_gdf' function.
    2) The pixel value, usually, ranges from 0 to 255. That is why the default of 'n_pixels_option_values' is 256.
    
    """
    
    if (255 in codes):
        raise ValueError('codes should not assume the value 255.')
    
    if not likelihood in ['Poisson', 'Gaussian']:
        raise ValueError('likelihood must one of \'Poisson\', \'Gaussian\'')
    
    print('Appending profile...')
    profiled_df = append_profile_in_gdf(geodataframe[['geometry', pop_string]], raster) # Use only two columns to build the weights (this avoids error, if the original dataset has already types appended on it).
    print('Append profile: Done.')
    
    # If the list is unsorted, the codes will be sorted to guarantee that the position of the weights will match
    codes.sort()
    
    # Formula WITHOUT intercept
    str_codes = [str(i) for i in codes] 
    formula_string = pop_string + ' ~ -1 + ' + " + ".join(['Type_' + s for s in str_codes])
    
    print('Starting to fit regression...')
    if (likelihood == 'Poisson'):
        results = smf.glm(formula_string, data = profiled_df, family = Poisson()).fit()
        
    if (likelihood == 'Gaussian'):
        results = smf.ols(formula_string, data = profiled_df).fit()
    
    weights = np.zeros(n_pixels_option_values)
    weights[codes] = results.params
    
    return weights








def return_weights_from_xgboost(geodataframe, 
                                raster, 
                                pop_string, 
                                codes = [21, 22, 23, 24], 
                                n_pixels_option_values = 256, 
                                tuned_xgb = False, 
                                gbm_hyperparam_grid = {'learning_rate': [0.001,0.01, 0.9],
                                                       'n_estimators': [200],
                                                       'subsample': [0.3, 0.5],
                                                       'max_depth': [4, 5, 6],
                                                       'num_boosting_rounds': [10, 20]}):
    
    """Function that returns the weights of each land type according to NLCD types/codes given by Extreme Gradient Boost model (XGBoost)
    
    Parameters
    ----------
    
    geodataframe           : a geopandas geoDataFrame used to build regression
    
    raster                 : a raster (from rasterio.open) that has the types of each pixel in the geodataframe
    
    pop_string             : the name of the variable on geodataframe that the regression shall be conducted
    
    codes                  : an integer list of codes values that should be considered as 'populated' from the National Land Cover Database (NLCD).
                             The description of each code can be found here: https://www.mrlc.gov/sites/default/files/metadata/landcover.html
                             The default is 21 (Developed, Open Space), 22 (Developed, Low Intensity), 23 (Developed, Medium Intensity) and 24 (Developed, High Intensity).
                             
    n_pixels_option_values : number of options of the pixel values of rasterior. Default is 256.
    
    tuned_xgb              : bool. Default is False.
                             If True the XGBoost model will be tuned making a grid search using gbm_hyperparam_grid dictionary a picking the best model in terms of mean squared error with some pre-defined number of cross-validation.
                             Otherwise, the XGBoost model is fitted with default values of xgboost.train function from xgboost Python library.
                             
    gbm_hyperparam_grid    : a dictionary that represent the grid for the grid search of XGBoost
    
    Notes
    -----
    1) The formula uses a substring called 'Type_' before the code number due to the 'append_profile_in_gdf' function.
    2) The pixel value, usually, ranges from 0 to 255. That is why the default of 'n_pixels_option_values' is 256.
    3) The returning weights represent the feature importance as the number of times a feature is used to split the data across all trees. That is, using the method .get_score(importance_type = 'weight')
    
    """
    
    if (255 in codes):
        raise ValueError('codes should not assume the value 255.')
    
    print('Appending profile...')
    profiled_df = append_profile_in_gdf(geodataframe[['geometry', pop_string]], raster) # Use only two columns to build the weights (this avoids error, if the original dataset has already types appended on it).
    print('Append profile: Done.')
    
    # If the list is unsorted, the codes will be sorted to guarantee that the position of the weights will match
    codes.sort()

    str_codes = [str(i) for i in codes] 
    feature_names = ['Type_' + s for s in str_codes]
    
    y = profiled_df[pop_string]
    X = profiled_df[feature_names]
    
    print('Starting to fit XGBoost...')
    if (tuned_xgb == False):
        
        # Create the DMatrix
        xgb_dmatrix = xgb.DMatrix(X, y)

        # Create the parameter dictionary
        params = {"objective":"reg:linear"}

        # Train the model
        xg_reg = xgb.train(params = params, dtrain = xgb_dmatrix)
        
    if (tuned_xgb == True):
        
        gbm = xgb.XGBRegressor()
        grid_mse = GridSearchCV(estimator = gbm,
                                param_grid = gbm_hyperparam_grid, 
                                scoring = 'neg_mean_squared_error', 
                                cv = 4,      # 4-fold crossvalidation
                                verbose = 3) # Prints the grid search profile

        # Fit the grid to the data
        grid_mse.fit(X, y)
        
        best_params = grid_mse.best_params_
        best_params["objective"] = "reg:linear"
        
        # Create the DMatrix
        xgb_dmatrix = xgb.DMatrix(X, y)
        
        # Train the model from the best parameters of the grid search
        xg_reg = xgb.train(params = best_params, dtrain = xgb_dmatrix)
    
    weights_from_xgb_object = xg_reg.get_score(importance_type = 'weight')
    weights_from_xgb = dict(sorted(weights_from_xgb_object.items())).values() # This orders according to type order string!
    
    weights = np.zeros(n_pixels_option_values)
    weights[codes] = list(weights_from_xgb) # Convert to list a dict_values
    
    return weights









def create_lon_lat(out_img, out_transform):
    
    '''Function that returns a tuple of longitudes and latitudes from numpy array and Affline
    
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
    
    '''
    
    lons = np.empty(shape = (1, out_img.shape[1], out_img.shape[2]))
    lats = np.empty(shape = (1, out_img.shape[1], out_img.shape[2]))
    
    for i in range(out_img.shape[1]):
        for j in range(out_img.shape[2]):
            lons[0][i][j] = (out_transform * (j, j))[0] # Only index j. 
            lats[0][i][j] = (out_transform * (i, i))[1] # Only indexes of longitudes!
            
    # Return two arrays: one is the longitudes of each pixel and one is the latitudes of each pixel
    
    return lons, lats






def create_non_zero_population_by_pixels_locations(geodataframe, raster, pop_string, weights = None, geodataframe_crs = {'init': 'epsg:3395'}, save_polygon_index = False):
    
    '''Function that returns the actual population of each pixel from a given geodataframe and variable.
    
    geodataframe       : a geopandas dataframe that it is contained in the raster
    
    raster             : the raster used from rasterio
    
    pop_string         : a string of the column name of the geodataframe that the estimation will be made
    
    weights            : vector of weights in each position of the pixel values according 'return_weights_from_regression' function. This must be provided by the user.
    
    geodataframe_crs   : projection of the dataframe (this is gonna be reprojeced inside the loop, as this must be the same as the raster).
                         This argument is only to avoid passing a polygon without crs and, therefore, raising an error.
                         
    save_polygon_index : bool. If True, it saves the polygon row index of each pixel. 
    
    '''
    
    geodataframe.crs = geodataframe_crs # This is gonna be reprojected inside the loop

    result_pops_array = np.array([])
    result_lons_array = np.array([])
    result_lats_array = np.array([])
    
    if (save_polygon_index == True):
        result_poly_index = np.array([])

    for line_index in range(len(geodataframe)):
        polygon = geodataframe.iloc[[line_index]]
        polygon_projected = polygon.to_crs(crs = raster.crs.data)
        
        coords = getFeatures(polygon_projected)
        
        out_img, out_transform = mask(dataset = raster, shapes = coords, crop = True)
        
        '''Calculating the population for each pixel'''
        trans_numpy = weights[out_img] # Pixel population from regression
        orig_estimate = polygon[pop_string]         # Original Population Value of The polygon
        correction_term = orig_estimate / trans_numpy.sum()
        final_pop_numpy_pre = trans_numpy * np.array(correction_term)
        
        flatten_final_pop_numpy_pre = np.ndarray.flatten(final_pop_numpy_pre)
        
        non_zero_pop_index = np.where(flatten_final_pop_numpy_pre != 0)
        
        final_pop_numpy = flatten_final_pop_numpy_pre[non_zero_pop_index]
        
        '''Retrieving location of each pixel'''
        lons, lats = create_lon_lat(out_img, out_transform)
        
        final_lons = np.ndarray.flatten(lons)[non_zero_pop_index]
        final_lats = np.ndarray.flatten(lats)[non_zero_pop_index]
        
        result_poly_index_aux = np.full(len(np.ndarray.flatten(lons)), line_index)[non_zero_pop_index]
        
        '''Append all flattens numpy arrays'''
        result_pops_array = np.append(result_pops_array, final_pop_numpy)
        result_lons_array = np.append(result_lons_array, final_lons)
        result_lats_array = np.append(result_lats_array, final_lats)
        
        if (save_polygon_index == True):
            result_poly_index = np.append(result_poly_index, result_poly_index_aux)
        
        print('Polygon {} processed out of {}'.format(line_index + 1, len(geodataframe)), end = "\r")
        
    if (save_polygon_index == False):
        data = {'pop_value': result_pops_array,
                'lons': result_lons_array.round().astype(int).tolist(), 
                'lats': result_lats_array.round().astype(int).tolist()}        

    if (save_polygon_index == True):
        data = {'pop_value': result_pops_array,
                'lons': result_lons_array.round().astype(int).tolist(), 
                'lats': result_lats_array.round().astype(int).tolist(),
                'polygon': result_poly_index}
        
    corresp = pd.DataFrame.from_dict(data)
        
    return corresp









def calculate_interpolated_polygon_population_from_correspondence_NLCD_table(polygon, raster, corresp_table):
    
    """Function that returns the interpolated population of a given polygon according to a correspondence table previous built
    
    Parameters
    ----------
    
    polygon       : polygon for the profile (it can be a row of a geopandas)
    
    raster        : the associated raster (from rasterio.open)
    
    corresp_table : correspondence table that has the interpolated population for each pixel. This object is created with the function 'create_non_zero_population_by_pixels_locations'.
    
    Notes
    -----
    When you clip a specific polygon, there are pixels that lie beyond the polygon extent, because the clipping is rectangular. 
    Therefore, the population could be wrongly summed from another spatial unit.
    The solution is to build a pandas and filter the pixels different than 255. This is done during the construction of the polygon summary for the resulting population of this function.
    
    """

    polygon.crs = {'init': 'epsg:3395'}
    polygon_projected = polygon.to_crs(crs = raster.crs.data)

    coords = getFeatures(polygon_projected)
    out_img, out_transform = mask(dataset = raster, shapes = coords, crop = True)
    lons, lats = create_lon_lat(out_img, out_transform)
    data = {'lons': np.ndarray.flatten(lons).round().astype(int).tolist(), 
            'lats': np.ndarray.flatten(lats).round().astype(int).tolist(),
            'pixel_value': np.ndarray.flatten(out_img)}
    polygon_summary_full = pd.DataFrame.from_dict(data)
    
    # Remove pixels of the polygon that do not belong to the spatial unit, but might be from another one
    polygon_summary = polygon_summary_full[polygon_summary_full.pixel_value != 255]
    
    merged_polygon = corresp_table.merge(polygon_summary, on = ['lons', 'lats'])
    
    pop = merged_polygon['pop_value'].sum()
    
    return pop








def calculate_interpolated_population_from_correspondence_NLCD_table(geodataframe, raster, corresp_table):
    
    """Function that returns the interpolated population of an entire geopandas according to a correspondence table previous built
    
    Parameters
    ----------
    
    geodataframe  : a GeoDataFrame from geopandas
    
    raster        : the associated raster (from rasterio.open)
    
    corresp_table : correspondence table that has the interpolated population for each pixel. This object is created with the function 'create_non_zero_population_by_pixels_locations'.
    
    Notes
    -----
    This function returns the same GeoDataFrame used as input with the addition of a new variable called 'interpolated_population', which is the resulting population.
    
    """
    
    final_geodataframe = geodataframe.copy()
    pop_final = np.empty(len(geodataframe))
    
    for line_index in range(len(geodataframe)):
        polygon = geodataframe.iloc[[line_index]]
        pop_aux = calculate_interpolated_polygon_population_from_correspondence_NLCD_table(polygon, raster, corresp_table)
        pop_final[line_index] = pop_aux
        
        print('Polygon {} processed out of {}'.format(line_index + 1, len(geodataframe)), end = "\r")
        
    final_geodataframe['interpolated_population'] = pop_final
        
    return final_geodataframe