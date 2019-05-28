"""
Functions to perform fast Population Interpolation using the National Land Cover Data (NLCD) using Scanlines.

This is a generic framework that can be used to distribute population more accurately in harmonized spatial structures. 

Inspired by Reibel, Michael, and Aditya Agrawal. "Areal interpolation of population counts using pre-classified land cover data." Population Research and Policy Review 26.5-6 (2007): 619-633.
The advantage of this approach is the use of the Scanlines proposed by "Eldawy, Ahmed, et al. "Large Scale Analytics of Vector + Raster Big Spatial Data." Proceedings of the 25th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems. ACM, 2017."

Note: This study was financed by the National Science Foundation (NSF) (Award #1831615). Renan X. Cortes is also grateful for the support of the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Process number 88881.170553/2018-01
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Samriddhi Singla <ssing068@ucr.edu>, Elijah Knaap <elijah.knaap@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Ahmed Eldawy <eldawy@ucr.com>"


import pandas as pd
import geopandas as gpd
import numpy as np
from rasterio.mask import mask

import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson, Gaussian

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# An option worth to consider to install `shap` (that could avoid dependency conflict) is direct from github:
# pip install git+https://github.com/slundberg/shap.git
import shap

from tobler.area_weighted import _check_crs
from tobler.vectorized_raster_interpolation import _check_presence_of_crs

import os
import tempfile
from subprocess import run
from shapely import wkt
import rasterio
import time

# Function that will return all count of all pixel types
def scanlines_count_pixels(source_gdf, raster_path):
    
    """Function that generates the count of all pixel types in a raster of a given set of polygons using scanlines
    
    Parameters
    ----------
    
    source_gdf      : geopandas GeoDataFrame with geometry column of polygon type for the source set of polygons desired.
    
    raster_path     : the path to the associated raster image.

    """
    
    t0_aux = time.time()
    
    _check_presence_of_crs(source_gdf)
    
    print('Opening raster metadata...')
    raster = rasterio.open(raster_path)
    
    print('Matching both crs\'s (reprojecting source_gdf to raster)...')
    source_gdf = source_gdf.to_crs(crs = raster.crs.data)
    
    # Check if Operational System is Windows
    if os.name == 'nt':
        sep_cmd = ';'
        sep_dir = '\\'
    else:
        sep_cmd = ':'
        sep_dir = '/'

    if ('geometry' not in source_gdf.columns):
        source_gdf['geometry'] = source_gdf[source_gdf._geometry_column_name]
        source_gdf = source_gdf.drop([source_gdf._geometry_column_name], axis = 1)
        source_gdf = source_gdf.set_geometry('geometry')
    
    # Create Temporary Directory
    source_gdf_temp_dir = tempfile.mkdtemp()
    
    # parquet like internal file
    print('Starting to create well-known text (wkt) of geometries...')
    source_gdf['geometry_wkt'] = source_gdf['geometry'].apply(lambda x: x.wkt) # Create well-know text (raw text) for the geometry column
    source_gdf = source_gdf.drop(['geometry'], axis = 1)
    source_gdf_temp_file_name = source_gdf_temp_dir + '{}source_gdf_temp.parquet'.format(sep_dir)
    
    # Just extract the useful column for optimization
    source_gdf = source_gdf[['geometry_wkt']]
    
    print('Starting to convert the GeoDataFrame to a temporary file...')
    source_gdf.to_parquet(source_gdf_temp_file_name)
    
    cmd = "java -client -cp dependency{}*{}ucrspatial-6.0-SNAPSHOT.jar histogram {} {}".format(sep_dir,
                                                                                               sep_cmd, 
                                                                                               raster_path, 
                                                                                               source_gdf_temp_file_name)
    
    t1_aux = time.time()
    
    print('Time of preparation before scanline (in seconds): {}'.format(t1_aux - t0_aux))
    

    
    print('Starting to perform the scanline...')
    t0_aux = time.time()
    run(cmd, shell = True, check = True) # Will generate an parquet for output: histogram.parquet
    t1_aux = time.time()
    print('Scanline: Done.')
    print('Time of scanline itself (in seconds): {}'.format(t1_aux - t0_aux))
    
    os.remove(source_gdf_temp_file_name)
    os.rmdir(source_gdf_temp_dir)
    
    profile_df = pd.read_parquet("histogram.parquet")
    
    os.remove("histogram.parquet")
    
    return profile_df




# Function that will interpolate the population for given set of weights and correction terms
def scanlines_interpolate(target_gdf, source_CTs, weights_long, raster_path):
    
    """Function that generates the interpolated values using scanlines with a given set of weights and Correction Terms using scanlines
    
    Parameters
    ----------
    
    target_gdf      : geopandas GeoDataFrame with geometry column of polygon type for the target set of polygons desired.
    
    source_CTs      : geopandas GeoDataFrame with the Correction Terms for source polygons.
    
    weights_long    : a numpy array with the weights for all land types in the raster.
    
    raster_path     : the path to the associated raster image.

    """
    
    t0_aux = time.time()
    
    _check_presence_of_crs(target_gdf)
    _check_presence_of_crs(source_CTs)
    
    if _check_crs(source_CTs, target_gdf):
        pass
    else:
        return None

    print('Opening raster metadata...')
    raster = rasterio.open(raster_path)
    
    print('Matching both crs\'s (reprojecting source_CTs to raster)...')
    source_CTs = source_CTs.to_crs(crs = raster.crs.data)
    print('...reprojecting target_gdf to raster)...')
    target_gdf = target_gdf.to_crs(crs = raster.crs.data)
    
    
    # Check if Operational System is Windows
    if os.name == 'nt':
        sep_cmd = ';'
        sep_dir = '\\'
    else:
        sep_cmd = ':'
        sep_dir = '/'
    
    if ('geometry' not in target_gdf.columns):
        target_gdf['geometry'] = target_gdf[target_gdf._geometry_column_name]
        target_gdf = target_gdf.drop([target_gdf._geometry_column_name], axis = 1)
        target_gdf = target_gdf.set_geometry('geometry')
    
    if ('geometry' not in source_CTs.columns):
        source_CTs['geometry'] = source_CTs[source_CTs._geometry_column_name]
        source_CTs = source_CTs.drop([source_CTs._geometry_column_name], axis = 1)
        source_CTs = source_CTs.set_geometry('geometry')
    
    
    # Create a temporary directory for ALL input files
    temp_dir = tempfile.mkdtemp()
    
    # parquet like internal file
    print('Starting to create well-known text (wkt) of geometries...')
    target_gdf['geometry_wkt'] = target_gdf['geometry'].apply(lambda x: x.wkt) # Create well-know text (raw text) for the geometry column
    target_gdf = target_gdf.drop(['geometry'], axis = 1)
    target_gdf_temp_file_name = temp_dir + '{}target_gdf_temp.parquet'.format(sep_dir)
    
    # Just extract the useful column for optimization
    target_gdf = target_gdf[['geometry_wkt']]
    
    print('Starting to convert the GeoDataFrame to a temporary file...')
    target_gdf.to_parquet(target_gdf_temp_file_name)
    
    # parquet like internal file
    print('Source CT: Starting to create well-known text (wkt) of geometries...')
    source_CTs['geometry_wkt'] = source_CTs['geometry'].apply(lambda x: x.wkt) # Create well-know text (raw text) for the geometry column
    source_CTs = source_CTs.drop(['geometry'], axis = 1)
    source_CTs_temp_file_name = temp_dir + '{}source_CTs_temp.parquet'.format(sep_dir)
    
    # Just extract the useful column for optimization
    # For source we need also the Correction Terms!
    source_CTs = source_CTs[['geometry_wkt', 'CT']]
    
    print('Starting to convert the GeoDataFrame to a temporary file...')
    source_CTs.to_parquet(source_CTs_temp_file_name)
    
    
    weights_temp_file_name = temp_dir + '{}input_weights.csv'.format(sep_dir)
    np.savetxt(weights_temp_file_name, weights_long, delimiter=",", header = 'weights', comments='')
    
    cmd = "java -cp dependency/*{}ucrspatial-6.0-SNAPSHOT.jar interpolate {} {} {} {}".format(sep_cmd, 
                                                                                              raster_path,
                                                                                              source_CTs_temp_file_name,
                                                                                              target_gdf_temp_file_name,
                                                                                              weights_temp_file_name)
    
    t1_aux = time.time()
    
    print('Time of preparation before scanline (in seconds): {}'.format(t1_aux - t0_aux))
    
    print('Starting to perform the scanline...')
    t0_aux = time.time()
    run(cmd, shell=True, check=True) # Will generate an parquet for output: interpolate.parquet
    t1_aux = time.time()
    print('Scanline: Done.')
    print('Time of scanline itself (in seconds): {}'.format(t1_aux - t0_aux))
    
    os.remove(target_gdf_temp_file_name)
    os.remove(source_CTs_temp_file_name)
    os.remove(weights_temp_file_name)
    
    os.rmdir(temp_dir)
    
    interpolated_df = pd.read_parquet("interpolate.parquet")
    
    os.remove("interpolate.parquet")
    
    return interpolated_df




def scanline_harmonization(source_gdf, 
                           target_gdf, 
                           pop_string, 
                           raster_path, 
                           auxiliary_type = 'nlcd',
                           regression_method = 'Poisson',
                           codes = [21, 22, 23, 24],
                           n_pixels_option_values = 256,
                           ReLU = True,
                           **kwargs):
    
    """Function that generates an interpolated population using scanlines with the entire pipeline.
    
    Parameters
    ----------
    
    source_gdf             : geopandas GeoDataFrame with geometry column of polygon type for the source set of polygons desired.
    
    target_gdf             : geopandas GeoDataFrame with geometry column of polygon type for the target set of polygons desired.
    
    pop_string             : the name of the variable on geodataframe that the interpolation shall be conducted.
    
    raster_path            : the path to the associated raster image.
    
    auxiliary_type         : string. The type of the auxiliary variable for the desired method of interpolation. Default is 'nlcd' for the National Land Cover Dataset. 
    
    regression_method      : the method used to estimate the weights of each land type and population. Default is "Poisson".
                        
        "Poisson"  : performs Generalized Linear Model with a Poisson likelihood with log-link function.
        "Gaussian" : ordinary least squares will be fitted.
        "XGBoost"  : an Extreme Gradient Boosting regression will be fitted and the weights will be extracted from the Shapelys value from each land type.

    codes                  : an integer list of codes values that should be considered as 'populated' for the raster file. See (1) in notes.
    
    n_pixels_option_values : number of options of the pixel values of rasterior. Default is 256.
    
    ReLU                   : bool. Default is True.
                             Wheter the Rectified Linear Units (ReLU) transformation will be used to avoid negative weights for the land types.
                             
    **kwargs               : additional arguments that can be passed to internal functions.
                             Currently `tuned_xgb` or `gbm_hyperparam_grid` can be passed to internal XGBoost approach.

    Notes
    -----

    1) Since this was inspired using the National Land Cover Database (NLCD), it is established some default values for this argument.
       The default is 21 (Developed, Open Space), 22 (Developed, Low Intensity), 23 (Developed, Medium Intensity) and 24 (Developed, High Intensity).
       The description of each code for NLCD can be found here: https://www.mrlc.gov/sites/default/files/metadata/landcover.html    
    
    """
    
    
    print('INITIALIZING FIRST SCANLINES')
    profiled_df_pre = scanlines_count_pixels(source_gdf, raster_path)
    
    profiled_df = pd.concat([source_gdf.reset_index(), profiled_df_pre], axis = 1)
    
    codes.sort()

    str_codes = [str(i) for i in codes] 
    formula_string = pop_string + ' ~ -1 + ' + " + ".join(['Type_' + s for s in str_codes])

    if (regression_method == 'Poisson'):
        results = smf.glm(formula_string, data = profiled_df, family = Poisson()).fit()
        weights = np.array(results.params)
    
    if (regression_method == 'Gaussian'):
        results = smf.glm(formula_string, data = profiled_df, family = Gaussian()).fit()
        weights = np.array(results.params)
        
    if (regression_method == 'XGBoost'):
        weights = _return_xgboost_weights(profiled_df, pop_string, str_codes, **kwargs)
        
    if ReLU:
        weights = np.where(weights < 0, 0, weights)
    
    # Correction Term (CT)
    profiled_df['denominator'] = (np.array(profiled_df[['Type_' + s for s in str_codes]]) * weights).sum(axis = 1)
    profiled_df['CT'] = np.nan_to_num(profiled_df[pop_string] / profiled_df['denominator'])
    scan_line_input_CT = profiled_df[['geometry', 'CT']]

    long_weights = np.zeros(n_pixels_option_values)
    long_weights[codes] = weights
    
    print('\nINITIALIZING SECOND SCANLINES')
    interpolate = scanlines_interpolate(target_gdf = target_gdf, 
                                        source_CTs = scan_line_input_CT, 
                                        weights_long = long_weights, 
                                        raster_path = raster_path)
    
    interpolate_df = pd.concat([target_gdf.reset_index(), interpolate], axis = 1)
    
    return interpolate_df








def _return_xgboost_weights(profiled_df, 
                            pop_string, 
                            str_codes, 
                            tuned_xgb = False, 
                            gbm_hyperparam_grid = {'learning_rate': [0.001,0.01, 0.1],
                                                  'n_estimators': [200],
                                                  'subsample': [0.3, 0.5],
                                                  'max_depth': [4, 5, 6],
                                                  'num_boosting_rounds': [10, 20]}):
    
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
                                verbose = 3, # Prints the grid search profile
                                n_jobs = -1) # Process the GridSearch in parallel all cores availables

        # Fit the grid to the data
        grid_mse.fit(X, y)
        
        best_params = grid_mse.best_params_
        best_params["objective"] = "reg:linear"
        
        # Create the DMatrix
        xgb_dmatrix = xgb.DMatrix(X, y)
        
        # Train the model from the best parameters of the grid search
        xg_reg = xgb.train(params = best_params, dtrain = xgb_dmatrix)
    
    # Build explainer and fit Shapley's values (https://github.com/slundberg/shap)
    explainer = shap.TreeExplainer(xg_reg)
    shap_values = explainer.shap_values(X)
    weights = np.array(shap_values.mean(axis = 0)) # This is already sorted by pixel Type
    
    return weights
    