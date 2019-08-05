"""
Useful functions for spatial interpolation methods of tobler
"""

import numpy as np

def _check_crs(source_df, target_df):
    """check if crs is identical"""
    if not (source_df.crs == target_df.crs):
        print("Source and target dataframes have different crs. Please correct.")
        return False
    return True


def _nan_check(df, column):
    """Check if variable has nan values.

    Warn and replace nan with 0.0.
    """
    values = df[column].values
    if np.any(np.isnan(values)):
        wherenan = np.isnan(values)
        values[wherenan] = 0.0
        print("nan values in variable: {var}, replacing with 0.0".format(var=column))
    return values


def _check_presence_of_crs(geoinput):
    """check if there is crs in the polygon/geodataframe"""
    if (geoinput.crs is None):
        raise KeyError('The polygon/geodataframe does not have a Coordinate Reference System (CRS). This must be set before using this function.')
    
    # Since the CRS can be an empty dictionary:
    if (len(geoinput.crs) == 0):
        raise KeyError('The polygon/geodataframe does not have a Coordinate Reference System (CRS). This must be set before using this function.')
