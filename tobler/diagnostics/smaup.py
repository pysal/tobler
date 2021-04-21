"""
A wrapper for using the S-maup statistical test in tobler interpolation

"""

from esda.smaup import Smaup
from esda.moran import Moran
from libpysal.weights import Rook
from warnings import warn

def _smaup(
    source_df,
    target_df,
    y,
    w=None):
    """
    A function that wraps esda's smaup function and automates some of the process of calculating smaup.
    For more about smaup read here: https://pysal.org/esda/generated/esda.Smaup.html#esda.Smaup

    Parameters
    ----------
    source_df :     geopandas.GeoDataFrame
                    source data to be converted. Used to construct spatial weights.
    target_df :     geopandas.GeoDataFrame
                    target geometries that will form the new representation of the input data. Used for k.
    y               : array
                      data for autocorellation calculation
    w               : libpysal.weights object
                      pysal spatial weights object for autocorellation calculation.
                      Rook recommended for smaup and used by default.

    Returns
    -------
    esda.smaup.Smaup:
        statistic that contains information regarding the extent to which the variable is affected by the MAUP.

    """
    if w is None:
        w = Rook.from_dataframe(source_df)
    rho = Moran(y, w).I
    n = len(y)
    k = len(target_df)
    stat = Smaup(n,k,rho)

    return stat
