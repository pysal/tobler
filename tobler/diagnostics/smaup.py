"""
A wrapper for using the S-maup statistical test in tobler interpolation

"""

from esda.smaup import Smaup
from esda.moran import Moran

def _smaup(k, y, w,):
    """
    A function that wraps esda's smaup function and automates some of the process of calculating smaup.
    For more about smaup read here: https://pysal.org/esda/generated/esda.Smaup.html#esda.Smaup

    Parameters
    ----------

    k               : int
                      number of regions
    y               : array
                      data for autocorellation calculation
    w               : libpysal.weights object
                      pysal spatial weights object for autocorellation calculation

    Returns
    -------
    esda.smaup.Smaup:
        statistic that contains information regarding the extent to which the variable is affected by the MAUP.


    """
    rho = Moran(y, w).I
    n = len(y)
    stat = Smaup(n,k,rho)
    return stat
    