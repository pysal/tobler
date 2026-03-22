"""test interpolation functions."""

import geopandas
from numpy.testing import assert_almost_equal

from tobler.pycno import pycno_interpolate


def test_pycno_interpolate(datasets):
    sac1, sac2 = datasets
    pyc = pycno_interpolate(
        source_df=sac1, target_df=sac2, variables=["TOT_POP"], cellsize=500
    )
    assert_almost_equal(pyc.TOT_POP.sum(), 1794618.503, decimal=1)


def test_custom_index(datasets):
    sac1, sac2 = datasets
    sac2 = sac2.set_index("ZIP")
    pyc = pycno_interpolate(
        source_df=sac1, target_df=sac2, variables=["TOT_POP"], cellsize=500
    )
    assert_almost_equal(pyc.TOT_POP.sum(), 1794618.503, decimal=1)
