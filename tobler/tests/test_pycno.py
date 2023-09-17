"""test interpolation functions."""
import geopandas

from libpysal.examples import load_example
from numpy.testing import assert_almost_equal
from tobler.pycno import pycno_interpolate


def datasets():
    sac1 = load_example("Sacramento1")
    sac2 = load_example("Sacramento2")
    sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))
    sac2 = geopandas.read_file(sac2.get_path("SacramentoMSA2.shp"))
    sac1 = sac1.to_crs(sac1.estimate_utm_crs())
    sac2 = sac2.to_crs(sac1.crs)
    sac1["pct_poverty"] = sac1.POV_POP / sac1.POV_TOT

    return sac1, sac2


def test_pycno_interpolate():
    sac1, sac2 = datasets()
    pyc = pycno_interpolate(
        source_df=sac1, target_df=sac2, variables=["TOT_POP"], cellsize=500
    )
    assert_almost_equal(pyc.TOT_POP.sum(), 1794618.503, decimal=1)

def test_custom_index():
    sac1, sac2 = datasets()
    sac2 = sac2.set_index("ZIP")
    pyc = pycno_interpolate(
        source_df=sac1, target_df=sac2, variables=["TOT_POP"], cellsize=500
    )
    assert_almost_equal(pyc.TOT_POP.sum(), 1794618.503, decimal=1)