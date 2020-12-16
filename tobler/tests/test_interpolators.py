"""test interpolation functions."""
import geopandas
try:
    import quilt3
    QUILTMISSING = False
except ImportError:
    QUILTMISSING = True

import os
from libpysal.examples import load_example
from numpy.testing import assert_almost_equal
from tobler.area_weighted import area_interpolate
import pytest


def datasets():
    if not QUILTMISSING:

        if not os.path.exists("nlcd_2011.tif"):
            p = quilt3.Package.browse("rasters/nlcd", "s3://spatial-ucr")
            p["nlcd_2011.tif"].fetch()
        sac1 = load_example('Sacramento1')
        sac2 = load_example('Sacramento2')
        sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))
        sac2 = geopandas.read_file(sac2.get_path("SacramentoMSA2.shp"))
        sac1['pct_poverty'] = sac1.POV_POP/sac1.POV_TOT

        return sac1, sac2
    else:
        pass


@pytest.mark.skipif(QUILTMISSING, reason="quilt3 not available.")
def test_area_interpolate():
    sac1, sac2 = datasets()
    area = area_interpolate(
        source_df=sac1, target_df=sac2, extensive_variables=["TOT_POP"],
        intensive_variables=["pct_poverty"]
    )
    assert_almost_equal(area.TOT_POP.sum(), 1796856, decimal=0)
    assert_almost_equal(area.pct_poverty.sum(), 2140, decimal=0)
