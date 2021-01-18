"""test utility functions."""

import geopandas
from libpysal.examples import load_example
from tobler.util import h3fy
from numpy.testing import assert_almost_equal

sac1 = load_example("Sacramento1")
sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))


def test_h3fy():
    sac_hex = h3fy(sac1, return_geoms=True)
    assert sac_hex.shape == (364, 1)


def test_h3fy_nogeoms():
    sac_hex = h3fy(sac1, return_geoms=False)
    assert len(sac_hex) == 364


def test_h3fy_diff_crs():
    sac1 = load_example("Sacramento1")
    sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))
    sac1 = sac1.to_crs(32710)
    sac_hex = h3fy(sac1)
    assert sac_hex.shape == (364, 1)
    assert sac_hex.crs.to_string() == "EPSG:32710"


def test_h3fy_clip():
    sac_hex = h3fy(sac1, clip=True)
    assert_almost_equal(
        sac_hex.to_crs(32710).unary_union.area, 13131736346.537416, decimal=4
    )

def test_h3_multipoly():
    va = geopandas.read_file(load_example('virginia').get_path('virginia.shp'))
    va = h3fy(va)
    assert_almost_equal(
        va.to_crs(2284).unary_union.area, 1106844905155.1118, decimal=0
    )