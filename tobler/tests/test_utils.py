"""test utility functions."""

import platform

import geopandas
import pytest
from libpysal.examples import load_example
from numpy.testing import assert_almost_equal

from tobler.util import h3fy


def test_h3fy():
    sac1 = load_example("Sacramento1")
    sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))
    sac_hex = h3fy(sac1, return_geoms=True)
    assert sac_hex.shape == (364, 1)


def test_h3fy_nogeoms():
    sac1 = load_example("Sacramento1")
    sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))
    sac_hex = h3fy(sac1, return_geoms=False)
    assert len(sac_hex) == 364


def test_h3fy_nocrs():
    sac1 = load_example("Sacramento1")
    sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))
    sac1.crs = None
    try:
        sac_hex = h3fy(sac1, return_geoms=True)
    except ValueError:
        pass


def test_h3fy_diff_crs():
    sac1 = load_example("Sacramento1")
    sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))
    sac1 = sac1.to_crs(32710)
    sac_hex = h3fy(sac1)
    assert sac_hex.shape == (364, 1)
    assert sac_hex.crs.to_string() == "EPSG:32710"


def test_h3fy_clip():
    sac1 = load_example("Sacramento1")
    sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))
    sac_hex = h3fy(sac1, clip=True)
    sac_hex = sac_hex.to_crs(sac_hex.estimate_utm_crs())
    assert_almost_equal(
        sac_hex.area.sum(), 13131736346.537422, decimal=0
    )


@pytest.mark.skipif(platform.system() == "Windows", reason='Unknown precision error on Windows. See #174 for details')
def test_h3_multipoly():
    va = geopandas.read_file(load_example("virginia").get_path("virginia.shp"))
    va = va.to_crs(va.estimate_utm_crs())

    va = h3fy(va)
    assert_almost_equal(va.area.sum(), 102888497504.47836, decimal=0)
