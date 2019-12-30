"""test interpolation functions."""
import pandas as pd
import geopandas
from libpysal.examples import load_example
from numpy.testing import assert_almost_equal
from tobler.dasymetric import glm, glm_pixel_adjusted, masked_area_interpolate
from tobler.area_weighted import area_interpolate


sac1 = load_example("Sacramento1")
sac2 = load_example("Sacramento2")

sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))
sac2 = geopandas.read_file(sac2.get_path("SacramentoMSA2.shp"))


def test_area_interpolate():
    area = area_interpolate(source_df=sac2, target_df=sac1, extensive_variables=['POP2001'])
    assert_almost_equal(area.POP2001.sum(), 1894018, decimal=0)


def test_masked_area_interpolate():
    masked = masked_area_interpolate(source_df=sac2, target_df=sac1, extensive_variables=['POP2001'])
    assert_almost_equal(masked.POP2001.sum(), 1894018, decimal=0)


def test_glm_pixel_adjusted():
    adjusted = glm_pixel_adjusted(source_df=sac2, target_df=sac1, variable='POP2001', ReLU=False)
    assert_almost_equal(adjusted.POP2001.sum(), 4054516, decimal=0)


def test_glm_poisson():

    glm_poisson = glm(source_df=sac2, target_df=sac1, variable="POP2001")
    assert glm_poisson.POP2001.sum() > 1469000
