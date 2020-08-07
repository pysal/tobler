"""test interpolation functions."""
import pandas as pd
import geopandas
import quilt3
import os
from libpysal.examples import load_example
from numpy.testing import assert_almost_equal
from tobler.dasymetric import masked_area_interpolate
from tobler.area_weighted import area_interpolate
from tobler.model import glm, glm_pixel_adjusted


def datasets():

    if not os.path.exists("nlcd_2011.tif"):
        p = quilt3.Package.browse("rasters/nlcd", "s3://spatial-ucr")
        p["nlcd_2011.tif"].fetch()

    sac1 = load_example("Sacramento1")
    sac2 = load_example("Sacramento2")

    sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))
    sac2 = geopandas.read_file(sac2.get_path("SacramentoMSA2.shp"))
    
    sac1['pct_poverty'] = sac1.POV_POP/sac1.POV_TOT

    return sac1, sac2


def test_area_interpolate():
    sac1, sac2 = datasets()
    area = area_interpolate(
        source_df=sac1, target_df=sac2, extensive_variables=["TOT_POP"], intensive_variables=["pct_poverty"]
    )
    assert_almost_equal(area.POP2001.sum(), 1796856, decimal=0)
    assert_almost_equal(area.pct_poverty.sum(), 1867, decimal=0)
    


def test_masked_area_interpolate():
    sac1, sac2 = datasets()
    masked = masked_area_interpolate(
        source_df=sac2,
        target_df=sac1,
        extensive_variables=["TOT_POP"],
        intensive_variables=['pct_poverty'],
        raster="nlcd_2011.tif",
    )
    assert masked.TOT_POP.sum() > 1500000
    assert masked.pct_poverty.sum() > 2000


def test_glm_pixel_adjusted():
    sac1, sac2 = datasets()
    adjusted = glm_pixel_adjusted(
        source_df=sac2,
        target_df=sac1,
        variable="POP2001",
        ReLU=False,
        raster="nlcd_2011.tif",
    )
    assert_almost_equal(adjusted.POP2001.sum(), 4054516, decimal=0)


def test_glm_poisson():
    sac1, sac2 = datasets()
    glm_poisson = glm(
        source_df=sac2, target_df=sac1, variable="POP2001", raster="nlcd_2011.tif"
    )
    assert glm_poisson.POP2001.sum() > 1469000
