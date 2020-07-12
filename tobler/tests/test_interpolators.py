"""test interpolation functions."""
import sys
import pandas as pd
import geopandas
try:
    import quilt3
    QUILTMISSING = False
except ImportError:
    QUILTMISSING = True

import os
from libpysal.examples import load_example
from numpy.testing import assert_almost_equal
from tobler.dasymetric import masked_area_interpolate
from tobler.area_weighted import area_interpolate
from tobler.model import glm, glm_pixel_adjusted
import pytest


def datasets():
    if not QUILTMISSING:

        if not os.path.exists("nlcd_2011.tif"):
            p = quilt3.Package.browse("rasters/nlcd", "s3://spatial-ucr")
            p["nlcd_2011.tif"].fetch()

        sac1 = load_example("Sacramento1")
        sac2 = load_example("Sacramento2")

        sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))
        sac2 = geopandas.read_file(sac2.get_path("SacramentoMSA2.shp"))

        return sac1, sac2
    else:
        pass

@pytest.mark.skipif(QUILTMISSING, reason="quilt3 not available.")
def test_area_interpolate():
    sac1, sac2 = datasets()
    area = area_interpolate(
        source_df=sac2, target_df=sac1, extensive_variables=["POP2001"]
    )
    assert_almost_equal(area.POP2001.sum(), 1894018, decimal=0)


@pytest.mark.skipif(QUILTMISSING, reason="quilt3 not available.")
def test_masked_area_interpolate():
    sac1, sac2 = datasets()
    masked = masked_area_interpolate(
        source_df=sac2,
        target_df=sac1,
        extensive_variables=["POP2001"],
        raster="nlcd_2011.tif",
    )
    assert masked.POP2001.sum() > 1500000


@pytest.mark.skipif(QUILTMISSING, reason="quilt3 not available.")
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


@pytest.mark.skipif(QUILTMISSING, reason="quilt3 not available.")
def test_glm_poisson():
    sac1, sac2 = datasets()
    glm_poisson = glm(
        source_df=sac2, target_df=sac1, variable="POP2001", raster="nlcd_2011.tif"
    )
    assert glm_poisson.POP2001.sum() > 1469000
