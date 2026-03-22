"""test interpolation functions."""

import geopandas
from libpysal.examples import load_example

from tobler.model import glm


def test_glm_poisson(datasets):
    sac1, sac2 = datasets
    glm_poisson = glm(
        source_df=sac2,
        target_df=sac1,
        variable="POP2001",
        raster="https://spatial-ucr.s3.amazonaws.com/nlcd/landcover/nlcd_landcover_2011.tif",
    )
    assert glm_poisson.POP2001.sum() > 1469000
