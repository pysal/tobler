"""test interpolation functions."""

import geopandas
import pytest
from libpysal.examples import load_example
from rasterstats.io import NodataWarning

from tobler.model import glm


def test_glm_poisson(datasets):
    sac1, sac2 = datasets
    with pytest.warns(
        NodataWarning,
        match="Setting nodata to -999; specify nodata explicitly",
    ):
        glm_poisson = glm(
            source_df=sac2,
            target_df=sac1,
            variable="POP2001",
            raster=(
                "https://spatial-ucr.s3.amazonaws.com/nlcd/"
                "landcover/nlcd_landcover_2011.tif"
            ),
        )
    assert glm_poisson.POP2001.sum() > 1469000
