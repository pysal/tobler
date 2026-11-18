"""test interpolation functions."""

import rasterio

from tobler.model import glm


def test_glm_poisson(datasets):
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        sac1, sac2 = datasets
        glm_poisson = glm(
            source_df=sac2,
            target_df=sac1,
            extensive_variable="POP2001",
            raster="s3://spatial-ucr/nlcd/landcover/nlcd_landcover_2011.tif",
        )
    assert glm_poisson.POP2001.sum() > 1469000
