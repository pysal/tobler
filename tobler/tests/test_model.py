"""test interpolation functions."""
import geopandas

try:
    import quilt3

    QUILTMISSING = False
except ImportError:
    QUILTMISSING = True

import os
from libpysal.examples import load_example
from tobler.model import glm
import pytest


def datasets():
    if not QUILTMISSING:

        if not os.path.exists("nlcd_2011.tif"):
            p = quilt3.Package.browse("rasters/nlcd", "s3://spatial-ucr")
            p["nlcd_2011.tif"].fetch()
        sac1 = load_example("Sacramento1")
        sac2 = load_example("Sacramento2")
        sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))
        sac1 = sac1.to_crs(sac1.estimate_utm_crs())
        sac2 = geopandas.read_file(sac2.get_path("SacramentoMSA2.shp"))
        sac2 = sac2.to_crs(sac2.estimate_utm_crs())
        sac1["pct_poverty"] = sac1.POV_POP / sac1.POV_TOT
        categories = ["cat", "dog", "donkey", "wombat", "capybara"]
        sac1["animal"] = (categories * ((len(sac1) // len(categories)) + 1))[
            : len(sac1)
        ]

        return sac1, sac2
    else:
        pass


@pytest.mark.skipif(QUILTMISSING, reason="quilt3 not available.")
def test_glm_poisson():
    sac1, sac2 = datasets()
    glm_poisson = glm(
        source_df=sac2, target_df=sac1, variable="POP2001", raster="nlcd_2011.tif",
    )
    assert glm_poisson.POP2001.sum() > 1469000
