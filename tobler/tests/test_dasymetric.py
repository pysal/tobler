"""test interpolation functions."""
import geopandas

from libpysal.examples import load_example
from tobler.dasymetric import masked_area_interpolate


def datasets():
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


def test_masked_area_interpolate():
    sac1, sac2 = datasets()
    masked = masked_area_interpolate(
        source_df=sac1,
        target_df=sac2,
        extensive_variables=["TOT_POP"],
        intensive_variables=["pct_poverty"],
        raster="https://spatial-ucr.s3.amazonaws.com/nlcd/landcover/nlcd_landcover_2011.tif",
        pixel_values=[21, 22, 23, 24],
    )
    assert masked.TOT_POP.sum().round(0) == sac1.TOT_POP.sum()
    assert masked.pct_poverty.sum() > 2000
