"""test interpolation functions."""

import pytest
from libpysal.examples import load_example

from tobler.dasymetric import masked_area_interpolate, masked_dot_density


def test_masked_area_interpolate(datasets):
    sac1, sac2 = datasets
    with pytest.WARN_VAR_VALS_INF:
        masked = masked_area_interpolate(
            source_df=sac1,
            target_df=sac2,
            extensive_variables=["TOT_POP"],
            intensive_variables=["pct_poverty"],
            raster=(
                "https://spatial-ucr.s3.amazonaws.com/nlcd/"
                "landcover/nlcd_landcover_2011.tif"
            ),
            pixel_values=[21, 22, 23, 24],
        )
    assert masked.TOT_POP.sum().round(0) == sac1.TOT_POP.sum()
    assert masked.pct_poverty.sum() > 2000


def test_masked_dot_density(datasets):
    sac1 = datasets[0]
    # Keep draws small so test remains fast and deterministic.
    sac1["dot_count"] = 1

    dots = masked_dot_density(
        source_df=sac1,
        raster=(
            "https://spatial-ucr.s3.amazonaws.com/nlcd/"
            "landcover/nlcd_landcover_2011.tif"
        ),
        pixel_values=[21, 22, 23, 24],
        columns=["dot_count"],
        rng=0,
    )

    assert len(dots) > 0
    assert set(dots["category"].unique()) == {"dot_count"}
    assert dots.geometry.geom_type.eq("Point").all()
